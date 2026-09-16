"""Persist encoded vectors so the long encode run happens once.

Encoding 8,000 images through ViT-B/32 on CPU takes tens of minutes. Doing that
at the start of every experiment would make iteration impossible, so the matrix
is written to a `.npy` file alongside a small JSON sidecar and reloaded
afterwards.

The sidecar is what makes the cache trustworthy. A bare `.npy` file cannot
answer "which model produced this?" or "which images are these rows, in what
order?", and a cache that silently answers those questions wrong is worse than
no cache: it returns vectors from a different model or a different image order,
and nothing downstream can tell. So every load re-checks the model id, the
revision, the dimensionality, and the exact id list before handing the array
back.

Files land in ``data/embeddings`` and are gitignored -- they are derived data,
regenerable from the pinned model and the pinned dataset.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from src import config

logger = logging.getLogger(__name__)

CACHE_FORMAT_VERSION = 1


class CacheError(RuntimeError):
    """Raised when a cache entry is missing, stale, or inconsistent."""


@dataclass(frozen=True)
class CacheEntry:
    """A cached embedding matrix together with the ids its rows belong to."""

    vectors: np.ndarray
    image_ids: tuple[str, ...]
    model_id: str
    revision: str

    def __post_init__(self) -> None:
        if len(self.image_ids) != self.vectors.shape[0]:
            raise CacheError(f"{len(self.image_ids)} ids for {self.vectors.shape[0]} vectors")

    @property
    def dim(self) -> int:
        return int(self.vectors.shape[1])

    def index_of(self, image_id: str) -> int:
        """Row number for one id, for callers that need to exclude it."""
        try:
            return self.image_ids.index(image_id)
        except ValueError as error:
            raise CacheError(f"{image_id!r} is not in this cache") from error


def _slug(model_id: str) -> str:
    """Filesystem-safe stem derived from a model id."""
    return model_id.replace("/", "__").replace(":", "_")


def cache_paths(model_id: str, tag: str = "images") -> tuple[Path, Path]:
    """Return the ``(vectors.npy, meta.json)`` pair for this model and tag."""
    stem = f"{_slug(model_id)}.{tag}"
    return (
        config.EMBEDDINGS_DIR / f"{stem}.npy",
        config.EMBEDDINGS_DIR / f"{stem}.meta.json",
    )


def _ids_digest(image_ids: tuple[str, ...]) -> str:
    """Hash the id list so row order can be verified without storing it twice."""
    hasher = hashlib.sha256()
    for image_id in image_ids:
        hasher.update(image_id.encode("utf-8"))
        hasher.update(b"\x00")
    return hasher.hexdigest()


def save(
    vectors: np.ndarray,
    image_ids: Sequence[str],
    *,
    model_id: str = config.CLIP_MODEL_ID,
    revision: str = config.CLIP_REVISION,
    tag: str = "images",
) -> Path:
    """Write vectors and their provenance to disk.

    Raises:
        CacheError: if the vectors are not float32, not 2-D, not unit-norm, or
            do not line up with the id list. Catching this at write time keeps
            a broken matrix out of the cache entirely.
    """
    image_ids = tuple(image_ids)

    if vectors.ndim != 2:
        raise CacheError(f"expected a 2-D matrix, got shape {vectors.shape}")
    if len(image_ids) != vectors.shape[0]:
        raise CacheError(f"{len(image_ids)} ids for {vectors.shape[0]} vectors")
    if len(set(image_ids)) != len(image_ids):
        raise CacheError("image ids are not unique")

    norms = np.linalg.norm(vectors, axis=1)
    if not np.allclose(norms, 1.0, atol=1e-4):
        worst = float(np.abs(norms - 1.0).max())
        raise CacheError(
            f"vectors are not unit-norm (worst deviation {worst:.2e}); "
            "normalise before caching or every similarity downstream is wrong"
        )

    config.EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
    vectors_path, meta_path = cache_paths(model_id, tag)

    np.save(vectors_path, vectors.astype(np.float32, copy=False))
    meta_path.write_text(
        json.dumps(
            {
                "format_version": CACHE_FORMAT_VERSION,
                "model_id": model_id,
                "revision": revision,
                "tag": tag,
                "count": len(image_ids),
                "dim": int(vectors.shape[1]),
                "ids_digest": _ids_digest(image_ids),
                "image_ids": list(image_ids),
                "written_at": datetime.now(UTC).isoformat(timespec="seconds"),
            },
            indent=1,
        ),
        encoding="utf-8",
    )

    logger.info(
        "cached %d x %d vectors to %s",
        vectors.shape[0],
        vectors.shape[1],
        vectors_path.name,
    )
    return vectors_path


def load(
    *,
    model_id: str = config.CLIP_MODEL_ID,
    revision: str = config.CLIP_REVISION,
    tag: str = "images",
    expected_ids: Sequence[str] | None = None,
) -> CacheEntry:
    """Read a cached matrix back, verifying it is the one the caller wants.

    Args:
        expected_ids: When given, the cached id list must match it exactly,
            in order. This is the check that catches a cache built from a
            different split or a different dataset revision.

    Raises:
        CacheError: if the files are absent, were written by another model or
            revision, or do not match ``expected_ids``.
    """
    vectors_path, meta_path = cache_paths(model_id, tag)

    if not vectors_path.exists() or not meta_path.exists():
        raise CacheError(
            f"no cached embeddings for {model_id} ({tag}). "
            "Run `python -m src.embedding.build_index` first."
        )

    meta = json.loads(meta_path.read_text(encoding="utf-8"))

    if meta.get("format_version") != CACHE_FORMAT_VERSION:
        raise CacheError(
            f"cache format {meta.get('format_version')} != expected "
            f"{CACHE_FORMAT_VERSION}; delete {vectors_path.parent} and rebuild"
        )
    if meta["model_id"] != model_id:
        raise CacheError(f"cache was built by {meta['model_id']}, not {model_id}")
    if meta["revision"] != revision:
        raise CacheError(
            f"cache was built at revision {meta['revision'][:12]}, "
            f"not {revision[:12]}; the weights changed, so the vectors did too"
        )

    vectors = np.load(vectors_path)
    image_ids = tuple(meta["image_ids"])

    if vectors.shape[0] != len(image_ids):
        raise CacheError(
            f"{vectors_path.name} holds {vectors.shape[0]} rows but the sidecar "
            f"lists {len(image_ids)} ids"
        )
    if _ids_digest(image_ids) != meta["ids_digest"]:
        raise CacheError(f"{meta_path.name} was edited after it was written")

    if expected_ids is not None:
        expected = tuple(expected_ids)
        if image_ids != expected:
            missing = len(set(expected) - set(image_ids))
            raise CacheError(
                f"cache holds {len(image_ids)} ids, caller expects {len(expected)} "
                f"({missing} of them absent from the cache); rebuild the index"
            )

    logger.info("loaded %d x %d cached vectors", vectors.shape[0], vectors.shape[1])
    return CacheEntry(
        vectors=vectors,
        image_ids=image_ids,
        model_id=meta["model_id"],
        revision=meta["revision"],
    )


def exists(*, model_id: str = config.CLIP_MODEL_ID, tag: str = "images") -> bool:
    """Whether both cache files are present, without validating them."""
    vectors_path, meta_path = cache_paths(model_id, tag)
    return vectors_path.exists() and meta_path.exists()
