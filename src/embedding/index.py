"""Build and query the persistent vector index.

ChromaDB holds the 8,000 image vectors together with the metadata a search
result needs -- the image id, its path, its split, and its captions -- so a
query returns everything needed to render a result without a second lookup
against the CSV.

At 8,000 vectors of 512 dimensions this is 16 MB of float32, and a brute-force
NumPy dot product would answer a query in milliseconds. Chroma is here because
it is the shape the problem takes at a scale where brute force stops working,
and because it persists to disk so the index survives a restart. The exact
search path stays available through `matrix()` for the evaluation harness,
which wants deterministic full rankings rather than approximate top-k.

One detail worth stating: Chroma is configured with **cosine** space. Vectors
are already unit-norm coming out of the encoder, so cosine and dot product
agree, but saying so explicitly means a future un-normalised vector degrades
gracefully instead of silently ranking by magnitude.

Chroma returns *distances*, not similarities. For cosine space the relationship
is ``similarity = 1 - distance``, and this module converts at the boundary so
nothing above it ever sees a distance.
"""

from __future__ import annotations

import logging
import shutil
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from src import config
from src.data.dataset import ImageRecord

logger = logging.getLogger(__name__)

COLLECTION_NAME = "flickr8k_clip"

# Chroma writes metadata values as scalars, so the five captions are stored as
# one string and split back on load. The separator has to be something no
# caption contains.
CAPTION_SEPARATOR = "\x1f"


class IndexError_(RuntimeError):
    """Raised when the index is missing, inconsistent, or cannot be built."""


@dataclass(frozen=True)
class IndexedImage:
    """One row of the index, as the search layer consumes it."""

    image_id: str
    path: str
    split: str
    captions: tuple[str, ...]


def _client():
    """Open the persistent Chroma client rooted at ``data/chroma``."""
    import chromadb
    from chromadb.config import Settings

    config.CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(
        path=str(config.CHROMA_DIR),
        settings=Settings(anonymized_telemetry=False, allow_reset=True),
    )


def build(
    records: Sequence[ImageRecord],
    vectors: np.ndarray,
    *,
    reset: bool = False,
) -> int:
    """Write vectors and their metadata into the persistent collection.

    Args:
        records: The images, in the same order as the rows of ``vectors``.
        vectors: Unit-norm embedding matrix.
        reset: Delete any existing collection first. Without this, rebuilding
            after a model change would leave stale vectors mixed with new ones.

    Returns:
        The number of rows in the collection afterwards.

    Raises:
        IndexError_: if records and vectors do not line up.
    """
    if len(records) != vectors.shape[0]:
        raise IndexError_(
            f"{len(records)} records for {vectors.shape[0]} vectors -- these must "
            "be the same list in the same order"
        )
    if not records:
        raise IndexError_("refusing to build an empty index")

    client = _client()

    if reset:
        try:
            client.delete_collection(COLLECTION_NAME)
            logger.info("dropped the existing collection")
        except Exception:
            pass

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={
            "hnsw:space": "cosine",
            "model_id": config.CLIP_MODEL_ID,
            "revision": config.CLIP_REVISION,
        },
    )

    # Chroma's add() holds everything in memory before flushing, so the write
    # is chunked. 1,000 keeps peak memory modest without making the write slow.
    chunk = 1000
    for start in range(0, len(records), chunk):
        window = records[start : start + chunk]
        collection.add(
            ids=[record.image_id for record in window],
            embeddings=vectors[start : start + chunk].tolist(),
            metadatas=[
                {
                    "path": str(record.path),
                    "split": record.split,
                    "captions": CAPTION_SEPARATOR.join(record.captions),
                }
                for record in window
            ],
        )
        logger.info("  indexed %d/%d", min(start + chunk, len(records)), len(records))

    count = collection.count()
    logger.info("collection %r holds %d vectors", COLLECTION_NAME, count)
    return count


def open_collection():
    """Open the existing collection, or explain how to create it."""
    client = _client()
    try:
        return client.get_collection(COLLECTION_NAME)
    except Exception as error:
        raise IndexError_(
            f"collection {COLLECTION_NAME!r} not found in {config.CHROMA_DIR}. "
            "Run `python -m src.embedding.build_index` first."
        ) from error


def query(
    vector: np.ndarray,
    k: int = 10,
    *,
    exclude: str | None = None,
) -> list[tuple[IndexedImage, float]]:
    """Return the ``k`` nearest images to ``vector``, best first.

    Args:
        vector: A single unit-norm query vector.
        k: How many results to return.
        exclude: An image id to drop from the results. Used by image→image
            search, where the query image is always its own nearest neighbour
            and returning it is useless.

    Returns:
        ``(IndexedImage, similarity)`` pairs, similarity in ``[-1, 1]``,
        descending.
    """
    if k < 1:
        raise IndexError_(f"k must be at least 1, got {k}")

    collection = open_collection()

    # Ask for one extra when a result will be dropped, so k results still come
    # back after the exclusion.
    n_results = min(k + (1 if exclude else 0), collection.count())

    response = collection.query(
        query_embeddings=[np.asarray(vector, dtype=np.float32).tolist()],
        n_results=n_results,
        include=["metadatas", "distances"],
    )

    results: list[tuple[IndexedImage, float]] = []
    for image_id, metadata, distance in zip(
        response["ids"][0],
        response["metadatas"][0],
        response["distances"][0],
        strict=True,
    ):
        if exclude is not None and image_id == exclude:
            continue
        results.append(
            (
                IndexedImage(
                    image_id=image_id,
                    path=str(metadata["path"]),
                    split=str(metadata["split"]),
                    captions=tuple(str(metadata["captions"]).split(CAPTION_SEPARATOR)),
                ),
                # Chroma cosine distance is 1 - cosine similarity.
                1.0 - float(distance),
            )
        )

    return results[:k]


def matrix() -> tuple[np.ndarray, tuple[str, ...]]:
    """Pull the whole index back as a matrix, for exact scoring.

    The evaluation harness needs full deterministic rankings over all 8,000
    images -- Recall@10 and MRR are meaningless if the retrieval was
    approximate. One dot product over a 16 MB matrix is fast enough that there
    is no reason to approximate.
    """
    collection = open_collection()
    response = collection.get(include=["embeddings"])

    ids = tuple(response["ids"])
    vectors = np.asarray(response["embeddings"], dtype=np.float32)

    if vectors.shape[0] != len(ids):
        raise IndexError_(f"index returned {vectors.shape[0]} vectors for {len(ids)} ids")

    return vectors, ids


def count() -> int:
    """How many vectors the collection currently holds."""
    return open_collection().count()


def drop() -> None:
    """Delete the persisted index entirely."""
    if config.CHROMA_DIR.exists():
        shutil.rmtree(config.CHROMA_DIR)
        logger.info("removed %s", config.CHROMA_DIR)
