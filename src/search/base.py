"""The shape every search path returns, and the machinery both of them share.

Text→image and image→image differ only in how the query vector is produced.
Once there is a vector, ranking is identical, so that part lives here once and
both paths call it. The evaluation harness and the API then treat the two
uniformly -- neither has to know which path produced a result.

Two ranking backends are available and they answer the same question
differently:

* **Chroma** -- the persisted index, which is what the API serves from.
* **Exact** -- one dot product against the whole cached matrix.

The evaluation harness uses the exact path deliberately. Recall@10 and MRR are
statements about where the right answer ranked among all 8,000 candidates, and
an approximate nearest-neighbour structure can quietly drop the right answer
out of the returned window, which would be scored as a model failure when it is
really an index artefact.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import numpy as np

from src import config
from src.data.dataset import ImageRecord, load_records
from src.embedding import cache, index
from src.embedding.clip_encoder import ClipEncoder

logger = logging.getLogger(__name__)


class SearchError(RuntimeError):
    """Raised when a search cannot be performed."""


@dataclass(frozen=True)
class SearchResult:
    """One ranked hit, identical in shape whatever produced it."""

    image_id: str
    score: float
    path: Path
    captions: tuple[str, ...]
    rank: int

    def __post_init__(self) -> None:
        # Cosine similarity between unit vectors cannot leave [-1, 1]. A value
        # outside it means something upstream stopped normalising, which is
        # worth catching at the point it becomes visible.
        if not -1.0001 <= self.score <= 1.0001:
            raise SearchError(
                f"{self.image_id}: score {self.score} is outside the cosine range; "
                "vectors are probably not unit-norm"
            )

    def describe(self, width: int = 62) -> str:
        caption = self.captions[0] if self.captions else ""
        return f"{self.rank:>2}. {self.score:+.4f}  {self.image_id:<18}  {caption[:width]}"


class BaseSearcher:
    """Shared state and ranking for both query directions.

    Loads the encoder, the record table and the cached matrix once, lazily, so
    constructing a searcher is cheap and the cost lands on first use.
    """

    def __init__(self, *, exact: bool = False, encoder: ClipEncoder | None = None) -> None:
        self.exact = exact
        self._encoder = encoder

    # -----------------------------------------------------------------
    # Lazily loaded resources
    # -----------------------------------------------------------------

    @cached_property
    def encoder(self) -> ClipEncoder:
        return self._encoder or ClipEncoder()

    @cached_property
    def records(self) -> dict[str, ImageRecord]:
        return {record.image_id: record for record in load_records()}

    @cached_property
    def _matrix(self) -> tuple[np.ndarray, tuple[str, ...]]:
        """The full embedding matrix and its row ids, for exact ranking."""
        try:
            entry = cache.load()
            return entry.vectors, entry.image_ids
        except cache.CacheError as error:
            raise SearchError(f"exact search needs the embedding cache: {error}") from error

    # -----------------------------------------------------------------
    # Ranking
    # -----------------------------------------------------------------

    def rank(
        self,
        vector: np.ndarray,
        k: int = 10,
        *,
        exclude: str | None = None,
        restrict_to: Sequence[str] | None = None,
    ) -> list[SearchResult]:
        """Rank the corpus against one query vector.

        Args:
            vector: A unit-norm query vector.
            k: How many results to return.
            exclude: An image id to drop, for image→image search.
            restrict_to: Only consider these ids. Forces the exact path, since
                Chroma's filtering would change what "top k" means.
        """
        if k < 1:
            raise SearchError(f"k must be at least 1, got {k}")

        if self.exact or restrict_to is not None:
            return self._rank_exact(vector, k, exclude=exclude, restrict_to=restrict_to)
        return self._rank_chroma(vector, k, exclude=exclude)

    def _rank_chroma(
        self, vector: np.ndarray, k: int, *, exclude: str | None
    ) -> list[SearchResult]:
        hits = index.query(vector, k=k, exclude=exclude)
        return [
            SearchResult(
                image_id=hit.image_id,
                score=score,
                path=Path(hit.path),
                captions=hit.captions,
                rank=position,
            )
            for position, (hit, score) in enumerate(hits, start=1)
        ]

    def _rank_exact(
        self,
        vector: np.ndarray,
        k: int,
        *,
        exclude: str | None,
        restrict_to: Sequence[str] | None,
    ) -> list[SearchResult]:
        vectors, ids = self._matrix
        scores = vectors @ np.asarray(vector, dtype=np.float32)

        keep = np.ones(len(ids), dtype=bool)
        if exclude is not None:
            keep &= np.array([image_id != exclude for image_id in ids])
        if restrict_to is not None:
            allowed = set(restrict_to)
            keep &= np.array([image_id in allowed for image_id in ids])

        candidates = np.flatnonzero(keep)
        if candidates.size == 0:
            raise SearchError("no candidates left after filtering")

        # argpartition finds the top k without sorting all 8,000, then only
        # those k are sorted. Matters when this runs 100 times in the latency
        # benchmark.
        subset_scores = scores[candidates]
        top = min(k, candidates.size)
        partitioned = np.argpartition(-subset_scores, top - 1)[:top]
        ordered = partitioned[np.argsort(-subset_scores[partitioned])]

        results: list[SearchResult] = []
        for position, offset in enumerate(ordered, start=1):
            row = candidates[offset]
            image_id = ids[row]
            record = self.records.get(image_id)
            results.append(
                SearchResult(
                    image_id=image_id,
                    score=float(scores[row]),
                    path=record.path if record else config.IMAGES_DIR / f"{image_id}.jpg",
                    captions=record.captions if record else (),
                    rank=position,
                )
            )
        return results

    def full_ranking(
        self, vector: np.ndarray, *, exclude: str | None = None
    ) -> tuple[np.ndarray, tuple[str, ...]]:
        """Score every image, unsorted, for metrics that need the whole ranking.

        Returns ``(scores, ids)`` aligned row-wise. Recall@k and MRR compute
        from this directly rather than asking for a top-k window that might not
        contain the gold image at all.
        """
        vectors, ids = self._matrix
        scores = vectors @ np.asarray(vector, dtype=np.float32)

        if exclude is not None:
            mask = np.array([image_id == exclude for image_id in ids])
            scores = np.where(mask, -np.inf, scores)

        return scores, ids
