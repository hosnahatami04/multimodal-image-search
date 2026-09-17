"""Recall@k, MRR, and the cross-modal alignment measurement.

Two metrics rather than one, because they answer different questions.

**Recall@k** is a yes/no question: was the gold image somewhere in the top k?
It matches how the system is actually used -- a person scanning a page of
results either finds what they wanted or does not. But it throws away
everything about *where* in that page the answer sat, so a system that always
ranks the answer first and one that always ranks it tenth score identically at
k=10.

**MRR** keeps that information. Each query contributes 1/rank, so rank 1 scores
1.0, rank 2 scores 0.5, rank 10 scores 0.1. A change in MRR with a flat
Recall@10 means the ordering improved inside the window, which is real progress
that Recall alone would hide.

Both are computed over the *full* ranking of all 8,000 images rather than over
a truncated top-k window. A gold image that never appears in the window would
otherwise be silently scored as a miss at every k, making a retrieval failure
indistinguishable from an index artefact.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_KS: tuple[int, ...] = (1, 5, 10)


class MetricError(RuntimeError):
    """Raised when a metric cannot be computed from the inputs given."""


@dataclass(frozen=True)
class QueryOutcome:
    """Where one query's gold image landed in the full ranking."""

    query_id: str
    query_text: str
    gold_image_id: str
    rank: int
    gold_score: float
    top_image_id: str
    top_score: float

    @property
    def reciprocal_rank(self) -> float:
        return 1.0 / self.rank

    def hit_at(self, k: int) -> bool:
        return self.rank <= k

    @property
    def margin(self) -> float:
        """How far the top result outscored the gold image.

        Zero when the gold image *is* the top result. A large margin on a miss
        says the model was confidently wrong, which is more interesting than a
        near-miss and is what Phase 6 wants to look at.
        """
        return self.top_score - self.gold_score


@dataclass(frozen=True)
class RetrievalReport:
    """Aggregate retrieval quality over a set of queries."""

    num_queries: int
    recall: dict[int, float]
    mrr: float
    median_rank: int
    mean_rank: float
    worst_rank: int
    outcomes: list[QueryOutcome] = field(default_factory=list, repr=False)

    def describe(self, label: str = "") -> str:
        head = f"{self.num_queries} queries" + (f"  [{label}]" if label else "")
        recalls = "   ".join(f"R@{k} {value:.3f}" for k, value in sorted(self.recall.items()))
        return (
            f"{head}\n"
            f"  {recalls}\n"
            f"  MRR {self.mrr:.3f}\n"
            f"  rank: median {self.median_rank}, mean {self.mean_rank:.1f}, "
            f"worst {self.worst_rank}"
        )

    def misses(self, k: int = 5) -> list[QueryOutcome]:
        """Queries whose gold image did not reach the top k, worst first."""
        return sorted(
            (outcome for outcome in self.outcomes if not outcome.hit_at(k)),
            key=lambda outcome: -outcome.rank,
        )


def rank_of(scores: np.ndarray, ids: Sequence[str], target_id: str) -> tuple[int, float]:
    """Find the 1-based rank of ``target_id`` in a descending score order.

    Ties are resolved pessimistically: an image tied with the gold image counts
    as ranking above it. Optimistic tie-breaking would let a model that assigns
    the same score to everything score perfectly.

    Raises:
        MetricError: if the target is not among the ids.
    """
    try:
        position = ids.index(target_id)
    except ValueError as error:
        raise MetricError(f"{target_id!r} is not in the scored set") from error

    gold_score = float(scores[position])
    # Count strictly-better plus tied-but-not-self, then add one for 1-based.
    better = int(np.sum(scores > gold_score))
    tied = int(np.sum(scores == gold_score)) - 1
    return better + tied + 1, gold_score


def evaluate(
    searcher,
    queries: Sequence[dict],
    *,
    ks: Sequence[int] = DEFAULT_KS,
    progress: bool = False,
) -> RetrievalReport:
    """Score every query and fold the ranks into a report.

    Args:
        searcher: Must expose ``score_all(text) -> (scores, ids)``.
        queries: Dicts with ``query_id``, ``query``, and ``gold_image_id``.
        ks: Which cut-offs to report Recall at.

    Raises:
        MetricError: on an empty query set or a gold id absent from the index.
    """
    if not queries:
        raise MetricError("cannot evaluate an empty query set")

    iterator = queries
    if progress:
        from tqdm import tqdm

        iterator = tqdm(queries, desc="  scoring", unit="query")

    outcomes: list[QueryOutcome] = []

    for entry in iterator:
        text = entry["query"]
        gold = entry["gold_image_id"]

        scores, ids = searcher.score_all(text)
        rank, gold_score = rank_of(scores, ids, gold)

        top_position = int(np.argmax(scores))
        outcomes.append(
            QueryOutcome(
                query_id=str(entry.get("query_id", "")),
                query_text=text,
                gold_image_id=gold,
                rank=rank,
                gold_score=gold_score,
                top_image_id=ids[top_position],
                top_score=float(scores[top_position]),
            )
        )

    return summarise(outcomes, ks=ks)


def summarise(
    outcomes: Sequence[QueryOutcome], *, ks: Sequence[int] = DEFAULT_KS
) -> RetrievalReport:
    """Fold per-query outcomes into aggregate metrics."""
    if not outcomes:
        raise MetricError("cannot summarise zero outcomes")

    ranks = np.array([outcome.rank for outcome in outcomes])

    return RetrievalReport(
        num_queries=len(outcomes),
        recall={k: float(np.mean(ranks <= k)) for k in ks},
        mrr=float(np.mean([outcome.reciprocal_rank for outcome in outcomes])),
        median_rank=int(np.median(ranks)),
        mean_rank=float(ranks.mean()),
        worst_rank=int(ranks.max()),
        outcomes=list(outcomes),
    )


def alignment(
    image_vectors: np.ndarray,
    caption_vectors: np.ndarray,
    owners: Sequence[int],
) -> dict[str, float]:
    """Compare image-to-own-caption similarity against image-to-random-caption.

    Args:
        image_vectors: Unit-norm image embeddings, one row per image.
        caption_vectors: Unit-norm caption embeddings.
        owners: For each caption row, the index of the image it belongs to.

    Returns:
        Means, percentiles, the gap between the two distributions, and the
        fraction of captions that score above the random baseline's mean --
        a crude but readable separation measure.

    A clean separation means the shared space is doing its job. Overlap is
    where it breaks down, and the overlapping cases are what Phase 6 examines.
    """
    if len(owners) != caption_vectors.shape[0]:
        raise MetricError(f"{len(owners)} owners for {caption_vectors.shape[0]} caption vectors")

    owners = np.asarray(owners)

    own = np.einsum("ij,ij->i", caption_vectors, image_vectors[owners])

    # Rotating the owner list guarantees no caption is paired with its own
    # image, which independent sampling would not.
    shifted = np.roll(owners, len(owners) // 2 + 1)
    random_pairs = np.einsum("ij,ij->i", caption_vectors, image_vectors[shifted])

    return {
        "own_mean": float(own.mean()),
        "own_p05": float(np.percentile(own, 5)),
        "own_p95": float(np.percentile(own, 95)),
        "random_mean": float(random_pairs.mean()),
        "random_p95": float(np.percentile(random_pairs, 95)),
        "gap": float(own.mean() - random_pairs.mean()),
        "separation": float(np.mean(own > random_pairs.mean())),
        "overlap": float(np.mean(own < np.percentile(random_pairs, 95))),
    }
