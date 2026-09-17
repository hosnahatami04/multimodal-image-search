"""Tests for the evaluation layer.

The metrics are tested against hand-built score arrays with known answers,
because that is the only way to be sure Recall@5 means what it is supposed to.
A metric verified only by running it on real data and finding the number
plausible is not verified at all -- the plausible-looking number is exactly what
an off-by-one in the rank calculation produces.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from src import config
from src.eval.retrieval_metrics import (
    MetricError,
    QueryOutcome,
    alignment,
    evaluate,
    rank_of,
    summarise,
)


def _outcome(rank: int, query_id: str = "q") -> QueryOutcome:
    return QueryOutcome(
        query_id=query_id,
        query_text="a query",
        gold_image_id="gold",
        rank=rank,
        gold_score=0.3,
        top_image_id="top",
        top_score=0.4,
    )


class FakeSearcher:
    """Returns a fixed score vector, so the expected rank is known."""

    def __init__(self, scores: np.ndarray, ids: list[str]) -> None:
        self._scores = scores
        self._ids = tuple(ids)

    def score_all(self, query: str):
        return self._scores, self._ids


# ---------------------------------------------------------------------------
# rank_of
# ---------------------------------------------------------------------------


def test_highest_score_ranks_first():
    scores = np.array([0.9, 0.5, 0.1])
    assert rank_of(scores, ["a", "b", "c"], "a") == (1, pytest.approx(0.9))


def test_lowest_score_ranks_last():
    scores = np.array([0.9, 0.5, 0.1])
    rank, _ = rank_of(scores, ["a", "b", "c"], "c")
    assert rank == 3


def test_rank_is_one_based():
    scores = np.array([0.9, 0.8])
    assert rank_of(scores, ["a", "b"], "a")[0] == 1


def test_ties_are_resolved_pessimistically():
    # Three images share the top score. The gold one is credited with rank 3,
    # not rank 1: optimistic tie-breaking would let a model that scores
    # everything identically achieve a perfect Recall@1.
    scores = np.array([0.5, 0.5, 0.5, 0.1])
    rank, _ = rank_of(scores, ["a", "b", "c", "d"], "b")
    assert rank == 3


def test_unknown_target_is_rejected():
    with pytest.raises(MetricError, match="not in the scored set"):
        rank_of(np.array([0.5]), ["a"], "zzz")


def test_negative_infinity_ranks_last():
    # This is how an excluded image is represented in a full ranking.
    scores = np.array([0.1, -np.inf, 0.9])
    rank, _ = rank_of(scores, ["a", "b", "c"], "b")
    assert rank == 3


# ---------------------------------------------------------------------------
# Recall and MRR
# ---------------------------------------------------------------------------


def test_recall_counts_ranks_at_or_below_k():
    report = summarise([_outcome(r) for r in (1, 3, 5, 6, 20)], ks=(1, 5, 10))

    assert report.recall[1] == pytest.approx(1 / 5)
    assert report.recall[5] == pytest.approx(3 / 5)
    assert report.recall[10] == pytest.approx(4 / 5)


def test_recall_at_k_is_monotonic():
    report = summarise([_outcome(r) for r in (2, 7, 40, 1, 9)], ks=(1, 5, 10))
    values = [report.recall[k] for k in (1, 5, 10)]

    assert values == sorted(values)


def test_perfect_retrieval_scores_one_everywhere():
    report = summarise([_outcome(1) for _ in range(6)], ks=(1, 5, 10))

    assert all(value == 1.0 for value in report.recall.values())
    assert report.mrr == 1.0


def test_mrr_is_the_mean_reciprocal_rank():
    # ranks 1, 2, 4 -> (1 + 0.5 + 0.25) / 3
    report = summarise([_outcome(r) for r in (1, 2, 4)])
    assert report.mrr == pytest.approx(1.75 / 3)


def test_mrr_separates_systems_that_recall_cannot():
    # Both hit every query inside the top 10, so Recall@10 is identical.
    # MRR is not, because one ranks the answer first and the other tenth.
    good = summarise([_outcome(1) for _ in range(5)], ks=(10,))
    poor = summarise([_outcome(10) for _ in range(5)], ks=(10,))

    assert good.recall[10] == poor.recall[10] == 1.0
    assert good.mrr > poor.mrr


def test_rank_statistics_are_reported():
    report = summarise([_outcome(r) for r in (1, 3, 5, 100)])

    assert report.median_rank == 4
    assert report.worst_rank == 100
    assert report.mean_rank == pytest.approx(109 / 4)


def test_misses_lists_only_queries_outside_k():
    report = summarise([_outcome(r, f"q{r}") for r in (1, 4, 9, 30)], ks=(5,))
    missed = report.misses(k=5)

    assert [outcome.rank for outcome in missed] == [30, 9]


def test_summarise_rejects_an_empty_set():
    with pytest.raises(MetricError, match="zero outcomes"):
        summarise([])


def test_margin_is_zero_when_the_gold_image_wins():
    outcome = QueryOutcome(
        query_id="q",
        query_text="t",
        gold_image_id="g",
        rank=1,
        gold_score=0.42,
        top_image_id="g",
        top_score=0.42,
    )
    assert outcome.margin == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# evaluate, end to end over a fake searcher
# ---------------------------------------------------------------------------


def test_evaluate_finds_the_expected_rank():
    scores = np.array([0.9, 0.7, 0.5, 0.3])
    searcher = FakeSearcher(scores, ["a", "b", "c", "d"])

    report = evaluate(
        searcher,
        [{"query_id": "q1", "query": "anything", "gold_image_id": "c"}],
        ks=(1, 5),
    )

    assert report.outcomes[0].rank == 3
    assert report.outcomes[0].top_image_id == "a"
    assert report.recall[1] == 0.0
    assert report.recall[5] == 1.0


def test_evaluate_records_the_margin():
    scores = np.array([0.9, 0.4])
    searcher = FakeSearcher(scores, ["a", "b"])

    report = evaluate(searcher, [{"query_id": "q", "query": "x", "gold_image_id": "b"}], ks=(1,))
    assert report.outcomes[0].margin == pytest.approx(0.5)


def test_evaluate_rejects_an_empty_query_set():
    with pytest.raises(MetricError, match="empty query set"):
        evaluate(FakeSearcher(np.array([1.0]), ["a"]), [])


# ---------------------------------------------------------------------------
# Alignment
# ---------------------------------------------------------------------------


def test_alignment_gap_is_positive_when_pairs_match():
    rng = np.random.default_rng(0)
    images = rng.normal(size=(10, 16)).astype(np.float32)
    images /= np.linalg.norm(images, axis=1, keepdims=True)

    # Each caption is its image plus a little noise, so own-pairs must score
    # far above random pairs.
    owners = list(range(10)) * 2
    captions = images[owners] + rng.normal(scale=0.1, size=(20, 16)).astype(np.float32)
    captions /= np.linalg.norm(captions, axis=1, keepdims=True)

    result = alignment(images, captions, owners)

    assert result["gap"] > 0.5
    assert result["own_mean"] > result["random_mean"]
    assert result["separation"] > 0.9


def test_alignment_gap_vanishes_for_unrelated_vectors():
    rng = np.random.default_rng(1)
    images = rng.normal(size=(20, 16)).astype(np.float32)
    images /= np.linalg.norm(images, axis=1, keepdims=True)
    captions = rng.normal(size=(20, 16)).astype(np.float32)
    captions /= np.linalg.norm(captions, axis=1, keepdims=True)

    result = alignment(images, captions, list(range(20)))

    assert abs(result["gap"]) < 0.2


def test_alignment_rejects_mismatched_owners():
    with pytest.raises(MetricError, match="owners for"):
        alignment(np.zeros((3, 4)), np.zeros((5, 4)), [0, 1])


# ---------------------------------------------------------------------------
# The committed evaluation sets
# ---------------------------------------------------------------------------


@pytest.mark.data
def test_query_set_is_well_formed():
    path = config.EVAL_SETS_DIR / "retrieval_queries.json"
    if not path.exists():
        pytest.skip("query set not built")

    queries = json.loads(path.read_text(encoding="utf-8"))

    assert len(queries) >= 50
    assert len({entry["query_id"] for entry in queries}) == len(queries)
    assert len({entry["gold_image_id"] for entry in queries}) == len(queries)
    for entry in queries:
        assert entry["query"].strip()
        assert entry["category"]


@pytest.mark.data
def test_query_gold_images_are_in_the_eval_split():
    path = config.EVAL_SETS_DIR / "retrieval_queries.json"
    if not path.exists():
        pytest.skip("query set not built")

    from src.data.dataset import load_records

    try:
        ids = {record.image_id for record in load_records(split=config.EVAL_SPLIT)}
    except Exception as error:
        pytest.skip(f"dataset unavailable: {error}")

    for entry in json.loads(path.read_text(encoding="utf-8")):
        assert entry["gold_image_id"] in ids


@pytest.mark.data
def test_hard_negative_groups_are_well_formed():
    path = config.EVAL_SETS_DIR / "hard_negatives.json"
    if not path.exists():
        pytest.skip("hard-negative set not built")

    groups = json.loads(path.read_text(encoding="utf-8"))

    seen: set[str] = set()
    for group in groups:
        assert 5 <= len(group["image_ids"]) <= 10
        assert group["target_image_id"] in group["image_ids"]
        assert group["query"].strip()
        # Groups must not overlap, or a target would sit among its own
        # distractors in two different closed worlds.
        assert not (seen & set(group["image_ids"]))
        seen |= set(group["image_ids"])
