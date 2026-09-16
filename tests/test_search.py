"""Tests for the search layer: result shape, ranking order, exclusion rules.

Ranking is tested against a hand-built matrix rather than against CLIP, because
the properties that matter here -- k results, descending scores, the query
excluded from its own results -- are properties of the ranking code, not of the
model. Testing them through CLIP would make them slow, non-deterministic and
harder to read, and would not test them any more thoroughly.

A few tests do run the real model end to end; they carry the `model` marker.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.data.dataset import ImageRecord
from src.embedding.clip_encoder import l2_normalise
from src.search.base import BaseSearcher, SearchError, SearchResult

# ---------------------------------------------------------------------------
# A searcher backed by a known matrix
# ---------------------------------------------------------------------------


class FakeSearcher(BaseSearcher):
    """BaseSearcher with the cache and record table replaced by fixtures."""

    def __init__(self, vectors: np.ndarray, ids: list[str], records=None) -> None:
        super().__init__(exact=True)
        self._fake_matrix = (vectors, tuple(ids))
        self._fake_records = records or {}

    @property
    def _matrix(self):
        return self._fake_matrix

    @property
    def records(self):
        return self._fake_records


@pytest.fixture
def searcher():
    """Eight images on a circle, so similarity order is known in advance.

    Row i sits at angle i * 45 degrees. A query pointing at 0 degrees is
    therefore closest to img_0, then to its two neighbours, and furthest from
    the one opposite it.
    """
    angles = np.arange(8) * (np.pi / 4)
    vectors = l2_normalise(np.stack([np.cos(angles), np.sin(angles)], axis=1).astype(np.float32))
    ids = [f"img_{i}" for i in range(8)]

    records = {
        image_id: ImageRecord(
            image_id=image_id,
            path=Path(f"/fake/{image_id}.jpg"),
            captions=tuple(f"caption {n} for {image_id}" for n in range(5)),
            split="test",
        )
        for image_id in ids
    }
    return FakeSearcher(vectors, ids, records)


def _query(angle_degrees: float) -> np.ndarray:
    radians = np.deg2rad(angle_degrees)
    return np.array([np.cos(radians), np.sin(radians)], dtype=np.float32)


# ---------------------------------------------------------------------------
# Result contract
# ---------------------------------------------------------------------------


def test_search_returns_exactly_k_results(searcher):
    for k in (1, 3, 5, 8):
        assert len(searcher.rank(_query(0), k=k)) == k


def test_k_larger_than_the_corpus_returns_everything(searcher):
    assert len(searcher.rank(_query(0), k=50)) == 8


def test_scores_are_sorted_descending(searcher):
    scores = [result.score for result in searcher.rank(_query(30), k=8)]
    assert scores == sorted(scores, reverse=True)


def test_ranks_are_consecutive_from_one(searcher):
    results = searcher.rank(_query(0), k=5)
    assert [result.rank for result in results] == [1, 2, 3, 4, 5]


def test_scores_stay_within_the_cosine_range(searcher):
    for result in searcher.rank(_query(123), k=8):
        assert -1.0001 <= result.score <= 1.0001


def test_nearest_result_is_the_aligned_vector(searcher):
    assert searcher.rank(_query(0), k=1)[0].image_id == "img_0"


def test_ranking_follows_angular_distance(searcher):
    # 0 degrees: img_0 exactly, then img_1 and img_7 at 45 degrees either side.
    results = searcher.rank(_query(0), k=3)
    assert results[0].image_id == "img_0"
    assert {results[1].image_id, results[2].image_id} == {"img_1", "img_7"}


def test_opposite_vector_ranks_last(searcher):
    results = searcher.rank(_query(0), k=8)
    assert results[-1].image_id == "img_4"  # 180 degrees away
    assert results[-1].score == pytest.approx(-1.0, abs=1e-5)


def test_results_carry_path_and_captions(searcher):
    result = searcher.rank(_query(0), k=1)[0]
    assert result.path == Path("/fake/img_0.jpg")
    assert len(result.captions) == 5


def test_zero_or_negative_k_is_rejected(searcher):
    for k in (0, -1):
        with pytest.raises(SearchError, match="k must be at least 1"):
            searcher.rank(_query(0), k=k)


# ---------------------------------------------------------------------------
# Exclusion -- an image must not be its own result
# ---------------------------------------------------------------------------


def test_excluded_id_is_absent_from_results(searcher):
    results = searcher.rank(_query(0), k=8, exclude="img_0")
    assert all(result.image_id != "img_0" for result in results)


def test_exclusion_still_returns_k_results(searcher):
    assert len(searcher.rank(_query(0), k=5, exclude="img_0")) == 5


def test_exclusion_promotes_the_next_best(searcher):
    assert searcher.rank(_query(0), k=1, exclude="img_0")[0].image_id in {"img_1", "img_7"}


# ---------------------------------------------------------------------------
# Restriction -- the hard-negative evaluation needs this
# ---------------------------------------------------------------------------


def test_restriction_limits_the_candidate_set(searcher):
    allowed = ["img_2", "img_3", "img_4"]
    results = searcher.rank(_query(0), k=3, restrict_to=allowed)

    assert {result.image_id for result in results} <= set(allowed)


def test_restriction_and_exclusion_compose(searcher):
    results = searcher.rank(
        _query(90), k=2, exclude="img_2", restrict_to=["img_1", "img_2", "img_3"]
    )
    assert {result.image_id for result in results} == {"img_1", "img_3"}


def test_restriction_to_nothing_is_an_error(searcher):
    with pytest.raises(SearchError, match="no candidates"):
        searcher.rank(_query(0), k=1, restrict_to=[])


# ---------------------------------------------------------------------------
# Full ranking, for Recall@k and MRR
# ---------------------------------------------------------------------------


def test_full_ranking_scores_every_image(searcher):
    scores, ids = searcher.full_ranking(_query(0))
    assert len(scores) == len(ids) == 8


def test_full_ranking_excludes_by_pushing_to_negative_infinity(searcher):
    scores, ids = searcher.full_ranking(_query(0), exclude="img_0")
    assert scores[ids.index("img_0")] == -np.inf


def test_full_ranking_agrees_with_top_k(searcher):
    scores, ids = searcher.full_ranking(_query(37))
    best = ids[int(np.argmax(scores))]

    assert searcher.rank(_query(37), k=1)[0].image_id == best


# ---------------------------------------------------------------------------
# SearchResult validation
# ---------------------------------------------------------------------------


def test_result_rejects_an_out_of_range_score():
    # A score above 1 means something upstream stopped normalising.
    with pytest.raises(SearchError, match="outside the cosine range"):
        SearchResult(image_id="x", score=1.7, path=Path("x.jpg"), captions=(), rank=1)


def test_result_allows_the_exact_endpoints():
    for score in (1.0, -1.0):
        SearchResult(image_id="x", score=score, path=Path("x.jpg"), captions=(), rank=1)


def test_result_describe_includes_id_and_score():
    result = SearchResult(
        image_id="img_7", score=0.4213, path=Path("x.jpg"), captions=("a dog",), rank=3
    )
    text = result.describe()

    assert "img_7" in text
    assert "0.4213" in text
    assert "a dog" in text


# ---------------------------------------------------------------------------
# Query validation
# ---------------------------------------------------------------------------


def test_empty_text_query_is_rejected():
    from src.search.text_to_image import TextToImageSearcher

    searcher = TextToImageSearcher()
    for query in ("", "   ", "\n"):
        with pytest.raises(SearchError, match="query is empty"):
            searcher.search(query)


def test_missing_query_image_is_rejected(tmp_path):
    from src.search.image_to_image import ImageToImageSearcher

    searcher = ImageToImageSearcher()
    with pytest.raises(SearchError, match="image not found"):
        searcher.search(tmp_path / "nope.jpg")


# ---------------------------------------------------------------------------
# End to end, against the real model and index
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def live_index():
    """Skip the module unless a built index and cache are both present."""
    from src.embedding import cache, index

    try:
        count = index.count()
    except Exception as error:
        pytest.skip(f"no index built: {error}")

    if not cache.exists():
        pytest.skip("no embedding cache")
    return count


@pytest.mark.model
@pytest.mark.data
def test_text_search_finds_something_plausible(live_index):
    from src.search.text_to_image import TextToImageSearcher

    results = TextToImageSearcher(exact=True).search("a dog running on grass", k=5)

    assert len(results) == 5
    assert results[0].score > results[-1].score
    # A real match should be clearly positive, not marginally above zero.
    assert results[0].score > 0.2


@pytest.mark.model
@pytest.mark.data
def test_image_search_excludes_the_query(live_index):
    from src.data.dataset import load_records
    from src.search.image_to_image import ImageToImageSearcher

    record = load_records(split="test")[0]
    results = ImageToImageSearcher(exact=True).search(record.path, k=5)

    assert all(result.image_id != record.image_id for result in results)


@pytest.mark.model
@pytest.mark.data
def test_image_search_including_self_ranks_it_first(live_index):
    from src.data.dataset import load_records
    from src.search.image_to_image import ImageToImageSearcher

    record = load_records(split="test")[0]
    results = ImageToImageSearcher(exact=True).search(record.path, k=3, exclude_self=False)

    assert results[0].image_id == record.image_id
    assert results[0].score == pytest.approx(1.0, abs=1e-3)


@pytest.mark.model
@pytest.mark.data
def test_chroma_and_exact_agree_on_the_top_hit(live_index):
    """The persisted index and the exact matrix must not disagree at rank 1."""
    from src.search.text_to_image import TextToImageSearcher

    query = "two dogs playing in the snow"
    chroma = TextToImageSearcher(exact=False).search(query, k=1)[0]
    exact = TextToImageSearcher(exact=True).search(query, k=1)[0]

    assert chroma.image_id == exact.image_id
    assert chroma.score == pytest.approx(exact.score, abs=1e-4)
