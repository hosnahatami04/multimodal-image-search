"""Tests for the inter-annotator agreement analysis.

The properties worth pinning down are ordering properties rather than exact
values: identical captions must score higher than unrelated ones, a word shared
by all five annotators must register as full consensus, and function words must
not be able to manufacture agreement on their own.
"""

from __future__ import annotations

import pytest

from src.data.agreement import (
    analyse,
    analyse_image,
    content_words,
    jaccard,
)
from src.data.dataset import DatasetError, ImageRecord, load_records
from tests.conftest import SAMPLE_CAPTIONS


def _record(captions: tuple[str, ...], image_id: str = "img") -> ImageRecord:
    from pathlib import Path

    return ImageRecord(image_id=image_id, path=Path(f"{image_id}.jpg"), captions=captions)


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------


def test_content_words_drops_function_words():
    assert content_words("A dog is on the grass") == {"dog", "grass"}


def test_content_words_strips_punctuation_and_case():
    assert content_words("Dog, RUNNING!") == {"dog", "running"}


def test_content_words_of_only_stopwords_is_empty():
    assert content_words("the a of and to") == set()


def test_jaccard_is_one_for_identical_sets():
    assert jaccard({"dog", "grass"}, {"dog", "grass"}) == 1.0


def test_jaccard_is_zero_for_disjoint_sets():
    assert jaccard({"dog"}, {"car"}) == 0.0


def test_jaccard_of_two_empty_sets_is_zero_not_one():
    # Two captions that say nothing informative agree about nothing; returning
    # 1.0 here would silently inflate the ceiling.
    assert jaccard(set(), set()) == 0.0


def test_jaccard_is_symmetric():
    left, right = {"a", "b", "c"}, {"b", "c", "d"}
    assert jaccard(left, right) == jaccard(right, left)


# ---------------------------------------------------------------------------
# Per-image agreement
# ---------------------------------------------------------------------------


def test_identical_captions_score_perfect_agreement():
    result = analyse_image(_record(("a dog runs",) * 5))

    assert result.mean_jaccard == 1.0
    assert result.consensus_ratio == 1.0
    assert result.length_stdev == 0.0


def test_high_agreement_scores_above_low_agreement():
    high = analyse_image(_record(SAMPLE_CAPTIONS["high"]))
    low = analyse_image(_record(SAMPLE_CAPTIONS["low"]))

    assert high.mean_jaccard > low.mean_jaccard


def test_consensus_word_is_shared_by_every_annotator():
    result = analyse_image(_record(SAMPLE_CAPTIONS["high"]))

    assert result.consensus_word in {"dog", "grass", "running", "runs", "brown"}
    assert result.consensus_ratio >= 0.8


def test_consensus_prefers_a_specific_noun_over_a_generic_one():
    captions = (
        "a man rides a skateboard",
        "a man on a skateboard",
        "a man doing a skateboard trick",
        "a man with his skateboard",
        "a man and a skateboard",
    )
    result = analyse_image(_record(captions))

    # "man" appears just as often, but says far less about the image.
    assert result.consensus_word == "skateboard"


def test_min_and_max_bracket_the_mean():
    result = analyse_image(_record(SAMPLE_CAPTIONS["medium"]))

    assert result.min_jaccard <= result.mean_jaccard <= result.max_jaccard


def test_unrelated_captions_score_near_zero():
    captions = (
        "a dog on grass",
        "a car on a highway",
        "a plate of pasta",
        "snow covered mountains",
        "a laptop on a desk",
    )
    result = analyse_image(_record(captions))

    assert result.mean_jaccard == 0.0


# ---------------------------------------------------------------------------
# Corpus report
# ---------------------------------------------------------------------------


def test_report_covers_every_record(tiny_dataset):
    report, per_image = analyse(load_records())

    assert report.num_images == tiny_dataset["num_images"]
    assert len(per_image) == tiny_dataset["num_images"]


def test_report_percentiles_are_ordered(tiny_dataset):
    report, _ = analyse(load_records())

    assert report.jaccard_p10 <= report.median_jaccard <= report.jaccard_p90
    assert 0.0 <= report.mean_jaccard <= 1.0


def test_report_fractions_are_proportions(tiny_dataset):
    report, _ = analyse(load_records())

    assert 0.0 <= report.full_consensus_fraction <= 1.0
    assert 0.0 <= report.no_consensus_fraction <= 1.0


def test_least_and_most_agreed_are_ordered(tiny_dataset):
    report, per_image = analyse(load_records())
    scores = {item.image_id: item.mean_jaccard for item in per_image}

    worst = max(scores[image_id] for image_id in report.least_agreed)
    best = min(scores[image_id] for image_id in report.most_agreed)
    assert worst <= best


def test_analyse_rejects_empty_input():
    with pytest.raises(DatasetError, match="empty record set"):
        analyse([])
