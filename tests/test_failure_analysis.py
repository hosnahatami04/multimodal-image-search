"""Tests for the failure-analysis layer.

The classification rules are what this phase's conclusions rest on, so they are
tested directly rather than only through the outputs they produce. A rule that
silently mislabels half the failures would still yield a plausible-looking
table.
"""

from __future__ import annotations

import json

import pytest

from src import config
from src.eval.retrieval_failures import (
    CONSTRAINT,
    FRAGMENT,
    FRAGMENT_THRESHOLD,
    GAP,
    PATTERN_DESCRIPTIONS,
    UNCLEAR,
    VOCABULARY_GAP_COUNT,
    _classify,
)

# ---------------------------------------------------------------------------
# Pattern classification
# ---------------------------------------------------------------------------


def test_a_word_absent_from_the_corpus_is_a_vocabulary_gap():
    assert _classify(coverage=0.9, missed=set(), rarest_count=0) == GAP


def test_the_vocabulary_gap_outranks_every_other_pattern():
    # Even a query the model clearly fragmented is reported as a gap when one of
    # its words appears nowhere: no ranking over those captions could have found
    # it, so the model is not what failed.
    assert _classify(coverage=0.0, missed={"crimson"}, rarest_count=0) == GAP


def test_a_rare_but_present_word_is_not_a_gap():
    # Appearing five times is little to learn from, but it is something -- and
    # treating it as an excuse would explain away real failures.
    assert _classify(coverage=0.2, missed={"bib"}, rarest_count=5) != GAP


def test_low_coverage_is_a_fragment_match():
    assert _classify(coverage=0.1, missed={"a", "b"}, rarest_count=500) == FRAGMENT


def test_zero_coverage_is_a_fragment_match():
    assert _classify(coverage=0.0, missed={"everything"}, rarest_count=500) == FRAGMENT


def test_high_coverage_with_something_missed_is_a_dropped_constraint():
    assert _classify(coverage=0.6, missed={"rowing"}, rarest_count=500) == CONSTRAINT


def test_full_coverage_fits_no_pattern():
    # Nothing was missed and coverage is complete, so the failure is not
    # explained by any of the three rules. Saying so is better than forcing it
    # into the nearest label.
    assert _classify(coverage=1.0, missed=set(), rarest_count=500) == UNCLEAR


def test_the_fragment_boundary_is_where_it_claims_to_be():
    below = _classify(FRAGMENT_THRESHOLD - 0.01, {"x"}, 500)
    at = _classify(FRAGMENT_THRESHOLD, {"x"}, 500)

    assert below == FRAGMENT
    assert at == CONSTRAINT


def test_every_pattern_has_a_description():
    for pattern in (FRAGMENT, CONSTRAINT, GAP, UNCLEAR):
        assert PATTERN_DESCRIPTIONS[pattern].strip()


def test_the_gap_threshold_means_absent_not_rare():
    # If this were raised, "rare" failures would start being excused as gaps.
    assert VOCABULARY_GAP_COUNT == 1


# ---------------------------------------------------------------------------
# The committed outputs
# ---------------------------------------------------------------------------


@pytest.mark.data
def test_retrieval_failures_file_is_coherent():
    path = config.RESULTS_DIR / "retrieval_failures.json"
    if not path.exists():
        pytest.skip("retrieval failure analysis not run")

    data = json.loads(path.read_text(encoding="utf-8"))
    summary, failures = data["summary"], data["failures"]

    assert summary["num_failures"] == len(failures)
    assert sum(summary["by_pattern"].values()) == len(failures)

    for failure in failures:
        assert 0.0 <= failure["coverage"] <= 1.0
        assert failure["pattern"] in PATTERN_DESCRIPTIONS
        # Coverage must agree with the word lists it was computed from.
        total = len(failure["covered_words"]) + len(failure["missed_words"])
        assert total == len(failure["query_words"])
        assert failure["coverage"] == pytest.approx(len(failure["covered_words"]) / total, abs=1e-3)


@pytest.mark.data
def test_covered_and_missed_words_do_not_overlap():
    path = config.RESULTS_DIR / "retrieval_failures.json"
    if not path.exists():
        pytest.skip("retrieval failure analysis not run")

    for failure in json.loads(path.read_text(encoding="utf-8"))["failures"]:
        assert not (set(failure["covered_words"]) & set(failure["missed_words"]))


@pytest.mark.data
def test_failure_cases_are_complete():
    path = config.RESULTS_DIR / "failure_cases.json"
    if not path.exists():
        pytest.skip("failure cases not built")

    cases = json.loads(path.read_text(encoding="utf-8"))

    # The plan asks for six to eight worked cases.
    assert 6 <= len(cases) <= 8
    assert len({case["case_id"] for case in cases}) == len(cases)

    for case in cases:
        assert case["system"] in {"retrieval", "vqa"}
        assert case["prompt"].strip()
        assert case["expected"].strip()
        assert case["produced"].strip()
        # A case without a diagnosis is a screenshot, not an analysis.
        assert len(case["diagnosis"]) > 120
        assert len(case["captions"]) == config.EXPECTED_CAPTIONS_PER_IMAGE


@pytest.mark.data
def test_failure_cases_cover_both_systems():
    path = config.RESULTS_DIR / "failure_cases.json"
    if not path.exists():
        pytest.skip("failure cases not built")

    systems = {case["system"] for case in json.loads(path.read_text(encoding="utf-8"))}
    assert systems == {"retrieval", "vqa"}


@pytest.mark.data
def test_failure_case_images_are_present():
    path = config.RESULTS_DIR / "failure_cases.json"
    if not path.exists():
        pytest.skip("failure cases not built")

    for case in json.loads(path.read_text(encoding="utf-8")):
        image = config.RESULTS_DIR / case["image_file"]
        assert image.is_file(), f"{case['case_id']}: {image} missing"
        assert image.stat().st_size > 1000


@pytest.mark.data
def test_the_report_embeds_the_failure_analysis():
    path = config.RESULTS_DIR / "report.md"
    if not path.exists():
        pytest.skip("report not generated")

    text = path.read_text(encoding="utf-8")

    assert "Where retrieval fails" in text
    assert "Worked failure cases" in text
    # Every figure the report references must exist next to it.
    for line in text.splitlines():
        if line.startswith("!["):
            target = line.split("](", 1)[1].rstrip(")")
            assert (config.RESULTS_DIR / target).is_file(), f"missing figure: {target}"
