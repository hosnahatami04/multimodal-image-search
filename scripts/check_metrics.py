"""Fail the build if a committed metric has regressed.

The committed files in ``results/`` are the project's claims about itself. This
checks that they still say what the README and the report say they say, and
that nothing has silently drifted downward.

Two kinds of check, and the distinction matters:

**Thresholds** are floors below which a number would mean the system has got
worse. They are set slightly under the values actually measured -- close enough
to catch a real regression, loose enough that rerunning the evaluation on a
different machine does not fail the build over the last decimal place.

**Consistency** checks are not about quality at all: they verify that the
result files agree with each other and with the eval sets. A retrieval report
claiming 51 queries when the query file holds 50 is a bug regardless of what
the accuracy says.

This runs in CI against the committed results, so a pull request that edits a
result file by hand -- or regenerates one against a different model -- fails
rather than merging quietly.

Run it with::

    python scripts/check_metrics.py
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config

logger = logging.getLogger("check-metrics")


@dataclass(frozen=True)
class Threshold:
    """One metric with a floor it must not drop below."""

    file: str
    path: tuple[str, ...]
    minimum: float
    label: str

    def read(self, data: dict) -> float:
        value: object = data
        for key in self.path:
            value = value[key]
        return float(value)


# Floors, set just under the measured values. Raising one of these after an
# improvement is how the baseline moves up; lowering one to make CI pass is how
# a project stops noticing that it got worse.
THRESHOLDS: tuple[Threshold, ...] = (
    Threshold("retrieval.json", ("recall", "5"), 0.40, "Recall@5"),
    Threshold("retrieval.json", ("recall", "10"), 0.52, "Recall@10"),
    Threshold("retrieval.json", ("mrr",), 0.24, "MRR"),
    Threshold("vqa.json", ("accuracy",), 0.75, "VQA accuracy"),
    Threshold("vqa.json", ("by_type", "object_presence", "accuracy"), 0.85, "VQA object presence"),
    Threshold("vqa.json", ("by_type", "counting", "accuracy"), 0.80, "VQA counting"),
    Threshold("embedding_space.json", ("alignment_gap",), 0.12, "alignment gap"),
    Threshold(
        "caption_agreement.json", ("report", "mean_consensus_ratio"), 0.75, "subject consensus"
    ),
)


def _load(name: str) -> dict | list | None:
    path = config.RESULTS_DIR / name
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def check_thresholds() -> list[str]:
    """Report every metric that has fallen below its floor."""
    failures: list[str] = []

    for threshold in THRESHOLDS:
        data = _load(threshold.file)
        if data is None:
            failures.append(f"{threshold.file} is missing; cannot check {threshold.label}")
            continue

        try:
            value = threshold.read(data)
        except (KeyError, TypeError, ValueError) as error:
            failures.append(
                f"{threshold.label}: cannot read {'.'.join(threshold.path)} "
                f"from {threshold.file} ({error})"
            )
            continue

        marker = "ok  " if value >= threshold.minimum else "FAIL"
        logger.info(
            "  %s %-24s %.4f  (floor %.2f)", marker, threshold.label, value, threshold.minimum
        )

        if value < threshold.minimum:
            failures.append(
                f"{threshold.label} is {value:.4f}, below the committed floor "
                f"of {threshold.minimum:.2f}"
            )

    return failures


def check_consistency() -> list[str]:
    """Report result files that disagree with each other or with the eval sets."""
    problems: list[str] = []

    retrieval = _load("retrieval.json")
    vqa = _load("vqa.json")
    false_negatives = _load("false_negatives.json")
    failures = _load("retrieval_failures.json")
    cases = _load("failure_cases.json")

    # The retrieval report must cover exactly the committed query set.
    query_file = config.EVAL_SETS_DIR / "retrieval_queries.json"
    if retrieval and query_file.exists():
        queries = json.loads(query_file.read_text(encoding="utf-8"))
        if retrieval["num_queries"] != len(queries):
            problems.append(
                f"retrieval.json scored {retrieval['num_queries']} queries but "
                f"the query set holds {len(queries)}"
            )
        if retrieval["corpus_size"] != config.EXPECTED_NUM_IMAGES:
            problems.append(
                f"retrieval.json ranked against {retrieval['corpus_size']} images, "
                f"expected {config.EXPECTED_NUM_IMAGES}"
            )

    # The VQA report must cover exactly the committed question set.
    question_file = config.EVAL_SETS_DIR / "vqa_questions.json"
    if vqa and question_file.exists():
        questions = json.loads(question_file.read_text(encoding="utf-8"))
        if vqa["n"] != len(questions):
            problems.append(
                f"vqa.json scored {vqa['n']} questions but the question set holds {len(questions)}"
            )
        counted = sum(item["n"] for item in vqa["by_type"].values())
        if counted != vqa["n"]:
            problems.append(f"vqa.json per-type counts sum to {counted}, not {vqa['n']}")

    # Every result file must name the pinned models, so a report produced
    # against different weights cannot be committed as if it were this one.
    if retrieval and retrieval.get("revision") != config.CLIP_REVISION:
        problems.append("retrieval.json was produced at a different CLIP revision")
    if vqa and vqa.get("revision") != config.BLIP_REVISION:
        problems.append("vqa.json was produced at a different BLIP revision")

    # The failure analysis must account for exactly the real misses.
    if false_negatives and failures:
        real = false_negatives["summary"]["real_misses"]
        analysed = failures["summary"]["num_failures"]
        if real != analysed:
            problems.append(
                f"false_negatives.json reports {real} real misses but "
                f"retrieval_failures.json analysed {analysed}"
            )

    # Every worked case must have its image on disk, or the report renders
    # broken.
    if cases:
        for case in cases:
            image = config.RESULTS_DIR / case["image_file"]
            if not image.is_file():
                problems.append(f"{case['case_id']}: image missing at {case['image_file']}")

    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check committed metrics for regressions.")
    parser.add_argument("--skip-consistency", action="store_true", help="only check the thresholds")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    logger.info("thresholds")
    failures = check_thresholds()

    problems: list[str] = []
    if not args.skip_consistency:
        logger.info("\nconsistency")
        problems = check_consistency()
        if not problems:
            logger.info("  ok   result files agree with each other and the eval sets")

    if failures or problems:
        logger.error("\n%d problem(s):", len(failures) + len(problems))
        for message in (*failures, *problems):
            logger.error("  - %s", message)
        return 1

    logger.info("\nall committed metrics hold")
    return 0


if __name__ == "__main__":
    sys.exit(main())
