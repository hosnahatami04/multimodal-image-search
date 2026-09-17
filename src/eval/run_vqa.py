"""Answer the evaluation questions and write the results.

Runs all 100 questions through BLIP, scores them, and writes both the aggregate
accuracy and every individual answer to ``results/``.

The per-question answers are committed, not just the totals. Phase 6 is the
failure analysis, and "which questions did it get wrong, and what did it say
instead?" cannot be answered from an accuracy figure. The wrong answers are the
material that phase works from.

Run it with::

    python -m src.eval.run_vqa --save
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import asdict

from src import config
from src.data.dataset import load_records
from src.eval.vqa_metrics import VqaMetricError, score
from src.vqa.blip_vqa import BlipVqa, VqaError

logger = logging.getLogger(__name__)


def load_questions() -> list[dict]:
    """Read the committed question set."""
    path = config.EVAL_SETS_DIR / "vqa_questions.json"
    if not path.exists():
        raise VqaMetricError(f"{path} not found. Run `python -m src.vqa.build_questions` first.")
    return json.loads(path.read_text(encoding="utf-8"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate BLIP-VQA on the question set.")
    parser.add_argument("--save", action="store_true", help="write results/vqa.json")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0, help="only the first N questions")
    parser.add_argument("--failures", type=int, default=10, help="how many failures to print")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

    try:
        questions = load_questions()
    except VqaMetricError as error:
        logger.error("%s", error)
        return 1

    if args.limit:
        questions = questions[: args.limit]

    records = {record.image_id: record for record in load_records()}
    missing = sorted({q["image_id"] for q in questions if q["image_id"] not in records})
    if missing:
        logger.error("questions reference unknown images: %s", ", ".join(missing[:5]))
        return 1

    pairs = [(records[q["image_id"]].path, q["question"]) for q in questions]

    logger.info("answering %d questions (batch %d)", len(pairs), args.batch_size)
    started = time.perf_counter()

    try:
        predictions = BlipVqa(batch_size=args.batch_size).answer_batch(pairs, show_progress=True)
    except VqaError as error:
        logger.error("%s", error)
        return 1

    elapsed = time.perf_counter() - started
    logger.info("answered in %.1f min (%.2f s per question)", elapsed / 60, elapsed / len(pairs))

    report = score(questions, predictions)

    print()
    print(report.describe())

    failures = report.failures()
    if failures:
        print(f"\n  {len(failures)} wrong answers:")
        for outcome in failures[: args.failures]:
            print(
                f"    [{outcome.question_type}] {outcome.question}\n"
                f"        gold {outcome.gold!r}  model {outcome.predicted!r}"
            )

    if args.save:
        config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        output = config.RESULTS_DIR / "vqa.json"
        output.write_text(
            json.dumps(
                {
                    "model_id": config.BLIP_VQA_MODEL_ID,
                    "revision": config.BLIP_REVISION,
                    "split": config.EVAL_SPLIT,
                    "n": report.n,
                    "accuracy": round(report.accuracy, 4),
                    "exact_accuracy": round(report.exact_accuracy, 4),
                    "rescued_by_normalisation": report.rescued_by_normalisation,
                    "seconds_per_question": round(elapsed / len(pairs), 3),
                    "binary_gold_yes_rate": round(report.binary_gold_yes_rate, 4),
                    "binary_predicted_yes_rate": round(report.binary_predicted_yes_rate, 4),
                    "by_type": {
                        name: {
                            "n": item.n,
                            "correct": item.correct,
                            "accuracy": round(item.accuracy, 4),
                            "exact_accuracy": round(item.exact_accuracy, 4),
                            "most_common_wrong": item.most_common_wrong,
                        }
                        for name, item in sorted(report.by_type.items())
                    },
                    "outcomes": [
                        {
                            "question_id": outcome.question_id,
                            "image_id": outcome.image_id,
                            "question": outcome.question,
                            "question_type": outcome.question_type,
                            "gold": outcome.gold,
                            "predicted": outcome.predicted,
                            "correct": outcome.correct,
                            "exact": outcome.exact,
                            **{
                                key: value
                                for key, value in asdict(outcome.detail).items()
                                if key.endswith("canonical")
                            },
                        }
                        for outcome in report.outcomes
                    ],
                },
                indent=1,
            ),
            encoding="utf-8",
        )
        logger.info("wrote %s", output.relative_to(config.PROJECT_ROOT))

    return 0


if __name__ == "__main__":
    sys.exit(main())
