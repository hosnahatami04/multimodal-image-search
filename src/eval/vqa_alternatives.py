"""Separate real VQA failures from answers the gold label was too narrow for.

Phase 4 found that retrieval Recall@1 understated the system because a query
describing "a dog catching a frisbee" has dozens of correct answers in the
corpus and only one was designated gold. The same problem appears here, in a
different shape.

For an open question like "what is the dog doing?", several answers are
correct at once. The image the model was shown is captioned "A dog is jumping
to catch a Frisbee": the gold answer is "jumping" and the model said "catching
frisbee". Scored strictly that is wrong; read honestly it is a different true
description of the same act.

Yes/no and counting questions do not have this property -- "3" is right and "4"
is not -- so the check applies only to the open-ended types.

The test is whether the model's answer is *supported by the image's captions*:
if two or more of the five annotators used the answer's content word, then two
or more people independently described the image that way and the answer is a
defensible reading. Requiring two rather than one keeps a single annotator's
idiosyncratic word from validating anything.

Words are compared on a crude stem rather than exactly, because the model
answers "catching frisbee" where the captions say "jumping to catch a Frisbee".
Those are the same verb and a literal comparison would call them unrelated --
the same lexical blind spot Phase 1 found in the caption agreement analysis and
Phase 4 found in the retrieval check. The stemmer only strips inflectional
endings (-ing, -s, -es, -ed); it does not conflate different words.

This is a lenient bound, reported alongside the strict number rather than
instead of it. Neither is the honest figure on its own.

Run it with::

    python -m src.eval.vqa_alternatives --save
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass

from src import config
from src.data.agreement import content_words
from src.data.dataset import load_records
from src.vqa.answer_matching import canonical
from src.vqa.question_types import ACTION, COLOUR, SCENE

logger = logging.getLogger(__name__)

# Only open-ended types can have more than one right answer. A counting
# question has exactly one, and a yes/no question has exactly two of which one
# is right, so crediting an "alternative" there would just be crediting error.
OPEN_TYPES = frozenset({ACTION, COLOUR, SCENE})

# How many of the five annotators must have used the word for the answer to
# count as supported. Two is the smallest number that excludes one person's
# unusual word choice.
MIN_ANNOTATORS = 2


def _stem(word: str) -> str:
    """Strip common inflectional endings so "catch" and "catching" compare equal.

    Deliberately minimal. A real stemmer would also fold "policy" into "polic"
    and create matches nobody intended; this handles the verb forms that
    actually differ between a question's answer and a caption's phrasing, and
    leaves everything else alone.
    """
    for suffix, minimum in (("ing", 5), ("ed", 4), ("es", 4), ("s", 4)):
        if word.endswith(suffix) and len(word) >= minimum:
            stem = word[: -len(suffix)]
            # "running" -> "runn" -> "run": undo the doubled consonant.
            if len(stem) > 2 and stem[-1] == stem[-2] and stem[-1] not in "aeiou":
                stem = stem[:-1]
            return stem
    return word


def _stems(words) -> set[str]:
    return {_stem(word) for word in words}


@dataclass(frozen=True)
class Alternative:
    """A wrong-by-the-label answer, judged against what the annotators wrote."""

    question_id: str
    image_id: str
    question: str
    question_type: str
    gold: str
    predicted: str
    supporting_annotators: int
    matched_words: tuple[str, ...]

    @property
    def defensible(self) -> bool:
        return self.supporting_annotators >= MIN_ANNOTATORS


def analyse() -> tuple[list[Alternative], dict]:
    """Judge every wrong open-ended answer against the image's captions."""
    results_path = config.RESULTS_DIR / "vqa.json"
    if not results_path.exists():
        raise FileNotFoundError(
            f"{results_path} not found. Run `python -m src.eval.run_vqa --save` first."
        )

    data = json.loads(results_path.read_text(encoding="utf-8"))
    records = {record.image_id: record for record in load_records()}

    alternatives: list[Alternative] = []

    for outcome in data["outcomes"]:
        if outcome["correct"] or outcome["question_type"] not in OPEN_TYPES:
            continue

        record = records[outcome["image_id"]]
        answer_words = _stems(content_words(canonical(outcome["predicted"])))
        if not answer_words:
            continue

        # Count annotators, not captions: one caption using two of the answer's
        # words is still one person agreeing.
        supporting = 0
        matched: set[str] = set()
        for caption in record.captions:
            overlap = answer_words & _stems(content_words(caption))
            if overlap:
                supporting += 1
                matched |= overlap

        alternatives.append(
            Alternative(
                question_id=outcome["question_id"],
                image_id=outcome["image_id"],
                question=outcome["question"],
                question_type=outcome["question_type"],
                gold=outcome["gold"],
                predicted=outcome["predicted"],
                supporting_annotators=supporting,
                matched_words=tuple(sorted(matched)),
            )
        )

    defensible = [item for item in alternatives if item.defensible]

    lenient_correct = data["accuracy"] * data["n"] + len(defensible)
    by_type_lenient: dict[str, float] = {}
    for name, stats in data["by_type"].items():
        extra = sum(1 for item in defensible if item.question_type == name)
        by_type_lenient[name] = round((stats["correct"] + extra) / stats["n"], 4)

    summary = {
        "open_ended_types": sorted(OPEN_TYPES),
        "min_annotators": MIN_ANNOTATORS,
        "wrong_open_ended": len(alternatives),
        "defensible": len(defensible),
        "strict_accuracy": data["accuracy"],
        "lenient_accuracy": round(lenient_correct / data["n"], 4),
        "strict_by_type": {k: v["accuracy"] for k, v in sorted(data["by_type"].items())},
        "lenient_by_type": dict(sorted(by_type_lenient.items())),
    }

    return alternatives, summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Find answers the gold label was too narrow for.")
    parser.add_argument("--save", action="store_true", help="write results/vqa_alternatives.json")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

    try:
        alternatives, summary = analyse()
    except FileNotFoundError as error:
        logger.error("%s", error)
        return 1

    print(
        f"\n{summary['wrong_open_ended']} wrong answers to open-ended questions "
        f"({', '.join(summary['open_ended_types'])})\n"
        f"  an answer counts as defensible when >= {summary['min_annotators']} of the five "
        f"annotators used its words\n"
    )
    print(f"  defensible          : {summary['defensible']}")
    print(f"  genuinely wrong     : {summary['wrong_open_ended'] - summary['defensible']}\n")
    print(f"  accuracy as measured: {summary['strict_accuracy']:.3f}")
    print(f"  crediting these     : {summary['lenient_accuracy']:.3f}\n")

    print("  by type, strict -> lenient")
    for name in sorted(summary["strict_by_type"]):
        strict = summary["strict_by_type"][name]
        lenient = summary["lenient_by_type"][name]
        arrow = "  ->" if lenient > strict else "    "
        print(f"    {name:<16} {strict:.3f}{arrow} {lenient:.3f}")

    print("\n  --- the model was describing the same act differently ---")
    for item in sorted(alternatives, key=lambda a: -a.supporting_annotators):
        if not item.defensible:
            continue
        print(f"    {item.question}  [{item.image_id}]")
        print(
            f"        gold {item.gold!r}  model {item.predicted!r}  "
            f"({item.supporting_annotators}/5 annotators used "
            f"{', '.join(item.matched_words)})"
        )

    print("\n  --- genuinely wrong ---")
    for item in sorted(alternatives, key=lambda a: a.supporting_annotators):
        if item.defensible:
            continue
        print(f"    {item.question}  [{item.image_id}]")
        print(f"        gold {item.gold!r}  model {item.predicted!r}")

    if args.save:
        config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        output = config.RESULTS_DIR / "vqa_alternatives.json"
        output.write_text(
            json.dumps(
                {
                    "summary": summary,
                    "alternatives": [
                        {
                            "question_id": item.question_id,
                            "image_id": item.image_id,
                            "question": item.question,
                            "question_type": item.question_type,
                            "gold": item.gold,
                            "predicted": item.predicted,
                            "supporting_annotators": item.supporting_annotators,
                            "matched_words": list(item.matched_words),
                            "defensible": item.defensible,
                        }
                        for item in alternatives
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
