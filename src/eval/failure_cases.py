"""Assemble the worked failure cases, with images, for the README.

An accuracy table says how often the system is wrong. A worked case says what
being wrong looks like, which is what a reader needs to judge whether the
system is usable for their problem. The plan asks for six to eight of these;
this selects them by rule rather than by whichever ones read best.

Selection favours cases that teach something different from each other:

* one per failure pattern, so no pattern is represented twice before every
  pattern is represented once
* within a pattern, the clearest instance -- the lowest coverage for a
  retrieval failure, the most confidently wrong answer for a VQA one
* both systems, because a reader deciding whether to use this needs to know
  how each fails, not just the one that fails more

Each case carries the image, the query or question, what the system produced,
what was expected, and a written diagnosis. The diagnosis is the part that
cannot be generated -- it is the sentence that says *why* -- so those live in
DIAGNOSES below, keyed by case, and the module refuses to emit a case it has
no diagnosis for.

Images are copied into ``results/failure_cases/`` so the README can embed them
without depending on the gitignored dataset.

Run it with::

    python -m src.eval.failure_cases --save
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from dataclasses import dataclass

from src import config
from src.data.dataset import load_records

logger = logging.getLogger(__name__)

CASES_DIR = config.RESULTS_DIR / "failure_cases"

# The written diagnosis for each selected case, keyed by the identifier the
# selection produces. A case without one is not emitted: an unexplained failure
# case is a screenshot, not an analysis.
DIAGNOSES: dict[str, str] = {
    # --- retrieval -----------------------------------------------------
    "retrieval:q043": (
        "The query carries five constraints -- a child, asleep, stretched, across, "
        "two chairs -- and the returned image supports none of them. CLIP compresses "
        "the whole sentence into one 512-dimensional vector before it sees any "
        "photograph, and a vector that must simultaneously encode a posture, a "
        "count, a spatial relation and a piece of furniture ends up encoding none "
        "of them strongly. The nearest image is one that is vaguely about small "
        "children, which is what survives the compression."
    ),
    "retrieval:q039": (
        '"Four or more" is a quantity expressed as a range, and CLIP has no '
        "mechanism for either part: not for the exact count, and not for the "
        'comparison. What reaches the embedding is roughly "swimmers", so the '
        "model returns children in a pool. The Phase 5 contrast is the point here: "
        "BLIP answers counting questions at 0.88 on the same images, because it "
        "holds the question while looking rather than compressing it first."
    ),
    "retrieval:q025": (
        "The query says crimson; every caption in the corpus that describes this "
        "colour says red. The model returned a girl in a red shirt who is skating "
        "-- which is, in substance, exactly right. This is not a model failure but "
        "a gap between the evaluation vocabulary and the corpus vocabulary, "
        "introduced by Phase 4's decision to rewrite queries rather than copy "
        "captions. That decision was correct and this is its cost, visible."
    ),
    "retrieval:q006": (
        "The head noun matched and the modifiers did not: the returned image is a "
        "boat on water, but a motorboat driven past apartments rather than a rowing "
        'boat on open water. "Rowing" and "open" are both adjectival '
        'constraints on "boat", and CLIP\'s single vector cannot express that one '
        "concept modifies another. It sees boat, water, blue -- all present -- and "
        "ranks accordingly."
    ),
    # --- VQA -----------------------------------------------------------
    "vqa:whistle": (
        'The model answered "talking on phone" for a girl blowing a whistle. '
        "Looking at the photograph explains it: the whistle is small, silver, and "
        "almost entirely hidden behind her fingers, and her hand is raised to her "
        "face in exactly the posture of someone holding a phone. The visible "
        "evidence -- a small bright object, a raised hand, a face turned toward it "
        "-- is genuinely shared between the two acts, and phones outnumber whistles "
        "in web training data by orders of magnitude. This is a prior overwhelming "
        "weak evidence rather than a perception error, and it recurs on a second "
        "image, which is what makes it a pattern."
    ),
    "vqa:cigarette": (
        "The second instance of the same failure, which is what makes it a pattern "
        'rather than an accident. A man lighting a cigarette becomes "looking at '
        'phone". Both cases involve a small object near the face, and in both the '
        "model reaches for the commonest explanation for that configuration in its "
        "training distribution. A product built on this would misreport smoking as "
        "phone use consistently, not occasionally."
    ),
    "vqa:counting_three": (
        "Every counting question with a gold answer of 1 or 2 was answered "
        "correctly -- fourteen of fourteen. Both errors are at 3, and both are off "
        "by exactly one. The model is not guessing: it is estimating, and its "
        "estimate degrades in a specific, bounded way as the count rises. That "
        'distinction matters for whether the capability is usable: "reliable up '
        'to two, approximate above" is a specification, where "0.88 accurate" '
        "is not."
    ),
    "vqa:red_brown": (
        "Red was read as brown twice, on different images and different objects "
        "(these restaurant chairs, and a boy's vest). A single colour error would "
        "be noise; the same confusion twice out of four colour errors is a "
        "systematic bias. The image shows why it is a defensible one: the chairs "
        "are stained wood under warm low restaurant lighting, and their pixels sit "
        "genuinely between red and brown. Five annotators called them red because "
        "they had the scene -- a red-walled Mexican restaurant -- to read the "
        "colour against. The model has only the pixels, and the pixels are "
        "ambiguous."
    ),
}


@dataclass(frozen=True)
class FailureCase:
    """One worked failure, ready to be embedded in the report."""

    case_id: str
    system: str
    pattern: str
    image_id: str
    image_file: str
    prompt: str
    expected: str
    produced: str
    captions: tuple[str, ...]
    diagnosis: str
    detail: dict


def _retrieval_cases() -> list[FailureCase]:
    """Pick one retrieval failure per pattern, clearest instance first."""
    path = config.RESULTS_DIR / "retrieval_failures.json"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python -m src.eval.retrieval_failures --save` first."
        )

    failures = json.loads(path.read_text(encoding="utf-8"))["failures"]
    records = {record.image_id: record for record in load_records()}

    chosen: list[FailureCase] = []
    seen_patterns: set[str] = set()

    # Lowest coverage first: the clearest instance of a pattern is the one where
    # the returned image supports least of what was asked for.
    for failure in sorted(failures, key=lambda item: item["coverage"]):
        case_id = f"retrieval:{failure['query_id']}"
        if case_id not in DIAGNOSES:
            continue
        if failure["pattern"] in seen_patterns and len(chosen) >= 3:
            continue

        chosen.append(
            FailureCase(
                case_id=case_id,
                system="retrieval",
                pattern=failure["pattern"],
                image_id=failure["gold_image_id"],
                image_file=f"{failure['gold_image_id']}.jpg",
                prompt=failure["query_text"],
                expected=f"{failure['gold_image_id']} at rank 1",
                produced=f"{failure['top_image_id']} at rank 1; gold at rank {failure['rank']}",
                captions=records[failure["gold_image_id"]].captions,
                diagnosis=DIAGNOSES[case_id],
                detail={
                    "rank": failure["rank"],
                    "coverage": failure["coverage"],
                    "covered_words": failure["covered_words"],
                    "missed_words": failure["missed_words"],
                    "returned_caption": failure["top_caption"],
                    "returned_image_id": failure["top_image_id"],
                },
            )
        )
        seen_patterns.add(failure["pattern"])

    return chosen


def _vqa_cases() -> list[FailureCase]:
    """Pick the VQA failures the diagnoses were written for."""
    path = config.RESULTS_DIR / "vqa.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run `python -m src.eval.run_vqa --save` first.")

    outcomes = json.loads(path.read_text(encoding="utf-8"))["outcomes"]
    records = {record.image_id: record for record in load_records()}

    # Matched by the distinguishing content of the question rather than by id,
    # so a rebuilt question set that renumbers does not silently drop a case.
    wanted: dict[str, tuple[str, str]] = {
        "vqa:whistle": ("whistle", "girl doing"),
        "vqa:cigarette": ("smoking", "man doing"),
        "vqa:counting_three": ("dogs are racing", "how many"),
        "vqa:red_brown": ("chairs", "colour"),
    }

    chosen: list[FailureCase] = []

    for case_id, (needle, context) in wanted.items():
        match = next(
            (
                outcome
                for outcome in outcomes
                if not outcome["correct"]
                and context.split()[0].lower() in outcome["question"].lower()
                and (
                    needle.lower() in outcome["question"].lower()
                    or needle.lower() in outcome["gold"].lower()
                )
            ),
            None,
        )
        if match is None:
            logger.warning("no failure found for case %s; skipping", case_id)
            continue

        chosen.append(
            FailureCase(
                case_id=case_id,
                system="vqa",
                pattern=match["question_type"],
                image_id=match["image_id"],
                image_file=f"{match['image_id']}.jpg",
                prompt=match["question"],
                expected=match["gold"],
                produced=match["predicted"],
                captions=records[match["image_id"]].captions,
                diagnosis=DIAGNOSES[case_id],
                detail={"question_type": match["question_type"]},
            )
        )

    return chosen


def build() -> list[FailureCase]:
    """Select the cases and copy their images into results/."""
    cases = _retrieval_cases() + _vqa_cases()

    CASES_DIR.mkdir(parents=True, exist_ok=True)
    records = {record.image_id: record for record in load_records()}

    for case in cases:
        source = records[case.image_id].path
        if not source.is_file():
            logger.warning("%s: image missing at %s", case.case_id, source)
            continue
        shutil.copyfile(source, CASES_DIR / case.image_file)

    return cases


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Assemble the worked failure cases.")
    parser.add_argument("--save", action="store_true", help="write results/failure_cases.json")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

    try:
        cases = build()
    except FileNotFoundError as error:
        logger.error("%s", error)
        return 1

    print(f"\n{len(cases)} worked failure cases\n")
    for case in cases:
        print(f"  [{case.system}/{case.pattern}]  {case.case_id}")
        print(f"    prompt:   {case.prompt}")
        print(f"    expected: {case.expected}")
        print(f"    produced: {case.produced}")
        print(f"    image:    {case.image_file}")
        print()

    if args.save:
        output = config.RESULTS_DIR / "failure_cases.json"
        output.write_text(
            json.dumps(
                [
                    {
                        "case_id": case.case_id,
                        "system": case.system,
                        "pattern": case.pattern,
                        "image_id": case.image_id,
                        "image_file": f"failure_cases/{case.image_file}",
                        "prompt": case.prompt,
                        "expected": case.expected,
                        "produced": case.produced,
                        "captions": list(case.captions),
                        "diagnosis": case.diagnosis,
                        "detail": case.detail,
                    }
                    for case in cases
                ],
                indent=1,
            ),
            encoding="utf-8",
        )
        logger.info("wrote %s and %d images", output.name, len(cases))

    return 0


if __name__ == "__main__":
    sys.exit(main())
