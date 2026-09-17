"""Write the 100-question VQA evaluation set.

Every question here was written by reading an image's five captions and asking
something those five annotators collectively establish. That sourcing rule is
what makes the gold answers defensible: if four captions say "three women" and
the fifth says "women", the answer to "how many women are there?" is 3 and the
disagreement is about wording, not fact. Where the captions do not settle a
question, the question is not asked.

Questions that a single caption supports but the others contradict are
excluded, because a gold answer that only one annotator would endorse measures
that annotator, not the image.

The distribution is deliberately even across the six types -- roughly 16-17
each -- so no category's accuracy rests on three questions. Phase 6 reports per
type, and a per-type number computed over a handful of questions has error bars
wider than any difference it could show.

Run it with::

    python -m src.vqa.build_questions
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter

from src import config
from src.data.dataset import DatasetError, load_records
from src.vqa.question_types import (
    ACTION,
    COLOUR,
    COUNTING,
    OBJECT_PRESENCE,
    SCENE,
    SPATIAL,
    TypedQuestion,
    check_consistency,
    distribution,
)

logger = logging.getLogger(__name__)

# (image_id, question, gold answer, type)
QUESTIONS: tuple[tuple[str, str, str, str], ...] = (
    # ---------------------------------------------------------------
    # object presence -- 17
    # ---------------------------------------------------------------
    ("test_00008", "Is there a dog in this image?", "yes", OBJECT_PRESENCE),
    ("test_00008", "Is the dog carrying something in its mouth?", "yes", OBJECT_PRESENCE),
    ("test_00731", "Is there a child in this image?", "yes", OBJECT_PRESENCE),
    ("test_00142", "Is the child wearing a helmet?", "yes", OBJECT_PRESENCE),
    ("test_00142", "Is there a bicycle in this image?", "yes", OBJECT_PRESENCE),
    ("test_00210", "Are there motorcycles in this image?", "yes", OBJECT_PRESENCE),
    ("test_00658", "Is anyone wearing sunglasses?", "yes", OBJECT_PRESENCE),
    ("test_00658", "Is there a car in this image?", "yes", OBJECT_PRESENCE),
    ("test_00855", "Is there a boat in this image?", "yes", OBJECT_PRESENCE),
    ("test_00730", "Is the woman holding a book?", "yes", OBJECT_PRESENCE),
    ("test_00810", "Are the women wearing head scarves?", "yes", OBJECT_PRESENCE),
    ("test_00854", "Is there a frisbee in this image?", "yes", OBJECT_PRESENCE),
    ("test_00928", "Is there a dumpster in this image?", "yes", OBJECT_PRESENCE),
    ("test_00595", "Is there a rock wall in this image?", "yes", OBJECT_PRESENCE),
    ("test_00764", "Are the stadium seats mostly empty?", "yes", OBJECT_PRESENCE),
    ("test_00418", "Are the dogs wearing muzzles?", "yes", OBJECT_PRESENCE),
    ("test_00158", "Is there playground equipment in this image?", "yes", OBJECT_PRESENCE),
    # ---------------------------------------------------------------
    # counting -- 17
    # ---------------------------------------------------------------
    ("test_00764", "How many people are sitting in the stadium?", "2", COUNTING),
    ("test_00210", "How many riders are on the track?", "2", COUNTING),
    ("test_00647", "How many women are in this image?", "3", COUNTING),
    ("test_00810", "How many women are in this image?", "3", COUNTING),
    ("test_00855", "How many men are in the boat?", "2", COUNTING),
    ("test_00418", "How many dogs are racing?", "3", COUNTING),
    ("test_00008", "How many dogs are in this image?", "1", COUNTING),
    ("test_00854", "How many dogs are in this image?", "1", COUNTING),
    ("test_00731", "How many girls are in this image?", "1", COUNTING),
    ("test_00527", "How many children are jumping into the pool?", "1", COUNTING),
    ("test_00142", "How many bicycles are in this image?", "1", COUNTING),
    ("test_00347", "How many motorcycles are in this image?", "1", COUNTING),
    ("test_00024", "How many cars are in this image?", "1", COUNTING),
    ("test_00658", "How many people are in this image?", "2", COUNTING),
    ("test_00769", "How many people are in this image?", "2", COUNTING),
    ("test_00730", "How many people are in this image?", "1", COUNTING),
    ("test_00595", "How many people are climbing?", "1", COUNTING),
    # ---------------------------------------------------------------
    # colour -- 17
    # ---------------------------------------------------------------
    ("test_00764", "What colour are the ponchos?", "green", COLOUR),
    ("test_00731", "What colour is the girl's dress?", "yellow", COLOUR),
    ("test_00142", "What colour is the bike?", "purple", COLOUR),
    ("test_00142", "What colour is the helmet?", "red", COLOUR),
    ("test_00008", "What colour is the dog?", "black", COLOUR),
    ("test_00008", "What colour is the ball in the dog's mouth?", "white", COLOUR),
    ("test_00647", "What colour are the women wearing?", "green", COLOUR),
    ("test_00807", "What colour is the girl's dress?", "blue", COLOUR),
    ("test_00658", "What colour is the car?", "red", COLOUR),
    ("test_00632", "What colour are the chairs?", "red", COLOUR),
    ("test_00158", "What colour is the jungle gym?", "red", COLOUR),
    ("test_00158", "What colour is the boy's vest?", "red", COLOUR),
    ("test_00928", "What colour is the man's shirt?", "white", COLOUR),
    ("test_00595", "What colour is the climber's hat?", "white", COLOUR),
    ("test_00527", "What colour is the pool water?", "blue", COLOUR),
    ("test_00412", "What colour is the dog?", "brown", COLOUR),
    ("test_00854", "What colour is the dog?", "white", COLOUR),
    # ---------------------------------------------------------------
    # spatial -- 16
    # ---------------------------------------------------------------
    ("test_00658", "Is the car behind the people?", "yes", SPATIAL),
    ("test_00764", "Are the two people surrounded by empty seats?", "yes", SPATIAL),
    ("test_00855", "Is one man sitting in front of the other?", "yes", SPATIAL),
    ("test_00008", "Is the dog coming out of the water?", "yes", SPATIAL),
    ("test_00807", "Is the girl inside the tube?", "yes", SPATIAL),
    ("test_00854", "Is the dog above the ground?", "yes", SPATIAL),
    ("test_00854", "Is the dog underwater?", "no", SPATIAL),
    ("test_00527", "Is the boy above the water?", "yes", SPATIAL),
    ("test_00769", "Is the skier next to a tree?", "yes", SPATIAL),
    ("test_00289", "Is the man standing behind a desk?", "yes", SPATIAL),
    ("test_00730", "Is the woman sitting on a doorstep?", "yes", SPATIAL),
    ("test_00928", "Is the man next to a dumpster?", "yes", SPATIAL),
    ("test_00632", "Is there an empty chair in the foreground?", "yes", SPATIAL),
    ("test_00720", "Are the people sitting on a table?", "yes", SPATIAL),
    ("test_00210", "Are the two riders next to each other?", "yes", SPATIAL),
    ("test_00024", "Is the car in a puddle?", "yes", SPATIAL),
    # ---------------------------------------------------------------
    # action -- 17
    # ---------------------------------------------------------------
    ("test_00731", "What is the girl doing?", "blowing a whistle", ACTION),
    ("test_00527", "What is the boy doing?", "jumping", ACTION),
    ("test_00807", "What is the girl doing?", "sliding", ACTION),
    ("test_00595", "What is the person doing?", "climbing", ACTION),
    ("test_00730", "What is the woman doing?", "reading", ACTION),
    ("test_00928", "What is the man doing?", "smoking", ACTION),
    ("test_00855", "What are the men doing?", "rowing", ACTION),
    ("test_00854", "What is the dog doing?", "jumping", ACTION),
    ("test_00412", "What is the dog doing?", "running", ACTION),
    ("test_00418", "What are the dogs doing?", "racing", ACTION),
    ("test_00347", "What is the man doing?", "wheelie", ACTION),
    ("test_00210", "What are the riders doing?", "racing", ACTION),
    ("test_00142", "What is the child doing?", "riding a bike", ACTION),
    ("test_00158", "What is the boy doing?", "playing", ACTION),
    ("test_00289", "What is the man doing?", "demonstrating", ACTION),
    ("test_00769", "What is the man with the camera doing?", "filming", ACTION),
    ("test_00008", "What is the dog doing?", "walking", ACTION),
    # ---------------------------------------------------------------
    # scene -- 16
    # ---------------------------------------------------------------
    ("test_00632", "Is this indoors or outdoors?", "indoors", SCENE),
    ("test_00764", "Where is this taking place?", "stadium", SCENE),
    ("test_00008", "Where is the dog?", "beach", SCENE),
    ("test_00855", "Where are the men?", "river", SCENE),
    ("test_00210", "What kind of track is this?", "dirt", SCENE),
    ("test_00527", "Where is the boy jumping?", "pool", SCENE),
    ("test_00158", "Is this indoors or outdoors?", "outdoors", SCENE),
    ("test_00810", "Is this indoors or outdoors?", "outdoors", SCENE),
    ("test_00928", "Is this indoors or outdoors?", "outdoors", SCENE),
    ("test_00730", "Is this indoors or outdoors?", "outdoors", SCENE),
    ("test_00595", "Is this indoors or outdoors?", "outdoors", SCENE),
    ("test_00024", "What kind of road is the car on?", "dirt", SCENE),
    ("test_00412", "What surface is the dog running on?", "gravel", SCENE),
    ("test_00769", "Is there snow in this image?", "yes", SCENE),
    ("test_00647", "Is this indoors or outdoors?", "outdoors", SCENE),
    ("test_00854", "What surface is the dog on?", "grass", SCENE),
)


def build() -> list[TypedQuestion]:
    """Validate every question against the dataset and return the typed set.

    Raises:
        DatasetError: if an image is absent from the evaluation split or is
            missing from disk. A question about an image nobody can load is a
            question that will be silently skipped at scoring time.
    """
    records = {record.image_id: record for record in load_records(split=config.EVAL_SPLIT)}

    missing = sorted({image_id for image_id, _, _, _ in QUESTIONS if image_id not in records})
    if missing:
        raise DatasetError(
            f"{len(missing)} images are not in the {config.EVAL_SPLIT!r} split: "
            f"{', '.join(missing[:5])}"
        )

    absent = sorted(
        {
            image_id
            for image_id, _, _, _ in QUESTIONS
            if image_id in records and not records[image_id].exists()
        }
    )
    if absent:
        raise DatasetError(f"{len(absent)} images are missing from disk: {', '.join(absent[:5])}")

    questions = [
        TypedQuestion(
            question_id=f"v{position:03d}",
            image_id=image_id,
            question=text,
            answer=answer,
            question_type=question_type,
        )
        for position, (image_id, text, answer, question_type) in enumerate(QUESTIONS, start=1)
    ]

    # The same question asked twice about the same image would double-count.
    seen = Counter((item.image_id, item.question.lower()) for item in questions)
    duplicates = [key for key, count in seen.items() if count > 1]
    if duplicates:
        raise DatasetError(f"duplicate question/image pairs: {duplicates[:3]}")

    return questions


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Write the VQA question set.")
    parser.add_argument("--show", action="store_true", help="print each question with its captions")
    parser.add_argument(
        "--strict", action="store_true", help="fail if the consistency check complains"
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

    try:
        questions = build()
    except DatasetError as error:
        logger.error("%s", error)
        return 1

    complaints = check_consistency(questions)
    if complaints:
        logger.warning("%d questions look mistyped:", len(complaints))
        for complaint in complaints[:12]:
            logger.warning("  %s", complaint)
        if args.strict:
            return 1

    config.EVAL_SETS_DIR.mkdir(parents=True, exist_ok=True)
    output = config.EVAL_SETS_DIR / "vqa_questions.json"
    output.write_text(
        json.dumps(
            [
                {
                    "question_id": item.question_id,
                    "image_id": item.image_id,
                    "question": item.question,
                    "answer": item.answer,
                    "question_type": item.question_type,
                }
                for item in questions
            ],
            indent=1,
        ),
        encoding="utf-8",
    )

    logger.info("wrote %s (%d questions)", output.name, len(questions))
    for question_type, count in sorted(distribution(questions).items()):
        logger.info("  %-16s %d", question_type, count)
    logger.info("  %-16s %d", "distinct images", len({item.image_id for item in questions}))

    if args.show:
        records = {r.image_id: r for r in load_records(split=config.EVAL_SPLIT)}
        for item in questions:
            print(f"\n{item.question_id}  [{item.question_type}]  {item.question}")
            print(f"  gold: {item.answer}")
            for caption in records[item.image_id].captions[:2]:
                print(f"    {caption[:72]}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
