"""The question taxonomy, and the rule that assigns a question to a type.

Reporting one averaged VQA accuracy hides the thing worth knowing. VQA models
do not fail uniformly -- they are strong on "is there a dog?" and weak on "how
many people?", and those two numbers averaged together describe neither. So
every question carries a type, and accuracy is reported per type.

The six types come from the project plan and match the ones the VQA literature
reports against:

* **object presence** -- "Is there a dog in this image?" The baseline. A model
  that fails here fails at seeing.
* **counting** -- "How many people are there?" Requires an exact quantity, and
  a model that groups "several" together cannot produce one.
* **colour** -- "What colour is the shirt?" Requires binding an attribute to
  the right object, not just noticing the colour is present.
* **spatial relation** -- "Is the ball to the left of the dog?" Requires a
  relation between two things, which a bag-of-concepts representation cannot
  express.
* **action** -- "What is the man doing?" Requires reading a verb off a still
  frame.
* **scene** -- "Is this indoors or outdoors?" Global context rather than any
  particular object.

Questions are typed by hand in the eval set. The classifier here exists to
catch a mistyped question, not to replace the hand labels: it is checked
against them, and a disagreement means one of the two is wrong and a person
should look.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

OBJECT_PRESENCE = "object_presence"
COUNTING = "counting"
COLOUR = "colour"
SPATIAL = "spatial"
ACTION = "action"
SCENE = "scene"

QUESTION_TYPES: tuple[str, ...] = (
    OBJECT_PRESENCE,
    COUNTING,
    COLOUR,
    SPATIAL,
    ACTION,
    SCENE,
)

TYPE_DESCRIPTIONS: dict[str, str] = {
    OBJECT_PRESENCE: "whether a named thing is in the image",
    COUNTING: "an exact quantity",
    COLOUR: "a colour bound to a specific object",
    SPATIAL: "a relation between two things",
    ACTION: "what is being done",
    SCENE: "the setting or global context",
}

# Answers that are expected to be yes/no. Used to check the eval set is
# internally consistent, not to constrain the model.
BINARY_TYPES = frozenset({OBJECT_PRESENCE, SPATIAL})

SPATIAL_WORDS = frozenset(
    {
        "left",
        "right",
        "behind",
        "front",
        "above",
        "below",
        "under",
        "underwater",
        "over",
        "beside",
        "next",
        "between",
        "top",
        "bottom",
        "near",
        "inside",
        "surrounded",
        "onto",
        "foreground",
        "background",
    }
)

# Phrases that place one thing relative to another. Checked as substrings
# because the relation is carried by the preposition plus its object, not by a
# single word: "on a table" is spatial, "on a sunny day" is not.
SPATIAL_PHRASES = (
    "coming out of",
    "out of the",
    "sitting on",
    "standing on",
    "sitting in",
    "standing in",
    "lying on",
    "in a puddle",
    "in the water",
    "on a table",
    "on a doorstep",
)

# Questions asking about setting, surface or location rather than any object.
SCENE_PHRASES = (
    "indoors",
    "outdoors",
    "inside or outside",
    "where is",
    "where are",
    "what kind of track",
    "what kind of road",
    "what kind of place",
    "what surface",
    "what season",
    "taking place",
    "is there snow",
)

COLOUR_WORDS = frozenset(
    {
        "black",
        "white",
        "red",
        "blue",
        "green",
        "yellow",
        "brown",
        "orange",
        "purple",
        "pink",
        "grey",
        "gray",
        "tan",
        "beige",
    }
)


class QuestionTypeError(ValueError):
    """Raised when a question cannot be typed or carries an unknown type."""


@dataclass(frozen=True)
class TypedQuestion:
    """One evaluation question with its image, gold answer and type."""

    question_id: str
    image_id: str
    question: str
    answer: str
    question_type: str

    def __post_init__(self) -> None:
        if self.question_type not in QUESTION_TYPES:
            raise QuestionTypeError(
                f"{self.question_id}: unknown type {self.question_type!r}; "
                f"expected one of {list(QUESTION_TYPES)}"
            )
        if not self.question.strip():
            raise QuestionTypeError(f"{self.question_id}: question is empty")
        if not self.answer.strip():
            raise QuestionTypeError(f"{self.question_id}: answer is empty")


def classify(question: str) -> str:
    """Guess a question's type from its wording.

    Ordered most-specific first: "how many red cars are there" is a counting
    question even though it mentions a colour, and "is the red ball behind the
    dog" is spatial even though it mentions a colour too.
    """
    text = question.lower().strip()
    words = set(re.findall(r"[a-z]+", text))

    if "how many" in text:
        return COUNTING

    # Scene before spatial: "where are the men?" mentions no relation, and
    # "is there snow in this image?" is about the setting rather than an object
    # someone put there.
    if any(phrase in text for phrase in SCENE_PHRASES):
        return SCENE

    if (words & SPATIAL_WORDS) or any(phrase in text for phrase in SPATIAL_PHRASES):
        return SPATIAL

    if "color" in text or "colour" in text or (words & COLOUR_WORDS):
        return COLOUR

    if "doing" in text or "what is happening" in text:
        return ACTION

    if text.startswith(("is there", "are there", "does the", "is this a", "can you see")):
        return OBJECT_PRESENCE

    return OBJECT_PRESENCE


def check_consistency(questions: list[TypedQuestion]) -> list[str]:
    """Report questions whose hand-assigned type looks wrong.

    Two checks:

    * the automatic classifier disagrees with the hand label
    * a binary-typed question has an answer that is not yes or no

    Returns a list of human-readable complaints; empty means the set is
    internally consistent. This is advisory -- the hand labels win -- but a
    complaint is worth a second look.
    """
    complaints: list[str] = []

    for item in questions:
        guessed = classify(item.question)
        if guessed != item.question_type:
            complaints.append(
                f"{item.question_id}: labelled {item.question_type!r} but reads as "
                f"{guessed!r} -- {item.question!r}"
            )

        if item.question_type in BINARY_TYPES:
            normalised = item.answer.strip().lower().rstrip(".")
            if normalised not in {"yes", "no"}:
                complaints.append(
                    f"{item.question_id}: {item.question_type!r} answered "
                    f"{item.answer!r}, expected yes or no"
                )

    return complaints


def distribution(questions: list[TypedQuestion]) -> dict[str, int]:
    """How many questions of each type, including types with none."""
    counts = dict.fromkeys(QUESTION_TYPES, 0)
    for item in questions:
        counts[item.question_type] += 1
    return counts
