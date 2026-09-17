"""Decide whether a predicted answer means the same thing as the gold answer.

VQA answers are one to three words, and exact string matching is far too
brittle for them. "two" and "2" are the same answer. So are "a dog" and "dog",
"yes." and "yes", "in a park" and "park". Scoring those as wrong would measure
formatting, not understanding, and would push accuracy down by roughly ten
points for reasons that have nothing to do with the model's vision.

The normalisation applied here, in order:

1. lowercase and strip surrounding punctuation
2. drop articles -- a, an, the
3. drop a leading preposition ("in a park" -> "park")
4. map number words to digits ("two" -> "2")
5. collapse whitespace

Two answers match when their normalised forms are equal. Nothing fuzzier is
used: edit distance would make "cat" match "hat", and synonym expansion would
need a lexicon whose coverage nobody could audit. Where a genuine synonym pair
matters ("bike" / "bicycle"), it is listed explicitly below, so every
equivalence the scorer believes in is visible in one place and can be argued
with.

The rule is documented here rather than buried in the metric, because a reader
who disagrees with it should be able to see exactly what was counted as correct.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Number words up to twenty plus the round tens, which covers every count a
# Flickr8k photograph can plausibly support.
NUMBER_WORDS: dict[str, str] = {
    "zero": "0",
    "none": "0",
    "no": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
    "eleven": "11",
    "twelve": "12",
    "thirteen": "13",
    "fourteen": "14",
    "fifteen": "15",
    "sixteen": "16",
    "seventeen": "17",
    "eighteen": "18",
    "nineteen": "19",
    "twenty": "20",
}

ARTICLES = frozenset({"a", "an", "the"})

# Stripped only when they lead the answer: "in a park" and "park" are the same
# answer to "where is this?".
LEADING_PREPOSITIONS = frozenset({"in", "on", "at", "of", "to", "by", "with", "into"})

# Explicit synonym classes. Every member of a set is treated as equal to every
# other. Kept deliberately small: each entry is a claim about the world that a
# reader can check, and a large list would stop being auditable.
SYNONYM_SETS: tuple[frozenset[str], ...] = (
    frozenset({"bike", "bicycle", "cycle"}),
    frozenset({"motorbike", "motorcycle"}),
    frozenset({"kid", "child", "youngster"}),
    frozenset({"kids", "children"}),
    frozenset({"man", "male", "guy"}),
    frozenset({"woman", "female", "lady"}),
    frozenset({"photo", "photograph", "picture", "image"}),
    frozenset({"sea", "ocean"}),
    frozenset({"outside", "outdoors", "outdoor"}),
    frozenset({"inside", "indoors", "indoor"}),
    frozenset({"football", "soccer"}),
    frozenset({"jumping", "leaping"}),
    frozenset({"running", "sprinting"}),
    frozenset({"frisbee", "disc"}),
)

_PUNCTUATION = re.compile(r"[.,!?;:\"'()\[\]]")
_WHITESPACE = re.compile(r"\s+")


class AnswerError(ValueError):
    """Raised when an answer cannot be normalised."""


def normalise(answer: str) -> str:
    """Reduce an answer to the form the comparison is made on.

    Raises:
        AnswerError: on an empty or whitespace-only answer. An empty prediction
            is a bug in the caller, and silently normalising it to "" would let
            it match a gold answer that is also empty.
    """
    if not answer or not answer.strip():
        raise AnswerError("answer is empty")

    text = _PUNCTUATION.sub(" ", answer.lower())
    text = _WHITESPACE.sub(" ", text).strip()

    words = text.split()

    # A leading preposition only -- "on the grass" -> "grass", but "sitting on
    # grass" keeps its verb.
    if words and words[0] in LEADING_PREPOSITIONS and len(words) > 1:
        words = words[1:]

    words = [word for word in words if word not in ARTICLES]
    words = [NUMBER_WORDS.get(word, word) for word in words]

    return " ".join(words)


def canonical(answer: str) -> str:
    """Normalise, then collapse any synonym to one representative of its class.

    The representative is the alphabetically first member, chosen only because
    it is stable; nothing depends on which one it is.
    """
    text = normalise(answer)

    words = []
    for word in text.split():
        replacement = word
        for group in SYNONYM_SETS:
            if word in group:
                replacement = min(group)
                break
        words.append(replacement)

    return " ".join(words)


def matches(predicted: str, gold: str) -> bool:
    """Whether a prediction should be scored as correct.

    Raises:
        AnswerError: if either side is empty.
    """
    return canonical(predicted) == canonical(gold)


@dataclass(frozen=True)
class MatchDetail:
    """Why a comparison came out the way it did, for inspecting failures."""

    predicted: str
    gold: str
    predicted_canonical: str
    gold_canonical: str
    correct: bool
    exact: bool

    @property
    def rescued_by_normalisation(self) -> bool:
        """True when normalisation turned a string mismatch into a match.

        Counting these says how much of the accuracy is real understanding and
        how much was formatting that exact matching would have thrown away.
        """
        return self.correct and not self.exact


def compare(predicted: str, gold: str) -> MatchDetail:
    """Compare two answers and report how the verdict was reached."""
    predicted_canonical = canonical(predicted)
    gold_canonical = canonical(gold)

    return MatchDetail(
        predicted=predicted,
        gold=gold,
        predicted_canonical=predicted_canonical,
        gold_canonical=gold_canonical,
        correct=predicted_canonical == gold_canonical,
        exact=predicted.strip().lower() == gold.strip().lower(),
    )
