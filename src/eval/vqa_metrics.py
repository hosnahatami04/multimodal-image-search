"""Score VQA answers, broken down by question type.

The breakdown is the point. A single accuracy figure over mixed question types
describes none of them: a model that answers "is there a dog?" almost always
and "how many people?" almost never averages to something that is true of
neither case, and the average moves when the question mix changes rather than
when the model does.

So accuracy is reported per type, with the counts alongside. A per-type figure
computed over sixteen questions carries real uncertainty, and stating n next to
it is what lets a reader judge how much weight the difference between two types
can bear.

Two extra numbers are tracked that most VQA reports omit:

* **exact-match accuracy** -- what the score would be without answer
  normalisation. The gap between it and the real accuracy says how much of the
  measurement was being lost to formatting rather than to vision.
* **the yes-rate on binary questions** -- a model that answers "yes" to
  everything scores well on a set where most answers are yes, and looks like it
  is seeing when it is guessing. Comparing the model's yes-rate to the gold
  yes-rate catches that.
"""

from __future__ import annotations

import logging
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field

from src.vqa.answer_matching import MatchDetail, compare
from src.vqa.question_types import BINARY_TYPES, QUESTION_TYPES

logger = logging.getLogger(__name__)


class VqaMetricError(RuntimeError):
    """Raised when accuracy cannot be computed from the inputs given."""


@dataclass(frozen=True)
class AnswerOutcome:
    """One scored question."""

    question_id: str
    image_id: str
    question: str
    question_type: str
    gold: str
    predicted: str
    detail: MatchDetail

    @property
    def correct(self) -> bool:
        return self.detail.correct

    @property
    def exact(self) -> bool:
        return self.detail.exact


@dataclass(frozen=True)
class TypeReport:
    """Accuracy for one question type."""

    question_type: str
    n: int
    correct: int
    exact_correct: int
    most_common_wrong: tuple[str, int] | None

    @property
    def accuracy(self) -> float:
        return self.correct / self.n if self.n else 0.0

    @property
    def exact_accuracy(self) -> float:
        return self.exact_correct / self.n if self.n else 0.0


@dataclass(frozen=True)
class VqaReport:
    """Overall and per-type VQA accuracy."""

    n: int
    correct: int
    exact_correct: int
    rescued_by_normalisation: int
    by_type: dict[str, TypeReport]
    binary_gold_yes_rate: float
    binary_predicted_yes_rate: float
    outcomes: list[AnswerOutcome] = field(default_factory=list, repr=False)

    @property
    def accuracy(self) -> float:
        return self.correct / self.n if self.n else 0.0

    @property
    def exact_accuracy(self) -> float:
        return self.exact_correct / self.n if self.n else 0.0

    def describe(self) -> str:
        lines = [
            f"{self.n} questions",
            f"  accuracy          {self.accuracy:.3f}  ({self.correct}/{self.n})",
            f"  exact match only  {self.exact_accuracy:.3f}  "
            f"(+{self.rescued_by_normalisation} rescued by normalisation)",
            "",
            "  by question type",
        ]
        width = max(len(name) for name in self.by_type) if self.by_type else 10
        for name in QUESTION_TYPES:
            report = self.by_type.get(name)
            if report is None or report.n == 0:
                continue
            wrong = ""
            if report.most_common_wrong:
                answer, count = report.most_common_wrong
                wrong = f'   most common wrong answer: "{answer}" (x{count})'
            lines.append(f"    {name:<{width}}  n={report.n:<3} acc {report.accuracy:.3f}{wrong}")

        lines += [
            "",
            f"  binary questions: gold says yes {self.binary_gold_yes_rate:.0%}, "
            f"model says yes {self.binary_predicted_yes_rate:.0%}",
        ]
        return "\n".join(lines)

    def failures(self, question_type: str | None = None) -> list[AnswerOutcome]:
        """Wrong answers, optionally restricted to one type."""
        return [
            outcome
            for outcome in self.outcomes
            if not outcome.correct
            and (question_type is None or outcome.question_type == question_type)
        ]


def score(
    questions: Sequence[dict],
    predictions: Sequence[str],
) -> VqaReport:
    """Score predictions against the gold answers.

    Args:
        questions: Dicts with ``question_id``, ``image_id``, ``question``,
            ``answer`` and ``question_type``.
        predictions: Model answers, aligned with ``questions``.

    Raises:
        VqaMetricError: on an empty set or a length mismatch. A silent
            misalignment here would score every answer against the wrong
            question and produce a plausible-looking number.
    """
    if not questions:
        raise VqaMetricError("cannot score an empty question set")
    if len(questions) != len(predictions):
        raise VqaMetricError(f"{len(questions)} questions but {len(predictions)} predictions")

    outcomes: list[AnswerOutcome] = []
    for entry, predicted in zip(questions, predictions, strict=True):
        text = predicted.strip() or "(no answer)"
        outcomes.append(
            AnswerOutcome(
                question_id=entry["question_id"],
                image_id=entry["image_id"],
                question=entry["question"],
                question_type=entry["question_type"],
                gold=entry["answer"],
                predicted=text,
                detail=compare(text, entry["answer"]),
            )
        )

    grouped: dict[str, list[AnswerOutcome]] = defaultdict(list)
    for outcome in outcomes:
        grouped[outcome.question_type].append(outcome)

    by_type: dict[str, TypeReport] = {}
    for question_type, items in grouped.items():
        wrong = Counter(
            outcome.detail.predicted_canonical for outcome in items if not outcome.correct
        )
        by_type[question_type] = TypeReport(
            question_type=question_type,
            n=len(items),
            correct=sum(1 for outcome in items if outcome.correct),
            exact_correct=sum(1 for outcome in items if outcome.exact),
            most_common_wrong=wrong.most_common(1)[0] if wrong else None,
        )

    binary = [outcome for outcome in outcomes if outcome.question_type in BINARY_TYPES]
    gold_yes = sum(1 for outcome in binary if outcome.gold.strip().lower() == "yes")
    predicted_yes = sum(1 for outcome in binary if outcome.detail.predicted_canonical == "yes")

    return VqaReport(
        n=len(outcomes),
        correct=sum(1 for outcome in outcomes if outcome.correct),
        exact_correct=sum(1 for outcome in outcomes if outcome.exact),
        rescued_by_normalisation=sum(
            1 for outcome in outcomes if outcome.detail.rescued_by_normalisation
        ),
        by_type=by_type,
        binary_gold_yes_rate=gold_yes / len(binary) if binary else 0.0,
        binary_predicted_yes_rate=predicted_yes / len(binary) if binary else 0.0,
        outcomes=outcomes,
    )
