"""Tests for the VQA layer: answer matching, question typing, scoring.

Answer matching gets the most attention here, because it is the component with
the most room to be quietly wrong. A normaliser that is too aggressive inflates
accuracy by accepting answers that do not mean the same thing; one that is too
timid deflates it by rejecting answers that do. Both produce a number that
looks reasonable, so the boundary cases are pinned down explicitly.
"""

from __future__ import annotations

import json

import pytest

from src import config
from src.eval.vqa_metrics import VqaMetricError, score
from src.vqa.answer_matching import (
    AnswerError,
    canonical,
    compare,
    matches,
    normalise,
)
from src.vqa.question_types import (
    ACTION,
    COLOUR,
    COUNTING,
    OBJECT_PRESENCE,
    QUESTION_TYPES,
    SCENE,
    SPATIAL,
    QuestionTypeError,
    TypedQuestion,
    check_consistency,
    classify,
    distribution,
)

# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------


def test_case_and_punctuation_are_ignored():
    assert normalise("Yes.") == "yes"
    assert normalise("  DOG!  ") == "dog"


def test_articles_are_dropped():
    assert normalise("a dog") == "dog"
    assert normalise("the man") == "man"
    assert normalise("an apple") == "apple"


def test_number_words_become_digits():
    assert normalise("two") == "2"
    assert normalise("Three.") == "3"
    assert normalise("twenty") == "20"


def test_leading_preposition_is_dropped():
    assert normalise("in a park") == "park"
    assert normalise("on the grass") == "grass"


def test_a_preposition_inside_the_answer_is_kept():
    # "sitting on grass" is not the same answer as "grass" -- only a *leading*
    # preposition is decoration.
    assert normalise("sitting on grass") == "sitting on grass"


def test_a_bare_preposition_survives():
    # Dropping it would leave an empty answer.
    assert normalise("in") == "in"


def test_whitespace_is_collapsed():
    assert normalise("a   brown    dog") == "brown dog"


def test_empty_answer_is_rejected():
    for value in ("", "   ", "\n"):
        with pytest.raises(AnswerError, match="empty"):
            normalise(value)


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("predicted", "gold"),
    [
        ("two", "2"),
        ("2", "two"),
        ("a dog", "dog"),
        ("yes.", "yes"),
        ("The Man", "man"),
        ("in a park", "park"),
        ("bicycle", "bike"),
        ("soccer", "football"),
        ("outdoors", "outside"),
        ("children", "kids"),
    ],
)
def test_equivalent_answers_match(predicted, gold):
    assert matches(predicted, gold)


@pytest.mark.parametrize(
    ("predicted", "gold"),
    [
        ("two", "3"),
        ("yes", "no"),
        ("dog", "cat"),
        ("red", "blue"),
        ("indoors", "outdoors"),
        ("running", "jumping"),
        ("grass", "sand"),
    ],
)
def test_different_answers_do_not_match(predicted, gold):
    assert not matches(predicted, gold)


def test_near_miss_words_do_not_match():
    # A fuzzy matcher based on edit distance would call these equal. This one
    # must not: "cat" and "hat" are one character apart and entirely different
    # answers.
    assert not matches("cat", "hat")
    assert not matches("dog", "log")


def test_synonyms_collapse_to_one_representative():
    assert canonical("bicycle") == canonical("bike") == canonical("cycle")


def test_compare_reports_how_the_verdict_was_reached():
    detail = compare("Two.", "2")

    assert detail.correct
    assert not detail.exact
    assert detail.rescued_by_normalisation


def test_an_exact_match_is_not_counted_as_rescued():
    detail = compare("yes", "yes")

    assert detail.correct
    assert detail.exact
    assert not detail.rescued_by_normalisation


def test_a_wrong_answer_is_never_rescued():
    detail = compare("no", "yes")

    assert not detail.correct
    assert not detail.rescued_by_normalisation


# ---------------------------------------------------------------------------
# Question typing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("How many people are there?", COUNTING),
        ("How many red cars are in this image?", COUNTING),
        ("What colour is the shirt?", COLOUR),
        ("What color is the dog?", COLOUR),
        ("Is the ball to the left of the dog?", SPATIAL),
        ("Is the cat behind the chair?", SPATIAL),
        ("What is the man doing?", ACTION),
        ("Is this indoors or outdoors?", SCENE),
        ("Where is the dog?", SCENE),
        ("Is there a dog in this image?", OBJECT_PRESENCE),
    ],
)
def test_classifier_recognises_each_type(question, expected):
    assert classify(question) == expected


def test_counting_wins_over_colour():
    # "how many red cars" mentions a colour but is a counting question.
    assert classify("How many red cars are there?") == COUNTING


def test_scene_wins_over_spatial_for_location_questions():
    # "where are the men?" asks for a setting, not a relation between things.
    assert classify("Where are the men?") == SCENE


def _question(question_type: str, question: str, answer: str) -> TypedQuestion:
    return TypedQuestion(
        question_id="v001",
        image_id="test_00000",
        question=question,
        answer=answer,
        question_type=question_type,
    )


def test_unknown_question_type_is_rejected():
    with pytest.raises(QuestionTypeError, match="unknown type"):
        _question("vibes", "What is this?", "a dog")


def test_typed_question_rejects_an_empty_question():
    with pytest.raises(QuestionTypeError, match="question is empty"):
        _question(ACTION, "   ", "running")


def test_typed_question_rejects_an_empty_answer():
    with pytest.raises(QuestionTypeError, match="answer is empty"):
        _question(ACTION, "What is happening?", "")


def test_consistency_check_flags_a_non_binary_answer():
    complaints = check_consistency(
        [_question(OBJECT_PRESENCE, "Is there a dog in this image?", "maybe")]
    )
    assert any("expected yes or no" in complaint for complaint in complaints)


def test_consistency_check_passes_a_well_formed_question():
    assert check_consistency([_question(COUNTING, "How many dogs are there?", "2")]) == []


def test_distribution_includes_types_with_no_questions():
    counts = distribution([_question(COUNTING, "How many dogs are there?", "2")])

    assert set(counts) == set(QUESTION_TYPES)
    assert counts[COUNTING] == 1
    assert counts[ACTION] == 0


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def _entry(question_id: str, question_type: str, answer: str) -> dict:
    return {
        "question_id": question_id,
        "image_id": "test_00000",
        "question": "a question",
        "answer": answer,
        "question_type": question_type,
    }


def test_accuracy_counts_normalised_matches():
    questions = [
        _entry("v1", COUNTING, "2"),
        _entry("v2", COUNTING, "3"),
        _entry("v3", COLOUR, "red"),
    ]
    report = score(questions, ["two", "5", "Red."])

    assert report.accuracy == pytest.approx(2 / 3)
    assert report.correct == 2


def test_exact_accuracy_is_lower_than_normalised():
    questions = [_entry("v1", COUNTING, "2"), _entry("v2", COLOUR, "red")]
    report = score(questions, ["two", "Red."])

    assert report.accuracy == 1.0
    assert report.exact_accuracy == 0.0
    assert report.rescued_by_normalisation == 2


def test_per_type_accuracy_is_reported_separately():
    questions = [
        _entry("v1", COUNTING, "2"),
        _entry("v2", COUNTING, "3"),
        _entry("v3", OBJECT_PRESENCE, "yes"),
        _entry("v4", OBJECT_PRESENCE, "yes"),
    ]
    report = score(questions, ["9", "9", "yes", "yes"])

    assert report.by_type[COUNTING].accuracy == 0.0
    assert report.by_type[OBJECT_PRESENCE].accuracy == 1.0
    # The average of the two describes neither.
    assert report.accuracy == 0.5


def test_most_common_wrong_answer_is_tracked():
    questions = [_entry(f"v{i}", COUNTING, "2") for i in range(3)]
    report = score(questions, ["5", "5", "7"])

    assert report.by_type[COUNTING].most_common_wrong == ("5", 2)


def test_yes_rate_exposes_a_model_that_always_says_yes():
    questions = [
        _entry("v1", OBJECT_PRESENCE, "yes"),
        _entry("v2", OBJECT_PRESENCE, "no"),
        _entry("v3", SPATIAL, "yes"),
        _entry("v4", SPATIAL, "no"),
    ]
    report = score(questions, ["yes"] * 4)

    assert report.binary_gold_yes_rate == 0.5
    assert report.binary_predicted_yes_rate == 1.0
    # Half right, entirely by guessing.
    assert report.accuracy == 0.5


def test_an_empty_prediction_is_scored_wrong_not_crashed():
    report = score([_entry("v1", COLOUR, "red")], ["   "])

    assert not report.outcomes[0].correct
    assert report.outcomes[0].predicted == "(no answer)"


def test_length_mismatch_is_rejected():
    with pytest.raises(VqaMetricError, match="2 questions but 1 predictions"):
        score([_entry("v1", COLOUR, "red"), _entry("v2", COLOUR, "blue")], ["red"])


def test_empty_question_set_is_rejected():
    with pytest.raises(VqaMetricError, match="empty question set"):
        score([], [])


def test_failures_can_be_filtered_by_type():
    questions = [_entry("v1", COUNTING, "2"), _entry("v2", COLOUR, "red")]
    report = score(questions, ["9", "blue"])

    assert len(report.failures()) == 2
    assert len(report.failures(COUNTING)) == 1


# ---------------------------------------------------------------------------
# The committed question set
# ---------------------------------------------------------------------------


@pytest.mark.data
def test_question_set_is_well_formed():
    path = config.EVAL_SETS_DIR / "vqa_questions.json"
    if not path.exists():
        pytest.skip("question set not built")

    questions = json.loads(path.read_text(encoding="utf-8"))

    assert len(questions) == 100
    assert len({q["question_id"] for q in questions}) == 100

    counts: dict[str, int] = dict.fromkeys(QUESTION_TYPES, 0)
    for entry in questions:
        assert entry["question"].strip()
        assert entry["answer"].strip()
        assert entry["question_type"] in QUESTION_TYPES
        counts[entry["question_type"]] += 1

    # Roughly even, so no category's accuracy rests on a handful of questions.
    assert all(count >= 15 for count in counts.values())


@pytest.mark.data
def test_question_set_types_agree_with_the_classifier():
    path = config.EVAL_SETS_DIR / "vqa_questions.json"
    if not path.exists():
        pytest.skip("question set not built")

    typed = [
        TypedQuestion(
            question_id=entry["question_id"],
            image_id=entry["image_id"],
            question=entry["question"],
            answer=entry["answer"],
            question_type=entry["question_type"],
        )
        for entry in json.loads(path.read_text(encoding="utf-8"))
    ]

    assert check_consistency(typed) == []


@pytest.mark.data
def test_question_images_are_in_the_eval_split():
    path = config.EVAL_SETS_DIR / "vqa_questions.json"
    if not path.exists():
        pytest.skip("question set not built")

    from src.data.dataset import load_records

    try:
        ids = {record.image_id for record in load_records(split=config.EVAL_SPLIT)}
    except Exception as error:
        pytest.skip(f"dataset unavailable: {error}")

    for entry in json.loads(path.read_text(encoding="utf-8")):
        assert entry["image_id"] in ids


# ---------------------------------------------------------------------------
# The real model
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def blip():
    """A real BLIP model, or a skip if the weights are not cached."""
    from src.vqa.blip_vqa import BlipVqa

    instance = BlipVqa(batch_size=2)
    try:
        _ = instance.model
    except Exception as error:
        pytest.skip(f"BLIP weights unavailable: {error}")
    return instance


@pytest.mark.model
@pytest.mark.data
def test_model_answers_a_real_image(blip):
    from src.data.dataset import load_records

    try:
        record = load_records(split=config.EVAL_SPLIT, require_files=True)[0]
    except Exception as error:
        pytest.skip(f"dataset unavailable: {error}")

    answer = blip.answer(record.path, "Is there an animal in this image?")

    assert answer.strip()
    assert len(answer.split()) <= 6


@pytest.mark.model
@pytest.mark.data
def test_answers_are_deterministic(blip):
    from src.data.dataset import load_records

    try:
        record = load_records(split=config.EVAL_SPLIT, require_files=True)[0]
    except Exception as error:
        pytest.skip(f"dataset unavailable: {error}")

    first = blip.answer(record.path, "What is in this image?")
    second = blip.answer(record.path, "What is in this image?")

    assert first == second


@pytest.mark.model
def test_missing_image_is_reported_clearly(blip):
    from src.vqa.blip_vqa import VqaError

    with pytest.raises(VqaError, match="image not found"):
        blip.answer(config.IMAGES_DIR / "does_not_exist_98765.jpg", "What is this?")


@pytest.mark.model
def test_model_rejects_an_empty_question(blip):
    from src.vqa.blip_vqa import VqaError

    with pytest.raises(VqaError, match="empty question"):
        blip.answer_batch([(config.IMAGES_DIR / "anything.jpg", "  ")])


@pytest.mark.model
def test_answering_nothing_returns_nothing(blip):
    assert blip.answer_batch([]) == []
