"""Find the patterns behind the retrieval failures.

Phase 4 established which queries failed and separated the real failures from
the near-duplicates. This asks the next question: do the real failures share
something, or are they 25 unrelated accidents?

They share something. Reading them, the recurring shape is that the model
latches onto one salient word of the query and ignores the rest --
"a footballer striking the ball with his head" returns a dog playing with a
ball, "three men assembling a sledge on snow" returns one child on a snowy
hill. Each returned image satisfies part of the query and contradicts the rest.

This module measures that rather than asserting it. For every failed query it
computes **anchor coverage**: what fraction of the query's content words the
returned image's captions actually support. A high coverage means the model
understood the query and picked a defensible image; a low one means it matched
on a fragment.

Three failure patterns are then labelled, each by a rule stated in code rather
than by eye:

* **fragment match** -- the returned image covers a minority of the query's
  words. The model anchored on one concept.
* **constraint dropped** -- the query carries several constraints and the
  returned image satisfies the head noun but not the modifiers.
* **vocabulary gap** -- the query's distinguishing word never appears in any
  caption in this corpus. This is the subtlest of the three and worth stating
  carefully: CLIP was trained on 400 million web pairs and *does* know words
  like "crimson" and "mid-air". What is missing is not the concept but any
  image in Flickr8k that a human ever described with that word, so there is no
  caption evidence the retrieval could have leaned on. It is a property of the
  evaluation set meeting the corpus, not of the model.

  Phase 4 deliberately rewrote every query to avoid copying captions verbatim,
  and this is the cost of that choice showing up: rewriting pushed some queries
  into vocabulary the corpus never uses.

Run it with::

    python -m src.eval.retrieval_failures --save
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter
from dataclasses import dataclass

from src import config
from src.data.agreement import STOPWORDS, content_words
from src.data.dataset import load_records

logger = logging.getLogger(__name__)

# Below this share of the query's content words supported by the returned
# image, the model matched a fragment rather than the query.
FRAGMENT_THRESHOLD = 0.34

# A word that appears in *no* corpus caption marks a vocabulary gap: the
# rewritten query reached for a word no annotator used. Words that appear a few
# times are not flagged -- the model has something to work with there, and
# treating "appears 5 times" as an excuse would explain away real failures.
VOCABULARY_GAP_COUNT = 1

FRAGMENT = "fragment_match"
CONSTRAINT = "constraint_dropped"
GAP = "vocabulary_gap"
UNCLEAR = "unclear"

PATTERN_DESCRIPTIONS: dict[str, str] = {
    FRAGMENT: "the returned image supports a minority of the query's words",
    CONSTRAINT: "the head noun matched but its modifiers were ignored",
    GAP: "a query word appears in no caption in this corpus",
    UNCLEAR: "no single pattern fits",
}


@dataclass(frozen=True)
class FailurePattern:
    """One failed query, with the pattern behind it named and measured."""

    query_id: str
    query_text: str
    category: str
    rank: int
    gold_image_id: str
    top_image_id: str
    top_caption: str

    query_words: tuple[str, ...]
    covered_words: tuple[str, ...]
    missed_words: tuple[str, ...]
    coverage: float
    rarest_word: str
    rarest_count: int
    pattern: str


def _corpus_word_counts() -> Counter[str]:
    """How many captions in the whole corpus contain each content word.

    Used to tell a query the model should have handled from one whose key word
    it had almost no chance to learn.
    """
    counts: Counter[str] = Counter()
    for record in load_records():
        seen: set[str] = set()
        for caption in record.captions:
            seen |= content_words(caption)
        counts.update(seen)
    return counts


def _classify(coverage: float, missed: set[str], rarest_count: int) -> str:
    """Name the pattern from the measurements, in priority order.

    The vocabulary gap is checked first because it disqualifies the query
    rather than describing the model: if no caption uses the word, no ranking
    over those captions could have found it.
    """
    if rarest_count < VOCABULARY_GAP_COUNT:
        return GAP
    if coverage < FRAGMENT_THRESHOLD:
        return FRAGMENT
    if missed:
        return CONSTRAINT
    return UNCLEAR


def analyse() -> tuple[list[FailurePattern], dict]:
    """Label every genuine retrieval failure with the pattern behind it."""
    path = config.RESULTS_DIR / "false_negatives.json"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python -m src.eval.false_negatives --save` first."
        )

    verdicts = json.loads(path.read_text(encoding="utf-8"))["verdicts"]
    records = {record.image_id: record for record in load_records()}
    corpus_counts = _corpus_word_counts()

    patterns: list[FailurePattern] = []

    for verdict in verdicts:
        # Plausible top-1 results are not failures; Phase 4 already separated
        # them out and re-examining them here would double-count.
        if verdict["plausible"]:
            continue

        query_words = content_words(verdict["query_text"]) - STOPWORDS
        if not query_words:
            continue

        returned_words: set[str] = set()
        for caption in records[verdict["top_image_id"]].captions:
            returned_words |= content_words(caption)

        covered = query_words & returned_words
        missed = query_words - returned_words

        rarest = min(query_words, key=lambda word: corpus_counts.get(word, 0))
        rarest_count = corpus_counts.get(rarest, 0)

        coverage = len(covered) / len(query_words)

        patterns.append(
            FailurePattern(
                query_id=verdict["query_id"],
                query_text=verdict["query_text"],
                category=verdict["category"],
                rank=verdict["rank"],
                gold_image_id=verdict["gold_image_id"],
                top_image_id=verdict["top_image_id"],
                top_caption=verdict["top_caption"],
                query_words=tuple(sorted(query_words)),
                covered_words=tuple(sorted(covered)),
                missed_words=tuple(sorted(missed)),
                coverage=coverage,
                rarest_word=rarest,
                rarest_count=rarest_count,
                pattern=_classify(coverage, missed, rarest_count),
            )
        )

    by_pattern = Counter(item.pattern for item in patterns)
    by_category = Counter(item.category for item in patterns)

    mean_coverage = sum(item.coverage for item in patterns) / len(patterns) if patterns else 0.0

    summary = {
        "num_failures": len(patterns),
        "mean_coverage": round(mean_coverage, 4),
        "fragment_threshold": FRAGMENT_THRESHOLD,
        "vocabulary_gap_count": VOCABULARY_GAP_COUNT,
        "by_pattern": dict(by_pattern.most_common()),
        "by_category": dict(by_category.most_common()),
        "pattern_descriptions": PATTERN_DESCRIPTIONS,
    }

    return patterns, summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Find patterns in the retrieval failures.")
    parser.add_argument("--save", action="store_true", help="write results/retrieval_failures.json")
    parser.add_argument("--show", type=int, default=4, help="examples per pattern")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

    try:
        patterns, summary = analyse()
    except FileNotFoundError as error:
        logger.error("%s", error)
        return 1

    print(
        f"\n{summary['num_failures']} genuine retrieval failures\n"
        f"  mean coverage: {summary['mean_coverage']:.2f} of the query's content words "
        f"are supported by the returned image\n"
    )

    print("  by pattern")
    for name, count in summary["by_pattern"].items():
        print(f"    {name:<20} {count:>3}   {PATTERN_DESCRIPTIONS[name]}")

    print("\n  by query category")
    for name, count in summary["by_category"].items():
        print(f"    {name:<20} {count:>3}")

    for pattern_name in (GAP, FRAGMENT, CONSTRAINT, UNCLEAR):
        matching = [item for item in patterns if item.pattern == pattern_name]
        if not matching:
            continue
        print(f"\n  --- {pattern_name}: {PATTERN_DESCRIPTIONS[pattern_name]} ---")
        for item in sorted(matching, key=lambda x: x.coverage)[: args.show]:
            print(f'    "{item.query_text}"  (rank {item.rank})')
            print(f"        returned: {item.top_caption[:64]}")
            print(
                f"        covered {item.coverage:.0%}: "
                f"{', '.join(item.covered_words) or 'nothing'}  |  "
                f"missed: {', '.join(item.missed_words)}"
            )

    if args.save:
        config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        output = config.RESULTS_DIR / "retrieval_failures.json"
        output.write_text(
            json.dumps(
                {
                    "summary": summary,
                    "failures": [
                        {
                            "query_id": item.query_id,
                            "query_text": item.query_text,
                            "category": item.category,
                            "rank": item.rank,
                            "gold_image_id": item.gold_image_id,
                            "top_image_id": item.top_image_id,
                            "top_caption": item.top_caption,
                            "query_words": list(item.query_words),
                            "covered_words": list(item.covered_words),
                            "missed_words": list(item.missed_words),
                            "coverage": round(item.coverage, 4),
                            "rarest_word": item.rarest_word,
                            "rarest_corpus_count": item.rarest_count,
                            "pattern": item.pattern,
                        }
                        for item in sorted(patterns, key=lambda x: (x.pattern, x.coverage))
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
