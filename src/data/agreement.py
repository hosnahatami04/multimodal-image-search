"""Measure how much the five human annotators agree with each other.

Flickr8k gives every image five captions written independently by five people.
That redundancy is usually treated as five times more training text. It is more
interesting than that: it is a measurement of the task's own ambiguity, and
therefore an estimate of the ceiling on what any model can achieve.

If five people look at one photograph and describe five different things, then
"the correct caption" does not exist for that image, and a retrieval system that
fails to return it for a given query is not necessarily wrong. Reading a
model's score without this number is reading a score with no scale on it.

Three complementary views are computed here, all lexical:

* **Jaccard overlap** over content words -- do the captions reuse vocabulary?
* **Subject consensus** -- does a single noun appear in most captions, i.e. do
  the annotators at least agree on what the photo is *of*?
* **Length dispersion** -- do they agree on how much detail the image warrants?

All three are deliberately shallow: they see "man" and "guy" as unrelated. The
semantic view, which does not, needs CLIP and arrives in Phase 2; these numbers
are its lower bound.

Run it with::

    python -m src.data.agreement
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics
import sys
from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from itertools import combinations

from src import config
from src.data.dataset import DatasetError, ImageRecord, load_records

logger = logging.getLogger(__name__)

# Function words carry no information about image content, and leaving them in
# makes every pair of captions look similar for reasons that have nothing to do
# with the picture. This list is deliberately small and explicit rather than
# pulled from a tokeniser package -- it is part of the method, so it should be
# readable and reviewable in the file that uses it.
STOPWORDS = frozenset(
    """
    a an the and or but if of to in on at by for with from into over under near
    is are was were be been being am do does did has have had
    this that these those there here it its his her their our your my
    up down out off above below between through across along around
    very some any each other another as than then so such no not
    """.split()
)

# Words that are grammatically nouns but describe people so generically that
# treating them as "the subject" hides real disagreement.
GENERIC_SUBJECTS = frozenset({"person", "people", "man", "woman", "boy", "girl", "child", "kid"})


@dataclass(frozen=True)
class ImageAgreement:
    """Agreement statistics for the five captions of one image."""

    image_id: str
    mean_jaccard: float
    min_jaccard: float
    max_jaccard: float
    consensus_word: str
    consensus_ratio: float
    length_mean: float
    length_stdev: float


@dataclass(frozen=True)
class AgreementReport:
    """Corpus-level summary of annotator agreement."""

    num_images: int
    split: str
    mean_jaccard: float
    median_jaccard: float
    jaccard_p10: float
    jaccard_p90: float
    mean_consensus_ratio: float
    full_consensus_fraction: float
    no_consensus_fraction: float
    mean_length: float
    mean_length_stdev: float
    least_agreed: list[str]
    most_agreed: list[str]

    def describe(self) -> str:
        return (
            f"{self.num_images} images in split {self.split!r}\n"
            f"  mean pairwise Jaccard      {self.mean_jaccard:.3f}  "
            f"(p10 {self.jaccard_p10:.3f}, median {self.median_jaccard:.3f}, "
            f"p90 {self.jaccard_p90:.3f})\n"
            f"  mean subject consensus     {self.mean_consensus_ratio:.3f}\n"
            f"  all 5 share a subject      {self.full_consensus_fraction:.1%}\n"
            f"  no majority subject        {self.no_consensus_fraction:.1%}\n"
            f"  caption length             {self.mean_length:.1f} words "
            f"(within-image sd {self.mean_length_stdev:.1f})"
        )


def content_words(caption: str) -> set[str]:
    """Reduce a caption to its informative words.

    Lowercases, strips punctuation, drops function words and pure numbers.
    Returns a set, so repetition within one caption does not inflate overlap.
    """
    words = set()
    for raw in caption.lower().split():
        word = raw.strip(".,!?;:\"'()[]-")
        if not word or word in STOPWORDS or word.isdigit():
            continue
        words.add(word)
    return words


def jaccard(left: set[str], right: set[str]) -> float:
    """Fraction of the combined vocabulary that both captions share.

    Returns 0.0 when both sides are empty rather than dividing by zero: two
    captions that say nothing informative are not in agreement about anything.
    """
    union = left | right
    if not union:
        return 0.0
    return len(left & right) / len(union)


def _subject_consensus(caption_words: Sequence[set[str]]) -> tuple[str, float]:
    """Find the word the most annotators used, and what fraction used it.

    Generic person-words are considered only if nothing more specific is shared,
    since "man" appearing in all five captions of a street scene says much less
    than "skateboard" appearing in all five.
    """
    counts: Counter[str] = Counter()
    for words in caption_words:
        counts.update(words)

    if not counts:
        return "", 0.0

    specific = {word: count for word, count in counts.items() if word not in GENERIC_SUBJECTS}
    pool = specific or dict(counts)

    best_word = max(pool, key=lambda word: (pool[word], -len(word)))
    return best_word, pool[best_word] / len(caption_words)


def analyse_image(record: ImageRecord) -> ImageAgreement:
    """Compute the agreement statistics for a single image's five captions."""
    caption_words = [content_words(caption) for caption in record.captions]

    pairwise = [
        jaccard(left, right) for left, right in combinations(caption_words, 2)
    ]

    lengths = [len(caption.split()) for caption in record.captions]
    consensus_word, consensus_ratio = _subject_consensus(caption_words)

    return ImageAgreement(
        image_id=record.image_id,
        mean_jaccard=statistics.mean(pairwise),
        min_jaccard=min(pairwise),
        max_jaccard=max(pairwise),
        consensus_word=consensus_word,
        consensus_ratio=consensus_ratio,
        length_mean=statistics.mean(lengths),
        length_stdev=statistics.stdev(lengths) if len(set(lengths)) > 1 else 0.0,
    )


def _percentile(values: Sequence[float], fraction: float) -> float:
    """Nearest-rank percentile, so the result is always an observed value."""
    if not values:
        raise DatasetError("cannot take a percentile of no values")
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round(fraction * (len(ordered) - 1))))
    return ordered[index]


def analyse(records: Sequence[ImageRecord], split: str = "all") -> tuple[AgreementReport, list[ImageAgreement]]:
    """Compute per-image agreement and fold it into a corpus-level report."""
    if not records:
        raise DatasetError("cannot measure agreement over an empty record set")

    per_image = [analyse_image(record) for record in records]
    jaccards = [item.mean_jaccard for item in per_image]
    consensus = [item.consensus_ratio for item in per_image]

    by_agreement = sorted(per_image, key=lambda item: item.mean_jaccard)

    report = AgreementReport(
        num_images=len(per_image),
        split=split,
        mean_jaccard=statistics.mean(jaccards),
        median_jaccard=statistics.median(jaccards),
        jaccard_p10=_percentile(jaccards, 0.10),
        jaccard_p90=_percentile(jaccards, 0.90),
        mean_consensus_ratio=statistics.mean(consensus),
        # Consensus is measured over five captions, so 1.0 means all five and
        # anything at or below 0.4 means at most two -- no majority.
        full_consensus_fraction=sum(1 for value in consensus if value >= 0.999) / len(consensus),
        no_consensus_fraction=sum(1 for value in consensus if value <= 0.4) / len(consensus),
        mean_length=statistics.mean(item.length_mean for item in per_image),
        mean_length_stdev=statistics.mean(item.length_stdev for item in per_image),
        least_agreed=[item.image_id for item in by_agreement[:5]],
        most_agreed=[item.image_id for item in by_agreement[-5:]],
    )

    return report, per_image


def _rounded(payload: dict[str, object], places: int = 4) -> dict[str, object]:
    """Trim float precision before the results file is committed.

    A Jaccard score carries four meaningful digits; the other thirteen that
    float repr prints are noise. Keeping them inflates the committed file by
    roughly a quarter and makes every diff of it unreadable.
    """
    return {
        key: round(value, places) if isinstance(value, float) else value
        for key, value in payload.items()
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Measure inter-annotator caption agreement.")
    parser.add_argument("--split", default=None, help="restrict to one split")
    parser.add_argument("--save", action="store_true", help="write results/caption_agreement.json")
    parser.add_argument("--examples", type=int, default=0, help="show N low-agreement examples")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

    try:
        records = load_records(split=args.split)
        report, per_image = analyse(records, split=args.split or "all")
    except DatasetError as error:
        logger.error("%s", error)
        return 1

    print(report.describe())

    if args.examples:
        lookup = {record.image_id: record for record in records}
        print("\nLowest-agreement images (the ceiling is lowest here):")
        for image_id in report.least_agreed[: args.examples]:
            print(f"\n  {image_id}  mean Jaccard "
                  f"{next(i.mean_jaccard for i in per_image if i.image_id == image_id):.3f}")
            for caption in lookup[image_id].captions:
                print(f"    - {caption}")

    if args.save:
        config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        output = config.RESULTS_DIR / "caption_agreement.json"
        output.write_text(
            json.dumps(
                {
                    "report": _rounded(asdict(report)),
                    "per_image": [_rounded(asdict(item)) for item in per_image],
                },
                indent=1,
            ),
            encoding="utf-8",
        )
        logger.info("wrote %s", output.relative_to(config.PROJECT_ROOT))

    return 0


if __name__ == "__main__":
    sys.exit(main())
