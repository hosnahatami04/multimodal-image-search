"""Separate real retrieval failures from artefacts of the evaluation set.

A retrieval metric assumes one correct answer per query. Flickr8k does not work
that way: it holds dozens of photographs of dogs catching frisbees, and a query
describing one of them matches all of them. When the model returns a different
frisbee-catching dog than the one designated gold, Recall@1 records a miss --
but the model did exactly what was asked.

Reporting Recall without measuring this overstates how badly the system
performs, and the overstatement is not small. So every non-rank-1 query is
checked: is the image the model returned actually depicting the same kind of
scene as the gold image?

The check uses **CLIP similarity between the two images**, not word overlap.
Word overlap was tried first and is too blunt -- it scores "scaling a rock face"
and "climbs up the side of a steep rock" as unrelated, which is the same
lexical blind spot Phase 1 documented in the caption agreement analysis.

The threshold is calibrated against the corpus itself rather than guessed: two
random images score about 0.52, so the cut is set well above that, at the point
where pairs are visibly the same kind of photograph.

Run it with::

    python -m src.eval.false_negatives --save
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass

from src import config
from src.data.dataset import load_records
from src.embedding import cache

logger = logging.getLogger(__name__)

# Two random Flickr8k images score ~0.52 against each other (measured in
# results/embedding_space.json). 0.75 sits far above that noise floor and, on
# inspection, is where pairs stop being "both outdoors" and start being "the
# same kind of photograph".
PLAUSIBLE_THRESHOLD = 0.75


@dataclass(frozen=True)
class Verdict:
    """Whether one missed query was a real failure or an evaluation artefact."""

    query_id: str
    query_text: str
    category: str
    rank: int
    gold_image_id: str
    top_image_id: str
    image_similarity: float
    top_caption: str
    gold_caption: str

    @property
    def plausible(self) -> bool:
        return self.image_similarity >= PLAUSIBLE_THRESHOLD


def analyse(threshold: float = PLAUSIBLE_THRESHOLD) -> tuple[list[Verdict], dict]:
    """Classify every non-rank-1 query as a plausible match or a real miss."""
    results_path = config.RESULTS_DIR / "retrieval.json"
    if not results_path.exists():
        raise FileNotFoundError(
            f"{results_path} not found. Run `python -m src.eval.run_retrieval --save` first."
        )

    data = json.loads(results_path.read_text(encoding="utf-8"))
    entry = cache.load()
    row_of = {image_id: position for position, image_id in enumerate(entry.image_ids)}
    records = {record.image_id: record for record in load_records()}

    verdicts: list[Verdict] = []

    for outcome in data["outcomes"]:
        if outcome["rank"] == 1:
            continue

        gold_id = outcome["gold_image_id"]
        top_id = outcome["top_image_id"]

        similarity = float(entry.vectors[row_of[gold_id]] @ entry.vectors[row_of[top_id]])

        verdicts.append(
            Verdict(
                query_id=outcome["query_id"],
                query_text=outcome["query_text"],
                category=outcome.get("category", "unknown"),
                rank=outcome["rank"],
                gold_image_id=gold_id,
                top_image_id=top_id,
                image_similarity=similarity,
                top_caption=records[top_id].captions[0],
                gold_caption=records[gold_id].captions[0],
            )
        )

    plausible = [verdict for verdict in verdicts if verdict.image_similarity >= threshold]
    real = [verdict for verdict in verdicts if verdict.image_similarity < threshold]

    # Recall@1 if plausible top-1 matches were credited as correct. This is not
    # the headline number -- it is the upper bound the headline sits inside.
    num_queries = data["num_queries"]
    strict_hits = num_queries - len(verdicts)
    lenient_at_1 = (strict_hits + len(plausible)) / num_queries

    summary = {
        "threshold": threshold,
        "num_queries": num_queries,
        "non_rank_1": len(verdicts),
        "plausible_top1": len(plausible),
        "real_misses": len(real),
        "strict_recall_at_1": round(data["recall"]["1"], 4),
        "lenient_recall_at_1": round(lenient_at_1, 4),
        "random_pair_baseline": 0.52,
    }

    return verdicts, summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Separate real misses from near-duplicates.")
    parser.add_argument("--threshold", type=float, default=PLAUSIBLE_THRESHOLD)
    parser.add_argument("--save", action="store_true", help="write results/false_negatives.json")
    parser.add_argument("--show", type=int, default=6, help="examples to print per class")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

    try:
        verdicts, summary = analyse(args.threshold)
    except FileNotFoundError as error:
        logger.error("%s", error)
        return 1

    print(
        f"\n{summary['num_queries']} queries, {summary['non_rank_1']} did not rank the "
        f"gold image first\n"
        f"  image-image similarity >= {args.threshold} counts as the same kind of scene\n"
        f"  (two random corpus images score about {summary['random_pair_baseline']})\n"
    )
    print(f"  plausible top-1, scored as a miss : {summary['plausible_top1']}")
    print(f"  genuinely wrong top-1             : {summary['real_misses']}\n")
    print(f"  Recall@1 as measured              : {summary['strict_recall_at_1']:.3f}")
    print(f"  Recall@1 crediting plausible hits : {summary['lenient_recall_at_1']:.3f}")

    ordered = sorted(verdicts, key=lambda verdict: -verdict.image_similarity)

    print("\n  --- returned a different photograph of the same thing ---")
    for verdict in [v for v in ordered if v.plausible][: args.show]:
        print(
            f"    sim {verdict.image_similarity:.3f}  rank {verdict.rank:<5} {verdict.query_text}"
        )
        print(f"        gold:     {verdict.gold_caption[:66]}")
        print(f"        returned: {verdict.top_caption[:66]}")

    print("\n  --- genuinely wrong ---")
    for verdict in [v for v in ordered if not v.plausible][-args.show :]:
        print(
            f"    sim {verdict.image_similarity:.3f}  rank {verdict.rank:<5} {verdict.query_text}"
        )
        print(f"        gold:     {verdict.gold_caption[:66]}")
        print(f"        returned: {verdict.top_caption[:66]}")

    if args.save:
        config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        output = config.RESULTS_DIR / "false_negatives.json"
        output.write_text(
            json.dumps(
                {
                    "summary": summary,
                    "verdicts": [
                        {
                            "query_id": verdict.query_id,
                            "query_text": verdict.query_text,
                            "category": verdict.category,
                            "rank": verdict.rank,
                            "gold_image_id": verdict.gold_image_id,
                            "top_image_id": verdict.top_image_id,
                            "image_similarity": round(verdict.image_similarity, 4),
                            "plausible": verdict.plausible,
                            "gold_caption": verdict.gold_caption,
                            "top_caption": verdict.top_caption,
                        }
                        for verdict in ordered
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
