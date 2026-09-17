"""Score the retrieval evaluation set and write the results.

Runs every query in ``eval_sets/retrieval_queries.json`` against the full
8,000-image index, records where each gold image ranked, and writes both the
aggregate metrics and the per-query outcomes to ``results/``.

Per-query outcomes are committed, not just the aggregates. Phase 6 needs to ask
"which queries missed, and do they share a pattern?", and that question cannot
be answered from a single Recall number.

Run it with::

    python -m src.eval.run_retrieval --save
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from dataclasses import asdict

from src import config
from src.eval.retrieval_metrics import (
    DEFAULT_KS,
    MetricError,
    RetrievalReport,
    evaluate,
    summarise,
)
from src.search.text_to_image import TextToImageSearcher

logger = logging.getLogger(__name__)


def load_queries() -> list[dict]:
    """Read the committed query set."""
    path = config.EVAL_SETS_DIR / "retrieval_queries.json"
    if not path.exists():
        raise MetricError(f"{path} not found. Run `python -m src.eval.build_queries` first.")
    return json.loads(path.read_text(encoding="utf-8"))


def by_category(report: RetrievalReport, queries: list[dict]) -> dict[str, dict]:
    """Break the aggregate down by query category.

    One averaged Recall hides which kinds of language the model cannot handle.
    Reporting per category is the point of tagging the queries in the first
    place, and it is what makes the counting and attribute weaknesses visible
    as numbers rather than anecdotes.
    """
    categories = {entry["query_id"]: entry["category"] for entry in queries}
    grouped: dict[str, list] = defaultdict(list)

    for outcome in report.outcomes:
        grouped[categories.get(outcome.query_id, "unknown")].append(outcome)

    out: dict[str, dict] = {}
    for category, outcomes in sorted(grouped.items()):
        sub = summarise(outcomes)
        out[category] = {
            "n": sub.num_queries,
            "recall": {str(k): round(v, 4) for k, v in sorted(sub.recall.items())},
            "mrr": round(sub.mrr, 4),
            "median_rank": sub.median_rank,
        }
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate text-to-image retrieval.")
    parser.add_argument("--save", action="store_true", help="write results/retrieval.json")
    parser.add_argument("--misses", type=int, default=8, help="how many misses to print")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

    try:
        queries = load_queries()
    except MetricError as error:
        logger.error("%s", error)
        return 1

    logger.info("scoring %d queries against the full index", len(queries))

    # The exact path, not Chroma: these metrics are claims about where the gold
    # image ranked among all 8,000, and an approximate index can drop it out of
    # the window entirely.
    searcher = TextToImageSearcher(exact=True)
    report = evaluate(searcher, queries, ks=DEFAULT_KS, progress=True)

    print()
    print(report.describe("text to image, all 8,000 candidates"))

    per_category = by_category(report, queries)
    print("\n  by category")
    width = max(len(name) for name in per_category)
    for category, stats in per_category.items():
        recalls = "  ".join(f"R@{k} {v:.2f}" for k, v in stats["recall"].items())
        print(
            f"    {category:<{width}}  n={stats['n']:<3} {recalls}  "
            f"MRR {stats['mrr']:.3f}  median rank {stats['median_rank']}"
        )

    misses = report.misses(k=5)
    if misses:
        print(f"\n  {len(misses)} queries missed the top 5:")
        categories = {entry["query_id"]: entry["category"] for entry in queries}
        for outcome in misses[: args.misses]:
            print(
                f"    rank {outcome.rank:>5}  [{categories.get(outcome.query_id, '?')}]  "
                f"{outcome.query_text}"
            )
            print(
                f"              gold {outcome.gold_score:+.4f} vs top "
                f"{outcome.top_score:+.4f} ({outcome.top_image_id}), "
                f"margin {outcome.margin:+.4f}"
            )
    else:
        print("\n  every query hit the top 5")

    if args.save:
        config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        output = config.RESULTS_DIR / "retrieval.json"
        payload = {
            "split": config.EVAL_SPLIT,
            "corpus_size": len(searcher.records),
            "model_id": config.CLIP_MODEL_ID,
            "revision": config.CLIP_REVISION,
            "num_queries": report.num_queries,
            "recall": {str(k): round(v, 4) for k, v in sorted(report.recall.items())},
            "mrr": round(report.mrr, 4),
            "median_rank": report.median_rank,
            "mean_rank": round(report.mean_rank, 2),
            "worst_rank": report.worst_rank,
            "by_category": per_category,
            "outcomes": [
                {
                    **asdict(outcome),
                    "gold_score": round(outcome.gold_score, 4),
                    "top_score": round(outcome.top_score, 4),
                    "category": next(
                        (e["category"] for e in queries if e["query_id"] == outcome.query_id),
                        "unknown",
                    ),
                }
                for outcome in report.outcomes
            ],
        }
        output.write_text(json.dumps(payload, indent=1), encoding="utf-8")
        logger.info("wrote %s", output.relative_to(config.PROJECT_ROOT))

    return 0


if __name__ == "__main__":
    sys.exit(main())
