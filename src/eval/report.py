"""Regenerate the evaluation report from the committed result files.

Every number in the report comes from a file in ``results/``. Nothing is typed
in by hand, so the report cannot drift away from the runs that produced it --
which is the failure mode of a README whose metrics were pasted in once and
never revisited.

The output is Markdown, written to ``results/report.md``, and the final README
in Phase 7 embeds it rather than restating it.

Run it with::

    python -m src.eval.report
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import UTC, datetime

from src import config

logger = logging.getLogger(__name__)


def _load(name: str) -> dict | None:
    path = config.RESULTS_DIR / name
    if not path.exists():
        logger.warning("%s missing; its section will be skipped", name)
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _agreement_section(data: dict | None) -> list[str]:
    if not data:
        return []
    report = data["report"]
    return [
        "## The human ceiling",
        "",
        "Each image carries five captions written independently by five people.",
        "Measured rather than spent as five times more training text, that",
        "redundancy says how ambiguous the task itself is -- and therefore how",
        "well any model could possibly do.",
        "",
        "| Measure | Value |",
        "|---|---|",
        f"| Mean pairwise Jaccard | {report['mean_jaccard']:.3f} |",
        f"| Mean subject consensus | {report['mean_consensus_ratio']:.3f} |",
        f"| All five share a subject | {report['full_consensus_fraction']:.1%} |",
        f"| No majority subject | {report['no_consensus_fraction']:.1%} |",
        f"| Caption length | {report['mean_length']:.1f} words |",
        "",
        f"The distance between {report['mean_jaccard']:.3f} and "
        f"{report['mean_consensus_ratio']:.3f} is the finding: annotators reuse very",
        "little of each other's vocabulary while agreeing on the subject far more",
        "often than not. That is the case for a shared embedding space over lexical",
        "matching, stated in the data before any model is involved.",
        "",
    ]


def _space_section(data: dict | None) -> list[str]:
    if not data:
        return []
    return [
        "## The embedding space",
        "",
        "| Comparison | Mean cosine similarity |",
        "|---|---|",
        f"| image vs image (random pairs) | {data['image_pair_mean']:.4f} |",
        f"| image vs its own captions | {data['caption_image_mean']:.4f} |",
        f"| image vs random captions | {data['random_caption_image_mean']:.4f} |",
        f"| **alignment gap** | **{data['alignment_gap']:+.4f}** |",
        "",
        f"Two unrelated images already score {data['image_pair_mean']:.2f}. CLIP's vectors",
        "occupy a narrow cone rather than the whole sphere, so an absolute similarity",
        "carries almost no information -- only the ranking, and the gap above the",
        "baseline, do. Image-image and image-text scores also sit on different scales",
        "and are not comparable to each other.",
        "",
        "![alignment](alignment.png)",
        "",
    ]


def _retrieval_section(data: dict | None, false_negatives: dict | None) -> list[str]:
    if not data:
        return []

    lines = [
        "## Text-to-image retrieval",
        "",
        f"{data['num_queries']} queries against all {data['corpus_size']:,} images. The queries are",
        "rewritten from held-out captions rather than copied: a query that is verbatim",
        "a caption in the index tests string matching dressed up as semantic search.",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Recall@1 | {data['recall']['1']:.3f} |",
        f"| Recall@5 | {data['recall']['5']:.3f} |",
        f"| Recall@10 | {data['recall']['10']:.3f} |",
        f"| MRR | {data['mrr']:.3f} |",
        f"| Median rank | {data['median_rank']} |",
        f"| Worst rank | {data['worst_rank']:,} |",
        "",
    ]

    if false_negatives:
        summary = false_negatives["summary"]
        lines += [
            "### Recall@1 is misleading here, and by how much",
            "",
            "A retrieval metric assumes one correct answer per query. Flickr8k holds",
            "dozens of photographs of dogs catching frisbees, so a query describing one",
            "of them matches all of them. When the model returns a *different* frisbee",
            "dog than the designated gold image, Recall@1 records a miss although the",
            "model did what was asked.",
            "",
            f"Of the {summary['non_rank_1']} queries that did not rank the gold image first,",
            f"**{summary['plausible_top1']}** returned an image scoring "
            f"at least {summary['threshold']} against the gold image -- well above the "
            f"{summary['random_pair_baseline']} two random images score, i.e. plainly the same",
            "kind of photograph.",
            "",
            "| | Recall@1 |",
            "|---|---|",
            f"| As measured | {summary['strict_recall_at_1']:.3f} |",
            f"| Crediting same-scene returns | {summary['lenient_recall_at_1']:.3f} |",
            "",
            "Neither number alone is honest. The first understates the system, the",
            "second is generous about what counts as correct, and the true figure sits",
            "between them -- which is why both are reported.",
            "",
        ]

    lines += [
        "### By query type",
        "",
        "One averaged recall hides which kinds of language the model cannot handle.",
        "",
        "| Type | n | R@1 | R@5 | R@10 | MRR | Median rank |",
        "|---|---|---|---|---|---|---|",
    ]
    for name, stats in sorted(
        data["by_category"].items(), key=lambda item: -item[1]["recall"]["10"]
    ):
        lines.append(
            f"| {name} | {stats['n']} | {stats['recall']['1']:.2f} | "
            f"{stats['recall']['5']:.2f} | {stats['recall']['10']:.2f} | "
            f"{stats['mrr']:.3f} | {stats['median_rank']} |"
        )
    lines += ["", "![recall by category](recall_by_category.png)", ""]
    return lines


def _vqa_section(data: dict | None) -> list[str]:
    if not data:
        return []

    lines = [
        "## Visual question answering",
        "",
        f"{data['n']} questions written by hand over "
        f"{len({o['image_id'] for o in data['outcomes']})} held-out images, each sourced from what",
        "the image's five annotators collectively establish rather than from one",
        "caption's wording.",
        "",
        "| Measure | Value |",
        "|---|---|",
        f"| Accuracy | **{data['accuracy']:.3f}** |",
        f"| Exact string match only | {data['exact_accuracy']:.3f} |",
        f"| Rescued by answer normalisation | {data['rescued_by_normalisation']} answers |",
        f"| Seconds per question (CPU) | {data['seconds_per_question']:.2f} |",
        "",
        f"The gap between {data['accuracy']:.3f} and {data['exact_accuracy']:.3f} is what exact string",
        'matching would have thrown away: "two" scored against "2", "a dog" against',
        '"dog". That is formatting, not vision, and counting it as error would',
        "misattribute the loss.",
        "",
        "### By question type",
        "",
        "This is the table the repository exists for. One averaged accuracy",
        "describes none of these categories.",
        "",
        "| Question type | n | Accuracy | Most common wrong answer |",
        "|---|---|---|---|",
    ]

    for name, stats in sorted(data["by_type"].items(), key=lambda item: -item[1]["accuracy"]):
        wrong = stats.get("most_common_wrong")
        wrong_text = f"`{wrong[0]}` (x{wrong[1]})" if wrong else "--"
        lines.append(
            f"| {name.replace('_', ' ')} | {stats['n']} | {stats['accuracy']:.3f} | {wrong_text} |"
        )

    gold_yes = data["binary_gold_yes_rate"]
    model_yes = data["binary_predicted_yes_rate"]
    lines += [
        "",
        f"On binary questions the gold answer is yes {gold_yes:.0%} of the time and the model",
        f"answers yes {model_yes:.0%} of the time. A model that simply always said yes would",
        f"score {gold_yes:.0%} on them, so that comparison is what separates seeing from",
        "guessing.",
        "",
        "![vqa accuracy](vqa_by_type.png)",
        "",
    ]
    return lines


def _hard_negative_section(data: dict | None) -> list[str]:
    if not data:
        return []
    gap = data["recall_at_1_within_group"] - data["recall_at_1_full_corpus"]
    return [
        "## Hard negatives",
        "",
        "Retrieval over 8,000 images where only a handful are even vaguely relevant",
        "is not a demanding test: the model can find the right dog photograph in a",
        'corpus of mostly beaches by recognising "dog". The harder question is what',
        "happens when every candidate is a dog photograph.",
        "",
        f"{data['num_groups']} groups of near-identical images were found by clustering the",
        f"embeddings (mean size {data['mean_group_size']}, mean within-group similarity",
        f"{data['mean_tightness']}), and a query was written by hand for one member of each.",
        "",
        "| Setting | Recall@1 |",
        "|---|---|",
        f"| Inside the group ({data['mean_group_size']} candidates) | "
        f"{data['recall_at_1_within_group']:.3f} |",
        f"| Whole corpus (8,000 candidates) | {data['recall_at_1_full_corpus']:.3f} |",
        f"| **Gap** | **{gap:+.3f}** |",
        "",
        'The within-group number is higher, which is the opposite of what "hard"',
        "usually implies, and the reason is worth stating: restricting the candidate",
        "set removes the thousands of unrelated images that could outrank the target",
        "by accident. What the gap measures is how much of the corpus-wide difficulty",
        "comes from sheer volume rather than from the near-duplicates -- and at this",
        "scale, most of it does.",
        "",
    ]


def _latency_section(data: dict | None) -> list[str]:
    if not data:
        return []
    stages = {stage["name"]: stage for stage in data["stages"]}
    encode, search, total = stages["encode"], stages["search"], stages["total"]
    share = encode["p50"] / (encode["p50"] + search["p50"])

    return [
        "## Latency",
        "",
        f"{data['runs']} queries, k={data['k']}, {data['corpus_size']:,} images, "
        f"{data['environment']['threads']} CPU threads.",
        "",
        "| Stage | p50 | p95 | p99 |",
        "|---|---|---|---|",
        f"| Text encode | {encode['p50']:.2f} ms | {encode['p95']:.2f} ms | "
        f"{encode['p99']:.2f} ms |",
        f"| Index search | {search['p50']:.2f} ms | {search['p95']:.2f} ms | "
        f"{search['p99']:.2f} ms |",
        f"| **End to end** | **{total['p50']:.2f} ms** | **{total['p95']:.2f} ms** | "
        f"{total['p99']:.2f} ms |",
        "",
        f"Text encoding is {share:.0%} of a query. The dot product over "
        f"{data['corpus_size']:,} vectors",
        "is the cheap half, which is why an approximate index would buy nothing at",
        f"this scale. Model and index load in {data['load_seconds']}s, once at startup.",
        "",
        "![latency](latency.png)",
        "",
    ]


def build() -> str:
    """Assemble the full report from whatever result files exist."""
    lines: list[str] = [
        "# Evaluation report",
        "",
        f"Generated {datetime.now(UTC).strftime('%Y-%m-%d')} from the files in `results/`.",
        f"Model `{config.CLIP_MODEL_ID}` at revision `{config.CLIP_REVISION[:12]}`,",
        f"dataset `{config.DATASET_ID}` at `{config.DATASET_REVISION[:12]}`.",
        "",
        "Every number below is read from a committed result file. Nothing is typed in",
        "by hand, so the report cannot drift away from the runs that produced it.",
        "",
        "---",
        "",
    ]

    lines += _agreement_section(_load("caption_agreement.json"))
    lines += _space_section(_load("embedding_space.json"))
    lines += _retrieval_section(_load("retrieval.json"), _load("false_negatives.json"))
    lines += _hard_negative_section(_load("hard_negatives.json"))
    lines += _vqa_section(_load("vqa.json"))
    lines += _latency_section(_load("latency.json"))

    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Regenerate results/report.md.")
    parser.add_argument("--print", action="store_true", help="also print to stdout")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

    text = build()
    output = config.RESULTS_DIR / "report.md"
    output.write_text(text, encoding="utf-8")
    logger.info("wrote %s (%d lines)", output.name, text.count("\n") + 1)

    if args.print:
        print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
