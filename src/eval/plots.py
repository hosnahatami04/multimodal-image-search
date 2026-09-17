"""Render the figures the evaluation produces.

Three plots, each answering a question that a table of numbers answers badly:

* **Alignment distributions** -- do images sit closer to their own captions than
  to random ones? Two overlapping histograms show the separation *and* the
  overlap; a pair of means shows only the separation and hides where the shared
  space breaks down.
* **Recall by category** -- one averaged Recall hides which kinds of language
  the model cannot handle. Bars side by side make the counting and
  compositional weaknesses immediate.
* **Latency distribution** -- a p50 and a p95 are two points on a shape. The
  shape says whether the tail is a long drift or a few outliers.

Figures are written to ``results/`` and committed, because the README embeds
them and regenerating them requires the model, the cache and the index.

Run it with::

    python -m src.eval.plots
"""

from __future__ import annotations

import argparse
import json
import logging
import sys

import numpy as np

from src import config

logger = logging.getLogger(__name__)

# Muted, print-safe, and distinguishable when converted to greyscale.
OWN_COLOUR = "#2b6b52"
RANDOM_COLOUR = "#99372c"
ACCENT = "#2f5d8c"
GRID = "#d3d9e2"


def _style(axis) -> None:
    """Strip the chart down to what carries information."""
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_color(GRID)
    axis.spines["bottom"].set_color(GRID)
    axis.tick_params(colors="#4a5568", labelsize=9)
    axis.grid(axis="y", color=GRID, linewidth=0.6, alpha=0.7)
    axis.set_axisbelow(True)


def plot_alignment(sample: int = 800) -> str:
    """Histogram image-to-own-caption against image-to-random-caption."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from src.data.dataset import load_records
    from src.embedding import cache
    from src.embedding.clip_encoder import ClipEncoder
    from src.eval.retrieval_metrics import alignment

    entry = cache.load()
    records = {record.image_id: record for record in load_records()}

    rng = np.random.default_rng(config.SEED)
    chosen = rng.choice(len(entry.image_ids), size=min(sample, len(entry.image_ids)), replace=False)

    captions: list[str] = []
    owners: list[int] = []
    for position in chosen:
        record = records.get(entry.image_ids[position])
        if record is None:
            continue
        for caption in record.captions:
            captions.append(caption)
            owners.append(int(position))

    logger.info("encoding %d captions", len(captions))
    caption_vectors = ClipEncoder().encode_texts(captions, show_progress=True)

    own = np.einsum("ij,ij->i", caption_vectors, entry.vectors[np.asarray(owners)])
    shifted = np.roll(np.asarray(owners), len(owners) // 2 + 1)
    random_pairs = np.einsum("ij,ij->i", caption_vectors, entry.vectors[shifted])

    stats = alignment(entry.vectors, caption_vectors, owners)

    figure, axis = plt.subplots(figsize=(7.5, 4.2), dpi=150)
    bins = np.linspace(
        min(own.min(), random_pairs.min()) - 0.01,
        max(own.max(), random_pairs.max()) + 0.01,
        60,
    )
    axis.hist(random_pairs, bins=bins, color=RANDOM_COLOUR, alpha=0.62, label="random caption")
    axis.hist(own, bins=bins, color=OWN_COLOUR, alpha=0.62, label="its own caption")

    axis.axvline(stats["random_mean"], color=RANDOM_COLOUR, linewidth=1.4, linestyle="--")
    axis.axvline(stats["own_mean"], color=OWN_COLOUR, linewidth=1.4, linestyle="--")

    axis.annotate(
        f"gap {stats['gap']:+.3f}",
        xy=((stats["own_mean"] + stats["random_mean"]) / 2, axis.get_ylim()[1] * 0.92),
        ha="center",
        fontsize=10,
        color="#151a21",
    )

    axis.set_xlabel("cosine similarity", fontsize=10)
    axis.set_ylabel("captions", fontsize=10)
    axis.set_title(
        f"Image-caption alignment  ({len(captions):,} captions, {len(chosen)} images)",
        fontsize=11.5,
        pad=12,
    )
    axis.legend(frameon=False, fontsize=9.5)
    _style(axis)

    output = config.RESULTS_DIR / "alignment.png"
    figure.tight_layout()
    figure.savefig(output, bbox_inches="tight")
    plt.close(figure)

    (config.RESULTS_DIR / "alignment.json").write_text(
        json.dumps(
            {**{k: round(v, 4) for k, v in stats.items()}, "num_captions": len(captions)},
            indent=1,
        ),
        encoding="utf-8",
    )
    return str(output.name)


def plot_recall_by_category() -> str:
    """Grouped bars of Recall@1/5/10 per query category."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = json.loads((config.RESULTS_DIR / "retrieval.json").read_text(encoding="utf-8"))
    per_category = data["by_category"]

    names = sorted(per_category, key=lambda name: -per_category[name]["recall"]["10"])
    ks = ["1", "5", "10"]
    shades = ["#1f4f3d", "#3d8a6b", "#8fc4ae"]

    positions = np.arange(len(names))
    width = 0.26

    figure, axis = plt.subplots(figsize=(8.4, 4.2), dpi=150)
    for offset, (k, shade) in enumerate(zip(ks, shades, strict=True)):
        values = [per_category[name]["recall"][k] for name in names]
        axis.bar(
            positions + (offset - 1) * width,
            values,
            width,
            label=f"R@{k}",
            color=shade,
        )

    axis.set_xticks(positions)
    axis.set_xticklabels([f"{name}\nn={per_category[name]['n']}" for name in names], fontsize=9)
    axis.set_ylim(0, 1.0)
    axis.set_ylabel("recall", fontsize=10)
    axis.set_title(
        f"Retrieval accuracy by query type  ({data['num_queries']} queries, "
        f"{data['corpus_size']:,} candidates)",
        fontsize=11.5,
        pad=12,
    )
    axis.legend(frameon=False, fontsize=9.5, ncols=3)
    _style(axis)

    output = config.RESULTS_DIR / "recall_by_category.png"
    figure.tight_layout()
    figure.savefig(output, bbox_inches="tight")
    plt.close(figure)
    return str(output.name)


def plot_latency() -> str:
    """Show where the time goes, and the shape of the tail."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = json.loads((config.RESULTS_DIR / "latency.json").read_text(encoding="utf-8"))
    stages = {stage["name"]: stage for stage in data["stages"]}

    figure, (left, right) = plt.subplots(1, 2, figsize=(9.6, 3.8), dpi=150)

    names = ["encode", "search"]
    values = [stages[name]["p50"] for name in names]
    left.barh(names, values, color=[ACCENT, "#8fb4d6"], height=0.5)
    for position, value in enumerate(values):
        left.text(
            value + max(values) * 0.02,
            position,
            f"{value:.2f} ms",
            va="center",
            fontsize=9.5,
            color="#151a21",
        )
    left.set_xlim(0, max(values) * 1.28)
    left.set_xlabel("p50 milliseconds", fontsize=10)
    left.set_title("Where the time goes", fontsize=11, pad=10)
    _style(left)
    left.grid(axis="x", color=GRID, linewidth=0.6, alpha=0.7)
    left.grid(axis="y", visible=False)

    total = stages["total"]
    points = ["min", "p50", "p95", "p99", "max"]
    ys = [total["minimum"], total["p50"], total["p95"], total["p99"], total["maximum"]]
    right.plot(points, ys, marker="o", color=ACCENT, linewidth=1.8, markersize=5)
    for position, value in enumerate(ys):
        right.annotate(
            f"{value:.1f}",
            (position, value),
            textcoords="offset points",
            xytext=(0, 7),
            ha="center",
            fontsize=9,
            color="#4a5568",
        )
    right.set_ylim(0, max(ys) * 1.22)
    right.set_ylabel("milliseconds", fontsize=10)
    right.set_title(f"End-to-end latency ({data['runs']} queries)", fontsize=11, pad=10)
    _style(right)

    output = config.RESULTS_DIR / "latency.png"
    figure.tight_layout()
    figure.savefig(output, bbox_inches="tight")
    plt.close(figure)
    return str(output.name)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Render the evaluation figures.")
    parser.add_argument("--skip-alignment", action="store_true", help="skip the slow one")
    parser.add_argument("--sample", type=int, default=800, help="images for the alignment plot")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if not args.skip_alignment:
        logger.info("wrote %s", plot_alignment(sample=args.sample))

    logger.info("wrote %s", plot_recall_by_category())
    logger.info("wrote %s", plot_latency())
    return 0


if __name__ == "__main__":
    sys.exit(main())
