"""Describe the geometry of the embedding space that was just built.

Two things are worth measuring before any retrieval metric is computed, because
both change how those metrics should be read.

**The space is anisotropic.** CLIP vectors do not spread over the whole unit
sphere; they occupy a narrow cone, so two unrelated images still score around
0.5 against each other. A raw similarity is therefore close to meaningless on
its own -- 0.30 sounds low and is in fact a strong match. Only the *ranking*
carries information, and only the *gap* between a matched and an unmatched
score is interpretable. Recording the random-pair baseline here means every
later number can be read against it.

**Image and text occupy different regions.** The two encoders were trained to
put matching pairs closer than non-matching ones, not to interleave the two
modalities, so image-text scores sit on a different scale from image-image
scores. Comparing across the two without knowing that leads to nonsense.

Run it with::

    python -m src.embedding.stats --save
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict, dataclass

import numpy as np

from src import config
from src.data.dataset import load_records
from src.embedding import cache

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SpaceStats:
    """Geometry of the embedding space, and the baselines it implies."""

    count: int
    dim: int

    image_pair_mean: float
    image_pair_p05: float
    image_pair_p95: float
    image_pair_max: float

    caption_image_mean: float
    caption_image_p05: float
    caption_image_p95: float

    random_caption_image_mean: float
    alignment_gap: float

    def describe(self) -> str:
        return (
            f"{self.count} vectors of dimension {self.dim}\n\n"
            f"  image vs image (random pairs)\n"
            f"    mean {self.image_pair_mean:+.4f}   "
            f"p05 {self.image_pair_p05:+.4f}   p95 {self.image_pair_p95:+.4f}   "
            f"max {self.image_pair_max:+.4f}\n\n"
            f"  image vs its OWN captions\n"
            f"    mean {self.caption_image_mean:+.4f}   "
            f"p05 {self.caption_image_p05:+.4f}   p95 {self.caption_image_p95:+.4f}\n\n"
            f"  image vs RANDOM captions\n"
            f"    mean {self.random_caption_image_mean:+.4f}\n\n"
            f"  alignment gap (own - random): {self.alignment_gap:+.4f}\n"
        )


def compute(sample: int = 500, seed: int = config.SEED) -> SpaceStats:
    """Measure the space using a random sample of images.

    Args:
        sample: How many images to draw for the caption comparisons. Encoding
            5 captions each is the cost here, so this stays well under the full
            corpus; 500 images is 2,500 captions and takes under a minute.
    """
    from src.embedding.clip_encoder import ClipEncoder

    entry = cache.load()
    vectors = entry.vectors
    rng = np.random.default_rng(seed)

    # --- image vs image --------------------------------------------------
    left = rng.integers(0, len(vectors), 5000)
    right = rng.integers(0, len(vectors), 5000)
    keep = left != right
    image_pairs = np.einsum("ij,ij->i", vectors[left[keep]], vectors[right[keep]])

    # --- image vs caption ------------------------------------------------
    records = {record.image_id: record for record in load_records()}
    chosen = rng.choice(len(entry.image_ids), size=min(sample, len(entry.image_ids)), replace=False)

    encoder = ClipEncoder()
    own_scores: list[float] = []
    random_scores: list[float] = []

    captions: list[str] = []
    owners: list[int] = []
    for position in chosen:
        record = records.get(entry.image_ids[position])
        if record is None:
            continue
        for caption in record.captions:
            captions.append(caption)
            owners.append(int(position))

    logger.info("encoding %d captions from %d images", len(captions), len(chosen))
    caption_vectors = encoder.encode_texts(captions, show_progress=True)

    # Shift the owner list by a fixed offset to build the random pairing: it
    # guarantees no caption is paired with its own image, which sampling
    # independently would not.
    shifted = np.roll(np.asarray(owners), len(owners) // 2 + 1)

    for position, (owner, other) in enumerate(zip(owners, shifted, strict=True)):
        own_scores.append(float(caption_vectors[position] @ vectors[owner]))
        random_scores.append(float(caption_vectors[position] @ vectors[other]))

    own = np.asarray(own_scores)
    random_pairs = np.asarray(random_scores)

    return SpaceStats(
        count=len(vectors),
        dim=int(vectors.shape[1]),
        image_pair_mean=float(image_pairs.mean()),
        image_pair_p05=float(np.percentile(image_pairs, 5)),
        image_pair_p95=float(np.percentile(image_pairs, 95)),
        image_pair_max=float(image_pairs.max()),
        caption_image_mean=float(own.mean()),
        caption_image_p05=float(np.percentile(own, 5)),
        caption_image_p95=float(np.percentile(own, 95)),
        random_caption_image_mean=float(random_pairs.mean()),
        alignment_gap=float(own.mean() - random_pairs.mean()),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Describe the embedding space.")
    parser.add_argument("--sample", type=int, default=500, help="images to sample for captions")
    parser.add_argument("--save", action="store_true", help="write results/embedding_space.json")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

    try:
        stats = compute(sample=args.sample)
    except cache.CacheError as error:
        logger.error("%s", error)
        return 1

    print()
    print(stats.describe())
    print(
        "  Reading these numbers:\n"
        "    Two unrelated images already score around "
        f"{stats.image_pair_mean:.2f}. CLIP's vectors occupy a narrow cone rather\n"
        "    than the whole sphere, so an absolute similarity carries little\n"
        "    information -- only the ranking, and the gap above the baseline, do.\n"
    )

    if args.save:
        config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        output = config.RESULTS_DIR / "embedding_space.json"
        payload = {
            key: round(value, 4) if isinstance(value, float) else value
            for key, value in asdict(stats).items()
        }
        payload["model_id"] = config.CLIP_MODEL_ID
        payload["revision"] = config.CLIP_REVISION
        output.write_text(json.dumps(payload, indent=1), encoding="utf-8")
        logger.info("wrote %s", output.relative_to(config.PROJECT_ROOT))

    return 0


if __name__ == "__main__":
    sys.exit(main())
