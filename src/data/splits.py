"""Freeze the train/validation/test assignment into a committed file.

Every later phase reads its image ids from ``data/splits.json`` rather than
re-deriving them. This is the single most effective guard against
irreproducible numbers: if the split is recomputed at each phase, a change in
ordering, in library version, or in seed handling silently changes which images
are being scored, and the resulting metric drift is indistinguishable from a
real change in model quality.

We adopt the dataset's own split (the standard Karpathy 6000/1000/1000) instead
of reshuffling with our own seed. Published Flickr8k retrieval numbers are
reported against that split, so keeping it makes our results comparable to the
literature rather than to nothing. The seeded-shuffle path remains available
behind ``--strategy random`` for anyone who wants an independent partition.

Run it with::

    python -m src.data.splits
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import random
import sys
from dataclasses import dataclass

from src import config
from src.data.dataset import DatasetError, load_records

logger = logging.getLogger(__name__)

STRATEGY_OFFICIAL = "official"
STRATEGY_RANDOM = "random"


@dataclass(frozen=True)
class Splits:
    """The frozen partition of image ids, plus the provenance to reproduce it."""

    strategy: str
    seed: int
    splits: dict[str, list[str]]
    digest: str

    @property
    def train(self) -> list[str]:
        return self.splits["train"]

    @property
    def validation(self) -> list[str]:
        return self.splits["validation"]

    @property
    def test(self) -> list[str]:
        return self.splits["test"]

    def ids_for(self, split: str) -> list[str]:
        if split not in self.splits:
            raise DatasetError(f"unknown split {split!r}; have {sorted(self.splits)}")
        return self.splits[split]

    def counts(self) -> dict[str, int]:
        return {name: len(ids) for name, ids in sorted(self.splits.items())}


def _digest(splits: dict[str, list[str]]) -> str:
    """Hash the exact split contents.

    Recorded alongside the split so a later phase can detect a hand-edited or
    regenerated file instead of trusting it.
    """
    hasher = hashlib.sha256()
    for name in sorted(splits):
        hasher.update(name.encode("utf-8"))
        for image_id in splits[name]:
            hasher.update(image_id.encode("utf-8"))
    return hasher.hexdigest()


def build_splits(strategy: str = STRATEGY_OFFICIAL, seed: int = config.SEED) -> Splits:
    """Compute the split assignment from the records on disk.

    Args:
        strategy: ``official`` keeps the dataset's own partition. ``random``
            pools every image and re-partitions with a seeded shuffle, using
            the same split sizes.
        seed: Seed for the ``random`` strategy; ignored by ``official``.
    """
    records = load_records()

    if strategy == STRATEGY_OFFICIAL:
        splits: dict[str, list[str]] = {name: [] for name in config.EXPECTED_SPLIT_SIZES}
        for record in records:
            if record.split not in splits:
                raise DatasetError(
                    f"{record.image_id}: unexpected split {record.split!r}; "
                    f"expected one of {sorted(splits)}"
                )
            splits[record.split].append(record.image_id)

    elif strategy == STRATEGY_RANDOM:
        # Sort before shuffling so the result depends only on the seed, never
        # on the order rows happened to come back from the CSV reader.
        image_ids = sorted(record.image_id for record in records)
        random.Random(seed).shuffle(image_ids)

        splits = {}
        cursor = 0
        for name, size in config.EXPECTED_SPLIT_SIZES.items():
            splits[name] = sorted(image_ids[cursor : cursor + size])
            cursor += size
        if cursor != len(image_ids):
            raise DatasetError(
                f"split sizes sum to {cursor} but there are {len(image_ids)} images"
            )

    else:
        raise DatasetError(
            f"unknown strategy {strategy!r}; expected {STRATEGY_OFFICIAL!r} or {STRATEGY_RANDOM!r}"
        )

    for name, expected in config.EXPECTED_SPLIT_SIZES.items():
        actual = len(splits[name])
        if actual != expected:
            raise DatasetError(f"split {name!r}: expected {expected} images, built {actual}")

    all_ids = [image_id for ids in splits.values() for image_id in ids]
    if len(set(all_ids)) != len(all_ids):
        raise DatasetError("the same image id appears in more than one split")

    return Splits(
        strategy=strategy,
        seed=seed,
        splits={name: sorted(ids) for name, ids in splits.items()},
        digest=_digest({name: sorted(ids) for name, ids in splits.items()}),
    )


def write_splits(splits: Splits) -> None:
    """Persist the split file that every later phase reads."""
    config.SPLITS_FILE.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "strategy": splits.strategy,
        "seed": splits.seed,
        "dataset_id": config.DATASET_ID,
        "dataset_revision": config.DATASET_REVISION,
        "counts": splits.counts(),
        "digest": splits.digest,
        "splits": splits.splits,
    }
    config.SPLITS_FILE.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info("wrote %s (%s)", config.SPLITS_FILE.name, splits.counts())


def load_splits(*, verify_digest: bool = True) -> Splits:
    """Read the committed split file.

    Args:
        verify_digest: Recompute the digest and raise if it disagrees with the
            stored one, catching a hand-edited split file.
    """
    if not config.SPLITS_FILE.exists():
        raise DatasetError(
            f"{config.SPLITS_FILE} not found. Run `python -m src.data.splits` first."
        )

    payload = json.loads(config.SPLITS_FILE.read_text(encoding="utf-8"))
    splits = Splits(
        strategy=payload["strategy"],
        seed=payload["seed"],
        splits=payload["splits"],
        digest=payload["digest"],
    )

    if verify_digest:
        recomputed = _digest(splits.splits)
        if recomputed != splits.digest:
            raise DatasetError(
                f"{config.SPLITS_FILE} digest mismatch: stored {splits.digest[:12]}, "
                f"recomputed {recomputed[:12]}. The file was modified after it was written."
            )

    return splits


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build and freeze the dataset splits.")
    parser.add_argument(
        "--strategy",
        choices=[STRATEGY_OFFICIAL, STRATEGY_RANDOM],
        default=STRATEGY_OFFICIAL,
        help="official = keep the dataset's own partition (default)",
    )
    parser.add_argument("--seed", type=int, default=config.SEED)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

    try:
        splits = build_splits(strategy=args.strategy, seed=args.seed)
    except DatasetError as error:
        logger.error("%s", error)
        return 1

    write_splits(splits)
    logger.info("digest %s", splits.digest[:16])
    return 0


if __name__ == "__main__":
    sys.exit(main())
