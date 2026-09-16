"""Fetch Flickr8k, materialise it on disk, and verify it is complete.

The governing rule of this module is that a partial download must fail loudly.
A silently truncated dataset does not raise anything on its own -- it just
quietly shifts every metric computed downstream, and nothing in the rest of the
pipeline can detect it. So every assumption about the data is checked here, and
a violated assumption raises.

What this produces under ``data/raw``::

    images/<image_id>.jpg   one file per image
    captions.csv            image_id, split, caption_0 .. caption_4
    manifest.json           per-file sizes and hashes, plus one digest

Run it with::

    python -m src.data.download
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

from src import config

logger = logging.getLogger(__name__)

# Number of bytes of each JPEG that feed the per-file hash. Hashing whole files
# for 8,000 images is needless IO: the first 64 KiB plus the exact byte length
# already identifies a file for integrity purposes, and a truncated or swapped
# file will differ in one or the other.
HASH_PREFIX_BYTES = 64 * 1024


class DatasetIncompleteError(RuntimeError):
    """Raised when the materialised dataset does not match expectations.

    A distinct exception type so callers and tests can assert on the failure
    mode rather than matching on message text.
    """


@dataclass(frozen=True)
class VerificationReport:
    """Outcome of checking what is on disk against what was expected."""

    num_images: int
    num_caption_rows: int
    split_counts: dict[str, int]
    manifest_digest: str

    def describe(self) -> str:
        splits = ", ".join(f"{name}={count}" for name, count in sorted(self.split_counts.items()))
        return (
            f"{self.num_images} images, {self.num_caption_rows} caption rows "
            f"({splits}), digest {self.manifest_digest[:12]}"
        )


def _hash_file(path: Path) -> str:
    """Return a short content hash for one file.

    Mixes the file's byte length into the digest so that a truncation beyond
    ``HASH_PREFIX_BYTES`` still changes the hash.
    """
    digest = hashlib.sha256()
    size = path.stat().st_size
    digest.update(str(size).encode("utf-8"))
    with path.open("rb") as handle:
        digest.update(handle.read(HASH_PREFIX_BYTES))
    return digest.hexdigest()


def _materialise(force: bool = False) -> None:
    """Download the dataset and write images plus captions to ``data/raw``.

    Importing ``datasets`` lazily keeps this module importable -- and its tests
    collectable -- on a machine where the heavy dependencies are not installed.
    """
    from datasets import load_dataset  # noqa: PLC0415 -- intentionally lazy
    from tqdm import tqdm  # noqa: PLC0415

    config.ensure_dirs()

    if config.CAPTIONS_FILE.exists() and not force:
        logger.info("captions.csv already present; skipping extraction (use --force to redo)")
        return

    logger.info(
        "loading %s at revision %s (first run downloads ~275 MB; "
        "on a slow link this is the long part)",
        config.DATASET_ID,
        config.DATASET_REVISION[:12],
    )
    dataset = load_dataset(config.DATASET_ID, revision=config.DATASET_REVISION)

    caption_columns = [f"caption_{i}" for i in range(config.EXPECTED_CAPTIONS_PER_IMAGE)]
    rows: list[dict[str, str]] = []

    for split_name in sorted(dataset.keys()):
        split = dataset[split_name]
        logger.info("extracting split %r (%d rows)", split_name, len(split))

        for index, record in enumerate(
            tqdm(split, desc=f"  {split_name}", unit="img", leave=False)
        ):
            # Stable, split-scoped id. The upstream parquet carries no filename,
            # so we mint ids ourselves -- deterministic because the row order in
            # a pinned revision is fixed.
            image_id = f"{split_name}_{index:05d}"
            image_path = config.IMAGES_DIR / f"{image_id}.jpg"

            if not image_path.exists() or force:
                image = record["image"]
                # Flickr8k is RGB, but a stray palette or grayscale image would
                # otherwise fail on JPEG save.
                if image.mode != "RGB":
                    image = image.convert("RGB")
                image.save(image_path, format="JPEG", quality=95)

            row = {"image_id": image_id, "split": split_name}
            for column in caption_columns:
                # Captions occasionally carry stray whitespace and newlines.
                row[column] = " ".join(str(record[column]).split())
            rows.append(row)

    with config.CAPTIONS_FILE.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image_id", "split", *caption_columns])
        writer.writeheader()
        writer.writerows(rows)

    logger.info("wrote %s (%d rows)", config.CAPTIONS_FILE.name, len(rows))


def build_manifest() -> dict[str, object]:
    """Hash every extracted image and fold the results into one digest.

    The per-file entries make it possible to point at *which* file changed; the
    combined digest makes it cheap to answer "is this the same dataset as the
    run that produced the committed metrics?".
    """
    paths = sorted(config.IMAGES_DIR.glob("*.jpg"))

    # Only worth a progress bar at dataset scale; the test fixtures have nine
    # images and should not print anything.
    if len(paths) > 100:
        from tqdm import tqdm  # noqa: PLC0415

        paths = tqdm(paths, desc="  hashing", unit="img", leave=False)

    entries: dict[str, dict[str, object]] = {}
    for image_path in paths:
        entries[image_path.name] = {
            "bytes": image_path.stat().st_size,
            "sha256_prefix": _hash_file(image_path),
        }

    combined = hashlib.sha256()
    for name, entry in entries.items():
        combined.update(name.encode("utf-8"))
        combined.update(str(entry["sha256_prefix"]).encode("utf-8"))

    return {
        "dataset_id": config.DATASET_ID,
        "dataset_revision": config.DATASET_REVISION,
        "num_images": len(entries),
        "digest": combined.hexdigest(),
        "files": entries,
    }


def verify(write_manifest: bool = True) -> VerificationReport:
    """Check the extracted dataset against every documented expectation.

    Raises:
        DatasetIncompleteError: on any mismatch -- a missing captions file, a
            wrong image count, a split of the wrong size, a row without exactly
            five non-empty captions, or a caption row with no image on disk.
    """
    if not config.CAPTIONS_FILE.exists():
        raise DatasetIncompleteError(
            f"{config.CAPTIONS_FILE} is missing. Run `python -m src.data.download` first."
        )

    caption_columns = [f"caption_{i}" for i in range(config.EXPECTED_CAPTIONS_PER_IMAGE)]
    split_counts: dict[str, int] = {}
    num_rows = 0
    missing_images: list[str] = []
    bad_caption_rows: list[str] = []

    with config.CAPTIONS_FILE.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        missing_columns = set(caption_columns) - set(reader.fieldnames or [])
        if missing_columns:
            raise DatasetIncompleteError(
                f"captions.csv is missing columns: {sorted(missing_columns)}"
            )

        for row in reader:
            num_rows += 1
            split_counts[row["split"]] = split_counts.get(row["split"], 0) + 1

            captions = [row[column].strip() for column in caption_columns]
            if any(not caption for caption in captions):
                bad_caption_rows.append(row["image_id"])

            if not (config.IMAGES_DIR / f"{row['image_id']}.jpg").exists():
                missing_images.append(row["image_id"])

    problems: list[str] = []

    if num_rows != config.EXPECTED_NUM_IMAGES:
        problems.append(f"expected {config.EXPECTED_NUM_IMAGES} caption rows, found {num_rows}")

    for split_name, expected in config.EXPECTED_SPLIT_SIZES.items():
        actual = split_counts.get(split_name, 0)
        if actual != expected:
            problems.append(f"split {split_name!r}: expected {expected} rows, found {actual}")

    if missing_images:
        preview = ", ".join(missing_images[:5])
        problems.append(f"{len(missing_images)} caption rows have no image file (e.g. {preview})")

    if bad_caption_rows:
        preview = ", ".join(bad_caption_rows[:5])
        problems.append(
            f"{len(bad_caption_rows)} rows do not have "
            f"{config.EXPECTED_CAPTIONS_PER_IMAGE} non-empty captions (e.g. {preview})"
        )

    num_files = sum(1 for _ in config.IMAGES_DIR.glob("*.jpg"))
    if num_files != config.EXPECTED_NUM_IMAGES:
        problems.append(f"expected {config.EXPECTED_NUM_IMAGES} image files, found {num_files}")

    if problems:
        raise DatasetIncompleteError(
            "Dataset verification failed:\n  - " + "\n  - ".join(problems)
        )

    manifest = build_manifest()
    if write_manifest:
        config.MANIFEST_FILE.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        logger.info("wrote %s", config.MANIFEST_FILE.name)

    return VerificationReport(
        num_images=num_files,
        num_caption_rows=num_rows,
        split_counts=split_counts,
        manifest_digest=str(manifest["digest"]),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Download and verify Flickr8k.")
    parser.add_argument(
        "--force", action="store_true", help="re-extract even if captions.csv already exists"
    )
    parser.add_argument(
        "--verify-only", action="store_true", help="skip downloading, only check what is on disk"
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

    if not args.verify_only:
        _materialise(force=args.force)

    try:
        report = verify()
    except DatasetIncompleteError as error:
        logger.error("%s", error)
        return 1

    logger.info("dataset OK: %s", report.describe())
    return 0


if __name__ == "__main__":
    sys.exit(main())
