"""Pair each image with its five captions and expose them as clean records.

The shape the rest of the pipeline consumes is ``ImageRecord``: an id, a path
to a JPEG on disk, and exactly five caption strings. Everything that knows
about CSV columns, splits, and file layout stops here.

``ImageRecord`` is frozen and validates itself on construction. Both choices
are deliberate. An accidental mutation of a record's captions somewhere in a
later phase would corrupt the data underneath every metric, and the corruption
would surface far from its cause; refusing the mutation outright is cheaper
than debugging it. Validating at construction means "exactly five captions per
image" is enforced once, at the boundary, rather than assumed in six places.
"""

from __future__ import annotations

import argparse
import csv
import logging
import statistics
import sys
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from src import config

logger = logging.getLogger(__name__)


class DatasetError(RuntimeError):
    """Raised when the on-disk dataset cannot be turned into valid records."""


@dataclass(frozen=True)
class ImageRecord:
    """One Flickr8k image together with its five independent human captions.

    Attributes:
        image_id: Stable identifier, also the JPEG's stem on disk.
        path: Absolute path to the image file.
        captions: Exactly five captions, written by five different annotators.
        split: Which official split this image belongs to.
    """

    image_id: str
    path: Path
    captions: tuple[str, ...]
    split: str = field(default="", compare=False)

    def __post_init__(self) -> None:
        if len(self.captions) != config.EXPECTED_CAPTIONS_PER_IMAGE:
            raise DatasetError(
                f"{self.image_id}: expected {config.EXPECTED_CAPTIONS_PER_IMAGE} captions, "
                f"got {len(self.captions)}"
            )
        if any(not caption.strip() for caption in self.captions):
            raise DatasetError(f"{self.image_id}: contains an empty caption")

    def exists(self) -> bool:
        """Whether the image file is actually present on disk."""
        return self.path.is_file()

    def require_file(self) -> Path:
        """Return the image path, raising a clear error if the file is missing.

        Callers that are about to open the image use this instead of ``path``
        so that a missing file produces one actionable message rather than a
        bare ``FileNotFoundError`` from deep inside an image library.
        """
        if not self.exists():
            raise DatasetError(
                f"{self.image_id}: image file not found at {self.path}. "
                "The dataset may be incomplete -- run `python -m src.data.download --verify-only`."
            )
        return self.path


def load_records(split: str | None = None, *, require_files: bool = False) -> list[ImageRecord]:
    """Read ``captions.csv`` and return the records it describes.

    Args:
        split: Restrict to one official split (``train``/``validation``/``test``).
            ``None`` returns every record.
        require_files: When true, raise if any record's image is missing from
            disk. Off by default so that metadata-only work (counting captions,
            measuring agreement) does not require the images to be present.

    Raises:
        DatasetError: if the captions file is absent, malformed, or -- with
            ``require_files`` -- refers to images that are not on disk.
    """
    if not config.CAPTIONS_FILE.exists():
        raise DatasetError(
            f"{config.CAPTIONS_FILE} not found. Run `python -m src.data.download` first."
        )

    caption_columns = [f"caption_{i}" for i in range(config.EXPECTED_CAPTIONS_PER_IMAGE)]
    records: list[ImageRecord] = []

    with config.CAPTIONS_FILE.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        missing_columns = set(["image_id", "split", *caption_columns]) - set(
            reader.fieldnames or []
        )
        if missing_columns:
            raise DatasetError(f"captions.csv is missing columns: {sorted(missing_columns)}")

        for row in reader:
            if split is not None and row["split"] != split:
                continue
            records.append(
                ImageRecord(
                    image_id=row["image_id"],
                    path=config.IMAGES_DIR / f"{row['image_id']}.jpg",
                    captions=tuple(row[column] for column in caption_columns),
                    split=row["split"],
                )
            )

    if split is not None and not records:
        raise DatasetError(f"no records found for split {split!r}")

    if require_files:
        missing = [record.image_id for record in records if not record.exists()]
        if missing:
            preview = ", ".join(missing[:5])
            raise DatasetError(
                f"{len(missing)} of {len(records)} images are missing from disk (e.g. {preview})"
            )

    return records


def iter_captions(records: Sequence[ImageRecord]) -> Iterator[tuple[str, str]]:
    """Yield ``(image_id, caption)`` for every caption across every record."""
    for record in records:
        for caption in record.captions:
            yield record.image_id, caption


def summarise(records: Sequence[ImageRecord]) -> dict[str, object]:
    """Compute descriptive statistics over a set of records.

    Caption length is reported in words rather than characters because that is
    the unit CLIP's text encoder ultimately works in, and because its context
    window is the thing worth knowing about: captions long enough to be
    truncated lose information before the model ever sees them.
    """
    if not records:
        raise DatasetError("cannot summarise an empty record set")

    word_counts = [len(caption.split()) for _, caption in iter_captions(records)]
    split_counts: dict[str, int] = {}
    for record in records:
        split_counts[record.split] = split_counts.get(record.split, 0) + 1

    vocabulary = {
        word.lower().strip(".,!?;:\"'")
        for _, caption in iter_captions(records)
        for word in caption.split()
    }
    vocabulary.discard("")

    return {
        "num_images": len(records),
        "num_captions": len(word_counts),
        "splits": split_counts,
        "caption_words_mean": round(statistics.mean(word_counts), 2),
        "caption_words_median": statistics.median(word_counts),
        "caption_words_min": min(word_counts),
        "caption_words_max": max(word_counts),
        "vocabulary_size": len(vocabulary),
        "images_present_on_disk": sum(1 for record in records if record.exists()),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Inspect the Flickr8k records.")
    parser.add_argument("--stats", action="store_true", help="print dataset statistics")
    parser.add_argument("--split", default=None, help="restrict to one split")
    parser.add_argument("--sample", type=int, default=0, help="print N example records")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

    try:
        records = load_records(split=args.split)
    except DatasetError as error:
        logger.error("%s", error)
        return 1

    if args.stats or not args.sample:
        stats = summarise(records)
        width = max(len(key) for key in stats)
        for key, value in stats.items():
            print(f"  {key:<{width}}  {value}")

    for record in records[: args.sample]:
        print(f"\n{record.image_id}  [{record.split}]  {record.path.name}")
        for index, caption in enumerate(record.captions):
            print(f"  {index}. {caption}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
