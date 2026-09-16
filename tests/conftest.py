"""Shared pytest fixtures.

The fixtures here build a miniature dataset with the same *shape* as Flickr8k
-- a captions CSV, one JPEG per row, five captions each -- inside a temporary
directory, and redirect ``src.config`` at it.

The point is that the Phase 1 logic can be tested without the real 1 GB
download. A test suite that only runs once the dataset is on disk is a test
suite that never runs in CI, which is where it matters most.
"""

from __future__ import annotations

import csv
from collections.abc import Iterator
from pathlib import Path

import pytest

from src import config

# A handful of caption sets with deliberately different agreement levels, so
# the agreement tests have something with real variation to measure.
SAMPLE_CAPTIONS: dict[str, tuple[str, ...]] = {
    "high": (
        "A brown dog runs across the green grass",
        "A brown dog running on grass",
        "A dog runs through a grass field",
        "The brown dog is running on the grass",
        "A dog running across green grass",
    ),
    "low": (
        "A man in a red jacket stands near a building",
        "Someone waits outside on a cold day",
        "A person by a brick wall",
        "A tourist looks at a map in the city",
        "An individual wearing winter clothing outdoors",
    ),
    "medium": (
        "Two children play with a ball in the park",
        "Kids playing ball outside",
        "Two young boys kick a ball on the lawn",
        "Children are playing in a park",
        "Two kids with a soccer ball",
    ),
}


def _write_jpeg(path: Path, colour: tuple[int, int, int]) -> None:
    """Write a tiny valid JPEG, or a plausible stand-in if Pillow is absent."""
    try:
        from PIL import Image
    except ImportError:  # pragma: no cover - Pillow is a hard dependency
        path.write_bytes(b"\xff\xd8\xff\xe0" + bytes(colour) * 64 + b"\xff\xd9")
        return

    Image.new("RGB", (32, 32), colour).save(path, format="JPEG")


@pytest.fixture
def tiny_dataset(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[dict[str, object]]:
    """Build a 9-image dataset on disk and point ``src.config`` at it.

    Yields a dict describing what was written, so tests can assert against the
    known-good expectations rather than re-deriving them.
    """
    raw_dir = tmp_path / "raw"
    images_dir = raw_dir / "images"
    images_dir.mkdir(parents=True)

    split_sizes = {"train": 5, "validation": 2, "test": 2}
    caption_columns = [f"caption_{i}" for i in range(5)]

    rows: list[dict[str, str]] = []
    styles = list(SAMPLE_CAPTIONS)

    for split_name, size in split_sizes.items():
        for index in range(size):
            image_id = f"{split_name}_{index:05d}"
            style = styles[(index + len(rows)) % len(styles)]
            _write_jpeg(images_dir / f"{image_id}.jpg", (index * 25 % 256, 120, 200))

            row = {"image_id": image_id, "split": split_name}
            for column, caption in zip(caption_columns, SAMPLE_CAPTIONS[style], strict=True):
                row[column] = caption
            rows.append(row)

    captions_file = raw_dir / "captions.csv"
    with captions_file.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image_id", "split", *caption_columns])
        writer.writeheader()
        writer.writerows(rows)

    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    monkeypatch.setattr(config, "RAW_DIR", raw_dir)
    monkeypatch.setattr(config, "IMAGES_DIR", images_dir)
    monkeypatch.setattr(config, "CAPTIONS_FILE", captions_file)
    monkeypatch.setattr(config, "MANIFEST_FILE", raw_dir / "manifest.json")
    monkeypatch.setattr(config, "SPLITS_FILE", tmp_path / "splits.json")
    monkeypatch.setattr(config, "EXPECTED_SPLIT_SIZES", split_sizes)
    monkeypatch.setattr(config, "EXPECTED_NUM_IMAGES", sum(split_sizes.values()))

    yield {
        "root": tmp_path,
        "images_dir": images_dir,
        "captions_file": captions_file,
        "split_sizes": split_sizes,
        "num_images": sum(split_sizes.values()),
        "rows": rows,
    }
