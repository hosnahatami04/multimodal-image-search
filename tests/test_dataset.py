"""Tests for the Phase 1 data pipeline.

These cover the three properties the plan calls out -- five captions per image,
a deterministic split, and a clear error for a missing image file -- plus the
verification logic, which is the piece that decides whether a half-downloaded
dataset is allowed through.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from src import config
from src.data import download, splits
from src.data.dataset import DatasetError, ImageRecord, load_records, summarise

# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


def test_loader_returns_five_captions_per_image(tiny_dataset):
    records = load_records()

    assert len(records) == tiny_dataset["num_images"]
    for record in records:
        assert len(record.captions) == 5
        assert all(caption.strip() for caption in record.captions)


def test_loader_filters_by_split(tiny_dataset):
    for split_name, expected in tiny_dataset["split_sizes"].items():
        records = load_records(split=split_name)
        assert len(records) == expected
        assert {record.split for record in records} == {split_name}


def test_loader_rejects_unknown_split(tiny_dataset):
    with pytest.raises(DatasetError, match="no records found"):
        load_records(split="nonexistent")


def test_record_is_immutable(tiny_dataset):
    record = load_records()[0]
    with pytest.raises(AttributeError):
        record.image_id = "changed"


def test_record_rejects_wrong_caption_count():
    with pytest.raises(DatasetError, match="expected 5 captions"):
        ImageRecord(image_id="x", path=Path("x.jpg"), captions=("only", "three", "here"))


def test_record_rejects_empty_caption():
    with pytest.raises(DatasetError, match="empty caption"):
        ImageRecord(image_id="x", path=Path("x.jpg"), captions=("a", "b", "  ", "d", "e"))


# ---------------------------------------------------------------------------
# Missing files
# ---------------------------------------------------------------------------


def test_missing_image_raises_clear_error(tiny_dataset):
    victim = next(iter(Path(tiny_dataset["images_dir"]).glob("*.jpg")))
    victim.unlink()

    records = load_records()
    broken = next(record for record in records if record.image_id == victim.stem)

    assert not broken.exists()
    with pytest.raises(DatasetError, match="image file not found"):
        broken.require_file()


def test_require_files_reports_how_many_are_missing(tiny_dataset):
    for victim in list(Path(tiny_dataset["images_dir"]).glob("*.jpg"))[:2]:
        victim.unlink()

    with pytest.raises(DatasetError, match=r"2 of .* images are missing"):
        load_records(require_files=True)


def test_loader_without_captions_file_is_explicit(tiny_dataset):
    Path(tiny_dataset["captions_file"]).unlink()

    with pytest.raises(DatasetError, match=re.escape("Run `python -m src.data.download`")):
        load_records()


# ---------------------------------------------------------------------------
# Verification -- a partial dataset must fail loudly
# ---------------------------------------------------------------------------


def test_verify_accepts_a_complete_dataset(tiny_dataset):
    report = download.verify(write_manifest=False)

    assert report.num_images == tiny_dataset["num_images"]
    assert report.num_caption_rows == tiny_dataset["num_images"]
    assert report.split_counts == tiny_dataset["split_sizes"]
    assert len(report.manifest_digest) == 64


def test_verify_rejects_missing_images(tiny_dataset):
    next(iter(Path(tiny_dataset["images_dir"]).glob("*.jpg"))).unlink()

    with pytest.raises(download.DatasetIncompleteError, match="no image file"):
        download.verify(write_manifest=False)


def test_verify_rejects_a_truncated_captions_file(tiny_dataset):
    path = Path(tiny_dataset["captions_file"])
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    path.write_text("".join(lines[:-2]), encoding="utf-8")

    with pytest.raises(download.DatasetIncompleteError, match="caption rows"):
        download.verify(write_manifest=False)


def test_verify_writes_a_manifest(tiny_dataset):
    download.verify(write_manifest=True)

    manifest = json.loads(config.MANIFEST_FILE.read_text(encoding="utf-8"))
    assert manifest["num_images"] == tiny_dataset["num_images"]
    assert len(manifest["files"]) == tiny_dataset["num_images"]
    assert manifest["dataset_revision"] == config.DATASET_REVISION


def test_manifest_digest_changes_when_an_image_changes(tiny_dataset):
    before = download.build_manifest()["digest"]

    victim = next(iter(Path(tiny_dataset["images_dir"]).glob("*.jpg")))
    victim.write_bytes(victim.read_bytes() + b"\x00" * 128)

    assert download.build_manifest()["digest"] != before


# ---------------------------------------------------------------------------
# Splits -- the same inputs must always produce the same partition
# ---------------------------------------------------------------------------


def test_official_split_matches_the_dataset(tiny_dataset):
    built = splits.build_splits(strategy=splits.STRATEGY_OFFICIAL)

    assert built.counts() == tiny_dataset["split_sizes"]
    assert built.strategy == splits.STRATEGY_OFFICIAL


def test_split_is_deterministic_across_runs(tiny_dataset):
    first = splits.build_splits(strategy=splits.STRATEGY_RANDOM, seed=config.SEED)
    second = splits.build_splits(strategy=splits.STRATEGY_RANDOM, seed=config.SEED)

    assert first.splits == second.splits
    assert first.digest == second.digest


def test_different_seeds_give_different_partitions(tiny_dataset):
    first = splits.build_splits(strategy=splits.STRATEGY_RANDOM, seed=1)
    second = splits.build_splits(strategy=splits.STRATEGY_RANDOM, seed=2)

    assert first.digest != second.digest


def test_splits_are_disjoint_and_complete(tiny_dataset):
    built = splits.build_splits(strategy=splits.STRATEGY_RANDOM)

    all_ids = [image_id for ids in built.splits.values() for image_id in ids]
    assert len(all_ids) == len(set(all_ids)) == tiny_dataset["num_images"]


def test_splits_round_trip_through_disk(tiny_dataset):
    built = splits.build_splits()
    splits.write_splits(built)
    loaded = splits.load_splits()

    assert loaded.splits == built.splits
    assert loaded.digest == built.digest


def test_tampered_split_file_is_detected(tiny_dataset):
    splits.write_splits(splits.build_splits())

    payload = json.loads(config.SPLITS_FILE.read_text(encoding="utf-8"))
    payload["splits"]["train"].pop()
    config.SPLITS_FILE.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(DatasetError, match="digest mismatch"):
        splits.load_splits()


def test_unknown_strategy_is_rejected(tiny_dataset):
    with pytest.raises(DatasetError, match="unknown strategy"):
        splits.build_splits(strategy="alphabetical")


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------


def test_summarise_counts_captions(tiny_dataset):
    stats = summarise(load_records())

    assert stats["num_images"] == tiny_dataset["num_images"]
    assert stats["num_captions"] == tiny_dataset["num_images"] * 5
    assert stats["caption_words_min"] >= 1
    assert stats["vocabulary_size"] > 0


def test_summarise_rejects_empty_input():
    with pytest.raises(DatasetError, match="empty record set"):
        summarise([])
