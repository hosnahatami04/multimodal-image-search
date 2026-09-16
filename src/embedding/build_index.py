"""Encode every image once, cache the vectors, and load them into the index.

This is the long-running step of the project. On CPU, 8,000 images through
ViT-B/32 takes tens of minutes; the point of the cache is that it happens once
and every later phase reads the `.npy` file instead.

The order of operations matters. Encoding comes first and is cached
immediately, before the index is touched, so that a failure while writing to
Chroma does not cost the encode run. Rebuilding the index from a warm cache
takes seconds.

Run it with::

    python -m src.embedding.build_index            # encode if needed, then index
    python -m src.embedding.build_index --force    # re-encode from scratch
    python -m src.embedding.build_index --limit 50 # a quick end-to-end smoke run
"""

from __future__ import annotations

import argparse
import logging
import sys
import time

from src import config
from src.data.dataset import load_records
from src.embedding import cache, index
from src.embedding.clip_encoder import ClipEncoder

logger = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Encode the corpus and build the index.")
    parser.add_argument("--force", action="store_true", help="re-encode even if cached")
    parser.add_argument(
        "--limit", type=int, default=0, help="only process the first N images (for smoke runs)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=config.ENCODE_BATCH_SIZE, help="images per forward pass"
    )
    parser.add_argument(
        "--skip-index", action="store_true", help="encode and cache, but do not touch Chroma"
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

    records = load_records()
    if args.limit:
        records = records[: args.limit]
        logger.info("limited to the first %d images", len(records))

    image_ids = [record.image_id for record in records]

    # ------------------------------------------------------------------
    # Encode, or reuse the cache
    # ------------------------------------------------------------------
    entry = None
    if not args.force:
        try:
            entry = cache.load(expected_ids=image_ids)
            logger.info("reusing cached embeddings; nothing to encode")
        except cache.CacheError as error:
            logger.info("cache unusable (%s); encoding", error)

    if entry is None:
        missing = [record.image_id for record in records if not record.exists()]
        if missing:
            logger.error(
                "%d images are missing from disk (e.g. %s). Run the download step first.",
                len(missing),
                ", ".join(missing[:3]),
            )
            return 1

        encoder = ClipEncoder(batch_size=args.batch_size)
        logger.info(
            "encoding %d images at batch size %d -- expect tens of minutes on CPU",
            len(records),
            args.batch_size,
        )

        started = time.perf_counter()
        vectors = encoder.encode_images([record.path for record in records], show_progress=True)
        elapsed = time.perf_counter() - started

        logger.info(
            "encoded %d images in %.1f min (%.1f img/s)",
            len(records),
            elapsed / 60,
            len(records) / elapsed,
        )

        cache.save(vectors, image_ids)
        entry = cache.load(expected_ids=image_ids)

    logger.info("embeddings: %d x %d", entry.vectors.shape[0], entry.dim)

    # ------------------------------------------------------------------
    # Index
    # ------------------------------------------------------------------
    if args.skip_index:
        logger.info("--skip-index given; done")
        return 0

    started = time.perf_counter()
    total = index.build(records, entry.vectors, reset=True)
    logger.info("indexed %d vectors in %.1fs", total, time.perf_counter() - started)

    if total != len(records):
        logger.error("index holds %d vectors for %d records", total, len(records))
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
