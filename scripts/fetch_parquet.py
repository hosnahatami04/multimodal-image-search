"""Resumable downloader for the dataset's parquet files.

`huggingface_hub` restarts a file from zero when its connection drops. On a
link that stalls mid-transfer -- which is what this project's development
machine sees against the Hub's CDN -- that means a 138 MB file never finishes:
each attempt gets partway, stalls, and begins again.

The CDN answers range requests with `206 Partial Content`, so the fix is to
drive the transfer directly: write to a `.part` file, and on a stall reconnect
with `Range: bytes=<what we already have>-`. Progress is never lost, and a
transfer that stalls ten times still completes.

The files land in the Hub's own cache layout, so `datasets.load_dataset()`
afterwards finds them already present and does no network IO at all.

Run it with::

    python scripts/fetch_parquet.py
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config

logger = logging.getLogger("fetch")

PARQUET_FILES = [
    "data/train-00000-of-00002-2f8f6bfa852eac4b.parquet",
    "data/train-00001-of-00002-2173151d8cd6c7fb.parquet",
    "data/validation-00000-of-00001-7025a2b596f14b7b.parquet",
    "data/test-00000-of-00001-42a2661d12c73e48.parquet",
]

BASE_URL = "https://huggingface.co/datasets/{repo}/resolve/{revision}/{path}"

CHUNK_BYTES = 256 * 1024
# A read that returns nothing for this long is treated as a stall and retried
# from the current offset rather than waited out.
SOCKET_TIMEOUT_SECONDS = 30
MAX_ATTEMPTS_PER_FILE = 200


def _human(num_bytes: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if abs(num_bytes) < 1024 or unit == "GB":
            return f"{num_bytes:,.1f} {unit}"
        num_bytes /= 1024
    return f"{num_bytes:,.1f} GB"


def _remote_size(url: str) -> int | None:
    """Ask the server how large the file is, following its redirect to the CDN."""
    try:
        with urlopen(Request(url, method="HEAD"), timeout=SOCKET_TIMEOUT_SECONDS) as response:
            length = response.headers.get("Content-Length")
            return int(length) if length else None
    except (HTTPError, URLError, TimeoutError, ValueError):
        return None


def download_resumable(url: str, destination: Path) -> Path:
    """Fetch ``url`` into ``destination``, resuming across stalls and drops.

    Returns the destination path. Raises RuntimeError if the file cannot be
    completed within ``MAX_ATTEMPTS_PER_FILE`` reconnections.
    """
    if destination.exists():
        logger.info(
            "%s already complete (%s)", destination.name, _human(destination.stat().st_size)
        )
        return destination

    partial = destination.with_suffix(destination.suffix + ".part")
    destination.parent.mkdir(parents=True, exist_ok=True)

    total = _remote_size(url)
    started_at = time.time()
    stall_count = 0

    for attempt in range(1, MAX_ATTEMPTS_PER_FILE + 1):
        have = partial.stat().st_size if partial.exists() else 0

        if total is not None and have >= total:
            break

        headers = {"Range": f"bytes={have}-"} if have else {}
        bytes_this_attempt = 0

        try:
            with (
                urlopen(Request(url, headers=headers), timeout=SOCKET_TIMEOUT_SECONDS) as response,
                partial.open("ab") as handle,
            ):
                while True:
                    chunk = response.read(CHUNK_BYTES)
                    if not chunk:
                        break
                    handle.write(chunk)
                    bytes_this_attempt += len(chunk)

                    done = have + bytes_this_attempt
                    elapsed = max(time.time() - started_at, 1e-6)
                    rate = done / elapsed
                    if total:
                        remaining = (total - done) / rate if rate > 0 else 0
                        print(
                            f"\r  {destination.name[:38]:38s} "
                            f"{done / total:6.1%}  {_human(done)} / {_human(total)}  "
                            f"{_human(rate)}/s  eta {remaining / 60:4.1f}m  "
                            f"stalls {stall_count}",
                            end="",
                            flush=True,
                        )

        except (HTTPError, URLError, TimeoutError, ConnectionError, OSError) as error:
            # A stall or reset is the expected case here, not an exception worth
            # aborting on: the next iteration reconnects from the current offset.
            stall_count += 1
            logger.debug("attempt %d interrupted (%s); resuming", attempt, type(error).__name__)

        have_now = partial.stat().st_size if partial.exists() else 0
        if total is not None and have_now >= total:
            break

        if bytes_this_attempt == 0:
            stall_count += 1
            # Back off a little when nothing at all came through, so a hard
            # block does not turn into a tight reconnect loop.
            time.sleep(min(2 + stall_count * 0.5, 15))
    else:
        raise RuntimeError(
            f"{destination.name}: gave up after {MAX_ATTEMPTS_PER_FILE} attempts "
            f"({_human(partial.stat().st_size if partial.exists() else 0)} of "
            f"{_human(total or 0)})"
        )

    print()
    final_size = partial.stat().st_size
    if total is not None and final_size != total:
        raise RuntimeError(
            f"{destination.name}: expected {_human(total)}, got {_human(final_size)}"
        )

    partial.replace(destination)
    logger.info(
        "%s done: %s in %.1f min (%d stalls)",
        destination.name,
        _human(final_size),
        (time.time() - started_at) / 60,
        stall_count,
    )
    return destination


def hub_cache_path(repo_id: str, revision: str, filename: str) -> Path:
    """Where `huggingface_hub` expects this file to live in its cache.

    Writing into this layout means `load_dataset()` finds the files already
    there and performs no network IO.
    """
    from huggingface_hub.constants import HF_HUB_CACHE

    repo_dir = Path(HF_HUB_CACHE) / f"datasets--{repo_id.replace('/', '--')}"
    return repo_dir / "snapshots" / revision / filename


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Resumably fetch the Flickr8k parquet files.")
    parser.add_argument("--repo", default=config.DATASET_ID)
    parser.add_argument("--revision", default=config.DATASET_REVISION)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

    logger.info("fetching %d files from %s @ %s", len(PARQUET_FILES), args.repo, args.revision[:12])

    for path in PARQUET_FILES:
        url = BASE_URL.format(repo=args.repo, revision=args.revision, path=path)
        destination = hub_cache_path(args.repo, args.revision, path)
        try:
            download_resumable(url, destination)
        except RuntimeError as error:
            logger.error("%s", error)
            return 1

    logger.info("all parquet files present under the Hub cache")
    return 0


if __name__ == "__main__":
    sys.exit(main())
