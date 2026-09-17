"""Resumably fetch a model's weights into the Hub cache.

The sibling of ``fetch_parquet.py`` and it exists for the same reason: on a
link that stalls mid-transfer, `huggingface_hub` restarts the file from zero,
so a 1.5 GB weight file never finishes. This drives the transfer with HTTP
range requests and resumes from wherever it got to.

Two details specific to models.

**Only one weight format is fetched.** A Hub model repo often carries the same
weights three times over -- `model.safetensors`, `pytorch_model.bin` and
`tf_model.h5` -- and downloading all of them would triple an already slow
transfer for nothing. safetensors is preferred (it loads without executing
pickled Python), with a fallback to the .bin if a repo has no safetensors.

**The files land in the Hub's snapshot layout**, so `from_pretrained` finds
them already present and performs no network IO. That is also why the config
and tokeniser files come along: without them `from_pretrained` would reach for
the network anyway and stall on the first request.

Run it with::

    python scripts/fetch_model.py --model Salesforce/blip-vqa-base
    python scripts/fetch_model.py --model openai/clip-vit-base-patch32
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.fetch_parquet import download_resumable
from src import config

logger = logging.getLogger("fetch-model")

BASE_URL = "https://huggingface.co/{repo}/resolve/{revision}/{path}"

# Weight files, most preferred first. Exactly one is downloaded.
WEIGHT_PREFERENCE = ("model.safetensors", "pytorch_model.bin")

# Everything `from_pretrained` needs besides the weights. Missing entries are
# skipped rather than failing: not every repo carries every one of these.
SUPPORT_FILES = (
    "config.json",
    "preprocessor_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.txt",
    "vocab.json",
    "merges.txt",
)

KNOWN_MODELS: dict[str, str] = {
    config.CLIP_MODEL_ID: config.CLIP_REVISION,
    config.BLIP_VQA_MODEL_ID: config.BLIP_REVISION,
}


def snapshot_path(repo_id: str, revision: str, filename: str) -> Path:
    """Where `from_pretrained` looks for this file in the Hub cache."""
    from huggingface_hub.constants import HF_HUB_CACHE

    repo_dir = Path(HF_HUB_CACHE) / f"models--{repo_id.replace('/', '--')}"
    return repo_dir / "snapshots" / revision / filename


def repo_files(repo_id: str, revision: str) -> set[str]:
    """List what the repo actually contains, so nothing absent is attempted."""
    from huggingface_hub import HfApi

    info = HfApi().model_info(repo_id, revision=revision)
    return {sibling.rfilename for sibling in info.siblings}


def fetch(repo_id: str, revision: str) -> int:
    """Download one weight file plus the support files, resumably."""
    available = repo_files(repo_id, revision)

    weights = next((name for name in WEIGHT_PREFERENCE if name in available), None)
    if weights is None:
        logger.error(
            "%s carries none of %s; nothing to download",
            repo_id,
            ", ".join(WEIGHT_PREFERENCE),
        )
        return 1

    wanted = [weights, *(name for name in SUPPORT_FILES if name in available)]
    logger.info("fetching %d files from %s @ %s", len(wanted), repo_id, revision[:12])
    logger.info("weights: %s (the other formats are skipped)", weights)

    for filename in wanted:
        url = BASE_URL.format(repo=repo_id, revision=revision, path=filename)
        destination = snapshot_path(repo_id, revision, filename)
        try:
            download_resumable(url, destination)
        except RuntimeError as error:
            logger.error("%s", error)
            return 1

    logger.info("%s is complete in the Hub cache", repo_id)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Resumably fetch model weights.")
    parser.add_argument(
        "--model",
        default=config.BLIP_VQA_MODEL_ID,
        help="repo id; defaults to the VQA model",
    )
    parser.add_argument(
        "--revision",
        default=None,
        help="commit sha; defaults to the pinned revision for a known model",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

    revision = args.revision or KNOWN_MODELS.get(args.model)
    if revision is None:
        logger.error(
            "%s is not a pinned model; pass --revision with an explicit commit sha "
            "rather than tracking a moving branch",
            args.model,
        )
        return 1

    return fetch(args.model, revision)


if __name__ == "__main__":
    sys.exit(main())
