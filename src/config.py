"""Central configuration: paths, seeds, and pinned model revisions.

Every module reads its paths and constants from here rather than hardcoding
them. Two reasons:

1. A path that appears in ten files is a path that will be changed in nine.
2. The random seed has to be one value shared by every phase. A seed defined
   per-module is not a seed, it is a decoration.

Model revisions are pinned to specific commit SHAs, not to `main`. A model
repository on the Hugging Face Hub is mutable: the weights behind
`openai/clip-vit-base-patch32` can be updated, and if that happens your metrics
change without a single line of your code changing. Pinning makes the run
reproducible. This is what the plan asks for in Phase 7.
"""

from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Hub transport
#
# `hf_xet` is Hugging Face's newer chunked transfer backend. It is faster where
# it works, but it negotiates over endpoints that some networks do not route,
# and when that happens a download does not error -- it sits at zero bytes
# indefinitely, which looks exactly like a slow connection.
#
# Falling back to plain HTTPS costs a little throughput and makes the download
# actually finish. Set MMS_USE_XET=1 to opt back in.
#
# This has to run before `huggingface_hub` is imported anywhere, which is why
# it lives at the top of the module every other module imports first.
# ---------------------------------------------------------------------------
if os.environ.get("MMS_USE_XET", "0") != "1":
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

# Give a slow link room to establish a connection before giving up.
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "60")

# ---------------------------------------------------------------------------
# Paths
#
# PROJECT_ROOT resolves from this file's location, so every path works no
# matter which directory the process was launched from.
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
IMAGES_DIR = RAW_DIR / "images"
EMBEDDINGS_DIR = DATA_DIR / "embeddings"
CHROMA_DIR = DATA_DIR / "chroma"

EVAL_SETS_DIR = PROJECT_ROOT / "eval_sets"
RESULTS_DIR = PROJECT_ROOT / "results"

# Written by the download step, read by the dataset loader.
CAPTIONS_FILE = RAW_DIR / "captions.csv"
MANIFEST_FILE = RAW_DIR / "manifest.json"

# Written by the split step, read by every later phase.
SPLITS_FILE = DATA_DIR / "splits.json"

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
SEED = 42

# The `jxie/flickr8k` build of Flickr8k carries the standard Karpathy split:
# 6,000 train / 1,000 validation / 1,000 test, each row holding one image and
# exactly 5 captions. These constants are assertions about the dataset, used to
# fail loudly on a partial download rather than silently computing metrics over
# whatever subset happened to arrive.
EXPECTED_SPLIT_SIZES = {"train": 6000, "validation": 1000, "test": 1000}
EXPECTED_NUM_IMAGES = sum(EXPECTED_SPLIT_SIZES.values())  # 8000
EXPECTED_CAPTIONS_PER_IMAGE = 5

# Which of the dataset's own splits the evaluation phases read from.
# We keep the official split rather than reshuffling: it is the split the
# published literature reports against, so our numbers stay comparable.
EVAL_SPLIT = "test"

# ---------------------------------------------------------------------------
# Models -- pinned to exact revisions
# ---------------------------------------------------------------------------
# Dataset, pinned to the exact commit we verified.
DATASET_ID = "jxie/flickr8k"
DATASET_REVISION = "56f58c967835f7c508d684f36bd7897cca9d7634"

CLIP_MODEL_ID = "openai/clip-vit-base-patch32"
CLIP_REVISION = "3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268"

BLIP_VQA_MODEL_ID = "Salesforce/blip-vqa-base"
BLIP_REVISION = "787b3d35d57e49572baabd22884b3d5a05acf072"

# CLIP ViT-B/32 produces 512-dimensional embeddings for both modalities.
EMBEDDING_DIM = 512

# Batch size for the initial encode run. Larger is faster up to the point
# where the batch stops fitting in RAM; 32 is comfortable on 8 GB.
ENCODE_BATCH_SIZE = 32


def ensure_dirs() -> None:
    """Create every directory the pipeline writes into.

    Safe to call repeatedly; `exist_ok=True` makes it idempotent.
    """
    for directory in (
        RAW_DIR,
        IMAGES_DIR,
        EMBEDDINGS_DIR,
        CHROMA_DIR,
        EVAL_SETS_DIR,
        RESULTS_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)
