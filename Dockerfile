# syntax=docker/dockerfile:1

# ---------------------------------------------------------------------------
# Multi-stage build for the search + VQA API.
#
# The model weights are baked into the image, so the container runs with no
# network access at all. That is the point: a container that downloads 2 GB of
# weights on first start is a container that fails in an air-gapped
# environment, fails when the Hub is down, and starts slowly forever.
#
# The cost is size. The image is roughly 4 GB, most of it PyTorch and the two
# model checkpoints, and that is expected rather than a problem to solve.
#
# What is NOT baked in: the 8,000 raw JPEGs. The API serves ranked image ids
# and captions, not image bytes, so the corpus is not needed at runtime. The
# embedding cache and the Chroma index are, and they are 46 MB together.
#
# Build:   docker build -t multimodal-search .
# Run:     docker run -p 8000:8000 multimodal-search
# Docs:    http://localhost:8000/docs
# ---------------------------------------------------------------------------


# ===========================================================================
# Stage 1: builder -- install dependencies and fetch the weights
# ===========================================================================
FROM python:3.11-slim AS builder

# Never write .pyc files, never buffer stdout: both are noise in a container.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /build

# Build-only dependencies. They stay in this stage and never reach the runtime
# image, which is the reason for the split.
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential git \
    && rm -rf /var/lib/apt/lists/*

# A virtualenv rather than the system Python, so the whole dependency tree can
# be copied to the runtime stage as one directory.
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# torch first, from the CPU index. A plain `pip install torch` would pull the
# CUDA build -- roughly 2.5 GB of GPU libraries this image has no use for.
RUN pip install --upgrade pip \
    && pip install torch==2.14.0 torchvision==0.29.0 \
       --index-url https://download.pytorch.org/whl/cpu

# Dependencies before source, so editing a source file does not invalidate the
# layer that took ten minutes to build.
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY src/ ./src/
COPY scripts/ ./scripts/

# Fetch both checkpoints into a known location. The pinned revisions come from
# src/config.py, so the image is built against exactly the weights the
# committed metrics were produced with.
ENV HF_HOME=/opt/models
RUN python scripts/fetch_model.py --model openai/clip-vit-base-patch32 \
    && python scripts/fetch_model.py --model Salesforce/blip-vqa-base


# ===========================================================================
# Stage 2: runtime -- only what is needed to serve
# ===========================================================================
FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    # Everything the models need is already on disk; refusing network access
    # makes that guarantee explicit rather than incidental.
    HF_HOME=/opt/models \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    PATH="/opt/venv/bin:$PATH"

# libgomp is OpenMP, which PyTorch links against for CPU threading. Without it
# torch imports and then segfaults on the first tensor operation, which is a
# confusing way to find out a shared library is missing.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Run as a non-root user. Nothing here needs privileges, and a container that
# runs as root by default is a container someone will run as root in
# production.
RUN useradd --create-home --uid 1000 app

COPY --from=builder /opt/venv /opt/venv
COPY --from=builder --chown=app:app /opt/models /opt/models

WORKDIR /app

COPY --chown=app:app src/ ./src/
COPY --chown=app:app eval_sets/ ./eval_sets/

# The embedding cache and the Chroma index. These are produced by
# `make encode` on the host and are gitignored, so the build expects them to
# exist locally -- a container cannot recompute them without the raw images.
COPY --chown=app:app data/embeddings/ ./data/embeddings/
COPY --chown=app:app data/chroma/ ./data/chroma/
COPY --chown=app:app data/raw/captions.csv ./data/raw/captions.csv
COPY --chown=app:app data/splits.json ./data/splits.json

USER app

EXPOSE 8000

# Startup loads CLIP and the index, which takes several seconds; the long
# start period keeps the orchestrator from killing the container during it.
HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=3 \
    CMD python -c "import urllib.request,sys; \
        sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=8).status == 200 else 1)"

# One worker on purpose. Each worker would hold its own copy of the models --
# roughly 2 GB resident each -- so scaling this means more containers, not more
# workers inside one.
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
