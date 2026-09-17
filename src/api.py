"""HTTP interface over the search index and the VQA model.

Three endpoints, one for each thing the project can do:

* ``POST /search/text``  -- a sentence, ranked images back
* ``POST /search/image`` -- an uploaded photograph, similar images back
* ``POST /vqa``          -- an uploaded photograph and a question, an answer back

**Models load once, at startup.** CLIP takes about six seconds to initialise
and BLIP rather longer; loading either per request would put that on every
call and make the service unusable. FastAPI's lifespan hook runs before the
first request is accepted, so by the time anything is served the models are in
memory and stay there.

That choice has a consequence worth stating: the process holds roughly 2 GB
resident. This is a single-process service sized for one machine, not something
to run forty copies of behind a load balancer.

**Uploads are bounded.** An endpoint that accepts an image accepts whatever is
sent, so both upload endpoints reject anything over a size cap before it
reaches the decoder rather than after.

Run it with::

    uvicorn src.api:app --reload
    # then open http://127.0.0.1:8000/docs
"""

from __future__ import annotations

import logging
import tempfile
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from pydantic import BaseModel, Field

from src import config

logger = logging.getLogger(__name__)

# Large enough for any ordinary photograph, small enough that a malformed or
# hostile upload cannot exhaust memory before it is rejected.
MAX_UPLOAD_BYTES = 12 * 1024 * 1024

ALLOWED_CONTENT_TYPES = frozenset(
    {"image/jpeg", "image/jpg", "image/png", "image/webp", "image/bmp"}
)

MAX_RESULTS = 50


# ---------------------------------------------------------------------------
# Response shapes
#
# Declared as models rather than bare dicts so FastAPI can publish an accurate
# schema at /docs. A client reading that page should not have to guess what
# comes back.
# ---------------------------------------------------------------------------


class SearchHit(BaseModel):
    """One ranked image."""

    image_id: str
    score: float = Field(description="Cosine similarity, in [-1, 1]")
    rank: int
    captions: list[str]


class SearchResponse(BaseModel):
    """A ranked list, plus what produced it."""

    query: str
    count: int
    took_ms: float
    results: list[SearchHit]


class VqaResponse(BaseModel):
    """One answer to one question about one image."""

    question: str
    answer: str
    took_ms: float


class HealthResponse(BaseModel):
    """Whether the service is ready to serve, and what it is serving."""

    status: str
    corpus_size: int
    clip_model: str
    blip_model: str
    models_loaded: list[str]


# ---------------------------------------------------------------------------
# Lifespan: load once, keep
# ---------------------------------------------------------------------------

_state: dict[str, object] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the models and index before the first request is served."""
    started = time.perf_counter()
    logger.info("loading models and index")

    from src.search.image_to_image import ImageToImageSearcher
    from src.search.text_to_image import TextToImageSearcher

    # Both searchers share one encoder: the model is the expensive part and
    # there is no reason to hold two copies of it.
    text_searcher = TextToImageSearcher(exact=True)
    _ = text_searcher.encoder.dim  # forces the CLIP load
    image_searcher = ImageToImageSearcher(exact=True, encoder=text_searcher.encoder)

    _state["text_searcher"] = text_searcher
    _state["image_searcher"] = image_searcher
    _state["corpus_size"] = len(text_searcher.records)

    # BLIP is loaded lazily on the first /vqa call rather than at startup: it
    # is 1.4 GB, and a deployment that only serves search should not pay for
    # it. The first VQA request is slow; every later one is not.
    _state["vqa"] = None

    logger.info(
        "ready in %.1fs (%d images indexed)",
        time.perf_counter() - started,
        _state["corpus_size"],
    )
    yield

    _state.clear()


app = FastAPI(
    title="Multimodal image search",
    description=(
        "Bidirectional CLIP search over Flickr8k plus BLIP visual question "
        "answering. Models are loaded once at startup."
    ),
    version="0.1.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _read_upload(upload: UploadFile) -> bytes:
    """Read an uploaded image, rejecting anything oversized or non-image.

    The size check runs while reading rather than afterwards, so a very large
    upload is refused without first being held in memory in full.
    """
    if upload.content_type and upload.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=415,
            detail=f"unsupported content type {upload.content_type!r}; "
            f"expected one of {sorted(ALLOWED_CONTENT_TYPES)}",
        )

    chunks: list[bytes] = []
    total = 0
    while chunk := await upload.read(1024 * 1024):
        total += len(chunk)
        if total > MAX_UPLOAD_BYTES:
            raise HTTPException(
                status_code=413,
                detail=f"image exceeds {MAX_UPLOAD_BYTES // (1024 * 1024)} MB",
            )
        chunks.append(chunk)

    if not chunks:
        raise HTTPException(status_code=400, detail="empty upload")

    return b"".join(chunks)


def _to_hits(results) -> list[SearchHit]:
    return [
        SearchHit(
            image_id=result.image_id,
            score=round(float(result.score), 6),
            rank=result.rank,
            captions=list(result.captions),
        )
        for result in results
    ]


def _require(key: str):
    value = _state.get(key)
    if value is None:
        raise HTTPException(status_code=503, detail="service is still starting")
    return value


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/health", response_model=HealthResponse, tags=["meta"])
def health() -> HealthResponse:
    """Readiness, and which models are actually resident."""
    loaded = ["clip"] if _state.get("text_searcher") else []
    if _state.get("vqa"):
        loaded.append("blip")

    return HealthResponse(
        status="ready" if loaded else "starting",
        corpus_size=int(_state.get("corpus_size", 0)),
        clip_model=config.CLIP_MODEL_ID,
        blip_model=config.BLIP_VQA_MODEL_ID,
        models_loaded=loaded,
    )


@app.post("/search/text", response_model=SearchResponse, tags=["search"])
def search_text(
    query: Annotated[str, Form(description="What to search for")],
    k: Annotated[int, Query(ge=1, le=MAX_RESULTS)] = 10,
) -> SearchResponse:
    """Find images matching a sentence.

    The query is encoded by CLIP's text encoder and compared against the
    precomputed image vectors, so this costs one forward pass over a short
    string regardless of how many images are indexed.
    """
    from src.search.base import SearchError

    searcher = _require("text_searcher")
    started = time.perf_counter()

    try:
        results = searcher.search(query, k=k)
    except SearchError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error

    return SearchResponse(
        query=query,
        count=len(results),
        took_ms=round((time.perf_counter() - started) * 1000, 2),
        results=_to_hits(results),
    )


@app.post("/search/image", response_model=SearchResponse, tags=["search"])
async def search_image(
    image: Annotated[UploadFile, File(description="The image to find matches for")],
    k: Annotated[int, Query(ge=1, le=MAX_RESULTS)] = 10,
) -> SearchResponse:
    """Find images similar to an uploaded one.

    This is semantic similarity, not pixel similarity: two photographs of
    different dogs on different lawns match, while an image and its own
    colour-inverted copy do not.
    """
    from src.search.base import SearchError

    searcher = _require("image_searcher")
    payload = await _read_upload(image)

    started = time.perf_counter()
    # The encoder takes a path, so the upload is spooled to a temporary file
    # that is removed before the response is built.
    with tempfile.NamedTemporaryFile(suffix=".img", delete=False) as handle:
        handle.write(payload)
        temporary = Path(handle.name)

    try:
        # exclude_self is off: an uploaded image is not in the corpus, so there
        # is nothing to exclude, and a caller who uploads a corpus image
        # probably wants to see it rank first as confirmation.
        results = searcher.search(temporary, k=k, exclude_self=False)
    except SearchError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    finally:
        temporary.unlink(missing_ok=True)

    return SearchResponse(
        query=image.filename or "uploaded image",
        count=len(results),
        took_ms=round((time.perf_counter() - started) * 1000, 2),
        results=_to_hits(results),
    )


@app.post("/vqa", response_model=VqaResponse, tags=["vqa"])
async def visual_question_answering(
    image: Annotated[UploadFile, File(description="The image to ask about")],
    question: Annotated[str, Form(description="A natural-language question")],
) -> VqaResponse:
    """Answer a question about an uploaded image.

    Unlike search, nothing here can be precomputed: BLIP fuses the image and
    the question inside the model, so every call is a full forward pass of
    roughly 400 ms on CPU.
    """
    from src.vqa.blip_vqa import BlipVqa, VqaError

    if not question.strip():
        raise HTTPException(status_code=400, detail="question is empty")

    payload = await _read_upload(image)

    if _state.get("vqa") is None:
        logger.info("loading BLIP on first use")
        _state["vqa"] = BlipVqa()

    started = time.perf_counter()
    with tempfile.NamedTemporaryFile(suffix=".img", delete=False) as handle:
        handle.write(payload)
        temporary = Path(handle.name)

    try:
        answer = _state["vqa"].answer(temporary, question)
    except VqaError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    finally:
        temporary.unlink(missing_ok=True)

    return VqaResponse(
        question=question,
        answer=answer,
        took_ms=round((time.perf_counter() - started) * 1000, 2),
    )
