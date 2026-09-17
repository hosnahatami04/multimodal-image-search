"""Tests for the HTTP layer.

Split by what they need. The request-handling tests -- validation, size limits,
content types, response shapes -- run everywhere, against a searcher stubbed
out at the module's state dict. They are the ones worth having on every commit:
a size cap that silently stopped working would not show up in any model test.

A second group drives the real models end to end and carries the usual markers.
"""

from __future__ import annotations

from contextlib import asynccontextmanager

import pytest
from fastapi.testclient import TestClient

from src import api
from src.api import MAX_UPLOAD_BYTES, app
from src.search.base import SearchResult

# ---------------------------------------------------------------------------
# A searcher that returns known results without loading anything
# ---------------------------------------------------------------------------


class StubSearcher:
    """Returns k synthetic hits, and records what it was asked."""

    def __init__(self, corpus_size: int = 8000) -> None:
        self.records = dict.fromkeys(range(corpus_size))
        self.calls: list[dict] = []

    def search(self, query, k: int = 10, **kwargs):
        self.calls.append({"query": str(query), "k": k, **kwargs})
        return [
            SearchResult(
                image_id=f"test_{position:05d}",
                score=0.9 - position * 0.01,
                path=__import__("pathlib").Path(f"/fake/test_{position:05d}.jpg"),
                captions=(f"caption for {position}",) * 5,
                rank=position + 1,
            )
            for position in range(k)
        ]


class StubVqa:
    def __init__(self, answer: str = "two") -> None:
        self.answer_text = answer
        self.calls: list[str] = []

    def answer(self, path, question: str) -> str:
        self.calls.append(question)
        return self.answer_text


def _png_bytes(size: int = 2048) -> bytes:
    """A payload with a PNG-ish header, padded to `size`."""
    header = b"\x89PNG\r\n\x1a\n"
    return header + b"\x00" * max(0, size - len(header))


@pytest.fixture
def client(monkeypatch):
    """A client with stubbed state and the real lifespan replaced.

    `TestClient` runs the app's lifespan on entry, which would load CLIP and
    overwrite whatever was put in `_state`. The stub lifespan leaves the state
    exactly as the fixture set it, so these tests exercise request handling
    without touching a model.
    """
    text = StubSearcher()
    image = StubSearcher()
    vqa = StubVqa()

    api._state.clear()
    api._state.update(
        {
            "text_searcher": text,
            "image_searcher": image,
            "vqa": vqa,
            "corpus_size": 8000,
        }
    )

    @asynccontextmanager
    async def stub_lifespan(_app):
        yield

    monkeypatch.setattr(app.router, "lifespan_context", stub_lifespan)

    with TestClient(app) as test_client:
        test_client.stub_text = text
        test_client.stub_image = image
        test_client.stub_vqa = vqa
        yield test_client

    api._state.clear()


@pytest.fixture
def bare_client(monkeypatch):
    """A client with no models loaded, to test the not-ready path."""
    api._state.clear()

    @asynccontextmanager
    async def stub_lifespan(_app):
        yield

    monkeypatch.setattr(app.router, "lifespan_context", stub_lifespan)

    with TestClient(app) as test_client:
        yield test_client


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------


def test_health_reports_ready_when_models_are_loaded(client):
    body = client.get("/health").json()

    assert body["status"] == "ready"
    assert body["corpus_size"] == 8000
    assert "clip" in body["models_loaded"]


def test_health_names_the_pinned_models(client):
    from src import config

    body = client.get("/health").json()

    assert body["clip_model"] == config.CLIP_MODEL_ID
    assert body["blip_model"] == config.BLIP_VQA_MODEL_ID


# ---------------------------------------------------------------------------
# Text search
# ---------------------------------------------------------------------------


def test_text_search_returns_ranked_results(client):
    response = client.post("/search/text", data={"query": "a dog on grass"}, params={"k": 5})

    assert response.status_code == 200
    body = response.json()
    assert body["count"] == 5
    assert len(body["results"]) == 5
    assert [hit["rank"] for hit in body["results"]] == [1, 2, 3, 4, 5]


def test_text_search_scores_descend(client):
    body = client.post("/search/text", data={"query": "anything"}, params={"k": 8}).json()
    scores = [hit["score"] for hit in body["results"]]

    assert scores == sorted(scores, reverse=True)


def test_text_search_reports_its_own_latency(client):
    body = client.post("/search/text", data={"query": "x"}).json()
    assert body["took_ms"] >= 0


def test_text_search_defaults_to_ten_results(client):
    assert client.post("/search/text", data={"query": "x"}).json()["count"] == 10


def test_text_search_rejects_a_missing_query(client):
    assert client.post("/search/text", data={}).status_code == 422


@pytest.mark.parametrize("k", [0, -1, 51, 1000])
def test_text_search_rejects_an_out_of_range_k(client, k):
    response = client.post("/search/text", data={"query": "x"}, params={"k": k})
    assert response.status_code == 422


def test_text_search_passes_k_through(client):
    client.post("/search/text", data={"query": "x"}, params={"k": 3})
    assert client.stub_text.calls[-1]["k"] == 3


def test_text_search_before_startup_is_unavailable(bare_client):
    response = bare_client.post("/search/text", data={"query": "x"})
    assert response.status_code == 503


# ---------------------------------------------------------------------------
# Image search
# ---------------------------------------------------------------------------


def test_image_search_accepts_an_upload(client):
    response = client.post(
        "/search/image",
        files={"image": ("photo.png", _png_bytes(), "image/png")},
        params={"k": 4},
    )

    assert response.status_code == 200
    assert response.json()["count"] == 4


def test_image_search_does_not_exclude_the_query(client):
    # An uploaded image is not in the corpus, so there is nothing to exclude --
    # and a caller uploading a corpus image wants to see it rank first.
    client.post("/search/image", files={"image": ("p.png", _png_bytes(), "image/png")})
    assert client.stub_image.calls[-1]["exclude_self"] is False


def test_image_search_rejects_an_oversized_upload(client):
    oversized = _png_bytes(MAX_UPLOAD_BYTES + 1024)
    response = client.post("/search/image", files={"image": ("big.png", oversized, "image/png")})

    assert response.status_code == 413
    assert "MB" in response.json()["detail"]


def test_image_search_rejects_a_non_image_content_type(client):
    response = client.post("/search/image", files={"image": ("notes.txt", b"hello", "text/plain")})

    assert response.status_code == 415


def test_image_search_rejects_an_empty_upload(client):
    response = client.post("/search/image", files={"image": ("empty.png", b"", "image/png")})

    assert response.status_code == 400


def test_image_search_rejects_a_missing_file(client):
    assert client.post("/search/image", files={}).status_code == 422


# ---------------------------------------------------------------------------
# VQA
# ---------------------------------------------------------------------------


def test_vqa_answers_a_question(client):
    response = client.post(
        "/vqa",
        files={"image": ("photo.png", _png_bytes(), "image/png")},
        data={"question": "How many dogs are there?"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "two"
    assert body["question"] == "How many dogs are there?"


def test_vqa_rejects_an_empty_question(client):
    response = client.post(
        "/vqa",
        files={"image": ("p.png", _png_bytes(), "image/png")},
        data={"question": "   "},
    )

    assert response.status_code == 400


def test_vqa_rejects_a_missing_question(client):
    response = client.post("/vqa", files={"image": ("p.png", _png_bytes(), "image/png")})
    assert response.status_code == 422


def test_vqa_applies_the_same_size_cap(client):
    response = client.post(
        "/vqa",
        files={"image": ("big.png", _png_bytes(MAX_UPLOAD_BYTES + 1), "image/png")},
        data={"question": "what?"},
    )

    assert response.status_code == 413


# ---------------------------------------------------------------------------
# The published schema
# ---------------------------------------------------------------------------


def test_openapi_schema_documents_every_endpoint(client):
    paths = client.get("/openapi.json").json()["paths"]

    assert set(paths) == {"/health", "/search/text", "/search/image", "/vqa"}


def test_docs_page_is_served(client):
    assert client.get("/docs").status_code == 200


# ---------------------------------------------------------------------------
# End to end, against the real models
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def live_client():
    """A client running the real lifespan, or a skip if assets are missing."""
    from src.embedding import cache

    if not cache.exists():
        pytest.skip("no embedding cache; run the encode step first")

    try:
        with TestClient(app) as test_client:
            yield test_client
    except Exception as error:
        pytest.skip(f"could not start the app: {error}")


@pytest.mark.model
@pytest.mark.data
def test_live_text_search_finds_something_relevant(live_client):
    body = live_client.post(
        "/search/text", data={"query": "a dog running on grass"}, params={"k": 5}
    ).json()

    assert body["count"] == 5
    # A real match should be clearly positive, not marginally above zero.
    assert body["results"][0]["score"] > 0.2
    assert body["results"][0]["captions"]


@pytest.mark.model
@pytest.mark.data
def test_live_image_search_ranks_the_uploaded_corpus_image_first(live_client):
    from src.data.dataset import load_records

    record = load_records(split="test", require_files=True)[0]
    payload = record.path.read_bytes()

    body = live_client.post(
        "/search/image",
        files={"image": (record.path.name, payload, "image/jpeg")},
        params={"k": 3},
    ).json()

    assert body["results"][0]["image_id"] == record.image_id
    assert body["results"][0]["score"] > 0.99


@pytest.mark.model
@pytest.mark.data
def test_live_vqa_answers_about_a_real_image(live_client):
    from src.data.dataset import load_records

    record = load_records(split="test", require_files=True)[0]

    body = live_client.post(
        "/vqa",
        files={"image": (record.path.name, record.path.read_bytes(), "image/jpeg")},
        data={"question": "Is there an animal in this image?"},
    ).json()

    assert body["answer"].strip()
    assert len(body["answer"].split()) <= 6
