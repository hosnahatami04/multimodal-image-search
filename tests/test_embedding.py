"""Tests for the embedding layer: normalisation, cache integrity, index shape.

Split in two. The pure-numpy tests -- normalisation, the cache's validation
rules, similarity arithmetic -- run everywhere including CI. The tests that
need CLIP's 600 MB of weights are marked `model` and skipped there.

The distinction matters: the cache's job is to refuse a matrix that would
produce wrong results downstream, and that logic is worth testing on every
commit, not only on a machine that has already downloaded the model.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from src import config
from src.embedding import cache
from src.embedding.clip_encoder import EncoderError, l2_normalise, similarity


@pytest.fixture
def embeddings_dir(tmp_path, monkeypatch):
    """Point the cache at a temporary directory."""
    directory = tmp_path / "embeddings"
    directory.mkdir()
    monkeypatch.setattr(config, "EMBEDDINGS_DIR", directory)
    return directory


def _unit_vectors(count: int, dim: int = 8, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return l2_normalise(rng.normal(size=(count, dim)).astype(np.float32))


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------


def test_normalise_gives_unit_rows():
    vectors = np.array([[3.0, 4.0], [1.0, 0.0], [-5.0, 12.0]], dtype=np.float32)
    normalised = l2_normalise(vectors)

    assert np.allclose(np.linalg.norm(normalised, axis=1), 1.0)


def test_normalise_preserves_direction():
    vectors = np.array([[3.0, 4.0]], dtype=np.float32)
    normalised = l2_normalise(vectors)

    assert np.allclose(normalised, [[0.6, 0.8]])


def test_normalise_is_idempotent():
    vectors = _unit_vectors(5)
    assert np.allclose(l2_normalise(vectors), vectors, atol=1e-6)


def test_normalise_handles_a_zero_vector():
    # Must not produce NaN: a NaN row would poison every similarity it touches
    # and the failure would surface far from here.
    result = l2_normalise(np.zeros((1, 4), dtype=np.float32))
    assert not np.isnan(result).any()


def test_normalise_returns_float32():
    assert l2_normalise(np.ones((2, 3), dtype=np.float64)).dtype == np.float32


# ---------------------------------------------------------------------------
# Similarity
# ---------------------------------------------------------------------------


def test_identical_vectors_score_one():
    vector = _unit_vectors(1)
    assert similarity(vector, vector)[0, 0] == pytest.approx(1.0, abs=1e-5)


def test_opposite_vectors_score_minus_one():
    vector = _unit_vectors(1)
    assert similarity(vector, -vector)[0, 0] == pytest.approx(-1.0, abs=1e-5)


def test_orthogonal_vectors_score_zero():
    left = np.array([[1.0, 0.0]], dtype=np.float32)
    right = np.array([[0.0, 1.0]], dtype=np.float32)
    assert similarity(left, right)[0, 0] == pytest.approx(0.0, abs=1e-6)


def test_similarity_stays_in_cosine_range():
    scores = similarity(_unit_vectors(20, seed=1), _unit_vectors(30, seed=2))
    assert scores.min() >= -1.0001
    assert scores.max() <= 1.0001


def test_similarity_shape_is_left_by_right():
    assert similarity(_unit_vectors(4), _unit_vectors(7)).shape == (4, 7)


def test_similarity_rejects_mismatched_dimensions():
    with pytest.raises(EncoderError, match="dimension mismatch"):
        similarity(_unit_vectors(2, dim=8), _unit_vectors(2, dim=16))


# ---------------------------------------------------------------------------
# Cache: round-trip
# ---------------------------------------------------------------------------


def test_cache_round_trips(embeddings_dir):
    vectors = _unit_vectors(6)
    ids = [f"img_{i:03d}" for i in range(6)]

    cache.save(vectors, ids)
    entry = cache.load()

    assert np.allclose(entry.vectors, vectors)
    assert entry.image_ids == tuple(ids)
    assert entry.dim == vectors.shape[1]


def test_cache_reports_row_for_an_id(embeddings_dir):
    ids = [f"img_{i:03d}" for i in range(4)]
    cache.save(_unit_vectors(4), ids)

    assert cache.load().index_of("img_002") == 2


def test_cache_rejects_an_unknown_id(embeddings_dir):
    cache.save(_unit_vectors(3), ["a", "b", "c"])

    with pytest.raises(cache.CacheError, match="not in this cache"):
        cache.load().index_of("zzz")


def test_exists_reflects_what_is_on_disk(embeddings_dir):
    assert not cache.exists()
    cache.save(_unit_vectors(2), ["a", "b"])
    assert cache.exists()


# ---------------------------------------------------------------------------
# Cache: refusing what would be wrong downstream
# ---------------------------------------------------------------------------


def test_cache_refuses_unnormalised_vectors(embeddings_dir):
    # The whole search layer treats a dot product as cosine similarity. Storing
    # un-normalised vectors would silently rank by magnitude instead.
    with pytest.raises(cache.CacheError, match="not unit-norm"):
        cache.save(np.full((3, 4), 2.0, dtype=np.float32), ["a", "b", "c"])


def test_cache_refuses_mismatched_id_count(embeddings_dir):
    with pytest.raises(cache.CacheError, match="3 ids for 5 vectors"):
        cache.save(_unit_vectors(5), ["a", "b", "c"])


def test_cache_refuses_duplicate_ids(embeddings_dir):
    with pytest.raises(cache.CacheError, match="not unique"):
        cache.save(_unit_vectors(3), ["a", "b", "a"])


def test_cache_refuses_a_one_dimensional_array(embeddings_dir):
    with pytest.raises(cache.CacheError, match="2-D"):
        cache.save(np.ones(8, dtype=np.float32), ["a"])


def test_loading_an_absent_cache_explains_what_to_run(embeddings_dir):
    with pytest.raises(cache.CacheError, match="build_index"):
        cache.load()


def test_cache_detects_a_different_model(embeddings_dir):
    cache.save(_unit_vectors(3), ["a", "b", "c"], model_id="model/one")

    with pytest.raises(cache.CacheError, match="no cached embeddings"):
        cache.load(model_id="model/two")


def test_cache_detects_a_changed_revision(embeddings_dir):
    # Same model id, different weights: the vectors are not comparable, and
    # silently mixing them would shift every metric.
    cache.save(_unit_vectors(3), ["a", "b", "c"], revision="a" * 40)

    with pytest.raises(cache.CacheError, match="revision"):
        cache.load(revision="b" * 40)


def test_cache_detects_an_id_list_that_does_not_match(embeddings_dir):
    cache.save(_unit_vectors(3), ["a", "b", "c"])

    with pytest.raises(cache.CacheError, match="rebuild the index"):
        cache.load(expected_ids=["a", "b", "c", "d"])


def test_cache_detects_a_reordered_id_list(embeddings_dir):
    # Same ids, different order: every row would be attributed to the wrong
    # image, and nothing downstream could notice.
    cache.save(_unit_vectors(3), ["a", "b", "c"])

    with pytest.raises(cache.CacheError, match="rebuild the index"):
        cache.load(expected_ids=["c", "b", "a"])


def test_cache_detects_an_edited_sidecar(embeddings_dir):
    cache.save(_unit_vectors(3), ["a", "b", "c"])

    _, meta_path = cache.cache_paths(config.CLIP_MODEL_ID)
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["image_ids"] = ["a", "b", "x"]
    meta_path.write_text(json.dumps(meta), encoding="utf-8")

    with pytest.raises(cache.CacheError, match="edited after it was written"):
        cache.load()


def test_cache_detects_a_stale_format_version(embeddings_dir):
    cache.save(_unit_vectors(2), ["a", "b"])

    _, meta_path = cache.cache_paths(config.CLIP_MODEL_ID)
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["format_version"] = 999
    meta_path.write_text(json.dumps(meta), encoding="utf-8")

    with pytest.raises(cache.CacheError, match="cache format"):
        cache.load()


def test_separate_tags_do_not_collide(embeddings_dir):
    cache.save(_unit_vectors(3, seed=1), ["a", "b", "c"], tag="images")
    cache.save(_unit_vectors(5, seed=2), list("vwxyz"), tag="captions")

    assert cache.load(tag="images").vectors.shape[0] == 3
    assert cache.load(tag="captions").vectors.shape[0] == 5


# ---------------------------------------------------------------------------
# The real encoder
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def encoder():
    """A real CLIP encoder, or a skip if the weights are not cached."""
    from src.embedding.clip_encoder import ClipEncoder

    instance = ClipEncoder(batch_size=4)
    try:
        _ = instance.dim
    except Exception as error:
        pytest.skip(f"CLIP weights unavailable: {error}")
    return instance


@pytest.mark.model
def test_encoder_reports_the_expected_dimension(encoder):
    assert encoder.dim == config.EMBEDDING_DIM


@pytest.mark.model
def test_encoded_text_is_unit_norm(encoder):
    vectors = encoder.encode_texts(["a dog on grass", "a red car"])

    assert vectors.shape == (2, encoder.dim)
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-5)


@pytest.mark.model
def test_encoding_is_deterministic(encoder):
    first = encoder.encode_text("a dog running through a field")
    second = encoder.encode_text("a dog running through a field")

    assert np.allclose(first, second, atol=1e-6)


@pytest.mark.model
def test_batch_and_single_encoding_agree(encoder):
    # Dropout left on, or the model left in train mode, would break this and
    # nothing else in the pipeline would notice.
    texts = ["a dog on grass", "a red car", "a plate of food", "a snowy mountain"]

    batched = encoder.encode_texts(texts)
    individually = np.stack([encoder.encode_text(text) for text in texts])

    assert np.allclose(batched, individually, atol=1e-5)


@pytest.mark.model
def test_related_text_scores_above_unrelated(encoder):
    anchor = encoder.encode_text("a brown dog running on green grass")
    related = encoder.encode_text("a dog plays outside on a lawn")
    unrelated = encoder.encode_text("a laptop on an office desk")

    assert similarity(anchor, related)[0, 0] > similarity(anchor, unrelated)[0, 0]


@pytest.mark.model
def test_empty_text_is_rejected(encoder):
    with pytest.raises(EncoderError, match="empty text"):
        encoder.encode_texts(["a dog", "   "])


@pytest.mark.model
def test_encoding_nothing_returns_an_empty_matrix(encoder):
    assert encoder.encode_texts([]).shape == (0, encoder.dim)
    assert encoder.encode_images([]).shape == (0, encoder.dim)


@pytest.mark.model
@pytest.mark.data
def test_image_is_closer_to_its_own_captions(encoder):
    """The sanity check the plan requires before anything is built on top."""
    from src.data.dataset import load_records

    try:
        records = load_records(split="test", require_files=True)
    except Exception as error:
        pytest.skip(f"dataset unavailable: {error}")

    subject, other = records[0], records[len(records) // 2]

    image = encoder.encode_image(subject.path)
    own = encoder.encode_texts(list(subject.captions))
    foreign = encoder.encode_texts(list(other.captions))

    assert similarity(image, own).mean() > similarity(image, foreign).mean()


@pytest.mark.model
@pytest.mark.data
def test_missing_image_file_is_reported_clearly(encoder):
    with pytest.raises(EncoderError, match="image not found"):
        encoder.encode_images([config.IMAGES_DIR / "does_not_exist_12345.jpg"])
