"""Tests for the deployment surface: the Dockerfile, CI, and reproducibility.

None of these build an image or start a container -- that belongs in CI, where
it can take twenty minutes. What they check is the set of properties that are
easy to lose in an edit and expensive to discover later: that the image still
runs as a non-root user, that it is still configured to work offline, that the
CPU wheel index is still pinned, and that every model the project uses is still
pinned to a commit rather than a moving branch.

Each of these has failed silently in some project somewhere. A Dockerfile that
quietly starts running as root, or a torch install that quietly starts pulling
the CUDA build and triples the image size, produces no error -- just a worse
artefact that nobody notices.
"""

from __future__ import annotations

import re

import pytest

from src import config

DOCKERFILE = config.PROJECT_ROOT / "Dockerfile"
DOCKERIGNORE = config.PROJECT_ROOT / ".dockerignore"
CI_WORKFLOW = config.PROJECT_ROOT / ".github" / "workflows" / "ci.yml"
REQUIREMENTS = config.PROJECT_ROOT / "requirements.txt"


def _dockerfile() -> str:
    if not DOCKERFILE.exists():
        pytest.skip("no Dockerfile")
    return DOCKERFILE.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Dockerfile
# ---------------------------------------------------------------------------


def test_the_build_is_multi_stage():
    # A single-stage image carries build-essential and git into production.
    stages = re.findall(r"^FROM .+ AS (\w+)", _dockerfile(), re.MULTILINE)
    assert len(stages) >= 2, f"expected a multi-stage build, found stages: {stages}"


def test_torch_comes_from_the_cpu_index():
    # Without the index URL, pip pulls the CUDA build: ~2.5 GB of GPU libraries
    # in an image that runs on CPU.
    text = _dockerfile()
    assert "download.pytorch.org/whl/cpu" in text


def test_the_container_does_not_run_as_root():
    text = _dockerfile()
    assert re.search(r"^USER \w+", text, re.MULTILINE), "no USER instruction"
    assert not re.search(r"^USER root\s*$", text, re.MULTILINE)


def test_the_runtime_is_configured_for_offline_use():
    # The weights are baked in. These variables make that a guarantee rather
    # than a coincidence: with them set, a missing file fails loudly instead of
    # silently reaching for the network.
    text = _dockerfile()
    assert "HF_HUB_OFFLINE=1" in text
    assert "TRANSFORMERS_OFFLINE=1" in text


def test_the_weights_are_fetched_during_the_build():
    text = _dockerfile()
    assert "fetch_model.py" in text
    assert "clip-vit-base-patch32" in text
    assert "blip-vqa-base" in text


def test_libgomp_is_installed():
    # torch links against OpenMP. Without libgomp1 the import succeeds and the
    # first tensor operation segfaults, which is a confusing way to discover a
    # missing shared library.
    assert "libgomp1" in _dockerfile()


def test_a_healthcheck_is_defined():
    text = _dockerfile()
    assert "HEALTHCHECK" in text
    assert "/health" in text


def test_the_healthcheck_allows_time_for_model_loading():
    # Startup loads CLIP and the index. A short start period would have the
    # orchestrator kill the container mid-load, repeatedly.
    match = re.search(r"--start-period=(\d+)s", _dockerfile())
    assert match, "no start-period on the healthcheck"
    assert int(match.group(1)) >= 60


def test_only_one_uvicorn_worker():
    # Each worker holds its own ~2 GB of models. Scaling means more containers.
    text = _dockerfile()
    assert "--workers" in text
    assert re.search(r'"--workers",\s*"1"', text)


def test_dependencies_are_installed_before_source_is_copied():
    """Editing a source file must not invalidate the dependency layer."""
    text = _dockerfile()
    requirements_at = text.index("COPY requirements.txt")
    source_at = text.index("COPY src/")
    assert requirements_at < source_at


# ---------------------------------------------------------------------------
# .dockerignore
# ---------------------------------------------------------------------------


def test_the_raw_corpus_is_excluded_from_the_build_context():
    if not DOCKERIGNORE.exists():
        pytest.skip("no .dockerignore")

    text = DOCKERIGNORE.read_text(encoding="utf-8")
    # 1 GB of JPEGs uploaded to the daemon on every build, for files the API
    # never reads.
    assert "data/raw/images/" in text


def test_git_history_is_excluded_from_the_build_context():
    if not DOCKERIGNORE.exists():
        pytest.skip("no .dockerignore")

    assert ".git/" in DOCKERIGNORE.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# CI
# ---------------------------------------------------------------------------


def _workflow() -> str:
    if not CI_WORKFLOW.exists():
        pytest.skip("no CI workflow")
    return CI_WORKFLOW.read_text(encoding="utf-8")


def test_ci_skips_the_model_and_data_tests():
    # Otherwise CI would need 2 GB of weights and 1 GB of images on every run.
    assert "not model and not data" in _workflow()


def test_ci_checks_formatting_as_well_as_lint():
    text = _workflow()
    assert "ruff check" in text
    assert "ruff format --check" in text


def test_ci_runs_the_metric_gate():
    assert "check_metrics.py" in _workflow()


def test_ci_uses_the_cpu_torch_index():
    assert "download.pytorch.org/whl/cpu" in _workflow()


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------


def test_model_revisions_are_commit_shas():
    """A Hub repo is mutable; `main` is not a version.

    Both revisions must be 40-character hex commit ids, so the weights behind a
    run are exactly the weights the committed metrics were produced with.
    """
    for label, revision in (
        ("CLIP", config.CLIP_REVISION),
        ("BLIP", config.BLIP_REVISION),
        ("dataset", config.DATASET_REVISION),
    ):
        assert re.fullmatch(r"[0-9a-f]{40}", revision), f"{label} revision is not a commit sha"


def test_no_revision_is_a_branch_name():
    for revision in (config.CLIP_REVISION, config.BLIP_REVISION, config.DATASET_REVISION):
        assert revision not in {"main", "master", "latest", "HEAD"}


def test_every_dependency_is_pinned_exactly():
    """A range lets a future install change the metrics without a code change."""
    if not REQUIREMENTS.exists():
        pytest.skip("no requirements.txt")

    unpinned: list[str] = []
    for line in REQUIREMENTS.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "==" not in line:
            unpinned.append(line)

    assert not unpinned, f"unpinned dependencies: {unpinned}"


def test_the_seed_is_fixed():
    assert isinstance(config.SEED, int)


def test_the_split_is_the_published_one():
    # Keeping the Karpathy split is what makes these numbers comparable to
    # published ones rather than to nothing.
    assert config.EXPECTED_SPLIT_SIZES == {"train": 6000, "validation": 1000, "test": 1000}
    assert config.EXPECTED_NUM_IMAGES == 8000
