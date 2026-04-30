"""Session-scoped fixtures that run real AutoGluon training on sample data.

These tests catch library-version drift (e.g., shap layout changes, sklearn
API changes) that mock-based unit tests miss. They are gated behind the
``slow`` pytest marker and deselected by the default ``addopts`` in
``pyproject.toml``.

Run them explicitly:

    uv run pytest -m slow tests/integration/

Artifacts are cached to ``tests/integration/_cache/`` (gitignored) so
repeated runs in the same session — or across sessions — do not retrain.
Delete the cache directory to force retraining on the next run.
"""

from __future__ import annotations

import logging
import shutil
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path

import pytest

from automl_model_training.tools import tool_train

# Cache location. Kept inside tests/integration/ so the gitignore rule is local.
_CACHE_ROOT = Path(__file__).parent / "_cache"

# Per-dataset cache dir names. Using a fixed name (not timestamped) makes
# artifact reuse possible across pytest invocations.
_FRAUD_CACHE = _CACHE_ROOT / "fraud"
_FLOWER_CACHE = _CACHE_ROOT / "flower"
_HOUSE_CACHE = _CACHE_ROOT / "house"


def _cached_training(
    cache_dir: Path,
    csv_path: str,
    label: str,
    preset: str = "medium",
    time_limit: int = 30,
    explain: bool = True,
) -> str:
    """Train once; return the cached run_dir on subsequent calls.

    AutoGluon writes a timestamped subdirectory inside the output dir we
    give it. We detect that by looking for any subdir matching
    ``llm_train_*`` after training, and re-use it on cache hits. If the
    cache is present but empty or malformed, we clear and re-train.
    """
    # Find an existing cached run, if any
    existing = sorted(cache_dir.glob("llm_train_*/"))
    if existing and (existing[-1] / "leaderboard_test.csv").exists():
        return str(existing[-1])

    # No cache — train from scratch. Remove any half-baked cache dir first.
    if cache_dir.exists():
        shutil.rmtree(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Suppress AutoGluon's extremely verbose stdout/stderr during training.
    # pytest -s would still expose these if a user really wants them.
    logging.getLogger().setLevel(logging.ERROR)
    with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
        result = tool_train(
            csv_path=csv_path,
            label=label,
            preset=preset,
            time_limit=time_limit,
            explain=explain,
            output_dir=str(cache_dir),
        )
    return str(result["run_dir"])


@pytest.fixture(scope="session")
def fraud_run_dir() -> str:
    """Real-trained fraud_detection.csv run directory (binary classification).

    Cached under ``tests/integration/_cache/fraud/``. First call takes
    ~10s; subsequent calls return instantly.
    """
    return _cached_training(
        _FRAUD_CACHE,
        csv_path="samples/fraud_detection.csv",
        label="is_fraud",
        preset="medium",
        time_limit=30,
        explain=True,
    )


@pytest.fixture(scope="session")
def flower_run_dir() -> str:
    """Real-trained flower_species.csv run directory (multiclass classification)."""
    return _cached_training(
        _FLOWER_CACHE,
        csv_path="samples/flower_species.csv",
        label="species",
        preset="medium",
        time_limit=30,
        explain=True,
    )


@pytest.fixture(scope="session")
def house_run_dir() -> str:
    """Real-trained house_prices.csv run directory (regression)."""
    return _cached_training(
        _HOUSE_CACHE,
        csv_path="samples/house_prices.csv",
        label="price",
        preset="medium",
        time_limit=60,  # Regression takes longer — 30s sometimes hits the limit
        explain=True,
    )
