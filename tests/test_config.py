"""Tests for automl_model_training.config."""

from pathlib import Path

from automl_model_training.config import (
    DEFAULT_LABEL,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_PRESET,
    DEFAULT_RANDOM_STATE,
    DEFAULT_TEST_SIZE,
    make_run_dir,
)


def test_defaults_are_sensible():
    assert DEFAULT_LABEL == "target"
    assert 0 < DEFAULT_TEST_SIZE < 1
    assert isinstance(DEFAULT_RANDOM_STATE, int)
    assert DEFAULT_PRESET == "best"
    assert DEFAULT_OUTPUT_DIR == "output"


def test_make_run_dir_creates_directory(tmp_path: Path):
    run_dir = make_run_dir(str(tmp_path), prefix="train")
    assert Path(run_dir).is_dir()
    assert "train_" in Path(run_dir).name


def test_make_run_dir_unique(tmp_path: Path):
    """Two calls should produce different directories."""
    import time

    d1 = make_run_dir(str(tmp_path), prefix="run")
    time.sleep(1.1)  # timestamp resolution is 1 second
    d2 = make_run_dir(str(tmp_path), prefix="run")
    assert d1 != d2


def test_make_run_dir_nested(tmp_path: Path):
    """Works even when base_dir doesn't exist yet."""
    nested = tmp_path / "a" / "b" / "c"
    run_dir = make_run_dir(str(nested), prefix="test")
    assert Path(run_dir).is_dir()
