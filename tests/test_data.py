"""Tests for automl_model_training.data."""

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from automl_model_training.data import load_and_prepare


def _write_csv(path: Path, n: int = 100, label: str = "target") -> str:
    """Write a small CSV and return its path as a string."""
    rng = np.random.RandomState(0)
    df = pd.DataFrame({
        "feat_a": rng.randn(n),
        "feat_b": rng.randn(n),
        "drop_me": rng.randn(n),
        label: rng.choice([0, 1], n),
    })
    csv_path = path / "data.csv"
    df.to_csv(csv_path, index=False)
    return str(csv_path)


def test_load_and_prepare_returns_correct_shapes(tmp_path: Path):
    csv = _write_csv(tmp_path)
    out = tmp_path / "out"

    train, test, train_n, test_n, num_cols = load_and_prepare(
        csv_path=csv,
        label="target",
        features_to_drop=[],
        test_size=0.2,
        random_state=42,
        output_dir=str(out),
    )

    assert len(train) + len(test) == 100
    assert len(train) == len(train_n)
    assert len(test) == len(test_n)
    assert set(train.columns) == set(test.columns)


def test_load_and_prepare_drops_features(tmp_path: Path):
    csv = _write_csv(tmp_path)
    out = tmp_path / "out"

    train, *_ = load_and_prepare(
        csv_path=csv,
        label="target",
        features_to_drop=["drop_me"],
        test_size=0.2,
        random_state=42,
        output_dir=str(out),
    )

    assert "drop_me" not in train.columns


def test_load_and_prepare_ignores_missing_drop_cols(tmp_path: Path):
    csv = _write_csv(tmp_path)
    out = tmp_path / "out"

    train, *_ = load_and_prepare(
        csv_path=csv,
        label="target",
        features_to_drop=["nonexistent_col"],
        test_size=0.2,
        random_state=42,
        output_dir=str(out),
    )

    # Should not raise, original columns intact
    assert "feat_a" in train.columns
