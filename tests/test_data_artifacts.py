"""Tests for CSV artifacts written by load_and_prepare."""

from pathlib import Path

import numpy as np
import pandas as pd

from automl_model_training.data import load_and_prepare


def _write_csv(path: Path) -> str:
    rng = np.random.RandomState(0)
    df = pd.DataFrame({
        "feat_a": rng.randn(100),
        "feat_b": rng.randn(100),
        "target": rng.choice([0, 1], 100),
    })
    csv_path = path / "data.csv"
    df.to_csv(csv_path, index=False)
    return str(csv_path)


def test_raw_csvs_written(tmp_path: Path):
    csv = _write_csv(tmp_path)
    out = tmp_path / "out"
    load_and_prepare(csv, "target", [], 0.2, 42, str(out))

    assert (out / "train_raw.csv").exists()
    assert (out / "test_raw.csv").exists()


def test_normalized_csvs_written(tmp_path: Path):
    csv = _write_csv(tmp_path)
    out = tmp_path / "out"
    load_and_prepare(csv, "target", [], 0.2, 42, str(out))

    assert (out / "train_normalized.csv").exists()
    assert (out / "test_normalized.csv").exists()


def test_normalized_values_differ_from_raw(tmp_path: Path):
    csv = _write_csv(tmp_path)
    out = tmp_path / "out"
    train_raw, _, train_norm, _, _ = load_and_prepare(
        csv, "target", [], 0.2, 42, str(out)
    )

    # Normalized numeric columns should differ from raw
    assert not np.allclose(
        train_raw["feat_a"].values, train_norm["feat_a"].values
    )
