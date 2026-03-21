"""Tests for evaluate.regression (train-time artifacts)."""

import json
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd

from automl_model_training.evaluate.regression import save_regression_artifacts


def _make_predictor_and_data():
    rng = np.random.RandomState(42)
    n = 50
    y_true = rng.randn(n) * 10
    y_pred = y_true + rng.randn(n) * 0.5  # small noise

    test_df = pd.DataFrame({"feat_a": rng.randn(n), "target": y_true})

    pred = MagicMock()
    pred.predict.return_value = pd.Series(y_pred)
    return pred, test_df


def test_saves_regression_files(tmp_path: Path):
    pred, test_df = _make_predictor_and_data()
    save_regression_artifacts(pred, test_df, "target", tmp_path)

    assert (tmp_path / "test_predictions.csv").exists()
    assert (tmp_path / "residual_stats.json").exists()
    assert (tmp_path / "residual_distribution.csv").exists()


def test_residual_stats_keys(tmp_path: Path):
    pred, test_df = _make_predictor_and_data()
    save_regression_artifacts(pred, test_df, "target", tmp_path)

    with open(tmp_path / "residual_stats.json") as f:
        stats = json.load(f)

    expected_keys = {
        "mean_residual", "median_residual", "std_residual",
        "min_residual", "max_residual", "mean_absolute_error",
        "root_mean_squared_error", "r2",
    }
    assert expected_keys == set(stats.keys())


def test_r2_near_one_for_good_predictions(tmp_path: Path):
    pred, test_df = _make_predictor_and_data()
    save_regression_artifacts(pred, test_df, "target", tmp_path)

    with open(tmp_path / "residual_stats.json") as f:
        stats = json.load(f)

    assert stats["r2"] > 0.95
