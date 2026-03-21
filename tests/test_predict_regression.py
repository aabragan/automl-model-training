"""Tests for evaluate.predict_regression (prediction-time artifacts)."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from automl_model_training.evaluate.predict_regression import save_regression_outputs


def test_saves_prediction_stats(tmp_path: Path):
    result = pd.DataFrame({
        "feat_a": [1.0, 2.0, 3.0],
        "target": [10.0, 20.0, 30.0],
        "target_predicted": [10.5, 19.5, 30.2],
    })

    save_regression_outputs(result, "target", tmp_path)

    assert (tmp_path / "prediction_stats.json").exists()
    with open(tmp_path / "prediction_stats.json") as f:
        stats = json.load(f)
    assert "mean" in stats
    assert "r2" in stats  # ground truth present


def test_no_residuals_without_ground_truth(tmp_path: Path):
    result = pd.DataFrame({
        "feat_a": [1.0, 2.0, 3.0],
        "price_predicted": [10.5, 19.5, 30.2],
    })

    save_regression_outputs(result, "price", tmp_path)

    with open(tmp_path / "prediction_stats.json") as f:
        stats = json.load(f)
    assert "mean" in stats
    assert "r2" not in stats  # no ground truth


def test_adds_residual_column_when_ground_truth(tmp_path: Path):
    result = pd.DataFrame({
        "target": [10.0, 20.0, 30.0],
        "target_predicted": [10.5, 19.5, 30.2],
    })

    save_regression_outputs(result, "target", tmp_path)

    assert "residual" in result.columns
