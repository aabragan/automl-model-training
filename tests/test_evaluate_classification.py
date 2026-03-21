"""Tests for evaluate.classification (train-time artifacts)."""

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd

from automl_model_training.evaluate.classification import save_classification_artifacts


def _make_predictor_and_data():
    """Build a mock predictor and test data for binary classification."""
    rng = np.random.RandomState(42)
    n = 50
    y_true = rng.choice([0, 1], n, p=[0.6, 0.4])
    test_df = pd.DataFrame(
        {
            "feat_a": rng.randn(n),
            "target": y_true,
        }
    )

    pred = MagicMock()
    pred.predict.return_value = pd.Series(y_true)  # perfect predictions
    proba = pd.DataFrame(
        {
            0: np.where(y_true == 0, 0.9, 0.1),
            1: np.where(y_true == 1, 0.9, 0.1),
        }
    )
    pred.predict_proba.return_value = proba
    return pred, test_df


def test_saves_all_classification_files(tmp_path: Path):
    pred, test_df = _make_predictor_and_data()
    save_classification_artifacts(pred, test_df, "target", tmp_path)

    expected = [
        "test_predictions.csv",
        "confusion_matrix.csv",
        "classification_report.csv",
        "roc_curve.csv",
        "roc_auc.json",
        "precision_recall_curve.csv",
        "average_precision.json",
    ]
    for fname in expected:
        assert (tmp_path / fname).exists(), f"Missing {fname}"


def test_roc_auc_is_valid(tmp_path: Path):
    import json

    pred, test_df = _make_predictor_and_data()
    save_classification_artifacts(pred, test_df, "target", tmp_path)

    with open(tmp_path / "roc_auc.json") as f:
        data = json.load(f)
    assert 0 <= data["roc_auc"] <= 1
