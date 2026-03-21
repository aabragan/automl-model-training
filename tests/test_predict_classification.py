"""Tests for evaluate.predict_classification (prediction-time artifacts)."""

from pathlib import Path

import numpy as np
import pandas as pd

from automl_model_training.evaluate.predict_classification import (
    save_classification_outputs,
)


def test_saves_probability_and_distribution(tmp_path, mock_binary_predictor, binary_data):
    _, test = binary_data
    result = test.copy()
    result["target_predicted"] = mock_binary_predictor.predict(test)

    save_classification_outputs(
        mock_binary_predictor, test, result, "target", tmp_path
    )

    assert (tmp_path / "probability_stats.csv").exists()
    assert (tmp_path / "prediction_distribution.csv").exists()


def test_adds_confidence_column(tmp_path, mock_binary_predictor, binary_data):
    _, test = binary_data
    result = test.copy()
    result["target_predicted"] = mock_binary_predictor.predict(test)

    save_classification_outputs(
        mock_binary_predictor, test, result, "target", tmp_path
    )

    assert "confidence" in result.columns
    assert (result["confidence"] >= 0).all()
    assert (result["confidence"] <= 1).all()


def test_saves_confusion_matrix_when_ground_truth(tmp_path, mock_binary_predictor, binary_data):
    _, test = binary_data
    result = test.copy()
    result["target_predicted"] = mock_binary_predictor.predict(test)

    save_classification_outputs(
        mock_binary_predictor, test, result, "target", tmp_path
    )

    # Ground truth column "target" exists in data → confusion matrix saved
    assert (tmp_path / "confusion_matrix.csv").exists()
    assert (tmp_path / "classification_report.csv").exists()
