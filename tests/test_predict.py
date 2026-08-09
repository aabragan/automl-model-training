"""Tests for prediction pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd

from automl_model_training.predict import load_predictor, predict_and_save


class TestPredictAndSave:
    def _make_predictor(self, problem_type: str = "binary") -> MagicMock:
        pred = MagicMock()
        pred.label = "target"
        pred.problem_type = problem_type
        pred.model_best = "LightGBM"
        pred.predict.return_value = pd.Series([0, 1, 0, 1, 0])
        pred.evaluate.return_value = {"f1": 0.85, "accuracy": 0.90}

        proba = pd.DataFrame({0: [0.8, 0.2, 0.7, 0.3, 0.9], 1: [0.2, 0.8, 0.3, 0.7, 0.1]})
        pred.predict_proba.return_value = proba
        return pred

    def test_saves_predictions_csv(self, tmp_path: Path):
        pred = self._make_predictor()
        data = pd.DataFrame({"feat_a": [1, 2, 3, 4, 5], "target": [0, 1, 0, 1, 0]})

        result = predict_and_save(pred, data, str(tmp_path))

        assert (tmp_path / "predictions.csv").exists()
        assert "target_predicted" in result.columns

    def test_saves_prediction_summary(self, tmp_path: Path):
        pred = self._make_predictor()
        data = pd.DataFrame({"feat_a": [1, 2, 3, 4, 5], "target": [0, 1, 0, 1, 0]})

        predict_and_save(pred, data, str(tmp_path))

        summary_path = tmp_path / "prediction_summary.json"
        assert summary_path.exists()
        summary = json.loads(summary_path.read_text())
        assert summary["problem_type"] == "binary"
        assert summary["num_rows"] == 5
        assert summary["has_ground_truth"] is True
        assert "eval_scores" in summary

    def test_no_ground_truth(self, tmp_path: Path):
        pred = self._make_predictor()
        # No "target" column in data
        data = pd.DataFrame({"feat_a": [1, 2, 3, 4, 5]})

        predict_and_save(pred, data, str(tmp_path))

        summary = json.loads((tmp_path / "prediction_summary.json").read_text())
        assert summary["has_ground_truth"] is False
        assert "eval_scores" not in summary

    def test_regression_predictions(self, tmp_path: Path):
        pred = self._make_predictor(problem_type="regression")
        pred.predict.return_value = pd.Series([1.1, 2.2, 3.3, 4.4, 5.5])
        data = pd.DataFrame({"feat_a": [1, 2, 3, 4, 5], "target": [1.0, 2.0, 3.0, 4.0, 5.0]})

        result = predict_and_save(pred, data, str(tmp_path))

        assert "target_predicted" in result.columns
        assert (tmp_path / "predictions.csv").exists()

    def test_does_not_mutate_input(self, tmp_path: Path):
        pred = self._make_predictor()
        data = pd.DataFrame({"feat_a": [1, 2, 3, 4, 5]})
        original_cols = list(data.columns)

        predict_and_save(pred, data, str(tmp_path))

        assert list(data.columns) == original_cols


class TestLoadPredictor:
    @patch("automl_model_training.predict.TabularPredictor")
    def test_loads_and_returns_predictor(self, mock_cls: MagicMock):
        mock_pred = MagicMock()
        mock_pred.problem_type = "binary"
        mock_pred.eval_metric = "f1"
        mock_pred.label = "target"
        mock_cls.load.return_value = mock_pred

        result = load_predictor("some/model/dir")

        mock_cls.load.assert_called_once_with("some/model/dir")
        assert result is mock_pred


class TestDriftDetection:
    def _make_predictor(self) -> MagicMock:
        pred = MagicMock()
        pred.label = "target"
        pred.problem_type = "binary"
        pred.model_best = "LightGBM"
        pred.predict.return_value = pd.Series([0, 1, 0, 1, 0])
        pred.evaluate.return_value = {"f1": 0.85}
        proba = pd.DataFrame({0: [0.8, 0.2, 0.7, 0.3, 0.9], 1: [0.2, 0.8, 0.3, 0.7, 0.1]})
        pred.predict_proba.return_value = proba
        return pred

    @patch("automl_model_training.predict.save_drift_report")
    @patch("automl_model_training.predict.detect_drift")
    def test_runs_drift_check_when_train_data_given(
        self, mock_detect: MagicMock, mock_save: MagicMock, tmp_path: Path
    ):
        pred = self._make_predictor()
        data = pd.DataFrame({"feat_a": [1, 2, 3, 4, 5], "target": [0, 1, 0, 1, 0]})
        train_data = pd.DataFrame({"feat_a": [5, 4, 3, 2, 1], "target": [1, 0, 1, 0, 1]})
        mock_detect.return_value = [{"feature": "feat_a", "psi": 0.3}]

        predict_and_save(pred, data, str(tmp_path), train_data=train_data)

        mock_detect.assert_called_once_with(train_data, data, "target")
        mock_save.assert_called_once()

    @patch("automl_model_training.predict.save_drift_report")
    @patch("automl_model_training.predict.detect_drift")
    def test_skips_report_when_no_drift_results(
        self, mock_detect: MagicMock, mock_save: MagicMock, tmp_path: Path
    ):
        pred = self._make_predictor()
        data = pd.DataFrame({"feat_a": [1, 2, 3, 4, 5], "target": [0, 1, 0, 1, 0]})
        train_data = pd.DataFrame({"feat_a": [5, 4, 3, 2, 1], "target": [1, 0, 1, 0, 1]})
        mock_detect.return_value = []

        predict_and_save(pred, data, str(tmp_path), train_data=train_data)

        mock_detect.assert_called_once()
        mock_save.assert_not_called()

    @patch("automl_model_training.predict.detect_drift")
    def test_no_drift_check_without_train_data(self, mock_detect: MagicMock, tmp_path: Path):
        pred = self._make_predictor()
        data = pd.DataFrame({"feat_a": [1, 2, 3, 4, 5], "target": [0, 1, 0, 1, 0]})

        predict_and_save(pred, data, str(tmp_path))

        mock_detect.assert_not_called()
