"""Tests for --auto-drop, --calibrate-threshold, --decision-threshold, and refit-best."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from automl_model_training.predict import predict_and_save
from automl_model_training.train import (
    _base_parser,
    _read_low_importance_features,
    train_and_evaluate,
)


def _make_mock_predictor(problem_type: str = "binary") -> MagicMock:
    pred = MagicMock()
    pred.label = "target"
    pred.problem_type = problem_type
    pred.eval_metric = "f1" if problem_type == "binary" else "root_mean_squared_error"
    pred.model_best = "LightGBM"
    pred.features.return_value = ["feat_a", "feat_b"]

    pred.predict.side_effect = lambda data, **kw: pd.Series(
        np.zeros(len(data), dtype=int), index=data.index
    )
    pred.predict_proba.side_effect = lambda data: pd.DataFrame(
        {0: np.full(len(data), 0.8), 1: np.full(len(data), 0.2)}, index=data.index
    )

    lb = pd.DataFrame(
        {
            "model": ["LightGBM", "CatBoost"],
            "score_val": [0.90, 0.85],
            "score_test": [0.88, 0.82],
            "fit_time": [10.0, 15.0],
            "pred_time_val": [0.1, 0.2],
        }
    )
    pred.leaderboard.return_value = lb
    pred.refit_full.return_value = {"LightGBM": "LightGBM_FULL"}
    pred.evaluate.return_value = {"f1": 0.88, "accuracy": 0.92}

    importance = pd.DataFrame(
        {"importance": [0.5, 0.3], "stddev": [0.01, 0.02]},
        index=["feat_a", "feat_b"],
    )
    pred.feature_importance.return_value = importance
    pred.info.return_value = {"model_info": {}}
    return pred


# --- _read_low_importance_features ---


class TestReadLowImportanceFeatures:
    def test_returns_low_importance_features(self, tmp_path: Path):
        imp = pd.DataFrame(
            {"importance": [0.5, 0.0005, -0.01]},
            index=["good", "near_zero", "harmful"],
        )
        imp.to_csv(tmp_path / "feature_importance.csv")

        result = _read_low_importance_features(str(tmp_path))

        assert "near_zero" in result
        assert "harmful" in result
        assert "good" not in result

    def test_returns_empty_when_no_file(self, tmp_path: Path):
        assert _read_low_importance_features(str(tmp_path)) == []

    def test_returns_empty_when_all_important(self, tmp_path: Path):
        imp = pd.DataFrame({"importance": [0.5, 0.3]}, index=["a", "b"])
        imp.to_csv(tmp_path / "feature_importance.csv")

        assert _read_low_importance_features(str(tmp_path)) == []


# --- --auto-drop CLI flag ---


class TestAutoDropFlag:
    def test_defaults_to_false(self):
        parser = _base_parser("test")
        args = parser.parse_args(["dummy.csv"])
        assert args.auto_drop is False

    def test_can_be_enabled(self):
        parser = _base_parser("test")
        args = parser.parse_args(["dummy.csv", "--auto-drop"])
        assert args.auto_drop is True


# --- --calibrate-threshold CLI flag ---


class TestCalibrateThresholdFlag:
    def test_defaults_to_none(self):
        parser = _base_parser("test")
        args = parser.parse_args(["dummy.csv"])
        assert args.calibrate_threshold is None

    def test_accepts_metric_name(self):
        parser = _base_parser("test")
        args = parser.parse_args(["dummy.csv", "--calibrate-threshold", "f1"])
        assert args.calibrate_threshold == "f1"


# --- calibrate_threshold in train_and_evaluate ---


class TestCalibrateThreshold:
    @patch("automl_model_training.train.TabularPredictor")
    def test_calibrates_threshold_for_binary(self, mock_cls: MagicMock, tmp_path: Path):
        mock_pred = _make_mock_predictor()
        mock_pred.calibrate_decision_threshold.return_value = 0.35
        mock_cls.return_value = mock_pred

        train_and_evaluate(
            train_raw=pd.DataFrame({"feat_a": [1, 2], "target": [0, 1]}),
            test_raw=pd.DataFrame({"feat_a": [3], "target": [1]}),
            label="target",
            problem_type="binary",
            eval_metric="f1",
            time_limit=None,
            preset="best",
            output_dir=str(tmp_path),
            calibrate_threshold="f1",
        )

        mock_pred.calibrate_decision_threshold.assert_called_once_with(metric="f1")
        mock_pred.set_decision_threshold.assert_called_once_with(0.35)

        info = json.loads((tmp_path / "model_info.json").read_text())
        assert info["decision_threshold"] == 0.35
        assert info["calibrated_for_metric"] == "f1"

    @patch("automl_model_training.train.TabularPredictor")
    def test_skips_calibration_for_regression(self, mock_cls: MagicMock, tmp_path: Path):
        mock_pred = _make_mock_predictor(problem_type="regression")
        mock_pred.predict.side_effect = lambda data, **kw: pd.Series(
            np.full(len(data), 1.5), index=data.index
        )
        mock_cls.return_value = mock_pred

        train_and_evaluate(
            train_raw=pd.DataFrame({"feat_a": [1, 2], "target": [1.0, 2.0]}),
            test_raw=pd.DataFrame({"feat_a": [3], "target": [3.0]}),
            label="target",
            problem_type="regression",
            eval_metric="root_mean_squared_error",
            time_limit=None,
            preset="best",
            output_dir=str(tmp_path),
            calibrate_threshold="f1",
        )

        mock_pred.calibrate_decision_threshold.assert_not_called()

    @patch("automl_model_training.train.TabularPredictor")
    def test_skips_calibration_when_none(self, mock_cls: MagicMock, tmp_path: Path):
        mock_pred = _make_mock_predictor()
        mock_cls.return_value = mock_pred

        train_and_evaluate(
            train_raw=pd.DataFrame({"feat_a": [1, 2], "target": [0, 1]}),
            test_raw=pd.DataFrame({"feat_a": [3], "target": [1]}),
            label="target",
            problem_type="binary",
            eval_metric="f1",
            time_limit=None,
            preset="best",
            output_dir=str(tmp_path),
        )

        mock_pred.calibrate_decision_threshold.assert_not_called()


# --- best_model_before_refit in model_info.json ---


class TestRefitBestModel:
    @patch("automl_model_training.train.TabularPredictor")
    def test_model_info_contains_refit_info(self, mock_cls: MagicMock, tmp_path: Path):
        mock_pred = _make_mock_predictor()
        mock_cls.return_value = mock_pred

        train_and_evaluate(
            train_raw=pd.DataFrame({"feat_a": [1, 2], "target": [0, 1]}),
            test_raw=pd.DataFrame({"feat_a": [3], "target": [1]}),
            label="target",
            problem_type="binary",
            eval_metric="f1",
            time_limit=None,
            preset="best",
            output_dir=str(tmp_path),
        )

        info = json.loads((tmp_path / "model_info.json").read_text())
        assert info["best_model_before_refit"] == "LightGBM"
        mock_pred.fit.assert_called_once()
        call_kwargs = mock_pred.fit.call_args.kwargs
        assert call_kwargs.get("refit_full") is True
        assert call_kwargs.get("set_best_to_refit_full") is True


# --- decision_threshold in predict_and_save ---


class TestDecisionThreshold:
    def _make_predictor(self) -> MagicMock:
        pred = MagicMock()
        pred.label = "target"
        pred.problem_type = "binary"
        pred.model_best = "LightGBM"
        pred.predict.return_value = pd.Series([0, 1, 0])
        pred.evaluate.return_value = {"f1": 0.85}
        proba = pd.DataFrame({0: [0.8, 0.3, 0.6], 1: [0.2, 0.7, 0.4]})
        pred.predict_proba.return_value = proba
        return pred

    def test_passes_threshold_to_predict(self, tmp_path: Path):
        pred = self._make_predictor()
        data = pd.DataFrame({"feat_a": [1, 2, 3], "target": [0, 1, 0]})

        predict_and_save(pred, data, str(tmp_path), decision_threshold=0.3)

        pred.predict.assert_called_once()
        call_kwargs = pred.predict.call_args[1]
        assert call_kwargs["decision_threshold"] == 0.3

    def test_no_threshold_kwarg_when_none(self, tmp_path: Path):
        pred = self._make_predictor()
        data = pd.DataFrame({"feat_a": [1, 2, 3], "target": [0, 1, 0]})

        predict_and_save(pred, data, str(tmp_path))

        call_kwargs = pred.predict.call_args[1]
        assert "decision_threshold" not in call_kwargs

    def test_ignores_threshold_for_regression(self, tmp_path: Path):
        pred = MagicMock()
        pred.label = "target"
        pred.problem_type = "regression"
        pred.model_best = "LightGBM"
        pred.predict.return_value = pd.Series([1.0, 2.0, 3.0])
        pred.evaluate.return_value = {"rmse": 0.5}
        data = pd.DataFrame({"feat_a": [1, 2, 3], "target": [1.1, 2.1, 3.1]})

        predict_and_save(pred, data, str(tmp_path), decision_threshold=0.3)

        call_kwargs = pred.predict.call_args[1]
        assert "decision_threshold" not in call_kwargs
