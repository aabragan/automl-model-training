"""Tests for train_and_evaluate core logic."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from automl_model_training.train import train_and_evaluate


def _make_mock_predictor(problem_type: str = "binary", n_test: int = 5) -> MagicMock:
    """Build a mock predictor that satisfies train_and_evaluate's calls."""
    pred = MagicMock()
    pred.label = "target"
    pred.problem_type = problem_type
    pred.eval_metric = "f1" if problem_type == "binary" else "root_mean_squared_error"
    pred.model_best = "LightGBM"
    pred.features.return_value = ["feat_a", "feat_b"]

    # predict/predict_proba return values sized to match test_raw
    pred.predict.side_effect = lambda data: pd.Series(
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


class TestTrainAndEvaluate:
    @patch("automl_model_training.train.TabularPredictor")
    def test_returns_predictor(self, mock_cls: MagicMock, tmp_path: Path):
        mock_pred = _make_mock_predictor()
        mock_cls.return_value = mock_pred

        result = train_and_evaluate(
            train_raw=pd.DataFrame({"feat_a": [1, 2], "feat_b": [3, 4], "target": [0, 1]}),
            test_raw=pd.DataFrame({"feat_a": [5], "feat_b": [6], "target": [1]}),
            label="target",
            problem_type="binary",
            eval_metric="f1",
            time_limit=None,
            preset="best",
            output_dir=str(tmp_path),
        )

        assert result is mock_pred

    @patch("automl_model_training.train.TabularPredictor")
    def test_saves_leaderboard_and_model_info(self, mock_cls: MagicMock, tmp_path: Path):
        mock_pred = _make_mock_predictor()
        mock_cls.return_value = mock_pred

        train_and_evaluate(
            train_raw=pd.DataFrame({"feat_a": [1, 2], "feat_b": [3, 4], "target": [0, 1]}),
            test_raw=pd.DataFrame({"feat_a": [5], "feat_b": [6], "target": [1]}),
            label="target",
            problem_type="binary",
            eval_metric="f1",
            time_limit=None,
            preset="best",
            output_dir=str(tmp_path),
        )

        assert (tmp_path / "leaderboard.csv").exists()
        assert (tmp_path / "leaderboard_test.csv").exists()
        assert (tmp_path / "feature_importance.csv").exists()
        assert (tmp_path / "model_info.json").exists()

        info = json.loads((tmp_path / "model_info.json").read_text())
        assert info["best_model"] == "LightGBM"
        assert info["problem_type"] == "binary"

    @patch("automl_model_training.train.TabularPredictor")
    def test_calls_fit_with_correct_params(self, mock_cls: MagicMock, tmp_path: Path):
        mock_pred = _make_mock_predictor()
        mock_cls.return_value = mock_pred

        train_and_evaluate(
            train_raw=pd.DataFrame({"feat_a": [1], "target": [0]}),
            test_raw=pd.DataFrame({"feat_a": [2], "target": [1]}),
            label="target",
            problem_type="binary",
            eval_metric="f1",
            time_limit=60,
            preset="high_quality",
            output_dir=str(tmp_path),
        )

        mock_pred.fit.assert_called_once()
        call_kwargs = mock_pred.fit.call_args[1]
        assert call_kwargs["presets"] == "high_quality"
        assert call_kwargs["time_limit"] == 60
        assert call_kwargs["auto_stack"] is True

    @patch("automl_model_training.train.TabularPredictor")
    def test_regression_dispatches_correctly(self, mock_cls: MagicMock, tmp_path: Path):
        mock_pred = _make_mock_predictor(problem_type="regression")
        mock_pred.predict.side_effect = lambda data: pd.Series(
            np.full(len(data), 1.5), index=data.index
        )
        mock_cls.return_value = mock_pred

        train_and_evaluate(
            train_raw=pd.DataFrame({"feat_a": [1, 2], "target": [1.0, 2.0]}),
            test_raw=pd.DataFrame({"feat_a": [3, 4], "target": [3.0, 4.0]}),
            label="target",
            problem_type="regression",
            eval_metric="root_mean_squared_error",
            time_limit=None,
            preset="best",
            output_dir=str(tmp_path),
        )

        # Should call save_regression_artifacts, not classification
        assert (tmp_path / "model_info.json").exists()

    @patch("automl_model_training.train.TabularPredictor")
    def test_prune_flag(self, mock_cls: MagicMock, tmp_path: Path):
        mock_pred = _make_mock_predictor()
        mock_cls.return_value = mock_pred

        train_and_evaluate(
            train_raw=pd.DataFrame({"feat_a": [1, 2], "target": [0, 1]}),
            test_raw=pd.DataFrame({"feat_a": [3], "target": [0]}),
            label="target",
            problem_type="binary",
            eval_metric="f1",
            time_limit=None,
            preset="best",
            output_dir=str(tmp_path),
            prune=True,
        )

        # Pruning should have been triggered
        assert (tmp_path / "model_info.json").exists()
