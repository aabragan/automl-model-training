"""Tests for cross-validation."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from automl_model_training.train import cross_validate


class TestCrossValidate:
    def _make_data(self, n: int = 100) -> pd.DataFrame:
        rng = np.random.RandomState(42)
        return pd.DataFrame(
            {
                "feat_a": rng.randn(n),
                "feat_b": rng.randn(n),
                "target": rng.choice([0, 1], n, p=[0.7, 0.3]),
            }
        )

    @patch("automl_model_training.train.TabularPredictor")
    def test_returns_summary_with_folds(self, mock_cls: MagicMock, tmp_path: Path):
        mock_pred = MagicMock()
        mock_pred.problem_type = "binary"
        mock_pred.model_best = "LightGBM"
        mock_pred.evaluate.return_value = {"f1": 0.85, "accuracy": 0.90}
        mock_cls.return_value = mock_pred

        data = self._make_data()
        summary = cross_validate(
            data=data,
            label="target",
            n_folds=3,
            problem_type="binary",
            eval_metric="f1",
            time_limit=None,
            preset="best",
            output_dir=str(tmp_path),
            random_state=42,
        )

        assert summary["n_folds"] == 3
        assert summary["total_rows"] == 100
        assert len(summary["folds"]) == 3
        assert "aggregate_scores" in summary
        assert "f1" in summary["aggregate_scores"]
        assert "mean" in summary["aggregate_scores"]["f1"]
        assert "std" in summary["aggregate_scores"]["f1"]

    @patch("automl_model_training.train.TabularPredictor")
    def test_saves_cv_summary_json(self, mock_cls: MagicMock, tmp_path: Path):
        mock_pred = MagicMock()
        mock_pred.problem_type = "binary"
        mock_pred.model_best = "LightGBM"
        mock_pred.evaluate.return_value = {"f1": 0.80}
        mock_cls.return_value = mock_pred

        data = self._make_data()
        cross_validate(
            data=data,
            label="target",
            n_folds=2,
            problem_type="binary",
            eval_metric="f1",
            time_limit=None,
            preset="best",
            output_dir=str(tmp_path),
            random_state=42,
        )

        cv_path = tmp_path / "cv_summary.json"
        assert cv_path.exists()
        saved = json.loads(cv_path.read_text())
        assert saved["n_folds"] == 2
        assert len(saved["folds"]) == 2

    @patch("automl_model_training.train.TabularPredictor")
    def test_creates_fold_directories(self, mock_cls: MagicMock, tmp_path: Path):
        mock_pred = MagicMock()
        mock_pred.problem_type = "regression"
        mock_pred.model_best = "CatBoost"
        mock_pred.evaluate.return_value = {"root_mean_squared_error": -2.5}
        mock_cls.return_value = mock_pred

        rng = np.random.RandomState(42)
        data = pd.DataFrame(
            {
                "feat_a": rng.randn(60),
                "target": rng.randn(60),
            }
        )

        cross_validate(
            data=data,
            label="target",
            n_folds=3,
            problem_type="regression",
            eval_metric="root_mean_squared_error",
            time_limit=None,
            preset="best",
            output_dir=str(tmp_path),
            random_state=42,
        )

        assert (tmp_path / "cv_fold_1").is_dir()
        assert (tmp_path / "cv_fold_2").is_dir()
        assert (tmp_path / "cv_fold_3").is_dir()

    @patch("automl_model_training.train.TabularPredictor")
    def test_aggregate_scores_are_correct(self, mock_cls: MagicMock, tmp_path: Path):
        # Return different scores per fold to verify aggregation
        scores = [{"f1": 0.80}, {"f1": 0.90}]
        mock_pred = MagicMock()
        mock_pred.problem_type = "binary"
        mock_pred.model_best = "LightGBM"
        mock_pred.evaluate.side_effect = scores
        mock_cls.return_value = mock_pred

        data = self._make_data(50)
        summary = cross_validate(
            data=data,
            label="target",
            n_folds=2,
            problem_type="binary",
            eval_metric="f1",
            time_limit=None,
            preset="best",
            output_dir=str(tmp_path),
            random_state=42,
        )

        agg = summary["aggregate_scores"]["f1"]
        assert agg["mean"] == 0.85  # (0.80 + 0.90) / 2
        assert agg["std"] > 0  # should have non-zero std
