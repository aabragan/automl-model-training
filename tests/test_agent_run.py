"""Tests for agent run_agent loop and helpers."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd

from automl_model_training.agent import _tabarena_available, run_agent


class TestTabarenaAvailable:
    @patch.dict("sys.modules", {"tabpfn": MagicMock()})
    def test_returns_true_when_installed(self):
        assert _tabarena_available() is True

    @patch.dict("sys.modules", {"tabpfn": None})
    def test_returns_false_when_missing(self):
        # Importing None raises ImportError
        assert _tabarena_available() is False


class TestRunAgent:
    @patch("automl_model_training.agent.train_and_evaluate")
    @patch("automl_model_training.agent.load_and_prepare")
    @patch("automl_model_training.agent._profile_and_get_drops")
    @patch("automl_model_training.agent.compare_experiments")
    @patch("automl_model_training.agent.record_experiment")
    @patch("automl_model_training.agent._extract_metric")
    def test_stops_when_target_reached(
        self,
        mock_extract: MagicMock,
        mock_record: MagicMock,
        mock_compare: MagicMock,
        mock_profile: MagicMock,
        mock_prepare: MagicMock,
        mock_train: MagicMock,
        tmp_path: Path,
    ):
        mock_profile.return_value = []
        mock_prepare.return_value = (
            pd.DataFrame({"feat_a": [1], "target": [0]}),
            pd.DataFrame({"feat_a": [2], "target": [1]}),
            None, None, [],
        )
        mock_train.return_value = MagicMock()
        mock_extract.return_value = 0.95  # above target
        mock_compare.return_value = pd.DataFrame()
        mock_record.return_value = {}

        result = run_agent(
            csv_path="dummy.csv",
            label="target",
            problem_type="binary",
            eval_metric="f1",
            target_metric="f1",
            target_value=0.90,
            max_iterations=5,
            output_dir=str(tmp_path),
        )

        assert result["target_met"] is True
        # Should stop after first iteration since target was met
        assert result["iterations"] == 1
        assert result["best_score"] == 0.95

    @patch("automl_model_training.agent.train_and_evaluate")
    @patch("automl_model_training.agent.load_and_prepare")
    @patch("automl_model_training.agent._profile_and_get_drops")
    @patch("automl_model_training.agent.compare_experiments")
    @patch("automl_model_training.agent.record_experiment")
    @patch("automl_model_training.agent._extract_metric")
    @patch("automl_model_training.agent._read_analysis")
    @patch("automl_model_training.agent._read_feature_importance")
    def test_runs_all_iterations_when_target_not_met(
        self,
        mock_importance: MagicMock,
        mock_analysis: MagicMock,
        mock_extract: MagicMock,
        mock_record: MagicMock,
        mock_compare: MagicMock,
        mock_profile: MagicMock,
        mock_prepare: MagicMock,
        mock_train: MagicMock,
        tmp_path: Path,
    ):
        mock_profile.return_value = []
        mock_prepare.return_value = (
            pd.DataFrame({"feat_a": [1], "target": [0]}),
            pd.DataFrame({"feat_a": [2], "target": [1]}),
            None, None, [],
        )
        mock_train.return_value = MagicMock()
        mock_extract.return_value = 0.50  # below target
        mock_compare.return_value = pd.DataFrame()
        mock_record.return_value = {}
        mock_analysis.return_value = {"findings": [], "recommendations": []}
        mock_importance.return_value = []

        result = run_agent(
            csv_path="dummy.csv",
            label="target",
            problem_type="binary",
            eval_metric="f1",
            target_metric="f1",
            target_value=0.99,
            max_iterations=3,
            output_dir=str(tmp_path),
        )

        assert result["target_met"] is False
        assert result["iterations"] == 3

    @patch("automl_model_training.agent.train_and_evaluate")
    @patch("automl_model_training.agent.load_and_prepare")
    @patch("automl_model_training.agent._profile_and_get_drops")
    @patch("automl_model_training.agent.compare_experiments")
    @patch("automl_model_training.agent.record_experiment")
    @patch("automl_model_training.agent._extract_metric")
    def test_regression_lower_is_better(
        self,
        mock_extract: MagicMock,
        mock_record: MagicMock,
        mock_compare: MagicMock,
        mock_profile: MagicMock,
        mock_prepare: MagicMock,
        mock_train: MagicMock,
        tmp_path: Path,
    ):
        mock_profile.return_value = []
        mock_prepare.return_value = (
            pd.DataFrame({"feat_a": [1], "target": [1.0]}),
            pd.DataFrame({"feat_a": [2], "target": [2.0]}),
            None, None, [],
        )
        mock_train.return_value = MagicMock()
        mock_extract.return_value = 3.0  # RMSE of 3.0, target is 5.0
        mock_compare.return_value = pd.DataFrame()
        mock_record.return_value = {}

        result = run_agent(
            csv_path="dummy.csv",
            label="target",
            problem_type="regression",
            eval_metric="root_mean_squared_error",
            target_metric="root_mean_squared_error",
            target_value=5.0,
            max_iterations=2,
            output_dir=str(tmp_path),
            higher_is_better=False,
        )

        assert result["target_met"] is True
        assert result["best_score"] == 3.0
