"""Tests for agent run_agent loop and helpers."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd

from automl_model_training.agent import (
    PRESETS_TO_TRY,
    PRESETS_WITH_EXTREME,
    PRESETS_WITH_NONCOMMERCIAL,
    _extreme_available,
    _noncommercial_available,
    _select_presets,
    run_agent,
)


class TestExtremeAvailable:
    @patch.dict("sys.modules", {"tabicl": MagicMock(), "tabdpt": MagicMock()})
    def test_returns_true_when_installed(self):
        assert _extreme_available() is True

    @patch.dict("sys.modules", {"tabicl": None, "tabdpt": None})
    def test_returns_false_when_missing(self):
        # Importing None raises ImportError
        assert _extreme_available() is False

    @patch.dict("sys.modules", {"tabicl": MagicMock(), "tabdpt": None})
    def test_returns_false_when_partially_installed(self):
        assert _extreme_available() is False


class TestNoncommercialAvailable:
    @patch.dict(
        "sys.modules",
        {"tabicl": MagicMock(), "tabdpt": MagicMock(), "tabpfn": MagicMock()},
    )
    def test_returns_true_when_all_installed(self):
        assert _noncommercial_available() is True

    @patch.dict(
        "sys.modules",
        {"tabicl": MagicMock(), "tabdpt": MagicMock(), "tabpfn": None},
    )
    def test_returns_false_without_tabpfn(self):
        assert _noncommercial_available() is False

    @patch.dict(
        "sys.modules",
        {"tabicl": None, "tabdpt": None, "tabpfn": MagicMock()},
    )
    def test_returns_false_without_extreme_models(self):
        # TabPFN alone is not enough — the noncommercial preset builds on extreme
        assert _noncommercial_available() is False


class TestSelectPresets:
    @patch.dict(
        "sys.modules",
        {"tabicl": MagicMock(), "tabdpt": MagicMock(), "tabpfn": MagicMock()},
    )
    def test_noncommercial_leads_when_everything_installed(self):
        presets = _select_presets()
        assert presets == PRESETS_WITH_NONCOMMERCIAL
        assert presets[0] == "noncommercial"
        assert presets[1] == "extreme"

    @patch.dict(
        "sys.modules",
        {"tabicl": MagicMock(), "tabdpt": MagicMock(), "tabpfn": None},
    )
    def test_extreme_leads_without_tabpfn(self):
        presets = _select_presets()
        assert presets == PRESETS_WITH_EXTREME
        assert presets[0] == "extreme"

    @patch.dict("sys.modules", {"tabicl": None, "tabdpt": None, "tabpfn": None})
    def test_cpu_presets_without_extras(self):
        assert _select_presets() == PRESETS_TO_TRY


class TestRunAgent:
    @patch("automl_model_training.agent.train_and_evaluate")
    @patch("automl_model_training.agent.load_and_prepare")
    @patch("automl_model_training.agent._profile_and_get_drops")
    @patch("automl_model_training.agent.compare_experiments")
    @patch("automl_model_training.agent.record_experiment")
    @patch("automl_model_training.agent.extract_metric")
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
            None,
            None,
            [],
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
    @patch("automl_model_training.agent.extract_metric")
    @patch("automl_model_training.agent.read_analysis")
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
            None,
            None,
            [],
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
    @patch("automl_model_training.agent.extract_metric")
    def test_regression_signed_rmse_convention(
        self,
        mock_extract: MagicMock,
        mock_record: MagicMock,
        mock_compare: MagicMock,
        mock_profile: MagicMock,
        mock_prepare: MagicMock,
        mock_train: MagicMock,
        tmp_path: Path,
    ):
        """RMSE scores arrive negated (AutoGluon higher-is-better convention).

        An RMSE of 3.0 is score -3.0; an RMSE goal of 5.0 is target -5.0.
        -3.0 >= -5.0, so the target is met.
        """
        mock_profile.return_value = []
        mock_prepare.return_value = (
            pd.DataFrame({"feat_a": [1], "target": [1.0]}),
            pd.DataFrame({"feat_a": [2], "target": [2.0]}),
            None,
            None,
            [],
        )
        mock_train.return_value = MagicMock()
        mock_extract.return_value = -3.0  # RMSE of 3.0, signed
        mock_compare.return_value = pd.DataFrame()
        mock_record.return_value = {}

        result = run_agent(
            csv_path="dummy.csv",
            label="target",
            problem_type="regression",
            eval_metric="root_mean_squared_error",
            target_metric="root_mean_squared_error",
            target_value=-5.0,
            max_iterations=2,
            output_dir=str(tmp_path),
        )

        assert result["target_met"] is True
        assert result["best_score"] == -3.0

    @patch("automl_model_training.agent.train_and_evaluate")
    @patch("automl_model_training.agent.load_and_prepare")
    @patch("automl_model_training.agent._profile_and_get_drops")
    @patch("automl_model_training.agent.compare_experiments")
    @patch("automl_model_training.agent.record_experiment")
    @patch("automl_model_training.agent.extract_metric")
    @patch("automl_model_training.agent.read_analysis")
    @patch("automl_model_training.agent._read_feature_importance")
    def test_handles_missing_score(
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
            None,
            None,
            [],
        )
        mock_train.return_value = MagicMock()
        mock_extract.return_value = None  # metric can't be extracted
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
            target_value=0.90,
            max_iterations=2,
            output_dir=str(tmp_path),
        )

        assert result["best_score"] is None
        assert result["target_met"] is False
        assert result["iterations"] == 2

    @patch("automl_model_training.agent.train_and_evaluate")
    @patch("automl_model_training.agent.load_and_prepare")
    @patch("automl_model_training.agent._profile_and_get_drops")
    @patch("automl_model_training.agent.compare_experiments")
    @patch("automl_model_training.agent.record_experiment")
    @patch("automl_model_training.agent.extract_metric")
    @patch("automl_model_training.agent.read_analysis")
    @patch("automl_model_training.agent._read_feature_importance")
    def test_adds_low_importance_drops_between_iterations(
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
            None,
            None,
            [],
        )
        mock_train.return_value = MagicMock()
        mock_extract.return_value = 0.50  # below target — keep iterating
        # Non-empty comparison exercises the summary logging path too
        mock_compare.return_value = pd.DataFrame([{"run_id": "r1", "metric_f1": 0.5}])
        mock_record.return_value = {}
        mock_analysis.return_value = {"findings": [], "recommendations": []}
        mock_importance.return_value = ["feat_low"]

        run_agent(
            csv_path="dummy.csv",
            label="target",
            problem_type="binary",
            eval_metric="f1",
            target_metric="f1",
            target_value=0.99,
            max_iterations=2,
            output_dir=str(tmp_path),
        )

        # Second iteration should include the low-importance feature in the drop list
        second_call_kwargs = mock_prepare.call_args_list[1][1]
        assert "feat_low" in second_call_kwargs["features_to_drop"]
