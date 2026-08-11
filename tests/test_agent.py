"""Tests for autonomous agent helper functions."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd

from automl_model_training.agent import (
    _decide_next_action,
    _profile_and_get_drops,
    _read_feature_importance,
)
from automl_model_training.run_artifacts import extract_metric, read_analysis


class TestReadAnalysis:
    def test_reads_valid_json(self, tmp_path: Path):
        data = {"findings": ["overfitting"], "recommendations": ["drop features"]}
        (tmp_path / "analysis.json").write_text(json.dumps(data))

        result = read_analysis(str(tmp_path))
        assert result["findings"] == ["overfitting"]

    def test_returns_empty_dict_when_missing(self, tmp_path: Path):
        result = read_analysis(str(tmp_path))
        assert result == {}


class TestReadFeatureImportance:
    def test_returns_low_importance_features(self, tmp_path: Path):
        df = pd.DataFrame(
            {"importance": [0.5, 0.001, -0.01, 0.0]},
            index=["good", "low", "negative", "zero"],
        )
        df.to_csv(tmp_path / "feature_importance.csv")

        result = _read_feature_importance(str(tmp_path))
        assert "low" in result
        assert "negative" in result
        assert "zero" in result
        assert "good" not in result

    def test_returns_empty_when_no_file(self, tmp_path: Path):
        result = _read_feature_importance(str(tmp_path))
        assert result == []

    def test_returns_empty_when_no_importance_column(self, tmp_path: Path):
        df = pd.DataFrame({"other_col": [1, 2, 3]})
        df.to_csv(tmp_path / "feature_importance.csv")

        result = _read_feature_importance(str(tmp_path))
        assert result == []


class TestExtractMetric:
    def _write_analysis(self, tmp_path: Path, test_scores: dict, eval_metric: str) -> None:
        analysis = {
            "best_model": "WeightedEnsemble_L2_FULL",
            "eval_metric": eval_metric,
            "test_scores": test_scores,
            "score_convention": "higher_is_better",
            "findings": [],
            "recommendations": [],
        }
        (tmp_path / "analysis.json").write_text(json.dumps(analysis))

    def test_extracts_named_metric_from_analysis(self, tmp_path: Path):
        self._write_analysis(tmp_path, {"f1": 0.92, "accuracy": 0.95}, "f1")

        assert extract_metric(str(tmp_path), "f1") == 0.92
        assert extract_metric(str(tmp_path), "accuracy") == 0.95

    def test_returns_deployed_models_signed_score(self, tmp_path: Path, leaderboard_with_refit):
        """The score must be the deployed (refit _FULL) model's, sign intact.

        The old implementation read abs(leaderboard_test.iloc[0]) — a
        non-deployed model (the _FULL row has NaN score_val and sorts last)
        with the sign destroyed. test_scores comes from predictor.evaluate()
        on the deployed model, so RMSE stays negated per AutoGluon's
        higher-is-better convention.
        """
        _, test_lb = leaderboard_with_refit
        # A stale leaderboard exists but must NOT be consulted
        test_lb.to_csv(tmp_path / "leaderboard_test.csv", index=False)
        self._write_analysis(
            tmp_path,
            {"root_mean_squared_error": -5.3, "r2": 0.81},
            "root_mean_squared_error",
        )

        score = extract_metric(str(tmp_path), "root_mean_squared_error")
        assert score == -5.3  # signed, not abs
        assert extract_metric(str(tmp_path), "r2") == 0.81

    def test_generic_score_falls_back_to_eval_metric(self, tmp_path: Path):
        self._write_analysis(tmp_path, {"f1": 0.88, "accuracy": 0.91}, "f1")

        assert extract_metric(str(tmp_path), "score") == 0.88

    def test_returns_none_when_no_files(self, tmp_path: Path):
        assert extract_metric(str(tmp_path), "f1") is None

    def test_returns_none_for_leaderboard_only_runs(self, tmp_path: Path):
        """Runs predating test_scores persistence yield None, not a wrong number."""
        lb = pd.DataFrame({"model": ["Best"], "score_test": [0.92]})
        lb.to_csv(tmp_path / "leaderboard_test.csv", index=False)

        assert extract_metric(str(tmp_path), "f1") is None

    def test_returns_none_when_metric_absent_and_no_eval_metric_match(self, tmp_path: Path):
        analysis = {"test_scores": {"f1": 0.9}, "eval_metric": "accuracy"}
        (tmp_path / "analysis.json").write_text(json.dumps(analysis))

        assert extract_metric(str(tmp_path), "mcc") is None


class TestDecideNextAction:
    def test_detects_overfitting_and_switches_preset(self):
        analysis = {
            "findings": ["Overfitting detected: val=0.95, test=0.80"],
            "recommendations": [],
        }
        action = _decide_next_action(analysis, 1, [], "best_quality")
        assert action["preset"] == "high_quality"
        assert "overfitting" in action["reason"].lower()

    def test_deescalates_from_foundation_model_presets_on_overfitting(self):
        analysis = {
            "findings": ["Overfitting detected: val=0.95, test=0.80"],
            "recommendations": [],
        }
        for preset in ("extreme", "noncommercial"):
            action = _decide_next_action(analysis, 1, [], preset)
            assert action["preset"] == "high_quality"

    def test_cycles_preset_when_no_issues(self):
        analysis = {"findings": ["No major issues"], "recommendations": []}
        presets = ["best_quality", "best_v150", "high_quality"]

        action = _decide_next_action(analysis, 1, [], "best_quality", presets=presets)
        assert action["preset"] == "best_v150"

    def test_wraps_around_preset_list(self):
        analysis = {"findings": [], "recommendations": []}
        presets = ["best_quality", "high_quality"]

        action = _decide_next_action(analysis, 2, [], "high_quality", presets=presets)
        assert action["preset"] == "best_quality"

    def test_detects_drop_recommendation(self):
        analysis = {
            "findings": [],
            "recommendations": ["Drop feature X — near-zero importance"],
        }
        action = _decide_next_action(analysis, 1, [], "best_quality")
        assert "drop" in action["reason"].lower()

    def test_empty_analysis(self):
        action = _decide_next_action({}, 1, [], "best_quality")
        assert action["preset"] is not None
        assert action["reason"] != ""

    def test_unknown_preset_falls_back_to_first(self):
        analysis = {"findings": [], "recommendations": []}
        presets = ["best_quality", "high_quality"]

        action = _decide_next_action(analysis, 1, [], "not_a_preset", presets=presets)
        assert action["preset"] == "best_quality"


class TestProfileAndGetDrops:
    @patch("automl_model_training.agent.save_profile_report")
    @patch("automl_model_training.agent.recommend_features_to_drop")
    @patch("automl_model_training.agent.find_highly_correlated_pairs", return_value=[])
    @patch("automl_model_training.agent.compute_correlation_matrix")
    def test_returns_recommended_drops(
        self,
        mock_corr: MagicMock,
        mock_pairs: MagicMock,
        mock_recs: MagicMock,
        mock_save: MagicMock,
        tmp_path: Path,
    ):
        csv = tmp_path / "data.csv"
        pd.DataFrame({"feat_a": [1, 2], "feat_b": [2, 1], "target": [0, 1]}).to_csv(
            csv, index=False
        )
        mock_corr.return_value = pd.DataFrame()
        mock_recs.return_value = [{"feature": "feat_b"}]

        result = _profile_and_get_drops(str(csv), "target", str(tmp_path / "out"))

        assert result == ["feat_b"]
        mock_save.assert_called_once()

    @patch("automl_model_training.agent.save_profile_report")
    @patch("automl_model_training.agent.recommend_features_to_drop", return_value=[])
    @patch("automl_model_training.agent.find_highly_correlated_pairs", return_value=[])
    @patch("automl_model_training.agent.compute_correlation_matrix")
    def test_returns_empty_when_no_recommendations(
        self,
        mock_corr: MagicMock,
        mock_pairs: MagicMock,
        mock_recs: MagicMock,
        mock_save: MagicMock,
        tmp_path: Path,
    ):
        csv = tmp_path / "data.csv"
        pd.DataFrame({"feat_a": [1, 2], "target": [0, 1]}).to_csv(csv, index=False)
        mock_corr.return_value = pd.DataFrame()

        result = _profile_and_get_drops(str(csv), "target", str(tmp_path / "out"))

        assert result == []
