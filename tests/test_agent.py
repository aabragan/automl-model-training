"""Tests for autonomous agent helper functions."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from automl_model_training.agent import (
    _decide_next_action,
    _extract_metric,
    _read_analysis,
    _read_feature_importance,
)


class TestReadAnalysis:
    def test_reads_valid_json(self, tmp_path: Path):
        data = {"findings": ["overfitting"], "recommendations": ["drop features"]}
        (tmp_path / "analysis.json").write_text(json.dumps(data))

        result = _read_analysis(str(tmp_path))
        assert result["findings"] == ["overfitting"]

    def test_returns_empty_dict_when_missing(self, tmp_path: Path):
        result = _read_analysis(str(tmp_path))
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
    def test_extracts_from_leaderboard(self, tmp_path: Path):
        lb = pd.DataFrame({"model": ["Best"], "score_test": [0.92]})
        lb.to_csv(tmp_path / "leaderboard_test.csv", index=False)

        score = _extract_metric(str(tmp_path), "f1")
        assert score == 0.92

    def test_returns_none_when_no_files(self, tmp_path: Path):
        score = _extract_metric(str(tmp_path), "f1")
        assert score is None

    def test_returns_absolute_value(self, tmp_path: Path):
        lb = pd.DataFrame({"model": ["Best"], "score_test": [-5.3]})
        lb.to_csv(tmp_path / "leaderboard_test.csv", index=False)

        score = _extract_metric(str(tmp_path), "rmse")
        assert score == 5.3


class TestDecideNextAction:
    def test_detects_overfitting_and_switches_preset(self):
        analysis = {
            "findings": ["Overfitting detected: val=0.95, test=0.80"],
            "recommendations": [],
        }
        action = _decide_next_action(analysis, 1, [], "best_quality")
        assert action["preset"] == "high_quality"
        assert "overfitting" in action["reason"].lower()

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
