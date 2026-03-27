"""Tests for ensemble pruning logic."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest

from automl_model_training.evaluate.prune import (
    _collect_dependencies,
    analyze_ensemble,
    prune_models,
    recommend_pruning,
    save_pruning_report,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def leaderboard() -> pd.DataFrame:
    """Fake leaderboard with 5 models."""
    return pd.DataFrame(
        {
            "model": [
                "WeightedEnsemble_L2",
                "LightGBM",
                "CatBoost",
                "RandomForest",
                "KNN",
            ],
            "score_val": [0.95, 0.93, 0.90, 0.80, 0.60],
            "score_test": [0.94, 0.92, 0.89, 0.78, 0.58],
            "fit_time": [10.0, 5.0, 6.0, 3.0, 1.0],
            "pred_time_val": [0.5, 0.1, 0.1, 0.2, 0.05],
            "can_infer": [True, True, True, True, True],
        }
    )


@pytest.fixture()
def mock_predictor(leaderboard: pd.DataFrame) -> MagicMock:
    pred = MagicMock()
    pred.model_best = "WeightedEnsemble_L2"
    pred.leaderboard.return_value = leaderboard
    pred.info.return_value = {
        "model_info": {
            "WeightedEnsemble_L2": {
                "children": ["LightGBM", "CatBoost"],
            },
            "LightGBM": {},
            "CatBoost": {},
            "RandomForest": {},
            "KNN": {},
        }
    }
    return pred


# ---------------------------------------------------------------------------
# analyze_ensemble
# ---------------------------------------------------------------------------


class TestAnalyzeEnsemble:
    def test_marks_best_model(self, mock_predictor: MagicMock):
        df = analyze_ensemble(mock_predictor, test_data=None)
        best_rows = df.loc[df["is_best"]]
        assert len(best_rows) == 1
        assert best_rows.iloc[0]["model"] == "WeightedEnsemble_L2"

    def test_marks_contributing_models(self, mock_predictor: MagicMock):
        df = analyze_ensemble(mock_predictor, test_data=None)
        contributing = set(df.loc[df["contributes_to_best"], "model"])
        assert "WeightedEnsemble_L2" in contributing
        assert "LightGBM" in contributing
        assert "CatBoost" in contributing
        assert "RandomForest" not in contributing
        assert "KNN" not in contributing


# ---------------------------------------------------------------------------
# recommend_pruning
# ---------------------------------------------------------------------------


class TestRecommendPruning:
    def test_prunes_low_scoring_non_contributing(self, mock_predictor: MagicMock):
        df = analyze_ensemble(mock_predictor, test_data=None)
        to_prune = recommend_pruning(df, score_threshold_pct=5.0)
        # RandomForest (0.80 vs 0.95 = 15.8% gap) and KNN (0.60 vs 0.95 = 36.8% gap)
        assert "RandomForest" in to_prune
        assert "KNN" in to_prune
        # Best and its deps should never be pruned
        assert "WeightedEnsemble_L2" not in to_prune
        assert "LightGBM" not in to_prune
        assert "CatBoost" not in to_prune

    def test_high_threshold_prunes_nothing(self, mock_predictor: MagicMock):
        df = analyze_ensemble(mock_predictor, test_data=None)
        to_prune = recommend_pruning(df, score_threshold_pct=99.0)
        assert to_prune == []

    def test_empty_dataframe(self):
        df = pd.DataFrame(columns=["model", "score_val", "is_best", "contributes_to_best"])
        assert recommend_pruning(df) == []


# ---------------------------------------------------------------------------
# prune_models
# ---------------------------------------------------------------------------


class TestPruneModels:
    def test_dry_run_does_not_delete(self, mock_predictor: MagicMock):
        result = prune_models(mock_predictor, ["KNN"], dry_run=True)
        assert result == []
        mock_predictor.delete_models.assert_not_called()

    def test_prune_calls_delete(self, mock_predictor: MagicMock):
        result = prune_models(mock_predictor, ["KNN", "RandomForest"], dry_run=False)
        assert result == ["KNN", "RandomForest"]
        mock_predictor.delete_models.assert_called_once_with(
            models_to_delete=["KNN", "RandomForest"],
            allow_delete_cascade=True,
            delete_from_disk=True,
            dry_run=False,
        )

    def test_empty_list_skips(self, mock_predictor: MagicMock):
        result = prune_models(mock_predictor, [], dry_run=False)
        assert result == []
        mock_predictor.delete_models.assert_not_called()


# ---------------------------------------------------------------------------
# save_pruning_report
# ---------------------------------------------------------------------------


class TestSavePruningReport:
    def test_saves_files(self, mock_predictor: MagicMock, tmp_path: Path):
        df = analyze_ensemble(mock_predictor, test_data=None)
        save_pruning_report(df, ["KNN"], tmp_path)

        assert (tmp_path / "ensemble_analysis.csv").exists()
        assert (tmp_path / "pruning_report.json").exists()

        import json

        report = json.loads((tmp_path / "pruning_report.json").read_text())
        assert report["total_models"] == 5
        assert report["pruned_count"] == 1
        assert report["pruned_models"] == ["KNN"]
        assert report["remaining_count"] == 4


# ---------------------------------------------------------------------------
# _collect_dependencies
# ---------------------------------------------------------------------------


class TestCollectDependencies:
    def test_recursive_deps(self):
        model_info = {
            "Ensemble": {"children": ["StackerA", "StackerB"]},
            "StackerA": {"children": ["LightGBM"]},
            "StackerB": {},
            "LightGBM": {},
        }
        result: set[str] = set()
        _collect_dependencies("Ensemble", model_info, result)
        assert result == {"Ensemble", "StackerA", "StackerB", "LightGBM"}

    def test_no_deps(self):
        model_info = {"LightGBM": {}}
        result: set[str] = set()
        _collect_dependencies("LightGBM", model_info, result)
        assert result == {"LightGBM"}
