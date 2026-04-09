"""Tests for model comparison."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from automl_model_training.compare import compare_runs, load_run_summary, save_comparison


def _create_run(path: Path, test_score: float = 0.90, n_models: int = 5) -> str:
    """Create a minimal fake training run directory."""
    path.mkdir(parents=True, exist_ok=True)

    model_info = {
        "problem_type": "binary",
        "eval_metric": "f1",
        "best_model": "LightGBM",
        "features": ["feat_a", "feat_b", "feat_c"],
    }
    (path / "model_info.json").write_text(json.dumps(model_info))

    analysis = {
        "findings": ["finding 1"],
        "recommendations": ["rec 1", "rec 2"],
    }
    (path / "analysis.json").write_text(json.dumps(analysis))

    lb = pd.DataFrame(
        {
            "model": [f"model_{i}" for i in range(n_models)],
            "score_test": [test_score - i * 0.01 for i in range(n_models)],
            "fit_time": [10.0] * n_models,
        }
    )
    lb.to_csv(path / "leaderboard_test.csv", index=False)

    lb_train = pd.DataFrame(
        {
            "model": [f"model_{i}" for i in range(n_models)],
            "score_val": [test_score - i * 0.01 for i in range(n_models)],
            "fit_time": [10.0] * n_models,
        }
    )
    lb_train.to_csv(path / "leaderboard.csv", index=False)

    imp = pd.DataFrame(
        {"importance": [0.5, 0.3, 0.1]},
        index=["feat_a", "feat_b", "feat_c"],
    )
    imp.to_csv(path / "feature_importance.csv")

    return str(path)


class TestLoadRunSummary:
    def test_loads_all_fields(self, tmp_path: Path):
        run_dir = _create_run(tmp_path / "run1")
        summary = load_run_summary(run_dir)

        assert summary["problem_type"] == "binary"
        assert summary["best_model"] == "LightGBM"
        assert summary["n_features"] == 3
        assert summary["best_test_score"] == 0.90
        assert summary["n_models"] == 5
        assert summary["total_fit_time"] == 50.0
        assert summary["top_5_features"] == ["feat_a", "feat_b", "feat_c"]

    def test_handles_empty_directory(self, tmp_path: Path):
        run_dir = tmp_path / "empty"
        run_dir.mkdir()
        summary = load_run_summary(str(run_dir))

        assert summary["run_dir"] == str(run_dir)
        assert "best_model" not in summary

    def test_loads_cv_summary(self, tmp_path: Path):
        run_dir = _create_run(tmp_path / "run_cv")
        cv = {
            "n_folds": 5,
            "aggregate_scores": {"f1": {"mean": 0.88, "std": 0.02}},
        }
        (tmp_path / "run_cv" / "cv_summary.json").write_text(json.dumps(cv))

        summary = load_run_summary(run_dir)
        assert summary["cv_folds"] == 5
        assert summary["cv_f1_mean"] == 0.88
        assert summary["cv_f1_std"] == 0.02


class TestCompareRuns:
    def test_compares_two_runs(self, tmp_path: Path):
        run1 = _create_run(tmp_path / "run1", test_score=0.90)
        run2 = _create_run(tmp_path / "run2", test_score=0.85, n_models=3)

        df = compare_runs([run1, run2])

        assert len(df) == 2
        assert "best_test_score" in df.columns
        assert df.iloc[0]["best_test_score"] == 0.90
        assert df.iloc[1]["best_test_score"] == 0.85
        assert df.iloc[1]["n_models"] == 3

    def test_single_run(self, tmp_path: Path):
        run1 = _create_run(tmp_path / "run1")
        df = compare_runs([run1])
        assert len(df) == 1

    def test_column_ordering(self, tmp_path: Path):
        run1 = _create_run(tmp_path / "run1")
        df = compare_runs([run1])

        # Priority columns should come first
        assert df.columns[0] == "run_dir"
        assert "best_test_score" in df.columns[:8].tolist()


class TestSaveComparison:
    def test_saves_csv_and_json(self, tmp_path: Path):
        run1 = _create_run(tmp_path / "run1")
        run2 = _create_run(tmp_path / "run2")
        df = compare_runs([run1, run2])

        out = tmp_path / "comparison_output"
        out.mkdir()
        save_comparison(df, out)

        assert (out / "comparison.csv").exists()
        assert (out / "comparison.json").exists()

        loaded = pd.read_csv(out / "comparison.csv")
        assert len(loaded) == 2
