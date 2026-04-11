"""Tests for the LLM agent tool layer (tools.py)."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_csv(tmp_path: Path, problem_type: str = "binary") -> Path:
    rng = np.random.RandomState(0)
    n = 100
    if problem_type == "binary":
        df = pd.DataFrame(
            {
                "feat_a": rng.randn(n),
                "feat_b": rng.randn(n),
                "target": rng.choice([0, 1], n),
            }
        )
    else:
        x = rng.randn(n)
        df = pd.DataFrame({"feat_a": x, "feat_b": rng.randn(n), "target": x * 2.0})
    p = tmp_path / "data.csv"
    df.to_csv(p, index=False)
    return p


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
    pred.evaluate.return_value = {"f1": 0.88}
    pred.feature_importance.return_value = pd.DataFrame(
        {"importance": [0.5, 0.0001], "stddev": [0.01, 0.001]},
        index=["feat_a", "feat_b"],
    )
    return pred


def _write_artifacts(run_dir: Path, problem_type: str = "binary") -> None:
    """Write the minimal artifacts that tool_train reads after training."""
    model_info = {
        "problem_type": problem_type,
        "eval_metric": "f1",
        "label": "target",
        "features": ["feat_a", "feat_b"],
        "best_model": "LightGBM",
    }
    (run_dir / "model_info.json").write_text(json.dumps(model_info))

    lb = pd.DataFrame(
        {"model": ["LightGBM"], "score_val": [0.90], "score_test": [0.88], "fit_time": [10.0]}
    )
    lb.to_csv(run_dir / "leaderboard_test.csv", index=False)

    imp = pd.DataFrame(
        {"importance": [0.5, 0.0001], "stddev": [0.01, 0.001]},
        index=["feat_a", "feat_b"],
    )
    imp.index.name = None
    imp.to_csv(run_dir / "feature_importance.csv")

    analysis = {
        "best_model": "LightGBM",
        "problem_type": problem_type,
        "eval_metric": "f1",
        "findings": ["No major issues detected."],
        "recommendations": ["Results look solid."],
    }
    (run_dir / "analysis.json").write_text(json.dumps(analysis))


# ---------------------------------------------------------------------------
# tool_profile
# ---------------------------------------------------------------------------


class TestToolProfile:
    def test_returns_expected_keys(self, tmp_path: Path):
        from automl_model_training.tools import tool_profile

        csv = _make_csv(tmp_path)
        result = tool_profile(str(csv), "target")

        assert set(result.keys()) == {
            "shape",
            "label_distribution",
            "missing_pct",
            "drop_recommendations",
            "numeric_features",
            "categorical_features",
        }

    def test_shape_matches_csv(self, tmp_path: Path):
        from automl_model_training.tools import tool_profile

        csv = _make_csv(tmp_path)
        result = tool_profile(str(csv), "target")
        assert result["shape"] == [100, 3]

    def test_label_not_in_feature_lists(self, tmp_path: Path):
        from automl_model_training.tools import tool_profile

        csv = _make_csv(tmp_path)
        result = tool_profile(str(csv), "target")
        assert "target" not in result["numeric_features"]
        assert "target" not in result["categorical_features"]

    def test_missing_pct_keys_match_columns(self, tmp_path: Path):
        from automl_model_training.tools import tool_profile

        csv = _make_csv(tmp_path)
        result = tool_profile(str(csv), "target")
        df = pd.read_csv(csv)
        assert set(result["missing_pct"].keys()) == set(df.columns)

    def test_drop_recommendations_is_list(self, tmp_path: Path):
        from automl_model_training.tools import tool_profile

        csv = _make_csv(tmp_path)
        result = tool_profile(str(csv), "target")
        assert isinstance(result["drop_recommendations"], list)


# ---------------------------------------------------------------------------
# tool_train
# ---------------------------------------------------------------------------


class TestToolTrain:
    @patch("automl_model_training.tools.train_and_evaluate")
    @patch("automl_model_training.tools.load_and_prepare")
    @patch("automl_model_training.tools.make_run_dir")
    def test_returns_expected_keys(self, mock_run_dir, mock_load, mock_train, tmp_path: Path):
        from automl_model_training.tools import tool_train

        run_dir = tmp_path / "run"
        run_dir.mkdir()
        mock_run_dir.return_value = str(run_dir)

        df = pd.DataFrame({"feat_a": [1, 2], "feat_b": [3, 4], "target": [0, 1]})
        mock_load.return_value = (df, df, df, df, ["feat_a", "feat_b"])
        mock_train.return_value = _make_mock_predictor()
        _write_artifacts(run_dir)

        csv = _make_csv(tmp_path)
        result = tool_train(str(csv), "target")

        assert set(result.keys()) == {
            "run_dir",
            "score",
            "model_info",
            "analysis",
            "leaderboard",
            "low_importance_features",
            "negative_importance_features",
        }

    @patch("automl_model_training.tools.train_and_evaluate")
    @patch("automl_model_training.tools.load_and_prepare")
    @patch("automl_model_training.tools.make_run_dir")
    def test_score_extracted_from_leaderboard(self, mock_run_dir, mock_load, mock_train, tmp_path):
        from automl_model_training.tools import tool_train

        run_dir = tmp_path / "run"
        run_dir.mkdir()
        mock_run_dir.return_value = str(run_dir)

        df = pd.DataFrame({"feat_a": [1, 2], "target": [0, 1]})
        mock_load.return_value = (df, df, df, df, ["feat_a"])
        mock_train.return_value = _make_mock_predictor()
        _write_artifacts(run_dir)

        result = tool_train(str(_make_csv(tmp_path)), "target")
        assert result["score"] == pytest.approx(0.88)

    @patch("automl_model_training.tools.train_and_evaluate")
    @patch("automl_model_training.tools.load_and_prepare")
    @patch("automl_model_training.tools.make_run_dir")
    def test_low_importance_features_detected(self, mock_run_dir, mock_load, mock_train, tmp_path):
        from automl_model_training.tools import tool_train

        run_dir = tmp_path / "run"
        run_dir.mkdir()
        mock_run_dir.return_value = str(run_dir)

        df = pd.DataFrame({"feat_a": [1, 2], "target": [0, 1]})
        mock_load.return_value = (df, df, df, df, ["feat_a"])
        mock_train.return_value = _make_mock_predictor()
        _write_artifacts(run_dir)  # feat_b has importance=0.0001 (near-zero)

        result = tool_train(str(_make_csv(tmp_path)), "target")
        assert "feat_b" in result["low_importance_features"]

    @patch("automl_model_training.tools.train_and_evaluate")
    @patch("automl_model_training.tools.load_and_prepare")
    @patch("automl_model_training.tools.make_run_dir")
    def test_negative_importance_features_detected(
        self, mock_run_dir, mock_load, mock_train, tmp_path
    ):
        from automl_model_training.tools import tool_train

        run_dir = tmp_path / "run"
        run_dir.mkdir()
        mock_run_dir.return_value = str(run_dir)

        df = pd.DataFrame({"feat_a": [1, 2], "target": [0, 1]})
        mock_load.return_value = (df, df, df, df, ["feat_a"])
        mock_train.return_value = _make_mock_predictor()

        # Override importance with a negative value
        imp = pd.DataFrame(
            {"importance": [0.5, -0.1], "stddev": [0.01, 0.001]},
            index=["feat_a", "feat_b"],
        )
        imp.to_csv(run_dir / "feature_importance.csv")
        _write_artifacts(run_dir)
        # Re-write importance to override _write_artifacts
        imp.to_csv(run_dir / "feature_importance.csv")

        result = tool_train(str(_make_csv(tmp_path)), "target")
        assert "feat_b" in result["negative_importance_features"]

    @patch("automl_model_training.tools.cross_validate")
    @patch("automl_model_training.tools.train_and_evaluate")
    @patch("automl_model_training.tools.load_and_prepare")
    @patch("automl_model_training.tools.make_run_dir")
    def test_cv_folds_triggers_cross_validate(
        self, mock_run_dir, mock_load, mock_train, mock_cv, tmp_path
    ):
        from automl_model_training.tools import tool_train

        run_dir = tmp_path / "run"
        run_dir.mkdir()
        mock_run_dir.return_value = str(run_dir)

        df = pd.DataFrame({"feat_a": [1, 2], "target": [0, 1]})
        mock_load.return_value = (df, df, df, df, ["feat_a"])
        mock_train.return_value = _make_mock_predictor()
        mock_cv.return_value = {}
        _write_artifacts(run_dir)

        tool_train(str(_make_csv(tmp_path)), "target", cv_folds=5)
        mock_cv.assert_called_once()
        assert mock_cv.call_args[1]["n_folds"] == 5

    @patch("automl_model_training.tools.train_and_evaluate")
    @patch("automl_model_training.tools.load_and_prepare")
    @patch("automl_model_training.tools.make_run_dir")
    def test_leaderboard_included_in_result(self, mock_run_dir, mock_load, mock_train, tmp_path):
        from automl_model_training.tools import tool_train

        run_dir = tmp_path / "run"
        run_dir.mkdir()
        mock_run_dir.return_value = str(run_dir)

        df = pd.DataFrame({"feat_a": [1, 2], "target": [0, 1]})
        mock_load.return_value = (df, df, df, df, ["feat_a"])
        mock_train.return_value = _make_mock_predictor()
        _write_artifacts(run_dir)

        result = tool_train(str(_make_csv(tmp_path)), "target")
        assert isinstance(result["leaderboard"], list)
        assert len(result["leaderboard"]) >= 1
        assert "model" in result["leaderboard"][0]


# ---------------------------------------------------------------------------
# tool_predict
# ---------------------------------------------------------------------------


class TestToolPredict:
    @patch("automl_model_training.tools.predict_and_save")
    @patch("automl_model_training.tools.load_predictor")
    @patch("automl_model_training.tools.make_run_dir")
    def test_returns_expected_keys(self, mock_run_dir, mock_load_pred, mock_predict, tmp_path):
        from automl_model_training.tools import tool_predict

        run_dir = tmp_path / "pred_run"
        run_dir.mkdir()
        mock_run_dir.return_value = str(run_dir)

        pred = _make_mock_predictor()
        mock_load_pred.return_value = pred

        result_df = pd.DataFrame({"feat_a": [1], "target_predicted": [0]})
        mock_predict.return_value = result_df

        summary = {"problem_type": "binary", "num_rows": 1}
        (run_dir / "prediction_summary.json").write_text(json.dumps(summary))

        csv = _make_csv(tmp_path)
        result = tool_predict(str(csv), "fake/model/dir")

        assert set(result.keys()) == {"run_dir", "num_rows", "columns", "summary"}
        assert result["num_rows"] == 1

    @patch("automl_model_training.tools.predict_and_save")
    @patch("automl_model_training.tools.load_predictor")
    @patch("automl_model_training.tools.make_run_dir")
    def test_passes_confidence_and_threshold(
        self, mock_run_dir, mock_load_pred, mock_predict, tmp_path
    ):
        from automl_model_training.tools import tool_predict

        run_dir = tmp_path / "pred_run"
        run_dir.mkdir()
        mock_run_dir.return_value = str(run_dir)
        mock_load_pred.return_value = _make_mock_predictor()
        mock_predict.return_value = pd.DataFrame({"target_predicted": [0]})
        (run_dir / "prediction_summary.json").write_text("{}")

        tool_predict(
            str(_make_csv(tmp_path)),
            "fake/model",
            min_confidence=0.7,
            decision_threshold=0.3,
        )

        _, kwargs = mock_predict.call_args
        assert kwargs["min_confidence"] == 0.7
        assert kwargs["decision_threshold"] == 0.3


# ---------------------------------------------------------------------------
# tool_read_analysis
# ---------------------------------------------------------------------------


class TestToolReadAnalysis:
    def test_returns_analysis_dict(self, tmp_path: Path):
        from automl_model_training.tools import tool_read_analysis

        analysis = {"findings": ["ok"], "recommendations": ["keep going"]}
        (tmp_path / "analysis.json").write_text(json.dumps(analysis))

        result = tool_read_analysis(str(tmp_path))
        assert result["findings"] == ["ok"]

    def test_returns_empty_dict_when_missing(self, tmp_path: Path):
        from automl_model_training.tools import tool_read_analysis

        result = tool_read_analysis(str(tmp_path))
        assert result == {}


# ---------------------------------------------------------------------------
# tool_compare_runs
# ---------------------------------------------------------------------------


class TestToolCompareRuns:
    @patch("automl_model_training.tools.compare_experiments")
    def test_returns_list_of_dicts(self, mock_compare):
        from automl_model_training.tools import tool_compare_runs

        mock_compare.return_value = pd.DataFrame(
            [{"param_preset": "best", "metric_score": 0.88}]
        )
        result = tool_compare_runs()
        assert isinstance(result, list)
        assert result[0]["param_preset"] == "best"

    @patch("automl_model_training.tools.compare_experiments")
    def test_returns_empty_list_when_no_experiments(self, mock_compare):
        from automl_model_training.tools import tool_compare_runs

        mock_compare.return_value = pd.DataFrame()
        result = tool_compare_runs()
        assert result == []

    @patch("automl_model_training.tools.compare_experiments")
    def test_passes_last_n(self, mock_compare):
        from automl_model_training.tools import tool_compare_runs

        mock_compare.return_value = pd.DataFrame()
        tool_compare_runs(last_n=3)
        mock_compare.assert_called_once_with(last_n=3)
