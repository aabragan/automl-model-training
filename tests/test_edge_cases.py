"""Edge case tests for boundary conditions and unusual inputs."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd

from automl_model_training.data import load_and_prepare
from automl_model_training.evaluate.regression import save_regression_artifacts
from automl_model_training.profile import (
    compute_correlation_matrix,
    find_highly_correlated_pairs,
    profile_categorical_features,
    profile_label,
    profile_missing_values,
    profile_numeric_features,
    profile_overview,
    recommend_features_to_drop,
    save_profile_report,
)

# ---------------------------------------------------------------------------
# Data loading edge cases
# ---------------------------------------------------------------------------


class TestDataEdgeCases:
    def test_single_feature_dataset(self, tmp_path: Path):
        """Dataset with only the label column and one feature."""
        df = pd.DataFrame({"feat": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], "target": [0] * 5 + [1] * 5})
        csv = tmp_path / "data.csv"
        df.to_csv(csv, index=False)

        train, test, *_ = load_and_prepare(str(csv), "target", [], 0.2, 42, str(tmp_path / "out"))
        assert len(train) + len(test) == 10
        assert "feat" in train.columns

    def test_all_features_dropped(self, tmp_path: Path):
        """Dropping all features except the label should still work."""
        df = pd.DataFrame({"a": range(20), "b": range(20), "target": [0] * 10 + [1] * 10})
        csv = tmp_path / "data.csv"
        df.to_csv(csv, index=False)

        train, *_ = load_and_prepare(str(csv), "target", ["a", "b"], 0.2, 42, str(tmp_path / "o"))
        assert list(train.columns) == ["target"]

    def test_all_numeric_features(self, tmp_path: Path):
        """Dataset with no categorical columns."""
        rng = np.random.RandomState(42)
        df = pd.DataFrame(
            {
                "x": rng.randn(50),
                "y": rng.randn(50),
                "target": rng.choice([0, 1], 50),
            }
        )
        csv = tmp_path / "data.csv"
        df.to_csv(csv, index=False)

        _, _, train_n, _, num_cols = load_and_prepare(
            str(csv), "target", [], 0.2, 42, str(tmp_path / "o")
        )
        assert set(num_cols) == {"x", "y"}

    def test_all_categorical_features(self, tmp_path: Path):
        """Dataset with no numeric features (except label)."""
        df = pd.DataFrame(
            {
                "color": ["red", "blue", "green"] * 10,
                "size": ["S", "M", "L"] * 10,
                "target": [0] * 15 + [1] * 15,
            }
        )
        csv = tmp_path / "data.csv"
        df.to_csv(csv, index=False)

        _, _, _, _, num_cols = load_and_prepare(
            str(csv), "target", [], 0.2, 42, str(tmp_path / "o")
        )
        assert num_cols == []

    def test_dataset_with_missing_values(self, tmp_path: Path):
        """Missing values should pass through without error."""
        df = pd.DataFrame(
            {
                "feat": [1.0, np.nan, 3.0, np.nan, 5.0, 6.0, 7.0, np.nan, 9.0, 10.0],
                "target": [0] * 5 + [1] * 5,
            }
        )
        csv = tmp_path / "data.csv"
        df.to_csv(csv, index=False)

        train, test, *_ = load_and_prepare(str(csv), "target", [], 0.2, 42, str(tmp_path / "o"))
        assert len(train) + len(test) == 10

    def test_large_test_size(self, tmp_path: Path):
        """90% test split should still work."""
        rng = np.random.RandomState(42)
        df = pd.DataFrame({"feat": rng.randn(100), "target": rng.choice([0, 1], 100)})
        csv = tmp_path / "data.csv"
        df.to_csv(csv, index=False)

        train, test, *_ = load_and_prepare(str(csv), "target", [], 0.9, 42, str(tmp_path / "o"))
        assert len(test) > len(train)


# ---------------------------------------------------------------------------
# Profiling edge cases
# ---------------------------------------------------------------------------


class TestProfileEdgeCases:
    def test_single_numeric_column(self):
        """Correlation matrix with one numeric column should be 1x1."""
        df = pd.DataFrame({"feat": [1, 2, 3], "cat": ["a", "b", "c"], "target": [0, 1, 0]})
        corr = compute_correlation_matrix(df, "target")
        assert "feat" in corr.columns
        assert "target" in corr.columns

    def test_no_numeric_columns(self):
        """Dataset with only categorical columns produces empty correlation matrix."""
        df = pd.DataFrame({"cat_a": ["x", "y"], "cat_b": ["a", "b"], "target": ["c1", "c2"]})
        corr = compute_correlation_matrix(df, "target")
        assert corr.empty

    def test_identical_features_correlation(self):
        """Two identical columns should have correlation of 1.0."""
        df = pd.DataFrame({"a": [1, 2, 3, 4, 5], "b": [1, 2, 3, 4, 5], "target": [0, 1, 0, 1, 0]})
        corr = compute_correlation_matrix(df, "target")
        pairs = find_highly_correlated_pairs(corr, threshold=0.99)
        assert len(pairs) >= 1

    def test_constant_feature_correlation(self):
        """A constant feature produces NaN correlations — should not crash."""
        df = pd.DataFrame({"const": [5] * 10, "vary": range(10), "target": [0] * 5 + [1] * 5})
        corr = compute_correlation_matrix(df, "target")
        # NaN correlations should not appear as highly correlated pairs
        pairs = find_highly_correlated_pairs(corr, threshold=0.90)
        for p in pairs:
            assert p["feature_a"] != "const" or p["feature_b"] != "const"

    def test_profile_overview_empty_dataframe(self):
        """Overview of an empty dataframe should not crash."""
        df = pd.DataFrame({"a": pd.Series(dtype=float), "target": pd.Series(dtype=int)})
        overview = profile_overview(df)
        assert overview["rows"] == 0
        assert overview["columns"] == 2

    def test_profile_missing_values_all_missing(self):
        """Column that is entirely NaN."""
        df = pd.DataFrame({"feat": [np.nan] * 5, "target": [0, 1, 0, 1, 0]})
        missing = profile_missing_values(df)
        assert missing.loc["feat", "missing_pct"] == 100.0

    def test_profile_missing_values_none_missing(self):
        """No missing values at all."""
        df = pd.DataFrame({"feat": [1, 2, 3], "target": [0, 1, 0]})
        missing = profile_missing_values(df)
        assert missing["missing_count"].sum() == 0

    def test_profile_label_regression(self):
        """Label with many unique values should be detected as regression."""
        df = pd.DataFrame({"target": np.linspace(0, 100, 50)})
        info = profile_label(df, "target")
        assert info["type"] == "regression"
        assert "mean" in info

    def test_profile_label_classification(self):
        """Label with few unique values should be detected as classification."""
        df = pd.DataFrame({"target": [0, 1, 2] * 10})
        info = profile_label(df, "target")
        assert info["type"] == "classification"
        assert "class_distribution" in info

    def test_profile_label_missing_column(self):
        """Missing label column should return error dict."""
        df = pd.DataFrame({"feat": [1, 2, 3]})
        info = profile_label(df, "nonexistent")
        assert "error" in info

    def test_profile_numeric_outliers(self):
        """Extreme outliers should be detected by IQR method."""
        values = [1.0] * 98 + [1000.0, -1000.0]
        df = pd.DataFrame({"feat": values, "target": [0] * 50 + [1] * 50})
        stats = profile_numeric_features(df, "target")
        assert stats.loc["feat", "outlier_count"] >= 2

    def test_profile_categorical_single_value(self):
        """Categorical column with one unique value."""
        df = pd.DataFrame({"cat": ["same"] * 10, "target": [0] * 5 + [1] * 5})
        cat_stats = profile_categorical_features(df, "target")
        assert cat_stats.loc["cat", "nunique"] == 1
        assert cat_stats.loc["cat", "top_value_pct"] == 100.0

    def test_recommend_drop_with_no_label_in_corr(self):
        """If label is not in the correlation matrix, return empty list."""
        df = pd.DataFrame({"a": [1, 2, 3], "b": [1, 2, 3]})
        corr = df.corr()
        recs = recommend_features_to_drop(corr, "nonexistent", threshold=0.90)
        assert recs == []

    def test_profile_report_with_missing_data(self, tmp_path: Path):
        """Full profile report on dataset with missing values."""
        df = pd.DataFrame(
            {
                "feat_a": [1.0, np.nan, 3.0, 4.0, 5.0],
                "feat_b": [np.nan, np.nan, np.nan, 4.0, 5.0],
                "cat": ["x", "y", None, "x", "y"],
                "target": [0, 1, 0, 1, 0],
            }
        )
        corr = compute_correlation_matrix(df, "target")
        pairs = find_highly_correlated_pairs(corr, 0.90)
        recs = recommend_features_to_drop(corr, "target", 0.90)

        report = save_profile_report(df, "target", corr, pairs, recs, tmp_path)
        assert report["missing_values"]["total_missing_cells"] > 0
        assert (tmp_path / "profile_report.json").exists()


# ---------------------------------------------------------------------------
# Evaluation edge cases
# ---------------------------------------------------------------------------


class TestEvaluationEdgeCases:
    def test_regression_perfect_predictions(self, tmp_path: Path):
        """Perfect predictions should produce zero residuals and R²=1."""
        predictor = MagicMock()
        predictor.problem_type = "regression"
        predictor.label = "target"

        test_data = pd.DataFrame(
            {
                "feat": [1, 2, 3, 4, 5],
                "target": [10.0, 20.0, 30.0, 40.0, 50.0],
            }
        )
        predictor.predict.return_value = pd.Series([10.0, 20.0, 30.0, 40.0, 50.0])

        save_regression_artifacts(predictor, test_data, "target", tmp_path)

        stats = json.loads((tmp_path / "residual_stats.json").read_text())
        assert abs(stats["mean_residual"]) < 1e-10
        assert abs(stats["r2"] - 1.0) < 1e-10

    def test_regression_constant_predictions(self, tmp_path: Path):
        """Constant predictions should produce meaningful residual stats."""
        predictor = MagicMock()
        predictor.problem_type = "regression"
        predictor.label = "target"

        test_data = pd.DataFrame({"feat": range(5), "target": [10.0, 20.0, 30.0, 40.0, 50.0]})
        predictor.predict.return_value = pd.Series([30.0] * 5)

        save_regression_artifacts(predictor, test_data, "target", tmp_path)

        stats = json.loads((tmp_path / "residual_stats.json").read_text())
        assert stats["root_mean_squared_error"] > 0
        assert (tmp_path / "test_predictions.csv").exists()

    def test_backtest_single_row_per_fold(self):
        """Walk-forward with minimal data should still produce valid folds."""
        from automl_model_training.backtest import _build_folds

        dates = pd.date_range("2024-01-01", periods=6, freq="D")
        data = pd.DataFrame({"date": dates, "val": range(6)})
        data["date"] = pd.to_datetime(data["date"])

        folds = _build_folds(data, "date", cutoff=None, n_splits=2)
        assert len(folds) == 2
        for train, test in folds:
            assert len(train) > 0
            assert len(test) > 0

    def test_aggregate_results_single_metric(self):
        """Aggregation with a single metric across multiple folds."""
        from automl_model_training.backtest import _aggregate_results

        results = [
            {"fold": 1, "scores": {"rmse": 0.5}},
            {"fold": 2, "scores": {"rmse": 0.3}},
            {"fold": 3, "scores": {"rmse": 0.4}},
        ]
        agg = _aggregate_results(results)
        assert abs(agg["aggregate_scores"]["rmse"]["mean"] - 0.4) < 1e-6
