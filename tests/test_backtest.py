"""Tests for the temporal backtesting module."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from automl_model_training.backtest import _aggregate_results, _build_folds, temporal_backtest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_dated_csv(tmp_path: Path, n: int = 200) -> tuple[Path, pd.DataFrame]:
    """Create a CSV with a date column, features, and a binary target."""
    rng = np.random.RandomState(42)
    dates = pd.date_range("2024-01-01", periods=n, freq="D")
    df = pd.DataFrame(
        {
            "date": dates,
            "feat_a": rng.randn(n),
            "feat_b": rng.randn(n),
            "target": rng.choice([0, 1], n),
        }
    )
    csv_path = tmp_path / "data.csv"
    df.to_csv(csv_path, index=False)
    return csv_path, df


# ---------------------------------------------------------------------------
# _build_folds
# ---------------------------------------------------------------------------


class TestBuildFolds:
    def test_single_cutoff(self):
        dates = pd.date_range("2024-01-01", periods=100, freq="D")
        data = pd.DataFrame({"date": dates, "val": range(100)})
        data["date"] = pd.to_datetime(data["date"])

        folds = _build_folds(data, "date", cutoff="2024-03-01", n_splits=1)
        assert len(folds) == 1
        train, test = folds[0]
        assert all(train["date"] < pd.Timestamp("2024-03-01"))
        assert all(test["date"] >= pd.Timestamp("2024-03-01"))

    def test_walk_forward_splits(self):
        dates = pd.date_range("2024-01-01", periods=300, freq="D")
        data = pd.DataFrame({"date": dates, "val": range(300)})
        data["date"] = pd.to_datetime(data["date"])

        folds = _build_folds(data, "date", cutoff=None, n_splits=3)
        assert len(folds) == 3

        # Each fold's training set should be larger than the previous
        for i in range(1, len(folds)):
            assert len(folds[i][0]) > len(folds[i - 1][0])

    def test_cutoff_empty_split_raises(self):
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        data = pd.DataFrame({"date": dates, "val": range(10)})
        data["date"] = pd.to_datetime(data["date"])

        with pytest.raises(ValueError, match="empty split"):
            _build_folds(data, "date", cutoff="2020-01-01", n_splits=1)

    def test_too_few_rows_raises(self):
        data = pd.DataFrame({"date": pd.to_datetime(["2024-01-01"]), "val": [1]})
        with pytest.raises(ValueError, match="Not enough data"):
            _build_folds(data, "date", cutoff=None, n_splits=5)


# ---------------------------------------------------------------------------
# _aggregate_results
# ---------------------------------------------------------------------------


class TestAggregateResults:
    def test_single_fold(self):
        results = [{"fold": 1, "scores": {"f1": 0.8, "accuracy": 0.9}}]
        agg = _aggregate_results(results)
        assert agg["aggregate_scores"]["f1"]["mean"] == 0.8
        assert agg["aggregate_scores"]["f1"]["std"] == 0.0

    def test_multiple_folds(self):
        results = [
            {"fold": 1, "scores": {"f1": 0.7}},
            {"fold": 2, "scores": {"f1": 0.9}},
        ]
        agg = _aggregate_results(results)
        assert agg["aggregate_scores"]["f1"]["mean"] == 0.8
        assert agg["aggregate_scores"]["f1"]["std"] > 0

    def test_empty_results(self):
        agg = _aggregate_results([])
        assert agg["aggregate_scores"] == {}


# ---------------------------------------------------------------------------
# temporal_backtest (integration with mocked train_and_evaluate)
# ---------------------------------------------------------------------------


class TestTemporalBacktest:
    @patch("automl_model_training.backtest.train_and_evaluate")
    def test_single_cutoff_backtest(self, mock_train, tmp_path: Path):
        csv_path, _ = _make_dated_csv(tmp_path)
        output_dir = str(tmp_path / "bt_output")

        mock_predictor = MagicMock()
        mock_predictor.evaluate.return_value = {"f1": 0.85, "accuracy": 0.90}
        mock_train.return_value = mock_predictor

        summary = temporal_backtest(
            csv_path=str(csv_path),
            date_column="date",
            label="target",
            cutoff="2024-04-01",
            n_splits=1,
            problem_type="binary",
            eval_metric="f1",
            time_limit=60,
            preset="best",
            output_dir=output_dir,
            features_to_drop=[],
        )

        assert summary["n_folds"] == 1
        assert len(summary["folds"]) == 1
        assert "f1" in summary["aggregate_scores"]
        assert (Path(output_dir) / "backtest_summary.json").exists()
        mock_train.assert_called_once()

    @patch("automl_model_training.backtest.train_and_evaluate")
    def test_walk_forward_backtest(self, mock_train, tmp_path: Path):
        csv_path, _ = _make_dated_csv(tmp_path, n=300)
        output_dir = str(tmp_path / "bt_output")

        mock_predictor = MagicMock()
        mock_predictor.evaluate.return_value = {"f1": 0.80}
        mock_train.return_value = mock_predictor

        summary = temporal_backtest(
            csv_path=str(csv_path),
            date_column="date",
            label="target",
            cutoff=None,
            n_splits=3,
            problem_type=None,
            eval_metric=None,
            time_limit=None,
            preset="best",
            output_dir=output_dir,
            features_to_drop=[],
        )

        assert summary["n_folds"] == 3
        assert mock_train.call_count == 3

    @patch("automl_model_training.backtest.train_and_evaluate")
    def test_features_dropped(self, mock_train, tmp_path: Path):
        csv_path, _ = _make_dated_csv(tmp_path)
        output_dir = str(tmp_path / "bt_output")

        mock_predictor = MagicMock()
        mock_predictor.evaluate.return_value = {"f1": 0.80}
        mock_train.return_value = mock_predictor

        temporal_backtest(
            csv_path=str(csv_path),
            date_column="date",
            label="target",
            cutoff="2024-04-01",
            n_splits=1,
            problem_type="binary",
            eval_metric="f1",
            time_limit=None,
            preset="best",
            output_dir=output_dir,
            features_to_drop=["feat_b"],
        )

        # The train call should not include feat_b or date
        call_args = mock_train.call_args
        train_df = call_args.kwargs.get("train_raw")
        if train_df is None:
            train_df = call_args[0][0]
        assert "feat_b" not in train_df.columns
        assert "date" not in train_df.columns
