"""Tests for data drift detection."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from automl_model_training.drift import compute_psi, detect_drift, save_drift_report


class TestComputePsi:
    def test_identical_distributions_return_zero(self):
        rng = np.random.RandomState(42)
        data = pd.Series(rng.randn(1000))
        psi = compute_psi(data, data)
        assert psi < 0.01

    def test_shifted_distribution_returns_high_psi(self):
        rng = np.random.RandomState(42)
        expected = pd.Series(rng.randn(1000))
        # Shift mean by 3 standard deviations
        actual = pd.Series(rng.randn(1000) + 3.0)
        psi = compute_psi(expected, actual)
        assert psi > 0.25

    def test_slightly_shifted_returns_moderate_psi(self):
        rng = np.random.RandomState(42)
        expected = pd.Series(rng.randn(1000))
        actual = pd.Series(rng.randn(1000) + 0.3)
        psi = compute_psi(expected, actual)
        assert 0.0 < psi < 1.0

    def test_handles_constant_feature(self):
        expected = pd.Series([1.0] * 100)
        actual = pd.Series([1.0] * 100)
        psi = compute_psi(expected, actual)
        assert psi == 0.0

    def test_handles_nans(self):
        rng = np.random.RandomState(42)
        expected = pd.Series([np.nan, *rng.randn(99).tolist()])
        actual = pd.Series(rng.randn(100))
        psi = compute_psi(expected, actual)
        assert isinstance(psi, float)


class TestDetectDrift:
    def _make_data(self, n: int, shift: float = 0.0) -> pd.DataFrame:
        rng = np.random.RandomState(42)
        return pd.DataFrame(
            {
                "feat_a": rng.randn(n) + shift,
                "feat_b": rng.randn(n),
                "cat_col": rng.choice(["x", "y"], n),
                "target": rng.choice([0, 1], n),
            }
        )

    def test_no_drift_on_same_distribution(self):
        train = self._make_data(500)
        predict = self._make_data(200)
        results = detect_drift(train, predict, "target")

        assert len(results) == 2  # feat_a and feat_b (cat_col excluded)
        for r in results:
            assert r["status"] == "no_drift"

    def test_detects_significant_drift(self):
        train = self._make_data(500)
        predict = self._make_data(200, shift=5.0)  # large shift on feat_a
        results = detect_drift(train, predict, "target")

        feat_a = next(r for r in results if r["feature"] == "feat_a")
        assert feat_a["status"] == "significant_drift"
        assert feat_a["psi"] > 0.25

    def test_excludes_label_column(self):
        train = self._make_data(500)
        predict = self._make_data(200)
        results = detect_drift(train, predict, "target")

        features = [r["feature"] for r in results]
        assert "target" not in features

    def test_excludes_categorical_columns(self):
        train = self._make_data(500)
        predict = self._make_data(200)
        results = detect_drift(train, predict, "target")

        features = [r["feature"] for r in results]
        assert "cat_col" not in features

    def test_sorted_by_psi_descending(self):
        train = self._make_data(500)
        predict = self._make_data(200, shift=3.0)
        results = detect_drift(train, predict, "target")

        psi_values = [r["psi"] for r in results]
        assert psi_values == sorted(psi_values, reverse=True)

    def test_returns_empty_for_no_shared_features(self):
        train = pd.DataFrame({"a": [1, 2, 3], "target": [0, 1, 0]})
        predict = pd.DataFrame({"b": [4, 5, 6], "target": [1, 0, 1]})
        results = detect_drift(train, predict, "target")
        assert results == []


class TestSaveDriftReport:
    def test_saves_json_and_csv(self, tmp_path: Path):
        results = [
            {
                "feature": "feat_a",
                "psi": 0.35,
                "status": "significant_drift",
                "train_mean": 0.0,
                "predict_mean": 3.0,
                "mean_shift_pct": 100.0,
                "train_std": 1.0,
                "predict_std": 1.0,
            },
            {
                "feature": "feat_b",
                "psi": 0.02,
                "status": "no_drift",
                "train_mean": 0.0,
                "predict_mean": 0.05,
                "mean_shift_pct": 5.0,
                "train_std": 1.0,
                "predict_std": 1.0,
            },
        ]

        summary = save_drift_report(results, tmp_path)

        assert (tmp_path / "drift_report.json").exists()
        assert (tmp_path / "drift_report.csv").exists()
        assert summary["significant_drift"] == 1
        assert summary["no_drift"] == 1
        assert summary["drifted_features"] == ["feat_a"]

    def test_summary_counts_are_correct(self, tmp_path: Path):
        results = [
            {
                "feature": "a",
                "psi": 0.30,
                "status": "significant_drift",
                "train_mean": 0,
                "predict_mean": 0,
                "mean_shift_pct": 0,
                "train_std": 1,
                "predict_std": 1,
            },
            {
                "feature": "b",
                "psi": 0.15,
                "status": "moderate_drift",
                "train_mean": 0,
                "predict_mean": 0,
                "mean_shift_pct": 0,
                "train_std": 1,
                "predict_std": 1,
            },
            {
                "feature": "c",
                "psi": 0.05,
                "status": "no_drift",
                "train_mean": 0,
                "predict_mean": 0,
                "mean_shift_pct": 0,
                "train_std": 1,
                "predict_std": 1,
            },
        ]

        summary = save_drift_report(results, tmp_path)

        assert summary["features_checked"] == 3
        assert summary["significant_drift"] == 1
        assert summary["moderate_drift"] == 1
        assert summary["no_drift"] == 1

    def test_handles_empty_results(self, tmp_path: Path):
        summary = save_drift_report([], tmp_path)

        assert summary["features_checked"] == 0
        assert (tmp_path / "drift_report.json").exists()
