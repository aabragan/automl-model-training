"""Tests for dataset profiling and correlation analysis."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from automl_model_training.profile import (
    compute_correlation_matrix,
    find_highly_correlated_pairs,
    plot_correlation_heatmap,
    recommend_features_to_drop,
    save_profile_report,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def correlated_data() -> pd.DataFrame:
    """Dataset with known correlations."""
    rng = np.random.RandomState(42)
    n = 200
    x = rng.randn(n)
    return pd.DataFrame(
        {
            "feat_a": x,
            "feat_b": x + rng.randn(n) * 0.01,  # ~1.0 correlation with feat_a
            "feat_c": rng.randn(n),  # independent
            "feat_d": x * 0.5 + rng.randn(n) * 0.5,  # moderate correlation
            "target": (x > 0).astype(int),
        }
    )


@pytest.fixture()
def uncorrelated_data() -> pd.DataFrame:
    rng = np.random.RandomState(42)
    n = 200
    return pd.DataFrame(
        {
            "feat_a": rng.randn(n),
            "feat_b": rng.randn(n),
            "feat_c": rng.randn(n),
            "target": rng.choice([0, 1], n),
        }
    )


# ---------------------------------------------------------------------------
# compute_correlation_matrix
# ---------------------------------------------------------------------------


class TestComputeCorrelationMatrix:
    def test_returns_square_matrix(self, correlated_data: pd.DataFrame):
        corr = compute_correlation_matrix(correlated_data, "target")
        assert corr.shape[0] == corr.shape[1]
        assert "target" in corr.columns

    def test_label_is_last_column(self, correlated_data: pd.DataFrame):
        corr = compute_correlation_matrix(correlated_data, "target")
        assert corr.columns[-1] == "target"

    def test_diagonal_is_one(self, correlated_data: pd.DataFrame):
        corr = compute_correlation_matrix(correlated_data, "target")
        for i in range(len(corr)):
            assert abs(corr.iloc[i, i] - 1.0) < 1e-10


# ---------------------------------------------------------------------------
# find_highly_correlated_pairs
# ---------------------------------------------------------------------------


class TestFindHighlyCorrelatedPairs:
    def test_finds_known_pair(self, correlated_data: pd.DataFrame):
        corr = compute_correlation_matrix(correlated_data, "target")
        pairs = find_highly_correlated_pairs(corr, threshold=0.90)
        pair_features = {(p["feature_a"], p["feature_b"]) for p in pairs}
        # feat_a and feat_b should be highly correlated
        assert ("feat_a", "feat_b") in pair_features or ("feat_b", "feat_a") in pair_features

    def test_no_pairs_below_threshold(self, uncorrelated_data: pd.DataFrame):
        corr = compute_correlation_matrix(uncorrelated_data, "target")
        pairs = find_highly_correlated_pairs(corr, threshold=0.90)
        assert len(pairs) == 0

    def test_sorted_by_abs_correlation(self, correlated_data: pd.DataFrame):
        corr = compute_correlation_matrix(correlated_data, "target")
        pairs = find_highly_correlated_pairs(corr, threshold=0.50)
        if len(pairs) > 1:
            for i in range(len(pairs) - 1):
                assert abs(pairs[i]["correlation"]) >= abs(pairs[i + 1]["correlation"])


# ---------------------------------------------------------------------------
# recommend_features_to_drop
# ---------------------------------------------------------------------------


class TestRecommendFeaturesToDrop:
    def test_recommends_drop_for_correlated_pair(self, correlated_data: pd.DataFrame):
        corr = compute_correlation_matrix(correlated_data, "target")
        recs = recommend_features_to_drop(corr, "target", threshold=0.90)
        drop_names = [r["feature"] for r in recs]
        # One of feat_a or feat_b should be recommended for drop
        assert "feat_a" in drop_names or "feat_b" in drop_names
        # But not both — we keep the one more correlated with target
        assert not ("feat_a" in drop_names and "feat_b" in drop_names)

    def test_no_recommendations_when_uncorrelated(self, uncorrelated_data: pd.DataFrame):
        corr = compute_correlation_matrix(uncorrelated_data, "target")
        recs = recommend_features_to_drop(corr, "target", threshold=0.90)
        assert len(recs) == 0

    def test_never_recommends_label(self, correlated_data: pd.DataFrame):
        corr = compute_correlation_matrix(correlated_data, "target")
        recs = recommend_features_to_drop(corr, "target", threshold=0.50)
        drop_names = [r["feature"] for r in recs]
        assert "target" not in drop_names

    def test_recommendation_has_required_keys(self, correlated_data: pd.DataFrame):
        corr = compute_correlation_matrix(correlated_data, "target")
        recs = recommend_features_to_drop(corr, "target", threshold=0.90)
        for rec in recs:
            assert "feature" in rec
            assert "reason" in rec
            assert "correlated_with" in rec
            assert "pair_correlation" in rec


# ---------------------------------------------------------------------------
# plot_correlation_heatmap
# ---------------------------------------------------------------------------


class TestPlotCorrelationHeatmap:
    def test_saves_png(self, correlated_data: pd.DataFrame, tmp_path: Path):
        corr = compute_correlation_matrix(correlated_data, "target")
        png = plot_correlation_heatmap(corr, tmp_path)
        assert png.exists()
        assert png.suffix == ".png"
        assert png.stat().st_size > 0


# ---------------------------------------------------------------------------
# save_profile_report
# ---------------------------------------------------------------------------


class TestSaveProfileReport:
    def test_saves_all_files(self, correlated_data: pd.DataFrame, tmp_path: Path):
        corr = compute_correlation_matrix(correlated_data, "target")
        pairs = find_highly_correlated_pairs(corr, 0.90)
        recs = recommend_features_to_drop(corr, "target", 0.90)

        save_profile_report(
            correlated_data,
            "target",
            corr,
            pairs,
            recs,
            tmp_path,
        )

        assert (tmp_path / "correlation_matrix.csv").exists()
        assert (tmp_path / "correlation_heatmap.png").exists()
        assert (tmp_path / "feature_stats.csv").exists()
        assert (tmp_path / "profile_report.json").exists()

        report = json.loads((tmp_path / "profile_report.json").read_text())
        assert report["total_rows"] == 200
        assert report["label"] == "target"
        assert isinstance(report["features_to_drop"], list)
        assert isinstance(report["highly_correlated_pairs"], list)
