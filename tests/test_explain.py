"""Tests for SHAP-based model explainability."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from automl_model_training.evaluate.explain import (
    build_shap_per_row,
    build_shap_summary,
    save_explainability_artifacts,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def sample_data() -> pd.DataFrame:
    rng = np.random.RandomState(42)
    return pd.DataFrame(
        {
            "feat_a": rng.randn(50),
            "feat_b": rng.randn(50),
            "feat_c": rng.randn(50),
            "target": rng.choice([0, 1], 50),
        }
    )


@pytest.fixture()
def shap_values_2d() -> np.ndarray:
    """SHAP values for regression/binary: (50, 3) features."""
    rng = np.random.RandomState(42)
    return rng.randn(50, 3)


@pytest.fixture()
def shap_values_3d() -> np.ndarray:
    """SHAP values for multiclass: (50, 3, 4) features x classes."""
    rng = np.random.RandomState(42)
    return rng.randn(50, 3, 4)


@pytest.fixture()
def feature_names() -> list[str]:
    return ["feat_a", "feat_b", "feat_c"]


# ---------------------------------------------------------------------------
# build_shap_summary
# ---------------------------------------------------------------------------


class TestBuildShapSummary:
    def test_2d_values(self, shap_values_2d: np.ndarray, feature_names: list[str]):
        summary = build_shap_summary(shap_values_2d, feature_names)
        assert len(summary) == 3
        assert list(summary.columns) == ["feature", "mean_abs_shap", "rank"]
        # Sorted descending by mean_abs_shap
        assert summary["mean_abs_shap"].is_monotonic_decreasing
        assert list(summary["rank"]) == [1, 2, 3]

    def test_3d_values(self, shap_values_3d: np.ndarray, feature_names: list[str]):
        summary = build_shap_summary(shap_values_3d, feature_names)
        assert len(summary) == 3
        assert all(summary["mean_abs_shap"] > 0)


# ---------------------------------------------------------------------------
# build_shap_per_row
# ---------------------------------------------------------------------------


class TestBuildShapPerRow:
    def test_returns_correct_rows(
        self,
        shap_values_2d: np.ndarray,
        sample_data: pd.DataFrame,
        feature_names: list[str],
    ):
        per_row = build_shap_per_row(shap_values_2d, sample_data, feature_names)
        assert len(per_row) == 50
        assert "top_features" in per_row.columns
        assert "row_index" in per_row.columns

    def test_top_features_capped_at_feature_count(
        self,
        sample_data: pd.DataFrame,
        feature_names: list[str],
    ):
        # Only 3 features, so top should be 3 not 5
        rng = np.random.RandomState(42)
        vals = rng.randn(50, 3)
        per_row = build_shap_per_row(vals, sample_data, feature_names)
        top = per_row.iloc[0]["top_features"]
        assert len(top) == 3  # min(5, 3 features)

    def test_3d_values(
        self,
        shap_values_3d: np.ndarray,
        sample_data: pd.DataFrame,
        feature_names: list[str],
    ):
        per_row = build_shap_per_row(shap_values_3d, sample_data, feature_names)
        assert len(per_row) == 50


# ---------------------------------------------------------------------------
# save_explainability_artifacts
# ---------------------------------------------------------------------------


class TestSaveExplainabilityArtifacts:
    @patch("automl_model_training.evaluate.explain.compute_shap_values")
    def test_saves_all_files(
        self,
        mock_compute: MagicMock,
        sample_data: pd.DataFrame,
        shap_values_2d: np.ndarray,
        tmp_path: Path,
    ):
        mock_compute.return_value = (
            shap_values_2d,
            np.array(0.5),
            ["feat_a", "feat_b", "feat_c"],
        )

        predictor = MagicMock()
        predictor.problem_type = "binary"
        predictor.label = "target"

        metadata = save_explainability_artifacts(
            predictor,
            sample_data,
            tmp_path,
            max_samples=50,
        )

        assert (tmp_path / "shap_summary.csv").exists()
        assert (tmp_path / "shap_values.csv").exists()
        assert (tmp_path / "shap_per_row.json").exists()
        assert (tmp_path / "shap_metadata.json").exists()

        # Verify metadata content
        assert metadata["problem_type"] == "binary"
        assert metadata["n_features"] == 3
        assert len(metadata["top_5_features"]) == 3

        # Verify JSON is valid
        with open(tmp_path / "shap_metadata.json") as f:
            saved = json.load(f)
        assert saved["n_samples_explained"] == 50

    @patch("automl_model_training.evaluate.explain.compute_shap_values")
    def test_multiclass_saves(
        self,
        mock_compute: MagicMock,
        sample_data: pd.DataFrame,
        shap_values_3d: np.ndarray,
        tmp_path: Path,
    ):
        mock_compute.return_value = (
            shap_values_3d,
            np.array([0.25, 0.25, 0.25, 0.25]),
            ["feat_a", "feat_b", "feat_c"],
        )

        predictor = MagicMock()
        predictor.problem_type = "multiclass"
        predictor.label = "target"

        metadata = save_explainability_artifacts(
            predictor,
            sample_data,
            tmp_path,
            max_samples=50,
        )

        assert metadata["problem_type"] == "multiclass"
        # shap_values.csv should be 2D (averaged across classes)
        shap_df = pd.read_csv(tmp_path / "shap_values.csv")
        assert shap_df.shape == (50, 3)
