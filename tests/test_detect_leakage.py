"""Tests for tool_detect_leakage."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from automl_model_training.tools import tool_detect_leakage


@pytest.fixture()
def binary_csv(tmp_path):
    """Binary classification CSV: two legit features, one leak (copy of target)."""
    rng = np.random.RandomState(42)
    n = 500
    df = pd.DataFrame(
        {
            "feat_a": rng.randn(n),
            "feat_b": rng.choice(["x", "y", "z"], n),
            "leaked_target": rng.choice([0, 1], n, p=[0.7, 0.3]),
        }
    )
    df["target"] = df["leaked_target"]  # perfect copy
    path = tmp_path / "binary.csv"
    df.to_csv(path, index=False)
    return path


@pytest.fixture()
def regression_csv(tmp_path):
    """Regression CSV: two legit features, one near-copy of target."""
    rng = np.random.RandomState(42)
    n = 500
    target = rng.randn(n) * 100 + 500
    df = pd.DataFrame(
        {
            "feat_a": rng.randn(n),
            "feat_b": rng.randn(n) * 10,
            "log_target_proxy": np.log(target + 1) + rng.randn(n) * 0.001,  # near-perfect
            "target": target,
        }
    )
    path = tmp_path / "regression.csv"
    df.to_csv(path, index=False)
    return path


@pytest.fixture()
def clean_csv(tmp_path):
    """CSV with no leakage — features have moderate signal only."""
    rng = np.random.RandomState(42)
    n = 500
    df = pd.DataFrame(
        {
            "feat_a": rng.randn(n),
            "feat_b": rng.randn(n),
            "feat_c": rng.randn(n),
            "target": rng.choice([0, 1], n, p=[0.6, 0.4]),
        }
    )
    path = tmp_path / "clean.csv"
    df.to_csv(path, index=False)
    return path


def test_missing_label_raises(tmp_path):
    df = pd.DataFrame({"a": [1, 2, 3]})
    path = tmp_path / "no_label.csv"
    df.to_csv(path, index=False)
    with pytest.raises(ValueError, match="Label column 'target' not in CSV"):
        tool_detect_leakage(str(path), label="target")


def test_detects_perfect_copy_binary(binary_csv):
    result = tool_detect_leakage(str(binary_csv), label="target")
    assert result["problem_type"] == "classification"
    leak_features = [s["feature"] for s in result["suspected_leaks"]]
    assert "leaked_target" in leak_features
    # Perfect copy should score at or near 1.0
    leaked_score = next(
        s["score"] for s in result["suspected_leaks"] if s["feature"] == "leaked_target"
    )
    assert leaked_score >= 0.99


def test_detects_near_copy_regression(regression_csv):
    result = tool_detect_leakage(str(regression_csv), label="target")
    assert result["problem_type"] == "regression"
    leak_features = [s["feature"] for s in result["suspected_leaks"]]
    assert "log_target_proxy" in leak_features


def test_clean_csv_no_leaks(clean_csv):
    result = tool_detect_leakage(str(clean_csv), label="target")
    assert result["suspected_leaks"] == []
    # All features should still be scored
    assert len(result["all_scores"]) == 3


def test_hints_include_drop_list(binary_csv):
    result = tool_detect_leakage(str(binary_csv), label="target")
    assert any("Suggested drop list" in h for h in result["hints"])
    assert any("leaked_target" in h for h in result["hints"])


def test_threshold_parameter_controls_sensitivity(binary_csv):
    # A very high threshold catches only perfect predictors
    strict = tool_detect_leakage(str(binary_csv), label="target", threshold=0.999)
    # A very low threshold flags legitimately-strong features too
    loose = tool_detect_leakage(str(binary_csv), label="target", threshold=0.5)
    assert len(loose["suspected_leaks"]) >= len(strict["suspected_leaks"])


def test_all_scores_sorted_by_score_desc(regression_csv):
    result = tool_detect_leakage(str(regression_csv), label="target")
    scores = [s["score"] for s in result["all_scores"]]
    assert scores == sorted(scores, reverse=True)


def test_handles_categorical_features(tmp_path):
    """Non-numeric feature columns get label-encoded automatically."""
    rng = np.random.RandomState(0)
    n = 500
    df = pd.DataFrame(
        {
            "category": rng.choice(["a", "b", "c"], n),
            "numeric": rng.randn(n),
            "target": rng.choice([0, 1], n),
        }
    )
    path = tmp_path / "cat.csv"
    df.to_csv(path, index=False)
    result = tool_detect_leakage(str(path), label="target")
    # Both features should appear in all_scores (no failures)
    feature_names = [s["feature"] for s in result["all_scores"]]
    assert "category" in feature_names
    assert "numeric" in feature_names


def test_handles_nan_target_rows(tmp_path):
    """Rows with NaN target should be dropped silently, not crash."""
    rng = np.random.RandomState(0)
    n = 100
    target = rng.choice([0, 1, np.nan], n, p=[0.45, 0.45, 0.1])
    df = pd.DataFrame({"feat_a": rng.randn(n), "target": target})
    path = tmp_path / "nan.csv"
    df.to_csv(path, index=False)
    result = tool_detect_leakage(str(path), label="target")
    # No crash; feat_a should be scored
    assert len(result["all_scores"]) == 1


def test_handles_all_nan_feature(tmp_path):
    """A feature that's 100% NaN is silently skipped."""
    df = pd.DataFrame(
        {
            "good_feat": [1.0, 2.0, 3.0, 4.0, 5.0] * 20,
            "nan_feat": [np.nan] * 100,
            "target": [0, 1] * 50,
        }
    )
    path = tmp_path / "partial_nan.csv"
    df.to_csv(path, index=False)
    result = tool_detect_leakage(str(path), label="target")
    scored = [s["feature"] for s in result["all_scores"]]
    assert "good_feat" in scored
    assert "nan_feat" not in scored


def test_sample_size_caps_large_dataset(tmp_path):
    """sample_size should subsample without changing the leak signal."""
    rng = np.random.RandomState(42)
    n = 20000
    df = pd.DataFrame(
        {
            "feat_a": rng.randn(n),
            "leaked": rng.choice([0, 1], n),
        }
    )
    df["target"] = df["leaked"]
    path = tmp_path / "big.csv"
    df.to_csv(path, index=False)
    # With default sample_size=5000, the leak is still caught trivially
    result = tool_detect_leakage(str(path), label="target", sample_size=1000)
    leak_features = [s["feature"] for s in result["suspected_leaks"]]
    assert "leaked" in leak_features
