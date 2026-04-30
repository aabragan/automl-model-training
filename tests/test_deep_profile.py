"""Tests for tool_deep_profile."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from automl_model_training.tools import tool_deep_profile


def test_deep_profile_missing_label_raises(tmp_path):
    df = pd.DataFrame({"a": [1, 2, 3]})
    path = tmp_path / "d.csv"
    df.to_csv(path, index=False)
    with pytest.raises(ValueError, match="Label column 'target' not in CSV"):
        tool_deep_profile(str(path), label="target")


def test_deep_profile_recommends_log_for_right_skewed(tmp_path):
    # Right-skewed positive feature should get log recommendation
    rng = np.random.RandomState(0)
    df = pd.DataFrame(
        {
            "skewed_price": np.exp(rng.randn(500) * 2),  # strongly right-skewed, positive
            "target": rng.randn(500),
        }
    )
    path = tmp_path / "skew.csv"
    df.to_csv(path, index=False)

    result = tool_deep_profile(str(path), label="target")
    assert "log" in result["suggested_transforms"]
    assert "skewed_price" in result["suggested_transforms"]["log"]
    log_recs = [f for f in result["numeric_features"] if f["feature"] == "skewed_price"]
    assert any("log transform" in f["recommendation"] for f in log_recs)


def test_deep_profile_recommends_onehot_for_low_cardinality(tmp_path):
    rng = np.random.RandomState(0)
    df = pd.DataFrame(
        {
            "category": rng.choice(["a", "b", "c"], 300),
            "target": rng.randn(300),
        }
    )
    path = tmp_path / "cat.csv"
    df.to_csv(path, index=False)

    result = tool_deep_profile(str(path), label="target")
    assert "onehot" in result["suggested_transforms"]
    assert "category" in result["suggested_transforms"]["onehot"]


def test_deep_profile_flags_high_cardinality(tmp_path):
    rng = np.random.RandomState(0)
    df = pd.DataFrame(
        {
            "user_id": [f"user_{i}" for i in rng.randint(0, 300, 500)],
            "target": rng.randn(500),
        }
    )
    path = tmp_path / "hc.csv"
    df.to_csv(path, index=False)

    result = tool_deep_profile(str(path), label="target")
    # High cardinality shouldn't be in onehot suggestions
    assert "user_id" not in result["suggested_transforms"].get("onehot", [])
    # But should appear in categorical_features with a target_mean/drop recommendation
    rec = next(f for f in result["categorical_features"] if f["feature"] == "user_id")
    assert "target_mean" in rec["recommendation"] or "drop" in rec["recommendation"]
