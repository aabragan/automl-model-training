"""Tests for feature_engineering.py and tool_engineer_features."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from automl_model_training.feature_engineering import (
    ONEHOT_TOP_N,
    apply_transformations,
)
from automl_model_training.tools import tool_engineer_features


@pytest.fixture()
def sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "price": [100.0, 200.0, 0.0, 400.0, -5.0],
            "area": [10.0, 20.0, 30.0, 40.0, 50.0],
            "quantity": [1, 2, 0, 4, 5],
            "city": ["a", "b", "a", "c", "b"],
            "sale_date": ["2024-01-15", "2024-06-20", "invalid", "2025-03-01", "2024-12-31"],
            "start": pd.to_datetime(["2024-01-01"] * 5),
            "end": pd.to_datetime(
                ["2024-02-01", "2024-03-01", "2024-04-01", "2024-05-01", "2024-06-01"]
            ),
            "target": [10, 20, 30, 40, 50],
        }
    )


def test_unknown_transform_raises(sample_df):
    with pytest.raises(ValueError, match="Unknown transformations"):
        apply_transformations(sample_df, {"fake_transform": ["price"]})


def test_missing_column_raises(sample_df):
    with pytest.raises(ValueError, match="columns not in DataFrame"):
        apply_transformations(sample_df, {"log": ["nonexistent"]})


def test_label_as_source_rejected(sample_df):
    with pytest.raises(ValueError, match="Label column 'target' cannot"):
        apply_transformations(sample_df, {"log": ["target"]}, label="target")


def test_label_in_pair_rejected(sample_df):
    with pytest.raises(ValueError, match="Label column 'target' cannot"):
        apply_transformations(sample_df, {"ratio": [["target", "area"]]}, label="target")


def test_log_transform(sample_df):
    out, report = apply_transformations(sample_df, {"log": ["price", "area"]})
    assert "log_price" in out.columns
    assert "log_area" in out.columns
    # log1p(100) == log(101)
    assert out["log_price"].iloc[0] == pytest.approx(np.log1p(100))
    # Negative value → NaN + warning
    assert np.isnan(out["log_price"].iloc[4])
    assert any("negative" in w for w in report["warnings"])


def test_sqrt_transform(sample_df):
    out, _ = apply_transformations(sample_df, {"sqrt": ["area"]})
    assert out["sqrt_area"].iloc[0] == pytest.approx(np.sqrt(10))


def test_ratio_handles_zero_denominator(sample_df):
    out, report = apply_transformations(sample_df, {"ratio": [["price", "quantity"]]})
    assert "price_per_quantity" in out.columns
    # quantity[2] == 0 → NaN (not Inf)
    assert np.isnan(out["price_per_quantity"].iloc[2])
    assert any("zero denominators" in w for w in report["warnings"])


def test_diff_numeric(sample_df):
    out, _ = apply_transformations(sample_df, {"diff": [["price", "area"]]})
    assert out["price_minus_area"].iloc[0] == 90.0


def test_diff_datetime(sample_df):
    out, _ = apply_transformations(sample_df, {"diff": [["end", "start"]]})
    # Days between 2024-01-01 and 2024-02-01 == 31
    assert out["end_minus_start"].iloc[0] == 31


def test_product(sample_df):
    out, _ = apply_transformations(sample_df, {"product": [["price", "quantity"]]})
    assert out["price_x_quantity"].iloc[1] == 400.0


def test_bin(sample_df):
    out, _ = apply_transformations(sample_df, {"bin": {"price": [-10, 50, 150, 500]}})
    assert "price_bin" in out.columns
    # Three distinct bins represented
    assert out["price_bin"].nunique() >= 2


def test_bin_requires_two_edges(sample_df):
    with pytest.raises(ValueError, match="at least 2 edges"):
        apply_transformations(sample_df, {"bin": {"price": [50]}})


def test_date_parts(sample_df):
    out, report = apply_transformations(sample_df, {"date_parts": ["sale_date"]})
    for suffix in ("year", "month", "day", "dayofweek", "is_weekend"):
        assert f"sale_date_{suffix}" in out.columns
    assert out["sale_date_year"].iloc[0] == 2024
    # "invalid" row → NaN + warning
    assert out["sale_date_year"].isna().sum() == 1
    assert any("unparseable" in w for w in report["warnings"])


def test_date_parts_fails_when_mostly_unparseable():
    df = pd.DataFrame({"d": ["not-a-date"] * 10})
    with pytest.raises(ValueError, match="only .* parsed as dates"):
        apply_transformations(df, {"date_parts": ["d"]})


def test_onehot_drops_source(sample_df):
    out, report = apply_transformations(sample_df, {"onehot": ["city"]})
    assert "city" not in out.columns
    assert "city" in report["dropped_features"]
    # Three distinct cities → three dummy columns
    assert sum(c.startswith("city_") for c in out.columns) == 3


def test_onehot_caps_high_cardinality():
    rng = np.random.RandomState(0)
    n = 500
    df = pd.DataFrame({"id": [f"v{i}" for i in rng.randint(0, 100, n)]})
    out, report = apply_transformations(df, {"onehot": ["id"]})
    dummy_cols = [c for c in out.columns if c.startswith("id_")]
    # Top N + potential _other bucket
    assert len(dummy_cols) <= ONEHOT_TOP_N + 1
    assert any("capped" in w for w in report["warnings"])


def test_target_mean_loo_no_leakage(sample_df):
    out, _ = apply_transformations(sample_df, {"target_mean": {"city": "target"}})
    # City 'a' has targets [10, 30]; LOO for row 0 is 30 (only other 'a')
    assert out["city_target_mean"].iloc[0] == 30
    assert out["city_target_mean"].iloc[2] == 10


def test_interact_top_k(sample_df, tmp_path):
    imp = pd.DataFrame({"importance": [0.5, 0.3, 0.1]}, index=["price", "area", "quantity"])
    imp_path = tmp_path / "feature_importance.csv"
    imp.to_csv(imp_path)
    out, _ = apply_transformations(
        sample_df,
        {"interact_top_k": {"k": 2, "importance_csv": str(imp_path)}},
    )
    assert "price_x_area" in out.columns


def test_original_df_not_mutated(sample_df):
    original = sample_df.copy()
    apply_transformations(sample_df, {"log": ["price"]})
    pd.testing.assert_frame_equal(sample_df, original)


def test_multiple_transforms_compose(sample_df):
    out, report = apply_transformations(
        sample_df,
        {
            "log": ["area"],
            "ratio": [["price", "area"]],
            "date_parts": ["sale_date"],
        },
    )
    assert "log_area" in out.columns
    assert "price_per_area" in out.columns
    assert "sale_date_year" in out.columns
    assert len(report["new_features"]) == 2 + 5  # log + ratio + 5 date parts


def test_tool_engineer_features_writes_csv_and_spec(sample_df, tmp_path):
    csv = tmp_path / "input.csv"
    sample_df.to_csv(csv, index=False)

    result = tool_engineer_features(
        csv_path=str(csv),
        transformations={"log": ["price"]},
        label="target",
        output_dir=str(tmp_path / "out"),
    )

    engineered = pd.read_csv(result["engineered_csv"])
    assert "log_price" in engineered.columns
    assert Path(result["spec_path"]).exists()
    spec = json.loads(Path(result["spec_path"]).read_text())
    assert spec["label"] == "target"
    assert spec["spec"] == {"log": ["price"]}
