"""Tests for tool_partial_dependence (1-way)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from automl_model_training.tools import tool_partial_dependence


@pytest.fixture()
def mock_pdp_run(tmp_path):
    """A run_dir with test_raw.csv and an AutogluonModels placeholder directory."""
    run_dir = tmp_path / "run"
    (run_dir / "AutogluonModels").mkdir(parents=True)
    test_raw = pd.DataFrame(
        {
            "feat_numeric": np.linspace(0, 100, 50),
            "feat_category": ["a"] * 25 + ["b"] * 25,
            "target": np.linspace(0, 10, 50),
        }
    )
    test_raw.to_csv(run_dir / "test_raw.csv", index=False)
    importance = pd.DataFrame({"importance": [0.5, 0.3]}, index=["feat_numeric", "feat_category"])
    importance.to_csv(run_dir / "feature_importance.csv")
    return run_dir


def test_partial_dependence_missing_files_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        tool_partial_dependence(str(tmp_path / "nonexistent"))


def test_partial_dependence_detects_monotonic(mock_pdp_run):
    """A model whose prediction increases linearly with the feature → monotonic hint."""
    mock_predictor = MagicMock()
    mock_predictor.label = "target"
    mock_predictor.problem_type = "regression"

    # Simulate a regression model where prediction = feat_numeric (perfectly monotonic)
    def predict(df):
        return pd.Series(df["feat_numeric"].values)

    mock_predictor.predict.side_effect = predict

    with patch(
        "automl_model_training.tools.explainability.load_predictor", return_value=mock_predictor
    ):
        result = tool_partial_dependence(
            str(mock_pdp_run), features=["feat_numeric"], sample_size=20
        )

    assert len(result["feature_curves"]) == 1
    curve = result["feature_curves"][0]
    assert curve["feature"] == "feat_numeric"
    assert curve["is_numeric"] is True
    assert len(curve["pdp_values"]) == 20
    assert any("monotonically increasing" in h for h in result["hints"])


def test_partial_dependence_handles_categorical(mock_pdp_run):
    mock_predictor = MagicMock()
    mock_predictor.label = "target"
    mock_predictor.problem_type = "regression"
    mock_predictor.predict.side_effect = lambda df: pd.Series([0.5] * len(df))

    with patch(
        "automl_model_training.tools.explainability.load_predictor", return_value=mock_predictor
    ):
        result = tool_partial_dependence(
            str(mock_pdp_run), features=["feat_category"], sample_size=20
        )

    curve = result["feature_curves"][0]
    assert curve["is_numeric"] is False
    assert set(curve["grid_values"]) <= {"a", "b"}


def test_partial_dependence_rejects_missing_feature(mock_pdp_run):
    mock_predictor = MagicMock()
    mock_predictor.label = "target"
    mock_predictor.problem_type = "regression"

    with (
        patch(
            "automl_model_training.tools.explainability.load_predictor", return_value=mock_predictor
        ),
        pytest.raises(ValueError, match="Features not in test data"),
    ):
        tool_partial_dependence(str(mock_pdp_run), features=["ghost_feature"])


def test_partial_dependence_rejects_invalid_grid_strategy(mock_pdp_run):
    mock_predictor = MagicMock()
    mock_predictor.label = "target"
    mock_predictor.problem_type = "regression"
    mock_predictor.predict.side_effect = lambda df: pd.Series([0.5] * len(df))

    with (
        patch(
            "automl_model_training.tools.explainability.load_predictor", return_value=mock_predictor
        ),
        pytest.raises(ValueError, match="grid_strategy"),
    ):
        tool_partial_dependence(
            str(mock_pdp_run),
            features=["feat_numeric"],
            grid_strategy="bogus",
        )


def test_partial_dependence_quantile_grid_concentrates_on_dense_regions(tmp_path):
    """Quantile grid should place more points where the data is dense, not evenly."""
    run_dir = tmp_path / "run"
    (run_dir / "AutogluonModels").mkdir(parents=True)
    # Heavily skewed feature: 90% of mass in [0, 1], long tail to 1000
    skewed = np.concatenate(
        [
            np.random.RandomState(0).uniform(0, 1, size=90),
            np.random.RandomState(0).uniform(100, 1000, size=10),
        ]
    )
    pd.DataFrame({"x": skewed, "target": skewed}).to_csv(run_dir / "test_raw.csv", index=False)

    mock_predictor = MagicMock()
    mock_predictor.label = "target"
    mock_predictor.problem_type = "regression"
    mock_predictor.predict.side_effect = lambda df: pd.Series(df["x"].values)

    with patch(
        "automl_model_training.tools.explainability.load_predictor", return_value=mock_predictor
    ):
        q_result = tool_partial_dependence(
            str(run_dir),
            features=["x"],
            sample_size=20,
            n_values=10,
            grid_strategy="quantile",
        )
        l_result = tool_partial_dependence(
            str(run_dir),
            features=["x"],
            sample_size=20,
            n_values=10,
            grid_strategy="linspace",
        )

    q_grid = q_result["feature_curves"][0]["grid_values"]
    l_grid = l_result["feature_curves"][0]["grid_values"]

    # Both grids should bracket the same min/max, but the quantile grid must place
    # strictly more points below the median of the data than the linspace grid does.
    # (Linspace spacing ≈ 100, quantile spacing near the mode ≈ 0.1.)
    median = float(np.median(skewed))
    q_below = sum(1 for v in q_grid if v <= median)
    l_below = sum(1 for v in l_grid if v <= median)
    assert q_below > l_below, (
        f"Quantile should concentrate in dense region; got q={q_below} l={l_below}"
    )


def test_partial_dependence_returns_pdp_std(mock_pdp_run):
    """pdp_std is the spread across sample rows at each grid point."""
    mock_predictor = MagicMock()
    mock_predictor.label = "target"
    mock_predictor.problem_type = "regression"
    # Predictions that vary across rows but not across grid → std > 0, PDP flat
    rng = np.random.RandomState(42)
    mock_predictor.predict.side_effect = lambda df: pd.Series(rng.uniform(0, 1, size=len(df)))

    with patch(
        "automl_model_training.tools.explainability.load_predictor", return_value=mock_predictor
    ):
        result = tool_partial_dependence(
            str(mock_pdp_run), features=["feat_numeric"], sample_size=30
        )

    curve = result["feature_curves"][0]
    assert "pdp_std" in curve
    assert len(curve["pdp_std"]) == len(curve["pdp_values"])
    # With random predictions, std across 30 samples must be positive
    assert all(s > 0 for s in curve["pdp_std"])


def test_partial_dependence_returns_ice_when_requested(mock_pdp_run):
    mock_predictor = MagicMock()
    mock_predictor.label = "target"
    mock_predictor.problem_type = "regression"
    mock_predictor.predict.side_effect = lambda df: pd.Series(df["feat_numeric"].values)

    with patch(
        "automl_model_training.tools.explainability.load_predictor", return_value=mock_predictor
    ):
        result = tool_partial_dependence(
            str(mock_pdp_run),
            features=["feat_numeric"],
            sample_size=15,
            n_values=10,
            return_ice=True,
        )

    curve = result["feature_curves"][0]
    assert "ice_values" in curve
    assert len(curve["ice_values"]) == len(curve["grid_values"])  # n_values rows
    assert all(len(row) == 15 for row in curve["ice_values"])  # sample_size cols


def test_partial_dependence_omits_ice_by_default(mock_pdp_run):
    mock_predictor = MagicMock()
    mock_predictor.label = "target"
    mock_predictor.problem_type = "regression"
    mock_predictor.predict.side_effect = lambda df: pd.Series([0.5] * len(df))

    with patch(
        "automl_model_training.tools.explainability.load_predictor", return_value=mock_predictor
    ):
        result = tool_partial_dependence(str(mock_pdp_run), features=["feat_numeric"])

    assert "ice_values" not in result["feature_curves"][0]


def test_partial_dependence_returns_per_class_for_multiclass(tmp_path):
    """Multiclass classification returns per_class_pdp_values with a curve per class."""
    run_dir = tmp_path / "run"
    (run_dir / "AutogluonModels").mkdir(parents=True)
    pd.DataFrame(
        {
            "x": np.linspace(0, 1, 40),
            "target": np.tile(["A", "B", "C", "D"], 10),
        }
    ).to_csv(run_dir / "test_raw.csv", index=False)

    mock_predictor = MagicMock()
    mock_predictor.label = "target"
    mock_predictor.problem_type = "multiclass"

    # Each class probability varies with x differently; columns sum to 1 per row
    def predict_proba(df):
        x = df["x"].values.astype(float)
        p_a = 0.4 - 0.3 * x
        p_b = 0.3
        p_c = 0.2
        p_d = 1.0 - (p_a + p_b + p_c)
        return pd.DataFrame({"A": p_a, "B": p_b, "C": p_c, "D": p_d})

    mock_predictor.predict_proba.side_effect = predict_proba

    with patch(
        "automl_model_training.tools.explainability.load_predictor", return_value=mock_predictor
    ):
        result = tool_partial_dependence(str(run_dir), features=["x"], sample_size=10, n_values=5)

    curve = result["feature_curves"][0]
    assert "per_class_pdp_values" in curve
    assert set(curve["per_class_pdp_values"].keys()) == {"A", "B", "C", "D"}
    for class_curve in curve["per_class_pdp_values"].values():
        assert len(class_curve) == len(curve["grid_values"])
    # Overall pdp_values uses the highest-sorted class ("D") — should rise with x
    pdp = curve["pdp_values"]
    assert pdp[-1] > pdp[0], f"pdp for class D should increase with x; got {pdp}"


def test_partial_dependence_preserves_int_dtype_when_grid_is_integer(mock_pdp_run):
    """Integer features whose grid is all-integer should not be promoted to float.

    Uses a low-cardinality int feature so the grid-construction code takes the
    'use unique values directly' path (no quantile interpolation → still ints).
    """
    test_raw = pd.DataFrame(
        {
            # 5 distinct int values repeated → unique_vals (5) <= n_values (5),
            # so the grid is exactly np.sort(unique_vals) — all integers.
            "int_feat": np.tile(np.array([1, 2, 3, 4, 5], dtype=np.int64), 10),
            "target": np.arange(50, dtype=float),
        }
    )
    test_raw.to_csv(mock_pdp_run / "test_raw.csv", index=False)

    seen_dtypes: list[np.dtype] = []

    def predict(df):
        seen_dtypes.append(df["int_feat"].dtype)
        return pd.Series(df["int_feat"].values.astype(float))

    mock_predictor = MagicMock()
    mock_predictor.label = "target"
    mock_predictor.problem_type = "regression"
    mock_predictor.predict.side_effect = predict

    with patch(
        "automl_model_training.tools.explainability.load_predictor", return_value=mock_predictor
    ):
        tool_partial_dependence(
            str(mock_pdp_run),
            features=["int_feat"],
            sample_size=10,
            n_values=5,
        )

    assert seen_dtypes, "predictor.predict was never called"
    assert np.issubdtype(seen_dtypes[0], np.integer), (
        f"int_feat dtype was promoted to {seen_dtypes[0]}"
    )
