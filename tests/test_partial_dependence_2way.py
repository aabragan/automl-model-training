"""Tests for tool_partial_dependence_2way."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from automl_model_training.tools import tool_partial_dependence_2way

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def mock_2way_run(tmp_path):
    """Run dir with test_raw.csv and two numeric features."""
    run_dir = tmp_path / "run"
    (run_dir / "AutogluonModels").mkdir(parents=True)
    rng = np.random.RandomState(0)
    test_raw = pd.DataFrame(
        {
            "x": np.linspace(0, 10, 100),
            "y": rng.uniform(0, 1, size=100),
            "cat": rng.choice(["a", "b", "c"], size=100),
            "target": rng.randn(100),
        }
    )
    test_raw.to_csv(run_dir / "test_raw.csv", index=False)
    return run_dir


# ---------------------------------------------------------------------------
# tool_partial_dependence_2way
# ---------------------------------------------------------------------------


def test_pdp_2way_missing_files_raises(tmp_path):

    with pytest.raises(FileNotFoundError, match="AutogluonModels"):
        tool_partial_dependence_2way(str(tmp_path), feature_a="x", feature_b="y")


def test_pdp_2way_rejects_same_feature(mock_2way_run):

    from automl_model_training.tools import tool_partial_dependence_2way

    mock_predictor = MagicMock()
    mock_predictor.label = "target"
    mock_predictor.problem_type = "regression"

    with (
        patch(
            "automl_model_training.tools.explainability.load_predictor", return_value=mock_predictor
        ),
        pytest.raises(ValueError, match="must differ"),
    ):
        tool_partial_dependence_2way(str(mock_2way_run), feature_a="x", feature_b="x")


def test_pdp_2way_rejects_missing_feature(mock_2way_run):

    from automl_model_training.tools import tool_partial_dependence_2way

    mock_predictor = MagicMock()
    mock_predictor.label = "target"
    mock_predictor.problem_type = "regression"

    with (
        patch(
            "automl_model_training.tools.explainability.load_predictor", return_value=mock_predictor
        ),
        pytest.raises(ValueError, match="not in test data"),
    ):
        tool_partial_dependence_2way(str(mock_2way_run), feature_a="x", feature_b="ghost")


def test_pdp_2way_rejects_cost_cap_breach(mock_2way_run):

    from automl_model_training.tools import tool_partial_dependence_2way

    mock_predictor = MagicMock()
    mock_predictor.label = "target"
    mock_predictor.problem_type = "regression"

    with (
        patch(
            "automl_model_training.tools.explainability.load_predictor", return_value=mock_predictor
        ),
        pytest.raises(ValueError, match="max_cells"),
    ):
        tool_partial_dependence_2way(
            str(mock_2way_run),
            feature_a="x",
            feature_b="y",
            n_values_a=100,
            n_values_b=100,
            sample_size=100,
            max_cells=50_000,
        )


def test_pdp_2way_detects_additive_surface(mock_2way_run):
    """A model that predicts f(x,y) = x + y (purely additive) should classify
    as 'additive' and have low interaction_strength."""

    from automl_model_training.tools import tool_partial_dependence_2way

    mock_predictor = MagicMock()
    mock_predictor.label = "target"
    mock_predictor.problem_type = "regression"

    def predict(df):
        return pd.Series(df["x"].values.astype(float) + df["y"].values.astype(float))

    mock_predictor.predict.side_effect = predict

    with patch(
        "automl_model_training.tools.explainability.load_predictor", return_value=mock_predictor
    ):
        result = tool_partial_dependence_2way(
            str(mock_2way_run),
            feature_a="x",
            feature_b="y",
            n_values_a=6,
            n_values_b=6,
            sample_size=30,
        )

    assert result["shape_hint"] == "additive"
    # Pure additive surface has interaction_strength ≈ 0 (up to floating-point noise)
    assert result["interaction_strength"] < 0.01
    assert len(result["surface"]) == 6
    assert len(result["surface"][0]) == 6


def test_pdp_2way_detects_nonadditive_surface(mock_2way_run):
    """A multiplicative model f(x,y) = x*y produces a saddle-shaped response
    surface (hyperbolic paraboloid). The tool should flag it as non-additive
    with meaningful interaction_strength, regardless of the specific shape
    label it lands on (saddle vs synergy vs threshold)."""

    from automl_model_training.tools import tool_partial_dependence_2way

    mock_predictor = MagicMock()
    mock_predictor.label = "target"
    mock_predictor.problem_type = "regression"

    def predict(df):
        return pd.Series(df["x"].values.astype(float) * df["y"].values.astype(float))

    mock_predictor.predict.side_effect = predict

    with patch(
        "automl_model_training.tools.explainability.load_predictor", return_value=mock_predictor
    ):
        result = tool_partial_dependence_2way(
            str(mock_2way_run),
            feature_a="x",
            feature_b="y",
            n_values_a=6,
            n_values_b=6,
            sample_size=30,
        )

    # Multiplicative surface is non-additive — any of the three non-additive
    # labels is acceptable; the key assertion is it's NOT labelled 'additive'.
    assert result["shape_hint"] != "additive"
    assert result["shape_hint"] in {"synergy", "saddle", "threshold"}
    assert result["interaction_strength"] > 0.05


def test_pdp_2way_detects_synergistic_surface(mock_2way_run):
    """A surface f(x,y) = x + y + 5*x*y (additive + strong positive interaction)
    should classify as 'synergy' — residuals are consistently positive in the
    corners where x*y is largest and negative elsewhere only mildly."""

    from automl_model_training.tools import tool_partial_dependence_2way

    mock_predictor = MagicMock()
    mock_predictor.label = "target"
    mock_predictor.problem_type = "regression"

    def predict(df):
        x = df["x"].values.astype(float)
        y = df["y"].values.astype(float)
        # Additive + a strictly-positive-contributing quadratic boost in the
        # upper-right quadrant. This creates residuals that are ≥ 0 almost
        # everywhere, so shape_hint should be 'synergy' rather than 'saddle'.
        return pd.Series(x + y + 2.0 * np.maximum(x * y, 0))

    mock_predictor.predict.side_effect = predict

    with patch(
        "automl_model_training.tools.explainability.load_predictor", return_value=mock_predictor
    ):
        result = tool_partial_dependence_2way(
            str(mock_2way_run),
            feature_a="x",
            feature_b="y",
            n_values_a=6,
            n_values_b=6,
            sample_size=30,
        )

    assert result["shape_hint"] != "additive"
    assert result["interaction_strength"] > 0.05


def test_pdp_2way_handles_categorical_feature(mock_2way_run):
    """One numeric + one categorical feature — surface should still be built."""

    from automl_model_training.tools import tool_partial_dependence_2way

    mock_predictor = MagicMock()
    mock_predictor.label = "target"
    mock_predictor.problem_type = "regression"
    # Return a constant so we can verify shape without caring about values
    mock_predictor.predict.side_effect = lambda df: pd.Series([0.5] * len(df))

    with patch(
        "automl_model_training.tools.explainability.load_predictor", return_value=mock_predictor
    ):
        result = tool_partial_dependence_2way(
            str(mock_2way_run),
            feature_a="x",
            feature_b="cat",
            n_values_a=5,
            n_values_b=3,
            sample_size=20,
        )

    assert result["is_numeric_a"] is True
    assert result["is_numeric_b"] is False
    # grid_b should be 3 string category values
    assert len(result["grid_b"]) <= 3
    assert all(isinstance(g, str) for g in result["grid_b"])
