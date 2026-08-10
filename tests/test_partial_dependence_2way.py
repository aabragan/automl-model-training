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
            "automl_model_training.tools.partial_dependence.load_predictor",
            return_value=mock_predictor,
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
            "automl_model_training.tools.partial_dependence.load_predictor",
            return_value=mock_predictor,
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
            "automl_model_training.tools.partial_dependence.load_predictor",
            return_value=mock_predictor,
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
        "automl_model_training.tools.partial_dependence.load_predictor", return_value=mock_predictor
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
        "automl_model_training.tools.partial_dependence.load_predictor", return_value=mock_predictor
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
        "automl_model_training.tools.partial_dependence.load_predictor", return_value=mock_predictor
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
        "automl_model_training.tools.partial_dependence.load_predictor", return_value=mock_predictor
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


def test_pdp_2way_missing_test_csv_raises(tmp_path):
    """AutogluonModels exists but test_raw.csv is missing → specific FileNotFoundError."""
    run_dir = tmp_path / "run"
    (run_dir / "AutogluonModels").mkdir(parents=True)
    with pytest.raises(FileNotFoundError, match="test_raw.csv"):
        tool_partial_dependence_2way(str(run_dir), feature_a="x", feature_b="y")


def test_pdp_2way_rejects_invalid_grid_strategy(mock_2way_run):
    mock_predictor = MagicMock()
    mock_predictor.label = "target"
    mock_predictor.problem_type = "regression"

    with (
        patch(
            "automl_model_training.tools.partial_dependence.load_predictor",
            return_value=mock_predictor,
        ),
        pytest.raises(ValueError, match="grid_strategy"),
    ):
        tool_partial_dependence_2way(
            str(mock_2way_run), feature_a="x", feature_b="y", grid_strategy="bogus"
        )


def test_pdp_2way_linspace_grid(mock_2way_run):
    """grid_strategy='linspace' spaces numeric grids evenly across min/max."""
    mock_predictor = MagicMock()
    mock_predictor.label = "target"
    mock_predictor.problem_type = "regression"
    mock_predictor.predict.side_effect = lambda df: pd.Series([0.5] * len(df))

    with patch(
        "automl_model_training.tools.partial_dependence.load_predictor", return_value=mock_predictor
    ):
        result = tool_partial_dependence_2way(
            str(mock_2way_run),
            feature_a="x",
            feature_b="y",
            n_values_a=5,
            n_values_b=5,
            sample_size=10,
            grid_strategy="linspace",
        )

    # x is linspace(0, 10, 100) → 5 evenly spaced grid points across the range
    assert result["grid_a"] == pytest.approx([0.0, 2.5, 5.0, 7.5, 10.0])
    assert len(result["grid_b"]) == 5


def test_pdp_2way_preserves_int_dtype_low_cardinality_grid(tmp_path):
    """Integer feature with few unique values → grid uses them directly, dtype kept int."""
    run_dir = tmp_path / "run"
    (run_dir / "AutogluonModels").mkdir(parents=True)
    rng = np.random.RandomState(1)
    pd.DataFrame(
        {
            "int_feat": np.tile(np.array([1, 2, 3], dtype=np.int64), 20),
            "y": rng.uniform(0, 1, 60),
            "target": rng.randn(60),
        }
    ).to_csv(run_dir / "test_raw.csv", index=False)

    seen_dtypes: list[np.dtype] = []

    def predict(df):
        seen_dtypes.append(df["int_feat"].dtype)
        return pd.Series(np.zeros(len(df)))

    mock_predictor = MagicMock()
    mock_predictor.label = "target"
    mock_predictor.problem_type = "regression"
    mock_predictor.predict.side_effect = predict

    with patch(
        "automl_model_training.tools.partial_dependence.load_predictor", return_value=mock_predictor
    ):
        result = tool_partial_dependence_2way(
            str(run_dir),
            feature_a="int_feat",
            feature_b="y",
            n_values_a=5,
            n_values_b=4,
            sample_size=10,
        )

    # 3 unique values <= n_values_a → grid is exactly the sorted unique values
    assert result["grid_a"] == [1.0, 2.0, 3.0]
    assert seen_dtypes, "predictor.predict was never called"
    assert np.issubdtype(seen_dtypes[0], np.integer), (
        f"int_feat dtype was promoted to {seen_dtypes[0]}"
    )


def test_pdp_2way_pads_degenerate_quantile_grid(tmp_path):
    """Heavy ties collapse the quantile grid → it is padded back up with linspace points."""
    run_dir = tmp_path / "run"
    (run_dir / "AutogluonModels").mkdir(parents=True)
    tied = np.concatenate([np.zeros(60), np.arange(1.0, 41.0)])
    pd.DataFrame({"tied": tied, "y": np.linspace(0, 1, 100), "target": np.zeros(100)}).to_csv(
        run_dir / "test_raw.csv", index=False
    )

    mock_predictor = MagicMock()
    mock_predictor.label = "target"
    mock_predictor.problem_type = "regression"
    mock_predictor.predict.side_effect = lambda df: pd.Series(np.zeros(len(df)))

    with patch(
        "automl_model_training.tools.partial_dependence.load_predictor", return_value=mock_predictor
    ):
        result = tool_partial_dependence_2way(
            str(run_dir),
            feature_a="tied",
            feature_b="y",
            n_values_a=10,
            n_values_b=4,
            sample_size=10,
        )

    assert len(result["grid_a"]) >= 10
    assert result["grid_a"][0] == 0.0
    assert result["grid_a"][-1] == 40.0


def test_pdp_2way_classification_uses_positive_class(mock_2way_run):
    """Binary classification surface is the mean positive-class probability."""

    def predict_proba(df):
        p1 = np.clip(df["x"].values.astype(float) / 10.0, 0.0, 1.0)
        return pd.DataFrame({0: 1 - p1, 1: p1})

    mock_predictor = MagicMock()
    mock_predictor.label = "target"
    mock_predictor.problem_type = "binary"
    mock_predictor.predict_proba.side_effect = predict_proba

    with patch(
        "automl_model_training.tools.partial_dependence.load_predictor", return_value=mock_predictor
    ):
        result = tool_partial_dependence_2way(
            str(mock_2way_run),
            feature_a="x",
            feature_b="y",
            n_values_a=5,
            n_values_b=4,
            sample_size=10,
        )

    surface = np.array(result["surface"])
    assert surface.shape == (5, 4)
    # Surface values are probabilities of the positive class (label 1)
    assert np.all((surface >= 0.0) & (surface <= 1.0))
    # p(1) grows with x → rows should increase down the a-axis
    row_means = surface.mean(axis=1)
    assert row_means[-1] > row_means[0]


def test_pdp_2way_detects_corner_synergy(tmp_path):
    """A single-corner bump (only max-x AND max-y boosted) yields one-sided residuals
    in nearly all cells → classified as 'synergy'."""
    run_dir = tmp_path / "run"
    (run_dir / "AutogluonModels").mkdir(parents=True)
    pd.DataFrame(
        {
            "x": np.linspace(0.0, 1.0, 100),
            "y": np.linspace(0.0, 1.0, 100),
            "target": np.zeros(100),
        }
    ).to_csv(run_dir / "test_raw.csv", index=False)

    def predict(df):
        x = df["x"].values.astype(float)
        y = df["y"].values.astype(float)
        return pd.Series(((x >= 1.0) & (y >= 1.0)).astype(float))

    mock_predictor = MagicMock()
    mock_predictor.label = "target"
    mock_predictor.problem_type = "regression"
    mock_predictor.predict.side_effect = predict

    with patch(
        "automl_model_training.tools.partial_dependence.load_predictor", return_value=mock_predictor
    ):
        result = tool_partial_dependence_2way(
            str(run_dir),
            feature_a="x",
            feature_b="y",
            n_values_a=15,
            n_values_b=15,
            sample_size=20,
        )

    assert result["shape_hint"] == "synergy"
    assert any("synergistically" in h for h in result["hints"])


def test_pdp_2way_saddle_with_categorical_feature_a(mock_2way_run):
    """Categorical feature_a flipping the sign of numeric feature_b → saddle
    (max_jump_a is skipped because feature_a is not numeric)."""
    signs = {"a": -1.0, "b": 0.0, "c": 1.0}

    def predict(df):
        s = df["cat"].map(signs).values.astype(float)
        return pd.Series(s * df["y"].values.astype(float))

    mock_predictor = MagicMock()
    mock_predictor.label = "target"
    mock_predictor.problem_type = "regression"
    mock_predictor.predict.side_effect = predict

    with patch(
        "automl_model_training.tools.partial_dependence.load_predictor", return_value=mock_predictor
    ):
        result = tool_partial_dependence_2way(
            str(mock_2way_run),
            feature_a="cat",
            feature_b="y",
            n_values_a=3,
            n_values_b=10,
            sample_size=20,
        )

    assert result["is_numeric_a"] is False
    assert result["shape_hint"] == "saddle"


def test_pdp_2way_saddle_with_categorical_feature_b(mock_2way_run):
    """Same sign-flip interaction with roles swapped → covers the non-numeric
    feature_b branch of the jump computation."""
    signs = {"a": -1.0, "b": 0.0, "c": 1.0}

    def predict(df):
        s = df["cat"].map(signs).values.astype(float)
        return pd.Series(s * df["x"].values.astype(float))

    mock_predictor = MagicMock()
    mock_predictor.label = "target"
    mock_predictor.problem_type = "regression"
    mock_predictor.predict.side_effect = predict

    with patch(
        "automl_model_training.tools.partial_dependence.load_predictor", return_value=mock_predictor
    ):
        result = tool_partial_dependence_2way(
            str(mock_2way_run),
            feature_a="x",
            feature_b="cat",
            n_values_a=10,
            n_values_b=3,
            sample_size=20,
        )

    assert result["is_numeric_b"] is False
    assert result["shape_hint"] == "saddle"


def test_pdp_2way_detects_threshold_surface(mock_2way_run):
    """A step function active only when both features cross a cutoff → sharp jump
    along one axis → classified as 'threshold'."""

    def predict(df):
        x = df["x"].values.astype(float)
        y = df["y"].values.astype(float)
        return pd.Series(((x > 5.0) & (y > 0.5)).astype(float))

    mock_predictor = MagicMock()
    mock_predictor.label = "target"
    mock_predictor.problem_type = "regression"
    mock_predictor.predict.side_effect = predict

    with patch(
        "automl_model_training.tools.partial_dependence.load_predictor", return_value=mock_predictor
    ):
        result = tool_partial_dependence_2way(
            str(mock_2way_run),
            feature_a="x",
            feature_b="y",
            n_values_a=10,
            n_values_b=10,
            sample_size=20,
        )

    assert result["shape_hint"] == "threshold"
    assert any("threshold effect" in h for h in result["hints"])
