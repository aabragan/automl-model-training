"""Tests for tier-2 LLM tools: deep_profile, shap_interactions, partial_dependence, tune_model."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from automl_model_training.tools import (
    tool_deep_profile,
    tool_partial_dependence,
    tool_shap_interactions,
    tool_tune_model,
)

# ---------------------------------------------------------------------------
# tool_deep_profile
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# tool_shap_interactions
# ---------------------------------------------------------------------------


def test_shap_interactions_missing_files_raises(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    with pytest.raises(FileNotFoundError, match="shap_values"):
        tool_shap_interactions(str(run_dir))


def test_shap_interactions_ranks_correlated_pairs(tmp_path):
    """Two highly-correlated SHAP columns should top the pairs list."""
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    rng = np.random.RandomState(0)
    n = 100
    base = rng.randn(n)
    shap_df = pd.DataFrame(
        {
            "feat_a": base + rng.randn(n) * 0.05,  # Highly correlated with base
            "feat_b": base + rng.randn(n) * 0.05,  # Highly correlated with base
            "feat_c": rng.randn(n),  # Independent
            "feat_d": rng.randn(n),
            "feat_e": rng.randn(n),
        }
    )
    shap_df.to_csv(run_dir / "shap_values.csv", index=False)

    summary = pd.DataFrame(
        {
            "feature": ["feat_a", "feat_b", "feat_c", "feat_d", "feat_e"],
            "mean_abs_shap": [1.0, 0.9, 0.5, 0.3, 0.1],
        }
    )
    summary.to_csv(run_dir / "shap_summary.csv", index=False)

    result = tool_shap_interactions(str(run_dir), top_k=5)
    # Top pair should be feat_a/feat_b with high correlation
    top_pair = result["pairs"][0]
    assert {top_pair["feature_a"], top_pair["feature_b"]} == {"feat_a", "feat_b"}
    assert top_pair["abs_corr"] > 0.9
    # Hints should flag the correlated pair
    assert any("redundant" in h or "correlated" in h for h in result["hints"])


def test_shap_interactions_handles_few_top_features(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    pd.DataFrame({"only_one": [0.1, 0.2]}).to_csv(run_dir / "shap_values.csv", index=False)
    pd.DataFrame({"feature": ["only_one"], "mean_abs_shap": [1.0]}).to_csv(
        run_dir / "shap_summary.csv", index=False
    )

    result = tool_shap_interactions(str(run_dir), top_k=5)
    assert result["pairs"] == []
    assert any("Fewer than 2" in h for h in result["hints"])


# ---------------------------------------------------------------------------
# tool_partial_dependence
# ---------------------------------------------------------------------------


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

    with patch("automl_model_training.tools.load_predictor", return_value=mock_predictor):
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

    with patch("automl_model_training.tools.load_predictor", return_value=mock_predictor):
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
        patch("automl_model_training.tools.load_predictor", return_value=mock_predictor),
        pytest.raises(ValueError, match="Features not in test data"),
    ):
        tool_partial_dependence(str(mock_pdp_run), features=["ghost_feature"])


# ---------------------------------------------------------------------------
# tool_tune_model
# ---------------------------------------------------------------------------


def test_tune_model_rejects_unknown_family(tmp_path):
    csv = tmp_path / "d.csv"
    pd.DataFrame({"x": [1, 2, 3], "y": [0, 1, 0]}).to_csv(csv, index=False)
    with pytest.raises(ValueError, match="model_family"):
        tool_tune_model(
            csv_path=str(csv),
            label="y",
            model_family="NONSENSE",
            n_trials=2,
            time_limit=10,
        )


def test_tune_model_passes_hyperparameter_kwargs_to_train(tmp_path):
    """Verify tune_model calls train_and_evaluate with hyperparameter_tune_kwargs."""
    csv = tmp_path / "d.csv"
    pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0], "y": [0, 1, 0, 1]}).to_csv(csv, index=False)

    with (
        patch("automl_model_training.tools.train_and_evaluate") as mock_train,
        patch("automl_model_training.tools.load_and_prepare") as mock_load,
    ):
        mock_load.return_value = (
            pd.DataFrame({"x": [1.0], "y": [0]}),
            pd.DataFrame({"x": [2.0], "y": [1]}),
            None,
            None,
            [],
        )
        mock_train.return_value = None

        tool_tune_model(
            csv_path=str(csv),
            label="y",
            model_family="GBM",
            n_trials=5,
            time_limit=60,
            output_dir=str(tmp_path / "out"),
        )

        # Verify train_and_evaluate was called with the right kwargs
        mock_train.assert_called_once()
        kwargs = mock_train.call_args.kwargs
        assert kwargs["hyperparameters"] == {"GBM": {}}
        assert kwargs["hyperparameter_tune_kwargs"]["num_trials"] == 5
        assert kwargs["time_limit"] == 60
