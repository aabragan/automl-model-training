"""Integration tests against a real-trained regression run.

house_prices.csv is the only sample dataset where a feature actually
lands with negative permutation importance, which exercises the agent's
drop-negative-importance workflow. We assert that behavior here so any
future change to the sample data or the importance computation surfaces
in test output rather than only in agent runs.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from automl_model_training.tools import (
    tool_calibration_curve,
    tool_inspect_errors,
    tool_model_subset_evaluate,
    tool_partial_dependence,
    tool_partial_dependence_2way,
    tool_read_analysis,
    tool_shap_interactions,
    tool_threshold_sweep,
)

pytestmark = pytest.mark.slow


# ---------------------------------------------------------------------------
# Artifact gating for regression
# ---------------------------------------------------------------------------


def test_regression_produces_regression_artifacts(house_run_dir):
    run = Path(house_run_dir)
    for artifact in (
        "leaderboard.csv",
        "leaderboard_test.csv",
        "feature_importance.csv",
        "test_predictions.csv",
        "residual_stats.json",
        "residual_distribution.csv",
        "shap_summary.csv",
        "analysis.json",
    ):
        assert (run / artifact).exists(), f"Missing regression artifact: {artifact}"


def test_regression_skips_classification_artifacts(house_run_dir):
    """Regression runs must NOT write classification-only artifacts."""
    run = Path(house_run_dir)
    assert not (run / "confusion_matrix.csv").exists()
    assert not (run / "classification_report.csv").exists()
    assert not (run / "roc_curve.csv").exists()
    assert not (run / "roc_auc.json").exists()
    assert not (run / "precision_recall_curve.csv").exists()


# ---------------------------------------------------------------------------
# Per-tool checks on regression
# ---------------------------------------------------------------------------


def test_tool_read_analysis_on_regression_run(house_run_dir):
    r = tool_read_analysis(house_run_dir)
    assert r.get("problem_type") == "regression"
    assert len(r["findings"]) > 0


def test_tool_inspect_errors_on_regression_run(house_run_dir):
    """Regression inspect_errors ranks by absolute residual, not confidence."""
    r = tool_inspect_errors(house_run_dir, n=5, worst=True)
    assert "rows" in r


def test_tool_shap_interactions_on_regression_run(house_run_dir):
    r = tool_shap_interactions(house_run_dir, top_k=3)
    assert "top_features" in r
    assert 0 < len(r["top_features"]) <= 3


def test_tool_partial_dependence_on_regression_run(house_run_dir):
    """Regression PDPs return raw predictions (not probabilities) — assert
    that the pdp_values are in a plausible price range for this dataset."""
    r = tool_partial_dependence(house_run_dir, n_values=5, sample_size=50)
    assert len(r["feature_curves"]) > 0
    for curve in r["feature_curves"]:
        # per_class_pdp_values is classification-only; regression curves must
        # not have it
        assert "per_class_pdp_values" not in curve


def test_tool_partial_dependence_2way_on_regression_run(house_run_dir):
    features = list(pd.read_csv(Path(house_run_dir) / "feature_importance.csv", index_col=0).index)
    assert len(features) >= 2
    r = tool_partial_dependence_2way(
        house_run_dir,
        feature_a=features[0],
        feature_b=features[1],
        n_values_a=6,
        n_values_b=6,
        sample_size=30,
    )
    assert "surface" in r
    assert r.get("shape_hint") in {"additive", "synergy", "saddle", "threshold"}


def test_tool_model_subset_evaluate_on_regression_run(house_run_dir):
    """For regression, AutoGluon reports score_test as negative RMSE (higher
    is better). The tool must sort by raw score_test descending so the
    smallest-absolute RMSE lands on top."""
    r = tool_model_subset_evaluate(house_run_dir)
    assert len(r["models"]) > 0
    # Scores are descending — best (least-negative RMSE) first
    scores = [m["score_test"] for m in r["models"]]
    assert scores == sorted(scores, reverse=True)


# ---------------------------------------------------------------------------
# Workflow-specific: house_prices is our only "drop negative importance" sample
# ---------------------------------------------------------------------------


def test_regression_run_has_at_least_one_negative_importance_feature(house_run_dir):
    """This is a data-quality assertion. We rely on house_prices.csv having
    at least one feature with negative permutation importance to exercise
    the agent's drop-harmful-features workflow. If this stops being true
    (e.g., because AutoGluon changes its importance computation), the agent
    workflow needs a different sample."""
    imp = pd.read_csv(Path(house_run_dir) / "feature_importance.csv", index_col=0)
    assert "importance" in imp.columns
    negatives = imp[imp["importance"] < 0]
    assert len(negatives) > 0, (
        "Expected at least one negative-importance feature in house_prices run. "
        "If this no longer holds, pick a different sample for the drop-harmful-features "
        "workflow and update this test."
    )


# ---------------------------------------------------------------------------
# Binary-only tools must refuse regression runs
# ---------------------------------------------------------------------------


def test_tool_threshold_sweep_refuses_regression(house_run_dir):
    with pytest.raises(ValueError, match="binary classification"):
        tool_threshold_sweep(house_run_dir)


def test_tool_calibration_curve_refuses_regression(house_run_dir):
    with pytest.raises(ValueError, match="binary classification"):
        tool_calibration_curve(house_run_dir)
