"""Integration tests against a real-trained binary-classification run.

These tests train an actual AutoGluon model on samples/fraud_detection.csv
(via the session-scoped fraud_run_dir fixture) and exercise every tool that
operates on a binary run. This catches library-version drift that
mock-based unit tests cannot — notably the two bugs these tests were
introduced to guard against:

1. shap 0.51 returns binary SHAP as (n_samples, n_features, n_classes).
   Older versions returned (n_classes, n_samples, n_features). The
   unit tests use hand-crafted SHAP arrays that happen to satisfy both
   layouts, so they could not catch the crash. A real-training test
   hitting the same path fails loudly on any future layout change.

2. save_classification_artifacts used to call roc_auc_score without
   multi_class= set, which works on binary but crashes on 3+ classes.
   The binary path should still produce every expected artifact here.

All tests are @pytest.mark.slow and are deselected by default pytest runs.
Invoke via `uv run pytest -m slow tests/integration/`.
"""

from __future__ import annotations

import json
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
# Artifact shape checks — catches bugs that only show up with real training
# ---------------------------------------------------------------------------


def test_binary_run_produces_all_expected_artifacts(fraud_run_dir):
    """Binary runs must write every classification artifact, including the
    binary-only ROC/PR curves that multiclass skips."""
    run = Path(fraud_run_dir)
    for artifact in (
        "leaderboard.csv",
        "leaderboard_test.csv",
        "feature_importance.csv",
        "test_predictions.csv",
        "test_raw.csv",
        "confusion_matrix.csv",
        "classification_report.csv",
        "roc_curve.csv",  # binary-only
        "roc_auc.json",
        "precision_recall_curve.csv",  # binary-only
        "average_precision.json",
        "analysis.json",
        "shap_summary.csv",
        "shap_values.csv",
        "shap_per_row.json",
        "shap_metadata.json",
    ):
        assert (run / artifact).exists(), f"Missing artifact after binary training: {artifact}"


def test_binary_shap_summary_has_one_row_per_feature(fraud_run_dir):
    """Regression test for the shap 0.51 layout bug.

    Before the fix, binary SHAP on shap 0.51 returned (n_samples, n_features,
    n_classes); the code picked shap_values[1] which was
    (n_features, n_classes) — wrong shape. build_shap_summary then
    crashed with 'All arrays must be of the same length'. This test hits
    the exact code path on a real fit and verifies the summary now aligns
    1:1 with the training features.
    """
    shap_summary = pd.read_csv(Path(fraud_run_dir) / "shap_summary.csv")
    feature_importance = pd.read_csv(Path(fraud_run_dir) / "feature_importance.csv", index_col=0)
    # SHAP summary has one row per training feature. Order is feature ranking.
    assert set(shap_summary["feature"]) == set(feature_importance.index)


def test_binary_shap_values_csv_is_wide_not_cube(fraud_run_dir):
    """SHAP values CSV should be (n_samples, n_features) — one column per
    feature, not per-class columns. Guards against reintroducing the cube
    layout bug."""
    shap_values = pd.read_csv(Path(fraud_run_dir) / "shap_values.csv")
    test_raw = pd.read_csv(Path(fraud_run_dir) / "test_raw.csv")
    # Same number of rows as the test set, same number of feature columns
    assert len(shap_values) == len(test_raw)
    assert set(shap_values.columns) == set(test_raw.columns) - {"is_fraud"}


# ---------------------------------------------------------------------------
# Per-tool checks
# ---------------------------------------------------------------------------


def test_tool_read_analysis_on_real_binary_run(fraud_run_dir):
    r = tool_read_analysis(fraud_run_dir)
    assert "findings" in r
    assert len(r["findings"]) > 0
    assert r.get("problem_type") == "binary"


def test_tool_inspect_errors_on_real_binary_run(fraud_run_dir):
    r = tool_inspect_errors(fraud_run_dir, n=5, worst=True)
    assert "rows" in r
    # Fraud is too separable — the test set may legitimately have zero
    # errors. But the rows list must be present and well-formed.
    for row in r["rows"]:
        assert "row_index" in row or "index" in row or "actual" in row


def test_tool_shap_interactions_on_real_binary_run(fraud_run_dir):
    r = tool_shap_interactions(fraud_run_dir, top_k=3)
    assert "pairs" in r
    assert "top_features" in r
    # top_features should be length 3 (or fewer if fewer features)
    assert 0 < len(r["top_features"]) <= 3


def test_tool_partial_dependence_on_real_binary_run(fraud_run_dir):
    r = tool_partial_dependence(fraud_run_dir, n_values=5, sample_size=50)
    assert "feature_curves" in r
    assert len(r["feature_curves"]) > 0
    for curve in r["feature_curves"]:
        assert "feature" in curve
        assert "grid_values" in curve
        assert "pdp_values" in curve
        assert len(curve["grid_values"]) == len(curve["pdp_values"])


def test_tool_partial_dependence_2way_on_real_binary_run(fraud_run_dir):
    # Pick two features from the trained run's feature list
    features = list(pd.read_csv(Path(fraud_run_dir) / "feature_importance.csv", index_col=0).index)
    assert len(features) >= 2
    r = tool_partial_dependence_2way(
        fraud_run_dir,
        feature_a=features[0],
        feature_b=features[1],
        n_values_a=6,
        n_values_b=6,
        sample_size=30,
    )
    assert "surface" in r
    assert len(r["surface"]) == 6 or len(r["surface"]) > 0
    assert "interaction_strength" in r
    assert r.get("shape_hint") in {"additive", "synergy", "saddle", "threshold"}


def test_tool_model_subset_evaluate_on_real_binary_run(fraud_run_dir):
    r = tool_model_subset_evaluate(fraud_run_dir)
    assert "models" in r
    assert len(r["models"]) > 0
    # Best model by score_test (descending, after absolute-value normalization)
    assert r.get("best_model") is not None
    # For binary, score_test is in [0, 1] range (f1 is the default binary metric)
    for m in r["models"]:
        assert 0 <= abs(m["score_test"]) <= 1


def test_tool_threshold_sweep_on_real_binary_run(fraud_run_dir):
    r = tool_threshold_sweep(fraud_run_dir, n_thresholds=21)
    # All five metrics present with expected shape
    for metric in ("f1", "precision", "recall", "mcc", "balanced_accuracy"):
        assert metric in r["curves"]
        assert len(r["curves"][metric]) == 21
        assert metric in r["best"]
        assert 0 < r["best"][metric]["threshold"] < 1


def test_tool_calibration_curve_on_real_binary_run(fraud_run_dir):
    r = tool_calibration_curve(fraud_run_dir, n_bins=5)
    assert r["direction"] in {"over_confident", "under_confident", "well_calibrated", "mixed"}
    assert 0 <= r["ece"] <= 1
    assert r["n_samples"] > 0
    # Each bin must have the full shape regardless of emptiness
    for b in r["bins"]:
        assert "prob_range" in b
        assert "n_samples" in b


def test_binary_roc_auc_json_has_scalar_auc(fraud_run_dir):
    """Binary runs write roc_auc.json with a scalar roc_auc field.
    Guards against the multiclass-only format leaking into binary."""
    data = json.loads((Path(fraud_run_dir) / "roc_auc.json").read_text())
    assert "roc_auc" in data
    assert 0 <= data["roc_auc"] <= 1
    # Should NOT have the multiclass-only key
    assert "roc_auc_macro_ovr" not in data
