"""Integration tests against a real-trained multiclass run.

Regression coverage for the roc_auc_score multiclass fix:
save_classification_artifacts used to call roc_auc_score(y_true, y_proba[c])
without multi_class=, which crashed with 'multi_class must be in (ovo, ovr)'.
The fix gates the binary-only ROC/PR curves behind len(labels)==2 and
computes a scalar macro-OVR AUC for multiclass. These tests verify both
sides of that gate on real training output.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from automl_model_training.tools import (
    tool_calibration_curve,
    tool_inspect_errors,
    tool_model_subset_evaluate,
    tool_partial_dependence,
    tool_read_analysis,
    tool_shap_interactions,
    tool_threshold_sweep,
)

pytestmark = pytest.mark.slow


# ---------------------------------------------------------------------------
# Artifact gating: what multiclass runs DO and DO NOT produce
# ---------------------------------------------------------------------------


def test_multiclass_produces_classification_artifacts(flower_run_dir):
    """Multiclass must write the classification artifacts that make sense for
    multiclass: confusion matrix, per-class classification report, scalar
    ROC AUC in the macro-OVR form."""
    run = Path(flower_run_dir)
    for artifact in (
        "leaderboard.csv",
        "leaderboard_test.csv",
        "feature_importance.csv",
        "test_predictions.csv",
        "confusion_matrix.csv",
        "classification_report.csv",
        "roc_auc.json",
        "shap_summary.csv",
    ):
        assert (run / artifact).exists(), f"Missing multiclass artifact: {artifact}"


def test_multiclass_skips_binary_only_artifacts(flower_run_dir):
    """ROC curve and PR curve files are binary-only concepts; they must NOT
    be written for multiclass runs. This is the direct regression test for
    the bug where AutoGluon called binary-only sklearn functions on a
    multiclass problem and crashed."""
    run = Path(flower_run_dir)
    # These files are binary-only — they must not exist on a multiclass run
    assert not (run / "roc_curve.csv").exists()
    assert not (run / "precision_recall_curve.csv").exists()
    assert not (run / "average_precision.json").exists()


def test_multiclass_roc_auc_json_has_macro_ovr(flower_run_dir):
    """Multiclass roc_auc.json has the scalar macro-OVR summary, not the
    binary-only 'roc_auc' + 'pos_label' pair."""
    data = json.loads((Path(flower_run_dir) / "roc_auc.json").read_text())
    assert "roc_auc_macro_ovr" in data
    assert 0 <= data["roc_auc_macro_ovr"] <= 1
    assert data.get("n_classes", 0) >= 3
    # And must not contain the binary-format keys
    assert "roc_auc" not in data
    assert "pos_label" not in data


# ---------------------------------------------------------------------------
# Per-tool checks on multiclass
# ---------------------------------------------------------------------------


def test_tool_read_analysis_on_multiclass_run(flower_run_dir):
    r = tool_read_analysis(flower_run_dir)
    assert r.get("problem_type") == "multiclass"
    assert len(r["findings"]) > 0


def test_tool_inspect_errors_on_multiclass_run(flower_run_dir):
    r = tool_inspect_errors(flower_run_dir, n=5, worst=True)
    assert "rows" in r


def test_tool_shap_interactions_on_multiclass_run(flower_run_dir):
    # shap on multiclass returns per-class arrays; this exercises the
    # averaging branch in build_shap_summary. A real run catches any
    # dimension bug.
    r = tool_shap_interactions(flower_run_dir, top_k=3)
    assert "pairs" in r
    assert "top_features" in r


def test_tool_partial_dependence_on_multiclass_run(flower_run_dir):
    # PDP on multiclass returns per_class_pdp_values in addition to the
    # overall curve (positive-class convention). Verify both are present.
    r = tool_partial_dependence(flower_run_dir, n_values=5, sample_size=50)
    assert len(r["feature_curves"]) > 0
    for curve in r["feature_curves"]:
        # per_class_pdp_values is only populated for numeric features —
        # the flower dataset is all-numeric, so every curve should have it
        if curve["is_numeric"]:
            assert "per_class_pdp_values" in curve
            assert len(curve["per_class_pdp_values"]) >= 2


def test_tool_model_subset_evaluate_on_multiclass_run(flower_run_dir):
    r = tool_model_subset_evaluate(flower_run_dir)
    assert len(r["models"]) > 0
    # All classes should be trained ⇒ at least one ensemble model
    assert any(m["is_ensemble"] for m in r["models"])


# ---------------------------------------------------------------------------
# Binary-only tools must refuse multiclass runs
# ---------------------------------------------------------------------------


def test_tool_threshold_sweep_refuses_multiclass(flower_run_dir):
    with pytest.raises(ValueError, match="binary classification"):
        tool_threshold_sweep(flower_run_dir)


def test_tool_calibration_curve_refuses_multiclass(flower_run_dir):
    with pytest.raises(ValueError, match="binary classification"):
        tool_calibration_curve(flower_run_dir)
