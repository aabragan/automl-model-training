"""Tests for evaluate.analyze — post-training analysis and recommendations."""

import json
from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd

from automl_model_training.evaluate.analyze import _model_family, analyze_and_recommend

# ---- Helpers ----


def _make_predictor(
    problem_type: str = "binary",
    label: str = "target",
    best_model: str = "LightGBM",
    features: list[str] | None = None,
) -> MagicMock:
    pred = MagicMock()
    pred.label = label
    pred.problem_type = problem_type
    pred.eval_metric = "f1"
    pred.model_best = best_model
    pred.features.return_value = features or ["feat_a", "feat_b"]
    return pred


def _make_leaderboards(
    best_model: str = "LightGBM",
    val_score: float = 0.90,
    test_score: float = 0.88,
    n_models: int = 8,
):
    models = [best_model] + [f"Model_{i}" for i in range(1, n_models)]
    lb = pd.DataFrame(
        {
            "model": models,
            "score_val": [val_score] + [val_score - 0.01 * i for i in range(1, n_models)],
            "fit_time": [10.0] * n_models,
            "pred_time_val": [0.1] * n_models,
        }
    )
    test_lb = pd.DataFrame(
        {
            "model": models,
            "score_test": [test_score] + [test_score - 0.01 * i for i in range(1, n_models)],
        }
    )
    return lb, test_lb


def _make_importance(features: list[str], values: list[float]):
    return pd.DataFrame({"importance": values}, index=features)


# ---- Tests ----


def test_no_issues_produces_positive_report(tmp_path: Path):
    pred = _make_predictor()
    lb, test_lb = _make_leaderboards(val_score=0.90, test_score=0.89)
    imp = _make_importance(["feat_a", "feat_b"], [0.15, 0.10])
    train = pd.DataFrame({"feat_a": range(500), "feat_b": range(500), "target": [0, 1] * 250})
    test = pd.DataFrame({"feat_a": range(150), "feat_b": range(150), "target": [0, 1] * 75})

    result = analyze_and_recommend(pred, train, test, lb, test_lb, imp, tmp_path)

    assert "Results look solid" in result["recommendations"][0]
    assert (tmp_path / "analysis.json").exists()
    assert (tmp_path / "analysis_report.txt").exists()


def test_overfitting_detected(tmp_path: Path):
    pred = _make_predictor()
    lb, test_lb = _make_leaderboards(val_score=0.95, test_score=0.80)  # 15.8% gap
    imp = _make_importance(["feat_a", "feat_b"], [0.15, 0.10])
    train = pd.DataFrame({"feat_a": range(200), "feat_b": range(200), "target": [0, 1] * 100})
    test = pd.DataFrame({"feat_a": range(50), "feat_b": range(50), "target": [0, 1] * 25})

    result = analyze_and_recommend(pred, train, test, lb, test_lb, imp, tmp_path)

    recs = " ".join(result["recommendations"])
    assert "overfitting" in recs.lower()


def test_moderate_gap_warning(tmp_path: Path):
    pred = _make_predictor()
    lb, test_lb = _make_leaderboards(val_score=0.90, test_score=0.84)  # ~6.7% gap
    imp = _make_importance(["feat_a", "feat_b"], [0.15, 0.10])
    train = pd.DataFrame({"feat_a": range(200), "feat_b": range(200), "target": [0, 1] * 100})
    test = pd.DataFrame({"feat_a": range(50), "feat_b": range(50), "target": [0, 1] * 25})

    result = analyze_and_recommend(pred, train, test, lb, test_lb, imp, tmp_path)

    recs = " ".join(result["recommendations"])
    assert "moderate" in recs.lower() or "monitor" in recs.lower()


def test_few_models_recommendation(tmp_path: Path):
    pred = _make_predictor()
    lb, test_lb = _make_leaderboards(n_models=3)
    imp = _make_importance(["feat_a", "feat_b"], [0.15, 0.10])
    train = pd.DataFrame({"feat_a": range(200), "feat_b": range(200), "target": [0, 1] * 100})
    test = pd.DataFrame({"feat_a": range(50), "feat_b": range(50), "target": [0, 1] * 25})

    result = analyze_and_recommend(pred, train, test, lb, test_lb, imp, tmp_path)

    recs = " ".join(result["recommendations"])
    assert "3 models" in recs or "time-limit" in recs.lower()


def test_low_importance_features_flagged(tmp_path: Path):
    pred = _make_predictor(features=["feat_a", "feat_b", "feat_c"])
    lb, test_lb = _make_leaderboards()
    imp = _make_importance(["feat_a", "feat_b", "feat_c"], [0.15, 0.0005, -0.02])
    train = pd.DataFrame(
        {
            "feat_a": range(200),
            "feat_b": range(200),
            "feat_c": range(200),
            "target": [0, 1] * 100,
        }
    )
    test = pd.DataFrame(
        {
            "feat_a": range(50),
            "feat_b": range(50),
            "feat_c": range(50),
            "target": [0, 1] * 25,
        }
    )

    result = analyze_and_recommend(pred, train, test, lb, test_lb, imp, tmp_path)

    recs = " ".join(result["recommendations"])
    assert "near-zero" in recs.lower() or "dropping" in recs.lower()
    assert "negative" in recs.lower() or "hurt" in recs.lower()


def test_class_imbalance_severe(tmp_path: Path):
    pred = _make_predictor()
    lb, test_lb = _make_leaderboards()
    imp = _make_importance(["feat_a", "feat_b"], [0.15, 0.10])
    # 95/5 split → 19:1 ratio
    train = pd.DataFrame(
        {
            "feat_a": range(200),
            "feat_b": range(200),
            "target": [0] * 190 + [1] * 10,
        }
    )
    test = pd.DataFrame({"feat_a": range(50), "feat_b": range(50), "target": [0, 1] * 25})

    result = analyze_and_recommend(pred, train, test, lb, test_lb, imp, tmp_path)

    recs = " ".join(result["recommendations"])
    assert "imbalance" in recs.lower()


def test_small_dataset_warning(tmp_path: Path):
    pred = _make_predictor(features=[f"f{i}" for i in range(50)])
    lb, test_lb = _make_leaderboards()
    imp = _make_importance([f"f{i}" for i in range(50)], [0.01] * 50)
    # 100 train rows, 50 features → ratio = 2x (< 10x threshold)
    train = pd.DataFrame({f"f{i}": range(100) for i in range(50)} | {"target": [0, 1] * 50})
    test = pd.DataFrame({f"f{i}": range(30) for i in range(50)} | {"target": [0, 1] * 15})

    result = analyze_and_recommend(pred, train, test, lb, test_lb, imp, tmp_path)

    recs = " ".join(result["recommendations"])
    assert "sample-to-feature" in recs.lower() or "ratio" in recs.lower()


def test_regression_skips_class_imbalance(tmp_path: Path):
    pred = _make_predictor(problem_type="regression")
    lb, test_lb = _make_leaderboards()
    imp = _make_importance(["feat_a", "feat_b"], [0.15, 0.10])
    train = pd.DataFrame({"feat_a": range(200), "feat_b": range(200), "target": range(200)})
    test = pd.DataFrame({"feat_a": range(50), "feat_b": range(50), "target": range(50)})

    result = analyze_and_recommend(pred, train, test, lb, test_lb, imp, tmp_path)

    recs = " ".join(result["recommendations"])
    assert "imbalance" not in recs.lower()


def test_model_family_extraction():
    assert _model_family("LightGBM_BAG_L1") == "lightgbm"
    assert _model_family("WeightedEnsemble_L2") == "weightedensemble"
    assert _model_family("CatBoost") == "catboost"
    assert _model_family("UnknownModel") == "UnknownModel"


def test_analysis_json_structure(tmp_path: Path):
    pred = _make_predictor()
    lb, test_lb = _make_leaderboards()
    imp = _make_importance(["feat_a", "feat_b"], [0.15, 0.10])
    train = pd.DataFrame({"feat_a": range(200), "feat_b": range(200), "target": [0, 1] * 100})
    test = pd.DataFrame({"feat_a": range(50), "feat_b": range(50), "target": [0, 1] * 25})

    analyze_and_recommend(pred, train, test, lb, test_lb, imp, tmp_path)

    with open(tmp_path / "analysis.json") as f:
        data = json.load(f)

    assert "best_model" in data
    assert "problem_type" in data
    assert "findings" in data
    assert "recommendations" in data
    assert isinstance(data["findings"], list)
    assert isinstance(data["recommendations"], list)


# ---------------------------------------------------------------------------
# SHAP vs permutation-importance disagreement detection
# ---------------------------------------------------------------------------


def _write_shap_summary(path: Path, rows: list[tuple[str, float]]) -> None:
    """Write a minimal shap_summary.csv. Rows are (feature, mean_abs_shap),
    already in descending order."""
    df = pd.DataFrame(
        [
            {"feature": feat, "mean_abs_shap": shap, "rank": i + 1}
            for i, (feat, shap) in enumerate(rows)
        ]
    )
    df.to_csv(path, index=False)


def test_shap_agreement_does_not_emit_disagreement_hint(tmp_path):
    """When SHAP and permutation importance agree on the top feature,
    no SHAP-vs-importance hint should be emitted."""
    from automl_model_training.evaluate.analyze import _check_shap_vs_importance_disagreement

    _write_shap_summary(
        tmp_path / "shap_summary.csv",
        [("feat_a", 0.5), ("feat_b", 0.3), ("feat_c", 0.1), ("feat_d", 0.05)],
    )
    imp = pd.DataFrame(
        {"importance": [0.4, 0.25, 0.12, 0.06]},
        index=["feat_a", "feat_b", "feat_c", "feat_d"],
    )
    result = _check_shap_vs_importance_disagreement(tmp_path / "shap_summary.csv", imp)
    assert result is None


def test_shap_disagreement_emits_hint_naming_both_features(tmp_path):
    """When SHAP ranks feat_a #1 but permutation ranks feat_d #1, and the
    #1s are mutually absent from the other's top-3, a hint should fire and
    name both features."""
    from automl_model_training.evaluate.analyze import _check_shap_vs_importance_disagreement

    # SHAP top-3 = [feat_a, feat_b, feat_c]; permutation top-3 = [feat_d, feat_e, feat_f]
    _write_shap_summary(
        tmp_path / "shap_summary.csv",
        [
            ("feat_a", 0.5),
            ("feat_b", 0.3),
            ("feat_c", 0.2),
            ("feat_d", 0.01),
            ("feat_e", 0.005),
            ("feat_f", 0.001),
        ],
    )
    imp = pd.DataFrame(
        {"importance": [0.5, 0.3, 0.2, 0.01, 0.005, 0.001]},
        index=["feat_d", "feat_e", "feat_f", "feat_a", "feat_b", "feat_c"],
    )
    result = _check_shap_vs_importance_disagreement(tmp_path / "shap_summary.csv", imp, top_k=3)
    assert result is not None
    assert "feat_a" in result["finding"]
    assert "feat_d" in result["finding"]
    assert "feat_a" in result["recommendation"]
    assert "feat_d" in result["recommendation"]
    assert "tool_partial_dependence" in result["recommendation"]


def test_shap_reordering_within_top_k_is_not_material(tmp_path):
    """When the #1s differ but each top feature is still in the other's
    top-3, the disagreement is not material and no hint should fire.
    This prevents noisy alerts on essentially-equivalent rankings."""
    from automl_model_training.evaluate.analyze import _check_shap_vs_importance_disagreement

    # SHAP: [a, b, c]. Permutation: [b, a, c]. #1s differ but each is in
    # the other's top-3 — minor reordering, not material.
    _write_shap_summary(
        tmp_path / "shap_summary.csv",
        [("feat_a", 0.5), ("feat_b", 0.45), ("feat_c", 0.1), ("feat_d", 0.05)],
    )
    imp = pd.DataFrame(
        {"importance": [0.5, 0.45, 0.1, 0.05]},
        index=["feat_b", "feat_a", "feat_c", "feat_d"],
    )
    result = _check_shap_vs_importance_disagreement(tmp_path / "shap_summary.csv", imp)
    assert result is None


def test_shap_disagreement_skipped_when_too_few_features(tmp_path):
    """With fewer than top_k + 1 features in common, the check returns None
    rather than firing on essentially arbitrary rankings."""
    from automl_model_training.evaluate.analyze import _check_shap_vs_importance_disagreement

    _write_shap_summary(tmp_path / "shap_summary.csv", [("feat_a", 0.5), ("feat_b", 0.3)])
    imp = pd.DataFrame({"importance": [0.5, 0.3]}, index=["feat_a", "feat_b"])
    result = _check_shap_vs_importance_disagreement(tmp_path / "shap_summary.csv", imp, top_k=3)
    assert result is None


def test_shap_disagreement_skipped_when_shap_file_missing(tmp_path):
    """If shap_summary.csv doesn't exist at the expected path, the check
    returns None (and analyze_and_recommend silently skips the hint)."""
    from automl_model_training.evaluate.analyze import _check_shap_vs_importance_disagreement

    imp = pd.DataFrame({"importance": [0.5, 0.3]}, index=["feat_a", "feat_b"])
    result = _check_shap_vs_importance_disagreement(tmp_path / "nonexistent.csv", imp)
    assert result is None


def test_analyze_and_recommend_integrates_shap_disagreement(tmp_path):
    """End-to-end: analyze_and_recommend writes the hint into findings and
    recommendations when a SHAP summary exists and disagrees with importance."""
    pred = _make_predictor(problem_type="binary", best_model="LightGBM")
    lb, test_lb = _make_leaderboards("LightGBM", val_score=0.90, test_score=0.89)

    # Large enough feature set, clear top disagreement
    _write_shap_summary(
        tmp_path / "shap_summary.csv",
        [
            ("feat_a", 0.6),
            ("feat_b", 0.4),
            ("feat_c", 0.2),
            ("feat_d", 0.01),
            ("feat_e", 0.005),
            ("feat_f", 0.001),
        ],
    )
    imp = _make_importance(
        ["feat_d", "feat_e", "feat_f", "feat_a", "feat_b", "feat_c"],
        [0.6, 0.4, 0.2, 0.01, 0.005, 0.001],
    )
    n = 400
    train = pd.DataFrame({f"feat_{c}": range(n) for c in "abcdef"} | {"target": [0, 1] * (n // 2)})
    test = pd.DataFrame({f"feat_{c}": range(100) for c in "abcdef"} | {"target": [0, 1] * 50})

    analysis = analyze_and_recommend(pred, train, test, lb, test_lb, imp, tmp_path)

    assert any("SHAP" in f and "feat_a" in f and "feat_d" in f for f in analysis["findings"]), (
        f"Expected SHAP-vs-importance finding in: {analysis['findings']}"
    )
    assert any("tool_partial_dependence" in r for r in analysis["recommendations"]), (
        f"Expected tool_partial_dependence recommendation in: {analysis['recommendations']}"
    )


# ---------------------------------------------------------------------------
# Refit _FULL model handling, test_scores persistence, regression diagnostics
# ---------------------------------------------------------------------------


def test_overfit_check_falls_back_to_pre_refit_model(tmp_path: Path, leaderboard_with_refit):
    """With set_best_to_refit_full, model_best is a _FULL model whose
    score_val is NaN. The gap check must fall back to the pre-refit base
    model instead of silently producing a NaN gap (dead check)."""
    lb, test_lb = leaderboard_with_refit
    # Severe gap on the base model: val 0.90 vs test 0.75 (16.7%)
    test_lb = test_lb.copy()
    test_lb.loc[test_lb["model"] == "WeightedEnsemble_L2", "score_test"] = 0.75

    pred = _make_predictor(best_model="WeightedEnsemble_L2_FULL")
    imp = _make_importance(["feat_a", "feat_b"], [0.15, 0.10])
    train = pd.DataFrame({"feat_a": range(200), "feat_b": range(200), "target": [0, 1] * 100})
    test = pd.DataFrame({"feat_a": range(50), "feat_b": range(50), "target": [0, 1] * 25})

    result = analyze_and_recommend(pred, train, test, lb, test_lb, imp, tmp_path)

    joined = " ".join(result["findings"])
    assert "WeightedEnsemble_L2" in joined
    assert "nan" not in joined.lower()
    # Severe gap emits an overfitting FINDING (not just a recommendation)
    # so the agent's decision logic can react to it.
    assert any("overfit" in f.lower() for f in result["findings"])
    assert any("overfitting" in r.lower() for r in result["recommendations"])


def test_test_scores_persisted_signed(tmp_path: Path):
    """analysis.json carries predictor.evaluate() output as the canonical
    metric source, signed per AutoGluon's higher-is-better convention."""
    pred = _make_predictor(problem_type="regression")
    lb, test_lb = _make_leaderboards()
    imp = _make_importance(["feat_a", "feat_b"], [0.15, 0.10])
    train = pd.DataFrame({"feat_a": range(200), "feat_b": range(200), "target": range(200)})
    test = pd.DataFrame({"feat_a": range(50), "feat_b": range(50), "target": range(50)})

    result = analyze_and_recommend(
        pred,
        train,
        test,
        lb,
        test_lb,
        imp,
        tmp_path,
        test_scores={"root_mean_squared_error": -4.83, "r2": 0.81},
    )

    assert result["test_scores"] == {"root_mean_squared_error": -4.83, "r2": 0.81}
    assert result["score_convention"] == "higher_is_better"
    saved = json.loads((tmp_path / "analysis.json").read_text())
    assert saved["test_scores"]["root_mean_squared_error"] == -4.83


def test_regression_diagnostics_bias_and_low_r2(tmp_path: Path):
    """Systematic bias and weak R² from residual_stats.json become findings."""
    (tmp_path / "residual_stats.json").write_text(
        json.dumps(
            {
                "mean_residual": 2.0,  # positive → under-predicting
                "mean_absolute_error": 3.0,  # |2.0| > 0.2 * 3.0 → bias fires
                "r2": 0.1,  # < 0.3 → weak fit fires
            }
        )
    )
    pred = _make_predictor(problem_type="regression")
    lb, test_lb = _make_leaderboards()
    imp = _make_importance(["feat_a", "feat_b"], [0.15, 0.10])
    train = pd.DataFrame({"feat_a": range(200), "feat_b": range(200), "target": range(200)})
    test = pd.DataFrame({"feat_a": range(50), "feat_b": range(50), "target": range(50)})

    result = analyze_and_recommend(pred, train, test, lb, test_lb, imp, tmp_path)

    joined = " ".join(result["findings"])
    assert "under-predicting" in joined
    assert "R²" in joined


def test_regression_diagnostics_heteroscedasticity(tmp_path: Path):
    """Error magnitude growing with the target value is flagged from
    test_predictions.csv."""
    import numpy as np

    rng = np.random.RandomState(0)
    actual = np.linspace(1, 100, 80)
    residual = actual * 0.1 * rng.choice([-1, 1], 80)  # error scales with target
    pd.DataFrame({"actual": actual, "predicted": actual - residual, "residual": residual}).to_csv(
        tmp_path / "test_predictions.csv", index=False
    )

    pred = _make_predictor(problem_type="regression")
    lb, test_lb = _make_leaderboards()
    imp = _make_importance(["feat_a", "feat_b"], [0.15, 0.10])
    train = pd.DataFrame({"feat_a": range(200), "feat_b": range(200), "target": range(200)})
    test = pd.DataFrame({"feat_a": range(50), "feat_b": range(50), "target": range(50)})

    result = analyze_and_recommend(pred, train, test, lb, test_lb, imp, tmp_path)

    assert any("Heteroscedasticity" in f for f in result["findings"])
    assert any("log-transform" in r for r in result["recommendations"])


def test_regression_diagnostics_skewed_target(tmp_path: Path):
    """A heavily skewed training target triggers a log-transform suggestion."""
    import numpy as np

    rng = np.random.RandomState(0)
    skewed = np.exp(rng.randn(200) * 2)  # log-normal → strong positive skew
    pred = _make_predictor(problem_type="regression")
    lb, test_lb = _make_leaderboards()
    imp = _make_importance(["feat_a", "feat_b"], [0.15, 0.10])
    train = pd.DataFrame({"feat_a": range(200), "feat_b": range(200), "target": skewed})
    test = pd.DataFrame({"feat_a": range(50), "feat_b": range(50), "target": range(50)})

    result = analyze_and_recommend(pred, train, test, lb, test_lb, imp, tmp_path)

    assert any("skew" in f.lower() for f in result["findings"])


def test_regression_diagnostics_silent_when_no_issues(tmp_path: Path):
    """Clean regression artifacts produce no regression-specific findings."""
    (tmp_path / "residual_stats.json").write_text(
        json.dumps({"mean_residual": 0.01, "mean_absolute_error": 3.0, "r2": 0.9})
    )
    pred = _make_predictor(problem_type="regression")
    lb, test_lb = _make_leaderboards()
    imp = _make_importance(["feat_a", "feat_b"], [0.15, 0.10])
    train = pd.DataFrame({"feat_a": range(200), "feat_b": range(200), "target": range(200)})
    test = pd.DataFrame({"feat_a": range(50), "feat_b": range(50), "target": range(50)})

    result = analyze_and_recommend(pred, train, test, lb, test_lb, imp, tmp_path)

    joined = " ".join(result["findings"])
    assert "Systematic bias" not in joined
    assert "Weak fit" not in joined
