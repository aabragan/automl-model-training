"""Post-training accuracy analysis and improvement recommendations."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd
from autogluon.tabular import TabularPredictor

from automl_model_training.config import (
    HETEROSCEDASTICITY_CORR_THRESHOLD,
    LOW_IMPORTANCE_THRESHOLD,
    OVERFITTING_MODERATE_GAP_PCT,
    OVERFITTING_SEVERE_GAP_PCT,
    REGRESSION_LOW_R2_THRESHOLD,
    RESIDUAL_BIAS_RATIO,
    TARGET_SKEW_THRESHOLD,
)

logger = logging.getLogger(__name__)


def analyze_and_recommend(
    predictor: TabularPredictor,
    train_raw: pd.DataFrame,
    test_raw: pd.DataFrame,
    leaderboard: pd.DataFrame,
    test_leaderboard: pd.DataFrame,
    importance: pd.DataFrame,
    output: Path,
    test_scores: dict | None = None,
) -> dict:
    """Analyze training results and write an improvement report.

    Returns the full analysis dict (also saved as ``analysis.json``).

    ``test_scores`` is the dict returned by ``predictor.evaluate(test_raw)``.
    It is persisted into ``analysis.json`` as the canonical test-set metric
    source. Values follow AutoGluon's internal convention: higher is always
    better (error metrics such as RMSE appear negated).
    """

    label = predictor.label
    problem_type = predictor.problem_type
    eval_metric = str(predictor.eval_metric)

    findings: list[str] = []
    recommendations: list[str] = []

    # ------------------------------------------------------------------
    # 1. Overfitting detection — compare val vs test scores
    # ------------------------------------------------------------------
    best_model = predictor.model_best
    # Refit "_FULL" models (set_best_to_refit_full=True) are trained on
    # train+val and have no validation score (NaN). Fall back to the
    # pre-refit base model so the val/test gap check still works.
    gap_model = best_model
    val_row = _val_scored_row(leaderboard, gap_model)
    if val_row.empty and best_model.endswith("_FULL"):
        gap_model = best_model[: -len("_FULL")]
        val_row = _val_scored_row(leaderboard, gap_model)
    test_row = test_leaderboard.loc[test_leaderboard["model"] == gap_model]

    # A large val/test gap suggests the model memorized training patterns
    # that don't generalize — the percentage gap normalizes across metrics
    if (
        not val_row.empty
        and not test_row.empty
        and not pd.isna(test_row["score_test"].iloc[0])
    ):
        val_score = float(val_row["score_val"].iloc[0])
        test_score = float(test_row["score_test"].iloc[0])
        gap = abs(val_score - test_score)
        gap_pct = (gap / abs(val_score) * 100) if val_score != 0 else 0

        findings.append(
            f"Best model '{gap_model}': val={val_score:.4f}, "
            f"test={test_score:.4f}, gap={gap:.4f} ({gap_pct:.1f}%)"
        )

        if gap_pct > OVERFITTING_SEVERE_GAP_PCT:
            # "Overfitting" must appear in findings (not just recommendations)
            # so the autonomous agent's decision logic can react to it.
            findings.append(
                f"Overfitting signal: val/test gap {gap_pct:.1f}% exceeds "
                f"{OVERFITTING_SEVERE_GAP_PCT:.0f}% threshold."
            )
            recommendations.append(
                f"Significant val/test gap detected (>{gap_pct:.0f}%). The model may be "
                "overfitting. Consider: increasing training data, reducing model "
                "complexity (try preset='high_quality' or 'good_quality'), or "
                "adding regularization via hyperparameter tuning."
            )
        elif gap_pct > OVERFITTING_MODERATE_GAP_PCT:
            recommendations.append(
                f"Moderate val/test gap ({gap_pct:.1f}%). Monitor for overfitting on "
                "future data. A larger test set or cross-validation may help "
                "get a more stable estimate."
            )

    # ------------------------------------------------------------------
    # 2. Model diversity — are ensembles helping?
    # ------------------------------------------------------------------
    n_models = len(leaderboard)
    findings.append(f"Total models trained: {n_models}")

    if n_models < 5:
        recommendations.append(
            f"Only {n_models} models were trained. Increasing --time-limit "
            "or using preset='best' would allow AutoGluon to explore more "
            "model families and stacking layers."
        )

    # Check if top models are all the same family
    if "model" in test_leaderboard.columns:
        top_models = test_leaderboard.head(3)["model"].tolist()
        families = {_model_family(m) for m in top_models}
        if len(families) == 1 and n_models > 3:
            recommendations.append(
                f"Top 3 models are all from the '{families.pop()}' family. "
                "The ensemble may benefit from more diverse model types. "
                "Try a longer time limit to allow other families to train."
            )

    # ------------------------------------------------------------------
    # 3. Feature importance analysis
    # ------------------------------------------------------------------
    if not importance.empty and "importance" in importance.columns:
        low_importance = importance[importance["importance"] <= LOW_IMPORTANCE_THRESHOLD]
        if len(low_importance) > 0:
            low_feats = low_importance.index.tolist()
            findings.append(
                f"Low-importance features ({len(low_feats)}): "
                f"{low_feats[:10]}{'...' if len(low_feats) > 10 else ''}"
            )
            recommendations.append(
                f"{len(low_feats)} feature(s) have near-zero permutation "
                "importance. Dropping them with --drop may reduce noise and "
                "speed up training without hurting accuracy."
            )

        # Negative importance = feature hurts the model
        negative = importance[importance["importance"] < 0]
        if len(negative) > 0:
            neg_feats = negative.index.tolist()
            findings.append(f"Harmful features (negative importance): {neg_feats}")
            recommendations.append(
                f"{len(neg_feats)} feature(s) have negative permutation "
                "importance — they actively hurt predictions. Strongly "
                f"consider dropping: {neg_feats}"
            )

    # ------------------------------------------------------------------
    # 4. Dataset size checks
    # ------------------------------------------------------------------
    n_train = len(train_raw)
    n_test = len(test_raw)
    n_features = len(predictor.features())

    findings.append(f"Dataset: {n_train} train rows, {n_test} test rows, {n_features} features")

    # 10x is a common rule of thumb for minimum samples-per-feature
    if n_train < n_features * 10:
        recommendations.append(
            f"Low sample-to-feature ratio ({n_train}/{n_features} = "
            f"{n_train / n_features:.1f}x). Consider collecting more data "
            "or reducing dimensionality to avoid overfitting."
        )

    if n_test < 100:
        recommendations.append(
            f"Test set is small ({n_test} rows). Evaluation metrics may be "
            "unreliable. Consider using a smaller --test-size or collecting "
            "more data."
        )

    # ------------------------------------------------------------------
    # 5. Class imbalance (classification only)
    # ------------------------------------------------------------------
    if problem_type in ("binary", "multiclass"):
        class_counts = train_raw[label].value_counts()
        majority = class_counts.iloc[0]
        minority = class_counts.iloc[-1]
        imbalance_ratio = majority / minority if minority > 0 else float("inf")

        findings.append(
            f"Class distribution: {class_counts.to_dict()}, imbalance ratio={imbalance_ratio:.1f}:1"
        )

        if imbalance_ratio > 10:
            recommendations.append(
                f"Severe class imbalance ({imbalance_ratio:.0f}:1). Consider "
                "using --eval-metric balanced_accuracy or f1, applying "
                "oversampling/undersampling, or collecting more minority "
                "class samples."
            )
        elif imbalance_ratio > 3:
            recommendations.append(
                f"Moderate class imbalance ({imbalance_ratio:.1f}:1). Ensure "
                "your eval metric accounts for this (e.g. f1, balanced_accuracy, "
                "roc_auc rather than plain accuracy)."
            )

    # ------------------------------------------------------------------
    # 5b. Regression diagnostics (regression/quantile only)
    # ------------------------------------------------------------------
    if problem_type in ("regression", "quantile"):
        reg = _regression_diagnostics(train_raw[label], output)
        findings.extend(reg["findings"])
        recommendations.extend(reg["recommendations"])

    # ------------------------------------------------------------------
    # 6. SHAP vs permutation-importance disagreement (when SHAP was computed)
    # ------------------------------------------------------------------
    shap_summary_path = output / "shap_summary.csv"
    if shap_summary_path.exists() and not importance.empty:
        disagreement = _check_shap_vs_importance_disagreement(
            shap_summary_path=shap_summary_path,
            importance=importance,
        )
        if disagreement is not None:
            findings.append(disagreement["finding"])
            recommendations.append(disagreement["recommendation"])

    # ------------------------------------------------------------------
    # 7. Preset / time-limit suggestions
    # ------------------------------------------------------------------
    if not recommendations:
        findings.append("No major issues detected.")
        recommendations.append(
            "Results look solid. To push further, try: increasing "
            "--time-limit, using preset='best' if not already, or "
            "engineering new features from domain knowledge."
        )

    # ------------------------------------------------------------------
    # Build and save report
    # ------------------------------------------------------------------
    analysis = {
        "best_model": best_model,
        "problem_type": problem_type,
        "eval_metric": eval_metric,
        # Canonical test-set scores for the deployed predictor, from
        # predictor.evaluate(). AutoGluon internal convention: higher is
        # always better (error metrics like RMSE appear negated).
        "test_scores": {
            str(k): float(v) for k, v in (test_scores or {}).items() if not pd.isna(v)
        },
        "score_convention": "higher_is_better",
        "findings": findings,
        "recommendations": recommendations,
    }

    with open(output / "analysis.json", "w") as f:
        json.dump(analysis, f, indent=2)

    # Human-readable report
    lines = [
        "=" * 60,
        "  POST-TRAINING ANALYSIS",
        "=" * 60,
        "",
        "Findings:",
    ]
    for i, finding in enumerate(findings, 1):
        lines.append(f"  {i}. {finding}")
    lines.append("")
    lines.append("Recommendations:")
    for i, rec in enumerate(recommendations, 1):
        lines.append(f"  {i}. {rec}")
    lines.append("")
    lines.append("=" * 60)

    report_text = "\n".join(lines)
    logger.info("\n%s", report_text)

    with open(output / "analysis_report.txt", "w") as f:
        f.write(report_text)
    logger.info("Analysis saved → %s", output / "analysis.json")
    logger.info("Report saved  → %s", output / "analysis_report.txt")

    return analysis


def _val_scored_row(leaderboard: pd.DataFrame, model: str) -> pd.DataFrame:
    """Return the leaderboard row for *model* if it has a real score_val."""
    row = leaderboard.loc[leaderboard["model"] == model]
    if row.empty or pd.isna(row["score_val"].iloc[0]):
        return row.iloc[0:0]
    return row


def _regression_diagnostics(y_train: pd.Series, output: Path) -> dict:
    """Regression-specific findings from artifacts written earlier in the run.

    Reads ``residual_stats.json`` and ``test_predictions.csv`` (written by
    ``save_regression_artifacts`` before analysis runs) and checks for:
    - systematic bias (mean residual large relative to MAE)
    - weak fit (low R²)
    - heteroscedasticity (error magnitude correlated with target value)
    - heavily skewed training target

    Residual convention (evaluate/regression.py): residual = actual - predicted,
    so a positive mean residual means the model under-predicts.
    """
    findings: list[str] = []
    recommendations: list[str] = []

    stats_path = output / "residual_stats.json"
    if stats_path.exists():
        try:
            stats = json.loads(stats_path.read_text())
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Could not read %s: %s", stats_path, e)
            stats = {}

        mean_resid = stats.get("mean_residual")
        mae = stats.get("mean_absolute_error")
        r2 = stats.get("r2")

        if mean_resid is not None and mae and abs(mean_resid) > mae * RESIDUAL_BIAS_RATIO:
            direction = "over-predicting" if mean_resid < 0 else "under-predicting"
            findings.append(
                f"Systematic bias: model is consistently {direction} "
                f"(mean residual = {mean_resid:.4f}, MAE = {mae:.4f})."
            )
            recommendations.append(
                f"The model is systematically {direction}. Check for target "
                "drift between train and test, or missing features that "
                "explain the offset."
            )

        if r2 is not None and r2 < REGRESSION_LOW_R2_THRESHOLD:
            findings.append(f"Weak fit: test R² = {r2:.3f}.")
            recommendations.append(
                f"Test R² is {r2:.3f} — the model explains little target "
                "variance. Consider engineering stronger features "
                "(tool_engineer_features) or verifying the target is "
                "predictable from the available columns."
            )

    preds_path = output / "test_predictions.csv"
    if preds_path.exists():
        try:
            preds = pd.read_csv(preds_path)
        except (pd.errors.ParserError, OSError) as e:
            logger.warning("Could not read %s: %s", preds_path, e)
            preds = pd.DataFrame()
        if (
            {"actual", "residual"} <= set(preds.columns)
            and len(preds) >= 20
            and preds["actual"].std() > 0
            and preds["residual"].abs().std() > 0
        ):
            corr = float(preds["actual"].corr(preds["residual"].abs()))
            if abs(corr) > HETEROSCEDASTICITY_CORR_THRESHOLD:
                findings.append(
                    f"Heteroscedasticity: error magnitude correlates with the "
                    f"target value (r = {corr:.2f})."
                )
                recommendations.append(
                    "Error size grows with the target value. Consider "
                    "log-transforming the target or switching --eval-metric "
                    "to mean_absolute_error."
                )

    has_variance = len(y_train) >= 3 and y_train.std() > 0
    skew = float(y_train.skew()) if has_variance else 0.0  # type: ignore[arg-type]
    if abs(skew) > TARGET_SKEW_THRESHOLD:
        findings.append(f"Training target is heavily skewed (skew = {skew:.2f}).")
        recommendations.append(
            "A heavily skewed target often trains better after a log "
            "transform (tool_engineer_features log transform on the label, "
            "or preprocess before training)."
        )

    return {"findings": findings, "recommendations": recommendations}


def _model_family(model_name: str) -> str:
    """Extract a rough model family from an AutoGluon model name."""
    name = model_name.lower()
    for family in (
        "weightedensemble",
        "catboost",
        "lightgbm",
        "xgboost",
        "randomforest",
        "extratrees",
        "knn",
        "neuralnet",
        "fastai",
        "tabular",
        "linear",
    ):
        if family in name:
            return family
    return model_name


def _check_shap_vs_importance_disagreement(
    shap_summary_path: Path,
    importance: pd.DataFrame,
    top_k: int = 3,
) -> dict | None:
    """Compare the top-k features by SHAP magnitude vs. permutation importance.

    Returns a ``{"finding": ..., "recommendation": ...}`` dict when the two
    ranking methods materially disagree, or ``None`` when they agree or the
    data is too thin to judge.

    Why this matters
    ----------------
    SHAP and permutation importance measure different things. Permutation
    importance asks "how much does the score drop if I shuffle this feature?"
    — it captures each feature's marginal contribution to the trained model.
    SHAP asks "how much does this feature push the prediction on each row,
    averaged across rows?" — it captures each feature's typical per-row
    influence regardless of whether the score depends on it.

    When the two rankings line up, they reinforce each other. When they
    disagree, one of two patterns is typically at play:

    1. Redundancy: feature A tops SHAP (high per-row influence) but B tops
       permutation (score drops when B is shuffled). A and B often carry
       overlapping signal; A drives the predictions, B is the one the model
       *must* have to maintain its score.

    2. Non-linearity the model captured, split across features: a tree model
       might route via feature B (high permutation drop if shuffled) but
       actually move the prediction via feature A (high SHAP). Classic in
       boosted trees with correlated features.

    Either way, the LLM agent should know. Calling tool_partial_dependence
    on both features makes the mechanism visible.

    Materiality is defined as: the top SHAP feature is not in the top-k
    permutation-importance list, OR vice versa. Spearman rank correlation
    would be cleaner but requires scipy; the top-k overlap check catches
    the same cases for the sample sizes we see in practice.
    """
    try:
        shap_summary = pd.read_csv(shap_summary_path)
    except Exception as e:  # noqa: BLE001 — a malformed file shouldn't kill analysis
        logger.warning("Could not read SHAP summary at %s: %s", shap_summary_path, e)
        return None

    if "feature" not in shap_summary.columns or "mean_abs_shap" not in shap_summary.columns:
        return None
    if "importance" not in importance.columns:
        return None

    # SHAP file is already sorted by mean_abs_shap descending; re-sort defensively
    shap_sorted = shap_summary.sort_values("mean_abs_shap", ascending=False)
    imp_sorted = importance.sort_values("importance", ascending=False)

    # Only compare features that exist in both rankings (e.g., low-importance
    # features might have been dropped during SHAP-only filtering).
    shap_features = [str(f) for f in shap_sorted["feature"].tolist()]
    imp_features = [str(f) for f in imp_sorted.index.tolist()]
    common = set(shap_features) & set(imp_features)
    if len(common) < top_k + 1:
        # Not enough features to form a meaningful top-k comparison.
        return None

    top_shap = [f for f in shap_features if f in common][:top_k]
    top_imp = [f for f in imp_features if f in common][:top_k]

    top_shap_feat = top_shap[0]
    top_imp_feat = top_imp[0]

    # Materiality: the #1 feature differs between methods AND the #1 of one
    # method is not in the other's top-k. This two-step check avoids
    # firing on trivial re-orderings within the top group.
    if top_shap_feat == top_imp_feat:
        return None
    if top_shap_feat in top_imp and top_imp_feat in top_shap:
        # #1s differ but both are still highly ranked by the other method —
        # the disagreement is not material.
        return None

    finding = (
        f"SHAP and permutation importance disagree on the top feature: "
        f"SHAP ranks '{top_shap_feat}' first (top-{top_k}={top_shap}), "
        f"permutation importance ranks '{top_imp_feat}' first "
        f"(top-{top_k}={top_imp})."
    )
    recommendation = (
        f"Top SHAP feature '{top_shap_feat}' and top permutation-importance "
        f"feature '{top_imp_feat}' differ. SHAP measures per-row influence; "
        "permutation importance measures score dependence. Disagreement "
        "usually means the two features carry overlapping signal (one drives "
        "predictions, the other guards the score). Run tool_partial_dependence "
        f"on both '{top_shap_feat}' and '{top_imp_feat}' to see which one "
        "actually moves predictions; consider dropping the redundant one or "
        "engineering a ratio/product."
    )
    return {"finding": finding, "recommendation": recommendation}
