from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from automl_model_training.agent import _extract_metric, _read_analysis
from automl_model_training.experiment import compare_experiments


def tool_read_analysis(run_dir: str) -> dict:
    """Read analysis.json findings from a completed training run.

    Use this to re-examine a previous run without retraining.

    Returns
    -------
    dict with keys: best_model, problem_type, eval_metric, findings, recommendations
    """
    return _read_analysis(run_dir)


def tool_compare_runs(last_n: int | None = None) -> list[dict]:
    """Compare all recorded training experiments to track iteration progress.

    Call after each tool_train to decide whether to continue iterating
    or accept the current best model.

    Parameters
    ----------
    last_n : int or None
        Return only the last N experiments (None = all).

    Returns
    -------
    list[dict]
        One dict per experiment with flattened params and metrics,
        sorted oldest to newest.
    """
    df = compare_experiments(last_n=last_n)
    return df.to_dict(orient="records") if not df.empty else []


def tool_inspect_errors(run_dir: str, n: int = 20, worst: bool = True) -> dict:
    """Return the N worst (or best) predictions from a completed training run.

    Use after ``tool_train`` to see actual failure modes rather than aggregate
    metrics. Classification errors are ranked by confidence (lowest first);
    regression errors are ranked by absolute residual (largest first).

    Parameters
    ----------
    run_dir : str
        Path to a training run directory (contains test_predictions.csv,
        test_raw.csv, model_info.json).
    n : int
        Number of rows to return (default 20).
    worst : bool
        If True, return worst predictions. If False, return best
        (most-confident-correct for classification, smallest residual
        for regression). Default True.

    Returns
    -------
    dict with keys:
        problem_type : "binary" | "multiclass" | "regression" | "quantile"
        label        : target column name
        rows         : list of row dicts with actual, predicted, error
                       metric, and all feature values
        summary      : dict of aggregate stats (error_count, error_rate,
                       mean_abs_residual, etc. — depends on problem type)
        hints        : list of pattern observations (e.g., "class 1 is
                       overrepresented in errors", "errors cluster at
                       low feature_x values")
    """
    run_path = Path(run_dir)
    info_path = run_path / "model_info.json"
    preds_path = run_path / "test_predictions.csv"
    test_path = run_path / "test_raw.csv"

    for p in (info_path, preds_path, test_path):
        if not p.exists():
            raise FileNotFoundError(f"tool_inspect_errors: missing {p}")

    with open(info_path) as f:
        info = json.load(f)
    problem_type = info.get("problem_type", "")
    label = info.get("label", "")

    preds = pd.read_csv(preds_path)
    test_raw = pd.read_csv(test_path)
    if len(preds) != len(test_raw):
        raise ValueError(
            f"Row count mismatch: {len(preds)} predictions vs {len(test_raw)} test rows"
        )
    feature_cols = [c for c in test_raw.columns if c != label]
    merged = pd.concat(
        [preds.reset_index(drop=True), test_raw[feature_cols].reset_index(drop=True)],
        axis=1,
    )

    if problem_type in ("binary", "multiclass"):
        return _inspect_classification_errors(merged, n, worst, label, problem_type, feature_cols)
    if problem_type in ("regression", "quantile"):
        return _inspect_regression_errors(merged, n, worst, label, problem_type, feature_cols)
    raise ValueError(f"Unsupported problem_type: {problem_type}")


def _inspect_classification_errors(
    df: pd.DataFrame,
    n: int,
    worst: bool,
    label: str,
    problem_type: str,
    feature_cols: list[str],
) -> dict:
    """Rank by confidence on errors and surface patterns in the failure set.

    Handles AutoGluon's class label conventions:
    - Integer labels (0, 1, 2) serialize as "prob_0", "prob_1" in CSV. After
      a round-trip, "predicted" may be float64 when any NaN is present, so
      the column key f"prob_{1.0}" would not match "prob_1". We normalize by
      trying both the raw string and the int-cast form.
    - Boolean labels become "prob_True"/"prob_False" after pd.read_csv.
    - String labels (e.g. "cat", "dog") pass through cleanly.

    "Confidence" is the probability assigned to the *predicted* class — the
    same quantity AutoGluon's predict_proba reports — so this mirrors what
    predict() would have used for the argmax decision. For multiclass we
    also compute the top-2 margin (max − second-max) so the LLM can tell
    "confidently wrong" errors from "barely-picked-wrong-class" errors.
    """
    prob_cols = [c for c in df.columns if c.startswith("prob_")]

    def _conf_key(pred: object) -> str | None:
        """Map a predicted label to its matching prob_* column name."""
        # pd.isna rejects arbitrary object types under strict mypy; a None check
        # plus float-NaN check covers AutoGluon's output dtypes (int, float, str, bool).
        if pred is None or (isinstance(pred, float) and pred != pred):
            return None
        # Try raw string form first (works for strings, bools after CSV round-trip,
        # and integers cleanly preserved in the CSV)
        candidates = [str(pred)]
        # If the CSV round-trip promoted an int to float64 ("1.0" from "1"),
        # also try the int cast
        if isinstance(pred, float) and pred.is_integer():
            candidates.append(str(int(pred)))
        # Booleans may appear as np.bool_ → str gives "True"/"False" which matches
        for candidate in candidates:
            if f"prob_{candidate}" in df.columns:
                return f"prob_{candidate}"
        return None

    def _conf(row: pd.Series) -> float:
        key = _conf_key(row["predicted"])
        return float(row[key]) if key is not None else float("nan")

    if prob_cols:
        df["confidence"] = df.apply(_conf, axis=1)
        # Top-2 margin: how much more confident was the predicted class than the
        # runner-up? Small margin = marginal decision; large margin = assertive.
        if len(prob_cols) >= 2:
            prob_values = df[prob_cols].values
            sorted_probs = np.sort(prob_values, axis=1)
            df["top2_margin"] = (sorted_probs[:, -1] - sorted_probs[:, -2]).round(4)
    else:
        df["confidence"] = float("nan")

    df["is_error"] = df["actual"] != df["predicted"]

    if worst:
        # Errors first, then by lowest confidence among them
        ranked = df.sort_values(by=["is_error", "confidence"], ascending=[False, True]).head(n)
    else:
        ranked = df[~df["is_error"]].sort_values("confidence", ascending=False).head(n)

    error_rate = float(df["is_error"].mean())
    errors_only = df[df["is_error"]]

    # Normalize keys to str for JSON stability (int/float64/bool all end up as
    # distinct dict keys otherwise, which is brittle across CSV round-trips)
    def _norm_keys(d: dict) -> dict:
        return {str(k): v for k, v in d.items()}

    if len(errors_only):
        class_error_rate = _norm_keys(
            errors_only["actual"].value_counts(normalize=True).round(4).to_dict()
        )
    else:
        class_error_rate = {}
    class_prevalence = _norm_keys(df["actual"].value_counts(normalize=True).round(4).to_dict())

    hints = []
    # Flag classes overrepresented among errors vs population
    for cls, err_pct in class_error_rate.items():
        prev = class_prevalence.get(cls, 0)
        if prev > 0 and err_pct > prev * 1.5:
            hints.append(
                f"Class {cls} is overrepresented in errors: "
                f"{err_pct:.1%} of errors vs {prev:.1%} of population"
            )
    # High-confidence errors → confidently wrong (label noise or leakage)
    if len(errors_only) and prob_cols:
        hi_conf_errors = int((errors_only["confidence"] > 0.9).sum())
        if hi_conf_errors:
            hints.append(
                f"{hi_conf_errors} errors have confidence > 0.9 — "
                "check for label noise or leakage in those rows"
            )
        # Low-margin errors (multiclass) → model is genuinely uncertain, not wrong
        if "top2_margin" in df.columns:
            close_call_errors = int(
                ((errors_only["top2_margin"] < 0.1) & (errors_only["confidence"] < 0.6)).sum()
            )
            if close_call_errors:
                hints.append(
                    f"{close_call_errors} errors are close-call (top-2 margin < 0.1, "
                    "confidence < 0.6) — model is uncertain, consider more training data "
                    "or features to disambiguate these classes"
                )

    return {
        "problem_type": problem_type,
        "label": label,
        "rows": ranked.to_dict(orient="records"),
        "summary": {
            "error_count": int(df["is_error"].sum()),
            "error_rate": round(error_rate, 4),
            "class_error_distribution": class_error_rate,
            "class_prevalence": class_prevalence,
        },
        "hints": hints,
    }


def _inspect_regression_errors(
    df: pd.DataFrame,
    n: int,
    worst: bool,
    label: str,
    problem_type: str,
    feature_cols: list[str],
) -> dict:
    """Rank by absolute residual and surface systematic patterns.

    AutoGluon's regression artifact writes ``residual = actual - predicted``
    (see ``evaluate/regression.py``), so:
    - ``residual > 0`` ⇒ actual exceeds prediction ⇒ model under-predicted
    - ``residual < 0`` ⇒ prediction exceeds actual ⇒ model over-predicted
    """
    if "residual" not in df.columns:
        # Defensive fallback; AutoGluon's regression.py always writes this column,
        # but some quantile runs or older artifacts may not.
        df["residual"] = df["actual"] - df["predicted"]
    df["abs_residual"] = df["residual"].abs()
    # Relative error as % of actual — guard against division by zero
    df["residual_pct"] = (df["abs_residual"] / df["actual"].replace(0, pd.NA) * 100).round(2)

    ranked = df.sort_values("abs_residual", ascending=not worst).head(n)

    hints = []
    # Systematic bias check (convention above: residual = actual - predicted)
    mean_resid = float(df["residual"].mean())
    if abs(mean_resid) > df["abs_residual"].mean() * 0.2:
        direction = "over-predicting" if mean_resid < 0 else "under-predicting"
        hints.append(f"Model is systematically {direction} (mean residual = {mean_resid:.2f})")
    # Check if errors correlate with actual magnitude (heteroscedasticity).
    # Skip when either series has zero variance (correlation undefined).
    if len(df) >= 20 and df["actual"].std() > 0 and df["abs_residual"].std() > 0:
        corr = df["actual"].corr(df["abs_residual"])
        if abs(corr) > 0.3:
            hints.append(
                f"Error magnitude correlates with target value (r = {corr:.2f}) — "
                "consider log-transforming the target"
            )

    rows = ranked.to_dict(orient="records")
    return {
        "problem_type": problem_type,
        "label": label,
        "rows": rows,
        "summary": {
            "mean_abs_residual": round(float(df["abs_residual"].mean()), 4),
            "median_abs_residual": round(float(df["abs_residual"].median()), 4),
            "max_abs_residual": round(float(df["abs_residual"].max()), 4),
            "mean_residual": round(mean_resid, 4),
        },
        "hints": hints,
    }


def tool_compare_importance(
    run_dir_before: str,
    run_dir_after: str,
    top_n: int = 10,
    significance_delta: float = 0.01,
) -> dict:
    """Diff feature importance across two training runs.

    Call after an iteration cycle (feature engineering, re-training with
    a different drop list, a different preset) to see which features
    gained or lost importance and whether any new feature dominates
    without meaningfully moving the score — a classic leakage signal.

    Reads ``feature_importance.csv`` from each run (permutation importance
    written by ``tool_train``). The score deltas come from each run's
    ``leaderboard_test.csv`` best row so the LLM can judge whether an
    importance change coincided with a score change.

    Parameters
    ----------
    run_dir_before, run_dir_after : str
        Two training run directories to compare. Order matters: the
        function reports "after - before" for every delta.
    top_n : int
        How many top-importance features (by max of before/after) to
        include in the detailed diff. Default 10.
    significance_delta : float
        Minimum absolute importance delta to consider a feature as
        "changed materially" — below this, the feature is filed as
        "unchanged" even if the raw number moved slightly. Default 0.01.

    Returns
    -------
    dict with keys:
        run_dir_before, run_dir_after
        score_before, score_after, score_delta
        gained_features    : features present only in `after`
        lost_features      : features present only in `before`
        changed_features   : list of {
                                feature,
                                importance_before, importance_after,
                                delta, pct_change
                             } for the top_n features ranked by |delta|
        dominant_new_feature : dict or None — flagged when a `gained_features`
                               entry is the top-importance feature in `after`
                               AND score_delta is within noise
        hints              : actionable observations

    Raises
    ------
    FileNotFoundError : if either run is missing feature_importance.csv
    """
    before_path = Path(run_dir_before) / "feature_importance.csv"
    after_path = Path(run_dir_after) / "feature_importance.csv"
    if not before_path.exists():
        raise FileNotFoundError(f"tool_compare_importance: missing {before_path}")
    if not after_path.exists():
        raise FileNotFoundError(f"tool_compare_importance: missing {after_path}")

    before = pd.read_csv(before_path, index_col=0)
    after = pd.read_csv(after_path, index_col=0)
    if "importance" not in before.columns or "importance" not in after.columns:
        raise ValueError(
            "feature_importance.csv must contain an 'importance' column; got "
            f"before={list(before.columns)}, after={list(after.columns)}"
        )

    before_imp = {str(k): float(v) for k, v in before["importance"].to_dict().items()}
    after_imp = {str(k): float(v) for k, v in after["importance"].to_dict().items()}

    before_set = set(before_imp)
    after_set = set(after_imp)
    gained = sorted(after_set - before_set)
    lost = sorted(before_set - after_set)
    common = before_set & after_set

    # Per-feature delta for common features
    changed: list[dict] = []
    for feat in common:
        b = before_imp[feat]
        a = after_imp[feat]
        delta = a - b
        if abs(delta) < significance_delta:
            continue
        pct = (delta / abs(b)) * 100.0 if abs(b) > 1e-12 else float("inf")
        changed.append(
            {
                "feature": feat,
                "importance_before": round(b, 6),
                "importance_after": round(a, 6),
                "delta": round(delta, 6),
                "pct_change": round(pct, 2) if np.isfinite(pct) else "inf",
            }
        )
    # Include gained features with before=0 and lost features with after=0 so
    # the combined ranking surfaces newcomers/departures alongside shifts
    for feat in gained:
        a = after_imp[feat]
        if abs(a) < significance_delta:
            continue
        changed.append(
            {
                "feature": feat,
                "importance_before": 0.0,
                "importance_after": round(a, 6),
                "delta": round(a, 6),
                "pct_change": "new",
            }
        )
    for feat in lost:
        b = before_imp[feat]
        if abs(b) < significance_delta:
            continue
        changed.append(
            {
                "feature": feat,
                "importance_before": round(b, 6),
                "importance_after": 0.0,
                "delta": round(-b, 6),
                "pct_change": "dropped",
            }
        )

    # Sort by absolute delta, keep top_n
    # Typing note: delta is always a float in our records; cast via float() to
    # silence mypy since the dict is typed as dict[str, object]
    changed.sort(key=lambda d: abs(float(d["delta"])), reverse=True)  # type: ignore[arg-type]
    changed = changed[:top_n]

    # Score deltas — best score per run (absolute value, matches
    # _extract_metric convention)
    score_before = _extract_metric(run_dir_before, "score")
    score_after = _extract_metric(run_dir_after, "score")
    score_delta = None
    if score_before is not None and score_after is not None:
        score_delta = round(score_after - score_before, 6)

    # Dominant-new-feature check: a new feature is the top of `after` AND
    # score barely moved → likely leakage or pointless feature engineering
    dominant_new: dict | None = None
    if gained and score_delta is not None:
        top_after_feat = max(after_imp, key=lambda k: after_imp[k])
        if top_after_feat in set(gained):
            # "Barely moved" = within 1% of the original score
            noise_threshold = 0.01 * max(abs(score_before or 0.0), 0.01)
            if abs(score_delta) < noise_threshold:
                dominant_new = {
                    "feature": top_after_feat,
                    "importance_after": round(after_imp[top_after_feat], 6),
                    "score_delta": score_delta,
                    "reason": (
                        "New feature dominates importance but score barely moved "
                        "— suspect leakage, redundancy with existing features, or "
                        "the feature is capturing label noise."
                    ),
                }

    hints: list[str] = []
    if dominant_new is not None:
        hints.append(
            f"New feature '{dominant_new['feature']}' is now the most important "
            f"(importance={dominant_new['importance_after']}) but score only "
            f"moved by {score_delta}. Investigate for leakage before trusting this run."
        )
    if not gained and not lost and not changed:
        hints.append(
            "No material importance changes detected. The two runs used "
            "effectively the same features; look at preset or time_limit "
            "differences to explain any score delta."
        )
    if score_delta is not None and score_delta < -0.01:
        hints.append(
            f"Score regressed by {abs(score_delta)} — review the 'lost_features' "
            "list; one of them may have been a genuine signal source."
        )
    if gained and score_delta is not None and score_delta > 0.01:
        hints.append(
            f"Score improved by {score_delta}; gained features contributed. "
            "Consider re-running profile/deep_profile on these new columns to "
            "catch emergent skew/leakage."
        )

    return {
        "run_dir_before": run_dir_before,
        "run_dir_after": run_dir_after,
        "score_before": score_before,
        "score_after": score_after,
        "score_delta": score_delta,
        "gained_features": gained,
        "lost_features": lost,
        "changed_features": changed,
        "dominant_new_feature": dominant_new,
        "hints": hints,
    }
