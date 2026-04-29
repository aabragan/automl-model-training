"""LLM agent tool layer.

Thin wrappers over the existing automl_model_training API that return
JSON-serializable dicts. Designed to be registered as tools with any
LLM agent framework (Bedrock Agents, LangChain, OpenAI function calling).

Iteration levers available to the LLM
--------------------------------------
preset           : Controls model diversity and training depth.
                   Ordered best→worst accuracy: extreme > best_quality/best >
                   best_v150 > high_quality/high > high_v150 > good > medium.
                   Start with "best", escalate if score is insufficient.

eval_metric      : The metric being optimized. Must match the problem type.
                   Binary: f1, roc_auc, accuracy, balanced_accuracy, mcc, log_loss
                   Multiclass: f1_macro, f1_weighted, accuracy, log_loss
                   Regression: root_mean_squared_error, mean_absolute_error, r2

problem_type     : Force binary, multiclass, regression, or quantile.
                   Use None to auto-detect from the label column.

drop             : List of feature column names to exclude. Use after
                   tool_train returns low-importance or harmful features
                   in analysis["findings"].

time_limit       : Seconds to train. None = train all models to completion.
                   Increase when the leaderboard shows few models trained.

test_size        : Fraction held out for evaluation (default 0.2).
                   Decrease if the test set is too small (<100 rows).

calibrate_threshold : Binary only. Metric name to calibrate the decision
                   threshold for (e.g. "f1"). Use when precision/recall
                   trade-off matters more than raw accuracy.

prune            : Remove underperforming ensemble members after training.
                   Reduces inference latency without hurting accuracy much.

explain          : Compute SHAP values. Use when feature attribution is
                   needed to understand model decisions.

cv_folds         : Run k-fold cross-validation before the final train/test
                   run. Use when the dataset is small (<1000 rows) or when
                   a single train/test split gives unstable metrics.

seed             : Random seed. Change to verify score stability across splits.

Decision guide for the LLM
---------------------------
1. Always call tool_profile first to get drop_recommendations and
   label_distribution before the first tool_train call.

2. After each tool_train call, read analysis["findings"] and
   analysis["recommendations"]:
   - "low-importance features" → add them to drop on next iteration
   - "harmful features (negative importance)" → add to drop immediately
   - "overfitting" → switch to a less aggressive preset or add cv_folds
   - "class imbalance" → switch eval_metric to f1 or balanced_accuracy
   - "few models trained" → increase time_limit

3. Preset escalation order for accuracy:
   best → best_v150 → high_quality (if overfitting, go the other direction)

4. Call tool_compare_runs after each iteration to track progress and
   decide whether to continue or stop.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from automl_model_training.agent import _extract_metric, _read_analysis
from automl_model_training.config import CLASSIFICATION_CARDINALITY_THRESHOLD, make_run_dir
from automl_model_training.data import load_and_prepare
from automl_model_training.experiment import compare_experiments
from automl_model_training.feature_engineering import apply_transformations
from automl_model_training.predict import load_predictor, predict_and_save
from automl_model_training.profile import (
    compute_correlation_matrix,
    recommend_features_to_drop,
)
from automl_model_training.train import cross_validate, train_and_evaluate


def tool_profile(csv_path: str, label: str) -> dict:
    """Analyze a dataset before training.

    Always call this first. Returns shape, label distribution, missing value
    percentages, and correlated feature drop recommendations.

    Parameters
    ----------
    csv_path : str
        Path to the input CSV.
    label : str
        Target column name.

    Returns
    -------
    dict with keys:
        shape               : [n_rows, n_cols]
        label_distribution  : {class: count} — use to detect class imbalance
        missing_pct         : {column: pct_missing} — flag columns >20% missing
        drop_recommendations: list of {feature, reason, correlated_with}
        numeric_features    : list of numeric column names
        categorical_features: list of categorical column names
    """
    data = pd.read_csv(csv_path)
    corr = compute_correlation_matrix(data, label)
    recs = recommend_features_to_drop(corr, label)
    numeric = data.select_dtypes(include="number").columns.tolist()
    categorical = data.select_dtypes(exclude="number").columns.tolist()
    return {
        "shape": list(data.shape),
        "label_distribution": data[label].value_counts().to_dict(),
        "missing_pct": (data.isnull().mean() * 100).round(2).to_dict(),
        "drop_recommendations": recs,
        "numeric_features": [c for c in numeric if c != label],
        "categorical_features": [c for c in categorical if c != label],
    }


def tool_train(
    csv_path: str,
    label: str,
    preset: str = "best",
    problem_type: str | None = None,
    eval_metric: str | None = None,
    time_limit: int | None = None,
    drop: list[str] | None = None,
    test_size: float = 0.2,
    seed: int = 42,
    prune: bool = False,
    explain: bool = False,
    cv_folds: int | None = None,
    calibrate_threshold: str | None = None,
    output_dir: str = "output",
) -> dict:
    """Train an AutoGluon model and return results for the next iteration decision.

    Parameters
    ----------
    csv_path : str
        Path to the training CSV.
    label : str
        Target column name.
    preset : str
        AutoGluon preset controlling model diversity and training depth.
        Options: extreme, best_quality, best (default), best_v150,
                 high_quality, high, high_v150, good, medium.
    problem_type : str or None
        Force: binary, multiclass, regression, quantile. None = auto-detect.
    eval_metric : str or None
        Metric to optimize. None = auto-detect from problem_type.
        Binary: f1, roc_auc, accuracy, balanced_accuracy, mcc, log_loss
        Multiclass: f1_macro, f1_weighted, accuracy, log_loss
        Regression: root_mean_squared_error, mean_absolute_error, r2
    time_limit : int or None
        Max training seconds. None = train all models to completion.
    drop : list[str] or None
        Feature columns to exclude. Add low/negative importance features here.
    test_size : float
        Fraction of data held out for evaluation (default 0.2).
    seed : int
        Random seed for reproducibility (default 42).
    prune : bool
        Remove underperforming ensemble members after training (default False).
    explain : bool
        Compute SHAP feature attributions after training (default False).
    cv_folds : int or None
        Run k-fold cross-validation before the final train/test run.
        Recommended for small datasets (<1000 rows).
    calibrate_threshold : str or None
        Binary only. Calibrate decision threshold for this metric (e.g. "f1").
    output_dir : str
        Base directory for run outputs (default "output").

    Returns
    -------
    dict with keys:
        run_dir      : path to this run's output directory
        score        : best model's test score (absolute value)
        model_info   : problem_type, eval_metric, features, best_model
        analysis     : findings and recommendations for the next iteration
        leaderboard  : list of {model, score_val, score_test} for top models
        low_importance_features  : features with near-zero importance to drop
        negative_importance_features : features that hurt the model — drop these
    """
    run_dir = make_run_dir(output_dir, prefix="llm_train")

    train_raw, test_raw, _, _, _ = load_and_prepare(
        csv_path=csv_path,
        label=label,
        features_to_drop=drop or [],
        test_size=test_size,
        random_state=seed,
        output_dir=run_dir,
    )

    if cv_folds is not None:
        full_data = pd.concat([train_raw, test_raw], ignore_index=True)
        cross_validate(
            data=full_data,
            label=label,
            n_folds=cv_folds,
            problem_type=problem_type,
            eval_metric=eval_metric,
            time_limit=time_limit,
            preset=preset,
            output_dir=run_dir,
            random_state=seed,
        )

    train_and_evaluate(
        train_raw=train_raw,
        test_raw=test_raw,
        label=label,
        problem_type=problem_type,
        eval_metric=eval_metric,
        time_limit=time_limit,
        preset=preset,
        output_dir=run_dir,
        prune=prune,
        explain=explain,
        calibrate_threshold=calibrate_threshold,
    )

    score = _extract_metric(run_dir, eval_metric or "score")
    analysis = _read_analysis(run_dir)

    model_info: dict = {}
    model_info_path = Path(run_dir) / "model_info.json"
    if model_info_path.exists():
        with open(model_info_path) as f:
            model_info = json.load(f)

    # Parse leaderboard for the LLM to see which model families performed best
    leaderboard: list[dict] = []
    lb_path = Path(run_dir) / "leaderboard_test.csv"
    if lb_path.exists():
        lb = pd.read_csv(lb_path)
        cols = [c for c in ["model", "score_val", "score_test", "fit_time"] if c in lb.columns]
        leaderboard = lb[cols].head(10).to_dict(orient="records")

    # Parse feature importance so the LLM can decide what to drop next
    low_importance: list[str] = []
    negative_importance: list[str] = []
    imp_path = Path(run_dir) / "feature_importance.csv"
    if imp_path.exists():
        imp = pd.read_csv(imp_path, index_col=0)
        if "importance" in imp.columns:
            low_importance = imp[imp["importance"].between(0, 0.001)].index.tolist()
            negative_importance = imp[imp["importance"] < 0].index.tolist()

    return {
        "run_dir": run_dir,
        "score": score,
        "model_info": model_info,
        "analysis": analysis,
        "leaderboard": leaderboard,
        "low_importance_features": low_importance,
        "negative_importance_features": negative_importance,
    }


def tool_predict(
    csv_path: str,
    model_dir: str,
    output_dir: str = "predictions_output",
    min_confidence: float | None = None,
    decision_threshold: float | None = None,
) -> dict:
    """Run inference on new data using a trained model.

    Parameters
    ----------
    csv_path : str
        Path to the prediction CSV.
    model_dir : str
        Path to the trained AutogluonModels/ directory from a tool_train run_dir.
    output_dir : str
        Base directory for prediction outputs.
    min_confidence : float or None
        Flag classification rows below this confidence (e.g. 0.7).
    decision_threshold : float or None
        Override binary classification decision threshold (e.g. 0.3).

    Returns
    -------
    dict with keys:
        run_dir  : path to prediction outputs
        num_rows : number of rows predicted
        columns  : output column names
        summary  : problem_type, best_model, eval_scores (if ground truth present)
    """
    run_dir = make_run_dir(output_dir, prefix="llm_predict")
    predictor = load_predictor(model_dir)
    data = pd.read_csv(csv_path)
    result = predict_and_save(
        predictor,
        data,
        run_dir,
        min_confidence=min_confidence,
        decision_threshold=decision_threshold,
    )

    summary: dict = {}
    summary_path = Path(run_dir) / "prediction_summary.json"
    if summary_path.exists():
        with open(summary_path) as f:
            summary = json.load(f)

    return {
        "run_dir": run_dir,
        "num_rows": len(result),
        "columns": list(result.columns),
        "summary": summary,
    }


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


def tool_engineer_features(
    csv_path: str,
    transformations: dict,
    label: str | None = None,
    output_dir: str = "output",
) -> dict:
    """Apply declarative feature transformations to a CSV.

    Use after ``tool_profile`` to create features the model can't derive on
    its own: log transforms for skewed distributions, ratios for relationships
    between features, date parts for temporal signal. Pass the returned
    ``engineered_csv`` path to ``tool_train`` for the next iteration.

    Parameters
    ----------
    csv_path : str
        Path to the input CSV.
    transformations : dict
        Spec dict. Supported keys and values:

        - ``log``: list[str] — log1p of each column → ``log_<col>``
        - ``sqrt``: list[str] — sqrt of each column → ``sqrt_<col>``
        - ``ratio``: list[[num, denom]] — ``<num>_per_<denom>``
        - ``diff``: list[[a, b]] — ``<a>_minus_<b>`` (numeric or day-delta)
        - ``product``: list[[a, b]] — ``<a>_x_<b>``
        - ``bin``: {col: [edges]} — ``<col>_bin`` (categorical)
        - ``date_parts``: list[str] — ``<col>_{year,month,day,dayofweek,is_weekend}``
        - ``onehot``: list[str] — one-hot encode (top 20 + _other), drops source column
        - ``target_mean``: {col: target_col} — leave-one-out target encoding
        - ``interact_top_k``: {"k": int, "importance_csv": str} — pairwise products
          of top-k features from a prior run's feature_importance.csv

    label : str or None
        Target column name. If provided, transformations referencing it are
        rejected to prevent leakage.
    output_dir : str
        Base directory for the engineered CSV.

    Returns
    -------
    dict with keys:
        engineered_csv   : path to new CSV — pass to tool_train
        new_features     : list of columns created
        dropped_features : list of source columns removed (from onehot)
        warnings         : list of issues (NaNs introduced, cardinality caps, etc.)
        spec_path        : path to transformations.json for reproducibility
    """
    run_dir = make_run_dir(output_dir, prefix="fe")
    df = pd.read_csv(csv_path)
    out, report = apply_transformations(df, transformations, label=label)

    engineered_csv = Path(run_dir) / "data.csv"
    out.to_csv(engineered_csv, index=False)

    spec_path = Path(run_dir) / "transformations.json"
    with open(spec_path, "w") as f:
        json.dump({"source_csv": csv_path, "label": label, "spec": transformations}, f, indent=2)

    return {
        "engineered_csv": str(engineered_csv),
        "new_features": report["new_features"],
        "dropped_features": report["dropped_features"],
        "warnings": report["warnings"],
        "spec_path": str(spec_path),
    }


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


def tool_detect_leakage(
    csv_path: str,
    label: str,
    threshold: float = 0.95,
    sample_size: int = 5000,
    seed: int = 42,
) -> dict:
    """Detect features that are suspiciously predictive of the target.

    Trains a depth-3 decision tree on each feature individually and scores
    it against the target. Any feature that alone achieves a score above
    ``threshold`` is almost certainly leaking — either a direct copy of the
    target, a derived variant (e.g., log of target), or a proxy computed
    from the target after the fact (e.g., "outcome_category" derived from
    the outcome column).

    Runs in seconds — use this BEFORE any AutoGluon training run to avoid
    wasting time optimizing a leaky model.

    Problem type is auto-detected using the same convention as the training
    pipeline: ≤20 unique label values → classification (tree classifier +
    accuracy), more → regression (tree regressor + R²).

    Parameters
    ----------
    csv_path : str
        Path to the input CSV.
    label : str
        Target column name.
    threshold : float
        Score above which a feature is flagged as leaking (default 0.95).
        Set lower (e.g., 0.85) to catch near-leaks; higher to reduce
        false positives on legitimately strong features.
    sample_size : int
        Number of rows to subsample for the leakage test (default 5000).
        More rows doesn't change the signal — single-feature trees saturate
        quickly.
    seed : int
        Random seed for subsampling and tree training (default 42).

    Returns
    -------
    dict with keys:
        problem_type    : "classification" | "regression"
        label           : target column name
        threshold       : score threshold used
        suspected_leaks : list of {feature, score, reason} sorted worst-first
        all_scores      : list of {feature, score} for every feature
        hints           : list of actionable observations
    """
    from sklearn.model_selection import cross_val_score
    from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

    df = pd.read_csv(csv_path)
    if label not in df.columns:
        raise ValueError(f"Label column '{label}' not in CSV: {list(df.columns)}")

    # Auto-detect problem type using the same rule as data.load_and_prepare
    is_classification = df[label].nunique() <= CLASSIFICATION_CARDINALITY_THRESHOLD
    problem_type = "classification" if is_classification else "regression"

    # Subsample for speed; single-feature trees saturate long before 5000 rows
    if len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=seed).reset_index(drop=True)

    y = df[label]
    # Drop rows where target is NaN — tree models can't score those
    mask = y.notna()
    df = df.loc[mask].reset_index(drop=True)
    y = y.loc[mask].reset_index(drop=True)

    feature_cols = [c for c in df.columns if c != label]
    scores: list[dict] = []

    for col in feature_cols:
        x = df[col]
        # Skip columns that are entirely NaN (no signal to measure)
        if x.isna().all():
            continue

        # Encode non-numeric columns via dense integer codes. Trees don't need
        # one-hot for this test — we're measuring dependence, not accuracy.
        if x.dtype == object or isinstance(x.dtype, pd.CategoricalDtype):
            codes = pd.Categorical(x).codes.astype(float)
            x_encoded: pd.Series = pd.Series(codes).replace(-1, pd.NA)
        else:
            x_encoded = x.astype(float)

        # Drop rows where this feature is NaN
        feat_mask = x_encoded.notna()
        if feat_mask.sum() < 10:
            continue
        x_clean = x_encoded.loc[feat_mask].to_numpy().reshape(-1, 1)
        y_clean = y.loc[feat_mask]

        if is_classification:
            clf = DecisionTreeClassifier(max_depth=3, random_state=seed)
            # Use balanced_accuracy so a feature that merely predicts the
            # majority class (trivially high raw accuracy on imbalanced data)
            # scores near 0.5, not 1.0. Genuine leaks still score near 1.0.
            # 3-fold CV prevents the tree from memorizing the training set.
            cv_scores = cross_val_score(clf, x_clean, y_clean, cv=3, scoring="balanced_accuracy")
            score = float(cv_scores.mean())
        else:
            reg = DecisionTreeRegressor(max_depth=3, random_state=seed)
            cv_scores = cross_val_score(reg, x_clean, y_clean, cv=3, scoring="r2")
            score = float(cv_scores.mean())

        scores.append({"feature": col, "score": round(score, 4)})

    scores.sort(key=lambda s: s["score"], reverse=True)

    suspected = []
    for entry in scores:
        if entry["score"] >= threshold:
            if entry["score"] >= 0.999:
                reason = (
                    "Perfect single-feature predictor — either a direct copy of "
                    "the target, a derived variant (e.g., log(target)), or a "
                    "feature computed after the target was known. If this is real "
                    "data, it's leakage. If this is synthetic/demo data, it may "
                    "be an unrealistically clean signal."
                )
            elif entry["score"] >= 0.98:
                reason = (
                    "Near-perfect single-feature score — likely a derived proxy "
                    "of the target or a post-hoc calculation that shouldn't be "
                    "available at inference time."
                )
            else:
                reason = (
                    "Single-feature score exceeds threshold — investigate how "
                    "this feature was computed and whether it's available at "
                    "inference time."
                )
            suspected.append({**entry, "reason": reason})

    hints = []
    if suspected:
        hints.append(
            f"{len(suspected)} feature(s) individually predict the target above the "
            f"{threshold:.2f} threshold — REMOVE these before training or you will "
            f"train a leaky model."
        )
        # Callers often want to pass the list to tool_train's drop parameter directly
        hints.append(f"Suggested drop list: {[s['feature'] for s in suspected]}")
    elif scores and scores[0]["score"] > 0.85:
        hints.append(
            f"No leaks above {threshold:.2f}, but '{scores[0]['feature']}' scores "
            f"{scores[0]['score']:.2f} on its own — verify it's a legitimate signal "
            f"and not a subtle proxy for the target."
        )

    return {
        "problem_type": problem_type,
        "label": label,
        "threshold": threshold,
        "suspected_leaks": suspected,
        "all_scores": scores,
        "hints": hints,
    }


def tool_deep_profile(csv_path: str, label: str) -> dict:
    """Extended per-feature profiling that maps signals to feature engineering actions.

    Use after ``tool_profile`` (which gives shape/missing/correlations) when
    you want to see per-feature skewness, outliers, and cardinality with
    direct ``tool_engineer_features`` spec suggestions. This is richer than
    ``tool_profile`` — call it when you plan to engineer features.

    Parameters
    ----------
    csv_path : str
    label : str

    Returns
    -------
    dict with keys:
        numeric_features : list of {feature, skew, outlier_pct, recommendation}
        categorical_features : list of {feature, cardinality, top_value, recommendation}
        suggested_transforms : dict ready to pass to tool_engineer_features, e.g.
            {"log": ["price"], "onehot": ["category"], "bin": {}}
        hints : list of summary observations
    """
    from automl_model_training.profile import (
        profile_categorical_features,
        profile_numeric_features,
    )

    df = pd.read_csv(csv_path)
    if label not in df.columns:
        raise ValueError(f"Label column '{label}' not in CSV: {list(df.columns)}")

    numeric_stats = profile_numeric_features(df, label)
    categorical_stats = profile_categorical_features(df, label)

    numeric_features = []
    suggested_log: list[str] = []
    for col_idx, row in numeric_stats.iterrows():
        col = str(col_idx)
        if col == label:
            continue
        skew = float(row.get("skew", 0))
        outlier_pct = float(row.get("outlier_pct", 0))
        # Skew > 1 (or < -1) is moderately-to-highly skewed; log1p typically helps
        # when all values are non-negative. Only recommend log for positive skew.
        recommend = []
        if skew > 1.0 and (df[col] >= 0).all():
            recommend.append("log transform (positive skew, non-negative values)")
            suggested_log.append(col)
        elif skew < -1.0:
            recommend.append("negative skew — consider reflecting + log, or keep as-is")
        if outlier_pct > 5.0:
            recommend.append(f"{outlier_pct:.1f}% outliers — tree models handle these fine")
        numeric_features.append(
            {
                "feature": col,
                "skew": round(skew, 3),
                "outlier_pct": round(outlier_pct, 2),
                "recommendation": "; ".join(recommend) if recommend else "no action",
            }
        )

    categorical_features = []
    suggested_onehot: list[str] = []
    for cat_idx, row in categorical_stats.iterrows():
        col = str(cat_idx)
        if col == label:
            continue
        cardinality = int(row.get("nunique", 0))
        top_value = row.get("top_value", "")
        recommend = []
        if cardinality == 2:
            recommend.append("binary categorical — one-hot is safe")
            suggested_onehot.append(str(col))
        elif 2 < cardinality <= 20:
            recommend.append(f"low cardinality ({cardinality}) — one-hot encode")
            suggested_onehot.append(str(col))
        elif 20 < cardinality <= 100:
            recommend.append(
                f"medium cardinality ({cardinality}) — one-hot with _other bucket or target encode"
            )
        else:
            recommend.append(
                f"high cardinality ({cardinality}) — consider target_mean encoding or drop"
            )
        categorical_features.append(
            {
                "feature": col,
                "cardinality": cardinality,
                "top_value": str(top_value),
                "recommendation": "; ".join(recommend),
            }
        )

    suggested_transforms: dict = {}
    if suggested_log:
        suggested_transforms["log"] = suggested_log
    if suggested_onehot:
        suggested_transforms["onehot"] = suggested_onehot

    hints = []
    if suggested_log:
        hints.append(
            f"{len(suggested_log)} numeric feature(s) are right-skewed — "
            f"pass suggested_transforms to tool_engineer_features for log1p transforms."
        )
    if any(int(f["cardinality"]) > 100 for f in categorical_features):  # type: ignore[call-overload]
        hints.append(
            "High-cardinality categorical(s) present — one-hot would explode columns. "
            "Use target_mean encoding or drop instead."
        )

    return {
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "suggested_transforms": suggested_transforms,
        "hints": hints,
    }


def tool_shap_interactions(run_dir: str, top_k: int = 5) -> dict:
    """Find pairs of features whose SHAP contributions correlate across rows.

    Uses the SHAP values already saved by ``tool_train(..., explain=True)``.
    Does NOT retrain anything.

    A pair of top-K features whose per-row SHAP contributions have high
    correlation suggests they carry redundant or coupled signal about the
    prediction. The LLM can use this to:
    - Drop one of a redundant pair (correlation ~ +1)
    - Engineer a ratio or product feature for a strongly-coupled pair
    - Investigate whether a counter-correlated pair (~ -1) indicates
      a hidden interaction the model is trying to express

    Parameters
    ----------
    run_dir : str
        Training run directory (must contain shap_values.csv — i.e.
        training must have used ``explain=True``).
    top_k : int
        Rank features by mean |SHAP| and analyze the top-k only. Default 5.
        Pairwise table size grows as k*(k-1)/2.

    Returns
    -------
    dict with keys:
        top_features : list of {feature, mean_abs_shap} sorted
        pairs        : list of {feature_a, feature_b, corr, abs_corr}
                       sorted by |corr| desc
        hints        : actionable observations
    """
    path = Path(run_dir)
    shap_path = path / "shap_values.csv"
    summary_path = path / "shap_summary.csv"
    if not shap_path.exists() or not summary_path.exists():
        raise FileNotFoundError(
            f"tool_shap_interactions: missing shap_values.csv or shap_summary.csv in "
            f"{run_dir}. Re-run training with explain=True."
        )

    shap_df = pd.read_csv(shap_path)
    summary = pd.read_csv(summary_path)

    # Top-k features by mean |SHAP|
    summary_sorted = summary.sort_values("mean_abs_shap", ascending=False).head(top_k)
    top_features = summary_sorted.to_dict(orient="records")
    top_feature_names = summary_sorted["feature"].tolist()

    # Restrict SHAP matrix to top-k features that actually exist as columns
    present = [f for f in top_feature_names if f in shap_df.columns]
    if len(present) < 2:
        return {
            "top_features": top_features,
            "pairs": [],
            "hints": ["Fewer than 2 top features found in shap_values.csv — no pairs to analyze"],
        }

    top_shap = shap_df[present]

    pairs = []
    for i, a in enumerate(present):
        for b in present[i + 1 :]:
            col_a = top_shap[a]
            col_b = top_shap[b]
            if col_a.std() == 0 or col_b.std() == 0:
                continue
            corr = float(col_a.corr(col_b))
            pairs.append(
                {
                    "feature_a": a,
                    "feature_b": b,
                    "corr": round(corr, 4),
                    "abs_corr": round(abs(corr), 4),
                }
            )
    pairs.sort(key=lambda p: p["abs_corr"], reverse=True)

    hints = []
    for p in pairs:
        if p["abs_corr"] > 0.7:
            if p["corr"] > 0:
                hints.append(
                    f"'{p['feature_a']}' and '{p['feature_b']}' SHAP values are "
                    f"highly correlated (r={p['corr']}) — they may carry redundant "
                    "signal. Try dropping one, or engineer their ratio/product."
                )
            else:
                hints.append(
                    f"'{p['feature_a']}' and '{p['feature_b']}' SHAP values are "
                    f"strongly counter-correlated (r={p['corr']}) — consider "
                    "engineering their difference or ratio."
                )
    if not hints and pairs:
        hints.append("No strongly interacting pairs among top features (all |r| ≤ 0.7).")

    return {"top_features": top_features, "pairs": pairs, "hints": hints}


def tool_partial_dependence(
    run_dir: str,
    features: list[str] | None = None,
    n_values: int = 20,
    sample_size: int = 200,
    grid_strategy: str = "quantile",
    return_ice: bool = False,
) -> dict:
    r"""Compute partial-dependence curves for selected features.

    Answers "how does feature X affect predictions across its range?" —
    which SHAP importance cannot (SHAP shows magnitude, PDP shows shape).
    Uses the already-trained AutoGluon predictor from ``run_dir``.

    Math
    ----
    For a feature of interest :math:`x_S` and complement features
    :math:`x_C` (everything else), the partial-dependence function is

    .. math::

        \hat{f}_S(x_S)
            = \frac{1}{n} \sum_{i=1}^{n} \hat{f}(x_S,\; x_C^{(i)})

    i.e., for each grid value of the feature, copy the sample rows,
    overwrite the feature column with that value in every row, ask the
    predictor what each row would look like, and average the predictions.

    Implementation is a single batched predictor call per feature: build a
    DataFrame of shape ``(n_values × sample_size, n_features)`` where the
    first ``sample_size`` rows have the feature set to grid value 0, the
    next ``sample_size`` rows to grid value 1, and so on. One
    ``predict`` / ``predict_proba`` call returns all predictions; reshape
    into ``(n_values, sample_size)`` and average over axis 1.

    For classification, the curve is the mean predicted probability of
    the highest-sorted class (same convention as
    ``evaluate/classification.py``). For multiclass, per-class curves are
    also returned in ``per_class_pdp_values``.

    ICE (Individual Conditional Expectation) curves — per-row PDP before
    averaging — are returned in ``ice_values`` when ``return_ice=True``.
    Averaging can hide Simpson's-paradox-like effects where subgroups
    respond oppositely; ICE makes them visible.

    Parameters
    ----------
    run_dir : str
        Training run directory containing AutogluonModels/.
    features : list[str] or None
        Features to compute PDP for. If None, uses the top 5 from
        feature_importance.csv if available, else the first 5 columns.
    n_values : int
        Grid points across numeric feature range (default 20). Ignored
        for categorical features (all observed categories are used).
    sample_size : int
        Number of rows to average over (default 200). Lower = faster.
    grid_strategy : str
        How to place numeric grid points. "quantile" (default) spaces
        points by data density — more resolution where the data lives.
        "linspace" spaces them evenly across the min/max range. Ignored
        for categorical features.
    return_ice : bool
        If True, include per-row ICE curves in the output as
        ``ice_values`` (shape ``n_values × sample_size``). Default False.

    Returns
    -------
    dict with keys:
        feature_curves : list of dicts, one per feature, with:
            feature              : column name
            is_numeric           : bool
            grid_values          : grid points (floats or strings)
            pdp_values           : averaged predictions per grid point
            pdp_std              : std across sample rows per grid point
            per_class_pdp_values : {class_label: [values]} — multiclass only
            ice_values           : n_values × sample_size matrix — if return_ice
        hints          : observations (monotonicity, threshold effects)
    """
    run_path = Path(run_dir)
    model_dir = run_path / "AutogluonModels"
    test_path = run_path / "test_raw.csv"
    if not model_dir.exists():
        raise FileNotFoundError(f"tool_partial_dependence: missing {model_dir}")
    if not test_path.exists():
        raise FileNotFoundError(f"tool_partial_dependence: missing {test_path}")
    if grid_strategy not in {"quantile", "linspace"}:
        raise ValueError(f"grid_strategy must be 'quantile' or 'linspace', got {grid_strategy!r}")

    predictor = load_predictor(str(model_dir))
    test_data = pd.read_csv(test_path)
    label = predictor.label
    feature_cols = [c for c in test_data.columns if c != label]

    # Pick features to analyze
    if features is None:
        imp_path = run_path / "feature_importance.csv"
        if imp_path.exists():
            imp = pd.read_csv(imp_path, index_col=0)
            if "importance" in imp.columns:
                ranked = imp.sort_values("importance", ascending=False).index.tolist()
                features = [f for f in ranked if f in feature_cols][:5]
        if not features:
            features = feature_cols[:5]
    missing = [f for f in features if f not in feature_cols]
    if missing:
        raise ValueError(f"Features not in test data: {missing}")

    # Subsample rows (PDP averages predictions → more rows = smoother, slower)
    sample = test_data.sample(n=min(sample_size, len(test_data)), random_state=42)
    sample_x = sample[feature_cols].reset_index(drop=True)
    n_samples = len(sample_x)

    # Memory guard: batched prediction materializes a (n_values * n_samples) DataFrame.
    # Cap at 100k rows and fall back to the per-grid-value loop above that threshold.
    MAX_BATCH_ROWS = 100_000

    problem_type = predictor.problem_type
    is_classification = problem_type in ("binary", "multiclass")

    feature_curves = []
    for feat in features:
        series = test_data[feat]
        is_numeric = pd.api.types.is_numeric_dtype(series)

        # --- Grid construction -------------------------------------------------
        if is_numeric:
            unique_vals = series.dropna().unique()
            if len(unique_vals) <= n_values:
                # Feature has few distinct values — use them directly, no interpolation
                grid = np.sort(unique_vals).tolist()
            elif grid_strategy == "quantile":
                quantiles = np.linspace(0, 1, n_values)
                grid_arr = np.quantile(series.dropna(), quantiles)
                # Quantile grid can produce duplicates on features with tied values;
                # dedupe and pad with linspace points to reach n_values when needed
                deduped = np.unique(grid_arr)
                if len(deduped) < n_values:
                    deduped = np.unique(
                        np.concatenate([deduped, np.linspace(series.min(), series.max(), n_values)])
                    )
                grid = deduped.tolist()
            else:  # linspace
                grid = np.linspace(series.min(), series.max(), n_values).tolist()
        else:
            grid = series.value_counts().head(n_values).index.tolist()

        # --- Batched prediction ------------------------------------------------
        # Build the (|grid| * n_samples, n_features) perturbed DataFrame: for each
        # grid value v, copy the sample rows and set the feature to v.
        total_rows = len(grid) * n_samples
        if total_rows <= MAX_BATCH_ROWS:
            # Tile sample rows |grid| times (keeps the original dtypes of the other cols)
            batched = pd.concat([sample_x] * len(grid), ignore_index=True)
            # Build the perturbed feature column with the right dtype. For numeric
            # features, preserve the original int dtype when every grid value is an
            # integer; otherwise let it promote (e.g., int → float for non-integer grids).
            # Extension dtypes (e.g., nullable Int64) are left as object and pandas
            # will coerce on assignment.
            grid_col = np.repeat(np.array(grid, dtype=object), n_samples)
            np_dtype = series.dtype if isinstance(series.dtype, np.dtype) else None
            if (
                is_numeric
                and np_dtype is not None
                and np.issubdtype(np_dtype, np.integer)
                and all(float(g).is_integer() for g in grid)
            ):
                grid_col = grid_col.astype(np_dtype)
            batched[feat] = grid_col

            if is_classification:
                proba = predictor.predict_proba(batched)
                # Reshape to (n_values, n_samples, n_classes). Force float dtype:
                # predict_proba can return object columns, which breaks .std().
                pred_matrix = np.asarray(proba.values, dtype=float).reshape(
                    len(grid), n_samples, -1
                )
                class_labels = list(proba.columns)
            else:
                preds = predictor.predict(batched)
                pred_matrix = np.asarray(preds, dtype=float).reshape(len(grid), n_samples)
                class_labels = None
        else:
            # Fallback: per-grid loop (legacy path). Triggered on very large
            # n_values * sample_size to keep memory bounded.
            if is_classification:
                per_value = []
                for v in grid:
                    p = sample_x.copy()
                    p[feat] = v
                    per_value.append(np.asarray(predictor.predict_proba(p).values, dtype=float))
                pred_matrix = np.stack(per_value, axis=0)  # (n_values, n_samples, n_classes)
                class_labels = list(predictor.predict_proba(sample_x.head(1)).columns)
            else:
                per_value = []
                for v in grid:
                    p = sample_x.copy()
                    p[feat] = v
                    per_value.append(np.asarray(predictor.predict(p), dtype=float))
                pred_matrix = np.stack(per_value, axis=0)  # (n_values, n_samples)
                class_labels = None

        # --- Aggregate into PDP / ICE -----------------------------------------
        curve: dict = {
            "feature": feat,
            "is_numeric": bool(is_numeric),
            "grid_values": [float(g) if is_numeric else str(g) for g in grid],
        }

        if is_classification and class_labels is not None:
            # Overall curve uses the highest-sorted class (consistent with
            # evaluate/classification.py::pos_label = labels[-1])
            pos_label = sorted(class_labels)[-1]
            pos_idx = class_labels.index(pos_label)
            pos_slice = pred_matrix[:, :, pos_idx]  # (n_values, n_samples)
            curve["pdp_values"] = [round(float(v), 6) for v in pos_slice.mean(axis=1)]
            curve["pdp_std"] = [round(float(v), 6) for v in pos_slice.std(axis=1)]
            # Per-class curves for multiclass; for binary this is just the two
            # complementary curves (cheap, still useful for threshold reasoning)
            curve["per_class_pdp_values"] = {
                str(cls): [round(float(v), 6) for v in pred_matrix[:, :, i].mean(axis=1)]
                for i, cls in enumerate(class_labels)
            }
            if return_ice:
                curve["ice_values"] = pos_slice.round(6).tolist()
        else:
            curve["pdp_values"] = [round(float(v), 6) for v in pred_matrix.mean(axis=1)]
            curve["pdp_std"] = [round(float(v), 6) for v in pred_matrix.std(axis=1)]
            if return_ice:
                curve["ice_values"] = pred_matrix.round(6).tolist()

        feature_curves.append(curve)

    # --- Hints ----------------------------------------------------------------
    hints = []
    for curve in feature_curves:
        raw_vals = curve["pdp_values"]
        assert isinstance(raw_vals, list)
        vals: list[float] = [float(v) for v in raw_vals]
        if len(vals) < 3:
            continue
        if curve["is_numeric"]:
            diffs = np.diff(np.array(vals))
            if np.all(diffs >= 0):
                hints.append(f"'{curve['feature']}' PDP is monotonically increasing.")
            elif np.all(diffs <= 0):
                hints.append(f"'{curve['feature']}' PDP is monotonically decreasing.")
            else:
                span = max(vals) - min(vals)
                if span > 0.05:
                    hints.append(
                        f"'{curve['feature']}' PDP is non-monotonic — the model learned "
                        "a threshold or peak effect; consider binning or polynomial terms."
                    )

    return {"feature_curves": feature_curves, "hints": hints}


def tool_tune_model(
    csv_path: str,
    label: str,
    model_family: str,
    n_trials: int = 20,
    time_limit: int = 300,
    drop: list[str] | None = None,
    test_size: float = 0.2,
    seed: int = 42,
    output_dir: str = "output",
) -> dict:
    """Run targeted hyperparameter tuning on a single model family.

    Use when the leaderboard from ``tool_train`` shows one family dominating
    (e.g., LightGBM wins the ensemble) and you want to squeeze more
    performance out of that family specifically, rather than retraining
    the whole ensemble with a better preset.

    This wraps AutoGluon's built-in ``hyperparameter_tune_kwargs``, which
    uses ray/tune under the hood with Optuna-style random/bayesian search.

    Parameters
    ----------
    csv_path : str
    label : str
    model_family : str
        AutoGluon model key: "GBM" (LightGBM), "XGB", "CAT" (CatBoost),
        "RF" (Random Forest), "XT" (Extra Trees), "NN_TORCH", "FASTAI".
    n_trials : int
        Number of hyperparameter configurations to try (default 20).
    time_limit : int
        Max seconds for the entire tuning run (default 300).
    drop : list[str] or None
        Features to exclude.
    test_size : float
    seed : int
    output_dir : str

    Returns
    -------
    dict with keys:
        run_dir        : path to run outputs
        model_family   : family that was tuned
        score          : best score achieved
        best_hyperparameters : the winning config (if AutoGluon saved it)
        leaderboard    : top-5 rows so the LLM can see per-trial scores
        analysis       : same shape as tool_train's analysis output
    """
    valid_families = {"GBM", "XGB", "CAT", "RF", "XT", "NN_TORCH", "FASTAI"}
    if model_family not in valid_families:
        raise ValueError(
            f"model_family '{model_family}' not supported. Choose from: {sorted(valid_families)}"
        )

    run_dir = make_run_dir(output_dir, prefix=f"tune_{model_family.lower()}")

    train_raw, test_raw, _, _, _ = load_and_prepare(
        csv_path=csv_path,
        label=label,
        features_to_drop=drop or [],
        test_size=test_size,
        random_state=seed,
        output_dir=run_dir,
    )

    # AutoGluon API: restrict to the chosen family and pass HPO config.
    # The family key maps to a dict of search-space hyperparameters; empty {}
    # lets AutoGluon use its default search space for that family.
    hyperparameters: dict[str, dict] = {model_family: {}}
    hyperparameter_tune_kwargs = {
        "num_trials": n_trials,
        "scheduler": "local",
        "searcher": "auto",
    }

    train_and_evaluate(
        train_raw=train_raw,
        test_raw=test_raw,
        label=label,
        problem_type=None,
        eval_metric=None,
        time_limit=time_limit,
        preset="medium",  # low-impact default; HPO drives accuracy, not the preset
        output_dir=run_dir,
        hyperparameters=hyperparameters,
        hyperparameter_tune_kwargs=hyperparameter_tune_kwargs,
    )

    score = _extract_metric(run_dir, "score")
    analysis = _read_analysis(run_dir)

    # Read the top 5 leaderboard rows for the LLM
    leaderboard: list[dict] = []
    lb_path = Path(run_dir) / "leaderboard_test.csv"
    if lb_path.exists():
        lb = pd.read_csv(lb_path)
        cols = [c for c in ["model", "score_val", "score_test", "fit_time"] if c in lb.columns]
        leaderboard = lb[cols].head(5).to_dict(orient="records")

    return {
        "run_dir": run_dir,
        "model_family": model_family,
        "score": score,
        "leaderboard": leaderboard,
        "analysis": analysis,
    }


def tool_threshold_sweep(
    run_dir: str,
    n_thresholds: int = 99,
    metrics: list[str] | None = None,
) -> dict:
    """Sweep the binary-classification decision threshold and report metric curves.

    Loads ``test_predictions.csv`` from a training run (always present after a
    binary run) and computes precision, recall, F1, MCC, and balanced accuracy
    at a grid of thresholds. Returns the full curves plus the argmax threshold
    for each metric, so an LLM agent can see the trade-off shape rather than a
    single calibrated number.

    Uses test-set predictions (not OOF probabilities) to stay compatible with
    refit-best predictors, which cannot produce OOF scores. The test set is
    held out from training, so this is still an unbiased threshold estimate.

    Parameters
    ----------
    run_dir : str
        Training run directory containing ``test_predictions.csv``.
    n_thresholds : int
        Number of thresholds to evaluate, spaced evenly in (0, 1). Default 99
        gives a 0.01 grid from 0.01 to 0.99.
    metrics : list[str] or None
        Which metrics to compute. Default: all of
        ``["f1", "precision", "recall", "mcc", "balanced_accuracy"]``.

    Returns
    -------
    dict with keys:
        thresholds : [float]                — grid points
        curves     : {metric: [float]}      — values per threshold
        best       : {metric: {"threshold", "value"}}
        hints      : [str]                  — observations (e.g., "F1-optimal
                                              threshold is within 0.01 of 0.5,
                                              calibration unlikely to help")

    Raises
    ------
    FileNotFoundError : if test_predictions.csv is missing.
    ValueError        : if the run is not binary classification (detected by
                        the prob_<class> columns present).
    """
    from sklearn.metrics import (
        balanced_accuracy_score,
        matthews_corrcoef,
        precision_score,
        recall_score,
    )

    run_path = Path(run_dir)
    preds_path = run_path / "test_predictions.csv"
    if not preds_path.exists():
        raise FileNotFoundError(f"tool_threshold_sweep: missing {preds_path}")

    preds = pd.read_csv(preds_path)
    prob_cols = sorted(c for c in preds.columns if c.startswith("prob_"))
    if len(prob_cols) != 2:
        raise ValueError(
            f"tool_threshold_sweep requires binary classification; found "
            f"{len(prob_cols)} prob_ columns in {preds_path.name}. Use "
            f"tool_read_analysis for non-binary runs."
        )
    if "actual" not in preds.columns:
        raise ValueError(
            f"tool_threshold_sweep: {preds_path.name} is missing 'actual' column; "
            "predictions were produced without ground truth."
        )

    # Positive class = highest-sorted label (same convention as
    # evaluate/classification.py)
    pos_prob_col = prob_cols[-1]  # e.g., "prob_1"
    pos_label_str = pos_prob_col.removeprefix("prob_")
    # Coerce the actual column to match the inferred positive label's dtype
    actual_raw = preds["actual"]
    try:
        pos_label: object = type(actual_raw.iloc[0])(pos_label_str)
    except (ValueError, TypeError):
        pos_label = pos_label_str
    y_true = (actual_raw == pos_label).astype(int).to_numpy()
    y_prob = preds[pos_prob_col].to_numpy(dtype=float)

    if metrics is None:
        metrics = ["f1", "precision", "recall", "mcc", "balanced_accuracy"]
    valid_metrics = {"f1", "precision", "recall", "mcc", "balanced_accuracy"}
    invalid = set(metrics) - valid_metrics
    if invalid:
        raise ValueError(f"Unknown metrics {sorted(invalid)}; valid: {sorted(valid_metrics)}")

    # Build threshold grid — exclude 0 and 1 because at those points many
    # metrics are undefined (division-by-zero) and the curves are uninformative
    thresholds = np.linspace(0, 1, n_thresholds + 2)[1:-1]

    curves: dict[str, list[float]] = {m: [] for m in metrics}
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        if "f1" in metrics:
            # F1 derived from precision and recall to avoid recomputing them twice
            # when both metrics are requested. Guarded for the degenerate case
            # where the predicted set is empty or all positives.
            p = precision_score(y_true, y_pred, zero_division=0)
            r = recall_score(y_true, y_pred, zero_division=0)
            f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
            curves["f1"].append(float(f1))
            if "precision" in metrics:
                curves["precision"].append(float(p))
            if "recall" in metrics:
                curves["recall"].append(float(r))
        else:
            if "precision" in metrics:
                curves["precision"].append(float(precision_score(y_true, y_pred, zero_division=0)))
            if "recall" in metrics:
                curves["recall"].append(float(recall_score(y_true, y_pred, zero_division=0)))
        if "mcc" in metrics:
            curves["mcc"].append(float(matthews_corrcoef(y_true, y_pred)))
        if "balanced_accuracy" in metrics:
            curves["balanced_accuracy"].append(float(balanced_accuracy_score(y_true, y_pred)))

    # Argmax threshold per metric
    best = {}
    for m in metrics:
        vals = np.asarray(curves[m])
        idx = int(np.argmax(vals))
        best[m] = {
            "threshold": round(float(thresholds[idx]), 4),
            "value": round(float(vals[idx]), 6),
        }

    # Hints: call out when the optimum is near 0.5 (calibration is likely
    # a wash) or at the edges (class imbalance / probability miscalibration)
    hints: list[str] = []
    if "f1" in best:
        f1_thresh = best["f1"]["threshold"]
        if abs(f1_thresh - 0.5) < 0.02:
            hints.append(
                f"F1-optimal threshold is {f1_thresh:.2f}, within 0.02 of 0.5 — "
                "threshold calibration is unlikely to materially help; "
                "focus on data/feature work instead."
            )
        elif f1_thresh < 0.2 or f1_thresh > 0.8:
            hints.append(
                f"F1-optimal threshold is {f1_thresh:.2f}, far from 0.5 — "
                "probabilities are poorly calibrated or classes are highly imbalanced. "
                "Consider tool_calibration_curve for a calibration diagnostic."
            )
    # Per-metric disagreement check: precision-optimal vs recall-optimal should
    # rarely be at the same threshold for non-trivial problems; if they are,
    # the model is likely under-fitting.
    if "precision" in best and "recall" in best:
        gap = abs(best["precision"]["threshold"] - best["recall"]["threshold"])
        if gap < 0.05:
            hints.append(
                "Precision and recall peak at nearly the same threshold — the "
                "model has little threshold sensitivity (flat probabilities); "
                "calibration or a larger model may help."
            )

    return {
        "run_dir": run_dir,
        "thresholds": [round(float(t), 4) for t in thresholds],
        "curves": {m: [round(v, 6) for v in vals] for m, vals in curves.items()},
        "best": best,
        "hints": hints,
    }
