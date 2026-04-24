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

import pandas as pd

from automl_model_training.agent import _extract_metric, _read_analysis
from automl_model_training.config import make_run_dir
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
    prob_cols = [c for c in df.columns if c.startswith("prob_")]
    # Confidence = probability assigned to the predicted class. Normalize the
    # class key — when predicted is float64 from CSV, f"prob_{1.0}" wouldn't
    # match the column name "prob_1".

    def _conf(row: pd.Series) -> float:
        pred = row["predicted"]
        candidates = [str(pred)]
        if pd.notna(pred) and isinstance(pred, float) and pred.is_integer():
            candidates.append(str(int(pred)))
        for candidate in candidates:
            if f"prob_{candidate}" in df.columns:
                return float(row[f"prob_{candidate}"])
        return float("nan")

    if prob_cols:
        df["confidence"] = df.apply(_conf, axis=1)
    else:
        df["confidence"] = float("nan")
    df["is_error"] = df["actual"] != df["predicted"]

    if worst:
        # Worst = errors first, then by lowest confidence among errors / correct with low confidence
        ranked = df.sort_values(by=["is_error", "confidence"], ascending=[False, True]).head(n)
    else:
        ranked = df[~df["is_error"]].sort_values("confidence", ascending=False).head(n)

    error_rate = float(df["is_error"].mean())
    errors_only = df[df["is_error"]]
    if len(errors_only):
        class_error_rate = errors_only["actual"].value_counts(normalize=True).round(4).to_dict()
    else:
        class_error_rate = {}
    class_prevalence = df["actual"].value_counts(normalize=True).round(4).to_dict()

    hints = []
    # Flag classes overrepresented among errors vs population
    for cls, err_pct in class_error_rate.items():
        prev = class_prevalence.get(cls, 0)
        if prev > 0 and err_pct > prev * 1.5:
            hints.append(
                f"Class {cls} is overrepresented in errors: "
                f"{err_pct:.1%} of errors vs {prev:.1%} of population"
            )
    # Low-confidence errors → model is uncertain where it fails (improve data)
    # High-confidence errors → model is confidently wrong (check for leakage or label noise)
    if len(errors_only) and prob_cols:
        hi_conf_errors = (errors_only["confidence"] > 0.9).sum()
        if hi_conf_errors:
            hints.append(
                f"{hi_conf_errors} errors have confidence > 0.9 — "
                "check for label noise or leakage in those rows"
            )

    rows = ranked.to_dict(orient="records")
    return {
        "problem_type": problem_type,
        "label": label,
        "rows": rows,
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
    if "residual" not in df.columns:
        df["residual"] = df["actual"] - df["predicted"]
    df["abs_residual"] = df["residual"].abs()
    # Relative error as % of actual — guard against division by zero
    df["residual_pct"] = (df["abs_residual"] / df["actual"].replace(0, pd.NA) * 100).round(2)

    ranked = df.sort_values("abs_residual", ascending=not worst).head(n)

    hints = []
    # Systematic bias check
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
