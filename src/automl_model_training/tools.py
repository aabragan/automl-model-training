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
