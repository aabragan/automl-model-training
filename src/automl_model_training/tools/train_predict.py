from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from automl_model_training.config import make_run_dir
from automl_model_training.data import load_and_prepare
from automl_model_training.predict import load_predictor, predict_and_save
from automl_model_training.run_artifacts import extract_metric, read_analysis
from automl_model_training.train import load_cv_train, train_and_evaluate


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

    load_cv_train(
        csv_path=csv_path,
        label=label,
        output_dir=run_dir,
        features_to_drop=drop or [],
        test_size=test_size,
        seed=seed,
        problem_type=problem_type,
        eval_metric=eval_metric,
        time_limit=time_limit,
        preset=preset,
        cv_folds=cv_folds,
        prune=prune,
        explain=explain,
        calibrate_threshold=calibrate_threshold,
    )

    score = extract_metric(run_dir, eval_metric or "score")
    analysis = read_analysis(run_dir)

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

    score = extract_metric(run_dir, "score")
    analysis = read_analysis(run_dir)

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
