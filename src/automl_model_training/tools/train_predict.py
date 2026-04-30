from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from automl_model_training.agent import _extract_metric, _read_analysis
from automl_model_training.config import make_run_dir
from automl_model_training.data import load_and_prepare
from automl_model_training.predict import load_predictor, predict_and_save
from automl_model_training.train import cross_validate, train_and_evaluate

if TYPE_CHECKING:
    import optuna


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


_HIGHER_IS_BETTER_METRICS = {
    "accuracy",
    "balanced_accuracy",
    "roc_auc",
    "f1",
    "f1_macro",
    "f1_micro",
    "f1_weighted",
    "precision",
    "recall",
    "r2",
    "mcc",
    "average_precision",
}


def _suggest_hyperparameters(
    trial: optuna.Trial,
    model_family: str,
) -> dict:
    """Build a concrete hyperparameter dict for one Optuna trial.

    Search spaces are curated per family based on common practice for
    tabular data. Each family returns a dict compatible with AutoGluon's
    ``hyperparameters`` argument (a non-search-space concrete config).
    Keep these conservative — the agent can tune further via custom
    search spaces later.
    """
    if model_family == "GBM":  # LightGBM
        return {
            "num_leaves": trial.suggest_int("num_leaves", 16, 256, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 5, 100, log=True),
            "lambda_l2": trial.suggest_float("lambda_l2", 1e-4, 10.0, log=True),
        }
    if model_family == "XGB":
        return {
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
            "gamma": trial.suggest_float("gamma", 1e-4, 5.0, log=True),
        }
    if model_family == "CAT":  # CatBoost
        return {
            "depth": trial.suggest_int("depth", 4, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 30.0, log=True),
            "random_strength": trial.suggest_float("random_strength", 0.0, 10.0),
            "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 10.0),
        }
    if model_family == "RF":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000, log=True),
            "max_depth": trial.suggest_int("max_depth", 5, 30),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", 0.5, 0.8]),
        }
    if model_family == "XT":  # Extra Trees (same space as RF)
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000, log=True),
            "max_depth": trial.suggest_int("max_depth", 5, 30),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", 0.5, 0.8]),
        }
    if model_family == "NN_TORCH":
        return {
            "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1e-1, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
            "dropout_prob": trial.suggest_float("dropout_prob", 0.0, 0.5),
            "num_layers": trial.suggest_int("num_layers", 2, 5),
            "hidden_size": trial.suggest_int("hidden_size", 64, 512, log=True),
        }
    if model_family == "FASTAI":
        return {
            "lr": trial.suggest_float("lr", 1e-4, 1e-1, log=True),
            "wd": trial.suggest_float("wd", 1e-6, 1e-2, log=True),
            "epochs": trial.suggest_int("epochs", 5, 30),
            "bs": trial.suggest_categorical("bs", [256, 512, 1024]),
        }
    raise ValueError(f"Unsupported model_family for Optuna search: {model_family}")


def _eval_metric_direction(eval_metric: str | None) -> str:
    """Return 'maximize' or 'minimize' for a given AutoGluon eval_metric name."""
    if eval_metric is None:
        return "maximize"  # AutoGluon's default picks higher-is-better by default
    return "maximize" if eval_metric in _HIGHER_IS_BETTER_METRICS else "minimize"


def tool_optuna_tune(
    csv_path: str,
    label: str,
    model_family: str,
    n_trials: int = 20,
    time_limit_per_trial: int = 60,
    eval_metric: str | None = None,
    problem_type: str | None = None,
    drop: list[str] | None = None,
    test_size: float = 0.2,
    seed: int = 42,
    output_dir: str = "output",
    study_name: str | None = None,
    storage: str | None = None,
    pruner: str = "median",
    n_startup_trials: int = 5,
) -> dict:
    """Optuna-driven hyperparameter search for a single AutoGluon model family.

    Runs an external Optuna loop where each trial:
      1. Optuna's TPE sampler proposes a concrete hyperparameter dict
      2. AutoGluon trains that one configuration (no internal HPO search)
      3. Score is read from leaderboard_test.csv and reported to Optuna

    Advantages over ``tool_tune_model`` (which calls AutoGluon's built-in
    ``hyperparameter_tune_kwargs``):

    - TPE is competitive with AutoGluon's bayes/random for tabular HPO.
    - Median pruning terminates worse-than-median trials early, cutting
      wall-clock by 2-3x on typical searches.
    - Study persistence via sqlite: if ``storage='sqlite:///path.db'``
      and ``study_name`` are set, subsequent calls resume the same study
      and the TPE model keeps improving across agent sessions.
    - Per-family search spaces are defined explicitly in code, not
      hidden in AutoGluon defaults.

    Parameters
    ----------
    csv_path, label, drop, test_size, seed, output_dir
        As in ``tool_train``.
    model_family : str
        AutoGluon model key: ``"GBM" | "XGB" | "CAT" | "RF" | "XT" | "NN_TORCH" | "FASTAI"``.
    n_trials : int
        Number of Optuna trials (default 20). Each trial is one full
        AutoGluon fit with a single concrete hyperparameter dict.
    time_limit_per_trial : int
        Seconds per trial (default 60). Total wall-clock is roughly
        ``n_trials * time_limit_per_trial`` minus pruning savings.
    eval_metric : str or None
        AutoGluon eval metric. If None, AutoGluon picks based on
        problem_type. Determines the Optuna study direction.
    problem_type : str or None
        ``"binary" | "multiclass" | "regression" | "quantile"`` or None
        for auto-detect.
    study_name : str or None
        Optuna study name. Used for study persistence — if None, a name
        is derived from ``csv_path + label + model_family``.
    storage : str or None
        Optuna storage URL (e.g., ``"sqlite:///path.db"``). If set with
        ``study_name``, the study persists across calls and the sampler
        keeps learning. If None, an in-memory study is used (no
        persistence).
    pruner : str
        ``"median"`` (default) prunes trials worse than the running
        median. ``"none"`` disables pruning.
    n_startup_trials : int
        Number of random trials before TPE starts modelling. Default 5.

    Returns
    -------
    dict with keys:
        run_dir                — directory of the best trial's run
        model_family
        best_score             — best test-set score achieved (absolute value)
        best_hyperparameters   — winning config as passed to AutoGluon
        direction              — "maximize" or "minimize"
        n_trials_run           — how many trials actually ran
        n_trials_pruned        — how many were pruned early
        param_importances      — Optuna's estimate of which hp mattered
                                 (fANOVA on completed trials, when >= 2 available)
        trial_history          — list of {trial_num, score, params, state}
        study_name
        storage
        hints                  — observations

    Raises
    ------
    ValueError        : unsupported model_family, invalid pruner
    """
    import optuna

    valid_families = {"GBM", "XGB", "CAT", "RF", "XT", "NN_TORCH", "FASTAI"}
    if model_family not in valid_families:
        raise ValueError(
            f"model_family '{model_family}' not supported. Choose from: {sorted(valid_families)}"
        )
    if pruner not in {"median", "none"}:
        raise ValueError(f"pruner must be 'median' or 'none', got {pruner!r}")

    direction = _eval_metric_direction(eval_metric)

    # Default study name keys on the dataset+label+family so sqlite-persisted
    # studies for different problems don't collide
    if study_name is None:
        study_name = f"{Path(csv_path).stem}__{label}__{model_family}"

    # Build the Optuna study; quiet its INFO-level logger
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    sampler = optuna.samplers.TPESampler(seed=seed, n_startup_trials=n_startup_trials)
    pruner_obj: optuna.pruners.BasePruner
    if pruner == "median":
        # Prune trials whose step-0 (only step we report, since AutoGluon
        # doesn't expose intermediate scores) is worse than median
        pruner_obj = optuna.pruners.MedianPruner(
            n_startup_trials=n_startup_trials,
            n_warmup_steps=0,
            interval_steps=1,
        )
    else:
        pruner_obj = optuna.pruners.NopPruner()

    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        load_if_exists=True,  # key to persistence: resume if same name+storage
        direction=direction,
        sampler=sampler,
        pruner=pruner_obj,
    )

    # Track each trial's run_dir so we can return the best one's path without
    # retraining. Optuna's trial.user_attrs persist across trials in the same
    # study even when storage is sqlite.
    trial_history: list[dict] = []

    def objective(trial: optuna.Trial) -> float:
        hp_dict = _suggest_hyperparameters(trial, model_family)
        trial_run_dir = make_run_dir(output_dir, prefix=f"optuna_{model_family.lower()}")

        try:
            train_raw, test_raw, _, _, _ = load_and_prepare(
                csv_path=csv_path,
                label=label,
                features_to_drop=drop or [],
                test_size=test_size,
                random_state=seed,
                output_dir=trial_run_dir,
            )
            train_and_evaluate(
                train_raw=train_raw,
                test_raw=test_raw,
                label=label,
                problem_type=problem_type,
                eval_metric=eval_metric,
                time_limit=time_limit_per_trial,
                preset="medium",  # HPO carries the quality, not the preset
                output_dir=trial_run_dir,
                hyperparameters={model_family: hp_dict},
                hyperparameter_tune_kwargs=None,  # concrete hp — no internal HPO
            )
        except Exception as e:  # noqa: BLE001 — surface the trial failure to Optuna
            trial.set_user_attr("failed", True)
            trial.set_user_attr("error", str(e)[:200])
            trial.set_user_attr("run_dir", trial_run_dir)
            # For maximize, -inf is worst; for minimize, +inf. TPE avoids these.
            return float("-inf") if direction == "maximize" else float("inf")

        score = _extract_metric(trial_run_dir, "score")
        trial.set_user_attr("run_dir", trial_run_dir)
        trial.set_user_attr("score", score if score is not None else float("nan"))
        if score is None:
            return float("-inf") if direction == "maximize" else float("inf")
        # Report once so MedianPruner can act (AutoGluon is one-shot per trial)
        trial.report(score, step=0)
        if trial.should_prune():
            raise optuna.TrialPruned()
        return score

    study.optimize(objective, n_trials=n_trials, catch=(optuna.TrialPruned,))

    # Gather per-trial records after the study finishes
    n_trials_pruned = 0
    for t in study.trials:
        trial_history.append(
            {
                "trial_num": t.number,
                "state": t.state.name,
                "score": t.user_attrs.get("score"),
                "params": t.params,
                "run_dir": t.user_attrs.get("run_dir"),
            }
        )
        if t.state == optuna.trial.TrialState.PRUNED:
            n_trials_pruned += 1

    # Best trial — raises if no trial completed OR if every trial failed
    # and returned ±inf as its objective value
    try:
        best_trial = study.best_trial
    except ValueError as e:
        raise RuntimeError(
            f"tool_optuna_tune: no trial completed successfully in study "
            f"'{study_name}'. All {n_trials} trials either failed or were "
            "pruned. Check time_limit_per_trial and search space."
        ) from e

    best_score = best_trial.value
    # If every trial failed internally, the objective returned ±inf. Surface
    # that as a clear error rather than reporting an infinite "best_score".
    if best_score is None or not np.isfinite(best_score):
        # Collect any recorded error strings for the message
        errors = {t.user_attrs.get("error", "") for t in study.trials if t.user_attrs.get("failed")}
        raise RuntimeError(
            f"tool_optuna_tune: no trial completed successfully in study "
            f"'{study_name}'. All {n_trials} trials failed. "
            f"Representative errors: {sorted(e for e in errors if e)[:3]}"
        )

    best_run_dir = best_trial.user_attrs.get("run_dir", "")

    # Parameter importances: only available with >= 2 completed trials
    param_importances: dict[str, float] = {}
    completed_count = sum(1 for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE)
    if completed_count >= 2:
        try:
            param_importances = optuna.importance.get_param_importances(study)
            param_importances = {k: round(float(v), 4) for k, v in param_importances.items()}
        except Exception:  # noqa: BLE001 — importance estimation is best-effort
            param_importances = {}

    hints: list[str] = []
    if n_trials_pruned > 0:
        savings_pct = n_trials_pruned / n_trials * 100
        hints.append(
            f"Median pruner terminated {n_trials_pruned}/{n_trials} trials early "
            f"(~{savings_pct:.0f}% wall-clock savings)."
        )
    if param_importances:
        top_param = next(iter(param_importances))  # already sorted by importance
        top_val = param_importances[top_param]
        if top_val > 0.5:
            hints.append(
                f"'{top_param}' dominates the search (importance {top_val:.2f}) — "
                "consider narrowing other parameters and searching harder on this one."
            )
    if storage is not None:
        hints.append(
            f"Study persisted to {storage}. Re-run with the same study_name to "
            "resume and extend the TPE model."
        )

    return {
        "run_dir": best_run_dir,
        "model_family": model_family,
        "best_score": abs(best_score) if best_score is not None else None,
        "best_hyperparameters": best_trial.params,
        "direction": direction,
        "n_trials_run": len(study.trials),
        "n_trials_pruned": n_trials_pruned,
        "param_importances": param_importances,
        "trial_history": trial_history,
        "study_name": study_name,
        "storage": storage,
        "hints": hints,
    }
