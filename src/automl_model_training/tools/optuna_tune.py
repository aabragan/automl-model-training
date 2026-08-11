from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from automl_model_training.config import make_run_dir
from automl_model_training.data import load_and_prepare
from automl_model_training.run_artifacts import extract_metric
from automl_model_training.train import train_and_evaluate

if TYPE_CHECKING:
    import optuna

logger = logging.getLogger(__name__)


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
    """Return the Optuna study direction. Always 'maximize'.

    ``extract_metric`` reports scores in AutoGluon's internal convention,
    where higher is always better (error metrics like RMSE are negated),
    so the study direction is metric-independent.
    """
    del eval_metric  # direction no longer depends on the metric name
    return "maximize"


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
        best_score             — best test-set score (AutoGluon signed
                                 convention: higher is better, RMSE negated)
        best_hyperparameters   — winning config as passed to AutoGluon
        direction              — always "maximize" (scores are signed)
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
            logger.warning(
                "Optuna trial %d (%s) failed: %s", trial.number, model_family, e, exc_info=True
            )
            trial.set_user_attr("failed", True)
            trial.set_user_attr("error", str(e)[:200])
            trial.set_user_attr("run_dir", trial_run_dir)
            # For maximize, -inf is worst; for minimize, +inf. TPE avoids these.
            return float("-inf") if direction == "maximize" else float("inf")

        score = extract_metric(trial_run_dir, "score")
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
        except Exception as e:  # noqa: BLE001 — importance estimation is best-effort
            logger.debug("Optuna param-importance estimation skipped: %s", e)
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
        # AutoGluon signed convention: higher is better; RMSE appears negated
        "best_score": float(best_score) if best_score is not None else None,
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
