"""
AutoGluon tabular model training.

Trains an AutoGluon TabularPredictor on raw (unscaled) data and
evaluates on a held-out test set.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd
from autogluon.tabular import TabularPredictor

from automl_model_training.config import (
    DEFAULT_EVAL_METRIC,
    DEFAULT_LABEL,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_PRESET,
    DEFAULT_PROBLEM_TYPE,
    DEFAULT_RANDOM_STATE,
    DEFAULT_TEST_SIZE,
    DEFAULT_TIME_LIMIT,
    FEATURES_TO_DROP,
    LOW_IMPORTANCE_THRESHOLD,
    RegressionThresholds,
    make_run_dir,
    setup_logging,
)
from automl_model_training.data import load_and_prepare
from automl_model_training.evaluate import (
    analyze_and_recommend,
    analyze_ensemble,
    prune_models,
    recommend_pruning,
    save_classification_artifacts,
    save_explainability_artifacts,
    save_pruning_report,
    save_regression_artifacts,
)
from automl_model_training.experiment import record_experiment
from automl_model_training.profile import (
    compute_correlation_matrix,
    find_highly_correlated_pairs,
    recommend_features_to_drop,
    save_profile_report,
)
from automl_model_training.run_artifacts import extract_metric

logger = logging.getLogger(__name__)


def _read_low_importance_features(output_dir: str) -> list[str]:
    """Return features with near-zero or negative permutation importance."""
    path = Path(output_dir) / "feature_importance.csv"
    if not path.exists():
        return []
    imp = pd.read_csv(path, index_col=0)
    if "importance" not in imp.columns:
        return []
    low = imp[imp["importance"] <= LOW_IMPORTANCE_THRESHOLD]
    return low.index.tolist()


def train_and_evaluate(
    train_raw: pd.DataFrame,
    test_raw: pd.DataFrame,
    label: str,
    problem_type: str | None,
    eval_metric: str | None,
    time_limit: int | None,
    preset: str,
    output_dir: str,
    prune: bool = False,
    explain: bool = False,
    calibrate_threshold: str | None = None,
    hyperparameters: dict | None = None,
    hyperparameter_tune_kwargs: dict | None = None,
    analysis_thresholds: RegressionThresholds | None = None,
) -> TabularPredictor:
    """Fit an AutoGluon TabularPredictor and evaluate on the test set.

    ``analysis_thresholds`` overrides the regression-diagnostics thresholds
    for this run (None = package defaults). Use for targets where the
    defaults are miscalibrated, e.g. hard-ceiling targets with a low
    achievable R².
    """

    output = Path(output_dir)
    model_path = str(output / "AutogluonModels")

    predictor = TabularPredictor(
        label=label,
        problem_type=problem_type,
        eval_metric=eval_metric,
        path=model_path,
        verbosity=2,
    )

    # Build fit kwargs; only pass HPO/hyperparameter args when caller provides
    # them, so default behaviour is unchanged.
    fit_kwargs: dict = {
        "train_data": train_raw,
        "presets": preset,
        "time_limit": time_limit,
        "refit_full": True,
        "set_best_to_refit_full": True,
        "dynamic_stacking": False,
    }
    if hyperparameters is not None:
        fit_kwargs["hyperparameters"] = hyperparameters
    if hyperparameter_tune_kwargs is not None:
        fit_kwargs["hyperparameter_tune_kwargs"] = hyperparameter_tune_kwargs

    predictor.fit(**fit_kwargs)

    # Keep models in memory so leaderboard/evaluate calls don't reload from disk
    predictor.persist()

    # Leaderboard (validation scores from internal CV)
    leaderboard = predictor.leaderboard(extra_info=True)
    leaderboard.to_csv(output / "leaderboard.csv", index=False)
    logger.info("Leaderboard saved → %s", output / "leaderboard.csv")
    logger.debug("%s", leaderboard[["model", "score_val", "fit_time", "pred_time_val"]].to_string())

    original_best = predictor.model_best

    # Evaluate on held-out test set
    logger.info("--- Test-set evaluation ---")
    test_scores = predictor.evaluate(test_raw)
    for metric_name, score in test_scores.items():
        logger.info("  %s: %.6f", metric_name, score)

    test_leaderboard = predictor.leaderboard(test_raw)
    test_leaderboard.to_csv(output / "leaderboard_test.csv", index=False)
    logger.info("Test leaderboard saved → %s", output / "leaderboard_test.csv")

    # Permutation-based importance: measures accuracy drop when each feature is shuffled
    importance = predictor.feature_importance(test_raw)
    importance.to_csv(output / "feature_importance.csv")
    logger.info("Feature importance saved → %s", output / "feature_importance.csv")

    # Model info summary
    model_info = {
        "problem_type": predictor.problem_type,
        "eval_metric": str(predictor.eval_metric),
        "label": label,
        "features": predictor.features(),
        "best_model": predictor.model_best,
        "best_model_before_refit": original_best,
    }

    # Post-fit decision threshold calibration for binary classification
    if calibrate_threshold and predictor.problem_type == "binary":
        threshold = predictor.calibrate_decision_threshold(metric=calibrate_threshold)
        predictor.set_decision_threshold(threshold)
        model_info["decision_threshold"] = threshold
        model_info["calibrated_for_metric"] = calibrate_threshold
        logger.info(
            "Calibrated decision threshold to %.4f for metric '%s'",
            threshold,
            calibrate_threshold,
        )
    elif calibrate_threshold:
        logger.warning(
            "--calibrate-threshold '%s' ignored: decision-threshold calibration "
            "only applies to binary classification (problem type is '%s').",
            calibrate_threshold,
            predictor.problem_type,
        )

    with open(output / "model_info.json", "w") as f:
        json.dump(model_info, f, indent=2)
    logger.info("Model info saved → %s", output / "model_info.json")

    # Problem-type-specific artifacts
    detected = predictor.problem_type
    if detected in ("binary", "multiclass"):
        save_classification_artifacts(predictor, test_raw, label, output)
    elif detected in ("regression", "quantile"):
        save_regression_artifacts(predictor, test_raw, label, output)

    # SHAP explainability (optional). Runs BEFORE analyze_and_recommend so the
    # analysis step can read shap_summary.csv and emit findings that compare
    # SHAP ranking to permutation-importance ranking.
    if explain:
        save_explainability_artifacts(predictor, test_raw, output)

    # Post-training analysis and recommendations. test_scores (from
    # predictor.evaluate above) is persisted into analysis.json as the
    # canonical test-set metric source for the deployed model.
    analyze_and_recommend(
        predictor=predictor,
        train_raw=train_raw,
        test_raw=test_raw,
        leaderboard=leaderboard,
        test_leaderboard=test_leaderboard,
        importance=importance,
        output=output,
        test_scores=test_scores,
        thresholds=analysis_thresholds,
    )

    # Ensemble pruning (optional)
    if prune:
        ensemble_df = analyze_ensemble(predictor, test_raw)
        to_prune = recommend_pruning(ensemble_df)
        pruned = prune_models(predictor, to_prune)
        save_pruning_report(ensemble_df, pruned, output)

    return predictor


def cross_validate(
    data: pd.DataFrame,
    label: str,
    n_folds: int,
    problem_type: str | None,
    eval_metric: str | None,
    time_limit: int | None,
    preset: str,
    output_dir: str,
    random_state: int,
    shuffle: bool = True,
) -> dict:
    """Run k-fold cross-validation and return aggregate scores.

    Trains a separate model per fold, evaluates on the held-out portion,
    and aggregates scores across folds. This is an accuracy estimate only —
    the deployable model is trained afterward by ``train_and_evaluate``.

    When ``shuffle`` is False, folds are contiguous slices in row order.
    This removes interleaving but is NOT forward-chaining: fold 1 still
    validates on the earliest rows while training on later (future) ones,
    so it is not a valid time-series estimate. Use ``backtest`` for causal,
    walk-forward temporal validation.
    """
    from sklearn.model_selection import KFold, StratifiedKFold

    from automl_model_training.config import CLASSIFICATION_CARDINALITY_THRESHOLD

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    # sklearn rejects random_state when shuffle=False
    seed = random_state if shuffle else None

    # Honor an explicit regression/quantile lock; the cardinality heuristic
    # only applies when the problem type is auto-detected. A regression
    # target with few unique values must NOT get stratified folds.
    if problem_type in ("regression", "quantile"):
        is_classification = False
    else:
        is_classification = data[label].nunique() <= CLASSIFICATION_CARDINALITY_THRESHOLD
    if is_classification:
        splitter = StratifiedKFold(n_splits=n_folds, shuffle=shuffle, random_state=seed)
        split_iter = splitter.split(data, data[label])
    else:
        splitter = KFold(n_splits=n_folds, shuffle=shuffle, random_state=seed)
        split_iter = splitter.split(data)

    fold_results: list[dict] = []

    for fold_num, (train_idx, val_idx) in enumerate(split_iter, 1):
        fold_dir = str(output / f"cv_fold_{fold_num}")
        Path(fold_dir).mkdir(parents=True, exist_ok=True)

        train_fold = data.iloc[train_idx].reset_index(drop=True)
        val_fold = data.iloc[val_idx].reset_index(drop=True)

        logger.info("=" * 60)
        logger.info("  CV FOLD %d / %d", fold_num, n_folds)
        logger.info("  Train: %d rows, Val: %d rows", len(train_fold), len(val_fold))
        logger.info("=" * 60)

        predictor = TabularPredictor(
            label=label,
            problem_type=problem_type,
            eval_metric=eval_metric,
            path=str(Path(fold_dir) / "AutogluonModels"),
            verbosity=1,
        )
        # Mirror train_and_evaluate's fit config where it affects scores
        # (dynamic_stacking; auto_stack is left to the preset default in
        # both paths) so fold scores estimate the same pipeline.
        # refit_full is deliberately omitted: folds are estimates, not
        # deployment artifacts, and refitting each fold doubles the cost.
        fold_fit_kwargs: dict = {
            "train_data": train_fold,
            "presets": preset,
            "time_limit": time_limit,
            "dynamic_stacking": False,
        }
        if is_classification:
            # Threshold calibration is meaningless for regression folds
            fold_fit_kwargs["calibrate_decision_threshold"] = "auto"
        predictor.fit(**fold_fit_kwargs)

        scores = predictor.evaluate(val_fold)
        fold_results.append(
            {
                "fold": fold_num,
                "train_rows": len(train_fold),
                "val_rows": len(val_fold),
                "scores": {k: float(v) for k, v in scores.items()},
                "best_model": predictor.model_best,
            }
        )

        for metric_name, score in scores.items():
            logger.info("  Fold %d %s: %.6f", fold_num, metric_name, score)

    # Aggregate scores across folds
    all_metrics = fold_results[0]["scores"].keys()
    agg: dict[str, dict[str, float]] = {}
    for metric in all_metrics:
        values = [f["scores"][metric] for f in fold_results]
        mean = sum(values) / len(values)
        variance = sum((v - mean) ** 2 for v in values) / len(values)
        agg[metric] = {"mean": round(mean, 6), "std": round(variance**0.5, 6)}

    summary = {
        "n_folds": n_folds,
        "total_rows": len(data),
        "aggregate_scores": agg,
        "folds": fold_results,
    }

    with open(output / "cv_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("CV summary saved → %s", output / "cv_summary.json")

    # Print aggregate results
    logger.info("")
    logger.info("=" * 60)
    logger.info("  CROSS-VALIDATION SUMMARY (%d folds)", n_folds)
    logger.info("=" * 60)
    for metric, stats in agg.items():
        logger.info("  %s: %.6f ± %.6f", metric, stats["mean"], stats["std"])
    logger.info("=" * 60)

    return summary


def load_cv_train(
    *,
    csv_path: str,
    label: str,
    output_dir: str,
    features_to_drop: list[str],
    test_size: float,
    seed: int,
    problem_type: str | None,
    eval_metric: str | None,
    time_limit: int | None,
    preset: str,
    cv_folds: int | None = None,
    cv_shuffle: bool = True,
    split_shuffle: bool = True,
    prune: bool = False,
    explain: bool = False,
    calibrate_threshold: str | None = None,
    hyperparameters: dict | None = None,
    hyperparameter_tune_kwargs: dict | None = None,
    analysis_thresholds: RegressionThresholds | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the standard load → optional CV → train_and_evaluate sequence.

    Used by both the CLI ``_run`` and ``tool_train`` so the LLM agent and
    the CLI go through the same code path. Returns the raw train/test
    splits so callers that need them (e.g., for auto-drop retrain) don't
    have to reload the CSV.

    ``split_shuffle=False`` makes the holdout split a contiguous tail slice
    in row order (see ``load_and_prepare``) — pair with ``cv_shuffle=False``
    for ordered data so neither the folds nor the holdout interleave time.
    """
    train_raw, test_raw, _, _, _ = load_and_prepare(
        csv_path=csv_path,
        label=label,
        features_to_drop=features_to_drop,
        test_size=test_size,
        random_state=seed,
        output_dir=output_dir,
        problem_type=problem_type,
        shuffle=split_shuffle,
    )

    if cv_folds is not None:
        # CV runs on the training split only. The held-out test set must
        # stay out of the folds so the final test score is an independent
        # estimate.
        cross_validate(
            data=train_raw,
            label=label,
            n_folds=cv_folds,
            problem_type=problem_type,
            eval_metric=eval_metric,
            time_limit=time_limit,
            preset=preset,
            output_dir=output_dir,
            random_state=seed,
            shuffle=cv_shuffle,
        )

    train_and_evaluate(
        train_raw=train_raw,
        test_raw=test_raw,
        label=label,
        problem_type=problem_type,
        eval_metric=eval_metric,
        time_limit=time_limit,
        preset=preset,
        output_dir=output_dir,
        prune=prune,
        explain=explain,
        calibrate_threshold=calibrate_threshold,
        hyperparameters=hyperparameters,
        hyperparameter_tune_kwargs=hyperparameter_tune_kwargs,
        analysis_thresholds=analysis_thresholds,
    )

    return train_raw, test_raw


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _base_parser(description: str) -> argparse.ArgumentParser:
    """Build an argument parser with the common training flags."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("csv", help="Path to the input CSV file.")
    parser.add_argument(
        "--label",
        default=DEFAULT_LABEL,
        help=f"Name of the target column (default: {DEFAULT_LABEL}).",
    )
    parser.add_argument(
        "--eval-metric",
        default=DEFAULT_EVAL_METRIC,
        help="Evaluation metric (default: auto-detect).",
    )
    parser.add_argument(
        "--preset",
        default=DEFAULT_PRESET,
        help=(
            f"AutoGluon preset: noncommercial, extreme, best, best_v150, high, "
            f"high_v150, good, medium (default: {DEFAULT_PRESET}). noncommercial/extreme "
            f"need a GPU and the matching extra installed."
        ),
    )
    parser.add_argument(
        "--time-limit",
        type=int,
        default=DEFAULT_TIME_LIMIT,
        help="Training time limit in seconds (default: no limit).",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=DEFAULT_TEST_SIZE,
        help=f"Fraction of data for test split (default: {DEFAULT_TEST_SIZE}).",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for all outputs (default: {DEFAULT_OUTPUT_DIR}).",
    )
    parser.add_argument(
        "--drop",
        nargs="*",
        default=FEATURES_TO_DROP,
        help="Feature column names to drop before training.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_RANDOM_STATE,
        help=f"Random seed for reproducibility (default: {DEFAULT_RANDOM_STATE}).",
    )
    parser.add_argument(
        "--prune",
        action="store_true",
        default=False,
        help="Prune underperforming models from the ensemble after training.",
    )
    parser.add_argument(
        "--explain",
        action="store_true",
        default=False,
        help="Compute SHAP values for model explainability after training.",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        default=False,
        help="Profile the dataset before training and auto-apply drop recommendations.",
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=None,
        help="Run k-fold cross-validation before the final train/test run (e.g. 5).",
    )
    parser.add_argument(
        "--cv-no-shuffle",
        action="store_true",
        default=False,
        help="Disable shuffling when building CV folds (folds become contiguous "
        "slices in row order). Only applies with --cv-folds. NOTE: folds are "
        "NOT forward-chaining — early folds still validate on past rows while "
        "training on later (future) ones, so this removes interleaving but is "
        "not a valid time-series estimate. Use the backtest command for "
        "causal, walk-forward validation.",
    )
    parser.add_argument(
        "--no-shuffle-split",
        action="store_true",
        default=False,
        help="Disable shuffling for the train/test holdout split: the last "
        "--test-size fraction of rows (in file order) becomes the test set. "
        "Use for ordered data where a random split would interleave time; "
        "disables stratification. For strict temporal validation, prefer "
        "the backtest command.",
    )
    parser.add_argument(
        "--calibrate-threshold",
        default=None,
        help="Calibrate the binary classification decision threshold for a specific metric "
        "(e.g. f1, balanced_accuracy, mcc). Only applies to binary problems.",
    )
    parser.add_argument(
        "--auto-drop",
        action="store_true",
        default=False,
        help="Train once, drop features with near-zero or negative importance, then retrain.",
    )
    thresholds = parser.add_argument_group(
        "regression analysis thresholds",
        "Per-run overrides for the regression diagnostics in post-training "
        "analysis (defaults come from config). Example: lower "
        "--low-r2-threshold for hard-ceiling targets where a low R² is "
        "close to the achievable maximum.",
    )
    thresholds.add_argument(
        "--low-r2-threshold",
        type=float,
        default=None,
        help="Test R² below this triggers a weak-fit warning "
        "(default: config.REGRESSION_LOW_R2_THRESHOLD).",
    )
    thresholds.add_argument(
        "--residual-bias-t",
        type=float,
        default=None,
        help="|t| = |mean residual| / (std residual / sqrt(n)) above this flags "
        "systematic bias (default: config.RESIDUAL_BIAS_T_THRESHOLD).",
    )
    thresholds.add_argument(
        "--heteroscedasticity-threshold",
        type=float,
        default=None,
        help="|corr(predicted, |residual|)| above this flags heteroscedasticity "
        "(default: config.HETEROSCEDASTICITY_CORR_THRESHOLD).",
    )
    thresholds.add_argument(
        "--target-skew-threshold",
        type=float,
        default=None,
        help="|skew| of the training target above this suggests a log transform "
        "(default: config.TARGET_SKEW_THRESHOLD).",
    )
    verbosity = parser.add_mutually_exclusive_group()
    verbosity.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        default=False,
        help="Enable debug-level logging.",
    )
    verbosity.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        default=False,
        help="Suppress info messages, show warnings and errors only.",
    )
    return parser


def _thresholds_from_args(args: argparse.Namespace) -> RegressionThresholds | None:
    """Build per-run analysis thresholds from CLI overrides (None = defaults)."""
    overrides = {
        "low_r2_threshold": args.low_r2_threshold,
        "residual_bias_t_threshold": args.residual_bias_t,
        "heteroscedasticity_corr_threshold": args.heteroscedasticity_threshold,
        "target_skew_threshold": args.target_skew_threshold,
    }
    overrides = {k: v for k, v in overrides.items() if v is not None}
    return RegressionThresholds(**overrides) if overrides else None


def _run(
    args: argparse.Namespace,
    problem_type: str | None,
    parser: argparse.ArgumentParser,
) -> None:
    """Shared run logic for all CLI entry points."""
    setup_logging(verbose=args.verbose, quiet=args.quiet)

    csv_path = Path(args.csv)
    if not csv_path.exists():
        parser.error(f"CSV file not found: {csv_path}")

    output_dir = make_run_dir(args.output_dir, prefix="train")

    # Profile dataset and auto-apply drop recommendations
    features_to_drop = list(args.drop)
    if args.profile:
        logger.info("--- Profiling dataset before training ---")
        profile_data = pd.read_csv(args.csv)
        corr = compute_correlation_matrix(profile_data, args.label)
        pairs = find_highly_correlated_pairs(corr)
        recs = recommend_features_to_drop(corr, args.label)
        profile_dir = Path(output_dir) / "profile"
        save_profile_report(profile_data, args.label, corr, pairs, recs, profile_dir)
        auto_drops = [r["feature"] for r in recs if r["feature"] not in features_to_drop]
        if auto_drops:
            logger.info("Profile recommends dropping: %s", auto_drops)
            features_to_drop.extend(auto_drops)

    analysis_thresholds = _thresholds_from_args(args)

    load_cv_train(
        csv_path=args.csv,
        label=args.label,
        output_dir=output_dir,
        features_to_drop=features_to_drop,
        test_size=args.test_size,
        seed=args.seed,
        problem_type=problem_type,
        eval_metric=args.eval_metric,
        time_limit=args.time_limit,
        preset=args.preset,
        cv_folds=args.cv_folds,
        cv_shuffle=not args.cv_no_shuffle,
        split_shuffle=not args.no_shuffle_split,
        prune=args.prune,
        explain=args.explain,
        calibrate_threshold=args.calibrate_threshold,
        analysis_thresholds=analysis_thresholds,
    )

    # Auto-drop: read importance from first run, drop bad features, retrain
    if args.auto_drop:
        low_feats = _read_low_importance_features(output_dir)
        new_drops = [f for f in low_feats if f not in features_to_drop]
        if new_drops:
            logger.info("--- Auto-drop: removing %d low-importance features ---", len(new_drops))
            logger.info("  Dropping: %s", new_drops)
            features_to_drop = features_to_drop + new_drops

            output_dir = make_run_dir(args.output_dir, prefix="train_autodrop")
            load_cv_train(
                csv_path=args.csv,
                label=args.label,
                output_dir=output_dir,
                features_to_drop=features_to_drop,
                test_size=args.test_size,
                seed=args.seed,
                problem_type=problem_type,
                eval_metric=args.eval_metric,
                time_limit=args.time_limit,
                preset=args.preset,
                cv_folds=None,  # auto-drop retrain skips CV
                split_shuffle=not args.no_shuffle_split,
                prune=args.prune,
                explain=args.explain,
                calibrate_threshold=args.calibrate_threshold,
                analysis_thresholds=analysis_thresholds,
            )
        else:
            logger.info("--- Auto-drop: no low-importance features found, skipping retrain ---")

    # Record experiment for comparison. The score comes from analysis.json's
    # test_scores (the deployed model's predictor.evaluate output, signed
    # higher-is-better) — the same source the agent and tool layer read.
    model_info_path = Path(output_dir) / "model_info.json"
    metrics: dict = {}
    if model_info_path.exists():
        with open(model_info_path) as f:
            info = json.load(f)
        eval_metric_name = info.get("eval_metric")
        score = extract_metric(output_dir, eval_metric_name or "score")
        if score is not None:
            metrics["best_test_score"] = score
            metrics["score_convention"] = "higher_is_better"
        metrics["best_model"] = info.get("best_model", "")

    record_experiment(
        output_dir=output_dir,
        params={
            "csv": args.csv,
            "label": args.label,
            "problem_type": str(problem_type),
            "eval_metric": str(args.eval_metric),
            "preset": args.preset,
            "time_limit": args.time_limit,
            "test_size": args.test_size,
            "seed": args.seed,
            "prune": args.prune,
            "explain": args.explain,
            "profile": args.profile,
            "drop": features_to_drop,
        },
        metrics=metrics,
    )


def main() -> None:
    parser = _base_parser("Train an AutoGluon tabular model from a CSV file.")
    parser.add_argument(
        "--problem-type",
        default=DEFAULT_PROBLEM_TYPE,
        choices=["binary", "multiclass", "regression", "quantile"],
        help="Problem type (default: auto-detect).",
    )
    args = parser.parse_args()
    _run(args, problem_type=args.problem_type, parser=parser)


def train_binary() -> None:
    """Entry point for binary classification (eval_metric=f1)."""
    parser = _base_parser("Train an AutoGluon binary classification model.")
    parser.set_defaults(eval_metric="f1")
    args = parser.parse_args()
    _run(args, problem_type="binary", parser=parser)


def train_regression() -> None:
    """Entry point for regression (eval_metric=root_mean_squared_error)."""
    parser = _base_parser("Train an AutoGluon regression model.")
    parser.set_defaults(eval_metric="root_mean_squared_error")
    args = parser.parse_args()
    _run(args, problem_type="regression", parser=parser)


if __name__ == "__main__":
    main()
