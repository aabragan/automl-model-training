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

logger = logging.getLogger(__name__)


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
) -> TabularPredictor:
    """Fit an AutoGluon TabularPredictor and evaluate on the test set."""

    output = Path(output_dir)
    model_path = str(output / "AutogluonModels")

    predictor = TabularPredictor(
        label=label,
        problem_type=problem_type,
        eval_metric=eval_metric,
        path=model_path,
        verbosity=2,
    )

    predictor.fit(
        train_data=train_raw,
        presets=preset,
        time_limit=time_limit,
        # auto_stack enables multi-layer stacking for better ensemble diversity
        auto_stack=True,
        # Automatically tune the decision threshold for binary classification
        calibrate_decision_threshold="auto",
    )

    # Keep models in memory so leaderboard/evaluate calls don't reload from disk
    predictor.persist()

    # Leaderboard (validation scores from internal CV)
    leaderboard = predictor.leaderboard(extra_info=True)
    leaderboard.to_csv(output / "leaderboard.csv", index=False)
    logger.info("Leaderboard saved → %s", output / "leaderboard.csv")
    logger.debug("%s", leaderboard[["model", "score_val", "fit_time", "pred_time_val"]].to_string())

    # Refit best models on the full training set (no CV holdout) for final deployment
    refit_map = predictor.refit_full()
    logger.info("Refit-full model map: %s", refit_map)

    # Switch to the refit version of the best model for deployment
    original_best = predictor.model_best
    if original_best in refit_map:
        predictor.set_model_best(refit_map[original_best])
        logger.info("Switched best model: %s → %s", original_best, refit_map[original_best])

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

    with open(output / "model_info.json", "w") as f:
        json.dump(model_info, f, indent=2)
    logger.info("Model info saved → %s", output / "model_info.json")

    # Problem-type-specific artifacts
    detected = predictor.problem_type
    if detected in ("binary", "multiclass"):
        save_classification_artifacts(predictor, test_raw, label, output)
    elif detected in ("regression", "quantile"):
        save_regression_artifacts(predictor, test_raw, label, output)

    # Post-training analysis and recommendations
    analyze_and_recommend(
        predictor=predictor,
        train_raw=train_raw,
        test_raw=test_raw,
        leaderboard=leaderboard,
        test_leaderboard=test_leaderboard,
        importance=importance,
        output=output,
    )

    # Ensemble pruning (optional)
    if prune:
        ensemble_df = analyze_ensemble(predictor, test_raw)
        to_prune = recommend_pruning(ensemble_df)
        pruned = prune_models(predictor, to_prune)
        save_pruning_report(ensemble_df, pruned, output)

    # SHAP explainability (optional)
    if explain:
        save_explainability_artifacts(predictor, test_raw, output)

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
) -> dict:
    """Run k-fold cross-validation and return aggregate scores.

    Trains a separate model per fold, evaluates on the held-out portion,
    and aggregates scores across folds. Also trains a final model on all
    data for deployment.
    """
    from sklearn.model_selection import KFold, StratifiedKFold

    from automl_model_training.config import CLASSIFICATION_CARDINALITY_THRESHOLD

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    is_classification = data[label].nunique() <= CLASSIFICATION_CARDINALITY_THRESHOLD
    if is_classification:
        splitter = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
        split_iter = splitter.split(data, data[label])
    else:
        splitter = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
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
        predictor.fit(
            train_data=train_fold,
            presets=preset,
            time_limit=time_limit,
            auto_stack=True,
            calibrate_decision_threshold="auto",
        )

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
        help=f"AutoGluon preset (default: {DEFAULT_PRESET}).",
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
        "--calibrate-threshold",
        default=None,
        help="Calibrate the binary classification decision threshold for a specific metric "
        "(e.g. f1, balanced_accuracy, mcc). Only applies to binary problems.",
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


def _run(args: argparse.Namespace, problem_type: str | None) -> None:
    """Shared run logic for all CLI entry points."""
    setup_logging(verbose=args.verbose, quiet=args.quiet)

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise SystemExit(f"ERROR: CSV file not found: {csv_path}")

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

    train_raw, test_raw, _, _, _ = load_and_prepare(
        csv_path=args.csv,
        label=args.label,
        features_to_drop=features_to_drop,
        test_size=args.test_size,
        random_state=args.seed,
        output_dir=output_dir,
    )

    # Cross-validation before the final train/test run
    if args.cv_folds is not None:
        full_data = pd.concat([train_raw, test_raw], ignore_index=True)
        cross_validate(
            data=full_data,
            label=args.label,
            n_folds=args.cv_folds,
            problem_type=problem_type,
            eval_metric=args.eval_metric,
            time_limit=args.time_limit,
            preset=args.preset,
            output_dir=output_dir,
            random_state=args.seed,
        )

    train_and_evaluate(
        train_raw=train_raw,
        test_raw=test_raw,
        label=args.label,
        problem_type=problem_type,
        eval_metric=args.eval_metric,
        time_limit=args.time_limit,
        preset=args.preset,
        output_dir=output_dir,
        prune=args.prune,
        explain=args.explain,
        calibrate_threshold=args.calibrate_threshold,
    )

    # Record experiment for comparison
    model_info_path = Path(output_dir) / "model_info.json"
    metrics: dict = {}
    if model_info_path.exists():
        with open(model_info_path) as f:
            info = json.load(f)
        # Load test scores from leaderboard_test if available
        test_lb_path = Path(output_dir) / "leaderboard_test.csv"
        if test_lb_path.exists():
            test_lb = pd.read_csv(test_lb_path)
            if not test_lb.empty:
                best_row = test_lb.iloc[0]
                metrics["best_test_score"] = float(best_row.get("score_test", 0))
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
    _run(args, problem_type=args.problem_type)


def train_binary() -> None:
    """Entry point for binary classification (eval_metric=f1)."""
    parser = _base_parser("Train an AutoGluon binary classification model.")
    parser.set_defaults(eval_metric="f1")
    args = parser.parse_args()
    _run(args, problem_type="binary")


def train_regression() -> None:
    """Entry point for regression (eval_metric=root_mean_squared_error)."""
    parser = _base_parser("Train an AutoGluon regression model.")
    parser.set_defaults(eval_metric="root_mean_squared_error")
    args = parser.parse_args()
    _run(args, problem_type="regression")


if __name__ == "__main__":
    main()
