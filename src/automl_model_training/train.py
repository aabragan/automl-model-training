"""
AutoGluon tabular model training.

Trains an AutoGluon TabularPredictor on raw (unscaled) data and
evaluates on a held-out test set.
"""

from __future__ import annotations

import argparse
import json
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
        auto_stack=True,
        calibrate_decision_threshold="auto",
    )

    # Leaderboard (validation scores from internal CV)
    leaderboard = predictor.leaderboard(extra_info=True)
    leaderboard.to_csv(output / "leaderboard.csv", index=False)
    print(f"\nLeaderboard saved → {output / 'leaderboard.csv'}")
    print(leaderboard[["model", "score_val", "fit_time", "pred_time_val"]].to_string())

    # Refit best models on full training data
    refit_map = predictor.refit_full()
    print(f"\nRefit-full model map: {refit_map}")

    # Evaluate on held-out test set
    print("\n--- Test-set evaluation ---")
    test_scores = predictor.evaluate(test_raw)
    for metric_name, score in test_scores.items():
        print(f"  {metric_name}: {score:.6f}")

    test_leaderboard = predictor.leaderboard(test_raw)
    test_leaderboard.to_csv(output / "leaderboard_test.csv", index=False)
    print(f"\nTest leaderboard saved → {output / 'leaderboard_test.csv'}")

    # Feature importance (permutation-based)
    importance = predictor.feature_importance(test_raw)
    importance.to_csv(output / "feature_importance.csv")
    print(f"Feature importance saved → {output / 'feature_importance.csv'}")

    # Model info summary
    model_info = {
        "problem_type": predictor.problem_type,
        "eval_metric": str(predictor.eval_metric),
        "label": label,
        "features": predictor.features(),
        "best_model": predictor.model_best,
    }
    with open(output / "model_info.json", "w") as f:
        json.dump(model_info, f, indent=2)
    print(f"Model info saved → {output / 'model_info.json'}")

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
    return parser


def _run(args: argparse.Namespace, problem_type: str | None) -> None:
    """Shared run logic for all CLI entry points."""
    output_dir = make_run_dir(args.output_dir, prefix="train")
    train_raw, test_raw, _, _, _ = load_and_prepare(
        csv_path=args.csv,
        label=args.label,
        features_to_drop=args.drop,
        test_size=args.test_size,
        random_state=DEFAULT_RANDOM_STATE,
        output_dir=output_dir,
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
