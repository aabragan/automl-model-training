"""
AutoGluon prediction script.

Loads a trained TabularPredictor, runs inference on a new CSV dataset,
and saves predictions with problem-type-specific outputs.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd
from autogluon.tabular import TabularDataset, TabularPredictor

from automl_model_training.config import DEFAULT_PREDICTIONS_DIR, make_run_dir, setup_logging
from automl_model_training.drift import detect_drift, save_drift_report
from automl_model_training.evaluate import (
    save_classification_outputs,
    save_regression_outputs,
)

logger = logging.getLogger(__name__)


def load_predictor(model_dir: str) -> TabularPredictor:
    """Load a previously trained AutoGluon predictor."""
    predictor = TabularPredictor.load(model_dir)
    logger.info("Loaded predictor from %s", model_dir)
    logger.info("  problem_type: %s", predictor.problem_type)
    logger.info("  eval_metric:  %s", predictor.eval_metric)
    logger.info("  label:        %s", predictor.label)
    return predictor


def predict_and_save(
    predictor: TabularPredictor,
    data: pd.DataFrame,
    output_dir: str,
    min_confidence: float | None = None,
    train_data: pd.DataFrame | None = None,
    decision_threshold: float | None = None,
) -> pd.DataFrame:
    """Run predictions and save problem-type-specific artifacts."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    label = predictor.label
    problem_type = predictor.problem_type
    # Work on a copy so the caller's DataFrame isn't mutated
    result = data.copy()

    predict_kwargs: dict = {}
    if decision_threshold is not None and problem_type == "binary":
        predict_kwargs["decision_threshold"] = decision_threshold
        logger.info("Using decision threshold: %.4f", decision_threshold)

    result[f"{label}_predicted"] = predictor.predict(data, **predict_kwargs)

    if problem_type in ("binary", "multiclass"):
        save_classification_outputs(predictor, data, result, label, output)
    elif problem_type in ("regression", "quantile"):
        save_regression_outputs(result, label, output)

    # Flag low-confidence rows for human review (classification only)
    if min_confidence is not None and "confidence" in result.columns:
        result["flagged_low_confidence"] = result["confidence"] < min_confidence
        n_flagged = int(result["flagged_low_confidence"].sum())
        logger.info(
            "Flagged %d / %d rows below %.0f%% confidence",
            n_flagged,
            len(result),
            min_confidence * 100,
        )

    # Drift detection: compare prediction data against training distribution
    if train_data is not None:
        drift_results = detect_drift(train_data, data, label)
        if drift_results:
            save_drift_report(drift_results, output)

    # Save merged result
    result.to_csv(output / "predictions.csv", index=False)
    logger.info("Merged predictions saved → %s", output / "predictions.csv")

    # Summary stats
    summary = {
        "problem_type": problem_type,
        "label": label,
        "num_rows": len(data),
        "num_features": len(data.columns),
        "best_model": predictor.model_best,
    }

    # When ground truth is present, evaluate so users can compare against training metrics
    if label in data.columns:
        summary["has_ground_truth"] = True
        scores = predictor.evaluate(data)
        summary["eval_scores"] = {k: float(v) for k, v in scores.items()}
        logger.info("--- Evaluation against ground truth ---")
        for metric_name, score in scores.items():
            logger.info("  %s: %.6f", metric_name, score)
    else:
        summary["has_ground_truth"] = False

    with open(output / "prediction_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("Prediction summary saved → %s", output / "prediction_summary.json")

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run predictions using a trained AutoGluon model.")
    parser.add_argument("csv", help="Path to the prediction CSV file.")
    parser.add_argument(
        "--model-dir",
        required=True,
        help="Path to the trained AutoGluon model directory.",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_PREDICTIONS_DIR,
        help=f"Directory for prediction outputs (default: {DEFAULT_PREDICTIONS_DIR}).",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=None,
        help="Flag classification rows below this confidence threshold (e.g. 0.7).",
    )
    parser.add_argument(
        "--drift-check",
        default=None,
        help="Path to training run directory for drift detection.",
    )
    parser.add_argument(
        "--decision-threshold",
        type=float,
        default=None,
        help="Override the binary classification decision threshold (e.g. 0.3). "
        "Only applies to binary models.",
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
    args = parser.parse_args()

    setup_logging(verbose=args.verbose, quiet=args.quiet)

    csv_path = Path(args.csv)
    if not csv_path.exists():
        parser.error(f"CSV file not found: {csv_path}")

    model_path = Path(args.model_dir)
    if not model_path.exists():
        parser.error(f"Model directory not found: {model_path}")

    output_dir = make_run_dir(args.output_dir, prefix="predict")
    predictor = load_predictor(args.model_dir)
    data = TabularDataset(args.csv)
    logger.info("Loaded %d rows x %d columns from %s", len(data), len(data.columns), args.csv)

    # Load training data for drift detection if requested
    train_data = None
    if args.drift_check:
        train_csv = Path(args.drift_check) / "train_raw.csv"
        if train_csv.exists():
            train_data = pd.read_csv(train_csv)
            logger.info("Loaded training data for drift check → %s", train_csv)
        else:
            logger.warning("train_raw.csv not found in %s — skipping drift check", args.drift_check)

    predict_and_save(
        predictor,
        data,
        output_dir,
        min_confidence=args.min_confidence,
        train_data=train_data,
        decision_threshold=args.decision_threshold,
    )


def predict_binary() -> None:
    """Convenience entry point — identical to main, just explicit naming."""
    main()


def predict_regression() -> None:
    """Convenience entry point — identical to main, just explicit naming."""
    main()


if __name__ == "__main__":
    main()
