"""
AutoGluon prediction script.

Loads a trained TabularPredictor, runs inference on a new CSV dataset,
and saves predictions with problem-type-specific outputs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from autogluon.tabular import TabularDataset, TabularPredictor

from automl_model_training.evaluate import (
    save_classification_outputs,
    save_regression_outputs,
)
from automl_model_training.config import DEFAULT_PREDICTIONS_DIR, make_run_dir


def load_predictor(model_dir: str) -> TabularPredictor:
    """Load a previously trained AutoGluon predictor."""
    predictor = TabularPredictor.load(model_dir)
    print(f"Loaded predictor from {model_dir}")
    print(f"  problem_type: {predictor.problem_type}")
    print(f"  eval_metric:  {predictor.eval_metric}")
    print(f"  label:        {predictor.label}")
    return predictor


def predict_and_save(
    predictor: TabularPredictor,
    data: pd.DataFrame,
    output_dir: str,
) -> pd.DataFrame:
    """Run predictions and save problem-type-specific artifacts."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    label = predictor.label
    problem_type = predictor.problem_type
    result = data.copy()

    result[f"{label}_predicted"] = predictor.predict(data)

    if problem_type in ("binary", "multiclass"):
        save_classification_outputs(predictor, data, result, label, output)
    elif problem_type in ("regression", "quantile"):
        save_regression_outputs(result, label, output)

    # Save merged result
    result.to_csv(output / "predictions.csv", index=False)
    print(f"\nMerged predictions saved → {output / 'predictions.csv'}")

    # Summary stats
    summary = {
        "problem_type": problem_type,
        "label": label,
        "num_rows": len(data),
        "num_features": len(data.columns),
        "best_model": predictor.model_best,
    }

    if label in data.columns:
        summary["has_ground_truth"] = True
        scores = predictor.evaluate(data)
        summary["eval_scores"] = {k: float(v) for k, v in scores.items()}
        print("\n--- Evaluation against ground truth ---")
        for metric_name, score in scores.items():
            print(f"  {metric_name}: {score:.6f}")
    else:
        summary["has_ground_truth"] = False

    with open(output / "prediction_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Prediction summary saved → {output / 'prediction_summary.json'}")

    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run predictions using a trained AutoGluon model."
    )
    parser.add_argument("csv", help="Path to the prediction CSV file.")
    parser.add_argument(
        "--model-dir", required=True,
        help="Path to the trained AutoGluon model directory.",
    )
    parser.add_argument(
        "--output-dir", default=DEFAULT_PREDICTIONS_DIR,
        help=f"Directory for prediction outputs (default: {DEFAULT_PREDICTIONS_DIR}).",
    )
    args = parser.parse_args()

    output_dir = make_run_dir(args.output_dir, prefix="predict")
    predictor = load_predictor(args.model_dir)
    data = TabularDataset(args.csv)
    print(f"Loaded {len(data)} rows x {len(data.columns)} columns from {args.csv}")

    predict_and_save(predictor, data, output_dir)


def predict_binary() -> None:
    """Convenience entry point — identical to main, just explicit naming."""
    main()


def predict_regression() -> None:
    """Convenience entry point — identical to main, just explicit naming."""
    main()


if __name__ == "__main__":
    main()
