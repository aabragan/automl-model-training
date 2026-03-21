"""
AutoGluon prediction script.

Loads a trained TabularPredictor, runs inference on a new CSV dataset,
and saves predictions with problem-type-specific outputs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from autogluon.tabular import TabularDataset, TabularPredictor


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

    # Core predictions
    result[f"{label}_predicted"] = predictor.predict(data)

    if problem_type in ("binary", "multiclass"):
        _save_classification_outputs(predictor, data, result, label, output)
    elif problem_type in ("regression", "quantile"):
        _save_regression_outputs(result, label, output)

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


def _save_classification_outputs(
    predictor: TabularPredictor,
    data: pd.DataFrame,
    result: pd.DataFrame,
    label: str,
    output: Path,
) -> None:
    """Append class probabilities and save classification-specific artifacts."""

    proba = predictor.predict_proba(data)

    # Merge probabilities into result
    for col in proba.columns:
        result[f"prob_{col}"] = proba[col].values

    # Confidence = probability of the predicted class
    predicted = result[f"{label}_predicted"]
    result["confidence"] = [
        proba.at[i, pred] for i, pred in zip(proba.index, predicted)
    ]

    # Probability distribution summary
    prob_stats = proba.describe().T
    prob_stats.to_csv(output / "probability_stats.csv")
    print(f"Probability stats saved → {output / 'probability_stats.csv'}")

    # Prediction distribution (value counts)
    dist = predicted.value_counts().reset_index()
    dist.columns = ["class", "count"]
    dist["percentage"] = (dist["count"] / len(predicted) * 100).round(2)
    dist.to_csv(output / "prediction_distribution.csv", index=False)
    print(f"Prediction distribution saved → {output / 'prediction_distribution.csv'}")

    # If ground truth exists, save confusion matrix and classification report
    if label in data.columns:
        from sklearn.metrics import classification_report, confusion_matrix

        y_true = data[label]
        y_pred = predicted
        labels = sorted(y_true.unique())

        cm = confusion_matrix(y_true, y_pred, labels=labels)
        cm_df = pd.DataFrame(cm, index=labels, columns=labels)
        cm_df.index.name = "actual"
        cm_df.columns.name = "predicted"
        cm_df.to_csv(output / "confusion_matrix.csv")
        print(f"Confusion matrix saved → {output / 'confusion_matrix.csv'}")

        report = classification_report(y_true, y_pred, output_dict=True)
        pd.DataFrame(report).T.to_csv(output / "classification_report.csv")
        print(f"Classification report saved → {output / 'classification_report.csv'}")


def _save_regression_outputs(
    result: pd.DataFrame,
    label: str,
    output: Path,
) -> None:
    """Save regression-specific artifacts."""

    predicted = result[f"{label}_predicted"]

    # Prediction distribution stats
    pred_stats = {
        "mean": float(predicted.mean()),
        "median": float(predicted.median()),
        "std": float(predicted.std()),
        "min": float(predicted.min()),
        "max": float(predicted.max()),
    }

    # If ground truth exists, add residual stats
    if label in result.columns:
        residuals = result[label] - predicted
        result["residual"] = residuals
        pred_stats.update({
            "mean_residual": float(residuals.mean()),
            "mean_absolute_error": float(residuals.abs().mean()),
            "root_mean_squared_error": float(np.sqrt((residuals**2).mean())),
            "r2": float(
                1 - (residuals**2).sum()
                / ((result[label] - result[label].mean()) ** 2).sum()
            ),
        })

    with open(output / "prediction_stats.json", "w") as f:
        json.dump(pred_stats, f, indent=2)
    print(f"Prediction stats saved → {output / 'prediction_stats.json'}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run predictions using a trained AutoGluon model."
    )
    parser.add_argument("csv", help="Path to the prediction CSV file.")
    parser.add_argument(
        "--model-dir",
        required=True,
        help="Path to the trained AutoGluon model directory.",
    )
    parser.add_argument(
        "--output-dir",
        default="predictions_output",
        help="Directory for prediction outputs (default: predictions_output).",
    )
    args = parser.parse_args()

    predictor = load_predictor(args.model_dir)
    data = TabularDataset(args.csv)
    print(f"Loaded {len(data)} rows x {len(data.columns)} columns from {args.csv}")

    predict_and_save(predictor, data, args.output_dir)


def predict_binary() -> None:
    """Convenience entry point — identical to main, just explicit naming."""
    main()


def predict_regression() -> None:
    """Convenience entry point — identical to main, just explicit naming."""
    main()


if __name__ == "__main__":
    main()
