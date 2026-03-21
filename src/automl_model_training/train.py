"""
AutoGluon tabular model training script.

Reads a CSV, drops specified features, splits/normalizes data,
trains an AutoGluon TabularPredictor, and evaluates on a held-out test set.

AutoGluon trains on raw (unscaled) data — it handles all internal
preprocessing automatically. RobustScaler-normalized copies of the
train/test splits are saved as CSV artifacts for external analysis,
but are NOT fed into the model.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from autogluon.tabular import TabularDataset, TabularPredictor
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler


# ---------------------------------------------------------------------------
# Configuration defaults
# ---------------------------------------------------------------------------
DEFAULT_LABEL = "target"
DEFAULT_TEST_SIZE = 0.2
DEFAULT_RANDOM_STATE = 42
DEFAULT_TIME_LIMIT = None  # no limit — train until all models are complete
DEFAULT_PRESET = "best"
DEFAULT_EVAL_METRIC = None  # auto-detect based on problem type
DEFAULT_PROBLEM_TYPE = None  # auto-detect from label column
DEFAULT_OUTPUT_DIR = "output"

# Features to drop before training (edit as needed)
FEATURES_TO_DROP: list[str] = [
    # "feature_a",
    # "feature_b",
]


def load_and_prepare(
    csv_path: str,
    label: str,
    features_to_drop: list[str],
    test_size: float,
    random_state: int,
    output_dir: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    """Load CSV, drop features, split, normalize, and persist artifacts.

    Returns (train_raw, test_raw, train_normalized, test_normalized, numeric_cols).
    """

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    # 1. Read CSV into TabularDataset
    data = TabularDataset(csv_path)
    print(f"Loaded {len(data)} rows x {len(data.columns)} columns from {csv_path}")

    # 2. Drop unwanted features (silently skip any that don't exist)
    cols_to_drop = [c for c in features_to_drop if c in data.columns]
    if cols_to_drop:
        data = data.drop(columns=cols_to_drop)
        print(f"Dropped features: {cols_to_drop}")

    # Identify numeric feature columns (exclude label) for scaling
    numeric_cols = [
        c for c in data.select_dtypes(include="number").columns if c != label
    ]

    # 3. Train / test split (stratify for classification labels)
    is_classification = data[label].nunique() <= 20  # heuristic
    stratify = data[label] if is_classification else None

    train_df, test_df = train_test_split(
        data,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    # Save raw (pre-normalization) splits
    train_df.to_csv(output / "train_raw.csv", index=False)
    test_df.to_csv(output / "test_raw.csv", index=False)
    print(f"Saved raw splits → {output / 'train_raw.csv'}, {output / 'test_raw.csv'}")

    # Normalize numeric features with RobustScaler (fit on train only)
    # Saved as artifacts for external analysis — AutoGluon trains on raw data.
    scaler = RobustScaler()
    train_norm = train_df.copy()
    test_norm = test_df.copy()
    train_norm[numeric_cols] = scaler.fit_transform(train_df[numeric_cols])
    test_norm[numeric_cols] = scaler.transform(test_df[numeric_cols])

    # Save normalized splits
    train_norm.to_csv(output / "train_normalized.csv", index=False)
    test_norm.to_csv(output / "test_normalized.csv", index=False)
    print(
        f"Saved normalized splits → "
        f"{output / 'train_normalized.csv'}, {output / 'test_normalized.csv'}"
    )

    return train_df, test_df, train_norm, test_norm, numeric_cols


def _save_binary_artifacts(
    predictor: TabularPredictor,
    test_raw: pd.DataFrame,
    label: str,
    output: Path,
) -> None:
    """Save binary-classification-specific evaluation artifacts."""

    y_true = test_raw[label]
    y_pred = predictor.predict(test_raw)
    y_proba = predictor.predict_proba(test_raw)

    # Predictions CSV (actual, predicted, probabilities per class)
    preds_df = pd.DataFrame({"actual": y_true, "predicted": y_pred})
    for col in y_proba.columns:
        preds_df[f"prob_{col}"] = y_proba[col].values
    preds_df.to_csv(output / "test_predictions.csv", index=False)
    print(f"Test predictions saved → {output / 'test_predictions.csv'}")

    # Confusion matrix
    labels = sorted(y_true.unique())
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)
    cm_df.index.name = "actual"
    cm_df.columns.name = "predicted"
    cm_df.to_csv(output / "confusion_matrix.csv")
    print(f"Confusion matrix saved → {output / 'confusion_matrix.csv'}")

    # Classification report (precision, recall, f1 per class)
    report = classification_report(y_true, y_pred, output_dict=True)
    pd.DataFrame(report).T.to_csv(output / "classification_report.csv")
    print(f"Classification report saved → {output / 'classification_report.csv'}")

    # ROC curve data + AUC
    pos_label = labels[-1]  # positive class
    fpr, tpr, thresholds = roc_curve(y_true, y_proba[pos_label], pos_label=pos_label)
    roc_auc = roc_auc_score(y_true, y_proba[pos_label])
    roc_df = pd.DataFrame({"fpr": fpr, "tpr": tpr, "threshold": thresholds})
    roc_df.to_csv(output / "roc_curve.csv", index=False)
    with open(output / "roc_auc.json", "w") as f:
        json.dump({"roc_auc": roc_auc, "pos_label": str(pos_label)}, f, indent=2)
    print(f"ROC curve saved → {output / 'roc_curve.csv'} (AUC={roc_auc:.6f})")

    # Precision-recall curve data + average precision
    precision, recall, pr_thresholds = precision_recall_curve(
        y_true, y_proba[pos_label], pos_label=pos_label
    )
    avg_precision = average_precision_score(y_true, y_proba[pos_label])
    pr_df = pd.DataFrame({
        "precision": precision,
        "recall": recall,
        "threshold": np.append(pr_thresholds, np.nan),
    })
    pr_df.to_csv(output / "precision_recall_curve.csv", index=False)
    with open(output / "average_precision.json", "w") as f:
        json.dump({"average_precision": avg_precision, "pos_label": str(pos_label)}, f, indent=2)
    print(
        f"Precision-recall curve saved → {output / 'precision_recall_curve.csv'} "
        f"(AP={avg_precision:.6f})"
    )


def _save_regression_artifacts(
    predictor: TabularPredictor,
    test_raw: pd.DataFrame,
    label: str,
    output: Path,
) -> None:
    """Save regression-specific evaluation artifacts."""

    y_true = test_raw[label]
    y_pred = predictor.predict(test_raw)
    residuals = y_true - y_pred

    # Predictions CSV (actual, predicted, residual)
    preds_df = pd.DataFrame({
        "actual": y_true,
        "predicted": y_pred,
        "residual": residuals,
    })
    preds_df.to_csv(output / "test_predictions.csv", index=False)
    print(f"Test predictions saved → {output / 'test_predictions.csv'}")

    # Residual statistics
    residual_stats = {
        "mean_residual": float(residuals.mean()),
        "median_residual": float(residuals.median()),
        "std_residual": float(residuals.std()),
        "min_residual": float(residuals.min()),
        "max_residual": float(residuals.max()),
        "mean_absolute_error": float(residuals.abs().mean()),
        "root_mean_squared_error": float(np.sqrt((residuals**2).mean())),
        "r2": float(1 - (residuals**2).sum() / ((y_true - y_true.mean()) ** 2).sum()),
    }
    with open(output / "residual_stats.json", "w") as f:
        json.dump(residual_stats, f, indent=2)
    print(f"Residual stats saved → {output / 'residual_stats.json'}")

    # Error distribution (binned residuals for histogram plotting)
    counts, bin_edges = np.histogram(residuals, bins=50)
    hist_df = pd.DataFrame({
        "bin_left": bin_edges[:-1],
        "bin_right": bin_edges[1:],
        "count": counts,
    })
    hist_df.to_csv(output / "residual_distribution.csv", index=False)
    print(f"Residual distribution saved → {output / 'residual_distribution.csv'}")


def train_and_evaluate(
    train_raw: pd.DataFrame,
    test_raw: pd.DataFrame,
    label: str,
    problem_type: str | None,
    eval_metric: str | None,
    time_limit: int | None,
    preset: str,
    output_dir: str,
) -> TabularPredictor:
    """Fit an AutoGluon TabularPredictor and evaluate on the test set.

    AutoGluon trains on raw (unscaled) data — it handles all internal
    preprocessing automatically.  Passing pre-scaled data can interfere
    with its pipelines and hurt tree-based model accuracy.
    """

    output = Path(output_dir)
    model_path = str(output / "AutogluonModels")

    # 4. Initialize and fit predictor
    predictor = TabularPredictor(
        label=label,
        problem_type=problem_type,  # None → auto-detect
        eval_metric=eval_metric,    # None → auto (accuracy / rmse)
        path=model_path,
        verbosity=2,
    )

    predictor.fit(
        train_data=train_raw,
        presets=preset,
        time_limit=time_limit,
        # Let AutoGluon decide stacking depth via dynamic stacking.
        # 'best' preset already enables auto_stack=True, num_bag_folds=8,
        # num_bag_sets=1, and the zeroshot hyperparameter portfolio.
        # We explicitly set these for clarity / in case a different preset is used:
        auto_stack=True,
        # Calibrate decision threshold for binary classification metrics
        # like f1, balanced_accuracy, mcc (no-op for regression).
        calibrate_decision_threshold="auto",
    )

    # 5. Save the leaderboard (validation scores from internal CV)
    leaderboard = predictor.leaderboard(extra_info=True)
    leaderboard.to_csv(output / "leaderboard.csv", index=False)
    print(f"\nLeaderboard saved → {output / 'leaderboard.csv'}")
    print(leaderboard[["model", "score_val", "fit_time", "pred_time_val"]].to_string())

    # Refit best models on the full training data (collapses bagged folds
    # into single models for faster inference without losing much accuracy).
    refit_map = predictor.refit_full()
    print(f"\nRefit-full model map: {refit_map}")

    # 6. Evaluate on held-out test set (also raw / unscaled)
    print("\n--- Test-set evaluation ---")
    test_scores = predictor.evaluate(test_raw)
    for metric_name, score in test_scores.items():
        print(f"  {metric_name}: {score:.6f}")

    # Leaderboard with test scores
    test_leaderboard = predictor.leaderboard(test_raw)
    test_leaderboard.to_csv(output / "leaderboard_test.csv", index=False)
    print(f"\nTest leaderboard saved → {output / 'leaderboard_test.csv'}")

    # Feature importance (permutation-based)
    importance = predictor.feature_importance(test_raw)
    importance.to_csv(output / "feature_importance.csv")
    print(f"Feature importance saved → {output / 'feature_importance.csv'}")

    # Save model info summary
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
    if detected == "binary":
        _save_binary_artifacts(predictor, test_raw, label, output)
    elif detected in ("regression", "quantile"):
        _save_regression_artifacts(predictor, test_raw, label, output)
    elif detected == "multiclass":
        # Reuse binary helper — it handles multiclass labels gracefully
        _save_binary_artifacts(predictor, test_raw, label, output)

    return predictor


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train an AutoGluon tabular model from a CSV file."
    )
    parser.add_argument("csv", help="Path to the input CSV file.")
    parser.add_argument(
        "--label",
        default=DEFAULT_LABEL,
        help=f"Name of the target column (default: {DEFAULT_LABEL}).",
    )
    parser.add_argument(
        "--problem-type",
        default=DEFAULT_PROBLEM_TYPE,
        choices=["binary", "multiclass", "regression", "quantile"],
        help="Problem type (default: auto-detect).",
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
    args = parser.parse_args()

    train_raw, test_raw, _, _, _ = load_and_prepare(
        csv_path=args.csv,
        label=args.label,
        features_to_drop=args.drop,
        test_size=args.test_size,
        random_state=DEFAULT_RANDOM_STATE,
        output_dir=args.output_dir,
    )

    train_and_evaluate(
        train_raw=train_raw,
        test_raw=test_raw,
        label=args.label,
        problem_type=args.problem_type,
        eval_metric=args.eval_metric,
        time_limit=args.time_limit,
        preset=args.preset,
        output_dir=args.output_dir,
    )


def _run_with_defaults(
    problem_type: str,
    eval_metric: str,
) -> None:
    """Shared helper for the binary / regression convenience entry points."""
    parser = argparse.ArgumentParser(
        description=f"Train an AutoGluon {problem_type} model from a CSV file."
    )
    parser.add_argument("csv", help="Path to the input CSV file.")
    parser.add_argument(
        "--label",
        default=DEFAULT_LABEL,
        help=f"Name of the target column (default: {DEFAULT_LABEL}).",
    )
    parser.add_argument(
        "--eval-metric",
        default=eval_metric,
        help=f"Evaluation metric (default: {eval_metric}).",
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
    args = parser.parse_args()

    train_raw, test_raw, _, _, _ = load_and_prepare(
        csv_path=args.csv,
        label=args.label,
        features_to_drop=args.drop,
        test_size=args.test_size,
        random_state=DEFAULT_RANDOM_STATE,
        output_dir=args.output_dir,
    )

    train_and_evaluate(
        train_raw=train_raw,
        test_raw=test_raw,
        label=args.label,
        problem_type=problem_type,
        eval_metric=args.eval_metric,
        time_limit=args.time_limit,
        preset=args.preset,
        output_dir=args.output_dir,
    )


def train_binary() -> None:
    """Entry point for binary classification (eval_metric=f1)."""
    _run_with_defaults(problem_type="binary", eval_metric="f1")


def train_regression() -> None:
    """Entry point for regression (eval_metric=root_mean_squared_error)."""
    _run_with_defaults(problem_type="regression", eval_metric="root_mean_squared_error")


if __name__ == "__main__":
    main()
