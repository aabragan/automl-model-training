"""Model explainability via SHAP values.

Computes SHAP values for the best model in the ensemble and saves
per-feature contribution data, a summary table, and per-row
explanations.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import shap
from autogluon.tabular import TabularPredictor

logger = logging.getLogger(__name__)


def compute_shap_values(
    predictor: TabularPredictor,
    data: pd.DataFrame,
    max_samples: int = 500,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Compute SHAP values using a KernelExplainer.

    Parameters
    ----------
    predictor : TabularPredictor
        Trained AutoGluon predictor.
    data : DataFrame
        Dataset to explain (typically the test set).
    max_samples : int
        Cap on rows to explain (SHAP can be slow on large datasets).

    Returns
    -------
    shap_values : ndarray
        SHAP values array (n_samples, n_features) for regression /
        binary, or (n_samples, n_features, n_classes) for multiclass.
    base_value : ndarray
        Expected value(s) from the explainer.
    feature_names : list[str]
        Feature column names in order.
    """
    label = predictor.label
    features = [c for c in data.columns if c != label]
    X = data[features]

    # KernelExplainer is model-agnostic but slow; cap rows to keep runtime reasonable
    if len(X) > max_samples:
        X = X.sample(n=max_samples, random_state=42)

    problem_type = predictor.problem_type

    # Classification needs probabilities for SHAP; regression uses raw predictions
    if problem_type in ("binary", "multiclass"):

        def predict_fn(x: np.ndarray) -> np.ndarray:
            return np.array(predictor.predict_proba(pd.DataFrame(x, columns=features)).values)

    else:

        def predict_fn(x: np.ndarray) -> np.ndarray:
            return np.array(predictor.predict(pd.DataFrame(x, columns=features)).values)

    # Background sample gives KernelExplainer a baseline distribution to compare against
    bg_size = min(100, len(X))
    background = shap.sample(X, bg_size, random_state=42)

    explainer = shap.KernelExplainer(predict_fn, background)
    shap_values = explainer.shap_values(X)

    # Normalize shape: ensure ndarray
    shap_values = np.array(shap_values)

    # For binary, SHAP returns [class_0, class_1] arrays — keep only the positive class
    # so downstream consumers get a simple (n_samples, n_features) matrix
    if problem_type == "binary" and shap_values.ndim == 3:
        shap_values = shap_values[1]  # (n_samples, n_features)

    base_value = np.array(explainer.expected_value)

    return shap_values, base_value, features


def build_shap_summary(
    shap_values: np.ndarray,
    feature_names: list[str],
) -> pd.DataFrame:
    """Build a summary DataFrame of mean absolute SHAP values per feature.

    For multiclass (3-D array), averages across classes first.
    """
    vals = shap_values
    vals = np.mean(np.abs(vals), axis=2) if vals.ndim == 3 else np.abs(vals)

    mean_abs = np.mean(vals, axis=0)
    summary = pd.DataFrame({"feature": feature_names, "mean_abs_shap": mean_abs}).sort_values(
        "mean_abs_shap", ascending=False
    )
    summary["rank"] = range(1, len(summary) + 1)
    return summary.reset_index(drop=True)


def build_shap_per_row(
    shap_values: np.ndarray,
    data: pd.DataFrame,
    feature_names: list[str],
    max_samples: int = 500,
) -> pd.DataFrame:
    """Build a DataFrame with per-row top contributing features.

    For each row, identifies the top 5 features by absolute SHAP value
    and their direction (positive/negative contribution).
    """
    label_col = [c for c in data.columns if c not in feature_names]
    X = data[feature_names]
    if len(X) > max_samples:
        X = X.sample(n=max_samples, random_state=42)

    vals = shap_values
    if vals.ndim == 3:
        vals = np.mean(vals, axis=2)

    rows: list[dict] = []
    for i in range(len(vals)):
        abs_vals = np.abs(vals[i])
        top_idx = np.argsort(abs_vals)[-5:][::-1]
        top_features = []
        for idx in top_idx:
            top_features.append(
                {
                    "feature": feature_names[idx],
                    "shap_value": round(float(vals[i][idx]), 6),
                    "feature_value": X.iloc[i, idx] if i < len(X) else None,
                }
            )
        row_data: dict = {"row_index": int(X.index[i]) if i < len(X) else i}
        if label_col:
            col = label_col[0]
            if col in data.columns and i < len(X):
                row_data["actual"] = data.loc[X.index[i], col] if X.index[i] in data.index else None
        row_data["top_features"] = top_features
        rows.append(row_data)

    return pd.DataFrame(rows)


def save_explainability_artifacts(
    predictor: TabularPredictor,
    test_data: pd.DataFrame,
    output: Path,
    max_samples: int = 500,
) -> dict:
    """Compute SHAP values and save all explainability artifacts.

    Artifacts saved:
    - ``shap_summary.csv`` — mean |SHAP| per feature, ranked
    - ``shap_values.csv`` — raw SHAP values matrix
    - ``shap_per_row.json`` — top 5 contributing features per row
    - ``shap_metadata.json`` — base values, problem type, sample count

    Returns the summary dict (also saved as shap_metadata.json).
    """
    output.mkdir(parents=True, exist_ok=True)

    logger.info("--- Computing SHAP values ---")
    shap_values, base_value, feature_names = compute_shap_values(
        predictor,
        test_data,
        max_samples=max_samples,
    )

    # Summary table
    summary_df = build_shap_summary(shap_values, feature_names)
    summary_df.to_csv(output / "shap_summary.csv", index=False)
    logger.info("SHAP summary saved → %s", output / "shap_summary.csv")
    logger.info("Top features by mean |SHAP|:")
    for _, row in summary_df.head(10).iterrows():
        logger.info("  %2d. %-30s %.6f", row["rank"], row["feature"], row["mean_abs_shap"])

    # Raw SHAP values matrix
    vals_2d = shap_values
    if vals_2d.ndim == 3:
        vals_2d = np.mean(vals_2d, axis=2)
    shap_df = pd.DataFrame(vals_2d, columns=feature_names)
    shap_df.to_csv(output / "shap_values.csv", index=False)
    logger.info("SHAP values saved  → %s", output / "shap_values.csv")

    # Per-row explanations
    per_row = build_shap_per_row(shap_values, test_data, feature_names, max_samples)
    per_row_records = per_row.to_dict(orient="records")
    with open(output / "shap_per_row.json", "w") as f:
        json.dump(per_row_records, f, indent=2, default=str)
    logger.info("Per-row SHAP saved → %s", output / "shap_per_row.json")

    # Metadata
    metadata = {
        "problem_type": predictor.problem_type,
        "base_value": base_value.tolist() if base_value.ndim > 0 else float(base_value),
        "n_samples_explained": len(vals_2d),
        "n_features": len(feature_names),
        "top_5_features": summary_df.head(5)["feature"].tolist(),
    }
    with open(output / "shap_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info("SHAP metadata saved → %s", output / "shap_metadata.json")

    return metadata
