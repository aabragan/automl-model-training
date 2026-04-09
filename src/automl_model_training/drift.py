"""Data drift detection between training and prediction datasets.

Compares numeric feature distributions to flag features that have
shifted significantly since training. Uses Population Stability Index
(PSI) as the primary metric — a standard measure from credit scoring
that quantifies how much a distribution has changed.

PSI interpretation:
  < 0.1  — no significant drift
  0.1–0.25 — moderate drift, worth monitoring
  > 0.25 — significant drift, model may be unreliable
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# PSI thresholds (standard industry values from credit risk modeling)
PSI_MODERATE = 0.1
PSI_SIGNIFICANT = 0.25
N_BINS = 10


def compute_psi(expected: pd.Series, actual: pd.Series, n_bins: int = N_BINS) -> float:
    """Compute Population Stability Index between two distributions.

    Bins the expected distribution into quantile-based buckets, then
    measures how much the actual distribution deviates from those
    same buckets.
    """
    # Use expected quantiles as bin edges so bins reflect training distribution
    edges = np.quantile(expected.dropna(), np.linspace(0, 1, n_bins + 1))
    edges = np.unique(edges)  # collapse duplicate edges from low-cardinality features
    if len(edges) < 2:
        return 0.0

    expected_counts = np.histogram(expected.dropna(), bins=edges)[0]
    actual_counts = np.histogram(actual.dropna(), bins=edges)[0]

    # Convert to proportions, clip to avoid log(0)
    expected_pct = np.clip(expected_counts / expected_counts.sum(), 1e-6, None)
    actual_pct = np.clip(actual_counts / actual_counts.sum(), 1e-6, None)

    psi = float(np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct)))
    return round(psi, 6)


def detect_drift(
    train_data: pd.DataFrame,
    predict_data: pd.DataFrame,
    label: str,
) -> list[dict]:
    """Compare feature distributions between training and prediction data.

    Returns a list of per-feature drift reports, sorted by PSI descending.
    Only numeric features present in both datasets are compared.
    """
    # Compare only numeric features that exist in both datasets (exclude label)
    train_numeric = set(train_data.select_dtypes(include="number").columns) - {label}
    predict_numeric = set(predict_data.select_dtypes(include="number").columns) - {label}
    shared = sorted(train_numeric & predict_numeric)

    if not shared:
        logger.warning("No shared numeric features to compare for drift detection")
        return []

    results: list[dict] = []
    for col in shared:
        train_col = train_data[col].dropna()
        predict_col = predict_data[col].dropna()

        if train_col.empty or predict_col.empty:
            continue

        psi = compute_psi(train_col, predict_col)
        train_mean = float(train_col.mean())
        predict_mean = float(predict_col.mean())
        train_std = float(train_col.std())
        predict_std = float(predict_col.std())

        # Relative mean shift as a percentage
        mean_shift_pct = (
            abs(predict_mean - train_mean) / abs(train_mean) * 100 if train_mean != 0 else 0.0
        )

        if psi >= PSI_SIGNIFICANT:
            status = "significant_drift"
        elif psi >= PSI_MODERATE:
            status = "moderate_drift"
        else:
            status = "no_drift"

        results.append(
            {
                "feature": col,
                "psi": psi,
                "status": status,
                "train_mean": round(train_mean, 6),
                "predict_mean": round(predict_mean, 6),
                "mean_shift_pct": round(mean_shift_pct, 2),
                "train_std": round(train_std, 6),
                "predict_std": round(predict_std, 6),
            }
        )

    return sorted(results, key=lambda r: r["psi"], reverse=True)


def save_drift_report(
    drift_results: list[dict],
    output: Path,
) -> dict:
    """Save drift detection results and return a summary."""
    output.mkdir(parents=True, exist_ok=True)

    n_significant = sum(1 for r in drift_results if r["status"] == "significant_drift")
    n_moderate = sum(1 for r in drift_results if r["status"] == "moderate_drift")
    n_clean = sum(1 for r in drift_results if r["status"] == "no_drift")

    summary = {
        "features_checked": len(drift_results),
        "significant_drift": n_significant,
        "moderate_drift": n_moderate,
        "no_drift": n_clean,
        "drifted_features": [r["feature"] for r in drift_results if r["status"] != "no_drift"],
        "details": drift_results,
    }

    with open(output / "drift_report.json", "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("Drift report saved → %s", output / "drift_report.json")

    # Also save as CSV for easy viewing
    if drift_results:
        pd.DataFrame(drift_results).to_csv(output / "drift_report.csv", index=False)
        logger.info("Drift report CSV saved → %s", output / "drift_report.csv")

    # Log warnings for drifted features
    if n_significant > 0:
        features = [r["feature"] for r in drift_results if r["status"] == "significant_drift"]
        logger.warning(
            "SIGNIFICANT DRIFT detected in %d feature(s): %s — model predictions may be unreliable",
            n_significant,
            features,
        )
    if n_moderate > 0:
        features = [r["feature"] for r in drift_results if r["status"] == "moderate_drift"]
        logger.warning(
            "Moderate drift detected in %d feature(s): %s — monitor prediction quality",
            n_moderate,
            features,
        )
    if n_significant == 0 and n_moderate == 0:
        logger.info("No significant drift detected across %d features", len(drift_results))

    return summary
