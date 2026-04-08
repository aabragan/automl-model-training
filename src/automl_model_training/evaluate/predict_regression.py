"""Regression artifacts generated at prediction time."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def save_regression_outputs(
    result: pd.DataFrame,
    label: str,
    output: Path,
) -> None:
    """Save regression-specific prediction artifacts."""

    predicted = result[f"{label}_predicted"]

    pred_stats = {
        "mean": float(predicted.mean()),
        "median": float(predicted.median()),
        "std": float(predicted.std()),
        "min": float(predicted.min()),
        "max": float(predicted.max()),
    }

    # Residual stats only make sense when we have actual values to compare against
    if label in result.columns:
        residuals = result[label] - predicted
        result["residual"] = residuals
        ss_res = (residuals**2).sum()
        ss_tot = ((result[label] - result[label].mean()) ** 2).sum()
        r2 = float(1 - ss_res / ss_tot) if ss_tot != 0 else 0.0
        pred_stats.update(
            {
                "mean_residual": float(residuals.mean()),
                "mean_absolute_error": float(residuals.abs().mean()),
                "root_mean_squared_error": float(np.sqrt((residuals**2).mean())),
                "r2": r2,
            }
        )

    with open(output / "prediction_stats.json", "w") as f:
        json.dump(pred_stats, f, indent=2)
    logger.info("Prediction stats saved → %s", output / "prediction_stats.json")
