"""Regression evaluation artifacts saved during training."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from autogluon.tabular import TabularPredictor

logger = logging.getLogger(__name__)


def save_regression_artifacts(
    predictor: TabularPredictor,
    test_raw: pd.DataFrame,
    label: str,
    output: Path,
) -> None:
    """Save regression evaluation artifacts to *output*."""

    y_true = test_raw[label]
    y_pred = predictor.predict(test_raw)
    residuals = y_true - y_pred

    # Predictions CSV (actual, predicted, residual)
    preds_df = pd.DataFrame(
        {
            "actual": y_true,
            "predicted": y_pred,
            "residual": residuals,
        }
    )
    preds_df.to_csv(output / "test_predictions.csv", index=False)
    logger.info("Test predictions saved → %s", output / "test_predictions.csv")

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
    logger.info("Residual stats saved → %s", output / "residual_stats.json")

    # Error distribution (binned residuals for histogram plotting)
    counts, bin_edges = np.histogram(residuals, bins=50)
    hist_df = pd.DataFrame(
        {
            "bin_left": bin_edges[:-1],
            "bin_right": bin_edges[1:],
            "count": counts,
        }
    )
    hist_df.to_csv(output / "residual_distribution.csv", index=False)
    logger.info("Residual distribution saved → %s", output / "residual_distribution.csv")
