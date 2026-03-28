"""Classification artifacts generated at prediction time."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
from autogluon.tabular import TabularPredictor

logger = logging.getLogger(__name__)


def save_classification_outputs(
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

    # Confidence = probability of the predicted class (vectorized lookup)
    predicted = result[f"{label}_predicted"]
    result["confidence"] = proba.to_numpy()[range(len(proba)), proba.columns.get_indexer(predicted)]

    # Probability distribution summary
    prob_stats = proba.describe().T
    prob_stats.to_csv(output / "probability_stats.csv")
    logger.info("Probability stats saved → %s", output / "probability_stats.csv")

    # Prediction distribution (value counts)
    dist = predicted.value_counts().reset_index()
    dist.columns = ["class", "count"]
    dist["percentage"] = (dist["count"] / len(predicted) * 100).round(2)
    dist.to_csv(output / "prediction_distribution.csv", index=False)
    logger.info("Prediction distribution saved → %s", output / "prediction_distribution.csv")

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
        logger.info("Confusion matrix saved → %s", output / "confusion_matrix.csv")

        report = classification_report(y_true, y_pred, output_dict=True)
        pd.DataFrame(report).T.to_csv(output / "classification_report.csv")
        logger.info("Classification report saved → %s", output / "classification_report.csv")
