"""Binary / multiclass evaluation artifacts saved during training."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from autogluon.tabular import TabularPredictor
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

logger = logging.getLogger(__name__)


def save_classification_artifacts(
    predictor: TabularPredictor,
    test_raw: pd.DataFrame,
    label: str,
    output: Path,
) -> None:
    """Save binary/multiclass evaluation artifacts to *output*."""

    y_true = test_raw[label]
    y_pred = predictor.predict(test_raw)
    y_proba = predictor.predict_proba(test_raw)

    # Predictions CSV (actual, predicted, probabilities per class)
    preds_df = pd.DataFrame({"actual": y_true, "predicted": y_pred})
    for col in y_proba.columns:
        preds_df[f"prob_{col}"] = y_proba[col].values
    preds_df.to_csv(output / "test_predictions.csv", index=False)
    logger.info("Test predictions saved → %s", output / "test_predictions.csv")

    # Confusion matrix
    labels = sorted(y_true.unique())
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)
    cm_df.index.name = "actual"
    cm_df.columns.name = "predicted"
    cm_df.to_csv(output / "confusion_matrix.csv")
    logger.info("Confusion matrix saved → %s", output / "confusion_matrix.csv")

    # Classification report (precision, recall, f1 per class)
    report = classification_report(y_true, y_pred, output_dict=True)
    pd.DataFrame(report).T.to_csv(output / "classification_report.csv")
    logger.info("Classification report saved → %s", output / "classification_report.csv")

    # ROC curve data + AUC
    pos_label = labels[-1]
    fpr, tpr, thresholds = roc_curve(y_true, y_proba[pos_label], pos_label=pos_label)
    roc_auc = roc_auc_score(y_true, y_proba[pos_label])
    roc_df = pd.DataFrame({"fpr": fpr, "tpr": tpr, "threshold": thresholds})
    roc_df.to_csv(output / "roc_curve.csv", index=False)
    with open(output / "roc_auc.json", "w") as f:
        json.dump({"roc_auc": roc_auc, "pos_label": str(pos_label)}, f, indent=2)
    logger.info("ROC curve saved → %s (AUC=%.6f)", output / "roc_curve.csv", roc_auc)

    # Precision-recall curve data + average precision
    precision, recall, pr_thresholds = precision_recall_curve(
        y_true, y_proba[pos_label], pos_label=pos_label
    )
    avg_precision = average_precision_score(y_true, y_proba[pos_label])
    pr_df = pd.DataFrame(
        {
            "precision": precision,
            "recall": recall,
            "threshold": np.append(pr_thresholds, np.nan),
        }
    )
    pr_df.to_csv(output / "precision_recall_curve.csv", index=False)
    with open(output / "average_precision.json", "w") as f:
        json.dump(
            {"average_precision": avg_precision, "pos_label": str(pos_label)},
            f,
            indent=2,
        )
    logger.info(
        "Precision-recall curve saved → %s (AP=%.6f)",
        output / "precision_recall_curve.csv",
        avg_precision,
    )
