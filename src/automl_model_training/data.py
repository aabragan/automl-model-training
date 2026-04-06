"""Data loading, splitting, and normalization."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
from autogluon.tabular import TabularDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler

from automl_model_training.config import CLASSIFICATION_CARDINALITY_THRESHOLD

logger = logging.getLogger(__name__)


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

    # Read CSV into TabularDataset
    data = TabularDataset(csv_path)
    logger.info("Loaded %d rows x %d columns from %s", len(data), len(data.columns), csv_path)

    # Drop unwanted features (silently skip any that don't exist)
    cols_to_drop = [c for c in features_to_drop if c in data.columns]
    if cols_to_drop:
        data = data.drop(columns=cols_to_drop)
        logger.info("Dropped features: %s", cols_to_drop)

    # Identify numeric feature columns (exclude label) for scaling
    numeric_cols = [c for c in data.select_dtypes(include="number").columns if c != label]

    # Train / test split (stratify for classification labels)
    is_classification = data[label].nunique() <= CLASSIFICATION_CARDINALITY_THRESHOLD
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
    logger.info("Saved raw splits → %s, %s", output / "train_raw.csv", output / "test_raw.csv")

    # Normalize numeric features with RobustScaler (fit on train only).
    # Saved as artifacts for external analysis — AutoGluon trains on raw data.
    if numeric_cols:
        scaler = RobustScaler()
        train_norm = train_df.copy()
        test_norm = test_df.copy()
        train_norm[numeric_cols] = scaler.fit_transform(train_df[numeric_cols])
        test_norm[numeric_cols] = scaler.transform(test_df[numeric_cols])
    else:
        train_norm = train_df
        test_norm = test_df

    # Save normalized splits
    train_norm.to_csv(output / "train_normalized.csv", index=False)
    test_norm.to_csv(output / "test_normalized.csv", index=False)
    logger.info(
        "Saved normalized splits → %s, %s",
        output / "train_normalized.csv",
        output / "test_normalized.csv",
    )

    return train_df, test_df, train_norm, test_norm, numeric_cols
