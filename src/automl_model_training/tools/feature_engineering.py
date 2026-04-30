from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from automl_model_training.config import make_run_dir
from automl_model_training.feature_engineering import apply_transformations


def tool_engineer_features(
    csv_path: str,
    transformations: dict,
    label: str | None = None,
    output_dir: str = "output",
) -> dict:
    """Apply declarative feature transformations to a CSV.

    Use after ``tool_profile`` to create features the model can't derive on
    its own: log transforms for skewed distributions, ratios for relationships
    between features, date parts for temporal signal. Pass the returned
    ``engineered_csv`` path to ``tool_train`` for the next iteration.

    Parameters
    ----------
    csv_path : str
        Path to the input CSV.
    transformations : dict
        Spec dict. Supported keys and values:

        - ``log``: list[str] — log1p of each column → ``log_<col>``
        - ``sqrt``: list[str] — sqrt of each column → ``sqrt_<col>``
        - ``ratio``: list[[num, denom]] — ``<num>_per_<denom>``
        - ``diff``: list[[a, b]] — ``<a>_minus_<b>`` (numeric or day-delta)
        - ``product``: list[[a, b]] — ``<a>_x_<b>``
        - ``bin``: {col: [edges]} — ``<col>_bin`` (categorical)
        - ``date_parts``: list[str] — ``<col>_{year,month,day,dayofweek,is_weekend}``
        - ``onehot``: list[str] — one-hot encode (top 20 + _other), drops source column
        - ``target_mean``: {col: target_col} — leave-one-out target encoding
        - ``interact_top_k``: {"k": int, "importance_csv": str} — pairwise products
          of top-k features from a prior run's feature_importance.csv

    label : str or None
        Target column name. If provided, transformations referencing it are
        rejected to prevent leakage.
    output_dir : str
        Base directory for the engineered CSV.

    Returns
    -------
    dict with keys:
        engineered_csv   : path to new CSV — pass to tool_train
        new_features     : list of columns created
        dropped_features : list of source columns removed (from onehot)
        warnings         : list of issues (NaNs introduced, cardinality caps, etc.)
        spec_path        : path to transformations.json for reproducibility
    """
    run_dir = make_run_dir(output_dir, prefix="fe")
    df = pd.read_csv(csv_path)
    out, report = apply_transformations(df, transformations, label=label)

    engineered_csv = Path(run_dir) / "data.csv"
    out.to_csv(engineered_csv, index=False)

    spec_path = Path(run_dir) / "transformations.json"
    with open(spec_path, "w") as f:
        json.dump({"source_csv": csv_path, "label": label, "spec": transformations}, f, indent=2)

    return {
        "engineered_csv": str(engineered_csv),
        "new_features": report["new_features"],
        "dropped_features": report["dropped_features"],
        "warnings": report["warnings"],
        "spec_path": str(spec_path),
    }
