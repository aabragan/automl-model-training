"""Ensemble pruning — identify and remove low-value models.

After training, AutoGluon may produce many models (base learners,
stacked layers, weighted ensembles).  Not all of them contribute
meaningfully to the final ensemble.  This module analyses the
leaderboard and optionally deletes underperforming models to reduce
disk footprint and inference latency.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd
from autogluon.tabular import TabularPredictor

logger = logging.getLogger(__name__)


def analyze_ensemble(
    predictor: TabularPredictor,
    test_data: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build a per-model analysis DataFrame.

    Columns returned:
    - model, score_val, score_test (if test_data given), fit_time,
      pred_time_val, is_best, can_infer, contributes_to_best
    """
    lb: pd.DataFrame = predictor.leaderboard(test_data, extra_info=True, silent=True)

    best = predictor.model_best

    # Determine which models the best model depends on
    info = predictor.info()
    model_info = info.get("model_info", {})
    contributing: set[str] = set()
    _collect_dependencies(best, model_info, contributing)

    lb["is_best"] = lb["model"] == best
    lb["contributes_to_best"] = lb["model"].isin(contributing)

    return lb


def recommend_pruning(
    ensemble_df: pd.DataFrame,
    score_threshold_pct: float = 5.0,
) -> list[str]:
    """Return a list of model names safe to prune.

    A model is recommended for pruning when it:
    1. Is NOT the best model.
    2. Does NOT contribute to the best model's dependency chain.
    3. Has a validation score more than *score_threshold_pct* % worse
       than the best model's validation score.

    Parameters
    ----------
    ensemble_df : DataFrame
        Output of :func:`analyze_ensemble`.
    score_threshold_pct : float
        Percentage gap from the best model's score below which a model
        is considered prunable (default 5 %).
    """
    best_row = ensemble_df.loc[ensemble_df["is_best"]]
    if best_row.empty:
        return []

    best_score = float(best_row["score_val"].iloc[0])

    to_prune: list[str] = []
    for _, row in ensemble_df.iterrows():
        if row["is_best"] or row["contributes_to_best"]:
            continue

        score = float(row["score_val"])
        gap_pct = abs(best_score - score) / abs(best_score) * 100 if best_score != 0 else 0.0

        if gap_pct > score_threshold_pct:
            to_prune.append(str(row["model"]))

    return to_prune


def prune_models(
    predictor: TabularPredictor,
    models_to_delete: list[str],
    dry_run: bool = False,
) -> list[str]:
    """Delete the specified models from the predictor.

    Returns the list of models actually deleted (empty on dry_run).
    """
    if not models_to_delete:
        logger.info("No models to prune.")
        return []

    if dry_run:
        logger.info(
            "[dry-run] Would prune %d model(s): %s", len(models_to_delete), models_to_delete
        )
        return []

    predictor.delete_models(
        models_to_delete=models_to_delete,
        allow_delete_cascade=True,
        delete_from_disk=True,
        dry_run=False,
    )
    logger.info("Pruned %d model(s): %s", len(models_to_delete), models_to_delete)
    return models_to_delete


def save_pruning_report(
    ensemble_df: pd.DataFrame,
    pruned: list[str],
    output: Path,
) -> None:
    """Persist the pruning analysis and results."""
    output.mkdir(parents=True, exist_ok=True)

    # Save full ensemble analysis
    cols = [
        c
        for c in ensemble_df.columns
        if c
        in (
            "model",
            "score_val",
            "score_test",
            "fit_time",
            "pred_time_val",
            "is_best",
            "contributes_to_best",
        )
    ]
    ensemble_df[cols].to_csv(output / "ensemble_analysis.csv", index=False)

    report = {
        "total_models": len(ensemble_df),
        "pruned_models": pruned,
        "pruned_count": len(pruned),
        "remaining_count": len(ensemble_df) - len(pruned),
    }
    with open(output / "pruning_report.json", "w") as f:
        json.dump(report, f, indent=2)

    logger.info("Ensemble analysis saved → %s", output / "ensemble_analysis.csv")
    logger.info("Pruning report saved   → %s", output / "pruning_report.json")


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _collect_dependencies(
    model_name: str,
    model_info: dict,
    result: set[str],
) -> None:
    """Recursively collect all models that *model_name* depends on."""
    result.add(model_name)
    info = model_info.get(model_name, {})

    # AutoGluon stores children/dependencies under various keys
    for key in ("children", "child_model_names", "dependencies"):
        deps = info.get(key, [])
        if isinstance(deps, str):
            deps = [deps]
        for dep in deps:
            if dep not in result:
                _collect_dependencies(dep, model_info, result)
