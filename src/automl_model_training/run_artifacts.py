"""Read artifacts produced by a training run.

These helpers are used by both the autonomous agent (``agent.py``) and the
LLM tool layer (``tools/``) to inspect a completed run's output directory
without re-running training.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def read_analysis(output_dir: str) -> dict:
    """Read the analysis.json from a training run."""
    path = Path(output_dir) / "analysis.json"
    if path.exists():
        with open(path) as f:
            result: dict = json.load(f)
            return result
    return {}


def extract_metric(output_dir: str, metric_name: str) -> float | None:
    """Extract a test-set metric for the deployed model from ``analysis.json``.

    Reads the ``test_scores`` dict that ``analyze_and_recommend`` persists
    from ``predictor.evaluate(test_raw)`` — the deployed predictor's scores,
    keyed by metric name.

    Values follow AutoGluon's internal convention: **higher is always
    better**. Error metrics like RMSE appear negated (an RMSE of 4.83 is
    stored as -4.83), so callers can compare scores directly without
    knowing the metric's natural direction.

    If ``metric_name`` is not present (e.g. the generic ``"score"``), the
    run's primary eval metric is returned instead. Returns ``None`` when no
    scores are available (e.g. runs from before test_scores was persisted).
    """
    analysis = read_analysis(output_dir)
    test_scores = analysis.get("test_scores") or {}
    if not test_scores:
        return None

    key = metric_name if metric_name in test_scores else analysis.get("eval_metric")
    if key in test_scores:
        value = test_scores[key]
        if value is not None and not pd.isna(value):
            return float(value)
    return None
