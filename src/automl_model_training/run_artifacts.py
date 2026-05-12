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
    """Extract a specific metric value from model_info or leaderboard."""
    # Try leaderboard_test first for the best model's score
    lb_path = Path(output_dir) / "leaderboard_test.csv"
    if lb_path.exists():
        lb = pd.read_csv(lb_path)
        if not lb.empty and "score_test" in lb.columns:
            return abs(float(lb.iloc[0]["score_test"]))

    # Fallback: read from analysis.json test_scores
    analysis = read_analysis(output_dir)
    test_scores = analysis.get("test_scores", {})
    if metric_name in test_scores:
        return abs(float(test_scores[metric_name]))

    return None
