"""Local experiment tracking.

Records training parameters, metrics, and output paths to a JSONL file
(one JSON object per line). Each training run appends a single entry,
making it easy to compare experiments without external services.
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from automl_model_training.config import setup_logging

logger = logging.getLogger(__name__)

DEFAULT_EXPERIMENT_LOG = "experiments.jsonl"


def record_experiment(
    output_dir: str,
    params: dict,
    metrics: dict,
    experiment_log: str = DEFAULT_EXPERIMENT_LOG,
) -> dict:
    """Append an experiment entry to the JSONL log file.

    Parameters
    ----------
    output_dir : str
        Path to the run's output directory (contains model artifacts).
    params : dict
        Training parameters (label, preset, problem_type, etc.).
    metrics : dict
        Evaluation metrics from the test set.
    experiment_log : str
        Path to the JSONL file. Created if it doesn't exist.

    Returns
    -------
    dict
        The experiment entry that was recorded.
    """
    entry = {
        "timestamp": datetime.now(UTC).isoformat(),
        "output_dir": output_dir,
        "params": params,
        "metrics": metrics,
    }

    # Read model_info.json if it exists for additional context
    model_info_path = Path(output_dir) / "model_info.json"
    if model_info_path.exists():
        with open(model_info_path) as f:
            entry["model_info"] = json.load(f)

    log_path = Path(experiment_log)
    with open(log_path, "a") as f:
        f.write(json.dumps(entry) + "\n")

    logger.info("Experiment recorded → %s", log_path)
    return entry


def load_experiments(experiment_log: str = DEFAULT_EXPERIMENT_LOG) -> list[dict]:
    """Load all experiment entries from the JSONL log."""
    log_path = Path(experiment_log)
    if not log_path.exists():
        return []
    entries = []
    for line in log_path.read_text().strip().splitlines():
        if line.strip():
            entries.append(json.loads(line))
    return entries


def compare_experiments(
    experiment_log: str = DEFAULT_EXPERIMENT_LOG,
    last_n: int | None = None,
) -> pd.DataFrame:
    """Build a comparison DataFrame from experiment entries.

    Parameters
    ----------
    experiment_log : str
        Path to the JSONL file.
    last_n : int or None
        If set, only show the last N experiments.

    Returns
    -------
    DataFrame
        One row per experiment with flattened params and metrics.
    """
    entries = load_experiments(experiment_log)
    if not entries:
        return pd.DataFrame()

    if last_n is not None:
        entries = entries[-last_n:]

    rows = []
    for entry in entries:
        row: dict = {
            "timestamp": entry.get("timestamp", ""),
            "output_dir": entry.get("output_dir", ""),
        }
        for k, v in entry.get("params", {}).items():
            row[f"param_{k}"] = v
        for k, v in entry.get("metrics", {}).items():
            row[f"metric_{k}"] = v
        model_info = entry.get("model_info", {})
        if model_info:
            row["best_model"] = model_info.get("best_model", "")
            row["problem_type"] = model_info.get("problem_type", "")
        rows.append(row)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    """CLI entry point for comparing experiments."""
    parser = argparse.ArgumentParser(
        description="Compare training experiments from the experiment log."
    )
    parser.add_argument(
        "--log",
        default=DEFAULT_EXPERIMENT_LOG,
        help=f"Path to the experiment log file (default: {DEFAULT_EXPERIMENT_LOG}).",
    )
    parser.add_argument(
        "--last",
        type=int,
        default=None,
        help="Show only the last N experiments.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Save comparison to CSV (default: print to stdout).",
    )
    verbosity = parser.add_mutually_exclusive_group()
    verbosity.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        default=False,
        help="Enable debug-level logging.",
    )
    verbosity.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        default=False,
        help="Suppress info messages.",
    )
    args = parser.parse_args()

    setup_logging(verbose=args.verbose, quiet=args.quiet)

    df = compare_experiments(experiment_log=args.log, last_n=args.last)

    if df.empty:
        logger.info("No experiments found in %s", args.log)
        return

    if args.output:
        df.to_csv(args.output, index=False)
        logger.info("Comparison saved → %s", args.output)
    else:
        print(df.to_string(index=False))

    logger.info("Total experiments: %d", len(df))
