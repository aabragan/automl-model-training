"""Compare training runs side by side.

Takes two or more run output directories and produces a consolidated
comparison of metrics, parameters, features, and model families.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd

from automl_model_training.config import setup_logging

logger = logging.getLogger(__name__)


def load_run_summary(run_dir: str) -> dict:
    """Load key artifacts from a training run directory.

    Returns a flat dict with metrics, params, and model info suitable
    for side-by-side comparison.
    """
    path = Path(run_dir)
    summary: dict = {"run_dir": run_dir}

    # model_info.json
    model_info_path = path / "model_info.json"
    if model_info_path.exists():
        with open(model_info_path) as f:
            info = json.load(f)
        summary["problem_type"] = info.get("problem_type", "")
        summary["eval_metric"] = info.get("eval_metric", "")
        summary["best_model"] = info.get("best_model", "")
        summary["n_features"] = len(info.get("features", []))
        summary["features"] = info.get("features", [])

    # analysis.json
    analysis_path = path / "analysis.json"
    if analysis_path.exists():
        with open(analysis_path) as f:
            analysis = json.load(f)
        summary["n_findings"] = len(analysis.get("findings", []))
        summary["n_recommendations"] = len(analysis.get("recommendations", []))

    # leaderboard_test.csv — best model's test score
    test_lb_path = path / "leaderboard_test.csv"
    if test_lb_path.exists():
        test_lb = pd.read_csv(test_lb_path)
        if not test_lb.empty:
            summary["best_test_score"] = float(test_lb.iloc[0].get("score_test", 0))
            summary["n_models"] = len(test_lb)

    # leaderboard.csv — training time
    lb_path = path / "leaderboard.csv"
    if lb_path.exists():
        lb = pd.read_csv(lb_path)
        if not lb.empty and "fit_time" in lb.columns:
            summary["total_fit_time"] = round(float(lb["fit_time"].sum()), 2)

    # feature_importance.csv — top features
    imp_path = path / "feature_importance.csv"
    if imp_path.exists():
        imp = pd.read_csv(imp_path, index_col=0)
        if "importance" in imp.columns and not imp.empty:
            top = imp.nlargest(5, "importance")
            summary["top_5_features"] = top.index.tolist()

    # cv_summary.json if cross-validation was used
    cv_path = path / "cv_summary.json"
    if cv_path.exists():
        with open(cv_path) as f:
            cv = json.load(f)
        summary["cv_folds"] = cv.get("n_folds", 0)
        for metric, stats in cv.get("aggregate_scores", {}).items():
            summary[f"cv_{metric}_mean"] = stats.get("mean")
            summary[f"cv_{metric}_std"] = stats.get("std")

    return summary


def compare_runs(run_dirs: list[str]) -> pd.DataFrame:
    """Build a comparison DataFrame from multiple run directories.

    Each row is a run, columns are metrics and parameters.
    """
    rows = [load_run_summary(d) for d in run_dirs]
    df = pd.DataFrame(rows)

    # Reorder columns for readability
    priority = [
        "run_dir",
        "problem_type",
        "eval_metric",
        "best_model",
        "best_test_score",
        "n_models",
        "n_features",
        "total_fit_time",
    ]
    ordered = [c for c in priority if c in df.columns]
    remaining = [c for c in df.columns if c not in ordered]
    return df[ordered + remaining]


def save_comparison(df: pd.DataFrame, output_path: Path) -> None:
    """Save comparison as CSV and JSON."""
    df.to_csv(output_path / "comparison.csv", index=False)
    logger.info("Comparison CSV saved → %s", output_path / "comparison.csv")

    # JSON version with lists preserved
    records = df.to_dict(orient="records")
    with open(output_path / "comparison.json", "w") as f:
        json.dump(records, f, indent=2, default=str)
    logger.info("Comparison JSON saved → %s", output_path / "comparison.json")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare two or more training runs side by side.")
    parser.add_argument(
        "runs",
        nargs="+",
        help="Paths to training run output directories.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Directory to save comparison files (default: print to stdout).",
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

    # Validate run directories exist
    for run_dir in args.runs:
        if not Path(run_dir).exists():
            parser.error(f"Run directory not found: {run_dir}")

    df = compare_runs(args.runs)

    if df.empty:
        logger.info("No data found in the provided run directories.")
        return

    if args.output:
        output_path = Path(args.output)
        output_path.mkdir(parents=True, exist_ok=True)
        save_comparison(df, output_path)
    else:
        print(df.to_string(index=False))

    logger.info("Compared %d runs", len(df))


if __name__ == "__main__":
    main()
