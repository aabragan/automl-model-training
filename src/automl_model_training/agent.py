"""Autonomous training agent.

Iteratively profiles, trains, analyzes results, and adjusts parameters
to reach a target metric threshold. Supports binary classification (F1)
and regression (RMSE) workflows.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd

from automl_model_training.config import (
    DEFAULT_LABEL,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_RANDOM_STATE,
    DEFAULT_TEST_SIZE,
    make_run_dir,
    setup_logging,
)
from automl_model_training.data import load_and_prepare
from automl_model_training.experiment import compare_experiments, record_experiment
from automl_model_training.profile import (
    compute_correlation_matrix,
    find_highly_correlated_pairs,
    recommend_features_to_drop,
    save_profile_report,
)
from automl_model_training.train import train_and_evaluate

logger = logging.getLogger(__name__)

PRESETS_TO_TRY = ["best_quality", "best_v150", "high_quality"]

# Extended presets when tabarena models are installed (GPU required)
PRESETS_WITH_EXTREME = ["extreme", "best_quality", "best_v150", "high_quality"]


def _tabarena_available() -> bool:
    """Check if the tabarena extra is installed for the extreme preset."""
    try:
        import tabpfn  # noqa: F401

        return True
    except ImportError:
        return False


def _profile_and_get_drops(
    csv_path: str,
    label: str,
    output_dir: str,
    threshold: float = 0.90,
) -> list[str]:
    """Profile the dataset and return recommended features to drop."""
    profile_dir = make_run_dir(output_dir, prefix="agent_profile")
    data = pd.read_csv(csv_path)
    logger.info("Profiling %d rows x %d columns", len(data), len(data.columns))

    corr = compute_correlation_matrix(data, label)
    pairs = find_highly_correlated_pairs(corr, threshold)
    recs = recommend_features_to_drop(corr, label, threshold)
    save_profile_report(data, label, corr, pairs, recs, Path(profile_dir))

    drop_list = [r["feature"] for r in recs]
    if drop_list:
        logger.info("Profile recommends dropping: %s", drop_list)
    else:
        logger.info("Profile found no features to drop")
    return drop_list


def _read_analysis(output_dir: str) -> dict:
    """Read the analysis.json from a training run."""
    path = Path(output_dir) / "analysis.json"
    if path.exists():
        with open(path) as f:
            result: dict = json.load(f)
            return result
    return {}


def _read_feature_importance(output_dir: str) -> list[str]:
    """Return features with near-zero or negative importance."""
    path = Path(output_dir) / "feature_importance.csv"
    if not path.exists():
        return []
    imp = pd.read_csv(path, index_col=0)
    if "importance" not in imp.columns:
        return []
    low = imp[imp["importance"] <= 0.001]
    return low.index.tolist()


def _extract_metric(output_dir: str, metric_name: str) -> float | None:
    """Extract a specific metric value from model_info or leaderboard."""
    # Try leaderboard_test first for the best model's score
    lb_path = Path(output_dir) / "leaderboard_test.csv"
    if lb_path.exists():
        lb = pd.read_csv(lb_path)
        if not lb.empty and "score_test" in lb.columns:
            return abs(float(lb.iloc[0]["score_test"]))

    # Fallback: read from analysis.json test_scores
    analysis = _read_analysis(output_dir)
    test_scores = analysis.get("test_scores", {})
    if metric_name in test_scores:
        return abs(float(test_scores[metric_name]))

    return None


def _decide_next_action(
    analysis: dict,
    iteration: int,
    current_drops: list[str],
    current_preset: str,
    presets: list[str] | None = None,
) -> dict:
    """Decide what to change for the next iteration based on analysis findings."""
    if presets is None:
        presets = PRESETS_TO_TRY
    action: dict = {"drops_to_add": [], "preset": current_preset, "reason": ""}

    findings = analysis.get("findings", [])
    recommendations = analysis.get("recommendations", [])

    # Check for low-importance features to drop
    for rec in recommendations:
        rec_lower = rec.lower()
        if "drop" in rec_lower and "feature" in rec_lower:
            action["reason"] = "Analysis recommends dropping low-value features"
            break

    # Check for overfitting
    for finding in findings:
        if "overfit" in finding.lower():
            action["reason"] = "Overfitting detected"
            if current_preset == "best_quality":
                action["preset"] = "high_quality"
                action["reason"] += " — switching to high_quality preset"
            break

    if not action["reason"]:
        action["reason"] = "Trying next preset for comparison"
        try:
            idx = presets.index(current_preset)
            action["preset"] = presets[(idx + 1) % len(presets)]
        except ValueError:
            action["preset"] = presets[0]

    return action


def run_agent(
    csv_path: str,
    label: str,
    problem_type: str,
    eval_metric: str,
    target_metric: str,
    target_value: float,
    max_iterations: int,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    test_size: float = DEFAULT_TEST_SIZE,
    higher_is_better: bool = True,
) -> dict:
    """Run the autonomous training loop.

    Parameters
    ----------
    csv_path : str
        Path to the input CSV.
    label : str
        Target column name.
    problem_type : str
        "binary" or "regression".
    eval_metric : str
        AutoGluon eval metric (e.g., "f1", "root_mean_squared_error").
    target_metric : str
        Metric name to check against target_value.
    target_value : float
        Stop when this metric value is reached.
    max_iterations : int
        Maximum training iterations.
    output_dir : str
        Base output directory.
    test_size : float
        Test split fraction.

    Returns
    -------
    dict
        Summary of the best run.
    """
    logger.info("=" * 60)
    logger.info("  AUTONOMOUS TRAINING AGENT")
    logger.info("=" * 60)
    logger.info("  Problem type: %s", problem_type)
    logger.info("  Eval metric: %s", eval_metric)
    logger.info(
        "  Target: %s %s %.4f", target_metric, ">=" if higher_is_better else "<=", target_value
    )
    logger.info("  Max iterations: %d", max_iterations)
    logger.info("=" * 60)

    # Step 1: Profile and get initial drop recommendations
    drop_features = _profile_and_get_drops(csv_path, label, output_dir)

    best_score: float | None = None
    best_run_dir: str = ""
    presets = PRESETS_WITH_EXTREME if _tabarena_available() else PRESETS_TO_TRY
    current_preset = presets[0]
    logger.info("  Available presets: %s", presets)

    def _is_better(new: float, old: float | None) -> bool:
        if old is None:
            return True
        return new > old if higher_is_better else new < old

    def _target_reached(score: float) -> bool:
        return score >= target_value if higher_is_better else score <= target_value

    for iteration in range(1, max_iterations + 1):
        logger.info("")
        logger.info("=" * 60)
        logger.info("  ITERATION %d / %d", iteration, max_iterations)
        logger.info("  Preset: %s", current_preset)
        logger.info("  Dropping: %s", drop_features or "(none)")
        logger.info("=" * 60)

        # Step 2: Train
        run_dir = make_run_dir(output_dir, prefix=f"agent_iter{iteration}")
        train_raw, test_raw, _, _, _ = load_and_prepare(
            csv_path=csv_path,
            label=label,
            features_to_drop=drop_features,
            test_size=test_size,
            random_state=DEFAULT_RANDOM_STATE,
            output_dir=run_dir,
        )

        train_and_evaluate(
            train_raw=train_raw,
            test_raw=test_raw,
            label=label,
            problem_type=problem_type,
            eval_metric=eval_metric,
            time_limit=None,
            preset=current_preset,
            output_dir=run_dir,
            prune=True,
            explain=False,
        )

        # Record experiment
        record_experiment(
            output_dir=run_dir,
            params={
                "csv": csv_path,
                "label": label,
                "problem_type": problem_type,
                "eval_metric": eval_metric,
                "preset": current_preset,
                "iteration": iteration,
                "drop": drop_features,
            },
            metrics={},
        )

        # Step 3: Check metric
        score = _extract_metric(run_dir, target_metric)
        if score is not None:
            logger.info("  %s = %.6f (target: %.4f)", target_metric, score, target_value)

            if _is_better(score, best_score):
                best_score = score
                best_run_dir = run_dir
                logger.info("  New best score")

            if _target_reached(score):
                logger.info("  Target reached — stopping")
                break
        else:
            logger.warning("  Could not extract %s from results", target_metric)

        # Step 4: Analyze and decide next action
        analysis = _read_analysis(run_dir)
        low_importance = _read_feature_importance(run_dir)

        action = _decide_next_action(analysis, iteration, drop_features, current_preset, presets)

        # Add low-importance features to drop list
        new_drops = [f for f in low_importance if f not in drop_features]
        if new_drops:
            logger.info("  Adding low-importance features to drop: %s", new_drops)
            drop_features = drop_features + new_drops

        current_preset = action["preset"]
        logger.info("  Next action: %s", action["reason"])

    # Summary
    logger.info("")
    logger.info("=" * 60)
    logger.info("  AGENT COMPLETE")
    logger.info("=" * 60)
    logger.info("  Best score: %s", best_score)
    logger.info("  Best run: %s", best_run_dir)
    logger.info("  Iterations used: %d / %d", min(iteration, max_iterations), max_iterations)
    target_met = best_score is not None and _target_reached(best_score)
    logger.info("  Target met: %s", target_met)
    logger.info("=" * 60)

    # Show experiment comparison
    df = compare_experiments()
    if not df.empty:
        logger.info("\nExperiment comparison:")
        logger.info("\n%s", df.to_string(index=False))

    return {
        "best_score": best_score,
        "best_run_dir": best_run_dir,
        "target_met": target_met,
        "iterations": min(iteration, max_iterations),
    }


# ---------------------------------------------------------------------------
# CLI entry points
# ---------------------------------------------------------------------------


def _base_agent_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("csv", help="Path to the input CSV file.")
    parser.add_argument(
        "--label",
        default=DEFAULT_LABEL,
        help=f"Name of the target column (default: {DEFAULT_LABEL}).",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        required=True,
        help="Maximum number of training iterations.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=DEFAULT_TEST_SIZE,
        help=f"Fraction of data for test split (default: {DEFAULT_TEST_SIZE}).",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for all outputs (default: {DEFAULT_OUTPUT_DIR}).",
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
    return parser


def agent_binary() -> None:
    """Autonomous agent for binary classification targeting F1."""
    parser = _base_agent_parser("Autonomous training agent for binary classification (target: F1).")
    parser.add_argument(
        "--target-f1",
        type=float,
        required=True,
        help="Target F1 score to reach (e.g., 0.90).",
    )
    args = parser.parse_args()
    setup_logging(verbose=args.verbose, quiet=args.quiet)

    run_agent(
        csv_path=args.csv,
        label=args.label,
        problem_type="binary",
        eval_metric="f1",
        target_metric="f1",
        target_value=args.target_f1,
        max_iterations=args.max_iterations,
        output_dir=args.output_dir,
        test_size=args.test_size,
    )


def agent_regression() -> None:
    """Autonomous agent for regression targeting RMSE."""
    parser = _base_agent_parser("Autonomous training agent for regression (target: RMSE).")
    parser.add_argument(
        "--target-rmse",
        type=float,
        required=True,
        help="Target RMSE to reach (e.g., 5.0). Lower is better.",
    )
    args = parser.parse_args()
    setup_logging(verbose=args.verbose, quiet=args.quiet)

    # For RMSE, lower is better — AutoGluon reports negative RMSE as score,
    # so we compare absolute values. The agent uses abs() internally.
    run_agent(
        csv_path=args.csv,
        label=args.label,
        problem_type="regression",
        eval_metric="root_mean_squared_error",
        target_metric="root_mean_squared_error",
        target_value=args.target_rmse,
        max_iterations=args.max_iterations,
        output_dir=args.output_dir,
        test_size=args.test_size,
        higher_is_better=False,
    )
