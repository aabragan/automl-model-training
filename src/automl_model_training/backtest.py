"""
Temporal backtesting for AutoGluon tabular models.

Splits data by a date column and trains/evaluates on temporal folds,
giving a realistic estimate of how the model performs on future data.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from automl_model_training.config import (
    DEFAULT_EVAL_METRIC,
    DEFAULT_LABEL,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_PRESET,
    DEFAULT_PROBLEM_TYPE,
    DEFAULT_TIME_LIMIT,
    FEATURES_TO_DROP,
    make_run_dir,
)
from automl_model_training.train import train_and_evaluate


def temporal_backtest(
    csv_path: str,
    date_column: str,
    label: str,
    cutoff: str | None,
    n_splits: int,
    problem_type: str | None,
    eval_metric: str | None,
    time_limit: int | None,
    preset: str,
    output_dir: str,
    features_to_drop: list[str],
) -> dict:
    """Run a walk-forward temporal backtest.

    Parameters
    ----------
    csv_path : str
        Path to the input CSV.
    date_column : str
        Column containing date/datetime values used for temporal ordering.
    label : str
        Target column name.
    cutoff : str or None
        Single cutoff date string (e.g. ``"2025-06-01"``).  When provided
        with ``n_splits=1`` (the default), a single train/test split is
        made at this date.  Ignored when ``n_splits > 1``.
    n_splits : int
        Number of walk-forward folds.  When > 1, the data is divided into
        ``n_splits + 1`` chronological chunks and each fold trains on all
        preceding chunks and tests on the next one.
    problem_type, eval_metric, time_limit, preset, output_dir, features_to_drop
        Forwarded to :func:`train_and_evaluate`.

    Returns
    -------
    dict
        Aggregate backtest summary written to ``backtest_summary.json``.
    """
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    data = pd.read_csv(csv_path)
    print(f"Loaded {len(data)} rows from {csv_path}")

    # Drop unwanted features (keep date_column for splitting)
    cols_to_drop = [c for c in features_to_drop if c in data.columns and c != date_column]
    if cols_to_drop:
        data = data.drop(columns=cols_to_drop)
        print(f"Dropped features: {cols_to_drop}")

    # Parse and sort by date
    data[date_column] = pd.to_datetime(data[date_column])
    data = data.sort_values(date_column).reset_index(drop=True)

    # Build fold boundaries
    folds = _build_folds(data, date_column, cutoff, n_splits)
    print(f"Backtest: {len(folds)} fold(s)")

    fold_results: list[dict] = []

    for i, (train_df, test_df) in enumerate(folds):
        fold_num = i + 1
        fold_dir = str(output / f"fold_{fold_num}")
        Path(fold_dir).mkdir(parents=True, exist_ok=True)

        print(f"\n{'=' * 60}")
        print(f"  FOLD {fold_num}/{len(folds)}")
        print(
            f"  Train: {len(train_df)} rows  "
            f"({train_df[date_column].min().date()} → {train_df[date_column].max().date()})"
        )
        print(
            f"  Test:  {len(test_df)} rows  "
            f"({test_df[date_column].min().date()} → {test_df[date_column].max().date()})"
        )
        print(f"{'=' * 60}")

        # Drop date column before training (it's not a feature)
        train_fold = train_df.drop(columns=[date_column])
        test_fold = test_df.drop(columns=[date_column])

        predictor = train_and_evaluate(
            train_raw=train_fold,
            test_raw=test_fold,
            label=label,
            problem_type=problem_type,
            eval_metric=eval_metric,
            time_limit=time_limit,
            preset=preset,
            output_dir=fold_dir,
        )

        scores = predictor.evaluate(test_fold)
        fold_results.append(
            {
                "fold": fold_num,
                "train_rows": len(train_df),
                "test_rows": len(test_df),
                "train_date_range": [
                    str(train_df[date_column].min().date()),
                    str(train_df[date_column].max().date()),
                ],
                "test_date_range": [
                    str(test_df[date_column].min().date()),
                    str(test_df[date_column].max().date()),
                ],
                "scores": {k: float(v) for k, v in scores.items()},
            }
        )

    # Aggregate summary
    summary = _aggregate_results(fold_results)
    summary["n_folds"] = len(folds)
    summary["total_rows"] = len(data)
    summary["date_column"] = date_column
    summary["folds"] = fold_results

    with open(output / "backtest_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nBacktest summary saved → {output / 'backtest_summary.json'}")

    _print_summary(summary)
    return summary


def _build_folds(
    data: pd.DataFrame,
    date_column: str,
    cutoff: str | None,
    n_splits: int,
) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    """Build temporal train/test folds."""
    if n_splits == 1 and cutoff:
        cutoff_dt = pd.to_datetime(cutoff)
        train = data[data[date_column] < cutoff_dt]
        test = data[data[date_column] >= cutoff_dt]
        if train.empty or test.empty:
            raise ValueError(
                f"Cutoff '{cutoff}' produces an empty split. "
                f"Date range: {data[date_column].min()} → {data[date_column].max()}"
            )
        return [(train, test)]

    # Walk-forward: divide into n_splits + 1 chronological chunks
    chunk_size = len(data) // (n_splits + 1)
    if chunk_size < 1:
        raise ValueError(f"Not enough data ({len(data)} rows) for {n_splits} splits.")

    folds: list[tuple[pd.DataFrame, pd.DataFrame]] = []
    for i in range(1, n_splits + 1):
        train_end = chunk_size * i
        test_end = chunk_size * (i + 1) if i < n_splits else len(data)
        train = data.iloc[:train_end]
        test = data.iloc[train_end:test_end]
        if not train.empty and not test.empty:
            folds.append((train.copy(), test.copy()))

    if not folds:
        raise ValueError("Could not create any valid folds from the data.")
    return folds


def _aggregate_results(fold_results: list[dict]) -> dict:
    """Compute mean and std of scores across folds."""
    if not fold_results:
        return {"aggregate_scores": {}}

    all_metrics = fold_results[0]["scores"].keys()
    agg: dict[str, dict[str, float]] = {}

    for metric in all_metrics:
        values = [f["scores"][metric] for f in fold_results if metric in f["scores"]]
        if values:
            mean = sum(values) / len(values)
            variance = sum((v - mean) ** 2 for v in values) / len(values)
            std = variance**0.5
            agg[metric] = {"mean": round(mean, 6), "std": round(std, 6)}

    return {"aggregate_scores": agg}


def _print_summary(summary: dict) -> None:
    """Print a human-readable backtest summary."""
    print(f"\n{'=' * 60}")
    print("  BACKTEST SUMMARY")
    print(f"{'=' * 60}")
    print(f"  Folds: {summary['n_folds']}")
    print(f"  Total rows: {summary['total_rows']}")
    print(f"  Date column: {summary['date_column']}")
    print()
    for metric, stats in summary["aggregate_scores"].items():
        print(f"  {metric}: {stats['mean']:.6f} ± {stats['std']:.6f}")
    print(f"{'=' * 60}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Temporal walk-forward backtesting for AutoGluon models."
    )
    parser.add_argument("csv", help="Path to the input CSV file.")
    parser.add_argument(
        "--date-column",
        required=True,
        help="Name of the date/datetime column for temporal splitting.",
    )
    parser.add_argument(
        "--cutoff",
        default=None,
        help="Cutoff date for a single train/test split (e.g. '2025-06-01').",
    )
    parser.add_argument(
        "--n-splits",
        type=int,
        default=1,
        help="Number of walk-forward folds (default: 1). Overrides --cutoff when > 1.",
    )
    parser.add_argument(
        "--label",
        default=DEFAULT_LABEL,
        help=f"Name of the target column (default: {DEFAULT_LABEL}).",
    )
    parser.add_argument(
        "--problem-type",
        default=DEFAULT_PROBLEM_TYPE,
        choices=["binary", "multiclass", "regression", "quantile"],
        help="Problem type (default: auto-detect).",
    )
    parser.add_argument(
        "--eval-metric",
        default=DEFAULT_EVAL_METRIC,
        help="Evaluation metric (default: auto-detect).",
    )
    parser.add_argument(
        "--preset",
        default=DEFAULT_PRESET,
        help=f"AutoGluon preset (default: {DEFAULT_PRESET}).",
    )
    parser.add_argument(
        "--time-limit",
        type=int,
        default=DEFAULT_TIME_LIMIT,
        help="Training time limit in seconds per fold (default: no limit).",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for all outputs (default: {DEFAULT_OUTPUT_DIR}).",
    )
    parser.add_argument(
        "--drop",
        nargs="*",
        default=FEATURES_TO_DROP,
        help="Feature column names to drop before training.",
    )
    args = parser.parse_args()

    output_dir = make_run_dir(args.output_dir, prefix="backtest")

    temporal_backtest(
        csv_path=args.csv,
        date_column=args.date_column,
        label=args.label,
        cutoff=args.cutoff,
        n_splits=args.n_splits,
        problem_type=args.problem_type,
        eval_metric=args.eval_metric,
        time_limit=args.time_limit,
        preset=args.preset,
        output_dir=output_dir,
        features_to_drop=args.drop,
    )


if __name__ == "__main__":
    main()
