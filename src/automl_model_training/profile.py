"""
Dataset profiling — correlation analysis and feature removal recommendations.

Computes a correlation matrix for numeric features, identifies highly
correlated pairs and low-variance features, generates a heatmap
visualization, and recommends features to drop.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")  # non-interactive backend for headless environments
import matplotlib.pyplot as plt  # noqa: E402

from automl_model_training.config import (
    DEFAULT_LABEL,
    DEFAULT_OUTPUT_DIR,
    make_run_dir,
)


def compute_correlation_matrix(
    data: pd.DataFrame,
    label: str,
) -> pd.DataFrame:
    """Compute Pearson correlation matrix for all numeric columns.

    The label column is included so users can see feature-target
    correlations alongside feature-feature correlations.
    """
    numeric = data.select_dtypes(include="number")
    if label in numeric.columns:
        # Move label to last column for readability
        cols = [c for c in numeric.columns if c != label] + [label]
        numeric = numeric[cols]
    return numeric.corr()


def find_highly_correlated_pairs(
    corr_matrix: pd.DataFrame,
    threshold: float = 0.90,
) -> list[dict]:
    """Find feature pairs with |correlation| above the threshold.

    Returns a list of dicts with keys: feature_a, feature_b, correlation.
    Only the upper triangle is scanned to avoid duplicates.
    """
    pairs: list[dict] = []
    cols = corr_matrix.columns.tolist()
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            corr = float(corr_matrix.iloc[i, j])  # type: ignore[arg-type]
            if abs(corr) >= threshold:
                pairs.append(
                    {
                        "feature_a": cols[i],
                        "feature_b": cols[j],
                        "correlation": round(float(corr), 6),
                    }
                )
    return sorted(pairs, key=lambda p: abs(p["correlation"]), reverse=True)


def recommend_features_to_drop(
    corr_matrix: pd.DataFrame,
    label: str,
    threshold: float = 0.90,
) -> list[dict]:
    """Recommend features to drop based on correlation analysis.

    For each highly correlated pair, the feature with the lower
    absolute correlation to the label is recommended for removal.
    Features are deduplicated — each appears at most once.
    """
    pairs = find_highly_correlated_pairs(corr_matrix, threshold)
    if not pairs or label not in corr_matrix.columns:
        return []

    label_corr = corr_matrix[label].abs()
    to_drop: dict[str, dict] = {}

    for pair in pairs:
        a, b = pair["feature_a"], pair["feature_b"]
        # Skip if either is the label itself
        if a == label or b == label:
            continue

        corr_a = float(label_corr.get(a, 0.0))
        corr_b = float(label_corr.get(b, 0.0))

        # Drop the one less correlated with the label
        drop, keep = (a, b) if corr_a <= corr_b else (b, a)

        if drop not in to_drop:
            to_drop[drop] = {
                "feature": drop,
                "reason": f"highly correlated with '{keep}' "
                f"(r={pair['correlation']:.4f}), "
                f"lower label correlation "
                f"(|r|={min(corr_a, corr_b):.4f} vs {max(corr_a, corr_b):.4f})",
                "correlated_with": keep,
                "pair_correlation": pair["correlation"],
                "label_correlation": round(float(min(corr_a, corr_b)), 6),
            }

    return list(to_drop.values())


def plot_correlation_heatmap(
    corr_matrix: pd.DataFrame,
    output_path: Path,
    figsize: tuple[int, int] | None = None,
) -> Path:
    """Save a correlation matrix heatmap as a PNG image."""
    n = len(corr_matrix)
    if figsize is None:
        size = max(8, n * 0.6)
        figsize = (int(size), int(size))

    fig, ax = plt.subplots(figsize=figsize)

    im = ax.imshow(corr_matrix.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    fig.colorbar(im, ax=ax, shrink=0.8, label="Pearson correlation")

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(corr_matrix.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(corr_matrix.columns, fontsize=8)

    # Annotate cells if matrix is small enough to be readable
    if n <= 20:
        for i in range(n):
            for j in range(n):
                val = float(corr_matrix.iloc[i, j])  # type: ignore[arg-type]
                color = "white" if abs(val) > 0.6 else "black"
                ax.text(
                    j,
                    i,
                    f"{val:.2f}",
                    ha="center",
                    va="center",
                    fontsize=max(5, 8 - n // 5),
                    color=color,
                )

    ax.set_title("Feature Correlation Matrix", fontsize=12, pad=12)
    fig.tight_layout()

    png_path = output_path / "correlation_heatmap.png"
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Heatmap saved → {png_path}")
    return png_path


def save_profile_report(
    data: pd.DataFrame,
    label: str,
    corr_matrix: pd.DataFrame,
    pairs: list[dict],
    drop_recommendations: list[dict],
    output: Path,
) -> dict:
    """Save all profiling artifacts and return the summary dict."""
    output.mkdir(parents=True, exist_ok=True)

    # Correlation matrix CSV
    corr_matrix.to_csv(output / "correlation_matrix.csv")
    print(f"Correlation matrix saved → {output / 'correlation_matrix.csv'}")

    # Heatmap
    plot_correlation_heatmap(corr_matrix, output)

    # Dataset stats
    numeric = data.select_dtypes(include="number")
    stats = numeric.describe().T
    stats["missing_pct"] = (data.isnull().sum() / len(data) * 100).reindex(stats.index)
    stats["nunique"] = data.nunique().reindex(stats.index)
    stats.to_csv(output / "feature_stats.csv")
    print(f"Feature stats saved → {output / 'feature_stats.csv'}")

    # Low-variance features (std < 0.01 relative to mean)
    low_variance: list[str] = []
    for col in numeric.columns:
        if col == label:
            continue
        std = float(numeric[col].std())
        mean = float(numeric[col].mean())
        if std < 0.01 and abs(mean) > 0 or std == 0:
            low_variance.append(col)

    # Build summary
    summary = {
        "total_rows": len(data),
        "total_columns": len(data.columns),
        "numeric_columns": len(numeric.columns),
        "categorical_columns": len(data.columns) - len(numeric.columns),
        "label": label,
        "highly_correlated_pairs": pairs,
        "features_to_drop": [r["feature"] for r in drop_recommendations],
        "drop_details": drop_recommendations,
        "low_variance_features": low_variance,
    }

    with open(output / "profile_report.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Profile report saved → {output / 'profile_report.json'}")

    # Human-readable report
    _print_report(summary, pairs, drop_recommendations, low_variance)

    return summary


def _print_report(
    summary: dict,
    pairs: list[dict],
    drop_recommendations: list[dict],
    low_variance: list[str],
) -> None:
    """Print a human-readable profiling report."""
    print(f"\n{'=' * 60}")
    print("  DATASET PROFILE")
    print(f"{'=' * 60}")
    print(f"  Rows: {summary['total_rows']}")
    print(
        f"  Columns: {summary['total_columns']} "
        f"({summary['numeric_columns']} numeric, "
        f"{summary['categorical_columns']} categorical)"
    )
    print(f"  Label: {summary['label']}")

    if pairs:
        print(f"\n  Highly correlated pairs ({len(pairs)}):")
        for p in pairs[:10]:
            print(f"    {p['feature_a']} ↔ {p['feature_b']}: r={p['correlation']:.4f}")
        if len(pairs) > 10:
            print(f"    ... and {len(pairs) - 10} more")

    if drop_recommendations:
        print(f"\n  Recommended features to drop ({len(drop_recommendations)}):")
        for r in drop_recommendations:
            print(f"    • {r['feature']} — {r['reason']}")
    else:
        print("\n  No features recommended for removal.")

    if low_variance:
        print(f"\n  Low-variance features ({len(low_variance)}): {low_variance}")

    print(f"{'=' * 60}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Profile a dataset: correlation analysis, feature recommendations, "
        "and heatmap visualization."
    )
    parser.add_argument("csv", help="Path to the input CSV file.")
    parser.add_argument(
        "--label",
        default=DEFAULT_LABEL,
        help=f"Name of the target column (default: {DEFAULT_LABEL}).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.90,
        help="Correlation threshold for flagging pairs (default: 0.90).",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for profile outputs (default: {DEFAULT_OUTPUT_DIR}).",
    )
    args = parser.parse_args()

    output_dir = make_run_dir(args.output_dir, prefix="profile")
    data = pd.read_csv(args.csv)
    print(f"Loaded {len(data)} rows × {len(data.columns)} columns from {args.csv}")

    corr = compute_correlation_matrix(data, args.label)
    pairs = find_highly_correlated_pairs(corr, args.threshold)
    recommendations = recommend_features_to_drop(corr, args.label, args.threshold)

    save_profile_report(data, args.label, corr, pairs, recommendations, Path(output_dir))

    if recommendations:
        drop_list = [r["feature"] for r in recommendations]
        print("\nTo use these recommendations, add to your train command:")
        print(f"  uv run train data.csv --drop {' '.join(drop_list)}")


if __name__ == "__main__":
    main()
