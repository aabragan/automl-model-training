"""
Dataset profiling — comprehensive data quality and distribution analysis.

Generates a full profiling report covering:
- Dataset overview (shape, types, memory)
- Missing values analysis
- Numeric feature distributions and outlier detection (IQR method)
- Categorical feature cardinality analysis
- Label/target distribution (class balance for classification)
- Correlation analysis and feature removal recommendations
- Heatmap visualization
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


def profile_overview(data: pd.DataFrame) -> dict:
    """Compute high-level dataset overview stats."""
    mem_mb = data.memory_usage(deep=True).sum() / 1_048_576
    dtype_counts = data.dtypes.value_counts()
    return {
        "rows": len(data),
        "columns": len(data.columns),
        "memory_mb": round(mem_mb, 2),
        "dtype_breakdown": {str(k): int(v) for k, v in dtype_counts.items()},
        "duplicate_rows": int(data.duplicated().sum()),
    }


def profile_missing_values(data: pd.DataFrame) -> pd.DataFrame:
    """Per-column missing value counts and percentages."""
    missing = data.isnull().sum()
    pct = (missing / len(data) * 100).round(2)
    df = pd.DataFrame({"missing_count": missing, "missing_pct": pct})
    return df.sort_values("missing_pct", ascending=False)


def profile_numeric_features(data: pd.DataFrame, label: str) -> pd.DataFrame:
    """Descriptive stats + outlier counts (IQR method) for numeric columns."""
    numeric = data.select_dtypes(include="number")
    stats = numeric.describe().T
    stats["missing_pct"] = (numeric.isnull().sum() / len(data) * 100).round(2)
    stats["skew"] = numeric.skew().round(4)
    stats["kurtosis"] = numeric.kurtosis().round(4)

    # Outlier detection via IQR
    outlier_counts = {}
    for col in numeric.columns:
        q1 = numeric[col].quantile(0.25)
        q3 = numeric[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        outlier_counts[col] = int(((numeric[col] < lower) | (numeric[col] > upper)).sum())
    stats["outlier_count"] = pd.Series(outlier_counts)
    stats["outlier_pct"] = (stats["outlier_count"] / len(data) * 100).round(2)
    stats["is_label"] = stats.index == label
    return stats


def profile_categorical_features(data: pd.DataFrame, label: str) -> pd.DataFrame:
    """Cardinality, top values, and missing stats for non-numeric columns."""
    cat_cols = data.select_dtypes(exclude="number").columns.tolist()
    if label in data.columns and label not in cat_cols:
        # label might be numeric — skip
        pass
    elif label in cat_cols:
        cat_cols = [c for c in cat_cols if c != label]

    if not cat_cols:
        return pd.DataFrame()

    rows = []
    for col in cat_cols:
        vc = data[col].value_counts()
        rows.append(
            {
                "column": col,
                "nunique": int(data[col].nunique()),
                "missing_count": int(data[col].isnull().sum()),
                "missing_pct": round(data[col].isnull().sum() / len(data) * 100, 2),
                "top_value": str(vc.index[0]) if len(vc) > 0 else None,
                "top_value_count": int(vc.iloc[0]) if len(vc) > 0 else 0,
                "top_value_pct": round(vc.iloc[0] / len(data) * 100, 2) if len(vc) > 0 else 0,
            }
        )
    return pd.DataFrame(rows).set_index("column")


def profile_label(data: pd.DataFrame, label: str) -> dict:
    """Analyze the target/label column distribution."""
    if label not in data.columns:
        return {"error": f"label '{label}' not found in dataset"}

    col = data[label]
    info: dict = {
        "label": label,
        "dtype": str(col.dtype),
        "missing_count": int(col.isnull().sum()),
        "nunique": int(col.nunique()),
    }

    if col.nunique() <= 20:
        # Classification — show class balance
        vc = col.value_counts()
        info["type"] = "classification"
        info["class_distribution"] = {
            str(k): {"count": int(v), "pct": round(v / len(data) * 100, 2)} for k, v in vc.items()
        }
        majority_pct = vc.iloc[0] / len(data) * 100
        minority_pct = vc.iloc[-1] / len(data) * 100
        info["imbalance_ratio"] = (
            round(majority_pct / minority_pct, 2) if minority_pct > 0 else None
        )
    else:
        # Regression — show distribution stats
        info["type"] = "regression"
        info["mean"] = round(float(col.mean()), 6)
        info["median"] = round(float(col.median(numeric_only=True)), 6)
        info["std"] = round(float(col.std()), 6)
        info["min"] = round(float(col.min()), 6)
        info["max"] = round(float(col.max()), 6)  # type: ignore[arg-type]
        info["skew"] = round(float(col.skew()), 4)  # type: ignore[arg-type]

    return info


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

    # --- Dataset overview ---
    overview = profile_overview(data)

    # --- Missing values ---
    missing_df = profile_missing_values(data)
    missing_df.to_csv(output / "missing_values.csv")
    print(f"Missing values saved → {output / 'missing_values.csv'}")

    # --- Numeric feature stats + outliers ---
    numeric_stats = profile_numeric_features(data, label)
    numeric_stats.to_csv(output / "numeric_feature_stats.csv")
    print(f"Numeric feature stats saved → {output / 'numeric_feature_stats.csv'}")

    # --- Categorical feature stats ---
    cat_stats = profile_categorical_features(data, label)
    if not cat_stats.empty:
        cat_stats.to_csv(output / "categorical_feature_stats.csv")
        print(f"Categorical feature stats saved → {output / 'categorical_feature_stats.csv'}")

    # --- Label distribution ---
    label_info = profile_label(data, label)

    # --- Correlation matrix CSV ---
    corr_matrix.to_csv(output / "correlation_matrix.csv")
    print(f"Correlation matrix saved → {output / 'correlation_matrix.csv'}")

    # --- Heatmap ---
    plot_correlation_heatmap(corr_matrix, output)

    # --- Low-variance features ---
    numeric = data.select_dtypes(include="number")
    low_variance: list[str] = []
    for col in numeric.columns:
        if col == label:
            continue
        std = float(numeric[col].std())
        mean = float(numeric[col].mean())
        if std < 0.01 and abs(mean) > 0 or std == 0:
            low_variance.append(col)

    # --- Columns with high missing rates ---
    high_missing = missing_df[missing_df["missing_pct"] > 50].index.tolist()

    # --- Outlier summary ---
    outlier_cols = numeric_stats[numeric_stats["outlier_pct"] > 5].index.tolist()

    # --- Build full summary ---
    summary = {
        "overview": overview,
        "label_analysis": label_info,
        "missing_values": {
            "total_missing_cells": int(data.isnull().sum().sum()),
            "columns_with_missing": int((data.isnull().sum() > 0).sum()),
            "high_missing_columns": high_missing,
        },
        "outlier_summary": {
            "columns_with_outliers_gt_5pct": outlier_cols,
        },
        "correlation_analysis": {
            "highly_correlated_pairs": pairs,
            "features_to_drop": [r["feature"] for r in drop_recommendations],
            "drop_details": drop_recommendations,
        },
        "low_variance_features": low_variance,
    }

    with open(output / "profile_report.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Profile report saved → {output / 'profile_report.json'}")

    _print_report(summary, pairs, drop_recommendations, low_variance)

    return summary


def _print_report(
    summary: dict,
    pairs: list[dict],
    drop_recommendations: list[dict],
    low_variance: list[str],
) -> None:
    """Print a human-readable profiling report."""
    ov = summary["overview"]
    label_info = summary["label_analysis"]
    missing = summary["missing_values"]

    print(f"\n{'=' * 60}")
    print("  DATASET PROFILE")
    print(f"{'=' * 60}")
    print(f"  Rows: {ov['rows']}")
    print(f"  Columns: {ov['columns']}")
    print(f"  Memory: {ov['memory_mb']} MB")
    print(f"  Duplicate rows: {ov['duplicate_rows']}")
    print(f"  Types: {ov['dtype_breakdown']}")

    # Missing values
    print("\n  Missing values:")
    print(f"    Total missing cells: {missing['total_missing_cells']}")
    print(f"    Columns with missing: {missing['columns_with_missing']}")
    if missing["high_missing_columns"]:
        print(f"    Columns >50% missing: {missing['high_missing_columns']}")

    # Label analysis
    print(f"\n  Label: {label_info.get('label', 'N/A')} ({label_info.get('type', 'unknown')})")
    if label_info.get("type") == "classification":
        dist = label_info.get("class_distribution", {})
        for cls, info in dist.items():
            print(f"    {cls}: {info['count']} ({info['pct']}%)")
        if label_info.get("imbalance_ratio"):
            print(f"    Imbalance ratio: {label_info['imbalance_ratio']}:1")
    elif label_info.get("type") == "regression":
        print(f"    mean={label_info['mean']}, std={label_info['std']}, skew={label_info['skew']}")

    # Outliers
    outlier_cols = summary.get("outlier_summary", {}).get("columns_with_outliers_gt_5pct", [])
    if outlier_cols:
        print(f"\n  Columns with >5% outliers ({len(outlier_cols)}): {outlier_cols}")

    # Correlation
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
