from __future__ import annotations

from pathlib import Path

import pandas as pd


def tool_shap_interactions(run_dir: str, top_k: int = 5) -> dict:
    """Find pairs of features whose SHAP contributions correlate across rows.

    Uses the SHAP values already saved by ``tool_train(..., explain=True)``.
    Does NOT retrain anything.

    A pair of top-K features whose per-row SHAP contributions have high
    correlation suggests they carry redundant or coupled signal about the
    prediction. The LLM can use this to:
    - Drop one of a redundant pair (correlation ~ +1)
    - Engineer a ratio or product feature for a strongly-coupled pair
    - Investigate whether a counter-correlated pair (~ -1) indicates
      a hidden interaction the model is trying to express

    Parameters
    ----------
    run_dir : str
        Training run directory (must contain shap_values.csv — i.e.
        training must have used ``explain=True``).
    top_k : int
        Rank features by mean |SHAP| and analyze the top-k only. Default 5.
        Pairwise table size grows as k*(k-1)/2.

    Returns
    -------
    dict with keys:
        top_features : list of {feature, mean_abs_shap} sorted
        pairs        : list of {feature_a, feature_b, corr, abs_corr}
                       sorted by |corr| desc
        hints        : actionable observations
    """
    path = Path(run_dir)
    shap_path = path / "shap_values.csv"
    summary_path = path / "shap_summary.csv"
    if not shap_path.exists() or not summary_path.exists():
        raise FileNotFoundError(
            f"tool_shap_interactions: missing shap_values.csv or shap_summary.csv in "
            f"{run_dir}. Re-run training with explain=True."
        )

    shap_df = pd.read_csv(shap_path)
    summary = pd.read_csv(summary_path)

    # Top-k features by mean |SHAP|
    summary_sorted = summary.sort_values("mean_abs_shap", ascending=False).head(top_k)
    top_features = summary_sorted.to_dict(orient="records")
    top_feature_names = summary_sorted["feature"].tolist()

    # Restrict SHAP matrix to top-k features that actually exist as columns
    present = [f for f in top_feature_names if f in shap_df.columns]
    if len(present) < 2:
        return {
            "top_features": top_features,
            "pairs": [],
            "hints": ["Fewer than 2 top features found in shap_values.csv — no pairs to analyze"],
        }

    top_shap = shap_df[present]

    pairs = []
    for i, a in enumerate(present):
        for b in present[i + 1 :]:
            col_a = top_shap[a]
            col_b = top_shap[b]
            if col_a.std() == 0 or col_b.std() == 0:
                continue
            corr = float(col_a.corr(col_b))
            pairs.append(
                {
                    "feature_a": a,
                    "feature_b": b,
                    "corr": round(corr, 4),
                    "abs_corr": round(abs(corr), 4),
                }
            )
    pairs.sort(key=lambda p: p["abs_corr"], reverse=True)

    hints = []
    for p in pairs:
        if p["abs_corr"] > 0.7:
            if p["corr"] > 0:
                hints.append(
                    f"'{p['feature_a']}' and '{p['feature_b']}' SHAP values are "
                    f"highly correlated (r={p['corr']}) — they may carry redundant "
                    "signal. Try dropping one, or engineer their ratio/product."
                )
            else:
                hints.append(
                    f"'{p['feature_a']}' and '{p['feature_b']}' SHAP values are "
                    f"strongly counter-correlated (r={p['corr']}) — consider "
                    "engineering their difference or ratio."
                )
    if not hints and pairs:
        hints.append("No strongly interacting pairs among top features (all |r| ≤ 0.7).")

    return {"top_features": top_features, "pairs": pairs, "hints": hints}
