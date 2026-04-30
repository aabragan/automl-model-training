"""Tests for tool_shap_interactions."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from automl_model_training.tools import tool_shap_interactions


def test_shap_interactions_missing_files_raises(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    with pytest.raises(FileNotFoundError, match="shap_values"):
        tool_shap_interactions(str(run_dir))


def test_shap_interactions_ranks_correlated_pairs(tmp_path):
    """Two highly-correlated SHAP columns should top the pairs list."""
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    rng = np.random.RandomState(0)
    n = 100
    base = rng.randn(n)
    shap_df = pd.DataFrame(
        {
            "feat_a": base + rng.randn(n) * 0.05,  # Highly correlated with base
            "feat_b": base + rng.randn(n) * 0.05,  # Highly correlated with base
            "feat_c": rng.randn(n),  # Independent
            "feat_d": rng.randn(n),
            "feat_e": rng.randn(n),
        }
    )
    shap_df.to_csv(run_dir / "shap_values.csv", index=False)

    summary = pd.DataFrame(
        {
            "feature": ["feat_a", "feat_b", "feat_c", "feat_d", "feat_e"],
            "mean_abs_shap": [1.0, 0.9, 0.5, 0.3, 0.1],
        }
    )
    summary.to_csv(run_dir / "shap_summary.csv", index=False)

    result = tool_shap_interactions(str(run_dir), top_k=5)
    # Top pair should be feat_a/feat_b with high correlation
    top_pair = result["pairs"][0]
    assert {top_pair["feature_a"], top_pair["feature_b"]} == {"feat_a", "feat_b"}
    assert top_pair["abs_corr"] > 0.9
    # Hints should flag the correlated pair
    assert any("redundant" in h or "correlated" in h for h in result["hints"])


def test_shap_interactions_handles_few_top_features(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    pd.DataFrame({"only_one": [0.1, 0.2]}).to_csv(run_dir / "shap_values.csv", index=False)
    pd.DataFrame({"feature": ["only_one"], "mean_abs_shap": [1.0]}).to_csv(
        run_dir / "shap_summary.csv", index=False
    )

    result = tool_shap_interactions(str(run_dir), top_k=5)
    assert result["pairs"] == []
    assert any("Fewer than 2" in h for h in result["hints"])
