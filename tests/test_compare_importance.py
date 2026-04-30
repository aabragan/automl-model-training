"""Tests for tool_compare_importance."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _write_importance_run(
    run_dir: Path,
    importances: dict[str, float],
    score: float,
) -> None:
    """Write a minimal feature_importance.csv + leaderboard_test.csv into run_dir."""
    run_dir.mkdir(parents=True, exist_ok=True)
    imp_df = pd.DataFrame(
        [
            {"": feat, "importance": imp, "stddev": 0.0, "p_value": 0.01, "n": 5}
            for feat, imp in importances.items()
        ]
    ).set_index("")
    imp_df.to_csv(run_dir / "feature_importance.csv")
    pd.DataFrame(
        {
            "model": ["fake"],
            "score_test": [score],
            "score_val": [score],
            "fit_time": [1.0],
        }
    ).to_csv(run_dir / "leaderboard_test.csv", index=False)


# ---------------------------------------------------------------------------
# tool_compare_importance
# ---------------------------------------------------------------------------


def test_compare_importance_missing_files_raises(tmp_path):
    from automl_model_training.tools import tool_compare_importance

    with pytest.raises(FileNotFoundError, match="feature_importance.csv"):
        tool_compare_importance(str(tmp_path / "a"), str(tmp_path / "b"))


def test_compare_importance_detects_gained_and_lost(tmp_path):
    from automl_model_training.tools import tool_compare_importance

    before = tmp_path / "before"
    after = tmp_path / "after"
    _write_importance_run(before, {"x": 0.4, "y": 0.3, "z": 0.2}, score=0.80)
    _write_importance_run(after, {"x": 0.45, "y": 0.25, "new_feat": 0.15}, score=0.82)

    result = tool_compare_importance(str(before), str(after))

    assert result["gained_features"] == ["new_feat"]
    assert result["lost_features"] == ["z"]
    assert result["score_before"] == 0.80
    assert result["score_after"] == 0.82
    assert result["score_delta"] == 0.02


def test_compare_importance_flags_dominant_new_feature_with_flat_score(tmp_path):
    """Gained feature dominates after-run importance but score barely moved
    → should raise a leakage hint."""
    from automl_model_training.tools import tool_compare_importance

    before = tmp_path / "before"
    after = tmp_path / "after"
    _write_importance_run(before, {"a": 0.2, "b": 0.2}, score=0.85)
    # New leakage-like feature with huge importance but score unchanged
    _write_importance_run(after, {"a": 0.1, "b": 0.1, "leak": 0.9}, score=0.851)

    result = tool_compare_importance(str(before), str(after))
    assert result["dominant_new_feature"] is not None
    assert result["dominant_new_feature"]["feature"] == "leak"
    assert any("leak" in h for h in result["hints"])


def test_compare_importance_no_flag_when_score_improves(tmp_path):
    """Same dominant new feature but score improved materially → no leakage flag."""
    from automl_model_training.tools import tool_compare_importance

    before = tmp_path / "before"
    after = tmp_path / "after"
    _write_importance_run(before, {"a": 0.2, "b": 0.2}, score=0.70)
    _write_importance_run(after, {"a": 0.1, "b": 0.1, "new": 0.9}, score=0.88)

    result = tool_compare_importance(str(before), str(after))
    assert result["dominant_new_feature"] is None


def test_compare_importance_ranks_top_n_by_abs_delta(tmp_path):
    from automl_model_training.tools import tool_compare_importance

    before = tmp_path / "before"
    after = tmp_path / "after"
    _write_importance_run(
        before,
        {"a": 0.50, "b": 0.40, "c": 0.30, "d": 0.20, "e": 0.10},
        score=0.80,
    )
    _write_importance_run(
        after,
        {"a": 0.50, "b": 0.60, "c": 0.10, "d": 0.22, "e": 0.11},  # b ↑, c ↓
        score=0.82,
    )
    result = tool_compare_importance(str(before), str(after), top_n=2)
    assert len(result["changed_features"]) == 2
    # Top 2 by |delta|: b (+0.20), c (-0.20)
    features_in_top = {r["feature"] for r in result["changed_features"]}
    assert features_in_top == {"b", "c"}


def test_compare_importance_ignores_sub_threshold_changes(tmp_path):
    """significance_delta filters out jitter."""
    from automl_model_training.tools import tool_compare_importance

    before = tmp_path / "before"
    after = tmp_path / "after"
    _write_importance_run(before, {"a": 0.5, "b": 0.5}, score=0.80)
    # Only tiny changes (< 0.01 default threshold)
    _write_importance_run(after, {"a": 0.501, "b": 0.499}, score=0.80)

    result = tool_compare_importance(str(before), str(after))
    assert result["changed_features"] == []
    assert any("No material importance changes" in h for h in result["hints"])
