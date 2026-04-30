"""Tests for tool_model_subset_evaluate."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _write_leaderboard_test(
    run_dir: Path,
    rows: list[dict],
) -> None:
    """Write a minimal leaderboard_test.csv into ``run_dir``.

    Each row must contain 'model', 'score_test', 'stack_level'.
    Optional columns: 'score_val', 'pred_time_test', 'fit_time'.
    """
    run_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(run_dir / "leaderboard_test.csv", index=False)


# ---------------------------------------------------------------------------
# tool_model_subset_evaluate
# ---------------------------------------------------------------------------


def test_model_subset_evaluate_missing_file_raises(tmp_path):
    from automl_model_training.tools import tool_model_subset_evaluate

    with pytest.raises(FileNotFoundError, match="leaderboard_test.csv"):
        tool_model_subset_evaluate(str(tmp_path))


def test_model_subset_evaluate_missing_columns_raises(tmp_path):
    """CSV without the required columns should yield a clear error."""
    from automl_model_training.tools import tool_model_subset_evaluate

    # Missing 'stack_level'
    pd.DataFrame({"model": ["A"], "score_test": [0.9]}).to_csv(
        tmp_path / "leaderboard_test.csv", index=False
    )
    with pytest.raises(ValueError, match="stack_level"):
        tool_model_subset_evaluate(str(tmp_path))


def test_model_subset_evaluate_empty_leaderboard_raises(tmp_path):
    """Well-formed-but-empty CSV → RuntimeError, not a silent wrong result."""
    from automl_model_training.tools import tool_model_subset_evaluate

    pd.DataFrame(columns=["model", "score_test", "stack_level"]).to_csv(
        tmp_path / "leaderboard_test.csv", index=False
    )
    with pytest.raises(RuntimeError, match="empty"):
        tool_model_subset_evaluate(str(tmp_path))


def test_model_subset_evaluate_returns_ordered_models(tmp_path):
    """Models should be returned sorted by absolute test score, descending."""
    from automl_model_training.tools import tool_model_subset_evaluate

    _write_leaderboard_test(
        tmp_path,
        [
            {"model": "LightGBM", "score_test": 0.88, "stack_level": 1, "pred_time_test": 0.05},
            {
                "model": "WeightedEnsemble_L2",
                "score_test": 0.91,
                "stack_level": 2,
                "pred_time_test": 0.20,
            },
            {"model": "CatBoost", "score_test": 0.85, "stack_level": 1, "pred_time_test": 0.10},
        ],
    )

    result = tool_model_subset_evaluate(str(tmp_path))

    # Ordering
    assert [m["model"] for m in result["models"]] == [
        "WeightedEnsemble_L2",
        "LightGBM",
        "CatBoost",
    ]
    # Ensemble flagged correctly
    ens = [m for m in result["models"] if m["is_ensemble"]]
    assert [m["model"] for m in ens] == ["WeightedEnsemble_L2"]

    assert result["best_model"]["model"] == "WeightedEnsemble_L2"
    assert result["best_single_model"]["model"] == "LightGBM"


def test_model_subset_evaluate_flags_cheaper_near_best(tmp_path):
    """A single model within score_tolerance of the ensemble and materially
    faster at inference should land in recommended_deploy with a speedup."""
    from automl_model_training.tools import tool_model_subset_evaluate

    _write_leaderboard_test(
        tmp_path,
        [
            {
                "model": "WeightedEnsemble_L2",
                "score_test": 0.905,
                "stack_level": 2,
                "pred_time_test": 0.40,  # slow
            },
            {
                "model": "LightGBM",
                "score_test": 0.900,  # only 0.005 below best, within default tolerance 0.01
                "stack_level": 1,
                "pred_time_test": 0.05,  # 8x faster than the ensemble
            },
        ],
    )

    result = tool_model_subset_evaluate(str(tmp_path))
    assert result["recommended_deploy"] is not None
    assert result["recommended_deploy"]["model"] == "LightGBM"
    # Speedup = 0.40 / 0.05 = 8.0
    assert result["recommended_deploy"]["speedup"] == 8.0
    assert any("LightGBM" in h for h in result["hints"])


def test_model_subset_evaluate_no_recommendation_when_ensemble_alone_qualifies(tmp_path):
    """If no non-ensemble model is within score_tolerance of the best, no
    deployment recommendation should be emitted."""
    from automl_model_training.tools import tool_model_subset_evaluate

    _write_leaderboard_test(
        tmp_path,
        [
            {
                "model": "WeightedEnsemble_L2",
                "score_test": 0.95,
                "stack_level": 2,
                "pred_time_test": 0.30,
            },
            {
                "model": "LightGBM",
                "score_test": 0.80,  # 0.15 gap — far outside tolerance
                "stack_level": 1,
                "pred_time_test": 0.05,
            },
        ],
    )
    result = tool_model_subset_evaluate(str(tmp_path))
    assert result["recommended_deploy"] is None


def test_model_subset_evaluate_flags_small_single_vs_ensemble_gap(tmp_path):
    """When the best single model is very close to the ensemble, a 'may not be
    worth its cost' hint should appear."""
    from automl_model_training.tools import tool_model_subset_evaluate

    _write_leaderboard_test(
        tmp_path,
        [
            {
                "model": "WeightedEnsemble_L2",
                "score_test": 0.901,
                "stack_level": 2,
                "pred_time_test": 0.30,
            },
            {
                "model": "LightGBM",
                "score_test": 0.900,  # score_gap = 0.001, well under tolerance
                "stack_level": 1,
                "pred_time_test": 0.05,
            },
        ],
    )
    result = tool_model_subset_evaluate(str(tmp_path))
    assert result["score_gap"] == pytest.approx(0.001)
    assert any("ensemble may not be worth" in h for h in result["hints"])


def test_model_subset_evaluate_stack_level_detects_ensemble(tmp_path):
    """A model that isn't named 'WeightedEnsemble*' but sits at stack_level>1
    should still be flagged as is_ensemble=True."""
    from automl_model_training.tools import tool_model_subset_evaluate

    _write_leaderboard_test(
        tmp_path,
        [
            {
                "model": "CustomStacker",  # doesn't start with WeightedEnsemble
                "score_test": 0.93,
                "stack_level": 2,  # but stack_level says it IS a stacker
                "pred_time_test": 0.30,
            },
            {
                "model": "LightGBM",
                "score_test": 0.88,
                "stack_level": 1,
                "pred_time_test": 0.05,
            },
        ],
    )
    result = tool_model_subset_evaluate(str(tmp_path))
    stacker = next(m for m in result["models"] if m["model"] == "CustomStacker")
    assert stacker["is_ensemble"] is True
    assert result["best_single_model"]["model"] == "LightGBM"


def test_model_subset_evaluate_single_model_run(tmp_path):
    """A leaderboard with exactly one model (e.g., a tune run) should emit the
    'only one model' hint and still return a well-formed result."""
    from automl_model_training.tools import tool_model_subset_evaluate

    _write_leaderboard_test(
        tmp_path,
        [{"model": "LightGBM", "score_test": 0.87, "stack_level": 1, "pred_time_test": 0.05}],
    )
    result = tool_model_subset_evaluate(str(tmp_path))
    assert len(result["models"]) == 1
    assert result["best_model"]["model"] == "LightGBM"
    assert result["best_single_model"]["model"] == "LightGBM"
    assert result["score_gap"] == 0.0
    assert any("Only one model" in h for h in result["hints"])


def test_model_subset_evaluate_handles_negative_regression_scores(tmp_path):
    """AutoGluon stores lower-is-better metrics (RMSE, log_loss) as negative
    values such that higher score_test is always better. The tool must sort
    by score_test descending so the 'best' model is the one closest to zero
    (least-negative) in regression, and the most positive in classification."""
    from automl_model_training.tools import tool_model_subset_evaluate

    _write_leaderboard_test(
        tmp_path,
        [
            # LightGBM: RMSE ≈ 2.0 → stored as -2.0 → BETTER regression model
            {"model": "LightGBM", "score_test": -2.0, "stack_level": 1, "pred_time_test": 0.05},
            # CatBoost: RMSE ≈ 3.5 → stored as -3.5 → WORSE regression model
            {
                "model": "CatBoost",
                "score_test": -3.5,
                "stack_level": 1,
                "pred_time_test": 0.10,
            },
        ],
    )
    result = tool_model_subset_evaluate(str(tmp_path))
    # Best model = the less-negative score_test (lower RMSE in the original scale)
    assert result["best_model"]["model"] == "LightGBM"
    # Order should be LightGBM (-2.0) > CatBoost (-3.5)
    assert [m["model"] for m in result["models"]] == ["LightGBM", "CatBoost"]
