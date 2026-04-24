"""Tests for tool_inspect_errors."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from automl_model_training.tools import tool_inspect_errors


def _write_run(
    path: Path,
    problem_type: str,
    label: str,
    preds: pd.DataFrame,
    test_raw: pd.DataFrame,
) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    preds.to_csv(path / "test_predictions.csv", index=False)
    test_raw.to_csv(path / "test_raw.csv", index=False)
    (path / "model_info.json").write_text(
        json.dumps({"problem_type": problem_type, "label": label})
    )
    return path


def test_missing_run_dir_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="missing"):
        tool_inspect_errors(str(tmp_path / "nonexistent"))


def test_row_count_mismatch_raises(tmp_path):
    preds = pd.DataFrame(
        {"actual": [0, 1], "predicted": [0, 1], "prob_0": [0.9, 0.1], "prob_1": [0.1, 0.9]}
    )
    test_raw = pd.DataFrame({"x": [1], "y": [0]})  # Only 1 row
    run = _write_run(tmp_path / "run", "binary", "y", preds, test_raw)
    with pytest.raises(ValueError, match="Row count mismatch"):
        tool_inspect_errors(str(run))


def test_binary_worst_errors_ranked_by_confidence(tmp_path):
    preds = pd.DataFrame(
        {
            "actual": [0, 0, 1, 1, 0],
            "predicted": [0, 1, 1, 0, 0],  # Row 1 wrong high-conf, row 3 wrong low-conf
            "prob_0": [0.95, 0.1, 0.2, 0.4, 0.88],
            "prob_1": [0.05, 0.9, 0.8, 0.6, 0.12],
        }
    )
    test_raw = pd.DataFrame({"feat_a": [10, 20, 30, 40, 50], "label": [0, 0, 1, 1, 0]})
    run = _write_run(tmp_path / "run", "binary", "label", preds, test_raw)

    result = tool_inspect_errors(str(run), n=3)

    assert result["problem_type"] == "binary"
    assert result["label"] == "label"
    assert len(result["rows"]) == 3
    # Errors (2 of them) should come first
    assert sum(1 for r in result["rows"] if r["actual"] != r["predicted"]) == 2
    # Feature values from test_raw must be present
    assert "feat_a" in result["rows"][0]


def test_binary_summary_counts_errors(tmp_path):
    preds = pd.DataFrame(
        {
            "actual": [0, 0, 1, 1],
            "predicted": [0, 1, 1, 0],
            "prob_0": [0.9, 0.3, 0.1, 0.7],
            "prob_1": [0.1, 0.7, 0.9, 0.3],
        }
    )
    test_raw = pd.DataFrame({"x": [1, 2, 3, 4], "label": [0, 0, 1, 1]})
    run = _write_run(tmp_path / "run", "binary", "label", preds, test_raw)

    result = tool_inspect_errors(str(run))

    assert result["summary"]["error_count"] == 2
    assert result["summary"]["error_rate"] == 0.5
    assert result["summary"]["class_prevalence"] == {0: 0.5, 1: 0.5}


def test_binary_flags_high_confidence_errors(tmp_path):
    # 3 errors at confidence > 0.9 should trigger the hint
    preds = pd.DataFrame(
        {
            "actual": [0, 0, 0, 1, 1],
            "predicted": [1, 1, 1, 1, 1],
            "prob_0": [0.05, 0.05, 0.05, 0.05, 0.05],
            "prob_1": [0.95, 0.95, 0.95, 0.95, 0.95],
        }
    )
    test_raw = pd.DataFrame({"x": [1, 2, 3, 4, 5], "label": [0, 0, 0, 1, 1]})
    run = _write_run(tmp_path / "run", "binary", "label", preds, test_raw)

    result = tool_inspect_errors(str(run))
    assert any("confidence > 0.9" in h for h in result["hints"])


def test_binary_flags_class_imbalance_in_errors(tmp_path):
    # Class 1 is 20% of data but 100% of errors
    preds = pd.DataFrame(
        {
            "actual": [0] * 8 + [1] * 2,
            "predicted": [0] * 8 + [0] * 2,  # All class-1 rows misclassified
            "prob_0": [0.9] * 8 + [0.7, 0.7],
            "prob_1": [0.1] * 8 + [0.3, 0.3],
        }
    )
    test_raw = pd.DataFrame({"x": list(range(10)), "label": [0] * 8 + [1] * 2})
    run = _write_run(tmp_path / "run", "binary", "label", preds, test_raw)

    result = tool_inspect_errors(str(run))
    assert any("overrepresented" in h for h in result["hints"])


def test_regression_worst_ranked_by_abs_residual(tmp_path):
    preds = pd.DataFrame(
        {
            "actual": [100, 200, 300, 400, 500],
            "predicted": [105, 250, 310, 250, 498],
            "residual": [-5, -50, -10, 150, 2],
        }
    )
    test_raw = pd.DataFrame({"feat_a": [1, 2, 3, 4, 5], "price": [100, 200, 300, 400, 500]})
    run = _write_run(tmp_path / "run", "regression", "price", preds, test_raw)

    result = tool_inspect_errors(str(run), n=2)

    assert result["problem_type"] == "regression"
    # Largest |residual| is 150, then 50
    assert result["rows"][0]["abs_residual"] == 150
    assert result["rows"][1]["abs_residual"] == 50


def test_regression_best_returns_smallest_residuals(tmp_path):
    preds = pd.DataFrame(
        {"actual": [100, 200, 300], "predicted": [105, 250, 301], "residual": [-5, -50, -1]}
    )
    test_raw = pd.DataFrame({"x": [1, 2, 3], "price": [100, 200, 300]})
    run = _write_run(tmp_path / "run", "regression", "price", preds, test_raw)

    result = tool_inspect_errors(str(run), n=1, worst=False)
    assert result["rows"][0]["abs_residual"] == 1


def test_regression_detects_systematic_bias(tmp_path):
    # Model consistently under-predicts by ~100
    n_rows = 30
    actual = np.arange(100, 100 + n_rows * 10, 10)
    predicted = actual - 100
    preds = pd.DataFrame(
        {"actual": actual, "predicted": predicted, "residual": actual - predicted}
    )
    test_raw = pd.DataFrame({"x": np.arange(n_rows), "price": actual})
    run = _write_run(tmp_path / "run", "regression", "price", preds, test_raw)

    result = tool_inspect_errors(str(run))
    assert any("under-predicting" in h for h in result["hints"])


def test_regression_summary_stats(tmp_path):
    preds = pd.DataFrame(
        {"actual": [100, 200, 300], "predicted": [110, 180, 320], "residual": [-10, 20, -20]}
    )
    test_raw = pd.DataFrame({"x": [1, 2, 3], "price": [100, 200, 300]})
    run = _write_run(tmp_path / "run", "regression", "price", preds, test_raw)

    result = tool_inspect_errors(str(run))
    assert result["summary"]["max_abs_residual"] == 20
    assert result["summary"]["mean_abs_residual"] == pytest.approx(50 / 3, rel=1e-3)


def test_unsupported_problem_type_raises(tmp_path):
    preds = pd.DataFrame({"actual": [1], "predicted": [1]})
    test_raw = pd.DataFrame({"x": [1], "label": [1]})
    run = _write_run(tmp_path / "run", "unknown_type", "label", preds, test_raw)

    with pytest.raises(ValueError, match="Unsupported problem_type"):
        tool_inspect_errors(str(run))
