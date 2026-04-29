"""Tests for new tier-3 LLM tools (threshold sweep, calibration, Optuna tune,
importance diff, 2-way PDP, model-subset eval) and Optuna study persistence."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from automl_model_training.tools import tool_threshold_sweep

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _write_binary_predictions(
    run_dir: Path,
    n: int = 200,
    signal_strength: float = 2.0,
    imbalance: float = 0.5,
    seed: int = 0,
) -> None:
    """Write a realistic binary test_predictions.csv into ``run_dir``.

    signal_strength : larger → prob_1 separates the classes better
    imbalance       : fraction of positives (0.5 = balanced)
    """
    rng = np.random.RandomState(seed)
    y = (rng.rand(n) < imbalance).astype(int)
    # Logistic-ish: positives pulled toward high prob_1, negatives toward low
    logit = signal_strength * (y - 0.5) + rng.randn(n) * 0.5
    prob_1 = 1.0 / (1.0 + np.exp(-logit))
    predicted = (prob_1 >= 0.5).astype(int)
    pd.DataFrame(
        {
            "actual": y,
            "predicted": predicted,
            "prob_0": 1.0 - prob_1,
            "prob_1": prob_1,
        }
    ).to_csv(run_dir / "test_predictions.csv", index=False)


# ---------------------------------------------------------------------------
# tool_threshold_sweep
# ---------------------------------------------------------------------------


def test_threshold_sweep_missing_predictions_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="test_predictions.csv"):
        tool_threshold_sweep(str(tmp_path))


def test_threshold_sweep_rejects_multiclass(tmp_path):
    """A run with 3+ prob_ columns is multiclass — tool should refuse."""
    pd.DataFrame(
        {
            "actual": [0, 1, 2, 0, 1, 2],
            "predicted": [0, 1, 2, 0, 1, 2],
            "prob_0": [0.9, 0.1, 0.1, 0.9, 0.1, 0.1],
            "prob_1": [0.05, 0.8, 0.1, 0.05, 0.8, 0.1],
            "prob_2": [0.05, 0.1, 0.8, 0.05, 0.1, 0.8],
        }
    ).to_csv(tmp_path / "test_predictions.csv", index=False)

    with pytest.raises(ValueError, match="binary classification"):
        tool_threshold_sweep(str(tmp_path))


def test_threshold_sweep_rejects_unknown_metric(tmp_path):
    _write_binary_predictions(tmp_path)
    with pytest.raises(ValueError, match="Unknown metrics"):
        tool_threshold_sweep(str(tmp_path), metrics=["f1", "nonsense"])


def test_threshold_sweep_returns_expected_shape(tmp_path):
    _write_binary_predictions(tmp_path, n=200)
    result = tool_threshold_sweep(str(tmp_path), n_thresholds=49)

    # Shape guarantees
    assert len(result["thresholds"]) == 49
    assert all(0 < t < 1 for t in result["thresholds"])
    for metric in ["f1", "precision", "recall", "mcc", "balanced_accuracy"]:
        assert metric in result["curves"]
        assert len(result["curves"][metric]) == 49
        assert metric in result["best"]
        assert 0 < result["best"][metric]["threshold"] < 1


def test_threshold_sweep_f1_peaks_near_05_for_balanced_strong_signal(tmp_path):
    """With balanced classes and strong signal, F1-optimal threshold should land
    somewhere in the middle band (0.2–0.8), not at the edges. F1 is asymmetric
    and can drift below 0.5 on noisy signals, so we only assert the middle band,
    not a tight window around 0.5."""
    _write_binary_predictions(tmp_path, n=500, signal_strength=4.0, imbalance=0.5)
    result = tool_threshold_sweep(str(tmp_path))
    f1_thresh = result["best"]["f1"]["threshold"]
    assert 0.2 < f1_thresh < 0.8, f"F1-optimal threshold should be in middle band, got {f1_thresh}"
    # And F1 itself should be high (strong signal + balanced classes)
    assert result["best"]["f1"]["value"] > 0.8


def test_threshold_sweep_flags_near_05_optimum(tmp_path):
    """When F1-optimal is within 0.02 of 0.5, a hint about calibration is returned."""
    _write_binary_predictions(tmp_path, n=1000, signal_strength=5.0, imbalance=0.5)
    result = tool_threshold_sweep(str(tmp_path), n_thresholds=99)
    if abs(result["best"]["f1"]["threshold"] - 0.5) < 0.02:
        assert any("within 0.02 of 0.5" in h for h in result["hints"])


def test_threshold_sweep_subset_metrics(tmp_path):
    """Requesting only some metrics should yield only those curves."""
    _write_binary_predictions(tmp_path)
    result = tool_threshold_sweep(str(tmp_path), metrics=["f1", "mcc"])
    assert set(result["curves"].keys()) == {"f1", "mcc"}
    assert set(result["best"].keys()) == {"f1", "mcc"}
