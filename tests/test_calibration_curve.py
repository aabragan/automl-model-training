"""Tests for tool_calibration_curve."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from automl_model_training.tools import tool_calibration_curve

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


def _write_miscalibrated_predictions(
    run_dir: Path,
    direction: str,
    n: int = 1000,
    seed: int = 0,
) -> None:
    """Write test_predictions.csv with a known miscalibration pattern.

    direction : "over_confident"  — prob_1 pushed toward 0/1 but actuals are noisier
                "under_confident" — prob_1 compressed toward 0.5 but actuals are decisive
                "well_calibrated" — prob_1 matches actual positive rate per bin
    """
    rng = np.random.RandomState(seed)
    true_prob = rng.uniform(0.01, 0.99, size=n)
    y = (rng.rand(n) < true_prob).astype(int)

    if direction == "over_confident":
        # Push reported probs away from 0.5 (sharper than reality)
        prob_1 = np.where(true_prob > 0.5, true_prob + (1 - true_prob) * 0.6, true_prob * 0.4)
    elif direction == "under_confident":
        # Compress reported probs toward 0.5 (softer than reality)
        prob_1 = 0.5 + (true_prob - 0.5) * 0.3
    elif direction == "well_calibrated":
        prob_1 = true_prob
    else:
        raise ValueError(f"Unknown direction: {direction}")

    prob_1 = np.clip(prob_1, 0.001, 0.999)
    pd.DataFrame(
        {
            "actual": y,
            "predicted": (prob_1 >= 0.5).astype(int),
            "prob_0": 1.0 - prob_1,
            "prob_1": prob_1,
        }
    ).to_csv(run_dir / "test_predictions.csv", index=False)


# ---------------------------------------------------------------------------
# tool_calibration_curve
# ---------------------------------------------------------------------------


def test_calibration_curve_missing_predictions_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="test_predictions.csv"):
        tool_calibration_curve(str(tmp_path))


def test_calibration_curve_rejects_multiclass(tmp_path):
    pd.DataFrame(
        {
            "actual": [0, 1, 2],
            "predicted": [0, 1, 2],
            "prob_0": [0.9, 0.1, 0.1],
            "prob_1": [0.05, 0.8, 0.1],
            "prob_2": [0.05, 0.1, 0.8],
        }
    ).to_csv(tmp_path / "test_predictions.csv", index=False)

    with pytest.raises(ValueError, match="binary classification"):
        tool_calibration_curve(str(tmp_path))


def test_calibration_curve_rejects_invalid_strategy(tmp_path):
    _write_binary_predictions(tmp_path)
    with pytest.raises(ValueError, match="strategy"):
        tool_calibration_curve(str(tmp_path), strategy="bogus")


def test_calibration_curve_returns_expected_shape(tmp_path):
    _write_binary_predictions(tmp_path, n=500)
    result = tool_calibration_curve(str(tmp_path), n_bins=10)
    assert result["n_samples"] == 500
    assert 0.0 <= result["ece"] <= 1.0
    assert 0.0 <= result["max_gap"] <= 1.0
    assert result["direction"] in {"over_confident", "under_confident", "well_calibrated", "mixed"}
    # Each bin has a prob_range of length 2
    for b in result["bins"]:
        assert len(b["prob_range"]) == 2
        assert b["prob_range"][0] <= b["prob_range"][1]


def test_calibration_curve_detects_overconfidence(tmp_path):
    _write_miscalibrated_predictions(tmp_path, direction="over_confident", n=2000)
    result = tool_calibration_curve(str(tmp_path), n_bins=10)
    assert result["direction"] == "over_confident"
    assert result["ece"] > 0.05
    assert any("over-confident" in h for h in result["hints"])


def test_calibration_curve_detects_underconfidence(tmp_path):
    _write_miscalibrated_predictions(tmp_path, direction="under_confident", n=2000)
    result = tool_calibration_curve(str(tmp_path), n_bins=10)
    assert result["direction"] == "under_confident"
    assert result["ece"] > 0.05


def test_calibration_curve_detects_well_calibrated(tmp_path):
    _write_miscalibrated_predictions(tmp_path, direction="well_calibrated", n=2000)
    result = tool_calibration_curve(str(tmp_path), n_bins=10)
    assert result["direction"] == "well_calibrated"
    assert result["max_gap"] < 0.1


def test_calibration_curve_missing_actual_column_raises(tmp_path):
    pd.DataFrame(
        {
            "predicted": [0, 1, 0, 1],
            "prob_0": [0.8, 0.2, 0.7, 0.3],
            "prob_1": [0.2, 0.8, 0.3, 0.7],
        }
    ).to_csv(tmp_path / "test_predictions.csv", index=False)

    with pytest.raises(ValueError, match="missing 'actual'"):
        tool_calibration_curve(str(tmp_path))


def test_calibration_curve_noncoercible_label_falls_back_to_string(tmp_path):
    """When the positive label can't be coerced to the actual column's dtype
    (int('dog') raises), the tool falls back to the raw string label."""
    pd.DataFrame(
        {
            "actual": [0, 1] * 10,
            "predicted": [0, 1] * 10,
            "prob_cat": [0.8, 0.2] * 10,
            "prob_dog": [0.2, 0.8] * 10,
        }
    ).to_csv(tmp_path / "test_predictions.csv", index=False)

    result = tool_calibration_curve(str(tmp_path), n_bins=4)
    # String label 'dog' matches no int actuals → every occupied bin has 0 positives
    occupied = [b for b in result["bins"] if b["n_samples"] > 0]
    assert occupied
    assert all(b["actual_positive_rate"] == 0.0 for b in occupied)


def test_calibration_curve_uniform_strategy_reports_empty_bins(tmp_path):
    """Uniform bins with probs clustered at 0.75 → 9 empty bins with None fields.
    Predicted 0.75 vs actual 0.25 has a big gap but identical extremeness
    (both 0.25 from 0.5), so the direction vote is empty → well_calibrated."""
    n = 100
    y = np.array([1] * 25 + [0] * 75)
    prob_1 = np.full(n, 0.75)
    pd.DataFrame(
        {
            "actual": y,
            "predicted": (prob_1 >= 0.5).astype(int),
            "prob_0": 1.0 - prob_1,
            "prob_1": prob_1,
        }
    ).to_csv(tmp_path / "test_predictions.csv", index=False)

    result = tool_calibration_curve(str(tmp_path), n_bins=10, strategy="uniform")
    assert result["strategy"] == "uniform"
    assert result["n_bins_effective"] == 1
    empty = [b for b in result["bins"] if b["n_samples"] == 0]
    assert len(empty) == 9
    assert all(b["mean_predicted"] is None and b["gap"] is None for b in empty)
    assert result["max_gap"] == pytest.approx(0.5)
    assert result["direction"] == "well_calibrated"


def test_calibration_curve_mixed_direction(tmp_path):
    """One over-extreme bin (0.9 predicted vs 0.6 actual) and one under-extreme
    bin (0.6 predicted vs 0.9 actual) → no dominant direction → 'mixed'."""
    y = np.concatenate(
        [
            np.array([1] * 60 + [0] * 40),  # cluster A: prob 0.9, actual rate 0.6
            np.array([1] * 90 + [0] * 10),  # cluster B: prob 0.6, actual rate 0.9
        ]
    )
    prob_1 = np.concatenate([np.full(100, 0.9), np.full(100, 0.6)])
    pd.DataFrame(
        {
            "actual": y,
            "predicted": (prob_1 >= 0.5).astype(int),
            "prob_0": 1.0 - prob_1,
            "prob_1": prob_1,
        }
    ).to_csv(tmp_path / "test_predictions.csv", index=False)

    result = tool_calibration_curve(str(tmp_path), n_bins=10, strategy="uniform")
    assert result["direction"] == "mixed"
    assert any("mixed miscalibration" in h for h in result["hints"])


def test_calibration_curve_flags_confident_but_wrong_bins(tmp_path):
    """Bins where the model is very confident (>0.9 / <0.1) but only 50% right
    trigger the high-confidence and low-confidence bucket hints."""
    y = np.concatenate(
        [
            np.array([1] * 25 + [0] * 25),  # prob 0.95, actual rate 0.5 → gap +0.45
            np.array([1] * 25 + [0] * 25),  # prob 0.05, actual rate 0.5 → gap -0.45
        ]
    )
    prob_1 = np.concatenate([np.full(50, 0.95), np.full(50, 0.05)])
    pd.DataFrame(
        {
            "actual": y,
            "predicted": (prob_1 >= 0.5).astype(int),
            "prob_0": 1.0 - prob_1,
            "prob_1": prob_1,
        }
    ).to_csv(tmp_path / "test_predictions.csv", index=False)

    result = tool_calibration_curve(str(tmp_path), n_bins=10, strategy="uniform")
    assert any("High-confidence bucket" in h for h in result["hints"])
    assert any("Low-confidence bucket" in h for h in result["hints"])
