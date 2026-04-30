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
