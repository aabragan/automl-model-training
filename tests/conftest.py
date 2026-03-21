"""Shared fixtures for unit tests.

All AutoGluon predictor interactions are mocked so tests run instantly
without needing real model training.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest


@pytest.fixture()
def tmp_output(tmp_path: Path) -> Path:
    """Return a temporary output directory."""
    out = tmp_path / "output"
    out.mkdir()
    return out


@pytest.fixture()
def binary_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Small binary classification dataset (train, test)."""
    rng = np.random.RandomState(42)
    n_train, n_test = 200, 50

    def _make(n: int) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "feat_a": rng.randn(n),
                "feat_b": rng.randn(n),
                "feat_c": rng.choice(["x", "y"], n),
                "target": rng.choice([0, 1], n, p=[0.7, 0.3]),
            }
        )

    return _make(n_train), _make(n_test)


@pytest.fixture()
def regression_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Small regression dataset (train, test)."""
    rng = np.random.RandomState(42)
    n_train, n_test = 200, 50

    def _make(n: int) -> pd.DataFrame:
        x = rng.randn(n)
        return pd.DataFrame(
            {
                "feat_a": x,
                "feat_b": rng.randn(n),
                "target": x * 2.5 + rng.randn(n) * 0.1,
            }
        )

    return _make(n_train), _make(n_test)


@pytest.fixture()
def mock_binary_predictor(binary_data):
    """A MagicMock that behaves like a trained binary TabularPredictor."""
    _, test = binary_data
    pred = MagicMock()
    pred.label = "target"
    pred.problem_type = "binary"
    pred.eval_metric = "f1"
    pred.model_best = "LightGBM"
    pred.features.return_value = ["feat_a", "feat_b", "feat_c"]

    y_pred = test["target"].copy()
    pred.predict.return_value = y_pred

    proba = pd.DataFrame(
        {
            0: np.where(y_pred == 0, 0.8, 0.2),
            1: np.where(y_pred == 1, 0.8, 0.2),
        }
    )
    pred.predict_proba.return_value = proba
    pred.evaluate.return_value = {"f1": 0.85, "accuracy": 0.90}
    return pred
