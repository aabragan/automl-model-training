"""Tests for compute_shap_values."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from automl_model_training.evaluate.explain import compute_shap_values


class TestComputeShapValues:
    def _make_predictor(self, problem_type: str = "binary") -> MagicMock:
        pred = MagicMock()
        pred.label = "target"
        pred.problem_type = problem_type
        pred.predict.return_value = pd.Series([1.0, 2.0, 3.0])
        pred.predict_proba.return_value = pd.DataFrame(
            {0: [0.8, 0.3, 0.6], 1: [0.2, 0.7, 0.4]}
        )
        return pred

    def _make_data(self, n: int = 10) -> pd.DataFrame:
        rng = np.random.RandomState(42)
        return pd.DataFrame({
            "feat_a": rng.randn(n),
            "feat_b": rng.randn(n),
            "target": rng.choice([0, 1], n),
        })

    @patch("automl_model_training.evaluate.explain.shap")
    def test_binary_returns_2d(self, mock_shap: MagicMock):
        pred = self._make_predictor("binary")
        data = self._make_data(10)

        # Simulate SHAP returning (2, n_samples, n_features) for binary
        mock_explainer = MagicMock()
        mock_shap.KernelExplainer.return_value = mock_explainer
        mock_shap.sample.return_value = data[["feat_a", "feat_b"]].head(5)
        mock_explainer.shap_values.return_value = [
            np.zeros((10, 2)),  # class 0
            np.ones((10, 2)),   # class 1
        ]
        mock_explainer.expected_value = np.array([0.5, 0.5])

        shap_vals, base, features = compute_shap_values(pred, data, max_samples=10)

        # Should select positive class (index 1) and return 2D
        assert shap_vals.ndim == 2
        assert shap_vals.shape == (10, 2)
        assert np.all(shap_vals == 1.0)
        assert features == ["feat_a", "feat_b"]

    @patch("automl_model_training.evaluate.explain.shap")
    def test_regression_returns_2d(self, mock_shap: MagicMock):
        pred = self._make_predictor("regression")
        data = self._make_data(10)

        mock_explainer = MagicMock()
        mock_shap.KernelExplainer.return_value = mock_explainer
        mock_shap.sample.return_value = data[["feat_a", "feat_b"]].head(5)
        mock_explainer.shap_values.return_value = np.ones((10, 2))
        mock_explainer.expected_value = np.array(3.0)

        shap_vals, base, features = compute_shap_values(pred, data, max_samples=10)

        assert shap_vals.ndim == 2
        assert shap_vals.shape == (10, 2)

    @patch("automl_model_training.evaluate.explain.shap")
    def test_subsamples_large_datasets(self, mock_shap: MagicMock):
        pred = self._make_predictor("regression")
        data = self._make_data(100)

        mock_explainer = MagicMock()
        mock_shap.KernelExplainer.return_value = mock_explainer
        mock_shap.sample.return_value = data[["feat_a", "feat_b"]].head(5)
        # Return shape matching max_samples=20, not full 100
        mock_explainer.shap_values.return_value = np.ones((20, 2))
        mock_explainer.expected_value = np.array(1.0)

        shap_vals, _, _ = compute_shap_values(pred, data, max_samples=20)

        assert shap_vals.shape[0] == 20
