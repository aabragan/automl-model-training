"""Tests for --min-confidence prediction filtering."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd

from automl_model_training.predict import predict_and_save


class TestMinConfidence:
    def _make_predictor(self) -> MagicMock:
        pred = MagicMock()
        pred.label = "target"
        pred.problem_type = "binary"
        pred.model_best = "LightGBM"
        pred.predict.return_value = pd.Series([0, 1, 0, 1])
        pred.evaluate.return_value = {"f1": 0.85}

        proba = pd.DataFrame({0: [0.9, 0.3, 0.6, 0.2], 1: [0.1, 0.7, 0.4, 0.8]})
        pred.predict_proba.return_value = proba
        return pred

    def test_flags_low_confidence_rows(self, tmp_path: Path):
        pred = self._make_predictor()
        data = pd.DataFrame({"feat_a": [1, 2, 3, 4], "target": [0, 1, 0, 1]})

        result = predict_and_save(pred, data, str(tmp_path), min_confidence=0.7)

        assert "flagged_low_confidence" in result.columns
        # Row 2 has confidence 0.6 (prob of predicted class 0), should be flagged
        assert result["flagged_low_confidence"].sum() >= 1

    def test_no_flag_column_without_threshold(self, tmp_path: Path):
        pred = self._make_predictor()
        data = pd.DataFrame({"feat_a": [1, 2, 3, 4], "target": [0, 1, 0, 1]})

        result = predict_and_save(pred, data, str(tmp_path))

        assert "flagged_low_confidence" not in result.columns

    def test_no_flag_for_regression(self, tmp_path: Path):
        pred = MagicMock()
        pred.label = "target"
        pred.problem_type = "regression"
        pred.model_best = "LightGBM"
        pred.predict.return_value = pd.Series([1.0, 2.0, 3.0])
        pred.evaluate.return_value = {"rmse": 0.5}
        data = pd.DataFrame({"feat_a": [1, 2, 3], "target": [1.1, 2.1, 3.1]})

        result = predict_and_save(pred, data, str(tmp_path), min_confidence=0.7)

        # Regression has no confidence column, so no flagging
        assert "flagged_low_confidence" not in result.columns
