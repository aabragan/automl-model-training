"""Tests for tool_tune_model."""

from __future__ import annotations

from unittest.mock import patch

import pandas as pd
import pytest

from automl_model_training.tools import tool_tune_model


def test_tune_model_rejects_unknown_family(tmp_path):
    csv = tmp_path / "d.csv"
    pd.DataFrame({"x": [1, 2, 3], "y": [0, 1, 0]}).to_csv(csv, index=False)
    with pytest.raises(ValueError, match="model_family"):
        tool_tune_model(
            csv_path=str(csv),
            label="y",
            model_family="NONSENSE",
            n_trials=2,
            time_limit=10,
        )


def test_tune_model_passes_hyperparameter_kwargs_to_train(tmp_path):
    """Verify tune_model calls train_and_evaluate with hyperparameter_tune_kwargs."""
    csv = tmp_path / "d.csv"
    pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0], "y": [0, 1, 0, 1]}).to_csv(csv, index=False)

    with (
        patch("automl_model_training.tools.train_predict.train_and_evaluate") as mock_train,
        patch("automl_model_training.tools.train_predict.load_and_prepare") as mock_load,
    ):
        mock_load.return_value = (
            pd.DataFrame({"x": [1.0], "y": [0]}),
            pd.DataFrame({"x": [2.0], "y": [1]}),
            None,
            None,
            [],
        )
        mock_train.return_value = None

        tool_tune_model(
            csv_path=str(csv),
            label="y",
            model_family="GBM",
            n_trials=5,
            time_limit=60,
            output_dir=str(tmp_path / "out"),
        )

        # Verify train_and_evaluate was called with the right kwargs
        mock_train.assert_called_once()
        kwargs = mock_train.call_args.kwargs
        assert kwargs["hyperparameters"] == {"GBM": {}}
        assert kwargs["hyperparameter_tune_kwargs"]["num_trials"] == 5
        assert kwargs["time_limit"] == 60


def test_tune_model_reads_leaderboard_when_present(tmp_path):
    """When the run wrote leaderboard_test.csv, the top-5 rows are returned
    with only the expected columns."""
    csv = tmp_path / "d.csv"
    pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0], "y": [0, 1, 0, 1]}).to_csv(csv, index=False)

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    lb = pd.DataFrame(
        {
            "model": [f"GBM_T{i}" for i in range(7)],
            "score_val": [0.9 - i * 0.01 for i in range(7)],
            "score_test": [0.88 - i * 0.01 for i in range(7)],
            "fit_time": [10.0 + i for i in range(7)],
            "pred_time_val": [0.1] * 7,  # extra column must be filtered out
        }
    )
    lb.to_csv(run_dir / "leaderboard_test.csv", index=False)

    with (
        patch("automl_model_training.tools.train_predict.train_and_evaluate") as mock_train,
        patch("automl_model_training.tools.train_predict.load_and_prepare") as mock_load,
        patch("automl_model_training.tools.train_predict.make_run_dir") as mock_run_dir,
    ):
        mock_run_dir.return_value = str(run_dir)
        mock_load.return_value = (
            pd.DataFrame({"x": [1.0], "y": [0]}),
            pd.DataFrame({"x": [2.0], "y": [1]}),
            None,
            None,
            [],
        )
        mock_train.return_value = None

        result = tool_tune_model(
            csv_path=str(csv),
            label="y",
            model_family="GBM",
            n_trials=3,
            time_limit=30,
            output_dir=str(tmp_path / "out"),
        )

    assert len(result["leaderboard"]) == 5  # capped at top 5 of 7 rows
    assert result["leaderboard"][0]["model"] == "GBM_T0"
    # Only the allowed columns survive
    assert set(result["leaderboard"][0]) == {"model", "score_val", "score_test", "fit_time"}
    assert result["score"] == pytest.approx(0.88)
