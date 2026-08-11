"""CLI smoke tests for every entry point declared in pyproject.toml.

Each test exercises one ``main()`` (or variant) by patching ``sys.argv`` and
mocking the heavy-lifting function it dispatches to. The goal is to catch:

  - argparse breakage (missing/required flags, type mismatches)
  - dispatch bugs (wrong problem_type, wrong eval_metric)
  - regressions in the glue between argparse and the implementation

Real training is never invoked.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest


@pytest.fixture()
def sample_csv(tmp_path: Path) -> Path:
    """Tiny CSV that file-existence checks can find."""
    p = tmp_path / "data.csv"
    pd.DataFrame({"feat_a": [1, 2, 3, 4], "target": [0, 1, 0, 1]}).to_csv(p, index=False)
    return p


@pytest.fixture()
def fake_run_dir(tmp_path: Path) -> Path:
    """A directory that compare/predict --model-dir checks accept as existing."""
    d = tmp_path / "run"
    d.mkdir()
    return d


# ---------------------------------------------------------------------------
# train / train-binary / train-regression
# ---------------------------------------------------------------------------


class TestTrainCli:
    @patch("automl_model_training.train._run")
    def test_train_main_dispatches_with_auto_problem_type(
        self, mock_run: MagicMock, monkeypatch, sample_csv: Path
    ):
        from automl_model_training.train import main

        monkeypatch.setattr(sys, "argv", ["train", str(sample_csv), "--label", "target"])
        main()

        mock_run.assert_called_once()
        args, kwargs = mock_run.call_args
        assert kwargs["problem_type"] is None  # auto-detect
        assert args[0].label == "target"

    @patch("automl_model_training.train._run")
    def test_train_main_passes_through_flags(
        self, mock_run: MagicMock, monkeypatch, sample_csv: Path
    ):
        from automl_model_training.train import main

        monkeypatch.setattr(
            sys,
            "argv",
            [
                "train",
                str(sample_csv),
                "--label",
                "target",
                "--time-limit",
                "30",
                "--preset",
                "medium",
                "--cv-folds",
                "5",
                "--cv-no-shuffle",
            ],
        )
        main()

        args, _ = mock_run.call_args
        ns = args[0]
        assert ns.time_limit == 30
        assert ns.preset == "medium"
        assert ns.cv_folds == 5
        assert ns.cv_no_shuffle is True

    @patch("automl_model_training.train._run")
    def test_train_main_cv_shuffle_defaults_on(
        self, mock_run: MagicMock, monkeypatch, sample_csv: Path
    ):
        from automl_model_training.train import main

        monkeypatch.setattr(sys, "argv", ["train", str(sample_csv), "--label", "target"])
        main()

        args, _ = mock_run.call_args
        assert args[0].cv_no_shuffle is False

    @patch("automl_model_training.train._run")
    def test_train_binary_forces_binary(self, mock_run: MagicMock, monkeypatch, sample_csv: Path):
        from automl_model_training.train import train_binary

        monkeypatch.setattr(sys, "argv", ["train-binary", str(sample_csv), "--label", "target"])
        train_binary()

        _, kwargs = mock_run.call_args
        assert kwargs["problem_type"] == "binary"

    @patch("automl_model_training.train._run")
    def test_train_regression_forces_regression_and_rmse(
        self, mock_run: MagicMock, monkeypatch, sample_csv: Path
    ):
        from automl_model_training.train import train_regression

        monkeypatch.setattr(sys, "argv", ["train-regression", str(sample_csv), "--label", "target"])
        train_regression()

        args, kwargs = mock_run.call_args
        assert kwargs["problem_type"] == "regression"
        # eval_metric default set by train_regression
        assert args[0].eval_metric == "root_mean_squared_error"


# ---------------------------------------------------------------------------
# predict / predict-binary / predict-regression
# ---------------------------------------------------------------------------


class TestPredictCli:
    @patch("automl_model_training.predict.predict_and_save")
    @patch("automl_model_training.predict.load_predictor")
    def test_predict_main_passes_required_args(
        self,
        mock_load: MagicMock,
        mock_save: MagicMock,
        monkeypatch,
        sample_csv: Path,
        fake_run_dir: Path,
    ):
        from automl_model_training.predict import main

        mock_load.return_value = MagicMock()

        monkeypatch.setattr(
            sys,
            "argv",
            ["predict", str(sample_csv), "--model-dir", str(fake_run_dir)],
        )
        main()

        mock_load.assert_called_once_with(str(fake_run_dir))
        mock_save.assert_called_once()

    @patch("automl_model_training.predict.predict_and_save")
    @patch("automl_model_training.predict.load_predictor")
    def test_predict_main_forwards_min_confidence(
        self,
        mock_load: MagicMock,
        mock_save: MagicMock,
        monkeypatch,
        sample_csv: Path,
        fake_run_dir: Path,
    ):
        from automl_model_training.predict import main

        mock_load.return_value = MagicMock()

        monkeypatch.setattr(
            sys,
            "argv",
            [
                "predict",
                str(sample_csv),
                "--model-dir",
                str(fake_run_dir),
                "--min-confidence",
                "0.7",
            ],
        )
        main()

        _, kwargs = mock_save.call_args
        assert kwargs["min_confidence"] == 0.7

    @patch("automl_model_training.predict.predict_and_save")
    @patch("automl_model_training.predict.load_predictor")
    def test_predict_main_errors_on_missing_csv(
        self,
        mock_load: MagicMock,
        mock_save: MagicMock,
        monkeypatch,
        tmp_path: Path,
        fake_run_dir: Path,
    ):
        from automl_model_training.predict import main

        monkeypatch.setattr(
            sys,
            "argv",
            ["predict", str(tmp_path / "ghost.csv"), "--model-dir", str(fake_run_dir)],
        )
        with pytest.raises(SystemExit):
            main()
        mock_load.assert_not_called()

    @patch("automl_model_training.predict.predict_and_save")
    @patch("automl_model_training.predict.load_predictor")
    def test_predict_main_errors_on_missing_model_dir(
        self,
        mock_load: MagicMock,
        mock_save: MagicMock,
        monkeypatch,
        sample_csv: Path,
        tmp_path: Path,
    ):
        from automl_model_training.predict import main

        monkeypatch.setattr(
            sys,
            "argv",
            ["predict", str(sample_csv), "--model-dir", str(tmp_path / "no_model")],
        )
        with pytest.raises(SystemExit):
            main()
        mock_load.assert_not_called()

    @patch("automl_model_training.predict.predict_and_save")
    @patch("automl_model_training.predict.load_predictor")
    def test_predict_main_loads_train_data_for_drift_check(
        self,
        mock_load: MagicMock,
        mock_save: MagicMock,
        monkeypatch,
        sample_csv: Path,
        fake_run_dir: Path,
        tmp_path: Path,
    ):
        from automl_model_training.predict import main

        mock_load.return_value = MagicMock()
        train_run = tmp_path / "train_run"
        train_run.mkdir()
        pd.DataFrame({"feat_a": [1, 2], "target": [0, 1]}).to_csv(
            train_run / "train_raw.csv", index=False
        )

        monkeypatch.setattr(
            sys,
            "argv",
            [
                "predict",
                str(sample_csv),
                "--model-dir",
                str(fake_run_dir),
                "--drift-check",
                str(train_run),
            ],
        )
        main()

        _, kwargs = mock_save.call_args
        assert isinstance(kwargs["train_data"], pd.DataFrame)
        assert list(kwargs["train_data"].columns) == ["feat_a", "target"]

    @patch("automl_model_training.predict.predict_and_save")
    @patch("automl_model_training.predict.load_predictor")
    def test_predict_main_warns_when_train_raw_missing(
        self,
        mock_load: MagicMock,
        mock_save: MagicMock,
        monkeypatch,
        sample_csv: Path,
        fake_run_dir: Path,
        tmp_path: Path,
    ):
        from automl_model_training.predict import main

        mock_load.return_value = MagicMock()
        empty_run = tmp_path / "empty_run"
        empty_run.mkdir()

        monkeypatch.setattr(
            sys,
            "argv",
            [
                "predict",
                str(sample_csv),
                "--model-dir",
                str(fake_run_dir),
                "--drift-check",
                str(empty_run),
            ],
        )
        main()

        _, kwargs = mock_save.call_args
        assert kwargs["train_data"] is None

    @patch("automl_model_training.predict.predict_and_save")
    @patch("automl_model_training.predict.load_predictor")
    def test_predict_main_forwards_decision_threshold(
        self,
        mock_load: MagicMock,
        mock_save: MagicMock,
        monkeypatch,
        sample_csv: Path,
        fake_run_dir: Path,
    ):
        from automl_model_training.predict import main

        mock_load.return_value = MagicMock()

        monkeypatch.setattr(
            sys,
            "argv",
            [
                "predict",
                str(sample_csv),
                "--model-dir",
                str(fake_run_dir),
                "--decision-threshold",
                "0.3",
            ],
        )
        main()

        _, kwargs = mock_save.call_args
        assert kwargs["decision_threshold"] == 0.3

    def test_predict_binary_and_regression_are_aliases_to_main(
        self, monkeypatch, sample_csv: Path, fake_run_dir: Path
    ):
        # predict_binary and predict_regression both call main() — verify they exist
        # and don't crash before argparse runs
        from automl_model_training.predict import predict_binary, predict_regression

        with (
            patch("automl_model_training.predict.predict_and_save"),
            patch("automl_model_training.predict.load_predictor", return_value=MagicMock()),
        ):
            monkeypatch.setattr(
                sys,
                "argv",
                ["predict-binary", str(sample_csv), "--model-dir", str(fake_run_dir)],
            )
            predict_binary()

            monkeypatch.setattr(
                sys,
                "argv",
                ["predict-regression", str(sample_csv), "--model-dir", str(fake_run_dir)],
            )
            predict_regression()


# ---------------------------------------------------------------------------
# backtest
# ---------------------------------------------------------------------------


class TestBacktestCli:
    @patch("automl_model_training.backtest.temporal_backtest")
    def test_backtest_requires_date_column(self, mock_bt: MagicMock, monkeypatch, sample_csv: Path):
        from automl_model_training.backtest import main

        monkeypatch.setattr(sys, "argv", ["backtest", str(sample_csv)])
        with pytest.raises(SystemExit):
            main()
        mock_bt.assert_not_called()

    @patch("automl_model_training.backtest.temporal_backtest")
    def test_backtest_forwards_date_column_and_n_splits(
        self, mock_bt: MagicMock, monkeypatch, sample_csv: Path
    ):
        from automl_model_training.backtest import main

        monkeypatch.setattr(
            sys,
            "argv",
            [
                "backtest",
                str(sample_csv),
                "--date-column",
                "ds",
                "--n-splits",
                "3",
            ],
        )
        main()

        _, kwargs = mock_bt.call_args
        assert kwargs["date_column"] == "ds"
        assert kwargs["n_splits"] == 3


# ---------------------------------------------------------------------------
# profile
# ---------------------------------------------------------------------------


class TestProfileCli:
    @patch("automl_model_training.profile.save_profile_report")
    @patch("automl_model_training.profile.recommend_features_to_drop", return_value=[])
    @patch("automl_model_training.profile.find_highly_correlated_pairs", return_value=[])
    @patch("automl_model_training.profile.compute_correlation_matrix")
    def test_profile_main_runs_to_completion(
        self,
        mock_corr: MagicMock,
        mock_pairs: MagicMock,
        mock_recs: MagicMock,
        mock_save: MagicMock,
        monkeypatch,
        sample_csv: Path,
    ):
        from automl_model_training.profile import main

        mock_corr.return_value = pd.DataFrame()

        monkeypatch.setattr(sys, "argv", ["profile", str(sample_csv), "--label", "target"])
        main()

        mock_corr.assert_called_once()
        mock_save.assert_called_once()


# ---------------------------------------------------------------------------
# experiments
# ---------------------------------------------------------------------------


class TestExperimentsCli:
    @patch("automl_model_training.experiment.compare_experiments")
    def test_experiments_prints_when_no_output(self, mock_compare: MagicMock, monkeypatch, capsys):
        from automl_model_training.experiment import main

        mock_compare.return_value = pd.DataFrame(
            [{"run_id": "r1", "param_preset": "best", "metric_score": 0.9}]
        )

        monkeypatch.setattr(sys, "argv", ["experiments"])
        main()

        captured = capsys.readouterr()
        assert "r1" in captured.out

    @patch("automl_model_training.experiment.compare_experiments")
    def test_experiments_handles_empty_log(self, mock_compare: MagicMock, monkeypatch):
        from automl_model_training.experiment import main

        mock_compare.return_value = pd.DataFrame()

        monkeypatch.setattr(sys, "argv", ["experiments"])
        main()  # Should not raise

    @patch("automl_model_training.experiment.compare_experiments")
    def test_experiments_writes_to_output(
        self, mock_compare: MagicMock, monkeypatch, tmp_path: Path
    ):
        from automl_model_training.experiment import main

        mock_compare.return_value = pd.DataFrame([{"run_id": "r1", "metric_score": 0.9}])

        out = tmp_path / "comparison.csv"
        monkeypatch.setattr(sys, "argv", ["experiments", "--output", str(out)])
        main()

        assert out.exists()


# ---------------------------------------------------------------------------
# compare
# ---------------------------------------------------------------------------


class TestCompareCli:
    @patch("automl_model_training.compare.compare_runs")
    def test_compare_rejects_missing_run_dir(
        self, mock_cmp: MagicMock, monkeypatch, tmp_path: Path
    ):
        from automl_model_training.compare import main

        ghost = tmp_path / "does_not_exist"
        monkeypatch.setattr(sys, "argv", ["compare", str(ghost)])
        with pytest.raises(SystemExit):
            main()
        mock_cmp.assert_not_called()

    @patch("automl_model_training.compare.compare_runs")
    def test_compare_accepts_multiple_runs(
        self, mock_cmp: MagicMock, monkeypatch, fake_run_dir: Path, tmp_path: Path
    ):
        from automl_model_training.compare import main

        run2 = tmp_path / "run2"
        run2.mkdir()
        mock_cmp.return_value = pd.DataFrame([{"run_id": "r1"}, {"run_id": "r2"}])

        monkeypatch.setattr(sys, "argv", ["compare", str(fake_run_dir), str(run2)])
        main()

        args, _ = mock_cmp.call_args
        assert len(args[0]) == 2


# ---------------------------------------------------------------------------
# agent-binary / agent-regression
# ---------------------------------------------------------------------------


class TestAgentCli:
    @patch("automl_model_training.agent.run_agent")
    def test_agent_binary_requires_target_f1(
        self, mock_run: MagicMock, monkeypatch, sample_csv: Path
    ):
        from automl_model_training.agent import agent_binary

        monkeypatch.setattr(sys, "argv", ["agent-binary", str(sample_csv), "--label", "target"])
        with pytest.raises(SystemExit):
            agent_binary()
        mock_run.assert_not_called()

    @patch("automl_model_training.agent.run_agent")
    def test_agent_binary_dispatches_with_f1(
        self, mock_run: MagicMock, monkeypatch, sample_csv: Path
    ):
        from automl_model_training.agent import agent_binary

        monkeypatch.setattr(
            sys,
            "argv",
            [
                "agent-binary",
                str(sample_csv),
                "--label",
                "target",
                "--target-f1",
                "0.9",
                "--max-iterations",
                "3",
            ],
        )
        agent_binary()

        _, kwargs = mock_run.call_args
        assert kwargs["problem_type"] == "binary"
        assert kwargs["eval_metric"] == "f1"
        assert kwargs["target_value"] == 0.9

    @patch("automl_model_training.agent.run_agent")
    def test_agent_regression_dispatches_with_rmse_and_lower_is_better(
        self, mock_run: MagicMock, monkeypatch, sample_csv: Path
    ):
        from automl_model_training.agent import agent_regression

        monkeypatch.setattr(
            sys,
            "argv",
            [
                "agent-regression",
                str(sample_csv),
                "--label",
                "target",
                "--target-rmse",
                "5.0",
                "--max-iterations",
                "3",
            ],
        )
        agent_regression()

        _, kwargs = mock_run.call_args
        assert kwargs["problem_type"] == "regression"
        assert kwargs["eval_metric"] == "root_mean_squared_error"
        assert kwargs["target_value"] == 5.0
        assert kwargs["higher_is_better"] is False


# ---------------------------------------------------------------------------
# agent-ollama
# ---------------------------------------------------------------------------


class TestOllamaAgentCli:
    @patch("automl_model_training.ollama_agent.run_ollama_agent")
    def test_ollama_main_uses_default_model(
        self, mock_run: MagicMock, monkeypatch, sample_csv: Path
    ):
        from automl_model_training.ollama_agent import main

        monkeypatch.setattr(sys, "argv", ["agent-ollama", str(sample_csv)])
        main()

        _, kwargs = mock_run.call_args
        assert kwargs["model"] == "qwen2.5:14b"
        assert kwargs["base_url"] == "http://localhost:11434/v1"

    @patch("automl_model_training.ollama_agent.run_ollama_agent")
    def test_ollama_main_forwards_overrides(
        self, mock_run: MagicMock, monkeypatch, sample_csv: Path
    ):
        from automl_model_training.ollama_agent import main

        monkeypatch.setattr(
            sys,
            "argv",
            [
                "agent-ollama",
                str(sample_csv),
                "--model",
                "llama3.1:8b",
                "--max-iterations",
                "10",
            ],
        )
        main()

        _, kwargs = mock_run.call_args
        assert kwargs["model"] == "llama3.1:8b"
        assert kwargs["max_iterations"] == 10
