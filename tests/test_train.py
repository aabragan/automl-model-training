"""Tests for train_and_evaluate core logic."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from automl_model_training.train import _base_parser, _run, train_and_evaluate


def _make_mock_predictor(problem_type: str = "binary", n_test: int = 5) -> MagicMock:
    """Build a mock predictor that satisfies train_and_evaluate's calls."""
    pred = MagicMock()
    pred.label = "target"
    pred.problem_type = problem_type
    pred.eval_metric = "f1" if problem_type == "binary" else "root_mean_squared_error"
    pred.model_best = "LightGBM"
    pred.features.return_value = ["feat_a", "feat_b"]

    # predict/predict_proba return values sized to match test_raw.
    # Alternate 0s and 1s so sklearn metrics have both classes to work with.
    pred.predict.side_effect = lambda data: pd.Series(
        np.arange(len(data)) % 2, index=data.index, dtype=int
    )
    pred.predict_proba.side_effect = lambda data: pd.DataFrame(
        {
            0: np.where(np.arange(len(data)) % 2 == 0, 0.8, 0.2),
            1: np.where(np.arange(len(data)) % 2 == 0, 0.2, 0.8),
        },
        index=data.index,
    )

    lb = pd.DataFrame(
        {
            "model": ["LightGBM", "CatBoost"],
            "score_val": [0.90, 0.85],
            "score_test": [0.88, 0.82],
            "fit_time": [10.0, 15.0],
            "pred_time_val": [0.1, 0.2],
        }
    )
    pred.leaderboard.return_value = lb

    pred.refit_full.return_value = {"LightGBM": "LightGBM_FULL"}
    pred.evaluate.return_value = {"f1": 0.88, "accuracy": 0.92}

    importance = pd.DataFrame(
        {"importance": [0.5, 0.3], "stddev": [0.01, 0.02]},
        index=["feat_a", "feat_b"],
    )
    pred.feature_importance.return_value = importance

    pred.info.return_value = {"model_info": {}}
    return pred


class TestTrainAndEvaluate:
    @patch("automl_model_training.train.TabularPredictor")
    def test_returns_predictor(self, mock_cls: MagicMock, tmp_path: Path):
        mock_pred = _make_mock_predictor()
        mock_cls.return_value = mock_pred

        result = train_and_evaluate(
            train_raw=pd.DataFrame(
                {"feat_a": [1, 2, 3, 4], "feat_b": [3, 4, 5, 6], "target": [0, 1, 0, 1]}
            ),
            test_raw=pd.DataFrame({"feat_a": [5, 6], "feat_b": [6, 7], "target": [0, 1]}),
            label="target",
            problem_type="binary",
            eval_metric="f1",
            time_limit=None,
            preset="best",
            output_dir=str(tmp_path),
        )

        assert result is mock_pred

    @patch("automl_model_training.train.TabularPredictor")
    def test_saves_leaderboard_and_model_info(self, mock_cls: MagicMock, tmp_path: Path):
        mock_pred = _make_mock_predictor()
        mock_cls.return_value = mock_pred

        train_and_evaluate(
            train_raw=pd.DataFrame(
                {"feat_a": [1, 2, 3, 4], "feat_b": [3, 4, 5, 6], "target": [0, 1, 0, 1]}
            ),
            test_raw=pd.DataFrame({"feat_a": [5, 6], "feat_b": [6, 7], "target": [0, 1]}),
            label="target",
            problem_type="binary",
            eval_metric="f1",
            time_limit=None,
            preset="best",
            output_dir=str(tmp_path),
        )

        assert (tmp_path / "leaderboard.csv").exists()
        assert (tmp_path / "leaderboard_test.csv").exists()
        assert (tmp_path / "feature_importance.csv").exists()
        assert (tmp_path / "model_info.json").exists()

        info = json.loads((tmp_path / "model_info.json").read_text())
        assert info["best_model"] == "LightGBM"
        assert info["problem_type"] == "binary"

    @patch("automl_model_training.train.TabularPredictor")
    def test_calls_fit_with_correct_params(self, mock_cls: MagicMock, tmp_path: Path):
        mock_pred = _make_mock_predictor()
        mock_cls.return_value = mock_pred

        train_and_evaluate(
            train_raw=pd.DataFrame({"feat_a": [1, 2, 3, 4], "target": [0, 1, 0, 1]}),
            test_raw=pd.DataFrame({"feat_a": [5, 6], "target": [0, 1]}),
            label="target",
            problem_type="binary",
            eval_metric="f1",
            time_limit=60,
            preset="high_quality",
            output_dir=str(tmp_path),
        )

        mock_pred.fit.assert_called_once()
        call_kwargs = mock_pred.fit.call_args[1]
        assert call_kwargs["presets"] == "high_quality"
        assert call_kwargs["time_limit"] == 60
        assert call_kwargs["refit_full"] is True
        assert call_kwargs["set_best_to_refit_full"] is True
        assert call_kwargs["dynamic_stacking"] is False

    @patch("automl_model_training.train.TabularPredictor")
    def test_regression_dispatches_correctly(self, mock_cls: MagicMock, tmp_path: Path):
        mock_pred = _make_mock_predictor(problem_type="regression")
        mock_pred.predict.side_effect = lambda data: pd.Series(
            np.full(len(data), 1.5), index=data.index
        )
        mock_cls.return_value = mock_pred

        train_and_evaluate(
            train_raw=pd.DataFrame({"feat_a": [1, 2], "target": [1.0, 2.0]}),
            test_raw=pd.DataFrame({"feat_a": [3, 4], "target": [3.0, 4.0]}),
            label="target",
            problem_type="regression",
            eval_metric="root_mean_squared_error",
            time_limit=None,
            preset="best",
            output_dir=str(tmp_path),
        )

        # Should call save_regression_artifacts, not classification
        assert (tmp_path / "model_info.json").exists()

    @patch("automl_model_training.train.TabularPredictor")
    def test_prune_flag(self, mock_cls: MagicMock, tmp_path: Path):
        mock_pred = _make_mock_predictor()
        mock_cls.return_value = mock_pred

        train_and_evaluate(
            train_raw=pd.DataFrame({"feat_a": [1, 2, 3, 4], "target": [0, 1, 0, 1]}),
            test_raw=pd.DataFrame({"feat_a": [5, 6], "target": [0, 1]}),
            label="target",
            problem_type="binary",
            eval_metric="f1",
            time_limit=None,
            preset="best",
            output_dir=str(tmp_path),
            prune=True,
        )

        # Pruning should have been triggered
        assert (tmp_path / "model_info.json").exists()


class TestFitKwargsPassthrough:
    @patch("automl_model_training.train.TabularPredictor")
    def test_hyperparameters_forwarded_to_fit(self, mock_cls: MagicMock, tmp_path: Path):
        mock_pred = _make_mock_predictor()
        mock_cls.return_value = mock_pred

        hp = {"GBM": {}}
        hpo = {"num_trials": 5, "searcher": "auto", "scheduler": "local"}
        train_and_evaluate(
            train_raw=pd.DataFrame({"feat_a": [1, 2, 3, 4], "target": [0, 1, 0, 1]}),
            test_raw=pd.DataFrame({"feat_a": [5, 6], "target": [0, 1]}),
            label="target",
            problem_type="binary",
            eval_metric="f1",
            time_limit=None,
            preset="best",
            output_dir=str(tmp_path),
            hyperparameters=hp,
            hyperparameter_tune_kwargs=hpo,
        )

        call_kwargs = mock_pred.fit.call_args[1]
        assert call_kwargs["hyperparameters"] == hp
        assert call_kwargs["hyperparameter_tune_kwargs"] == hpo

    @patch("automl_model_training.train.TabularPredictor")
    def test_no_hyperparameter_kwargs_by_default(self, mock_cls: MagicMock, tmp_path: Path):
        mock_pred = _make_mock_predictor()
        mock_cls.return_value = mock_pred

        train_and_evaluate(
            train_raw=pd.DataFrame({"feat_a": [1, 2, 3, 4], "target": [0, 1, 0, 1]}),
            test_raw=pd.DataFrame({"feat_a": [5, 6], "target": [0, 1]}),
            label="target",
            problem_type="binary",
            eval_metric="f1",
            time_limit=None,
            preset="best",
            output_dir=str(tmp_path),
        )

        call_kwargs = mock_pred.fit.call_args[1]
        assert "hyperparameters" not in call_kwargs
        assert "hyperparameter_tune_kwargs" not in call_kwargs


class TestExplainFlag:
    @patch("automl_model_training.train.save_explainability_artifacts")
    @patch("automl_model_training.train.TabularPredictor")
    def test_explain_true_saves_shap_artifacts(
        self, mock_cls: MagicMock, mock_explain: MagicMock, tmp_path: Path
    ):
        mock_pred = _make_mock_predictor()
        mock_cls.return_value = mock_pred

        train_and_evaluate(
            train_raw=pd.DataFrame({"feat_a": [1, 2, 3, 4], "target": [0, 1, 0, 1]}),
            test_raw=pd.DataFrame({"feat_a": [5, 6], "target": [0, 1]}),
            label="target",
            problem_type="binary",
            eval_metric="f1",
            time_limit=None,
            preset="best",
            output_dir=str(tmp_path),
            explain=True,
        )

        mock_explain.assert_called_once()

    @patch("automl_model_training.train.save_explainability_artifacts")
    @patch("automl_model_training.train.TabularPredictor")
    def test_explain_false_skips_shap(
        self, mock_cls: MagicMock, mock_explain: MagicMock, tmp_path: Path
    ):
        mock_pred = _make_mock_predictor()
        mock_cls.return_value = mock_pred

        train_and_evaluate(
            train_raw=pd.DataFrame({"feat_a": [1, 2, 3, 4], "target": [0, 1, 0, 1]}),
            test_raw=pd.DataFrame({"feat_a": [5, 6], "target": [0, 1]}),
            label="target",
            problem_type="binary",
            eval_metric="f1",
            time_limit=None,
            preset="best",
            output_dir=str(tmp_path),
        )

        mock_explain.assert_not_called()


# --- _run CLI orchestration ---


def _parse_run_args(csv_path: Path, tmp_path: Path, *extra: str):
    parser = _base_parser("test")
    args = parser.parse_args(
        [str(csv_path), "--label", "target", "--output-dir", str(tmp_path / "out"), *extra]
    )
    return args, parser


@pytest.fixture()
def train_csv(tmp_path: Path) -> Path:
    p = tmp_path / "data.csv"
    pd.DataFrame({"feat_a": [1, 2, 3, 4], "feat_b": [4, 3, 2, 1], "target": [0, 1, 0, 1]}).to_csv(
        p, index=False
    )
    return p


class TestRunCli:
    @patch("automl_model_training.train.record_experiment")
    @patch("automl_model_training.train.load_cv_train")
    def test_basic_run_trains_and_records(
        self, mock_train: MagicMock, mock_record: MagicMock, train_csv: Path, tmp_path: Path
    ):
        args, parser = _parse_run_args(train_csv, tmp_path)

        _run(args, problem_type=None, parser=parser)

        mock_train.assert_called_once()
        kwargs = mock_train.call_args[1]
        assert kwargs["csv_path"] == str(train_csv)
        assert kwargs["label"] == "target"
        mock_record.assert_called_once()
        # No model_info.json written by the mocked training → empty metrics
        assert mock_record.call_args[1]["metrics"] == {}

    @patch("automl_model_training.train.record_experiment")
    @patch("automl_model_training.train.load_cv_train")
    def test_missing_csv_errors(
        self, mock_train: MagicMock, mock_record: MagicMock, tmp_path: Path
    ):
        args, parser = _parse_run_args(tmp_path / "ghost.csv", tmp_path)

        with pytest.raises(SystemExit):
            _run(args, problem_type=None, parser=parser)
        mock_train.assert_not_called()

    @patch("automl_model_training.train.record_experiment")
    @patch("automl_model_training.train.load_cv_train")
    def test_experiment_metrics_read_from_artifacts(
        self, mock_train: MagicMock, mock_record: MagicMock, train_csv: Path, tmp_path: Path
    ):
        def _write_artifacts(**kwargs):
            out = Path(kwargs["output_dir"])
            (out / "model_info.json").write_text(
                json.dumps({"best_model": "LightGBM_FULL", "eval_metric": "f1"})
            )
            analysis = {
                "eval_metric": "f1",
                "test_scores": {"f1": 0.91, "accuracy": 0.95},
                "score_convention": "higher_is_better",
            }
            (out / "analysis.json").write_text(json.dumps(analysis))
            return (pd.DataFrame(), pd.DataFrame())

        mock_train.side_effect = _write_artifacts
        args, parser = _parse_run_args(train_csv, tmp_path)

        _run(args, problem_type="binary", parser=parser)

        metrics = mock_record.call_args[1]["metrics"]
        assert metrics["best_test_score"] == 0.91
        assert metrics["score_convention"] == "higher_is_better"
        assert metrics["best_model"] == "LightGBM_FULL"

    @patch("automl_model_training.train.record_experiment")
    @patch("automl_model_training.train.load_cv_train")
    @patch("automl_model_training.train.save_profile_report")
    @patch("automl_model_training.train.recommend_features_to_drop")
    @patch("automl_model_training.train.find_highly_correlated_pairs", return_value=[])
    @patch("automl_model_training.train.compute_correlation_matrix")
    def test_profile_flag_adds_recommended_drops(
        self,
        mock_corr: MagicMock,
        mock_pairs: MagicMock,
        mock_recs: MagicMock,
        mock_save: MagicMock,
        mock_train: MagicMock,
        mock_record: MagicMock,
        train_csv: Path,
        tmp_path: Path,
    ):
        mock_corr.return_value = pd.DataFrame()
        mock_recs.return_value = [{"feature": "feat_b"}]
        args, parser = _parse_run_args(train_csv, tmp_path, "--profile")

        _run(args, problem_type=None, parser=parser)

        mock_save.assert_called_once()
        kwargs = mock_train.call_args[1]
        assert "feat_b" in kwargs["features_to_drop"]

    @patch("automl_model_training.train.record_experiment")
    @patch("automl_model_training.train.load_cv_train")
    @patch("automl_model_training.train.save_profile_report")
    @patch("automl_model_training.train.recommend_features_to_drop")
    @patch("automl_model_training.train.find_highly_correlated_pairs", return_value=[])
    @patch("automl_model_training.train.compute_correlation_matrix")
    def test_profile_flag_skips_already_dropped_features(
        self,
        mock_corr: MagicMock,
        mock_pairs: MagicMock,
        mock_recs: MagicMock,
        mock_save: MagicMock,
        mock_train: MagicMock,
        mock_record: MagicMock,
        train_csv: Path,
        tmp_path: Path,
    ):
        mock_corr.return_value = pd.DataFrame()
        mock_recs.return_value = [{"feature": "feat_b"}]
        args, parser = _parse_run_args(train_csv, tmp_path, "--profile", "--drop", "feat_b")

        _run(args, problem_type=None, parser=parser)

        kwargs = mock_train.call_args[1]
        assert kwargs["features_to_drop"].count("feat_b") == 1

    @patch("automl_model_training.train.record_experiment")
    @patch("automl_model_training.train._read_low_importance_features")
    @patch("automl_model_training.train.load_cv_train")
    def test_auto_drop_retrains_with_low_importance_features(
        self,
        mock_train: MagicMock,
        mock_low: MagicMock,
        mock_record: MagicMock,
        train_csv: Path,
        tmp_path: Path,
    ):
        mock_low.return_value = ["feat_b"]
        args, parser = _parse_run_args(train_csv, tmp_path, "--auto-drop")

        _run(args, problem_type=None, parser=parser)

        assert mock_train.call_count == 2
        retrain_kwargs = mock_train.call_args_list[1][1]
        assert "feat_b" in retrain_kwargs["features_to_drop"]
        assert retrain_kwargs["cv_folds"] is None

    @patch("automl_model_training.train.record_experiment")
    @patch("automl_model_training.train._read_low_importance_features")
    @patch("automl_model_training.train.load_cv_train")
    def test_auto_drop_skips_retrain_when_nothing_to_drop(
        self,
        mock_train: MagicMock,
        mock_low: MagicMock,
        mock_record: MagicMock,
        train_csv: Path,
        tmp_path: Path,
    ):
        mock_low.return_value = []
        args, parser = _parse_run_args(train_csv, tmp_path, "--auto-drop")

        _run(args, problem_type=None, parser=parser)

        assert mock_train.call_count == 1
        mock_record.assert_called_once()

    @patch("automl_model_training.train.record_experiment")
    @patch("automl_model_training.train._read_low_importance_features")
    @patch("automl_model_training.train.load_cv_train")
    def test_auto_drop_ignores_features_already_dropped(
        self,
        mock_train: MagicMock,
        mock_low: MagicMock,
        mock_record: MagicMock,
        train_csv: Path,
        tmp_path: Path,
    ):
        mock_low.return_value = ["feat_b"]
        args, parser = _parse_run_args(train_csv, tmp_path, "--auto-drop", "--drop", "feat_b")

        _run(args, problem_type=None, parser=parser)

        # feat_b was already in the drop list → no new drops → single training run
        assert mock_train.call_count == 1


class TestNoShuffleSplitFlag:
    @patch("automl_model_training.train.record_experiment")
    @patch("automl_model_training.train.load_cv_train")
    def test_flag_forwards_split_shuffle_false(
        self, mock_train: MagicMock, mock_record: MagicMock, train_csv: Path, tmp_path: Path
    ):
        args, parser = _parse_run_args(train_csv, tmp_path, "--no-shuffle-split")

        _run(args, problem_type=None, parser=parser)

        assert mock_train.call_args[1]["split_shuffle"] is False

    @patch("automl_model_training.train.record_experiment")
    @patch("automl_model_training.train.load_cv_train")
    def test_default_keeps_shuffled_split(
        self, mock_train: MagicMock, mock_record: MagicMock, train_csv: Path, tmp_path: Path
    ):
        args, parser = _parse_run_args(train_csv, tmp_path)

        _run(args, problem_type=None, parser=parser)

        assert mock_train.call_args[1]["split_shuffle"] is True

    @patch("automl_model_training.train.record_experiment")
    @patch("automl_model_training.train._read_low_importance_features")
    @patch("automl_model_training.train.load_cv_train")
    def test_auto_drop_retrain_preserves_split_shuffle(
        self,
        mock_train: MagicMock,
        mock_low: MagicMock,
        mock_record: MagicMock,
        train_csv: Path,
        tmp_path: Path,
    ):
        mock_low.return_value = ["feat_b"]
        args, parser = _parse_run_args(
            train_csv, tmp_path, "--auto-drop", "--no-shuffle-split"
        )

        _run(args, problem_type=None, parser=parser)

        assert mock_train.call_count == 2
        assert mock_train.call_args_list[1][1]["split_shuffle"] is False


class TestLoadCvTrainSplitShuffle:
    @patch("automl_model_training.train.train_and_evaluate")
    @patch("automl_model_training.train.load_and_prepare")
    def test_split_shuffle_forwarded_to_load_and_prepare(
        self, mock_load: MagicMock, mock_train: MagicMock, tmp_path: Path
    ):
        from automl_model_training.train import load_cv_train

        mock_load.return_value = (pd.DataFrame(), pd.DataFrame(), None, None, [])

        load_cv_train(
            csv_path="dummy.csv",
            label="target",
            output_dir=str(tmp_path),
            features_to_drop=[],
            test_size=0.2,
            seed=42,
            problem_type="regression",
            eval_metric=None,
            time_limit=None,
            preset="best",
            split_shuffle=False,
        )

        assert mock_load.call_args[1]["shuffle"] is False
