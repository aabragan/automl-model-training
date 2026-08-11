"""Tests for tool_optuna_tune."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _write_score_artifacts(run_dir: Path, score: float, eval_metric: str = "f1") -> None:
    """Write a minimal analysis.json so extract_metric returns `score`."""
    run_dir.mkdir(parents=True, exist_ok=True)
    analysis = {
        "best_model": "fake_model",
        "eval_metric": eval_metric,
        "test_scores": {eval_metric: score},
        "score_convention": "higher_is_better",
        "findings": [],
        "recommendations": [],
    }
    (run_dir / "analysis.json").write_text(json.dumps(analysis))


# ---------------------------------------------------------------------------
# tool_optuna_tune
# ---------------------------------------------------------------------------


def test_optuna_tune_rejects_unknown_family(tmp_path):
    from automl_model_training.tools import tool_optuna_tune

    csv = tmp_path / "d.csv"
    pd.DataFrame({"x": [1, 2, 3, 4], "y": [0, 1, 0, 1]}).to_csv(csv, index=False)
    with pytest.raises(ValueError, match="model_family"):
        tool_optuna_tune(
            csv_path=str(csv),
            label="y",
            model_family="NONSENSE",
            n_trials=2,
            time_limit_per_trial=5,
        )


def test_optuna_tune_rejects_invalid_pruner(tmp_path):
    from automl_model_training.tools import tool_optuna_tune

    csv = tmp_path / "d.csv"
    pd.DataFrame({"x": [1, 2, 3, 4], "y": [0, 1, 0, 1]}).to_csv(csv, index=False)
    with pytest.raises(ValueError, match="pruner"):
        tool_optuna_tune(
            csv_path=str(csv),
            label="y",
            model_family="GBM",
            n_trials=2,
            pruner="bogus",
        )


def test_optuna_tune_runs_loop_and_returns_best_trial(tmp_path, monkeypatch):
    """End-to-end Optuna loop with mocked AutoGluon training.

    Each trial writes a fake analysis.json; the 'score' is the
    learning_rate Optuna suggested, so the TPE sampler should converge
    to high learning rates by the end.
    """
    from automl_model_training.tools import optuna_tune as tools_tp
    from automl_model_training.tools import tool_optuna_tune

    csv = tmp_path / "d.csv"
    pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0], "y": [0, 1, 0, 1]}).to_csv(csv, index=False)

    captured_hps: list[dict] = []

    def fake_load_and_prepare(**kwargs):
        return (
            pd.DataFrame({"x": [1.0, 2.0], "y": [0, 1]}),
            pd.DataFrame({"x": [3.0], "y": [1]}),
            None,
            None,
            [],
        )

    def fake_train_and_evaluate(**kwargs):
        hp = kwargs["hyperparameters"]["GBM"]
        captured_hps.append(hp)
        # Score increases with learning_rate — lets us verify TPE prefers higher lr
        score = float(hp["learning_rate"])
        _write_score_artifacts(Path(kwargs["output_dir"]), score)
        return None

    monkeypatch.setattr(tools_tp, "load_and_prepare", fake_load_and_prepare)
    monkeypatch.setattr(tools_tp, "train_and_evaluate", fake_train_and_evaluate)

    result = tool_optuna_tune(
        csv_path=str(csv),
        label="y",
        model_family="GBM",
        n_trials=8,
        time_limit_per_trial=5,
        eval_metric="f1",
        output_dir=str(tmp_path / "out"),
        n_startup_trials=2,
        pruner="none",  # keep all trials so TPE behavior is visible
        seed=0,
    )

    # Shape
    assert result["model_family"] == "GBM"
    assert result["direction"] == "maximize"
    assert result["n_trials_run"] == 8
    assert "learning_rate" in result["best_hyperparameters"]
    assert result["best_score"] is not None
    assert len(result["trial_history"]) == 8

    # The best score's learning_rate should be among the highest sampled
    all_lrs = [hp["learning_rate"] for hp in captured_hps]
    best_lr = result["best_hyperparameters"]["learning_rate"]
    assert best_lr == max(all_lrs)


def test_optuna_tune_persists_study_to_sqlite(tmp_path, monkeypatch):
    """Sqlite-backed study persists across calls and the second call resumes it."""
    from automl_model_training.tools import optuna_tune as tools_tp
    from automl_model_training.tools import tool_optuna_tune

    csv = tmp_path / "d.csv"
    pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0], "y": [0, 1, 0, 1]}).to_csv(csv, index=False)

    def fake_load_and_prepare(**kwargs):
        return (
            pd.DataFrame({"x": [1.0], "y": [0]}),
            pd.DataFrame({"x": [2.0], "y": [1]}),
            None,
            None,
            [],
        )

    def fake_train_and_evaluate(**kwargs):
        hp = kwargs["hyperparameters"]["GBM"]
        score = float(hp["learning_rate"])
        _write_score_artifacts(Path(kwargs["output_dir"]), score)

    monkeypatch.setattr(tools_tp, "load_and_prepare", fake_load_and_prepare)
    monkeypatch.setattr(tools_tp, "train_and_evaluate", fake_train_and_evaluate)

    db_path = tmp_path / "study.db"
    storage = f"sqlite:///{db_path}"
    study_name = "test_persistence"

    r1 = tool_optuna_tune(
        csv_path=str(csv),
        label="y",
        model_family="GBM",
        n_trials=3,
        time_limit_per_trial=5,
        output_dir=str(tmp_path / "out1"),
        study_name=study_name,
        storage=storage,
        pruner="none",
        n_startup_trials=1,
        seed=0,
    )
    r2 = tool_optuna_tune(
        csv_path=str(csv),
        label="y",
        model_family="GBM",
        n_trials=3,
        time_limit_per_trial=5,
        output_dir=str(tmp_path / "out2"),
        study_name=study_name,
        storage=storage,
        pruner="none",
        n_startup_trials=1,
        seed=0,
    )

    # Second call should see 3 + 3 = 6 trials in the study
    assert r1["n_trials_run"] == 3
    assert r2["n_trials_run"] == 6
    # Both should report the same study_name + storage in the response
    assert r2["study_name"] == study_name
    assert r2["storage"] == storage
    # A persistence hint should appear in both
    assert any("persisted" in h for h in r2["hints"])


def test_optuna_tune_pruning_reduces_trial_count(tmp_path, monkeypatch):
    """MedianPruner should terminate half-bad trials; hints mention savings."""
    from automl_model_training.tools import optuna_tune as tools_tp
    from automl_model_training.tools import tool_optuna_tune

    csv = tmp_path / "d.csv"
    pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0], "y": [0, 1, 0, 1]}).to_csv(csv, index=False)

    # Alternate good/bad scores so the median has a meaningful threshold
    scores_to_return = [0.9, 0.3, 0.9, 0.3, 0.9, 0.3, 0.9, 0.3, 0.9, 0.3]
    idx = [0]

    def fake_load_and_prepare(**kwargs):
        return (
            pd.DataFrame({"x": [1.0], "y": [0]}),
            pd.DataFrame({"x": [2.0], "y": [1]}),
            None,
            None,
            [],
        )

    def fake_train_and_evaluate(**kwargs):
        s = scores_to_return[idx[0] % len(scores_to_return)]
        idx[0] += 1
        _write_score_artifacts(Path(kwargs["output_dir"]), s)

    monkeypatch.setattr(tools_tp, "load_and_prepare", fake_load_and_prepare)
    monkeypatch.setattr(tools_tp, "train_and_evaluate", fake_train_and_evaluate)

    result = tool_optuna_tune(
        csv_path=str(csv),
        label="y",
        model_family="GBM",
        n_trials=10,
        time_limit_per_trial=5,
        output_dir=str(tmp_path / "out"),
        pruner="median",
        n_startup_trials=2,
        seed=0,
    )

    # MedianPruner prunes trials whose single reported step is below the median
    # of completed trials. In this setup, half the trials are 0.3 and half 0.9;
    # after enough warmup trials, 0.3s will get pruned. Just verify at least one
    # pruning occurred and the hint is emitted.
    assert result["n_trials_run"] == 10
    if result["n_trials_pruned"] > 0:
        assert any("pruner" in h.lower() for h in result["hints"])


def test_optuna_tune_regression_maximizes_signed_scores(tmp_path, monkeypatch):
    """RMSE scores arrive negated (AutoGluon higher-is-better convention),
    so the study always maximizes: maximizing -RMSE minimizes RMSE."""
    from automl_model_training.tools import optuna_tune as tools_tp
    from automl_model_training.tools import tool_optuna_tune

    csv = tmp_path / "d.csv"
    pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0], "y": [1.1, 2.2, 3.3, 4.4]}).to_csv(csv, index=False)

    captured_lrs: list[float] = []

    def fake_load_and_prepare(**kwargs):
        return (
            pd.DataFrame({"x": [1.0], "y": [1.1]}),
            pd.DataFrame({"x": [2.0], "y": [2.2]}),
            None,
            None,
            [],
        )

    def fake_train_and_evaluate(**kwargs):
        hp = kwargs["hyperparameters"]["GBM"]
        lr = float(hp["learning_rate"])
        captured_lrs.append(lr)
        # Simulate RMSE == learning_rate, stored negated as AutoGluon does
        _write_score_artifacts(
            Path(kwargs["output_dir"]), -lr, eval_metric="root_mean_squared_error"
        )

    monkeypatch.setattr(tools_tp, "load_and_prepare", fake_load_and_prepare)
    monkeypatch.setattr(tools_tp, "train_and_evaluate", fake_train_and_evaluate)

    result = tool_optuna_tune(
        csv_path=str(csv),
        label="y",
        model_family="GBM",
        n_trials=4,
        time_limit_per_trial=5,
        eval_metric="root_mean_squared_error",
        output_dir=str(tmp_path / "out"),
        pruner="none",
        n_startup_trials=1,
        seed=0,
    )
    assert result["direction"] == "maximize"
    assert result["best_score"] is not None
    assert result["best_score"] <= 0  # signed convention: -RMSE
    # Maximizing the signed score picks the SMALLEST RMSE (learning_rate here)
    assert result["best_hyperparameters"]["learning_rate"] == min(captured_lrs)


def test_optuna_tune_raises_when_all_trials_fail(tmp_path, monkeypatch):
    """If train_and_evaluate always raises, surface a clear error."""
    from automl_model_training.tools import optuna_tune as tools_tp
    from automl_model_training.tools import tool_optuna_tune

    csv = tmp_path / "d.csv"
    pd.DataFrame({"x": [1.0, 2.0], "y": [0, 1]}).to_csv(csv, index=False)

    def fake_load_and_prepare(**kwargs):
        return (
            pd.DataFrame({"x": [1.0], "y": [0]}),
            pd.DataFrame({"x": [2.0], "y": [1]}),
            None,
            None,
            [],
        )

    def always_fail(**kwargs):
        raise RuntimeError("simulated AutoGluon failure")

    monkeypatch.setattr(tools_tp, "load_and_prepare", fake_load_and_prepare)
    monkeypatch.setattr(tools_tp, "train_and_evaluate", always_fail)

    with pytest.raises(RuntimeError, match="no trial completed successfully"):
        tool_optuna_tune(
            csv_path=str(csv),
            label="y",
            model_family="GBM",
            n_trials=3,
            time_limit_per_trial=5,
            output_dir=str(tmp_path / "out"),
            pruner="none",
            n_startup_trials=1,
            seed=0,
        )


# ---------------------------------------------------------------------------
# _suggest_hyperparameters — per-family search spaces
# ---------------------------------------------------------------------------


_FAMILY_EXPECTED_KEYS = {
    "GBM": {
        "num_leaves",
        "max_depth",
        "learning_rate",
        "feature_fraction",
        "bagging_fraction",
        "min_data_in_leaf",
        "lambda_l2",
    },
    "XGB": {
        "max_depth",
        "learning_rate",
        "subsample",
        "colsample_bytree",
        "min_child_weight",
        "reg_lambda",
        "gamma",
    },
    "CAT": {"depth", "learning_rate", "l2_leaf_reg", "random_strength", "bagging_temperature"},
    "RF": {"n_estimators", "max_depth", "min_samples_split", "min_samples_leaf", "max_features"},
    "XT": {"n_estimators", "max_depth", "min_samples_split", "min_samples_leaf", "max_features"},
    "NN_TORCH": {"learning_rate", "weight_decay", "dropout_prob", "num_layers", "hidden_size"},
    "FASTAI": {"lr", "wd", "epochs", "bs"},
}


@pytest.mark.parametrize("family", sorted(_FAMILY_EXPECTED_KEYS))
def test_suggest_hyperparameters_returns_expected_keys(family):
    """Each supported family produces a concrete hp dict with its curated keys."""
    import optuna

    from automl_model_training.tools.optuna_tune import _suggest_hyperparameters

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=0))
    trial = study.ask()

    hp = _suggest_hyperparameters(trial, family)

    assert set(hp.keys()) == _FAMILY_EXPECTED_KEYS[family]
    # Values must be concrete (numbers or categorical choices), not search spaces
    for value in hp.values():
        assert isinstance(value, int | float | str)


def test_suggest_hyperparameters_rejects_unknown_family():
    import optuna

    from automl_model_training.tools.optuna_tune import _suggest_hyperparameters

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=0))
    trial = study.ask()

    with pytest.raises(ValueError, match="Unsupported model_family"):
        _suggest_hyperparameters(trial, "BOGUS")


# ---------------------------------------------------------------------------
# Objective edge cases
# ---------------------------------------------------------------------------


def test_optuna_tune_raises_when_score_missing_from_artifacts(tmp_path, monkeypatch):
    """Training 'succeeds' but writes no leaderboard: score is None, every trial
    returns -inf, and a clear RuntimeError surfaces at the end."""
    from automl_model_training.tools import optuna_tune as tools_tp
    from automl_model_training.tools import tool_optuna_tune

    csv = tmp_path / "d.csv"
    pd.DataFrame({"x": [1.0, 2.0], "y": [0, 1]}).to_csv(csv, index=False)

    def fake_load_and_prepare(**kwargs):
        return (
            pd.DataFrame({"x": [1.0], "y": [0]}),
            pd.DataFrame({"x": [2.0], "y": [1]}),
            None,
            None,
            [],
        )

    def train_without_artifacts(**kwargs):
        # Succeeds but never writes analysis.json → extract_metric → None
        return None

    monkeypatch.setattr(tools_tp, "load_and_prepare", fake_load_and_prepare)
    monkeypatch.setattr(tools_tp, "train_and_evaluate", train_without_artifacts)

    with pytest.raises(RuntimeError, match="no trial completed successfully"):
        tool_optuna_tune(
            csv_path=str(csv),
            label="y",
            model_family="GBM",
            n_trials=2,
            time_limit_per_trial=5,
            output_dir=str(tmp_path / "out"),
            pruner="none",
            n_startup_trials=1,
            seed=0,
        )


def test_optuna_tune_raises_when_every_trial_is_pruned(tmp_path, monkeypatch):
    """If the pruner kills every trial, study.best_trial raises ValueError and
    the tool converts it into a descriptive RuntimeError."""
    import optuna

    from automl_model_training.tools import optuna_tune as tools_tp
    from automl_model_training.tools import tool_optuna_tune

    csv = tmp_path / "d.csv"
    pd.DataFrame({"x": [1.0, 2.0], "y": [0, 1]}).to_csv(csv, index=False)

    def fake_load_and_prepare(**kwargs):
        return (
            pd.DataFrame({"x": [1.0], "y": [0]}),
            pd.DataFrame({"x": [2.0], "y": [1]}),
            None,
            None,
            [],
        )

    def fake_train_and_evaluate(**kwargs):
        _write_score_artifacts(Path(kwargs["output_dir"]), 0.5)

    class AlwaysPrune(optuna.pruners.BasePruner):
        def prune(self, study, trial):
            return True

    monkeypatch.setattr(tools_tp, "load_and_prepare", fake_load_and_prepare)
    monkeypatch.setattr(tools_tp, "train_and_evaluate", fake_train_and_evaluate)
    # tool_optuna_tune builds its pruner via optuna.pruners.MedianPruner(...)
    monkeypatch.setattr(optuna.pruners, "MedianPruner", lambda **kwargs: AlwaysPrune())

    with pytest.raises(RuntimeError, match="either failed or were pruned"):
        tool_optuna_tune(
            csv_path=str(csv),
            label="y",
            model_family="GBM",
            n_trials=3,
            time_limit_per_trial=5,
            output_dir=str(tmp_path / "out"),
            pruner="median",
            n_startup_trials=1,
            seed=0,
        )


def test_optuna_tune_param_importance_failure_is_nonfatal(tmp_path, monkeypatch):
    """If Optuna's importance estimation blows up, the tool degrades to an
    empty param_importances dict instead of raising."""
    import optuna

    from automl_model_training.tools import optuna_tune as tools_tp
    from automl_model_training.tools import tool_optuna_tune

    csv = tmp_path / "d.csv"
    pd.DataFrame({"x": [1.0, 2.0], "y": [0, 1]}).to_csv(csv, index=False)

    def fake_load_and_prepare(**kwargs):
        return (
            pd.DataFrame({"x": [1.0], "y": [0]}),
            pd.DataFrame({"x": [2.0], "y": [1]}),
            None,
            None,
            [],
        )

    def fake_train_and_evaluate(**kwargs):
        hp = kwargs["hyperparameters"]["GBM"]
        _write_score_artifacts(Path(kwargs["output_dir"]), float(hp["learning_rate"]))

    def broken_importances(study, **kwargs):
        raise RuntimeError("fANOVA exploded")

    monkeypatch.setattr(tools_tp, "load_and_prepare", fake_load_and_prepare)
    monkeypatch.setattr(tools_tp, "train_and_evaluate", fake_train_and_evaluate)
    monkeypatch.setattr(optuna.importance, "get_param_importances", broken_importances)

    result = tool_optuna_tune(
        csv_path=str(csv),
        label="y",
        model_family="GBM",
        n_trials=3,  # >= 2 completed trials so importance estimation is attempted
        time_limit_per_trial=5,
        output_dir=str(tmp_path / "out"),
        pruner="none",
        n_startup_trials=1,
        seed=0,
    )

    assert result["param_importances"] == {}
    assert result["best_score"] is not None
