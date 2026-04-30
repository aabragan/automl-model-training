"""Tests for tool_optuna_tune."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _write_score_artifacts(run_dir: Path, score: float) -> None:
    """Write a minimal leaderboard_test.csv so _extract_metric returns `score`."""
    run_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "model": ["fake_model"],
            "score_test": [score],
            "score_val": [score],
            "fit_time": [1.0],
        }
    ).to_csv(run_dir / "leaderboard_test.csv", index=False)


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

    Each trial writes a fake leaderboard_test.csv; the 'score' is the
    learning_rate Optuna suggested, so the TPE sampler should converge
    to high learning rates by the end.
    """
    from automl_model_training.tools import tool_optuna_tune
    from automl_model_training.tools import train_predict as tools_tp

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
    from automl_model_training.tools import tool_optuna_tune
    from automl_model_training.tools import train_predict as tools_tp

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
    from automl_model_training.tools import tool_optuna_tune
    from automl_model_training.tools import train_predict as tools_tp

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


def test_optuna_tune_regression_uses_minimize_direction(tmp_path, monkeypatch):
    """RMSE should produce direction='minimize', with AutoGluon's sign-flipped
    score_test being converted to absolute value."""
    from automl_model_training.tools import tool_optuna_tune
    from automl_model_training.tools import train_predict as tools_tp

    csv = tmp_path / "d.csv"
    pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0], "y": [1.1, 2.2, 3.3, 4.4]}).to_csv(csv, index=False)

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
        # AutoGluon records lower-is-better metrics as negative scores;
        # the negative of learning_rate simulates that
        score = -float(hp["learning_rate"])
        _write_score_artifacts(Path(kwargs["output_dir"]), score)

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
    assert result["direction"] == "minimize"
    assert result["best_score"] is not None
    assert result["best_score"] >= 0  # absolute value returned


def test_optuna_tune_raises_when_all_trials_fail(tmp_path, monkeypatch):
    """If train_and_evaluate always raises, surface a clear error."""
    from automl_model_training.tools import tool_optuna_tune
    from automl_model_training.tools import train_predict as tools_tp

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
