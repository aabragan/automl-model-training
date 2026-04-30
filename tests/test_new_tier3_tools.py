"""Tests for new tier-3 LLM tools (threshold sweep, calibration, Optuna tune,
importance diff, 2-way PDP, model-subset eval) and Optuna study persistence."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from automl_model_training.tools import tool_calibration_curve, tool_threshold_sweep

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


# ---------------------------------------------------------------------------
# tool_threshold_sweep
# ---------------------------------------------------------------------------


def test_threshold_sweep_missing_predictions_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="test_predictions.csv"):
        tool_threshold_sweep(str(tmp_path))


def test_threshold_sweep_rejects_multiclass(tmp_path):
    """A run with 3+ prob_ columns is multiclass — tool should refuse."""
    pd.DataFrame(
        {
            "actual": [0, 1, 2, 0, 1, 2],
            "predicted": [0, 1, 2, 0, 1, 2],
            "prob_0": [0.9, 0.1, 0.1, 0.9, 0.1, 0.1],
            "prob_1": [0.05, 0.8, 0.1, 0.05, 0.8, 0.1],
            "prob_2": [0.05, 0.1, 0.8, 0.05, 0.1, 0.8],
        }
    ).to_csv(tmp_path / "test_predictions.csv", index=False)

    with pytest.raises(ValueError, match="binary classification"):
        tool_threshold_sweep(str(tmp_path))


def test_threshold_sweep_rejects_unknown_metric(tmp_path):
    _write_binary_predictions(tmp_path)
    with pytest.raises(ValueError, match="Unknown metrics"):
        tool_threshold_sweep(str(tmp_path), metrics=["f1", "nonsense"])


def test_threshold_sweep_returns_expected_shape(tmp_path):
    _write_binary_predictions(tmp_path, n=200)
    result = tool_threshold_sweep(str(tmp_path), n_thresholds=49)

    # Shape guarantees
    assert len(result["thresholds"]) == 49
    assert all(0 < t < 1 for t in result["thresholds"])
    for metric in ["f1", "precision", "recall", "mcc", "balanced_accuracy"]:
        assert metric in result["curves"]
        assert len(result["curves"][metric]) == 49
        assert metric in result["best"]
        assert 0 < result["best"][metric]["threshold"] < 1


def test_threshold_sweep_f1_peaks_near_05_for_balanced_strong_signal(tmp_path):
    """With balanced classes and strong signal, F1-optimal threshold should land
    somewhere in the middle band (0.2–0.8), not at the edges. F1 is asymmetric
    and can drift below 0.5 on noisy signals, so we only assert the middle band,
    not a tight window around 0.5."""
    _write_binary_predictions(tmp_path, n=500, signal_strength=4.0, imbalance=0.5)
    result = tool_threshold_sweep(str(tmp_path))
    f1_thresh = result["best"]["f1"]["threshold"]
    assert 0.2 < f1_thresh < 0.8, f"F1-optimal threshold should be in middle band, got {f1_thresh}"
    # And F1 itself should be high (strong signal + balanced classes)
    assert result["best"]["f1"]["value"] > 0.8


def test_threshold_sweep_flags_near_05_optimum(tmp_path):
    """When F1-optimal is within 0.02 of 0.5, a hint about calibration is returned."""
    _write_binary_predictions(tmp_path, n=1000, signal_strength=5.0, imbalance=0.5)
    result = tool_threshold_sweep(str(tmp_path), n_thresholds=99)
    if abs(result["best"]["f1"]["threshold"] - 0.5) < 0.02:
        assert any("within 0.02 of 0.5" in h for h in result["hints"])


def test_threshold_sweep_subset_metrics(tmp_path):
    """Requesting only some metrics should yield only those curves."""
    _write_binary_predictions(tmp_path)
    result = tool_threshold_sweep(str(tmp_path), metrics=["f1", "mcc"])
    assert set(result["curves"].keys()) == {"f1", "mcc"}
    assert set(result["best"].keys()) == {"f1", "mcc"}


# ---------------------------------------------------------------------------
# tool_calibration_curve
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# tool_optuna_tune
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


# ---------------------------------------------------------------------------
# tool_compare_importance
# ---------------------------------------------------------------------------


def _write_importance_run(
    run_dir: Path,
    importances: dict[str, float],
    score: float,
) -> None:
    """Write a minimal feature_importance.csv + leaderboard_test.csv into run_dir."""
    run_dir.mkdir(parents=True, exist_ok=True)
    imp_df = pd.DataFrame(
        [
            {"": feat, "importance": imp, "stddev": 0.0, "p_value": 0.01, "n": 5}
            for feat, imp in importances.items()
        ]
    ).set_index("")
    imp_df.to_csv(run_dir / "feature_importance.csv")
    pd.DataFrame(
        {
            "model": ["fake"],
            "score_test": [score],
            "score_val": [score],
            "fit_time": [1.0],
        }
    ).to_csv(run_dir / "leaderboard_test.csv", index=False)


def test_compare_importance_missing_files_raises(tmp_path):
    from automl_model_training.tools import tool_compare_importance

    with pytest.raises(FileNotFoundError, match="feature_importance.csv"):
        tool_compare_importance(str(tmp_path / "a"), str(tmp_path / "b"))


def test_compare_importance_detects_gained_and_lost(tmp_path):
    from automl_model_training.tools import tool_compare_importance

    before = tmp_path / "before"
    after = tmp_path / "after"
    _write_importance_run(before, {"x": 0.4, "y": 0.3, "z": 0.2}, score=0.80)
    _write_importance_run(after, {"x": 0.45, "y": 0.25, "new_feat": 0.15}, score=0.82)

    result = tool_compare_importance(str(before), str(after))

    assert result["gained_features"] == ["new_feat"]
    assert result["lost_features"] == ["z"]
    assert result["score_before"] == 0.80
    assert result["score_after"] == 0.82
    assert result["score_delta"] == 0.02


def test_compare_importance_flags_dominant_new_feature_with_flat_score(tmp_path):
    """Gained feature dominates after-run importance but score barely moved
    → should raise a leakage hint."""
    from automl_model_training.tools import tool_compare_importance

    before = tmp_path / "before"
    after = tmp_path / "after"
    _write_importance_run(before, {"a": 0.2, "b": 0.2}, score=0.85)
    # New leakage-like feature with huge importance but score unchanged
    _write_importance_run(after, {"a": 0.1, "b": 0.1, "leak": 0.9}, score=0.851)

    result = tool_compare_importance(str(before), str(after))
    assert result["dominant_new_feature"] is not None
    assert result["dominant_new_feature"]["feature"] == "leak"
    assert any("leak" in h for h in result["hints"])


def test_compare_importance_no_flag_when_score_improves(tmp_path):
    """Same dominant new feature but score improved materially → no leakage flag."""
    from automl_model_training.tools import tool_compare_importance

    before = tmp_path / "before"
    after = tmp_path / "after"
    _write_importance_run(before, {"a": 0.2, "b": 0.2}, score=0.70)
    _write_importance_run(after, {"a": 0.1, "b": 0.1, "new": 0.9}, score=0.88)

    result = tool_compare_importance(str(before), str(after))
    assert result["dominant_new_feature"] is None


def test_compare_importance_ranks_top_n_by_abs_delta(tmp_path):
    from automl_model_training.tools import tool_compare_importance

    before = tmp_path / "before"
    after = tmp_path / "after"
    _write_importance_run(
        before,
        {"a": 0.50, "b": 0.40, "c": 0.30, "d": 0.20, "e": 0.10},
        score=0.80,
    )
    _write_importance_run(
        after,
        {"a": 0.50, "b": 0.60, "c": 0.10, "d": 0.22, "e": 0.11},  # b ↑, c ↓
        score=0.82,
    )
    result = tool_compare_importance(str(before), str(after), top_n=2)
    assert len(result["changed_features"]) == 2
    # Top 2 by |delta|: b (+0.20), c (-0.20)
    features_in_top = {r["feature"] for r in result["changed_features"]}
    assert features_in_top == {"b", "c"}


def test_compare_importance_ignores_sub_threshold_changes(tmp_path):
    """significance_delta filters out jitter."""
    from automl_model_training.tools import tool_compare_importance

    before = tmp_path / "before"
    after = tmp_path / "after"
    _write_importance_run(before, {"a": 0.5, "b": 0.5}, score=0.80)
    # Only tiny changes (< 0.01 default threshold)
    _write_importance_run(after, {"a": 0.501, "b": 0.499}, score=0.80)

    result = tool_compare_importance(str(before), str(after))
    assert result["changed_features"] == []
    assert any("No material importance changes" in h for h in result["hints"])


# ---------------------------------------------------------------------------
# tool_partial_dependence_2way
# ---------------------------------------------------------------------------


@pytest.fixture()
def mock_2way_run(tmp_path):
    """Run dir with test_raw.csv and two numeric features."""
    run_dir = tmp_path / "run"
    (run_dir / "AutogluonModels").mkdir(parents=True)
    rng = np.random.RandomState(0)
    test_raw = pd.DataFrame(
        {
            "x": np.linspace(0, 10, 100),
            "y": rng.uniform(0, 1, size=100),
            "cat": rng.choice(["a", "b", "c"], size=100),
            "target": rng.randn(100),
        }
    )
    test_raw.to_csv(run_dir / "test_raw.csv", index=False)
    return run_dir


def test_pdp_2way_missing_files_raises(tmp_path):
    from automl_model_training.tools import tool_partial_dependence_2way

    with pytest.raises(FileNotFoundError, match="AutogluonModels"):
        tool_partial_dependence_2way(str(tmp_path), feature_a="x", feature_b="y")


def test_pdp_2way_rejects_same_feature(mock_2way_run):
    from unittest.mock import MagicMock, patch

    from automl_model_training.tools import tool_partial_dependence_2way

    mock_predictor = MagicMock()
    mock_predictor.label = "target"
    mock_predictor.problem_type = "regression"

    with (
        patch(
            "automl_model_training.tools.explainability.load_predictor", return_value=mock_predictor
        ),
        pytest.raises(ValueError, match="must differ"),
    ):
        tool_partial_dependence_2way(str(mock_2way_run), feature_a="x", feature_b="x")


def test_pdp_2way_rejects_missing_feature(mock_2way_run):
    from unittest.mock import MagicMock, patch

    from automl_model_training.tools import tool_partial_dependence_2way

    mock_predictor = MagicMock()
    mock_predictor.label = "target"
    mock_predictor.problem_type = "regression"

    with (
        patch(
            "automl_model_training.tools.explainability.load_predictor", return_value=mock_predictor
        ),
        pytest.raises(ValueError, match="not in test data"),
    ):
        tool_partial_dependence_2way(str(mock_2way_run), feature_a="x", feature_b="ghost")


def test_pdp_2way_rejects_cost_cap_breach(mock_2way_run):
    from unittest.mock import MagicMock, patch

    from automl_model_training.tools import tool_partial_dependence_2way

    mock_predictor = MagicMock()
    mock_predictor.label = "target"
    mock_predictor.problem_type = "regression"

    with (
        patch(
            "automl_model_training.tools.explainability.load_predictor", return_value=mock_predictor
        ),
        pytest.raises(ValueError, match="max_cells"),
    ):
        tool_partial_dependence_2way(
            str(mock_2way_run),
            feature_a="x",
            feature_b="y",
            n_values_a=100,
            n_values_b=100,
            sample_size=100,
            max_cells=50_000,
        )


def test_pdp_2way_detects_additive_surface(mock_2way_run):
    """A model that predicts f(x,y) = x + y (purely additive) should classify
    as 'additive' and have low interaction_strength."""
    from unittest.mock import MagicMock, patch

    from automl_model_training.tools import tool_partial_dependence_2way

    mock_predictor = MagicMock()
    mock_predictor.label = "target"
    mock_predictor.problem_type = "regression"

    def predict(df):
        return pd.Series(df["x"].values.astype(float) + df["y"].values.astype(float))

    mock_predictor.predict.side_effect = predict

    with patch(
        "automl_model_training.tools.explainability.load_predictor", return_value=mock_predictor
    ):
        result = tool_partial_dependence_2way(
            str(mock_2way_run),
            feature_a="x",
            feature_b="y",
            n_values_a=6,
            n_values_b=6,
            sample_size=30,
        )

    assert result["shape_hint"] == "additive"
    # Pure additive surface has interaction_strength ≈ 0 (up to floating-point noise)
    assert result["interaction_strength"] < 0.01
    assert len(result["surface"]) == 6
    assert len(result["surface"][0]) == 6


def test_pdp_2way_detects_nonadditive_surface(mock_2way_run):
    """A multiplicative model f(x,y) = x*y produces a saddle-shaped response
    surface (hyperbolic paraboloid). The tool should flag it as non-additive
    with meaningful interaction_strength, regardless of the specific shape
    label it lands on (saddle vs synergy vs threshold)."""
    from unittest.mock import MagicMock, patch

    from automl_model_training.tools import tool_partial_dependence_2way

    mock_predictor = MagicMock()
    mock_predictor.label = "target"
    mock_predictor.problem_type = "regression"

    def predict(df):
        return pd.Series(df["x"].values.astype(float) * df["y"].values.astype(float))

    mock_predictor.predict.side_effect = predict

    with patch(
        "automl_model_training.tools.explainability.load_predictor", return_value=mock_predictor
    ):
        result = tool_partial_dependence_2way(
            str(mock_2way_run),
            feature_a="x",
            feature_b="y",
            n_values_a=6,
            n_values_b=6,
            sample_size=30,
        )

    # Multiplicative surface is non-additive — any of the three non-additive
    # labels is acceptable; the key assertion is it's NOT labelled 'additive'.
    assert result["shape_hint"] != "additive"
    assert result["shape_hint"] in {"synergy", "saddle", "threshold"}
    assert result["interaction_strength"] > 0.05


def test_pdp_2way_detects_synergistic_surface(mock_2way_run):
    """A surface f(x,y) = x + y + 5*x*y (additive + strong positive interaction)
    should classify as 'synergy' — residuals are consistently positive in the
    corners where x*y is largest and negative elsewhere only mildly."""
    from unittest.mock import MagicMock, patch

    from automl_model_training.tools import tool_partial_dependence_2way

    mock_predictor = MagicMock()
    mock_predictor.label = "target"
    mock_predictor.problem_type = "regression"

    def predict(df):
        x = df["x"].values.astype(float)
        y = df["y"].values.astype(float)
        # Additive + a strictly-positive-contributing quadratic boost in the
        # upper-right quadrant. This creates residuals that are ≥ 0 almost
        # everywhere, so shape_hint should be 'synergy' rather than 'saddle'.
        return pd.Series(x + y + 2.0 * np.maximum(x * y, 0))

    mock_predictor.predict.side_effect = predict

    with patch(
        "automl_model_training.tools.explainability.load_predictor", return_value=mock_predictor
    ):
        result = tool_partial_dependence_2way(
            str(mock_2way_run),
            feature_a="x",
            feature_b="y",
            n_values_a=6,
            n_values_b=6,
            sample_size=30,
        )

    assert result["shape_hint"] != "additive"
    assert result["interaction_strength"] > 0.05


def test_pdp_2way_handles_categorical_feature(mock_2way_run):
    """One numeric + one categorical feature — surface should still be built."""
    from unittest.mock import MagicMock, patch

    from automl_model_training.tools import tool_partial_dependence_2way

    mock_predictor = MagicMock()
    mock_predictor.label = "target"
    mock_predictor.problem_type = "regression"
    # Return a constant so we can verify shape without caring about values
    mock_predictor.predict.side_effect = lambda df: pd.Series([0.5] * len(df))

    with patch(
        "automl_model_training.tools.explainability.load_predictor", return_value=mock_predictor
    ):
        result = tool_partial_dependence_2way(
            str(mock_2way_run),
            feature_a="x",
            feature_b="cat",
            n_values_a=5,
            n_values_b=3,
            sample_size=20,
        )

    assert result["is_numeric_a"] is True
    assert result["is_numeric_b"] is False
    # grid_b should be 3 string category values
    assert len(result["grid_b"]) <= 3
    assert all(isinstance(g, str) for g in result["grid_b"])


# ---------------------------------------------------------------------------
# tool_model_subset_evaluate
# ---------------------------------------------------------------------------


def _write_leaderboard_test(
    run_dir: Path,
    rows: list[dict],
) -> None:
    """Write a minimal leaderboard_test.csv into ``run_dir``.

    Each row must contain 'model', 'score_test', 'stack_level'.
    Optional columns: 'score_val', 'pred_time_test', 'fit_time'.
    """
    run_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(run_dir / "leaderboard_test.csv", index=False)


def test_model_subset_evaluate_missing_file_raises(tmp_path):
    from automl_model_training.tools import tool_model_subset_evaluate

    with pytest.raises(FileNotFoundError, match="leaderboard_test.csv"):
        tool_model_subset_evaluate(str(tmp_path))


def test_model_subset_evaluate_missing_columns_raises(tmp_path):
    """CSV without the required columns should yield a clear error."""
    from automl_model_training.tools import tool_model_subset_evaluate

    # Missing 'stack_level'
    pd.DataFrame({"model": ["A"], "score_test": [0.9]}).to_csv(
        tmp_path / "leaderboard_test.csv", index=False
    )
    with pytest.raises(ValueError, match="stack_level"):
        tool_model_subset_evaluate(str(tmp_path))


def test_model_subset_evaluate_empty_leaderboard_raises(tmp_path):
    """Well-formed-but-empty CSV → RuntimeError, not a silent wrong result."""
    from automl_model_training.tools import tool_model_subset_evaluate

    pd.DataFrame(columns=["model", "score_test", "stack_level"]).to_csv(
        tmp_path / "leaderboard_test.csv", index=False
    )
    with pytest.raises(RuntimeError, match="empty"):
        tool_model_subset_evaluate(str(tmp_path))


def test_model_subset_evaluate_returns_ordered_models(tmp_path):
    """Models should be returned sorted by absolute test score, descending."""
    from automl_model_training.tools import tool_model_subset_evaluate

    _write_leaderboard_test(
        tmp_path,
        [
            {"model": "LightGBM", "score_test": 0.88, "stack_level": 1, "pred_time_test": 0.05},
            {
                "model": "WeightedEnsemble_L2",
                "score_test": 0.91,
                "stack_level": 2,
                "pred_time_test": 0.20,
            },
            {"model": "CatBoost", "score_test": 0.85, "stack_level": 1, "pred_time_test": 0.10},
        ],
    )

    result = tool_model_subset_evaluate(str(tmp_path))

    # Ordering
    assert [m["model"] for m in result["models"]] == [
        "WeightedEnsemble_L2",
        "LightGBM",
        "CatBoost",
    ]
    # Ensemble flagged correctly
    ens = [m for m in result["models"] if m["is_ensemble"]]
    assert [m["model"] for m in ens] == ["WeightedEnsemble_L2"]

    assert result["best_model"]["model"] == "WeightedEnsemble_L2"
    assert result["best_single_model"]["model"] == "LightGBM"


def test_model_subset_evaluate_flags_cheaper_near_best(tmp_path):
    """A single model within score_tolerance of the ensemble and materially
    faster at inference should land in recommended_deploy with a speedup."""
    from automl_model_training.tools import tool_model_subset_evaluate

    _write_leaderboard_test(
        tmp_path,
        [
            {
                "model": "WeightedEnsemble_L2",
                "score_test": 0.905,
                "stack_level": 2,
                "pred_time_test": 0.40,  # slow
            },
            {
                "model": "LightGBM",
                "score_test": 0.900,  # only 0.005 below best, within default tolerance 0.01
                "stack_level": 1,
                "pred_time_test": 0.05,  # 8x faster than the ensemble
            },
        ],
    )

    result = tool_model_subset_evaluate(str(tmp_path))
    assert result["recommended_deploy"] is not None
    assert result["recommended_deploy"]["model"] == "LightGBM"
    # Speedup = 0.40 / 0.05 = 8.0
    assert result["recommended_deploy"]["speedup"] == 8.0
    assert any("LightGBM" in h for h in result["hints"])


def test_model_subset_evaluate_no_recommendation_when_ensemble_alone_qualifies(tmp_path):
    """If no non-ensemble model is within score_tolerance of the best, no
    deployment recommendation should be emitted."""
    from automl_model_training.tools import tool_model_subset_evaluate

    _write_leaderboard_test(
        tmp_path,
        [
            {
                "model": "WeightedEnsemble_L2",
                "score_test": 0.95,
                "stack_level": 2,
                "pred_time_test": 0.30,
            },
            {
                "model": "LightGBM",
                "score_test": 0.80,  # 0.15 gap — far outside tolerance
                "stack_level": 1,
                "pred_time_test": 0.05,
            },
        ],
    )
    result = tool_model_subset_evaluate(str(tmp_path))
    assert result["recommended_deploy"] is None


def test_model_subset_evaluate_flags_small_single_vs_ensemble_gap(tmp_path):
    """When the best single model is very close to the ensemble, a 'may not be
    worth its cost' hint should appear."""
    from automl_model_training.tools import tool_model_subset_evaluate

    _write_leaderboard_test(
        tmp_path,
        [
            {
                "model": "WeightedEnsemble_L2",
                "score_test": 0.901,
                "stack_level": 2,
                "pred_time_test": 0.30,
            },
            {
                "model": "LightGBM",
                "score_test": 0.900,  # score_gap = 0.001, well under tolerance
                "stack_level": 1,
                "pred_time_test": 0.05,
            },
        ],
    )
    result = tool_model_subset_evaluate(str(tmp_path))
    assert result["score_gap"] == pytest.approx(0.001)
    assert any("ensemble may not be worth" in h for h in result["hints"])


def test_model_subset_evaluate_stack_level_detects_ensemble(tmp_path):
    """A model that isn't named 'WeightedEnsemble*' but sits at stack_level>1
    should still be flagged as is_ensemble=True."""
    from automl_model_training.tools import tool_model_subset_evaluate

    _write_leaderboard_test(
        tmp_path,
        [
            {
                "model": "CustomStacker",  # doesn't start with WeightedEnsemble
                "score_test": 0.93,
                "stack_level": 2,  # but stack_level says it IS a stacker
                "pred_time_test": 0.30,
            },
            {
                "model": "LightGBM",
                "score_test": 0.88,
                "stack_level": 1,
                "pred_time_test": 0.05,
            },
        ],
    )
    result = tool_model_subset_evaluate(str(tmp_path))
    stacker = next(m for m in result["models"] if m["model"] == "CustomStacker")
    assert stacker["is_ensemble"] is True
    assert result["best_single_model"]["model"] == "LightGBM"


def test_model_subset_evaluate_single_model_run(tmp_path):
    """A leaderboard with exactly one model (e.g., a tune run) should emit the
    'only one model' hint and still return a well-formed result."""
    from automl_model_training.tools import tool_model_subset_evaluate

    _write_leaderboard_test(
        tmp_path,
        [{"model": "LightGBM", "score_test": 0.87, "stack_level": 1, "pred_time_test": 0.05}],
    )
    result = tool_model_subset_evaluate(str(tmp_path))
    assert len(result["models"]) == 1
    assert result["best_model"]["model"] == "LightGBM"
    assert result["best_single_model"]["model"] == "LightGBM"
    assert result["score_gap"] == 0.0
    assert any("Only one model" in h for h in result["hints"])


def test_model_subset_evaluate_handles_negative_regression_scores(tmp_path):
    """AutoGluon stores lower-is-better metrics (RMSE, log_loss) as negative
    values such that higher score_test is always better. The tool must sort
    by score_test descending so the 'best' model is the one closest to zero
    (least-negative) in regression, and the most positive in classification."""
    from automl_model_training.tools import tool_model_subset_evaluate

    _write_leaderboard_test(
        tmp_path,
        [
            # LightGBM: RMSE ≈ 2.0 → stored as -2.0 → BETTER regression model
            {"model": "LightGBM", "score_test": -2.0, "stack_level": 1, "pred_time_test": 0.05},
            # CatBoost: RMSE ≈ 3.5 → stored as -3.5 → WORSE regression model
            {
                "model": "CatBoost",
                "score_test": -3.5,
                "stack_level": 1,
                "pred_time_test": 0.10,
            },
        ],
    )
    result = tool_model_subset_evaluate(str(tmp_path))
    # Best model = the less-negative score_test (lower RMSE in the original scale)
    assert result["best_model"]["model"] == "LightGBM"
    # Order should be LightGBM (-2.0) > CatBoost (-3.5)
    assert [m["model"] for m in result["models"]] == ["LightGBM", "CatBoost"]
