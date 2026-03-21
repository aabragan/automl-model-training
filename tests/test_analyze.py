"""Tests for evaluate.analyze — post-training analysis and recommendations."""

import json
from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd

from automl_model_training.evaluate.analyze import _model_family, analyze_and_recommend

# ---- Helpers ----


def _make_predictor(
    problem_type: str = "binary",
    label: str = "target",
    best_model: str = "LightGBM",
    features: list[str] | None = None,
) -> MagicMock:
    pred = MagicMock()
    pred.label = label
    pred.problem_type = problem_type
    pred.eval_metric = "f1"
    pred.model_best = best_model
    pred.features.return_value = features or ["feat_a", "feat_b"]
    return pred


def _make_leaderboards(
    best_model: str = "LightGBM",
    val_score: float = 0.90,
    test_score: float = 0.88,
    n_models: int = 8,
):
    models = [best_model] + [f"Model_{i}" for i in range(1, n_models)]
    lb = pd.DataFrame(
        {
            "model": models,
            "score_val": [val_score] + [val_score - 0.01 * i for i in range(1, n_models)],
            "fit_time": [10.0] * n_models,
            "pred_time_val": [0.1] * n_models,
        }
    )
    test_lb = pd.DataFrame(
        {
            "model": models,
            "score_test": [test_score] + [test_score - 0.01 * i for i in range(1, n_models)],
        }
    )
    return lb, test_lb


def _make_importance(features: list[str], values: list[float]):
    return pd.DataFrame({"importance": values}, index=features)


# ---- Tests ----


def test_no_issues_produces_positive_report(tmp_path: Path):
    pred = _make_predictor()
    lb, test_lb = _make_leaderboards(val_score=0.90, test_score=0.89)
    imp = _make_importance(["feat_a", "feat_b"], [0.15, 0.10])
    train = pd.DataFrame({"feat_a": range(500), "feat_b": range(500), "target": [0, 1] * 250})
    test = pd.DataFrame({"feat_a": range(150), "feat_b": range(150), "target": [0, 1] * 75})

    result = analyze_and_recommend(pred, train, test, lb, test_lb, imp, tmp_path)

    assert "Results look solid" in result["recommendations"][0]
    assert (tmp_path / "analysis.json").exists()
    assert (tmp_path / "analysis_report.txt").exists()


def test_overfitting_detected(tmp_path: Path):
    pred = _make_predictor()
    lb, test_lb = _make_leaderboards(val_score=0.95, test_score=0.80)  # 15.8% gap
    imp = _make_importance(["feat_a", "feat_b"], [0.15, 0.10])
    train = pd.DataFrame({"feat_a": range(200), "feat_b": range(200), "target": [0, 1] * 100})
    test = pd.DataFrame({"feat_a": range(50), "feat_b": range(50), "target": [0, 1] * 25})

    result = analyze_and_recommend(pred, train, test, lb, test_lb, imp, tmp_path)

    recs = " ".join(result["recommendations"])
    assert "overfitting" in recs.lower()


def test_moderate_gap_warning(tmp_path: Path):
    pred = _make_predictor()
    lb, test_lb = _make_leaderboards(val_score=0.90, test_score=0.84)  # ~6.7% gap
    imp = _make_importance(["feat_a", "feat_b"], [0.15, 0.10])
    train = pd.DataFrame({"feat_a": range(200), "feat_b": range(200), "target": [0, 1] * 100})
    test = pd.DataFrame({"feat_a": range(50), "feat_b": range(50), "target": [0, 1] * 25})

    result = analyze_and_recommend(pred, train, test, lb, test_lb, imp, tmp_path)

    recs = " ".join(result["recommendations"])
    assert "moderate" in recs.lower() or "monitor" in recs.lower()


def test_few_models_recommendation(tmp_path: Path):
    pred = _make_predictor()
    lb, test_lb = _make_leaderboards(n_models=3)
    imp = _make_importance(["feat_a", "feat_b"], [0.15, 0.10])
    train = pd.DataFrame({"feat_a": range(200), "feat_b": range(200), "target": [0, 1] * 100})
    test = pd.DataFrame({"feat_a": range(50), "feat_b": range(50), "target": [0, 1] * 25})

    result = analyze_and_recommend(pred, train, test, lb, test_lb, imp, tmp_path)

    recs = " ".join(result["recommendations"])
    assert "3 models" in recs or "time-limit" in recs.lower()


def test_low_importance_features_flagged(tmp_path: Path):
    pred = _make_predictor(features=["feat_a", "feat_b", "feat_c"])
    lb, test_lb = _make_leaderboards()
    imp = _make_importance(["feat_a", "feat_b", "feat_c"], [0.15, 0.0005, -0.02])
    train = pd.DataFrame(
        {
            "feat_a": range(200),
            "feat_b": range(200),
            "feat_c": range(200),
            "target": [0, 1] * 100,
        }
    )
    test = pd.DataFrame(
        {
            "feat_a": range(50),
            "feat_b": range(50),
            "feat_c": range(50),
            "target": [0, 1] * 25,
        }
    )

    result = analyze_and_recommend(pred, train, test, lb, test_lb, imp, tmp_path)

    recs = " ".join(result["recommendations"])
    assert "near-zero" in recs.lower() or "dropping" in recs.lower()
    assert "negative" in recs.lower() or "hurt" in recs.lower()


def test_class_imbalance_severe(tmp_path: Path):
    pred = _make_predictor()
    lb, test_lb = _make_leaderboards()
    imp = _make_importance(["feat_a", "feat_b"], [0.15, 0.10])
    # 95/5 split → 19:1 ratio
    train = pd.DataFrame(
        {
            "feat_a": range(200),
            "feat_b": range(200),
            "target": [0] * 190 + [1] * 10,
        }
    )
    test = pd.DataFrame({"feat_a": range(50), "feat_b": range(50), "target": [0, 1] * 25})

    result = analyze_and_recommend(pred, train, test, lb, test_lb, imp, tmp_path)

    recs = " ".join(result["recommendations"])
    assert "imbalance" in recs.lower()


def test_small_dataset_warning(tmp_path: Path):
    pred = _make_predictor(features=[f"f{i}" for i in range(50)])
    lb, test_lb = _make_leaderboards()
    imp = _make_importance([f"f{i}" for i in range(50)], [0.01] * 50)
    # 100 train rows, 50 features → ratio = 2x (< 10x threshold)
    train = pd.DataFrame({f"f{i}": range(100) for i in range(50)} | {"target": [0, 1] * 50})
    test = pd.DataFrame({f"f{i}": range(30) for i in range(50)} | {"target": [0, 1] * 15})

    result = analyze_and_recommend(pred, train, test, lb, test_lb, imp, tmp_path)

    recs = " ".join(result["recommendations"])
    assert "sample-to-feature" in recs.lower() or "ratio" in recs.lower()


def test_regression_skips_class_imbalance(tmp_path: Path):
    pred = _make_predictor(problem_type="regression")
    lb, test_lb = _make_leaderboards()
    imp = _make_importance(["feat_a", "feat_b"], [0.15, 0.10])
    train = pd.DataFrame({"feat_a": range(200), "feat_b": range(200), "target": range(200)})
    test = pd.DataFrame({"feat_a": range(50), "feat_b": range(50), "target": range(50)})

    result = analyze_and_recommend(pred, train, test, lb, test_lb, imp, tmp_path)

    recs = " ".join(result["recommendations"])
    assert "imbalance" not in recs.lower()


def test_model_family_extraction():
    assert _model_family("LightGBM_BAG_L1") == "lightgbm"
    assert _model_family("WeightedEnsemble_L2") == "weightedensemble"
    assert _model_family("CatBoost") == "catboost"
    assert _model_family("UnknownModel") == "UnknownModel"


def test_analysis_json_structure(tmp_path: Path):
    pred = _make_predictor()
    lb, test_lb = _make_leaderboards()
    imp = _make_importance(["feat_a", "feat_b"], [0.15, 0.10])
    train = pd.DataFrame({"feat_a": range(200), "feat_b": range(200), "target": [0, 1] * 100})
    test = pd.DataFrame({"feat_a": range(50), "feat_b": range(50), "target": [0, 1] * 25})

    analyze_and_recommend(pred, train, test, lb, test_lb, imp, tmp_path)

    with open(tmp_path / "analysis.json") as f:
        data = json.load(f)

    assert "best_model" in data
    assert "problem_type" in data
    assert "findings" in data
    assert "recommendations" in data
    assert isinstance(data["findings"], list)
    assert isinstance(data["recommendations"], list)
