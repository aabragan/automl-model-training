"""Evaluation artifact generators for training and prediction."""

from automl_model_training.evaluate.analyze import analyze_and_recommend
from automl_model_training.evaluate.classification import save_classification_artifacts
from automl_model_training.evaluate.explain import save_explainability_artifacts
from automl_model_training.evaluate.predict_classification import save_classification_outputs
from automl_model_training.evaluate.predict_regression import save_regression_outputs
from automl_model_training.evaluate.prune import (
    analyze_ensemble,
    prune_models,
    recommend_pruning,
    save_pruning_report,
)
from automl_model_training.evaluate.regression import save_regression_artifacts

__all__ = [
    "analyze_and_recommend",
    "analyze_ensemble",
    "prune_models",
    "recommend_pruning",
    "save_classification_artifacts",
    "save_classification_outputs",
    "save_explainability_artifacts",
    "save_pruning_report",
    "save_regression_artifacts",
    "save_regression_outputs",
]
