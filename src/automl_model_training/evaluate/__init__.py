"""Evaluation artifact generators for training and prediction."""

from automl_model_training.evaluate.classification import save_classification_artifacts
from automl_model_training.evaluate.predict_classification import save_classification_outputs
from automl_model_training.evaluate.predict_regression import save_regression_outputs
from automl_model_training.evaluate.regression import save_regression_artifacts

__all__ = [
    "save_classification_artifacts",
    "save_classification_outputs",
    "save_regression_artifacts",
    "save_regression_outputs",
]
