"""LLM agent tool layer.

Thin wrappers over the existing automl_model_training API that return
JSON-serializable dicts. Designed to be registered as tools with any
LLM agent framework (Bedrock Agents, LangChain, OpenAI function calling).

This package was split out of a single ``tools.py`` module for readability.
The public API is unchanged — every ``tool_*`` function can still be
imported directly from ``automl_model_training.tools``.

Organization
------------
- profile.py          : tool_profile, tool_deep_profile, tool_detect_leakage
- train_predict.py    : tool_train, tool_predict, tool_tune_model, tool_optuna_tune
- feature_engineering.py : tool_engineer_features
- analysis.py         : tool_read_analysis, tool_compare_runs, tool_inspect_errors,
                         tool_compare_importance
- explainability.py   : tool_shap_interactions, tool_partial_dependence,
                         tool_partial_dependence_2way
- calibration.py      : tool_threshold_sweep, tool_calibration_curve
- model_eval.py       : tool_model_subset_evaluate

Iteration levers available to the LLM
--------------------------------------
preset           : Controls model diversity and training depth.
                   Ordered best→worst accuracy: extreme > best_quality/best >
                   best_v150 > high_quality/high > high_v150 > good > medium.
                   Start with "best", escalate if score is insufficient.

eval_metric      : The metric being optimized. Must match the problem type.
                   Binary: f1, roc_auc, accuracy, balanced_accuracy, mcc, log_loss
                   Multiclass: f1_macro, f1_weighted, accuracy, log_loss
                   Regression: root_mean_squared_error, mean_absolute_error, r2

problem_type     : Force binary, multiclass, regression, or quantile.
                   Use None to auto-detect from the label column.

drop             : List of feature column names to exclude. Use after
                   tool_train returns low-importance or harmful features
                   in analysis["findings"].

time_limit       : Seconds to train. None = train all models to completion.

test_size        : Fraction held out for evaluation (default 0.2).

calibrate_threshold : Binary only. Metric name to calibrate the decision
                   threshold for (e.g. "f1").

prune            : Remove underperforming ensemble members after training.

explain          : Compute SHAP values. Use when feature attribution is needed.

cv_folds         : Run k-fold cross-validation before the final train/test run.

seed             : Random seed. Change to verify score stability across splits.

Decision guide for the LLM
---------------------------
1. Always call tool_profile first to get drop_recommendations and
   label_distribution before the first tool_train call.

2. After each tool_train call, read analysis["findings"] and
   analysis["recommendations"]:
   - "low-importance features" → add them to drop on next iteration
   - "harmful features (negative importance)" → add to drop immediately
   - "overfitting" → switch to a less aggressive preset or add cv_folds
   - "class imbalance" → switch eval_metric to f1 or balanced_accuracy
   - "few models trained" → increase time_limit

3. Preset escalation order for accuracy:
   best → best_v150 → high_quality (if overfitting, go the other direction)

4. Call tool_compare_runs after each iteration to track progress and
   decide whether to continue or stop.
"""

from __future__ import annotations

from automl_model_training.tools.analysis import (
    tool_compare_importance,
    tool_compare_runs,
    tool_inspect_errors,
    tool_read_analysis,
)
from automl_model_training.tools.calibration import (
    tool_calibration_curve,
    tool_threshold_sweep,
)
from automl_model_training.tools.explainability import (
    tool_partial_dependence,
    tool_partial_dependence_2way,
    tool_shap_interactions,
)
from automl_model_training.tools.feature_engineering import tool_engineer_features
from automl_model_training.tools.model_eval import tool_model_subset_evaluate
from automl_model_training.tools.profile import (
    tool_deep_profile,
    tool_detect_leakage,
    tool_profile,
)
from automl_model_training.tools.train_predict import (
    tool_optuna_tune,
    tool_predict,
    tool_train,
    tool_tune_model,
)

__all__ = [
    "tool_calibration_curve",
    "tool_compare_importance",
    "tool_compare_runs",
    "tool_deep_profile",
    "tool_detect_leakage",
    "tool_engineer_features",
    "tool_inspect_errors",
    "tool_model_subset_evaluate",
    "tool_optuna_tune",
    "tool_partial_dependence",
    "tool_partial_dependence_2way",
    "tool_predict",
    "tool_profile",
    "tool_read_analysis",
    "tool_shap_interactions",
    "tool_threshold_sweep",
    "tool_train",
    "tool_tune_model",
]
