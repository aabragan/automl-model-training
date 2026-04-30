"""Ollama-powered LLM agent for iterative AutoML training.

Uses Ollama's OpenAI-compatible API to drive the tool loop.
Requires: uv add openai  +  ollama pull qwen2.5:14b

Usage:
    uv run python -m automl_model_training.ollama_agent data.csv --label target
    uv run python -m automl_model_training.ollama_agent data.csv --label price --model llama3.1:8b
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

from openai import OpenAI

from automl_model_training.config import DEFAULT_LABEL, DEFAULT_OUTPUT_DIR, setup_logging
from automl_model_training.tools import (
    tool_calibration_curve,
    tool_compare_importance,
    tool_compare_runs,
    tool_deep_profile,
    tool_detect_leakage,
    tool_engineer_features,
    tool_inspect_errors,
    tool_model_subset_evaluate,
    tool_optuna_tune,
    tool_partial_dependence,
    tool_partial_dependence_2way,
    tool_predict,
    tool_profile,
    tool_read_analysis,
    tool_shap_interactions,
    tool_threshold_sweep,
    tool_train,
    tool_tune_model,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tool schema — describes each tool to the LLM in OpenAI function-call format
# ---------------------------------------------------------------------------

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "tool_profile",
            "description": "Analyze a dataset before training. Always call this first.",
            "parameters": {
                "type": "object",
                "properties": {
                    "csv_path": {"type": "string", "description": "Path to the input CSV."},
                    "label": {"type": "string", "description": "Target column name."},
                },
                "required": ["csv_path", "label"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "tool_train",
            "description": (
                "Train an AutoGluon model. Returns score, analysis findings, leaderboard, "
                "low_importance_features, and negative_importance_features. "
                "Preset options (best→worst accuracy): extreme, best_quality, best, best_v150, "
                "high_quality, high, good, medium. "
                "eval_metric options — binary: f1, roc_auc, accuracy, balanced_accuracy; "
                "regression: root_mean_squared_error, mean_absolute_error, r2."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "csv_path": {"type": "string"},
                    "label": {"type": "string"},
                    "preset": {"type": "string", "default": "best"},
                    "problem_type": {
                        "type": "string",
                        "enum": ["binary", "multiclass", "regression", "quantile"],
                        "description": "Omit to auto-detect.",
                    },
                    "eval_metric": {"type": "string", "description": "Omit to auto-detect."},
                    "time_limit": {"type": "integer", "description": "Seconds. Omit for no limit."},
                    "drop": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Feature columns to exclude.",
                    },
                    "test_size": {"type": "number", "default": 0.2},
                    "seed": {"type": "integer", "default": 42},
                    "prune": {"type": "boolean", "default": False},
                    "cv_folds": {
                        "type": "integer",
                        "description": "k-fold CV. Use for small datasets (<1000 rows).",
                    },
                    "calibrate_threshold": {
                        "type": "string",
                        "description": "Binary only. Metric to calibrate threshold for (e.g. f1).",
                    },
                    "output_dir": {"type": "string", "default": "output"},
                },
                "required": ["csv_path", "label"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "tool_detect_leakage",
            "description": (
                "Detect features that are suspiciously predictive of the target. "
                "Trains a depth-3 decision tree on each feature individually; any feature "
                "that alone scores above the threshold (default 0.95) is almost certainly "
                "leaking (a copy or proxy of the target). Call this BEFORE tool_train to "
                "avoid wasting time optimizing a leaky model."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "csv_path": {"type": "string"},
                    "label": {"type": "string"},
                    "threshold": {
                        "type": "number",
                        "default": 0.95,
                        "description": "Score above which a feature is flagged as leaking.",
                    },
                    "sample_size": {
                        "type": "integer",
                        "default": 5000,
                        "description": "Rows to subsample for the test.",
                    },
                    "seed": {"type": "integer", "default": 42},
                },
                "required": ["csv_path", "label"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "tool_engineer_features",
            "description": (
                "Apply declarative feature transformations to a CSV. "
                "Use after tool_profile to create features the model can't derive itself: "
                "log for skewed distributions, ratio for relationships, date_parts for dates. "
                "Pass the returned engineered_csv to tool_train. Supported transforms: "
                "log, sqrt, ratio, diff, product, bin, date_parts, onehot, target_mean, "
                "interact_top_k."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "csv_path": {"type": "string"},
                    "transformations": {
                        "type": "object",
                        "description": (
                            "Spec dict. Examples: "
                            '{"log": ["price"], "ratio": [["debt", "income"]], '
                            '"date_parts": ["sale_date"], "bin": {"age": [0, 18, 65, 120]}}'
                        ),
                    },
                    "label": {
                        "type": "string",
                        "description": (
                            "Target column — rejected as transform source to prevent leakage."
                        ),
                    },
                    "output_dir": {"type": "string", "default": "output"},
                },
                "required": ["csv_path", "transformations"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "tool_predict",
            "description": "Run inference on new data using a trained model.",
            "parameters": {
                "type": "object",
                "properties": {
                    "csv_path": {"type": "string"},
                    "model_dir": {
                        "type": "string",
                        "description": "Path to AutogluonModels/ inside a run_dir.",
                    },
                    "output_dir": {"type": "string", "default": "predictions_output"},
                    "min_confidence": {"type": "number"},
                    "decision_threshold": {"type": "number"},
                },
                "required": ["csv_path", "model_dir"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "tool_read_analysis",
            "description": (
                "Re-read analysis.json from a completed training run without retraining."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "run_dir": {
                        "type": "string",
                        "description": "Path to the training run directory.",
                    }
                },
                "required": ["run_dir"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "tool_inspect_errors",
            "description": (
                "Return the N worst predictions from a training run with feature values and "
                "pattern hints. Use after tool_train to see actual failure modes rather than "
                "aggregate metrics — helpful for spotting label noise, leakage, systematic bias, "
                "or subpopulations the model can't handle."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "run_dir": {
                        "type": "string",
                        "description": "Path to a completed training run directory.",
                    },
                    "n": {
                        "type": "integer",
                        "default": 20,
                        "description": "Number of rows to return.",
                    },
                    "worst": {
                        "type": "boolean",
                        "default": True,
                        "description": "True for worst predictions, False for best.",
                    },
                },
                "required": ["run_dir"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "tool_deep_profile",
            "description": (
                "Extended per-feature profiling with direct suggested_transforms for "
                "tool_engineer_features. Call after tool_profile when you plan to engineer "
                "features. Returns numeric skewness, categorical cardinality, outlier %, "
                "and a ready-to-use transforms spec."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "csv_path": {"type": "string"},
                    "label": {"type": "string"},
                },
                "required": ["csv_path", "label"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "tool_shap_interactions",
            "description": (
                "Find pairs of top-k important features whose SHAP contributions "
                "correlate across rows. Surfaces redundant features (drop one) and "
                "strongly-coupled pairs (engineer their ratio/product). Requires a "
                "training run done with explain=True."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "run_dir": {"type": "string"},
                    "top_k": {"type": "integer", "default": 5},
                },
                "required": ["run_dir"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "tool_partial_dependence",
            "description": (
                "Compute partial-dependence curves showing how each feature affects "
                "predictions across its range. Use when SHAP shows a feature is important "
                "but you want to know HOW it matters (monotonic, threshold effect, etc.)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "run_dir": {"type": "string"},
                    "features": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Features to analyze. Omit to use top-5 by importance.",
                    },
                    "n_values": {"type": "integer", "default": 20},
                    "sample_size": {"type": "integer", "default": 200},
                },
                "required": ["run_dir"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "tool_tune_model",
            "description": (
                "Run targeted hyperparameter tuning on a single AutoGluon model family. "
                "Use when the leaderboard shows one family dominating (e.g., GBM wins) "
                "and you want to squeeze more performance out of it specifically."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "csv_path": {"type": "string"},
                    "label": {"type": "string"},
                    "model_family": {
                        "type": "string",
                        "enum": ["GBM", "XGB", "CAT", "RF", "XT", "NN_TORCH", "FASTAI"],
                    },
                    "n_trials": {"type": "integer", "default": 20},
                    "time_limit": {"type": "integer", "default": 300},
                    "drop": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["csv_path", "label", "model_family"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "tool_optuna_tune",
            "description": (
                "Optuna-driven hyperparameter search for a single model family. Competitive "
                "alternative to tool_tune_model: uses TPE (better than random on tabular HPO), "
                "MedianPruner (2-3x wall-clock savings), and sqlite-backed study persistence. "
                "Leave study_name and storage unset — the agent supplies stable defaults so "
                "repeated calls within or across sessions resume the same study and the "
                "sampler keeps improving. Override them only to start a fresh search."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "csv_path": {"type": "string"},
                    "label": {"type": "string"},
                    "model_family": {
                        "type": "string",
                        "enum": ["GBM", "XGB", "CAT", "RF", "XT", "NN_TORCH", "FASTAI"],
                    },
                    "n_trials": {"type": "integer", "default": 20},
                    "time_limit_per_trial": {
                        "type": "integer",
                        "default": 60,
                        "description": "Seconds per trial. Total ≈ n_trials × this.",
                    },
                    "eval_metric": {"type": "string"},
                    "problem_type": {
                        "type": "string",
                        "enum": ["binary", "multiclass", "regression", "quantile"],
                    },
                    "drop": {"type": "array", "items": {"type": "string"}},
                    "pruner": {
                        "type": "string",
                        "enum": ["median", "none"],
                        "default": "median",
                    },
                    "study_name": {
                        "type": "string",
                        "description": "Override to start a fresh study.",
                    },
                    "storage": {
                        "type": "string",
                        "description": "Override to relocate the sqlite DB.",
                    },
                },
                "required": ["csv_path", "label", "model_family"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "tool_threshold_sweep",
            "description": (
                "Binary classification: sweep the decision threshold and return per-metric "
                "curves (F1, precision, recall, MCC, balanced_accuracy) with the argmax "
                "threshold for each. Use after tool_train on a binary problem to see the "
                "full precision/recall trade-off shape rather than relying on the single "
                "value produced by calibrate_threshold."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "run_dir": {"type": "string"},
                    "n_thresholds": {"type": "integer", "default": 99},
                    "metrics": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "Subset of f1, precision, recall, mcc, balanced_accuracy. "
                            "Omit for all five."
                        ),
                    },
                },
                "required": ["run_dir"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "tool_calibration_curve",
            "description": (
                "Binary classification: reliability diagram answering 'when the model says "
                "0.8, is it right 80% of the time?'. Returns per-bin predicted vs actual "
                "positive rate, ECE, and classifies miscalibration direction "
                "(over_confident, under_confident, well_calibrated, mixed). Call this before "
                "threshold work — if probabilities are well-calibrated, threshold tuning "
                "usually does not help."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "run_dir": {"type": "string"},
                    "n_bins": {"type": "integer", "default": 10},
                    "strategy": {
                        "type": "string",
                        "enum": ["quantile", "uniform"],
                        "default": "quantile",
                    },
                },
                "required": ["run_dir"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "tool_compare_importance",
            "description": (
                "Diff feature importance between two training runs to see what a "
                "feature-engineering or drop-list change actually did. Flags "
                "dominant_new_feature (new feature tops the importance list but score "
                "barely moved — likely leakage), gained/lost features, and top changes "
                "ranked by |delta|."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "run_dir_before": {"type": "string"},
                    "run_dir_after": {"type": "string"},
                    "top_n": {"type": "integer", "default": 10},
                    "significance_delta": {"type": "number", "default": 0.01},
                },
                "required": ["run_dir_before", "run_dir_after"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "tool_partial_dependence_2way",
            "description": (
                "2-way partial-dependence surface for two features. Complements "
                "tool_shap_interactions by returning the actual shape of the interaction "
                "(additive, synergy, saddle, threshold) rather than just a magnitude. "
                "Use after SHAP flags an interesting pair. Cost-capped at 50k prediction "
                "rows by default; reduce n_values_a/n_values_b or sample_size if the "
                "tool refuses the call."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "run_dir": {"type": "string"},
                    "feature_a": {"type": "string"},
                    "feature_b": {"type": "string"},
                    "n_values_a": {"type": "integer", "default": 10},
                    "n_values_b": {"type": "integer", "default": 10},
                    "sample_size": {"type": "integer", "default": 200},
                },
                "required": ["run_dir", "feature_a", "feature_b"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "tool_model_subset_evaluate",
            "description": (
                "Per-model report from the training leaderboard. Returns every model's "
                "test score, inference time, and stack level, and flags a "
                "recommended_deploy model when a faster single model is within "
                "score_tolerance of the ensemble — useful for deciding whether the "
                "ensemble complexity is actually worth its inference cost."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "run_dir": {"type": "string"},
                    "score_tolerance": {"type": "number", "default": 0.01},
                },
                "required": ["run_dir"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "tool_compare_runs",
            "description": "Compare all recorded training experiments. Call after each iteration.",
            "parameters": {
                "type": "object",
                "properties": {
                    "last_n": {
                        "type": "integer",
                        "description": "Return only the last N experiments.",
                    }
                },
            },
        },
    },
]

# Map tool names to callables
_TOOL_MAP: dict[str, Callable[..., Any]] = {
    "tool_profile": tool_profile,
    "tool_deep_profile": tool_deep_profile,
    "tool_detect_leakage": tool_detect_leakage,
    "tool_engineer_features": tool_engineer_features,
    "tool_train": tool_train,
    "tool_tune_model": tool_tune_model,
    "tool_optuna_tune": tool_optuna_tune,
    "tool_predict": tool_predict,
    "tool_inspect_errors": tool_inspect_errors,
    "tool_shap_interactions": tool_shap_interactions,
    "tool_partial_dependence": tool_partial_dependence,
    "tool_partial_dependence_2way": tool_partial_dependence_2way,
    "tool_threshold_sweep": tool_threshold_sweep,
    "tool_calibration_curve": tool_calibration_curve,
    "tool_compare_importance": tool_compare_importance,
    "tool_model_subset_evaluate": tool_model_subset_evaluate,
    "tool_read_analysis": tool_read_analysis,
    "tool_compare_runs": tool_compare_runs,
}

SYSTEM_PROMPT = """\
You are an AutoML training agent. Your goal is to iteratively train the best possible model.

Workflow:
1. Call tool_profile first to understand the dataset and get drop recommendations.
2. Call tool_detect_leakage to catch features that perfectly predict the target.
   Any feature in suspected_leaks MUST be added to the drop list passed to tool_train —
   training on leaked features produces a worthless model.
3. Consider tool_engineer_features before the first tool_train when profile shows:
   - heavily skewed numeric features → use "log" or "sqrt"
   - related pairs that should become ratios → use "ratio"
   - datetime columns → use "date_parts"
   - high-cardinality categorical with ordinal meaning → use "bin"
   Pass the returned engineered_csv to tool_train instead of the original CSV.
4. Call tool_train with preset="best" and the recommended drops as a baseline.
5. After each training run, read analysis["findings"] and decide:
   - "negative_importance_features" → add to drop immediately
   - "low_importance_features" → add to drop if score hasn't improved
   - "overfitting" → switch to a less aggressive preset (best → high_quality)
   - "class imbalance" → switch eval_metric to f1 or balanced_accuracy
   - "few models trained" → increase time_limit
   When the score plateaus, call tool_inspect_errors to see the worst predictions —
   patterns in the errors (clustered values, systematic bias, high-confidence
   mistakes) often reveal data issues that no aggregate metric can.
6. For hyperparameter tuning, prefer tool_optuna_tune over tool_tune_model. Optuna
   uses TPE + pruning (faster) and the agent persists the study across calls — when
   you call tool_optuna_tune multiple times for the same (csv, label, model_family),
   the sampler keeps learning from prior trials. Let study_name and storage default.
7. After you have a binary classifier, call tool_calibration_curve to check whether
   probabilities are trustworthy. If direction is "over_confident" or "under_confident",
   calibration is the problem — not the threshold. If "well_calibrated", move to
   tool_threshold_sweep to see the precision/recall trade-off shape.
8. Call tool_compare_runs after each iteration to track progress. When comparing two
   runs where features changed, also call tool_compare_importance to see which
   features gained/lost importance — a new feature that dominates importance but
   barely moves the score is almost always leakage.
9. Stop when the score stops improving or you have iterated 5 times.
10. Summarize the best run and explain what worked.

Always explain your reasoning before calling a tool.
"""


def _make_optuna_defaults(
    csv_path: str,
    label: str,
    model_family: str,
    output_dir: str,
) -> tuple[str, str]:
    """Return (study_name, storage) defaults for Optuna persistence.

    The storage is a single sqlite DB shared by every Optuna study in this
    session, placed at ``{output_dir}/optuna_studies.db``. The study_name
    is keyed on the triple (csv_path, label, model_family) via a short SHA-1
    hash so separate datasets and model families don't collide, and so
    repeated calls within or across sessions resume the same study (letting
    the TPE sampler keep improving).
    """
    key = f"{Path(csv_path).resolve()}\n{label}\n{model_family}"
    digest = hashlib.sha1(key.encode()).hexdigest()[:10]
    study_name = f"optuna_{model_family.lower()}_{digest}"
    storage = f"sqlite:///{Path(output_dir).resolve() / 'optuna_studies.db'}"
    return study_name, storage


def _dispatch_tool(
    fn_name: str,
    fn_args: dict,
    agent_output_dir: str,
) -> Any:
    """Run a tool call and return its result.

    Wraps _TOOL_MAP to inject session-level defaults. The main one is Optuna
    study persistence: if the LLM calls tool_optuna_tune without an explicit
    study_name or storage, we fill them in with stable defaults so the study
    is resumed on subsequent calls rather than restarted from scratch.
    """
    if fn_name == "tool_optuna_tune" and (
        not fn_args.get("study_name") or not fn_args.get("storage")
    ):
        # Need csv_path, label, model_family (all required args of the tool)
        default_name, default_storage = _make_optuna_defaults(
            csv_path=fn_args["csv_path"],
            label=fn_args["label"],
            model_family=fn_args["model_family"],
            output_dir=fn_args.get("output_dir", agent_output_dir),
        )
        fn_args.setdefault("study_name", default_name)
        fn_args.setdefault("storage", default_storage)
        logger.debug(
            "Applied Optuna persistence defaults: study_name=%s storage=%s",
            fn_args["study_name"],
            fn_args["storage"],
        )
    return _TOOL_MAP[fn_name](**fn_args)


def run_ollama_agent(
    csv_path: str,
    label: str,
    model: str = "qwen2.5:14b",
    base_url: str = "http://localhost:11434/v1",
    max_iterations: int = 5,
    output_dir: str = DEFAULT_OUTPUT_DIR,
) -> None:
    """Run the Ollama-powered AutoML agent loop."""
    client = OpenAI(base_url=base_url, api_key="ollama")

    messages: list[dict] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Train the best possible model for '{csv_path}' with label='{label}'. "
                f"Output directory: '{output_dir}'. Max iterations: {max_iterations}."
            ),
        },
    ]

    logger.info("Starting Ollama agent with model=%s", model)

    for _ in range(max_iterations * 3):  # generous upper bound on total LLM calls
        response = client.chat.completions.create(  # type: ignore[call-overload]
            model=model,
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
        )

        msg = response.choices[0].message
        messages.append(msg.model_dump(exclude_none=True))

        # No tool calls — agent is done
        if not msg.tool_calls:
            logger.info("\nAgent final response:\n%s", msg.content)
            print(msg.content)
            break

        # Execute each tool call and feed results back
        for call in msg.tool_calls:
            fn_name = call.function.name
            fn_args = json.loads(call.function.arguments)

            logger.info("Calling %s(%s)", fn_name, fn_args)

            try:
                result = _dispatch_tool(fn_name, fn_args, agent_output_dir=output_dir)
            except Exception as exc:
                result = {"error": str(exc)}

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": json.dumps(result, default=str),
                }
            )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Ollama AutoML agent.")
    parser.add_argument("csv", help="Path to the input CSV file.")
    parser.add_argument("--label", default=DEFAULT_LABEL, help="Target column name.")
    parser.add_argument("--model", default="qwen2.5:14b", help="Ollama model name.")
    parser.add_argument(
        "--base-url", default="http://localhost:11434/v1", help="Ollama API base URL."
    )
    parser.add_argument("--max-iterations", type=int, default=5, help="Max training iterations.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Base output directory.")
    verbosity = parser.add_mutually_exclusive_group()
    verbosity.add_argument("--verbose", "-v", action="store_true", default=False)
    verbosity.add_argument("--quiet", "-q", action="store_true", default=False)
    args = parser.parse_args()

    setup_logging(verbose=args.verbose, quiet=args.quiet)
    run_ollama_agent(
        csv_path=args.csv,
        label=args.label,
        model=args.model,
        base_url=args.base_url,
        max_iterations=args.max_iterations,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
