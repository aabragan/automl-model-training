"""Ollama-powered LLM agent for iterative AutoML training.

Uses Ollama's OpenAI-compatible API to drive the tool loop.
Requires: uv add openai  +  ollama pull qwen2.5:14b

Usage:
    uv run python -m automl_model_training.ollama_agent data.csv --label target
    uv run python -m automl_model_training.ollama_agent data.csv --label price --model llama3.1:8b
"""

from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Callable
from typing import Any

from openai import OpenAI

from automl_model_training.config import DEFAULT_LABEL, DEFAULT_OUTPUT_DIR, setup_logging
from automl_model_training.tools import (
    tool_compare_runs,
    tool_detect_leakage,
    tool_engineer_features,
    tool_inspect_errors,
    tool_predict,
    tool_profile,
    tool_read_analysis,
    tool_train,
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
    "tool_detect_leakage": tool_detect_leakage,
    "tool_engineer_features": tool_engineer_features,
    "tool_train": tool_train,
    "tool_predict": tool_predict,
    "tool_inspect_errors": tool_inspect_errors,
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
6. Call tool_compare_runs after each iteration to track progress.
7. Stop when the score stops improving or you have iterated 5 times.
8. Summarize the best run and explain what worked.

Always explain your reasoning before calling a tool.
"""


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
                result = _TOOL_MAP[fn_name](**fn_args)
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
