# Error Handling Guide

## Table of Contents

- [Data Loading Errors](#data-loading-errors)
  - [FileNotFoundError: CSV path does not exist](#filenotfounderror-csv-path-does-not-exist)
  - [KeyError: Label column not found](#keyerror-label-column-not-found)
  - [ValueError: Stratified split fails](#valueerror-stratified-split-fails)
- [Training Errors](#training-errors)
  - [AutoGluon: No valid models trained](#autogluon-no-valid-models-trained)
  - [MemoryError during training](#memoryerror-during-training)
  - [SHAP KernelExplainer timeout (--explain)](#shap-kernelexplainer-timeout---explain)
- [Prediction Errors](#prediction-errors)
  - [FileNotFoundError: Model directory not found](#filenotfounderror-model-directory-not-found)
  - [Column mismatch between training and prediction data](#column-mismatch-between-training-and-prediction-data)
  - [Prediction with ground truth evaluation fails](#prediction-with-ground-truth-evaluation-fails)
- [Profiling Errors](#profiling-errors)
  - [No numeric columns found](#no-numeric-columns-found)
  - [Matplotlib backend errors](#matplotlib-backend-errors)
- [Backtest Errors](#backtest-errors)
  - [ValueError: Empty split from cutoff](#valueerror-empty-split-from-cutoff)
  - [ValueError: Not enough data for splits](#valueerror-not-enough-data-for-splits)
  - [Date parsing failures](#date-parsing-failures)
- [General Troubleshooting](#general-troubleshooting)
  - [Check your Python version](#check-your-python-version)
  - [Dependency issues](#dependency-issues)
  - [Disk space](#disk-space)
  - [Inspecting a failed run](#inspecting-a-failed-run)
- [Cross-Validation Errors](#cross-validation-errors)
  - [Too few samples for the number of folds](#too-few-samples-for-the-number-of-folds)
- [Drift Detection Errors](#drift-detection-errors)
  - [train_raw.csv not found](#train_rawcsv-not-found)
  - [No shared numeric features](#no-shared-numeric-features)
- [Model Comparison Errors](#model-comparison-errors)
  - [Run directory not found](#run-directory-not-found)
- [Ollama Agent Errors](#ollama-agent-errors)
  - [Connection refused](#connection-refused)
  - [Model not found](#model-not-found)
  - [Tool call JSON malformed / agent loops without progress](#tool-call-json-malformed--agent-loops-without-progress)
  - [Tool execution error returned to LLM](#tool-execution-error-returned-to-llm)
  - [openai package not installed](#openai-package-not-installed)
- [LLM Tool Errors](#llm-tool-errors)
  - [tool_engineer_features: Label rejected / missing columns / date parsing](#tool_engineer_features-label-column-rejected-as-source)
  - [tool_detect_leakage: Label column not in CSV](#tool_detect_leakage-label-column-not-in-csv)
  - [tool_inspect_errors: Missing run artifacts / row count mismatch](#tool_inspect_errors-missing-run-artifacts)
  - [tool_shap_interactions: Missing SHAP artifacts](#tool_shap_interactions-missing-shap-artifacts)
  - [tool_partial_dependence: Missing model / invalid feature](#tool_partial_dependence-missing-autogluonmodels-directory)
  - [tool_partial_dependence_2way: Cost cap / same feature](#tool_partial_dependence_2way-cost-cap-exceeded)
  - [tool_tune_model: Unsupported family / HPO out of time](#tool_tune_model-unsupported-model-family)
  - [tool_optuna_tune: All trials failed / invalid pruner](#tool_optuna_tune-all-trials-failed)
  - [tool_threshold_sweep / tool_calibration_curve: Non-binary run / invalid args](#tool_threshold_sweep--tool_calibration_curve-non-binary-run)
  - [tool_compare_importance: Missing or malformed feature_importance.csv](#tool_compare_importance-missing-feature_importancecsv)
  - [tool_model_subset_evaluate: Missing or malformed leaderboard](#tool_model_subset_evaluate-missing-or-malformed-leaderboard)

This document covers common errors you may encounter when running the training, prediction, profiling, and backtesting pipelines, along with their causes and fixes.

## Data Loading Errors

### FileNotFoundError: CSV path does not exist

```
FileNotFoundError: [Errno 2] No such file or directory: 'data.csv'
```

The CSV path passed as the first positional argument does not exist. Verify the path is correct and the file is readable.

### KeyError: Label column not found

```
KeyError: 'target'
```

The `--label` column name does not exist in the CSV. Check your column names with:

```bash
head -1 data.csv
```

Then pass the correct name: `--label your_column_name`.

### ValueError: Stratified split fails

```
ValueError: The least populated class in y has only 1 member
```

When the label has 20 or fewer unique values, the pipeline uses stratified splitting. If any class has fewer than 2 samples, scikit-learn cannot stratify. Solutions:

- Remove rare classes from the dataset before training
- Increase dataset size
- If the label is actually continuous (e.g., a score from 1-20), the heuristic misclassifies it as categorical — use `--problem-type regression` to bypass stratification

## Training Errors

### AutoGluon: No valid models trained

```
autogluon.common.utils.log_utils: WARNING - No models were trained
```

This typically means:

- `--time-limit` was too short for any model to complete — remove the flag or increase the value
- The dataset is too small or has too many missing values for any model to fit
- All features are constant or the label has zero variance

### MemoryError during training

AutoGluon with `presets='best'` and `auto_stack=True` trains many models with bagging and stacking, which is memory-intensive. The `extreme` preset is even more demanding as it loads Tabular Foundation Models. Options:

- Use `--preset high_quality` or `good` to reduce model count
- Use `--preset best_v150` or `high_v150` for v1.5 optimized presets (better quality, 5x faster)
- Add `--time-limit` to cap training duration
- Reduce dataset size or feature count

### SHAP KernelExplainer timeout (--explain)

SHAP's KernelExplainer is computationally expensive. If it hangs or runs out of memory on large datasets, the explainer automatically subsamples to 500 rows. For very wide datasets (hundreds of features), consider profiling first and dropping low-importance features with `--drop`.

## Prediction Errors

### FileNotFoundError: Model directory not found

```
FileNotFoundError: AutogluonModels directory does not exist
```

The `--model-dir` path must point to the `AutogluonModels/` directory inside a training run output. Example:

```bash
uv run predict data.csv --model-dir output/train_20260321_120530/AutogluonModels
```

### Column mismatch between training and prediction data

```
KeyError: "['feature_x'] not in index"
```

The prediction CSV must contain the same feature columns the model was trained on (minus the label, which is optional). Check `model_info.json` in the training output for the expected feature list.

### Prediction with ground truth evaluation fails

If the label column exists in the prediction CSV, the pipeline automatically evaluates against it. If the label values have a different format (e.g., strings vs integers), evaluation may fail. Ensure label encoding matches the training data.

## Profiling Errors

### No numeric columns found

If the dataset has no numeric columns, the correlation matrix will be empty and the heatmap will not be generated. The profiling report will still include categorical stats and missing value analysis.

### Matplotlib backend errors

```
RuntimeError: Invalid DISPLAY variable
```

The profiling script sets `matplotlib.use("Agg")` for headless environments. If you see display-related errors, ensure you are not importing matplotlib before the profile module.

## Backtest Errors

### ValueError: Empty split from cutoff

```
ValueError: Cutoff '2025-06-01' produces an empty split.
```

The `--cutoff` date falls outside the data's date range, producing an empty train or test set. The error message shows the actual date range — pick a cutoff within it.

### ValueError: Not enough data for splits

```
ValueError: Not enough data (50 rows) for 10 splits.
```

Walk-forward backtesting divides data into `n_splits + 1` chunks. If the dataset is too small relative to the number of splits, reduce `--n-splits`.

### Date parsing failures

```
ParserError: Unknown string format
```

The `--date-column` values must be parseable by `pd.to_datetime()`. Common formats like `YYYY-MM-DD`, `MM/DD/YYYY`, and ISO 8601 work automatically. For non-standard formats, preprocess the date column before running backtest.

## General Troubleshooting

### Check your Python version

This project requires Python >= 3.12. Verify with:

```bash
python --version
uv run python --version
```

### Dependency issues

If imports fail, sync dependencies:

```bash
uv sync
```

### Disk space

Training with `presets='best'` can produce large model directories (multiple GB for stacked ensembles). Use `--prune` to remove underperforming models after training, or check available disk space before long runs.

### Inspecting a failed run

Every run creates a timestamped output directory. Even if training fails partway through, partial artifacts (raw splits, early leaderboard entries) may already be saved. Check the run directory for any files that were written before the failure.

## Cross-Validation Errors

### Too few samples for the number of folds

```
ValueError: Cannot have number of splits n_splits=10 greater than the number of members in each class.
```

Reduce `--cv-folds` or increase dataset size. For stratified classification, each class needs at least as many samples as folds.

## Drift Detection Errors

### train_raw.csv not found

```
WARNING: train_raw.csv not found in output/train_<ts> — skipping drift check
```

The `--drift-check` path must point to the training run directory (not the `AutogluonModels/` subdirectory). The directory must contain `train_raw.csv`, which is generated during training.

### No shared numeric features

If the prediction data has no numeric features in common with the training data, drift detection is skipped with a warning. This can happen if column names changed between training and prediction.

## Model Comparison Errors

### Run directory not found

```
error: Run directory not found: output/nonexistent_run
```

All paths passed to `uv run compare` must be existing training run directories. Check the path and ensure the training run completed successfully.

## Ollama Agent Errors

### Connection refused

```
openai.APIConnectionError: Connection error.
```

The Ollama server is not running. Start it with:

```bash
ollama serve
```

Then verify it's reachable:

```bash
curl http://localhost:11434/api/tags
```

### Model not found

```
openai.NotFoundError: model 'qwen2.5:14b' not found
```

The model hasn't been pulled yet. Pull it first:

```bash
ollama pull qwen2.5:14b
```

### Tool call JSON malformed / agent loops without progress

Some smaller models (under 7B parameters) produce malformed tool call JSON or ignore the tool schema. Switch to a larger model:

```bash
uv run agent-ollama data.csv --label target --model qwen2.5:14b
```

`qwen2.5:14b` is the most reliable for tool-calling. `llama3.1:8b` is a good fallback.

### Tool execution error returned to LLM

If a tool call fails (e.g., bad CSV path, training error), the error is caught and returned to the LLM as `{"error": "..."}` rather than crashing the loop. The LLM will see the error message and can adjust its approach. Check the logs for the underlying cause:

```bash
uv run agent-ollama data.csv --label target --verbose
```

### openai package not installed

```
ModuleNotFoundError: No module named 'openai'
```

Run `uv sync` to install the `openai` dependency that was added to `pyproject.toml`.

## LLM Tool Errors

Errors specific to the tool layer (`automl_model_training.tools`) when driven by an LLM agent or called directly from a notebook.

### `tool_engineer_features`: Label column rejected as source

```
ValueError: Label column 'price' cannot be used in 'log' transformation
```

The leakage-prevention guard blocked a transformation that referenced the target column. Remove the label from your transformation spec and re-call with only feature columns.

### `tool_engineer_features`: Columns not in DataFrame

```
ValueError: log: columns not in DataFrame: ['nonexistent_feature']
```

The spec referenced a column that doesn't exist in the CSV. Check for typos — all column names must match exactly (case-sensitive).

### `tool_engineer_features`: date_parts parsing too few rows

```
ValueError: date_parts 'created_at': only 30% of values parsed as dates
```

Fewer than 50% of values in the column parsed as valid dates. Either clean the column first (normalize formats, remove sentinel values) or drop `date_parts` for this column.

### `tool_detect_leakage`: Label column not in CSV

```
ValueError: Label column 'target' not in CSV: [...]
```

The `label` argument doesn't match any column name in the CSV. Check the exact column headers — label column names are case-sensitive.

### `tool_inspect_errors`: Missing run artifacts

```
FileNotFoundError: tool_inspect_errors: missing .../test_predictions.csv
```

The `run_dir` you passed isn't a complete training run — it's missing `test_predictions.csv`, `test_raw.csv`, or `model_info.json`. Make sure you pass the output directory of a successful `tool_train` call (not the parent `output/` directory, not an `AutogluonModels/` subdirectory).

### `tool_inspect_errors`: Row count mismatch

```
ValueError: Row count mismatch: 100 predictions vs 99 test rows
```

The saved predictions don't align with the test split — this shouldn't normally happen. If you see this, the run directory is corrupted; re-run training.

### `tool_shap_interactions`: Missing SHAP artifacts

```
FileNotFoundError: missing shap_values.csv or shap_summary.csv in ...
```

The training run was done without `explain=True`, so no SHAP values were saved. Re-run with:

```python
tool_train(csv_path, label, explain=True, ...)
```

### `tool_partial_dependence`: Missing AutogluonModels directory

```
FileNotFoundError: tool_partial_dependence: missing .../AutogluonModels
```

The run directory is incomplete or training failed before the model was saved. Check the training logs for the underlying error.

### `tool_partial_dependence`: Features not in test data

```
ValueError: Features not in test data: ['ghost_feature']
```

You passed a feature name that doesn't match the training data columns. Omit `features=` to let the tool pick the top-5 by importance, or correct the spelling.

### `tool_tune_model`: Unsupported model family

```
ValueError: model_family 'NONSENSE' not supported. Choose from: [...]
```

Supported families are: `GBM`, `XGB`, `CAT`, `RF`, `XT`, `NN_TORCH`, `FASTAI`. Check the AutoGluon documentation for what each key maps to.

### `tool_tune_model`: HPO runs out of time

AutoGluon may log `Not enough time to try all hyperparameter configurations`. The `time_limit` was too short for the requested `n_trials`. Either increase `time_limit`, reduce `n_trials`, or pick a faster family (e.g., GBM is faster than NN_TORCH).

### `tool_optuna_tune`: All trials failed

```
RuntimeError: tool_optuna_tune: no trial completed successfully in study '...'. All 10 trials failed.
Representative errors: ['NumFeatures too high for FASTAI', ...]
```

Every Optuna trial hit an AutoGluon exception. The representative error messages point at the root cause — most often an incompatible `model_family` for the dataset (e.g., FASTAI on very wide data), a `time_limit_per_trial` too short for the family (NN_TORCH needs more than a few seconds), or a memory issue with the search space. Lower `n_trials`, raise `time_limit_per_trial`, or switch families.

### `tool_optuna_tune`: Unsupported model family

Same valid set as `tool_tune_model`: `GBM`, `XGB`, `CAT`, `RF`, `XT`, `NN_TORCH`, `FASTAI`.

### `tool_optuna_tune`: Invalid pruner

```
ValueError: pruner must be 'median' or 'none', got 'bogus'
```

Only `"median"` (MedianPruner) and `"none"` (NopPruner) are supported for now. `"median"` is the default; use `"none"` only to disable pruning for a direct wall-clock comparison with `tool_tune_model`.

### `tool_partial_dependence_2way`: Cost cap exceeded

```
ValueError: 2-way PDP would materialize 200,000 prediction rows, exceeding the max_cells=50,000 cap.
Reduce n_values_a and n_values_b to ≈14x14, or lower sample_size from 200, or raise max_cells if memory allows.
```

The 2-way PDP materializes `n_values_a * n_values_b * sample_size` prediction rows in a single DataFrame. The default cap catches unsafe configurations before OOM. Follow the suggestion in the error message — either shrink the grid, shrink the sample, or (if you have the memory) raise `max_cells`.

### `tool_partial_dependence_2way`: feature_a == feature_b

```
ValueError: feature_a and feature_b must differ
```

2-way PDP requires two distinct features. For a 1-D curve, use `tool_partial_dependence` instead.

### `tool_threshold_sweep` / `tool_calibration_curve`: Non-binary run

```
ValueError: tool_threshold_sweep requires binary classification; found 3 prob_ columns in test_predictions.csv.
```

These tools are binary-only — they infer the positive class from the `prob_<class>` columns. For multiclass, use `tool_read_analysis` or the per-class reports in `classification_report.csv`. For regression, neither tool applies.

### `tool_threshold_sweep`: Unknown metric

```
ValueError: Unknown metrics ['bogus']; valid: ['balanced_accuracy', 'f1', 'mcc', 'precision', 'recall']
```

Pass only the supported names. Omit `metrics=` to compute all five.

### `tool_calibration_curve`: Invalid strategy

```
ValueError: strategy must be 'quantile' or 'uniform', got 'bogus'
```

`"quantile"` (equal-frequency bins) is the default and usually preferred. Use `"uniform"` (equal-width bins) only when probabilities are evenly spread across `[0, 1]`.

### `tool_compare_importance`: Missing feature_importance.csv

```
FileNotFoundError: tool_compare_importance: missing .../feature_importance.csv
```

One or both of the run directories don't have `feature_importance.csv`. Every successful `tool_train` call writes one by default; if you see this error, the earlier run crashed partway through. Re-train and try again.

### `tool_compare_importance`: Unexpected schema

```
ValueError: feature_importance.csv must contain an 'importance' column
```

The CSV is malformed — usually because it was produced by a much older or patched version of AutoGluon. Re-train on the current version to regenerate the artifact.

### `tool_model_subset_evaluate`: Missing or malformed leaderboard

```
FileNotFoundError: tool_model_subset_evaluate: missing .../leaderboard_test.csv
ValueError: leaderboard_test.csv is missing expected columns ['stack_level']
RuntimeError: tool_model_subset_evaluate: leaderboard_test.csv at ... is empty.
```

All three point to an incomplete training run. `leaderboard_test.csv` is written on every successful `tool_train`; if it's missing, empty, or lacks `stack_level`, training crashed partway through. Re-train and try again.

### General: tool call returns `{"error": "..."}` in agent loop

When an LLM agent is driving the loop, any tool exception is caught and returned as a dict rather than crashing. Run the tool directly from Python or with `--verbose` to see the full traceback:

```bash
uv run agent-ollama data.csv --label target --verbose
```
