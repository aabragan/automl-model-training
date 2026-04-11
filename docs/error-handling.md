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
