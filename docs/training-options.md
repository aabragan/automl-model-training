# Training Options

This project uses [uv](https://docs.astral.sh/uv/) for dependency management and script execution. All CLI commands are defined as `[project.scripts]` in `pyproject.toml` and should be invoked via `uv run`:

```bash
uv run <command> [ARGS]
```

`uv run` resolves the project's virtual environment and dependencies automatically — no manual `pip install` or `source .venv/bin/activate` needed.

## Project Structure

```
src/automl_model_training/
├── config.py                          # Shared defaults & constants
├── data.py                            # CSV loading, train/test split, normalization
├── train.py                           # Model training + CLI entry points
├── predict.py                         # Inference + CLI entry points
├── backtest.py                        # Temporal walk-forward backtesting
└── evaluate/
    ├── analyze.py                     # Post-training accuracy analysis & recommendations
    ├── classification.py              # Train-time binary/multiclass artifacts
    ├── regression.py                  # Train-time regression artifacts
    ├── predict_classification.py      # Predict-time classification artifacts
    └── predict_regression.py          # Predict-time regression artifacts

tests/
├── conftest.py                        # Shared fixtures and mock predictors
├── test_analyze.py                    # Analysis & recommendation logic
├── test_config.py                     # Config defaults and make_run_dir
├── test_data.py                       # Data loading, splitting, feature dropping
├── test_data_artifacts.py             # CSV artifact generation
├── test_evaluate_classification.py    # Train-time classification artifacts
├── test_evaluate_regression.py        # Train-time regression artifacts
├── test_predict_classification.py     # Predict-time classification artifacts
├── test_predict_regression.py         # Predict-time regression artifacts
└── test_backtest.py                   # Temporal backtesting logic
```

## Entry Points

| Command                              | Problem Type   | Default Eval Metric          | Use When                                      |
|--------------------------------------|----------------|------------------------------|-----------------------------------------------|
| `uv run train`                       | Auto-detect    | Auto-detect                  | You want full control or have a non-standard task |
| `uv run train-binary`               | Binary         | `f1`                         | Your target column has exactly two classes     |
| `uv run train-regression`           | Regression     | `root_mean_squared_error`    | Your target column is continuous               |
| `uv run predict`                     | Auto-detect    | —                            | Run inference with a trained model             |
| `uv run predict-binary`             | Binary         | —                            | Run inference (binary convenience alias)       |
| `uv run predict-regression`         | Regression     | —                            | Run inference (regression convenience alias)   |
| `uv run backtest`                    | Auto-detect    | Auto-detect                  | Temporal walk-forward backtesting               |

---

## General (auto-detect) training

```bash
uv run train data.csv [OPTIONS]
```

Automatically detects the problem type and evaluation metric from the target column.

### Options

| Flag              | Default    | Description                                                        |
|-------------------|------------|--------------------------------------------------------------------|
| `--label`         | `target`   | Name of the target column in the CSV.                              |
| `--problem-type`  | auto       | Force a problem type: `binary`, `multiclass`, `regression`, `quantile`. |
| `--eval-metric`   | auto       | Evaluation metric (e.g. `f1`, `accuracy`, `roc_auc`, `rmse`).     |
| `--preset`        | `best`     | AutoGluon preset. Options include `best`, `high_quality`, `good_quality`, `medium_quality`. |
| `--time-limit`    | no limit   | Max training time in seconds. Omit to let all models finish.       |
| `--test-size`     | `0.2`      | Fraction of data held out for testing.                             |
| `--output-dir`    | `output`   | Directory where all artifacts are written.                         |
| `--drop`          | none       | Space-separated list of feature columns to exclude before training.|

### Example

```bash
# Auto-detect everything, 20% test split, output to ./output
uv run train data.csv

# Explicit binary classification with a 60-second time limit
uv run train data.csv --label churn --problem-type binary --eval-metric roc_auc --time-limit 60

# Drop two columns and write results to a custom directory
uv run train data.csv --drop feature_a feature_b --output-dir results/run1
```

---

## Binary classification training

```bash
uv run train-binary data.csv [OPTIONS]
```

A convenience wrapper that locks `--problem-type` to `binary` and defaults `--eval-metric` to `f1`. Accepts the same options as `train` except `--problem-type` is not available.

### Example

```bash
uv run train-binary data.csv --label is_fraud --time-limit 120
```

---

## Regression training

```bash
uv run train-regression data.csv [OPTIONS]
```

A convenience wrapper that locks `--problem-type` to `regression` and defaults `--eval-metric` to `root_mean_squared_error`. Accepts the same options as `train` except `--problem-type` is not available.

### Example

```bash
uv run train-regression data.csv --label price --test-size 0.3
```

---

## Prediction

After training, use the predict commands to run inference on new data. All three variants behave identically — the named versions are just convenience aliases.

```bash
uv run predict data.csv --model-dir output/AutogluonModels [OPTIONS]
```

### Options

| Flag              | Default               | Description                                      |
|-------------------|-----------------------|--------------------------------------------------|
| `--model-dir`     | (required)            | Path to the trained `AutogluonModels/` directory. |
| `--output-dir`    | `predictions_output`  | Directory for prediction outputs.                |

### Example

```bash
# Predict using a previously trained model
uv run predict new_data.csv --model-dir output/AutogluonModels

# Use the binary/regression aliases (identical behavior)
uv run predict-binary new_data.csv --model-dir output/AutogluonModels
uv run predict-regression new_data.csv --model-dir output/AutogluonModels --output-dir results/preds
```

---

## Backtesting

Temporal backtesting splits data by a date column to simulate how the model would have performed on future data. Supports a single cutoff split or multi-fold walk-forward evaluation.

```bash
uv run backtest data.csv --date-column date [OPTIONS]
```

### Options

| Flag              | Default    | Description                                                        |
|-------------------|------------|--------------------------------------------------------------------|
| `--date-column`   | (required) | Name of the date/datetime column for temporal splitting.           |
| `--cutoff`        | none       | Cutoff date for a single split (e.g. `2025-06-01`).               |
| `--n-splits`      | `1`        | Number of walk-forward folds. Overrides `--cutoff` when > 1.      |
| `--label`         | `target`   | Name of the target column.                                        |
| `--problem-type`  | auto       | Force: `binary`, `multiclass`, `regression`, `quantile`.          |
| `--eval-metric`   | auto       | Evaluation metric.                                                 |
| `--preset`        | `best`     | AutoGluon preset.                                                  |
| `--time-limit`    | no limit   | Training time limit in seconds per fold.                           |
| `--output-dir`    | `output`   | Base directory for backtest outputs.                               |
| `--drop`          | none       | Feature columns to exclude (the date column is always excluded).   |

### Example

```bash
# Single cutoff — train on data before June 2025, test on data after
uv run backtest data.csv --date-column date --cutoff 2025-06-01 --label price

# Walk-forward with 3 folds — each fold trains on all prior data
uv run backtest data.csv --date-column transaction_date --n-splits 3 --label churn

# With time limit per fold and custom output
uv run backtest data.csv --date-column date --n-splits 5 --time-limit 120 --output-dir backtest_results
```

### How It Works

1. Data is sorted chronologically by `--date-column`.
2. With `--cutoff`: rows before the cutoff become training data, rows on/after become test data.
3. With `--n-splits N`: data is divided into N+1 chronological chunks. Fold 1 trains on chunk 1 and tests on chunk 2; fold 2 trains on chunks 1–2 and tests on chunk 3; and so on.
4. Each fold runs a full `train_and_evaluate` cycle with its own model, leaderboard, and analysis.
5. Aggregate scores (mean ± std across folds) are saved to `backtest_summary.json`.

### Output

Each backtest run creates a timestamped directory (e.g. `output/backtest_20260322_140530/`) containing:

| Path                          | Description                                      |
|-------------------------------|--------------------------------------------------|
| `fold_1/`, `fold_2/`, ...     | Full training output for each fold (same as a regular training run). |
| `backtest_summary.json`       | Per-fold scores and aggregate mean ± std.        |

---

## Output Artifacts

Each training run creates a timestamped subfolder under `--output-dir`, e.g. `output/train_20260321_120530/`. Each prediction run does the same under its output dir, e.g. `predictions_output/predict_20260321_121045/`. This keeps every run isolated and prevents overwriting previous results.

Training runs write the following:

| File                          | Description                                                  |
|-------------------------------|--------------------------------------------------------------|
| `train_raw.csv`               | Raw training split.                                          |
| `test_raw.csv`                | Raw test split.                                              |
| `train_normalized.csv`        | RobustScaler-normalized training split (for external analysis only). |
| `test_normalized.csv`         | RobustScaler-normalized test split (for external analysis only).     |
| `leaderboard.csv`             | Validation scores for every trained model.                   |
| `leaderboard_test.csv`        | Test-set scores for every trained model.                     |
| `feature_importance.csv`      | Permutation-based feature importance on the test set.        |
| `model_info.json`             | Problem type, eval metric, features, and best model name.    |
| `analysis.json`               | Structured findings and improvement recommendations.         |
| `analysis_report.txt`         | Human-readable analysis report.                              |
| `AutogluonModels/`            | Serialized model directory (used by `predict` commands).     |

### Binary / multiclass extras

| File                          | Description                              |
|-------------------------------|------------------------------------------|
| `test_predictions.csv`        | Actual vs predicted with class probabilities. |
| `confusion_matrix.csv`        | Confusion matrix.                        |
| `classification_report.csv`   | Per-class precision, recall, F1.         |
| `roc_curve.csv`               | FPR, TPR, thresholds for ROC plotting.   |
| `roc_auc.json`                | ROC AUC score.                           |
| `precision_recall_curve.csv`  | Precision-recall curve data.             |
| `average_precision.json`      | Average precision score.                 |

### Regression extras

| File                          | Description                              |
|-------------------------------|------------------------------------------|
| `test_predictions.csv`        | Actual, predicted, and residual values.  |
| `residual_stats.json`         | MAE, RMSE, R², residual distribution stats. |
| `residual_distribution.csv`   | Binned residuals for histogram plotting. |


---

## Testing

Tests use pytest with mocked AutoGluon predictors so they run in seconds without real model training.

```bash
# Run all tests
uv run pytest tests/ -v

# Run a specific test file
uv run pytest tests/test_analyze.py -v

# Run a single test
uv run pytest tests/test_analyze.py::test_overfitting_detected -v
```

### Test Coverage

| Test File                          | Module Tested                        | What's Covered                                      |
|------------------------------------|--------------------------------------|-----------------------------------------------------|
| `test_config.py`                   | `config.py`                          | Default values, `make_run_dir` creation and uniqueness |
| `test_data.py`                     | `data.py`                            | Splitting, feature dropping, missing column handling |
| `test_data_artifacts.py`           | `data.py`                            | Raw and normalized CSV artifact generation           |
| `test_evaluate_classification.py`  | `evaluate/classification.py`         | All train-time classification files and ROC AUC validity |
| `test_evaluate_regression.py`      | `evaluate/regression.py`             | Residual stats, R² accuracy, file generation         |
| `test_predict_classification.py`   | `evaluate/predict_classification.py` | Probabilities, confidence, confusion matrix with ground truth |
| `test_predict_regression.py`       | `evaluate/predict_regression.py`     | Prediction stats with and without ground truth       |
| `test_analyze.py`                  | `evaluate/analyze.py`                | Overfitting, class imbalance, feature importance, dataset size, model diversity, JSON structure |
| `test_backtest.py`                 | `backtest.py`                        | Fold building, cutoff splits, walk-forward splits, aggregation, feature dropping               |
