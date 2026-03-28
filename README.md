# automl-model-training

AutoML training and prediction pipeline built on [AutoGluon](https://auto.gluon.ai/). Point it at a CSV, and it trains an ensemble of models, evaluates on a held-out test set, analyzes accuracy, and recommends improvements — all from a single command.

## Features

- **Auto-detect or explicit** problem types: binary, multiclass, regression, quantile
- **Ensemble training** with automatic stacking, bagging, and model selection via AutoGluon
- **Post-training analysis** that flags overfitting, class imbalance, low-value features, and dataset issues — with actionable recommendations saved to every run
- **Prediction pipeline** that loads a trained model and runs inference on new data with problem-type-specific artifacts
- **Timestamped run directories** so every training and prediction run is isolated and nothing gets overwritten
- **Normalized data artifacts** (RobustScaler) saved alongside raw splits for external analysis
- **CI pipelines** for tests, linting (ruff), and type checking (mypy) on every PR

## Requirements

- Python >= 3.12
- [uv](https://docs.astral.sh/uv/) package manager

## Setup

```bash
git clone <repo-url>
cd automl-model-training

# Install all dependencies (uv manages the venv automatically)
uv sync
```

No manual `pip install` or `source .venv/bin/activate` needed — `uv run` handles everything.

## Quick Start

```bash
# Train with auto-detection — reads the "target" column, picks problem type and metric
uv run train data.csv

# Binary classification optimized for F1
uv run train-binary data.csv --label is_fraud --time-limit 120

# Regression optimized for RMSE
uv run train-regression data.csv --label price

# Run predictions on new data using a trained model
uv run predict new_data.csv --model-dir output/train_20260321_120530/AutogluonModels

# Backtest with a temporal cutoff
uv run backtest data.csv --date-column date --cutoff 2025-06-01 --label price

# Profile dataset before training — get correlation analysis and drop recommendations
uv run profile data.csv --label price
```

## Project Structure

```
src/automl_model_training/
├── config.py                          # Shared defaults & constants
├── data.py                            # CSV loading, train/test split, normalization
├── train.py                           # Model training + CLI entry points
├── predict.py                         # Inference + CLI entry points
├── backtest.py                        # Temporal walk-forward backtesting
├── profile.py                         # Dataset profiling and correlation analysis
└── evaluate/
    ├── analyze.py                     # Post-training accuracy analysis & recommendations
    ├── classification.py              # Train-time binary/multiclass artifacts
    ├── regression.py                  # Train-time regression artifacts
    ├── predict_classification.py      # Predict-time classification artifacts
    ├── predict_regression.py          # Predict-time regression artifacts
    ├── prune.py                       # Ensemble pruning analysis and model deletion
    └── explain.py                     # SHAP-based model explainability

tests/                                 # 71 unit tests with mocked predictors
docs/training-options.md               # Detailed CLI reference
.github/workflows/                     # CI: tests, lint, type check
```

## CLI Reference

### Profiling Command

| Command            | Description                                              |
|--------------------|----------------------------------------------------------|
| `uv run profile`   | Correlation analysis, feature recommendations, heatmap  |

### Profiling Options

| Flag             | Default   | Description                                             |
|------------------|-----------|---------------------------------------------------------|
| `--label`        | `target`  | Target column name                                      |
| `--threshold`    | `0.90`    | Correlation threshold for flagging pairs                |
| `--output-dir`   | `output`  | Base directory for outputs                              |

### Profiling Example

```bash
# Profile and get drop recommendations
uv run profile data.csv --label price --threshold 0.85
```

### Training Commands

| Command                    | Problem Type | Default Metric             | Use When                                         |
|----------------------------|--------------|----------------------------|--------------------------------------------------|
| `uv run train`             | Auto-detect  | Auto-detect                | Full control or non-standard task                |
| `uv run train-binary`      | Binary       | `f1`                       | Target column has exactly two classes            |
| `uv run train-regression`  | Regression   | `root_mean_squared_error`  | Target column is continuous                      |

### Training Options

| Flag             | Default   | Description                                                         |
|------------------|-----------|---------------------------------------------------------------------|
| `--label`        | `target`  | Name of the target column in the CSV                                |
| `--problem-type` | auto      | Force: `binary`, `multiclass`, `regression`, `quantile` (train only)|
| `--eval-metric`  | auto      | Evaluation metric (e.g. `f1`, `accuracy`, `roc_auc`, `rmse`)       |
| `--preset`       | `best`    | AutoGluon preset: `best`, `high_quality`, `good_quality`, `medium_quality` |
| `--time-limit`   | no limit  | Max training time in seconds                                        |
| `--test-size`    | `0.2`     | Fraction of data held out for testing                               |
| `--output-dir`   | `output`  | Base directory for run outputs                                      |
| `--drop`         | none      | Space-separated feature columns to exclude                          |
| `--prune`        | off       | Prune underperforming models from the ensemble after training       |
| `--explain`      | off       | Compute SHAP values for model explainability after training         |

### Training Examples

```bash
# Auto-detect everything
uv run train data.csv

# Explicit binary with ROC AUC metric and 60s time limit
uv run train data.csv --label churn --problem-type binary --eval-metric roc_auc --time-limit 60

# Drop features and write to a custom directory
uv run train data.csv --drop feature_a feature_b --output-dir results/experiment1

# Binary convenience wrapper (locks problem type, defaults to F1)
uv run train-binary data.csv --label is_fraud --time-limit 120

# Regression convenience wrapper (locks problem type, defaults to RMSE)
uv run train-regression data.csv --label price --test-size 0.3

# Train and prune underperforming models from the ensemble
uv run train data.csv --prune

# Train with SHAP explainability
uv run train data.csv --explain
```

### Prediction Commands

| Command                       | Description                              |
|-------------------------------|------------------------------------------|
| `uv run predict`              | Run inference (auto-detect problem type) |
| `uv run predict-binary`       | Convenience alias for binary models      |
| `uv run predict-regression`   | Convenience alias for regression models  |

All three behave identically — the named versions exist for clarity.

### Prediction Options

| Flag           | Default              | Description                                       |
|----------------|----------------------|---------------------------------------------------|
| `--model-dir`  | (required)           | Path to the trained `AutogluonModels/` directory   |
| `--output-dir` | `predictions_output` | Base directory for prediction outputs              |

### Prediction Examples

```bash
# Basic prediction
uv run predict new_data.csv --model-dir output/train_20260321_120530/AutogluonModels

# Custom output directory
uv run predict-regression new_data.csv \
  --model-dir output/train_20260321_120530/AutogluonModels \
  --output-dir results/preds
```

### Backtest Command

| Command            | Description                                      |
|--------------------|--------------------------------------------------|
| `uv run backtest`  | Temporal walk-forward backtesting                |

### Backtest Options

| Flag             | Default    | Description                                                    |
|------------------|------------|----------------------------------------------------------------|
| `--date-column`  | (required) | Date/datetime column for temporal splitting                    |
| `--cutoff`       | none       | Cutoff date for a single split (e.g. `2025-06-01`)            |
| `--n-splits`     | `1`        | Number of walk-forward folds (overrides `--cutoff` when > 1)  |
| `--label`        | `target`   | Target column name                                             |
| `--problem-type` | auto       | Force: `binary`, `multiclass`, `regression`, `quantile`       |
| `--eval-metric`  | auto       | Evaluation metric                                              |
| `--preset`       | `best`     | AutoGluon preset                                               |
| `--time-limit`   | no limit   | Training time limit per fold in seconds                        |
| `--output-dir`   | `output`   | Base directory for outputs                                     |
| `--drop`         | none       | Feature columns to exclude (date column always excluded)       |

### Backtest Examples

```bash
# Single cutoff — train before June 2025, test after
uv run backtest data.csv --date-column date --cutoff 2025-06-01 --label price

# Walk-forward with 3 folds
uv run backtest data.csv --date-column transaction_date --n-splits 3 --label churn

# With time limit per fold
uv run backtest data.csv --date-column date --n-splits 5 --time-limit 120
```

## Output Artifacts

Each run creates a timestamped subfolder (e.g. `output/train_20260321_120530/`) so previous results are never overwritten.

### Training Outputs

| File                        | Description                                                    |
|-----------------------------|----------------------------------------------------------------|
| `train_raw.csv`             | Raw training split                                             |
| `test_raw.csv`              | Raw test split                                                 |
| `train_normalized.csv`      | RobustScaler-normalized training split (external analysis only)|
| `test_normalized.csv`       | RobustScaler-normalized test split (external analysis only)    |
| `leaderboard.csv`           | Validation scores for every trained model                      |
| `leaderboard_test.csv`      | Test-set scores for every trained model                        |
| `feature_importance.csv`    | Permutation-based feature importance on the test set           |
| `model_info.json`           | Problem type, eval metric, features, best model name           |
| `analysis.json`             | Structured findings and improvement recommendations            |
| `analysis_report.txt`       | Human-readable analysis report                                 |
| `ensemble_analysis.csv`     | Per-model scores, timing, and contribution flags (with --prune)|
| `pruning_report.json`       | Pruned model list and remaining count (with --prune)           |
| `shap_summary.csv`          | Mean |SHAP| per feature, ranked (with --explain)               |
| `shap_values.csv`           | Raw SHAP values matrix (with --explain)                        |
| `shap_per_row.json`         | Top 5 contributing features per row (with --explain)           |
| `shap_metadata.json`        | Base values, problem type, top features (with --explain)       |
| `AutogluonModels/`          | Serialized model directory (used by predict commands)          |

#### Binary / Multiclass Extras

| File                        | Description                                |
|-----------------------------|--------------------------------------------|
| `test_predictions.csv`      | Actual vs predicted with class probabilities|
| `confusion_matrix.csv`      | Confusion matrix                           |
| `classification_report.csv` | Per-class precision, recall, F1            |
| `roc_curve.csv`             | FPR, TPR, thresholds for ROC plotting      |
| `roc_auc.json`              | ROC AUC score                              |
| `precision_recall_curve.csv`| Precision-recall curve data                |
| `average_precision.json`    | Average precision score                    |

#### Regression Extras

| File                        | Description                                |
|-----------------------------|--------------------------------------------|
| `test_predictions.csv`      | Actual, predicted, and residual values     |
| `residual_stats.json`       | MAE, RMSE, R², residual distribution stats |
| `residual_distribution.csv` | Binned residuals for histogram plotting    |

### Prediction Outputs

| File                          | Description                                          |
|-------------------------------|------------------------------------------------------|
| `predictions.csv`             | Input data with predicted values and probabilities   |
| `prediction_summary.json`     | Problem type, row count, best model, eval scores     |
| `probability_stats.csv`       | Class probability distribution (classification only) |
| `prediction_distribution.csv` | Predicted class counts and percentages (classification only) |
| `prediction_stats.json`       | Prediction distribution and residual stats (regression only) |

### Backtest Outputs

Each backtest run creates a timestamped directory (e.g. `output/backtest_20260322_140530/`):

| Path                          | Description                                          |
|-------------------------------|------------------------------------------------------|
| `fold_1/`, `fold_2/`, ...     | Full training output for each fold                   |
| `backtest_summary.json`       | Per-fold scores and aggregate mean ± std             |

## Post-Training Analysis

Every training run automatically produces an analysis report (`analysis.json` + `analysis_report.txt`) that checks for:

- **Overfitting** — compares validation vs test scores and flags significant gaps
- **Model diversity** — warns if top models are all from the same family
- **Feature importance** — identifies near-zero and negative-importance features worth dropping
- **Dataset size** — flags low sample-to-feature ratios and small test sets
- **Class imbalance** — detects skewed class distributions and suggests appropriate metrics

Each finding comes with a specific, actionable recommendation.

## Development

### Install Dev Dependencies

```bash
uv sync  # installs both runtime and dev dependencies
```

### Run Tests

Tests use mocked AutoGluon predictors so they run in seconds without real model training.

```bash
# All tests
uv run pytest tests/ -v

# Single file
uv run pytest tests/test_analyze.py -v

# Single test
uv run pytest tests/test_analyze.py::test_overfitting_detected -v
```

### Lint & Format

```bash
# Check for lint issues
uv run ruff check src/ tests/

# Auto-fix lint issues
uv run ruff check src/ tests/ --fix

# Check formatting
uv run ruff format --check src/ tests/

# Apply formatting
uv run ruff format src/ tests/
```

### Type Check

```bash
uv run mypy src/
```

### Test Coverage Map

| Test File                        | Module                               | Coverage                                                          |
|----------------------------------|--------------------------------------|-------------------------------------------------------------------|
| `test_config.py`                 | `config.py`                          | Default values, `make_run_dir` creation and uniqueness            |
| `test_data.py`                   | `data.py`                            | Splitting, feature dropping, missing column handling              |
| `test_data_artifacts.py`         | `data.py`                            | Raw and normalized CSV artifact generation                        |
| `test_evaluate_classification.py`| `evaluate/classification.py`         | All classification files, ROC AUC validity                        |
| `test_evaluate_regression.py`    | `evaluate/regression.py`             | Residual stats, R² accuracy, file generation                     |
| `test_predict_classification.py` | `evaluate/predict_classification.py` | Probabilities, confidence, confusion matrix with ground truth     |
| `test_predict_regression.py`     | `evaluate/predict_regression.py`     | Prediction stats with and without ground truth                    |
| `test_analyze.py`                | `evaluate/analyze.py`                | Overfitting, imbalance, feature importance, dataset size, diversity|
| `test_backtest.py`               | `backtest.py`                        | Fold building, cutoff splits, walk-forward, aggregation, feature dropping |
| `test_prune.py`                  | `evaluate/prune.py`                  | Ensemble analysis, pruning recommendations, model deletion, dependencies |
| `test_explain.py`                | `evaluate/explain.py`                | SHAP summary, per-row explanations, artifact generation, multiclass      |
| `test_profile.py`                | `profile.py`                         | Correlation matrix, pair detection, drop recommendations, heatmap        |

## CI Pipelines

Three GitHub Actions workflows run on every pull request to `main`:

| Workflow       | File                              | What It Does                        |
|----------------|-----------------------------------|-------------------------------------|
| Tests          | `.github/workflows/test.yml`      | `uv run pytest tests/ -v`          |
| Lint & Format  | `.github/workflows/lint.yml`      | `ruff check` + `ruff format --check`|
| Type Check     | `.github/workflows/typecheck.yml` | `uv run mypy src/`                  |

## How It Works

1. **Profiling** (`profile.py`) — optional pre-training step that computes a Pearson correlation matrix, identifies highly correlated feature pairs, recommends which to drop (keeping the one more correlated with the label), flags low-variance features, and generates a heatmap visualization.

2. **Data prep** (`data.py`) — loads the CSV, drops specified features, splits into train/test (stratified for classification), normalizes numeric features with RobustScaler, and saves all splits as CSV artifacts.

3. **Training** (`train.py`) — feeds raw (unscaled) data to AutoGluon's `TabularPredictor` with automatic stacking and bagging. AutoGluon handles all internal preprocessing — the normalized artifacts are for external analysis only. After training, it refits the best models on the full training set for faster inference.

4. **Evaluation** (`evaluate/`) — generates problem-type-specific artifacts: confusion matrices, ROC curves, precision-recall curves for classification; residual stats and distributions for regression.

5. **Analysis** (`evaluate/analyze.py`) — inspects the leaderboard, feature importance, dataset characteristics, and class distribution to produce findings and actionable recommendations.

6. **Ensemble pruning** (`evaluate/prune.py`) — when `--prune` is passed, analyzes which models contribute to the best ensemble, identifies underperformers (>5% worse than best, not in the dependency chain), and deletes them from disk to reduce footprint and inference latency.

7. **Explainability** (`evaluate/explain.py`) — when `--explain` is passed, computes SHAP values via KernelExplainer on the test set. Produces a global feature importance ranking (mean |SHAP|), raw SHAP values matrix, and per-row top-5 feature contributions with direction. Handles binary, multiclass, and regression.

8. **Prediction** (`predict.py`) — loads a trained model, runs inference on new data, and saves predictions with probabilities (classification) or residuals (regression). Evaluates against ground truth if the label column is present.

9. **Backtesting** (`backtest.py`) — splits data temporally by a date column and runs train/evaluate on each fold. Supports single-cutoff and walk-forward modes. Aggregates scores across folds to estimate real-world performance on future data.

## License

See [LICENSE](LICENSE) for details.
