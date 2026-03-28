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
├── profile.py                         # Dataset profiling and correlation analysis
├── experiment.py                      # Local experiment tracking and comparison
└── evaluate/
    ├── analyze.py                     # Post-training accuracy analysis & recommendations
    ├── classification.py              # Train-time binary/multiclass artifacts
    ├── regression.py                  # Train-time regression artifacts
    ├── predict_classification.py      # Predict-time classification artifacts
    ├── predict_regression.py          # Predict-time regression artifacts
    ├── prune.py                       # Ensemble pruning analysis and model deletion
    └── explain.py                     # SHAP-based model explainability

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
├── test_backtest.py                   # Temporal backtesting logic
└── test_prune.py                      # Ensemble pruning logic
└── test_explain.py                    # SHAP explainability logic
└── test_profile.py                    # Dataset profiling and correlation logic
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
| `uv run profile`                     | —              | —                            | Dataset profiling and correlation analysis       |
| `uv run experiments`                 | —              | —                            | Compare recorded training experiments            |
| `uv run backtest`                    | Auto-detect    | Auto-detect                  | Temporal walk-forward backtesting               |
| `uv run profile`                     | —              | —                            | Dataset profiling and correlation analysis       |

---

## Dataset Profiling

Run `profile` before training to analyze your dataset's quality, structure, and correlation patterns.

```bash
uv run profile data.csv [OPTIONS]
```

### Options

| Flag              | Default    | Description                                                        |
|-------------------|------------|--------------------------------------------------------------------|
| `--label`         | `target`   | Name of the target column.                                         |
| `--threshold`     | `0.90`     | Correlation threshold for flagging pairs.                          |
| `--output-dir`    | `output`   | Directory for profile outputs.                                     |

### Example

```bash
# Profile with default 0.90 threshold
uv run profile data.csv --label price

# Lower threshold to catch more correlated pairs
uv run profile data.csv --label churn --threshold 0.80

# Custom output directory
uv run profile data.csv --output-dir analysis/
```

### How It Works

1. Computes a dataset overview (shape, types, memory usage, duplicate rows).
2. Analyzes missing values per column (counts and percentages).
3. Profiles numeric features: descriptive stats, skew, kurtosis, and outlier detection via IQR method.
4. Profiles categorical features: cardinality, top values, missing rates.
5. Analyzes the label column: class distribution and imbalance ratio (classification) or distribution stats (regression).
6. Computes a Pearson correlation matrix and identifies highly correlated pairs.
7. For each correlated pair, recommends dropping the feature with the lower absolute correlation to the label.
8. Flags low-variance features (near-zero standard deviation).
9. Generates a heatmap visualization of the correlation matrix.

### Profile Outputs

| File                            | Description                                                    |
|---------------------------------|----------------------------------------------------------------|
| `missing_values.csv`            | Per-column missing counts and percentages.                     |
| `numeric_feature_stats.csv`     | Descriptive stats, skew, kurtosis, outlier counts (IQR method).|
| `categorical_feature_stats.csv` | Cardinality, top values, missing rates (if categorical cols exist). |
| `correlation_matrix.csv`        | Full Pearson correlation matrix.                               |
| `correlation_heatmap.png`       | Annotated heatmap visualization.                               |
| `profile_report.json`           | Full summary: overview, label analysis, missing values, outliers, correlations, drop recommendations. |

The CLI prints a ready-to-use `--drop` flag you can paste into your train command.

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
| `--prune`         | off        | Prune underperforming models from the ensemble after training.     |
| `--explain`       | off        | Compute SHAP values for model explainability after training.       |

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

## Multiclass classification

There is no dedicated `train-multiclass` convenience command. Use the generic `train` command — AutoGluon auto-detects multiclass when the label has more than two unique values, or force it explicitly:

```bash
uv run train data.csv --label category --problem-type multiclass
```

### How multiclass differs from binary

**Auto-detection:** AutoGluon classifies a label as multiclass when it has 3–20 unique values. Above 20, it defaults to regression. Use `--problem-type multiclass` to override the heuristic if needed.

**Default eval metric:** `accuracy`. For imbalanced multiclass problems, consider `--eval-metric f1_macro`, `f1_weighted`, or `log_loss`.

**Recommended metrics by scenario:**

| Scenario | Metric | Why |
|----------|--------|-----|
| Balanced classes | `accuracy` | Simple, interpretable |
| Imbalanced classes | `f1_macro` | Treats all classes equally regardless of size |
| Imbalanced, larger classes matter more | `f1_weighted` | Weights by class frequency |
| Probability calibration matters | `log_loss` | Penalizes confident wrong predictions |

**Decision threshold calibration:** The `calibrate_decision_threshold="auto"` parameter in the training pipeline is a no-op for multiclass — it only applies to binary classification.

**Stratified splitting:** The train/test split uses stratified sampling for multiclass (same as binary), preserving class proportions in both sets. This requires at least 2 samples per class.

### Evaluation artifacts

Multiclass produces the same artifact files as binary, but with differences in content:

| Artifact | Multiclass behavior |
|----------|-------------------|
| `test_predictions.csv` | One `prob_<class>` column per class |
| `confusion_matrix.csv` | N×N matrix (one row/column per class) |
| `classification_report.csv` | Per-class precision, recall, F1 plus macro/weighted averages |
| `roc_curve.csv` | Computed for the last sorted class only (one-vs-rest) |
| `roc_auc.json` | AUC for the last sorted class only |
| `precision_recall_curve.csv` | Computed for the last sorted class only |

**Current limitation:** ROC and precision-recall curves are computed for a single class (the last in sorted order). For full per-class ROC/PR analysis, use the per-class probabilities in `test_predictions.csv` to compute one-vs-rest curves externally.

### SHAP with multiclass

When `--explain` is used with multiclass:
- SHAP values are 3-dimensional: `(n_samples, n_features, n_classes)`
- `shap_summary.csv` averages across classes to produce a single global ranking
- `shap_per_row.json` also averages across classes for the top-5 feature selection
- `shap_values.csv` stores the class-averaged 2-D matrix

For per-class SHAP analysis, the raw 3-D values would need to be computed separately — the current pipeline does not save per-class SHAP matrices.

### Example

```bash
# Auto-detect multiclass
uv run train data.csv --label species

# Explicit multiclass with macro F1
uv run train data.csv --label category --problem-type multiclass --eval-metric f1_macro

# With profiling and pruning
uv run profile data.csv --label category
uv run train data.csv --label category --problem-type multiclass --prune --explain
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

## Ensemble Pruning

After training, AutoGluon may produce many models (base learners, stacked layers, weighted ensembles). Not all contribute meaningfully to the final prediction. The `--prune` flag analyzes the ensemble and removes models that:

1. Are not the best model
2. Are not in the best model's dependency chain (e.g. base models feeding a stacker)
3. Score more than 5% worse than the best model on validation data

```bash
# Train and prune in one step
uv run train data.csv --prune

# Works with all training variants
uv run train-binary data.csv --label is_fraud --prune
uv run train-regression data.csv --label price --prune
```

### Pruning Outputs

| File                     | Description                                              |
|--------------------------|----------------------------------------------------------|
| `ensemble_analysis.csv`  | Per-model scores, timing, and contribution flags.        |
| `pruning_report.json`    | Total models, pruned list, and remaining count.          |

Pruned models are deleted from disk, reducing the `AutogluonModels/` directory size and speeding up inference.

---

## Model Explainability

The `--explain` flag computes SHAP (SHapley Additive exPlanations) values after training. SHAP values show how each feature contributes to individual predictions — not just which features matter (permutation importance), but in which direction and by how much.

```bash
# Train and compute SHAP explanations
uv run train data.csv --explain

# Combine with pruning
uv run train-binary data.csv --label is_fraud --explain --prune
```

SHAP uses a KernelExplainer with up to 500 test samples (configurable). For large datasets, it subsamples automatically.

### Explainability Outputs

| File                     | Description                                              |
|--------------------------|----------------------------------------------------------|
| `shap_summary.csv`       | Mean |SHAP| per feature, ranked by importance.           |
| `shap_values.csv`        | Raw SHAP values matrix (n_samples × n_features).        |
| `shap_per_row.json`      | Top 5 contributing features per row with direction.      |
| `shap_metadata.json`     | Base values, problem type, sample count, top features.   |

### How to Read SHAP Values

- Positive SHAP value → feature pushes prediction higher (toward class 1 or higher regression value)
- Negative SHAP value → feature pushes prediction lower
- Magnitude → strength of the contribution
- `shap_summary.csv` gives the global view; `shap_per_row.json` gives per-prediction explanations

### Performance Characteristics

SHAP's KernelExplainer is model-agnostic but computationally expensive because it evaluates the prediction function many times per sample. Understanding the cost drivers helps you plan accordingly.

**What determines runtime:**

| Factor | Impact | Current Default |
|--------|--------|-----------------|
| Samples to explain | Linear — 2× samples ≈ 2× time | 500 (auto-subsampled) |
| Number of features | Exponential — KernelExplainer evaluates 2^F coalitions (approximated) | No limit |
| Background dataset size | Linear — more background samples = more accurate but slower | 100 |
| Model inference speed | Linear — every SHAP evaluation calls `predictor.predict()` or `predict_proba()` | Depends on ensemble |

**Rough timing estimates (single CPU):**

| Dataset Shape | Approximate Time |
|---------------|-----------------|
| 500 rows × 20 features | 1–5 minutes |
| 500 rows × 50 features | 5–15 minutes |
| 500 rows × 100 features | 15–45 minutes |
| 500 rows × 200+ features | 1+ hours |

These are rough estimates — actual time depends heavily on model complexity (stacked ensembles are slower than single models).

**Recommendations for large datasets:**

- Profile first (`uv run profile`) and drop low-importance or highly correlated features with `--drop` before training with `--explain`
- Prune the ensemble first (`--prune --explain`) — fewer models in the ensemble means faster SHAP inference
- For initial exploration, train with `--preset good_quality` instead of `best` to get a simpler ensemble
- If SHAP is too slow, permutation-based feature importance (`feature_importance.csv`) is always generated and runs much faster — it shows which features matter but not the direction of contribution

**Memory usage:**

The SHAP values matrix is `n_samples × n_features` of float64. For 500 samples × 100 features, that's about 400 KB — negligible. The memory bottleneck is the KernelExplainer holding the background dataset and intermediate coalition evaluations in memory, which scales with feature count.

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

## Experiment Tracking

Every training run automatically records its parameters, metrics, and output path to a local JSONL file (`experiments.jsonl`). This provides a lightweight experiment log without external services.

### How It Works

After each `train`, `train-binary`, or `train-regression` run completes, the pipeline appends a JSON entry containing:
- Timestamp (UTC)
- All CLI parameters (csv, label, preset, problem_type, eval_metric, etc.)
- Test-set metrics and best model name
- Output directory path
- Contents of `model_info.json`

### Comparing Experiments

```bash
uv run experiments [OPTIONS]
```

### Options

| Flag       | Default              | Description                                |
|------------|----------------------|--------------------------------------------|
| `--log`    | `experiments.jsonl`  | Path to the experiment log file            |
| `--last`   | all                  | Show only the last N experiments           |
| `--output` | stdout               | Save comparison to CSV                     |

### Examples

```bash
# View all experiments side by side
uv run experiments

# Last 3 experiments
uv run experiments --last 3

# Export to CSV
uv run experiments --output comparison.csv
```

### Log Format

Each line in `experiments.jsonl` is a self-contained JSON object:

```json
{
  "timestamp": "2026-03-27T23:30:00+00:00",
  "output_dir": "output/train_20260327_233000",
  "params": {
    "csv": "data.csv",
    "label": "target",
    "problem_type": "binary",
    "preset": "best",
    "eval_metric": "f1"
  },
  "metrics": {
    "best_model": "WeightedEnsemble_L2",
    "best_test_score": 0.87
  },
  "model_info": { ... }
}
```

The file is append-only and never overwritten. Delete it to reset the experiment history.

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
| `test_prune.py`                    | `evaluate/prune.py`                  | Ensemble analysis, pruning recommendations, model deletion, dependency collection              |
| `test_explain.py`                  | `evaluate/explain.py`                | SHAP summary, per-row explanations, artifact generation, multiclass handling                   |
| `test_profile.py`                  | `profile.py`                         | Correlation matrix, pair detection, drop recommendations, heatmap, report generation           |
