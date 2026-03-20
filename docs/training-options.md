# Training Options

This project uses [uv](https://docs.astral.sh/uv/) for dependency management and script execution. All CLI commands are defined as `[project.scripts]` in `pyproject.toml` and should be invoked via `uv run`:

```bash
uv run <command> [ARGS]
```

`uv run` resolves the project's virtual environment and dependencies automatically — no manual `pip install` or `source .venv/bin/activate` needed.

## Entry Points

| Command                              | Problem Type   | Default Eval Metric          | Use When                                      |
|--------------------------------------|----------------|------------------------------|-----------------------------------------------|
| `uv run train`                       | Auto-detect    | Auto-detect                  | You want full control or have a non-standard task |
| `uv run train-binary`               | Binary         | `f1`                         | Your target column has exactly two classes     |
| `uv run train-regression`           | Regression     | `root_mean_squared_error`    | Your target column is continuous               |
| `uv run predict`                     | Auto-detect    | —                            | Run inference with a trained model             |
| `uv run predict-binary`             | Binary         | —                            | Run inference (binary convenience alias)       |
| `uv run predict-regression`         | Regression     | —                            | Run inference (regression convenience alias)   |

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

## Output Artifacts

All commands write the following to `--output-dir`:

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
