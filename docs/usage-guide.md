# Usage Guide

A step-by-step walkthrough of the full AutoML training pipeline — from raw CSV to production predictions.

## The Model Training Lifecycle

Every ML project follows the same cycle: understand the data, train a model, evaluate it, iterate until satisfied, deploy to production, and monitor for degradation. This pipeline automates each step.

```
Profile → Train → Evaluate → Iterate → Predict → Monitor
```

| Step     | Command                 | Why                                                |
| -------- | ----------------------- | -------------------------------------------------- |
| Profile  | `uv run profile`        | Understand data quality before training            |
| Train    | `uv run train`          | Build the best ensemble model                      |
| Evaluate | Review output artifacts | Check accuracy, overfitting, feature importance    |
| Iterate  | Adjust params, retrain  | Improve based on analysis recommendations          |
| Validate | `uv run backtest`       | Test on temporal data to simulate production       |
| Compare  | `uv run compare`        | Pick the best run from multiple experiments        |
| Predict  | `uv run predict`        | Run inference on new data                          |
| Monitor  | `--drift-check`         | Detect when production data diverges from training |

## Step 1: Setup

```bash
git clone <repo-url>
cd automl-model-training
uv sync
```

## Step 2: Profile Your Dataset

**Goal:** Identify data quality issues, redundant features, and class imbalance before spending time on training.

```bash
uv run profile data.csv --label target
```

Review the output:

- `profile_report.json` — drop recommendations, missing value flags, outlier counts
- `correlation_heatmap.png` — visual overview of feature relationships
- The CLI prints a `--drop` flag you can paste directly into your train command

## Step 3: Train the Model

**Goal:** Build the best ensemble model for your problem type.

```bash
# Auto-detect problem type
uv run train data.csv

# Or use convenience wrappers
uv run train-binary data.csv --label is_fraud
uv run train-regression data.csv --label price
```

**Integrated profiling** — skip the separate profile step:

```bash
uv run train data.csv --profile --label price
```

**Cross-validation** — get more reliable accuracy estimates:

```bash
uv run train data.csv --cv-folds 5
```

**Reproducibility** — verify results with different seeds:

```bash
uv run train data.csv --seed 42
uv run train data.csv --seed 123
```

**Full-featured run:**

```bash
uv run train data.csv --label target --profile --cv-folds 5 --prune --explain
```

## Step 4: Review Training Results

Check the key files in the timestamped output directory:

1. `model_info.json` — confirms problem type, eval metric, best model
2. `leaderboard_test.csv` — test-set scores for every model
3. `feature_importance.csv` — which features matter most
4. `analysis.json` — automated findings and actionable recommendations
5. `cv_summary.json` — cross-validation mean ± std (if `--cv-folds` was used)

The analysis report flags overfitting, class imbalance, low-value features, and dataset size issues — each with a specific recommendation.

## Step 5: Iterate

| Issue                            | Action                                                      |
| -------------------------------- | ----------------------------------------------------------- |
| Overfitting (val >> test score)  | Increase data, reduce features, try `--preset high_quality` |
| Low accuracy                     | Add features, increase `--time-limit`, check data quality   |
| Class imbalance flagged          | Use `--eval-metric f1_macro` or `balanced_accuracy`         |
| Too many low-importance features | Use `--drop` or `--profile` to auto-remove them             |
| Ensemble too large               | Add `--prune`                                               |
| Need more reliable estimates     | Add `--cv-folds 5`                                          |

Each run creates a new timestamped directory — previous results are preserved.

## Step 6: Validate with Backtesting

**Goal:** For time-dependent data, test how the model performs on future data rather than random splits.

```bash
# Single cutoff
uv run backtest data.csv --date-column date --cutoff 2025-06-01 --label price

# Walk-forward with 3 folds
uv run backtest data.csv --date-column date --n-splits 3 --label churn
```

## Step 7: Compare Runs

**Goal:** After multiple experiments, pick the best one objectively.

```bash
# Compare two runs side by side
uv run compare output/train_20260321_120530 output/train_20260322_090000

# Or use the experiment log
uv run experiments --last 5
```

The `compare` command shows test scores, model families, feature counts, training times, and CV results in a single table.

## Step 8: Run Predictions

**Goal:** Apply the trained model to new data.

```bash
uv run predict new_data.csv --model-dir output/train_<ts>/AutogluonModels
```

**Flag uncertain predictions** for human review:

```bash
uv run predict new_data.csv --model-dir output/train_<ts>/AutogluonModels --min-confidence 0.7
```

**Check for data drift** against the training distribution:

```bash
uv run predict new_data.csv \
  --model-dir output/train_<ts>/AutogluonModels \
  --drift-check output/train_<ts>
```

Drift detection uses Population Stability Index (PSI) to flag features whose distributions have shifted since training. Significant drift (PSI > 0.25) means the model may be unreliable on this data.

## Complete Example

```bash
# 1. Setup
uv sync

# 2. Profile + train with CV, pruning, and explainability
uv run train-binary fraud_data.csv --label is_fraud \
    --profile --cv-folds 5 --prune --explain

# 3. Review results
cat output/train_<ts>/analysis_report.txt
cat output/train_<ts>/cv_summary.json

# 4. Backtest for temporal validation
uv run backtest fraud_data.csv --date-column date --n-splits 3 --label is_fraud

# 5. Compare all runs
uv run compare output/train_<ts1> output/train_<ts2>

# 6. Predict on new data with drift check and confidence filtering
uv run predict new_transactions.csv \
    --model-dir output/train_<ts>/AutogluonModels \
    --drift-check output/train_<ts> \
    --min-confidence 0.7

# 7. Review experiment history
uv run experiments
```

## Verbosity Control

All commands support `--verbose` / `--quiet`:

```bash
uv run train data.csv --verbose    # DEBUG level
uv run train data.csv              # INFO level (default)
uv run train data.csv --quiet      # WARNING level
```
