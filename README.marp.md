---
marp: true
theme: gaia
_class: lead
paginate: true
backgroundColor: #0c0c0c
color: #e0e0e0
style: |
  section {
    font-family: 'Inter', 'Segoe UI', Roboto, sans-serif;
    font-size: 28px;
    letter-spacing: -0.01em;
  }
  h1, h2, h3 {
    color: #ffffff;
    text-shadow: 0 2px 10px rgba(0,0,0,0.5);
  }
  h1 { font-size: 60px; margin-bottom: 20px; }
  h2 { font-size: 45px; border-bottom: 2px solid #3d5afe; padding-bottom: 10px; margin-bottom: 40px; }
  strong { color: #3d5afe; }
  code {
    background: #1e1e1e;
    color: #82aaff;
    border-radius: 6px;
    padding: 2px 8px;
  }
  pre {
    background: #121212 !important;
    border: 1px solid #333;
    border-radius: 12px;
    box-shadow: 0 10px 30px rgba(0,0,0,0.5);
  }
  .columns {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 2rem;
  }
  .lead h1 {
    background: linear-gradient(135deg, #3d5afe 0%, #ff4081 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 80px;
    font-weight: 800;
  }
  .tagline {
    font-size: 32px;
    color: #b0b0b0;
    margin-top: -20px;
  }

---

<!-- _backgroundColor: #000 -->
<!-- _backgroundImage: "linear-gradient(rgba(0,0,0,0.7), rgba(0,0,0,0.7)), url('./automl_header_1775762842851.png')" -->
<!-- _class: lead -->
<!-- _footer: "" -->
<!-- _header: "" -->
<style scoped>
  section {
    justify-content: flex-end;
    align-items: flex-start;
    padding-bottom: 80px;
    padding-left: 80px;
  }
</style>

# AutoML Training Pipeline

<div class="tagline">An Intelligent, Autonomous Workflow to build State-of-the-Art Models</div>

---

## 🚀 The Vision

AutoML Model Training is a high-performance pipeline built on **AutoGluon**.

- **Point and Shoot:** Provide a CSV — the system handles the rest
- **Ensemble Power:** Automatic stacking, bagging, and model selection
- **Tabular Foundation Models:** TabPFNv2, TabICL, Mitra, TabDPT, TabM
- **Reproducible:** Every run is timestamped, seeded, and logged
- **Agentic:** LLM-driven autonomous iteration and reasoning

---

## 🛠️ Full Feature Set

<div class="columns">
<div>

**Training**
- Auto-detect problem type
- Cross-validation (k-fold)
- Ensemble pruning
- SHAP explainability
- Decision threshold calibration
- Auto-drop low-importance features
- Integrated dataset profiling

</div>
<div>

**Production**
- Confidence filtering
- PSI-based drift detection
- Temporal backtesting
- Experiment tracking (JSONL)
- Model comparison reports
- Autonomous training agent
- Ollama LLM agent

</div>
</div>

---

## 📋 Setup

Managing dependencies with **uv** for speed and isolation.

```bash
# Clone and install
git clone <repo-url>
cd automl-model-training
uv sync

# Optional: GPU-accelerated Foundation Models
uv sync --extra extreme
```

Requires **Python ≥ 3.12**. No manual `pip install` or venv activation needed.

---

## ⚡ Quick Start

```bash
# Auto-detect problem type and metric
uv run train data.csv

# Binary classification (defaults to F1)
uv run train-binary data.csv --label is_fraud --time-limit 120

# Regression (defaults to RMSE)
uv run train-regression data.csv --label price

# Profile + train in one step
uv run train data.csv --profile --label price

# 5-fold cross-validation then final run
uv run train data.csv --cv-folds 5
```

---

## 📊 Dataset Profiling

Understand data quality before spending time on training.

```bash
uv run profile data.csv --label price --threshold 0.85
```

| Output | Description |
| :--- | :--- |
| `missing_values.csv` | Per-column missing rates |
| `correlation_matrix.csv` | Pearson correlation matrix |
| `correlation_heatmap.png` | Visual feature relationships |
| `profile_report.json` | Drop recommendations + full summary |

The CLI prints a ready-to-use `--drop` flag to paste into your train command.

---

## 🔍 Post-Training Analysis

Every run generates `analysis.json` + `analysis_report.txt` automatically.

- **Overfitting:** Val vs test score gap (flags >5% moderate, >10% severe)
- **Feature Importance:** Permutation-based ranking — flags near-zero and negative
- **Auto-drop:** `--auto-drop` trains once, drops bad features, retrains
- **Model Diversity:** Warns if top models are all from the same family
- **Class Imbalance:** Detects skewed distributions and suggests metrics
- **Dataset Size:** Flags low sample-to-feature ratios

---

## 🎯 Training Options

```bash
# Prune underperforming ensemble models
uv run train data.csv --prune

# SHAP explainability
uv run train data.csv --explain

# Calibrate binary decision threshold for F1
uv run train-binary data.csv --calibrate-threshold f1

# Auto-drop low-importance features and retrain
uv run train data.csv --auto-drop

# Reproducibility — verify with different seeds
uv run train data.csv --seed 42
uv run train data.csv --seed 123
```

---

## 🔮 Prediction & Safety

```bash
# Basic inference
uv run predict new_data.csv \
  --model-dir output/train_run/AutogluonModels

# Flag low-confidence predictions for human review
uv run predict new_data.csv \
  --model-dir output/train_run/AutogluonModels \
  --min-confidence 0.7

# Check for data drift against training distribution
uv run predict new_data.csv \
  --model-dir output/train_run/AutogluonModels \
  --drift-check output/train_run
```

---

## 📉 Data Drift Detection

**Why it matters:** Models degrade silently when production data shifts from training data.

Uses **Population Stability Index (PSI)** — a standard metric from credit risk modeling.

| PSI | Status |
| :--- | :--- |
| < 0.1 | No significant drift |
| 0.1 – 0.25 | Moderate drift — monitor |
| > 0.25 | **Significant drift — model may be unreliable** |

Produces `drift_report.json` + `drift_report.csv` with per-feature scores.

---

## 📈 Backtesting & Experiments

```bash
# Walk-forward temporal backtest (3 folds)
uv run backtest data.csv \
  --date-column date --n-splits 3 --label price

# Compare all experiments side-by-side
uv run experiments --last 5

# Side-by-side model run comparison
uv run compare output/run_A output/run_B --output results/diff
```

Every training run auto-logs to `experiments.jsonl` — no setup required.

---

## 🤖 Autonomous Training Agent

Automates the full loop: **profile → train → analyze → adjust → repeat**.

```bash
# Stop when F1 ≥ 0.90 or after 5 iterations
uv run agent-binary data.csv \
  --label is_fraud --target-f1 0.90 --max-iterations 5

# Stop when RMSE ≤ 5.0 or after 3 iterations
uv run agent-regression data.csv \
  --label price --target-rmse 5.0 --max-iterations 3
```

Cycles through `best_quality → best_v150 → high_quality` presets. Drops low-importance features between iterations. Records every run to `experiments.jsonl`.

---

## 🧠 Ollama LLM Agent

Drive the full pipeline via a **local LLM** using tool-calling.

```bash
# Pull a model and start Ollama
ollama pull qwen2.5:14b
ollama serve

# Run the conversational training agent
uv run python -m automl_model_training.ollama_agent \
  data.csv --label target --model qwen2.5:14b
```

The agent reads `analysis.json`, reasons about findings, and decides what to change next — adjusting presets, drop lists, and eval metrics autonomously.

---

## 🧩 LLM Tool Layer (`tools.py`)

Exposes the full pipeline as JSON-serializable functions for any LLM framework.

| Tool | Purpose |
| :--- | :--- |
| `tool_profile` | Analyze dataset, get drop recommendations |
| `tool_train` | Train model, return scores + findings |
| `tool_predict` | Run inference on new data |
| `tool_read_analysis` | Re-read analysis without retraining |
| `tool_compare_runs` | Compare experiment history |

**Compatible with:** Bedrock Agents, LangChain, OpenAI function calling.

---

## 🏗️ Project Architecture

```
src/automl_model_training/
├── config.py          # Shared defaults, thresholds, logging
├── train.py           # Training, cross-validation, CLI
├── predict.py         # Inference, confidence filtering, drift
├── drift.py           # PSI-based drift detection
├── profile.py         # Dataset profiling
├── compare.py         # Run comparison
├── experiment.py      # Experiment tracking
├── agent.py           # Autonomous training agent
├── ollama_agent.py    # Ollama LLM agent
├── tools.py           # LLM tool layer
└── evaluate/          # Artifact generators (analyze, prune, explain, ...)
```

---

## 🧪 Quality & Testing

Built for reliability and maintainability.

- **217 tests** — mocked AutoGluon predictors, runs in seconds
- **84%+ coverage** — core logic fully tested
- **ruff** — linting + formatting (`line-length = 100`, rules E,F,I,W,UP,B,SIM)
- **mypy** — static type checking across all 21 source files
- **CI/CD** — automated pipelines on every PR (tests, lint, types, coverage)

```bash
uv run python -m pytest tests/ -v
uv run ruff check src/ tests/
uv run mypy src/
```

---

<!-- _class: lead -->

# Ready to Train?

Modern AutoML. Reproducible. Agentic.

[GitHub Repository](https://github.com/aabragan/automl-model-training)
