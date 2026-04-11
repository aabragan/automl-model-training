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

AutoML Model Training is a high-performance training and prediction pipeline built on **AutoGluon**.

*   **Point and Shoot:** Just provide a CSV, and the system handles the rest.
*   **Ensemble Power:** Automatic stacking, bagging, and model selection.
*   **Tabular Foundation Models:** Cutting-edge models like TabPFNv2, TabICL, and TabDPT.
*   **Agentic Future:** LLM-driven autonomous iteration and reasoning.

---

## 🛠️ Key Features

- **Auto-Detection:** Detects problem types (Binary, Multiclass, Regression, Quantile).
- **Extreme Precision:** Leverages Tabular Foundation Models for <100K samples.
- **Smart Analysis:** Flags overfitting, imbalance, and low-value features.
- **Explainability:** Integrated SHAP-based global and local feature importance.
- **Ollama Agent:** Local LLM reasoning to autonomously drive improvements.

---

## 📋 Requirements & Setup

Managing dependencies with **uv** for speed and isolation.

```bash
# Clone and enter
git clone <repo-url>
cd automl-model-training

# Install dependencies (automatic venv)
uv sync

# Optional: GPU-accelerated Foundation Models
uv sync --extra extreme
```

---

## ⚡ Quick Start: Training

Training is as simple as one command.

```bash
# Auto-detect everything
uv run train data.csv

# Target specific metric with time limit
uv run train-binary data.csv --label is_fraud --time-limit 120

# Regression optimized for RMSE
uv run train-regression data.csv --label price
```

---

## 🤖 Autonomous LLM Agent (Ollama)

Run a local LLM to autonomously drive the full training pipeline.

```bash
# Install and pull model
ollama pull qwen2.5:14b
ollama serve

# Let the LLM iterate to find the best model
uv run agent-ollama data.csv --label target
```

- **Reasoning:** Agent interprets `analysis.json` findings.
- **Decisions:** Agent adjusts presets, drop lists, and metrics.
- **Transparency:** Agent summarizes exactly *why* it made its changes.

---

## 🧩 Agent Tools (`tools.py`)

Exposes the pipeline as JSON-serializable functions for any LLM framework.

| Function | Purpose |
| :--- | :--- |
| `tool_profile` | Analyze dataset and get drop recommendations |
| `tool_train` | Returns scores, findings, and importance-based drops |
| `tool_predict` | Run inference on new data |
| `tool_compare` | Compare run history to track progress |

**Compatible with:** Bedrock Agents, LangChain, OpenAI Function Calling.

---

## 📊 Dataset Profiling

Before training, understand your data quality.

```bash
uv run profile data.csv --label price --threshold 0.85
```

| Output Artifact | Description |
| :--- | :--- |
| `missing_values.csv` | Rates and counts per column |
| `correlation_matrix.csv` | Pearson correlation metrics |
| `correlation_heatmap.png` | Visual feature relationships |
| `profile_report.json` | Actionable drop recommendations |

---

## 🔍 Post-Training Analysis

Every run generates a deep-dive report (`analysis_report.txt`).

- **Overfitting Check:** Gaps between validation and test scores.
- **Feature Importance:** Permutation-based ranking.
- **Model Diversity:** Ensuring the ensemble isn't monolithic.
- **Ensemble Pruning:** Identification and removal of underperforming models.
- **Improvement Tips:** Specific, actionable recommendations.

---

## 🔮 Prediction & Drift

Running inference is robust and safe.

```bash
# Basic inference
uv run predict new_data.csv --model-dir output/train_run/AutogluonModels

# Flag low-confidence predictions
uv run predict new_data.csv --model-dir ... --min-confidence 0.7

# With data drift check
uv run predict new_data.csv --model-dir ... --drift-check output/train_run
```

- **Drift Detection:** PSI-based feature shift analysis.
- **Confidence Flagging:** Mark low-probability predictions for review.

---

## 📈 Backtesting & Experiments

Validate your models against history and compare runs.

```bash
# Temporal walk-forward backtest (3 splits)
uv run backtest data.csv --date-column date --n-splits 3 --label price

# Compare all experiments side-by-side
uv run experiments

# Side-by-side model run comparison (Diff)
uv run compare output/run_A output/run_B
```

---

## 🏗️ Project Architecture

```
src/automl_model_training/
├── agent.py         # Autonomous iterative logic
├── ollama_agent.py  # LLM-driven reasoning agent
├── tools.py         # LLM tool layer implementations
├── train.py         # Core trainer & CLI entry points
├── predict.py       # Inference & drift engine
├── profile.py       # Data quality analyzer
└── evaluate/        # Specialized artifact generators
    ├── analyze.py      # Recommendation engine
    └── ...
```

---

## 🧪 Development & Quality

Built for reliability and maintainability.

- **Tests:** `uv run pytest tests/` (90%+ coverage, mocked for speed).
- **Linting:** `uv run ruff check` for clean code.
- **Typing:** `uv run mypy src/` for static safety.
- **CI/CD:** Automated pipelines for every PR (tests, lint, types).

---

<!-- _class: lead -->

# Ready to Train?

Modern AutoML. Autonomous. Agentic.

[GitHub Repository](https://github.com/aabragan/automl-model-training)
