# NLI Partial Input Reliance Diagnostic System 🔍

A modular, end-to-end Python pipeline for evaluating Natural Language Inference (NLI) models under partial-input constraints. This system calculates the **Partial Input Reliance Index (PIRI)** to quantify how much state-of-the-art transformer models rely on dataset artifacts (hypothesis bias) and syntactic heuristics.

**Author:** Uddhav Khetan 
**Registration Number:** 22BCT0153
**Institution:** Vellore Institute of Technology (VIT)  

---

## 📖 Project Overview

Modern NLI models often achieve high accuracy not by learning true semantic entailment, but by exploiting crowd-worker artifacts or syntactic overlaps. This diagnostic system forces models to predict under three strict conditions:
1. **Full Input:** Premise + Hypothesis
2. **Hypothesis-Only:** [Empty String] + Hypothesis
3. **Premise-Only:** Premise + [Empty String]

By comparing the accuracy drop-off, the system calculates **PIRI**. A $PIRI_{hyp}$ score near 1.0 indicates severe reliance on the premise (no hypothesis bias), while a score near 0.0 indicates total reliance on hypothesis-only shortcuts.

### Supported Datasets & Models
* **Datasets:** SNLI, MNLI (Matched), HANS, ANLI, SICK, XNLI.
* **Models:** RoBERTa, DeBERTa-v3, DistilRoBERTa, DistilBERT, and custom PIRI-regularized models.

---

## 🗂️ Repository Structure

```text
├── config.py              # Central registry for models and datasets
├── data.py                # Hugging Face dataset loaders and normalizers
├── models.py              # Wrapper classes and eager-attention extraction
├── run_diagnostics.py     # Core 3-condition inference engine 
├── analyze.py             # Calculates PIRI and aggregates summary.csv
├── plots.py               # Generates cross-dataset Altair/Matplotlib charts
├── dashboard.py           # Streamlit UI and Live Interpretability Probe
├── run_all.sh             # Bash script to execute the full pipeline
├── tests/                 # Pytest suite for pipeline validation
└── results/               # Generated CSV logs and summary metrics
```

---

## ⚙️ Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/yourusername/nli-piri-diagnostics.git](https://github.com/yourusername/nli-piri-diagnostics.git)
   cd nli-piri-diagnostics
   ```

2. **Install dependencies:**
   Ensure you have Python 3.8+ installed. 
   ```bash
   pip install torch torchvision torchaudio
   pip install transformers datasets pandas numpy streamlit altair pytest
   ```
   *(Note: For Apple Silicon (M-series), ensure you install the MPS-compatible PyTorch build).*

---

## 🚀 Usage Guide

You can run the entire pipeline end-to-end using the provided shell script, or execute the modules step-by-step.

### Step 1: Run Inference
Evaluate a specific model and dataset under the 3 masking conditions:
```bash
python run_diagnostics.py --model roberta --dataset snli --samples 1000 --seed 42
```

### Step 2: Calculate PIRI Metrics
Aggregate the raw `.csv` logs from the `results/` folder into a unified `summary.csv`:
```bash
python analyze.py
```

### Step 3: Generate Visualizations
Generate the $PIRI_{hyp}$ vs. HANS scatter plots and accuracy bar charts:
```bash
python plots.py
```

### Step 4: Launch the Dashboard
Start the interactive Streamlit UI to explore metrics and run the Live Interpretability Probe:
```bash
streamlit run dashboard.py
```

---

## 🔬 Live Interpretability Probe

The Streamlit dashboard includes a **Live Inference** tab. By forcing Hugging Face's `attn_implementation="eager"`, the system extracts final-layer `[CLS]` token attention weights and renders a dynamic HTML heatmap over the text. This allows researchers to visually inspect which tokens the model relies on when making entailment decisions.

***