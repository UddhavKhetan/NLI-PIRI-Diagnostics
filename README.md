# NLI-PIRI-Diagnostics: Quantifying and Mitigating Input Over-Reliance in NLI

Welcome to the **NLI-PIRI-Diagnostics** repository. This project provides a unified, architecture-agnostic pipeline to evaluate, quantify, and mitigate dataset biases and heuristic over-reliance in Natural Language Inference (NLI) models.

---

## 1. Project Overview

### What is NLI?
Natural Language Inference (NLI) is a foundational NLP task where a model determines the relationship between two sentences: a **Premise** and a **Hypothesis**. The model must classify whether the premise *entails*, *contradicts*, or is *neutral* toward the hypothesis. 

### The Problem: Hypothesis-Only Bias
Many NLI datasets contain statistical artifacts (e.g., negation words heavily correlating with contradiction). As a result, models often "cheat" by looking *only* at the hypothesis to make a prediction, bypassing the actual cross-sentence reasoning the task requires. 

### What is PIRI?
To measure this, we utilize the **Partial Input Reliance Index (PIRI)**. PIRI intuitively quantifies how much a model's performance degrades when it is deprived of the full context. If a model still achieves high accuracy when seeing *only* the hypothesis, it is highly biased. PIRI calculates the relative drop in accuracy between a full-input baseline and a partial-input condition. 

### Pipeline Features & Key Contributions
*   **Architecture-Agnostic Metric:** Standardized computation of PIRI and a mathematically rigorous **Chance-Corrected PIRI** that accounts for random-guessing baselines.
*   **Triple-Condition Ablation:** Automated evaluation across three conditions: Full Input, Premise-Only, and Hypothesis-Only.
*   **Unified Evaluation Framework:** Seamless testing across diverse datasets (SNLI, MNLI, HANS, ANLI, XNLI, SICK) and models (RoBERTa, DeBERTa, DistilBERT, BART, Flan-T5).
*   **Mitigation Training:** An experimental training loop utilizing KL-divergence regularization to actively unlearn hypothesis-only biases during fine-tuning.
*   **Interpretability Dashboard:** A live, interactive Streamlit frontend for probing model decisions using Attention heatmaps and Integrated Gradients.

---

## 2. Repository Structure

This repository uses a flat, modular file structure in the root directory for straightforward execution and import management.

| File / Directory | Description |
| :--- | :--- |
| `config.py` | Centralized registries for supported models, datasets, and ablation strategies. |
| `data.py` / `models.py` | Core utilities for loading HF datasets, tokenization, and model wrapping. |
| `ablation.py` | Implements the logic for input ablations (e.g., masking, emptying, or replacing text with neutral templates). |
| `metrics.py` | Houses the mathematical formulations for standard accuracy, Macro F1, original PIRI, and Chance-Corrected PIRI. |
| `error_analysis.py` | Unoptimized, from-scratch implementations of confusion matrices and per-label accuracy tracking for maximum transparency. |
| `stats_utils.py` | Implements rigorous statistical tests (Paired Bootstrap, McNemar's, Permutation tests) to measure the significance of accuracy differences. |
| `full_eval_bruteforce.py` | CLI entry point for exhaustive, multi-seed evaluation sweeps across all models and datasets. |
| `multilingual_eval_bruteforce.py` | Specialized evaluation script for sweeping across multiple languages in the XNLI dataset. |
| `mitigation_train.py` | Training script that implements dual-forward-pass KL-divergence regularization to reduce model bias. |
| `interpretability.py` | Backend logic for gradient-based attribution (Integrated Gradients using Captum). |
| `dashboard.py` | The Streamlit web application for interactive, live model probing. |
| `results/` | Directory where all generated CSVs, JSON metrics, and confusion matrices are saved. |
| `images/` | Directory for storing generated plots and visualizations. |

---

## 3. Installation and Setup

### Prerequisites
*   Python 3.8+
*   A machine with a CUDA-enabled GPU is *highly recommended* for running mitigation training and large-model evaluations.

### Step-by-Step Setup
1. **Clone the repository:**
   ```bash
   git clone [https://github.com/UddhavKhetan/NLI-PIRI-Diagnostics.git](https://github.com/UddhavKhetan/NLI-PIRI-Diagnostics.git)
   cd NLI-PIRI-Diagnostics

```

2. **Create and activate a virtual environment:**
```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate

```


3. **Install dependencies:**
```bash
pip install -r requirements.txt

```


4. **Hugging Face Authentication (Optional):**
If you plan to use gated models (like LLaMA variants in the future), ensure you have authenticated via the Hugging Face CLI:
```bash
huggingface-cli login

```



---

## 4. Basic Usage: Quick Start

To run a quick "Hello World" diagnostic check, you can execute the brute-force script and limit it to a single model and dataset by editing the targets or passing arguments (if configured in your CLI).

By default, executing the script will run evaluations based on the internal configuration arrays:

```bash
python full_eval_bruteforce.py

```

### Where to find the outputs

Upon completion, navigate to the `results/bruteforce_eval/` directory. You will find:

* `{model}_{dataset}_seed{seed}_results.json`: Contains overall accuracies, PIRI, Chance-Corrected PIRI, and Macro F1 scores.
* `{model}_{dataset}_seed{seed}_{condition}_cm.csv`: A raw, label-by-label confusion matrix for deep error analysis.

---

## 5. Full Brute-Force Evaluation

The core philosophy of the brute-force scripts is **transparency over optimization**. To avoid hidden caching bugs, the script explicitly re-loads models, re-tokenizes datasets, and computes outputs from scratch for every combination of Model, Dataset, and Seed.

To run a massive sweep across all configurations (e.g., RoBERTa, DeBERTa, BART, DistilBERT on SNLI, MNLI, HANS, SICK across 10 seeds), simply run:

```bash
python full_eval_bruteforce.py

```

> **⚠️ Runtime Note:** Because this script intentionally avoids caching to ensure pure, isolated experimental runs, executing all models across all datasets for 10+ seeds can take **several hours to days** depending on your GPU. You can limit the scope by commenting out models/datasets in the script's global lists.

---

## 6. Mitigation Experiments

The pipeline includes a specialized training framework to actively "unlearn" biases.

**How it works:**
During training, the model performs two forward passes:

1. A standard pass with the **Full Input** (Premise + Hypothesis).
2. A partial pass with the **Hypothesis-Only** (Premise removed).

We calculate the Cross-Entropy loss for the full input. We then calculate the **KL-divergence** between the full-input distribution and the hypothesis-only distribution. By adding this KL-divergence as a penalty (weighted by hyperparameter $\alpha$), we force the model to rely *less* on the hypothesis-only signal.

**Running Mitigation:**
We provide presets for larger models (RoBERTa, DeBERTa, BART-Large, Flan-T5).

```bash
# Example: Mitigating hypothesis bias in RoBERTa
python mitigation_train.py --model roberta

# Example: Mitigating bias in an encoder-decoder (Flan-T5)
python mitigation_train.py --model flan_t5_small

```

Post-mitigation models can be plugged back into `full_eval_bruteforce.py` to compare the new PIRI scores against the baseline.

---

## 7. Interpretability Dashboard

We provide a Streamlit dashboard to visually inspect *why* a model is making a specific prediction.

**Launch the dashboard:**

```bash
streamlit run dashboard.py

```

**UI Features:**

* **Model Selection:** Choose from the loaded Hugging Face models.
* **Live Inference:** Type your own custom Premise and Hypothesis to see live predictions and confidence scores.
* **Attribution Methods:**
* **Attention Heatmaps:** Visualizes the raw attention weights from the [CLS] token in the final layer.
* **Integrated Gradients:** A robust, gradient-based attribution method (powered by Captum) that highlights exactly which tokens positively or negatively contributed to a specific class prediction.



> **Caveat:** Attention heatmaps are purely illustrative. Integrated Gradients provide a stronger mathematical attribution signal, but neither should be interpreted as strict, human-like causal reasoning.

---

## 8. Error Analysis & Statistics

Raw accuracy metrics can be deceiving. The pipeline automatically performs deep error analysis.

* **Confusion Matrices:** Generated automatically during evaluation. Rows represent the *True Label* and columns represent the *Predicted Label*. A heavy skew in specific columns during hypothesis-only evaluation often indicates a learned dataset artifact (e.g., always guessing "Contradiction").
* **Statistical Tests:** The `stats_utils.py` module includes functions for:
* *Paired Bootstrap Tests:* To establish 95% confidence intervals around accuracy drops.
* *McNemar’s Test:* To evaluate if the predictions of a baseline model vs. a mitigated model are significantly different.
* *Permutation Tests:* To establish exact p-values over accuracy differences.



---

## 9. Multilingual XNLI and Larger Models

To evaluate whether English-centric artifacts transfer to other languages, we include a brute-force script for the **XNLI** (Cross-lingual NLI) dataset.

Supported languages currently include English (`en`), French (`fr`), Spanish (`es`), German (`de`), and Chinese (`zh`).

```bash
python multilingual_eval_bruteforce.py

```

**Generative LLMs (Flan-T5/BART):**
Evaluating generative encoder-decoder models requires special handling for text generation vs. classification heads. The pipeline wraps these appropriately, but be aware that running multi-seed sweeps on models like `flan-t5` or `bart-large-mnli` dramatically increases VRAM usage and runtime.

---

## 10. Reproducing Key Results

To reproduce the full suite of experiments for a research manuscript, follow this recipe:

1. **Baseline Generation:** Run `python full_eval_bruteforce.py` to establish the baseline Accuracy and PIRI scores for all standard datasets.
2. **Cross-Lingual Check:** Run `python multilingual_eval_bruteforce.py` to generate the XNLI robustness tables.
3. **Mitigation:** Run `python mitigation_train.py --model [target_model]` for your model of choice.
4. **Post-Mitigation Eval:** Temporarily point the models in `config.py` to your newly saved local checkpoints and re-run step 1.
5. **Statistical Validation:** Use `stats_utils.py` on the resulting CSVs to generate p-values for your manuscript.

---

## 11. Limitations and Notes

* **Mathematical Assumptions:** The Chance-Corrected PIRI assumes a uniform class distribution and a chance baseline of $1/3$ (for 3-class NLI tasks like SNLI/MNLI) or $1/2$ (for HANS). If a model performs at or below random guessing on full inputs, PIRI mathematically breaks down and is clamped to $0.0$.
* **Resource Constraints:** Training and evaluating large models (DeBERTa-v3-Large, Flan-T5) with KL-divergence dual-forward passes requires significant GPU memory (16GB+ VRAM recommended).
* **Scope:** While mitigation shows promise in aligning model behavior, these results reflect a limited subset of architectures. They do not definitively prove the existence of a universal "robustness tax" across all LLM paradigms.

---

## 12. Key References & Inspirations

This framework builds upon the foundational work of several researchers in the field of NLI artifacts and adversarial robustness:

* **Gururangan et al. (2018):** Annotation Artifacts in Natural Language Inference (Hypothesis-only baselines).
* **Poliak et al. (2018):** Hypothesis Only Baselines in NLI.
* **McCoy et al. (2019):** Right for the Wrong Reasons (HANS dataset and lexical overlap heuristics).
* **Ribeiro et al. (2020):** Beyond Accuracy: Behavioral Testing of NLP Models with CheckList.
* **Wang et al. (2018/2019):** GLUE and SuperGLUE benchmark suites.
* **Nie et al. (2020):** Adversarial NLI (ANLI).

```

```