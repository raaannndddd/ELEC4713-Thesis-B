# ELEC4713 Thesis B — Medical Chatbot Bias Analysis

**ELEC4713 — Honours Thesis B, University of Sydney**

Automated bias analysis of two consumer medical AI chatbots — **Doctronic** and **DrKhan** — across demographic variables (race, gender, age) and clinical features.

---

## Overview

This project investigates whether the responses of Doctronic and DrKhan differ systematically across patient demographics when presented with matched clinical scenarios. It consists of two phases:

1. **Data collection** — Automated browser sessions send standardised patient prompts to both chatbots and record their responses.
2. **Bias analysis** — Three complementary analyses characterise and test for demographic differences:
   - **Clinical feature analysis** (`clinical_bias_analysis_v3.py`) — Extracts 10 ordinal/continuous clinical features (urgency, empathy, medication specificity, etc.) and tests for demographic effects via mixed-effects models.
   - **Quantitative comparison** (`chatbot_quantitative_comparison.py`) — Extracts 19 quantitative linguistic features and runs paired Wilcoxon signed-rank tests on matched prompt conditions.
   - **ML distinguishability** (`chatbot_ml_comparison.py`) — Trains classifiers (TF-IDF + embeddings) to measure how linguistically separable the two chatbots are.

---

## Repository Structure

```
.
├── data/
│   ├── web_conversations.json      # Short-form responses (Doctronic + DrKhan)
│   └── web_convo_short.json        # Additional short-form conversation set
├── web_automation/
│   ├── base_web_client.py          # Playwright base class
│   ├── doctronic_client.py         # Doctronic browser automation
│   ├── drkhan_client.py            # DrKhan browser automation
│   ├── conversation_manager_v2.py  # Multi-turn LLM-patient manager
│   ├── web_storage.py              # Save conversations to disk
│   ├── run_chatbots.py             # Single-turn prompt runner
│   └── run_comprehensive_experiment.py  # Multi-turn experiment runner
├── bias_analysis/
│   ├── clinical_bias_analysis_v3.py       # Method 1: clinical feature analysis
│   ├── chatbot_quantitative_comparison.py  # Method 2: quantitative comparison
│   ├── chatbot_ml_comparison.py            # Method 3: ML distinguishability
│   ├── analysis_constants.py       # Chatbot names, colours, labels
│   ├── analysis_utils.py           # FDR, effect-size helpers
│   ├── feature_registry.py         # Feature metadata registry
│   ├── prompt_blocks.py            # Matched prompt-pair construction
│   ├── schema_validation.py        # Input data validation
│   ├── shared_clinical_features.py # Regex feature extractors (clinical)
│   ├── shared_quantitative_features.py  # Feature extractors (quantitative)
│   └── validation/                 # Inter-rater reliability tools
├── models/
│   ├── ollama_chat_client.py       # LLM patient-simulator (Ollama)
│   └── ollama_client.py
├── config.py                       # Experiment configuration
├── requirements.txt
└── README.md
```

---

## Setup

### 1. Install Python dependencies

```bash
pip install -r requirements.txt
```

Python 3.11+ is recommended.

### 2. Install Playwright browsers (for data collection only)

```bash
playwright install chromium
```

### 3. (Optional) Install Ollama for multi-turn experiments

Download from https://ollama.ai and pull the model used as the patient simulator:

```bash
ollama pull llama3
```

---

## Running the Analysis

All commands should be run from the **project root** directory.

### Method 1 — Clinical Feature Analysis

Extracts clinical features and tests for demographic effects using mixed-effects models.

```bash
cd bias_analysis
python clinical_bias_analysis_v3.py
```

**Options:**
- `--use-nlp` — Use sentence-transformer anchor scoring (requires `sentence-transformers`)
- `--seed N` — Set random seed (default: 42)

**Outputs:** `bias_analysis/plots/clinical_v3/`, `bias_analysis/clinical_features_v3.csv`

---

### Method 2 — Quantitative Comparison

Extracts 19 quantitative linguistic features and runs Wilcoxon signed-rank tests on matched prompt pairs with BH-FDR correction.

```bash
cd bias_analysis
python chatbot_quantitative_comparison.py
```

**Options:**
- `--no-nlp` — Skip semantic similarity (faster, no GPU needed)
- `--data-path PATH` — Override the default data file

**Outputs:** `bias_analysis/plots/quant/`, `bias_analysis/quant_features.csv`, `bias_analysis/quant_paired_stats.csv`

---

### Method 3 — ML Distinguishability

Trains TF-IDF and embedding classifiers to measure how linguistically separable the chatbots are.

```bash
cd bias_analysis
python chatbot_ml_comparison.py
```

**Options:**
- `--min-df N` — Minimum document frequency for TF-IDF (default: 1)
- `--data-path PATH` — Override the default data file

**Outputs:** `bias_analysis/plots/model_comparison_f1.png`, `bias_analysis/plots/confusion_matrix_best.png`

---

## Collecting New Data (Optional)

If you want to re-run the data collection against the live chatbot websites:

```bash
# Single-turn prompts (Doctronic and DrKhan)
python web_automation/run_chatbots.py

# Multi-turn comprehensive experiment (one scenario, up to 6 turns)
python -m web_automation.run_comprehensive_experiment --chatbot doctronic
python -m web_automation.run_comprehensive_experiment --chatbot drkhan
```

> **Note:** Data collection requires an active internet connection and the chatbot websites to be accessible. Web automation is done via Playwright (headless browser).

---

## Statistical Design

- **Primary test:** Wilcoxon signed-rank on within-prompt matched pairs (Doctronic vs DrKhan), with Benjamini-Hochberg FDR correction across all features.
- **Secondary:** Mixed-effects linear regression (MixedLM) with race, gender, and age as fixed effects; symptom as random intercept.
- **Exploratory:** Kruskal-Wallis / one-way ANOVA by demographic group; post-hoc Dunn's test (FDR-corrected); semantic embedding PERMANOVA.
- **Effect sizes:** r (Wilcoxon), η² / ε² (group tests), standardised β (MixedLM).

---

## Requirements

See `requirements.txt`. Key dependencies:

| Package | Version | Purpose |
|---------|---------|---------|
| scipy | 1.13.1 | Wilcoxon, KW, Spearman |
| statsmodels | 0.14.6 | MixedLM, FDR, ANOVA |
| scikit-learn | 1.6.1 | ML classifiers, TF-IDF |
| sentence-transformers | 5.3.0 | Semantic similarity, NLP scoring |
| playwright | — | Web browser automation |
| matplotlib / seaborn | — | Visualisation |
