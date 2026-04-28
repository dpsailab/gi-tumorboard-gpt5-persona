# LLM Specialist Role Personas in Oncological Decision Support
## Reproducibility Repository

This repository contains the complete analysis and experimental pipeline accompanying the manuscript:

> **"Role Prompting Modulates Linguistic Style but Not Clinical Decision Structure in GPT-5-Based Gastrointestinal Tumour Board Simulation"**
> *(Submitted for peer review)*


---

## Study Overview

This study evaluates whether assigning specialist role personas—Surgeon, Medical Oncologist, and Radiation Oncologist—to large language models improves alignment with multidisciplinary tumour board (MTB) treatment recommendations.

The analysis framework combines:

- Clinical agreement evaluation
- Embedding-space geometric validation
- Statistical robustness testing
- Prompt engineering reproducibility experiments

Embedding analyses are used to determine whether different clinical personas induce:
- Distinct semantic representations
- Stable role identities in latent space
- Measurable clinical decision structure separation

---

## Repository Structure

```
.
├── config.py                         # Centralised constants (colours, column names, weights)
├── utils.py                          # Shared utility functions (parsing, statistics, comparisons)
│
├── 00_demographics.py                # Dataset population statistics
├── 01_agreement_analysis.py          # Agreement rates, Cochran Q, McNemar, Wilson CIs
├── 02_embedding_analysis.py          # Cosine similarity, PCA, UMAP, Jensen–Shannon divergence
├── 03_persona_stability_analysis.py  # PSI, CRI, boundary entropy, clinical risk score
├── 04_advanced_analysis.py           # Confusion matrices, Cohen κ, correlations
├── 05_sensitivity_analysis_composite_indices.py           # sensitivity analysis, convergent validity, component contribution 
│
├── data/
│ ├── anonymized_dataset/ # Study dataset (not publicly distributed)
│ └── dummy_patients/ # Synthetic reproducibility cases
│
├── prompts/                          # Prompt engineering experimentation module
│ ├── openai_client.py
│ ├── prompt_templates.py
│ ├── run_framework_experiment.py
│ └── evaluation.md
│
├── output/                           # outputs created at runtime
│
├── requirements.txt                  # Python dependencies
└── README.md

```

## Data Availability

The `data/` directory contains:

- **Derived study dataset:** GPT-5 outputs for all six 
  prompting frameworks, treatment category classifications, 
  embedding vectors, and MDT reference labels for all 
  100 cases. This enables full reproduction of all 
  statistical analyses and figures reported in the 
  manuscript.
- **Original clinical vignettes:** The source patient 
  summaries cannot be distributed due to institutional 
  data governance constraints, despite anonymization. 
  Researchers requiring access for replication may 
  contact the corresponding author.
- **Dummy patient cases** (`dummy_patients/`): Synthetic 
  cases matching the input schema, provided for 
  testing the prompt execution pipeline 
  (Frameworks 1–5) without requiring patient data 
  or API access to study outputs.

---

## Prompts Module

The `prompts/` module enables controlled experimental prompt testing for GPT-based tumour board reasoning.

Purpose:
- Ensure methodological transparency
- Isolate prompt engineering effects from clinical dataset effects

Experimental frameworks include:

### Framework 1 — Standard Tumour Board
Multidisciplinary reasoning with direct treatment recommendation output.

### Framework 2 — Multi-expert deliberation Reasoning
The model generates independent reasoning paths for:
- Surgical oncology
- Medical oncology
- Radiation oncology

Then synthesises a consensus decision.


### Frameworks 3–5 — Specialist Personas
Single-specialty clinical perspective prompting.

Configuration requires user-provided API keys.

---

## Running the Analyses

### 1. Install dependencies

```bash
pip install -r requirements.txt
```



## Running the Analyses

### 1. Install dependencies

```bash
pip install -r requirements.txt
```


### 2. Dataset Placement

Place the anonymized dataset in the repository root (already in folder data).


### 3. Execution Order

Each script is self-contained and can be run independently, but the standard analysis order is:

```bash
python 00_demographics.py               # ~1 min
python 01_agreement_analysis.py         # ~1 min
python 02_embedding_analysis.py         # ~2 min
python 03_persona_stability_analysis.py # ~2 min
python 04_advanced_analysis.py          # ~2 min
python 05_sensitivity_analysis_composite_indices.py  # ~1 min
```
---

## Core Statistical Methods

### Agreement Evaluation

- Cochran’s Q test  
- McNemar pairwise testing  
- Holm–Bonferroni correction  
- Wilson confidence intervals  

### Embedding Space Analysis

- Pairwise cosine similarity matrices  
- PCA manifold structural validation  
- Jensen–Shannon distribution divergence  
- Centroid geometric separation  
- Tumour-stratified structural validation  

### Statistical Robustness Testing

- Kruskal–Wallis projection separation testing  
- Effect size estimation via variance decomposition  
- Case-level and population-level signal testing  

---

## Persona Stability Metrics

The study introduces composite clinical robustness metrics.

 ⚠ **Methodological note:** The PSI and CRI are exploratory 
 composite indices introduced in this study. They have not been 
 externally validated against independent cohorts or clinical 
 outcomes. Weights are theory-driven and defined a priori in
 `config.py`; no data-driven optimisation was performed. 
 These metrics should be interpreted as descriptive summaries 
 of multi-dimensional model behaviour within this evaluation 
 framework, not as generalizable clinical performance measures.

### Persona Stability Index (PSI)
A composite index summarising four dimensions of role-specific 
behavioural consistency: semantic identity preservation 
(embedding cosine similarity across prompting conditions), 
role-specific clinical content presence, boundary control 
(penalising cross-specialty decision leakage), and treatment 
recommendation accuracy. Weights are defined in `config.py` 
and justified in the manuscript Supplementary Material.

### Composite Robustness Index (CRI)
Extends the PSI with a global entropy stability penalty term 
reflecting distributional unpredictability of treatment 
recommendations across cases. Full construction rationale 
and sensitivity analysis are provided in the manuscript 
Supplementary Material.

## Experimental Configuration

All analysis parameters are centrally controlled via:
````python
config.py
````

All analysis parameters, column mappings, and composite index weights are defined in `config.py`.
Key parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `SHOW_PLOTS` | `False` | Display plots interactively |
| `CRI_WEIGHTS` | see file | Composite Robustness Index weights |
| `PSI_WEIGHTS` | see file | Persona Stability Index weights |
| `RISK_WEIGHTS` | see file | Clinical Risk Penalty weights |

---
## Outputs

Generated outputs include:

### Agreement Analysis

- Treatment concordance statistics  
- Inter-method reliability tests  

### Embedding Analysis

- Similarity matrices  
- PCA coordinates and centroids  
- Jensen–Shannon divergence metrics  

### Stability Analysis

- Persona robustness indices  
- Bootstrap confidence intervals  

### Advanced Analysis

- Clinical signal correlation testing  
- Risk score modelling  

---

## Requirements

Core dependencies:

- Python ≥ 3.9  
- pandas  
- numpy 
- openai
- scipy  
- statsmodels  
- scikit-learn  
- matplotlib  
- seaborn  
- python-docx  
- openpyxl  

---

## Reproducibility Protocol

To ensure reproducibility:

- Fix the model version in prompt experiments.  
- Do not modify prompt templates.  
- Log raw model outputs before post-processing.  
- Maintain dataset preprocessing pipeline consistency.  

---

## Ethical Statement

This project is for research purposes only.

⚠ LLM-generated clinical outputs must **not** be used for real patient care.

---

## License

MIT License.

See `LICENSE`.

---

## Citation

If you use this work, please cite:

> Citation will be added after peer review acceptance.

---

## Contact

Corresponding author contact details will be provided upon publication.
