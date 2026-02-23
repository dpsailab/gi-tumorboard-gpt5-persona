# LLM Specialist Role Personas in Oncological Decision Support
## Reproducibility Repository

This repository contains the complete analysis code accompanying the manuscript:

> **"Effect of Specialist Role Personas on LLM Alignment with Multidisciplinary Tumour Board Recommendations in Gastrointestinal Oncology"**
> *(Submitted for peer review)*

---

## Study Summary

We evaluated whether assigning specialist role personas—Surgeon, Oncologist, and Radio-Oncologist—to a large language model (GPT-4) improves the alignment of its treatment recommendations with institutional multidisciplinary tumour board (MTB) decisions across 5 gastrointestinal cancer types.  Embedding-space analyses were used to quantify whether each persona produces a geometrically distinct, stable clinical identity.

---

## Repository Structure

```
.
├── config.py                         # Centralised constants (colours, column names, weights)
├── utils.py                          # Shared utility functions (parsing, statistics, comparisons)
│
├── 01_agreement_analysis.py          # Agreement rates, Cochran Q, McNemar, Wilson CIs
├── 02_embedding_analysis.py          # Cosine similarity, PCA, UMAP, Jensen–Shannon divergence
├── 03_persona_stability_analysis.py  # PSI, CRI, boundary entropy, clinical risk score
├── 04_advanced_analysis.py           # Confusion matrices, Cohen κ, GEE, correlations
│
├── role/                             # Excel outputs (created at runtime)
├── img/
│   ├── role/                         # Figures: agreement, treatment distribution
│   └── advanced/                     # Figures: PCA, UMAP, confusion matrices
│
└── requirements.txt                  # Python dependencies
```

> **Note:** The input data file (`Ultima_dataset with embeddings_role_treatment.xlsx`) is **not** included in this repository to protect patient privacy.  A fully de-identified synthetic dataset matching the column schema is available upon request from the corresponding author.

---

## Running the Analyses

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

To enable UMAP visualisations, also install:

```bash
pip install umap-learn
```

### 2. Place the dataset

Copy `Ultima_dataset with embeddings_role_treatment.xlsx` into the repository root.

### 3. Execute scripts in order

Each script is self-contained and can be run independently, but the standard analysis order is:

```bash
python 01_agreement_analysis.py      # ~1 min
python 02_embedding_analysis.py      # ~2–5 min (longer with UMAP enabled)
python 03_persona_stability_analysis.py  # ~3 min (bootstrap = 500 iterations)
python 04_advanced_analysis.py       # ~2 min
```

To enable UMAP, set `DO_UMAP = True` in `config.py`.

---

## Configuration

All analysis parameters, column mappings, and composite index weights are defined in `config.py`.  Key parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `SHOW_PLOTS` | `False` | Display plots interactively |
| `DO_UMAP` | `False` | Run UMAP dimensionality reduction |
| `CRI_WEIGHTS` | see file | Composite Robustness Index weights |
| `PSI_WEIGHTS` | see file | Persona Stability Index weights |
| `RISK_WEIGHTS` | see file | Clinical Risk Penalty weights |

---

## Key Outputs

| File | Description |
|------|-------------|
| `role/overall_agreement_ci.xlsx` | Overall accuracy with 95 % Wilson CIs |
| `role/ci_by_tumor_type.xlsx` | Stratified accuracy by tumour type |
| `role/mcnemar_matrix.xlsx` | Pairwise McNemar χ² statistics |
| `role/holm_bonferroni_results.xlsx` | Holm–Bonferroni corrected p-values |
| `role/kappa_results.xlsx` | Cohen's κ for all method pairs |
| `role/similarity_matrix_roles.xlsx` | Mean pairwise cosine similarities |
| `role/persona_drift_full_analysis.xlsx` | Per-case cosine drift (single vs SC) |
| `role/persona_stability_index.xlsx` | PSI per role |
| `role/composite_robustness_index.xlsx` | CRI per role |
| `role/composite_robustness_bootstrap_ci.xlsx` | Bootstrap 95 % CIs for CRI |
| `role/clinical_risk_score.xlsx` | Clinical Risk Penalty Score |
| `role/Analysis_Tables_role.docx` | Word tables for manuscript |

---

## Statistical Methods

- **Cochran's Q + McNemar** (Holm–Bonferroni corrected): omnibus and pairwise tests for differences in accuracy across methods (matched design).
- **Wilson score confidence intervals**: 95 % CIs for binomial proportions.
- **Cohen's κ**: inter-rater agreement for treatment category classification.
- **Generalised Estimating Equations (GEE)**: accounts for within-patient correlation across role conditions.
- **Point-biserial correlation**: association between role-specific content / pitch invasion and treatment correctness.
- **Jensen–Shannon divergence**: symmetric distributional distance between role embedding distributions.
- **Bootstrap resampling** (n = 500): 95 % CIs for composite indices.

---

## Requirements

See `requirements.txt`.  Core dependencies:

- Python ≥ 3.9
- pandas, numpy, scipy, statsmodels
- scikit-learn
- matplotlib, seaborn
- python-docx
- openpyxl

---

## License

MIT License — see `LICENSE` for details.

## Citation

If you use this code, please cite:

```
[Citation will be added upon manuscript acceptance]
```

## Contact

[Author contact details will be added upon publication]
