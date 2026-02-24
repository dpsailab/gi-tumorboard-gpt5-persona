"""
config.py
=========
Centralised configuration for all analysis scripts.

Defines column names, display labels, value mappings, colour schemes,
and global runtime flags.  Importing this module from any other script
guarantees that every parameter is consistent across the pipeline.

Study context
-------------
This code accompanies a study evaluating whether assigning specialist
role personas (Surgeon, Oncologist, Radio-Oncologist) to a large language
model (GPT-5 / ChatGPT) improves alignment with multidisciplinary tumour
board (MTB) recommendations for gastrointestinal oncology cases.
"""

# ---------------------------------------------------------------------------
# Runtime flags
# ---------------------------------------------------------------------------

# Set to True during development to display plots interactively.
# Must be False for headless (cluster / CI) execution.
SHOW_PLOTS: bool = False

# Set to True to run UMAP dimensionality reduction (requires `umap-learn`).
# UMAP is computationally expensive; disable for fast iteration.
DO_UMAP: bool = False

# ---------------------------------------------------------------------------
# Data source
# ---------------------------------------------------------------------------

DATA_FILE: str = "data/LLM_MDTB_dataset_repository.csv"

# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------
# Coral gradient palette used consistently across all figures.
COLOR_1: str = "#FADADD"   # light coral  → Surgeon
COLOR_2: str = "#F08080"   # medium coral → Oncologist
COLOR_3: str = "#FFA07A"   # salmon coral → Radio-Oncologist

# Role-specific colour mapping (used in grouped bar charts and UMAP plots).
ROLE_COLORS: dict = {
    "Surgeon":          "#E9C46A",
    "Oncologist":       "#2A9D8F",
    "Radio-Oncologist": "#457B9D",
}

# Bar-chart palette for sequential display of up to 6 groups.
BAR_COLORS: list = [
    "#E63946",   # red
    "#F4A261",   # orange
    "#E9C46A",   # yellow
    "#2A9D8F",   # teal
    "#457B9D",   # blue
    "#6A4C93",   # purple
]

# ---------------------------------------------------------------------------
# Specialist role columns (treatment predictions)
# ---------------------------------------------------------------------------

SPECIALIST_COLS: list = [
    "F3_persona_surgeon_treatment",
    "F4_persona_medical_oncologist_treatment",
    "F5_persona_radiation_oncologist_treatment",
]

# All LLM output columns to compare against the reference standard
# (index 0 = reference; indices 1-N = comparison columns).
COLUMNS_ANSWER: list = [
    "tumorboard_treatment",                             # Reference: MTB decision                   # Reference: MTB decision (primary)
    "F1_MDTB_simulation",                               # Baseline (no persona)
    "F2_multi_expert_consensus",                        # Multi-expert prompt
    "F3_persona_surgeon",                               # Surgeon persona
    "F4_persona_medical_oncologist",                    # Oncologist persona
    "F5_persona_radiation_oncologist",                  # Radio-Oncologist persona
    "F6_majority_vote",                                 # Majority vote across roles
]

# Human-readable labels corresponding to COLUMNS_ANSWER[1:] (reference excluded)
COLUMNS_ANSWER_RENAME: list = [
    "Simulated Tumorboard",
    "Multi-Expert",
    "Surgeon",
    "Oncologist",
    "Radio-Oncologist",
    "Majority Vote",
]

# Mapping from raw column names to display labels (for plot axes / tables)
RENAME_DICT: dict = dict(zip(COLUMNS_ANSWER[1:], COLUMNS_ANSWER_RENAME))

# ---------------------------------------------------------------------------
# Axis and column title mappings
# ---------------------------------------------------------------------------

TITLE_COLUMN_RENAME: dict = {
    "tumour_type":         "Tumour Type",
    "presentation":                   "Consultation Type",
    "tumorboard_treatment": "Tumour Board Recommendation",
    "tumorboard_primary_treatment": "Tumour Board Primary Recommendation",
}

# ---------------------------------------------------------------------------
# Value label mappings (German → English)
# ---------------------------------------------------------------------------

VALUE_RENAME: dict = {
    # Tumour types
    "Ösophagus":  "Oesophagus-Ca",
    "Pankreas":   "Pancreatic-Ca",
    "Magen":      "Gastric-Ca",
    "Kolon":      "Colorectal-Ca",
    "Leber":      "Hepatobiliary-Ca",
    # Consultation type
    "1":         "First Presentation",
    "2":         "FUP-Consultation",
    # Treatment categories
    "lokale Therapie":  "Local Therapy",
    "Diagnostik":       "Further Diagnostics",
    "Endo-Resektion":   "Endoscopic Resection",
    "Follow-up":              "Active Surveillance",
    "Systematic Therapy": "Systemic Therapy",
}

# ==========================================================
# Role mapping (framework prefix)
# ==========================================================

ROLE_PREFIX_MAP = {
    "Surgeon": "F3_persona_surgeon",
    "Oncologist": "F4_persona_medical_oncologist",
    "Radio-Oncologist": "F5_persona_radiation_oncologist",
}

ROLES = list(ROLE_PREFIX_MAP.keys())


# ==========================================================
# Framework mapping
# ==========================================================

FRAMEWORK_PREFIX_MAP = {
    "single": "F1_MDTB_simulation",
    "self_consistency": "F2_multi_expert_consensus",
}

# ---------------------------------------------------------------------------
# Embedding column definitions
# ---------------------------------------------------------------------------

# Mapping: readable role name → embedding column in the DataFrame
EMBEDDING_COLS: dict = {
    "Surgeon":          "F3_persona_surgeon_embedding",
    "Oncologist":       "F4_persona_medical_oncologist_embeddings",
    "Radio-Oncologist": "F5_persona_radiation_oncologist_embeddings",
    "Simulated Tumorboard": "F1_MDTB_simulation_embeddings",
}

# Self-consistency embedding columns
SELF_CONSISTENCY_EMBEDDING_COLS: dict = {
    "surgeon":          "F2_multi_expert_consensus_surgeon_embeddings",
    "oncologist":       "F2_multi_expert_consensus_oncologist_embeddings",
    "radio-oncologist": "F2_multi_expert_consensus_radio-oncologist_embeddings",
}



# ---------------------------------------------------------------------------
# Composite Robustness Index weights
# (documented in the Methods section of the manuscript)
# ---------------------------------------------------------------------------

CRI_WEIGHTS: dict = {
    "cosine_similarity":   0.35,
    "specificity_rate":    0.25,
    "pitch_control":       0.20,   # weight applied to (1 - boundary_violation_rate)
    "accuracy":            0.15,
    "entropy_stability":   0.05,   # weight applied to (1 - global_entropy)
}

# Persona Stability Index weights
PSI_WEIGHTS: dict = {
    "cosine_similarity": 0.40,
    "specificity_rate":  0.30,
    "pitch_control":     0.20,   # weight applied to (1 - boundary_violation_rate)
    "accuracy":          0.10,
}

# Clinical Risk Penalty weights
RISK_WEIGHTS: dict = {
    "boundary_violation": 0.60,
    "non_specificity": 0.40,   # weight applied to (1 - specificity_rate)
}

# ---------------------------------------------------------------------------
# Output directories
# ---------------------------------------------------------------------------

OUTPUT_DIR_ROLE:     str = "output"
OUTPUT_DIR_ADVANCED: str = "output/advanced"
OUTPUT_DIR_IMG_ROLE: str = "output/img"
