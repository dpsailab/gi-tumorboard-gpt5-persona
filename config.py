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
model (GPT-4 / ChatGPT) improves alignment with multidisciplinary tumour
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

DATA_FILE: str = "Ultima_dataset with embeddings_role_treatment.xlsx"

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
    "ChatGPT_single_request_5_surgeon_treatment",
    "ChatGPT_single_request_5_oncologist_treatment",
    "ChatGPT_single_request_5_radio-oncologist_treatment",
]

# All LLM output columns to compare against the reference standard
# (index 0 = reference; indices 1-N = comparison columns).
COLUMNS_ANSWER: list = [
    "Konferenzbeschluss",                               # Reference: MTB decision
    "ChatGPT_single_request_5",                         # Baseline (no persona)
    "ChatGPT_single_request_5_self-consistency",        # Self-consistency baseline
    "ChatGPT_single_request_5_surgeon",                 # Surgeon persona
    "ChatGPT_single_request_5_oncologist",              # Oncologist persona
    "ChatGPT_single_request_5_radio-oncologist",        # Radio-Oncologist persona
    "majority",                                         # Majority vote across roles
]

# Human-readable labels corresponding to COLUMNS_ANSWER[1:] (reference excluded)
COLUMNS_ANSWER_RENAME: list = [
    "Simulated Tumorboard",
    "Self Consistency",
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
    "Anmeldediagnose":         "Tumour Type",
    "EV/WV":                   "Consultation Type",
    "Konferenzbeschluss_treatment": "Tumour Board Recommendation",
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
    "EV":         "First Presentation",
    "WV":         "FUP-Consultation",
    # Treatment categories
    "lokale Therapie":  "Local Therapy",
    "Diagnostik":       "Further Diagnostics",
    "Endo-Resektion":   "Endoscopic Resection",
    "FUP":              "Active Surveillance",
    "Systematic Therapy": "Systemic Therapy",
}

# ---------------------------------------------------------------------------
# Embedding column definitions
# ---------------------------------------------------------------------------

# Mapping: readable role name → embedding column in the DataFrame
EMBEDDING_COLS: dict = {
    "Surgeon":          "ChatGPT_single_request_5_surgeon_embeddings",
    "Oncologist":       "ChatGPT_single_request_5_oncologist_embeddings",
    "Radio-Oncologist": "ChatGPT_single_request_5_radio-oncologist_embeddings",
    "Simulated Tumorboard": "ChatGPT_single_request_5_embeddings",
}

# Self-consistency embedding columns
SELF_CONSISTENCY_EMBEDDING_COLS: dict = {
    "surgeon":          "ChatGPT_single_request_5_self-consistency_surgeon_embeddings",
    "oncologist":       "ChatGPT_single_request_5_self-consistency_oncologist_embeddings",
    "radio-oncologist": "ChatGPT_single_request_5_self-consistency_radio-oncologist_embeddings",
}

# ---------------------------------------------------------------------------
# Tumour map (English ↔ German, used for embedding plots)
# ---------------------------------------------------------------------------

TUMOR_MAP: dict = {
    "Esophagus": "Ösophagus",
    "Pancreas":  "Pankreas",
    "Gastric":   "Magen",
    "Colon":     "Kolon",
    "Liver":     "Leber",
}

TUMOR_LIST: list = list(TUMOR_MAP.keys())

# ---------------------------------------------------------------------------
# Composite Robustness Index weights
# (documented in the Methods section of the manuscript)
# ---------------------------------------------------------------------------

CRI_WEIGHTS: dict = {
    "cosine_similarity":   0.35,
    "specificity_rate":    0.25,
    "pitch_control":       0.20,   # weight applied to (1 - pitch_invasion_rate)
    "accuracy":            0.15,
    "entropy_stability":   0.05,   # weight applied to (1 - global_entropy)
}

# Persona Stability Index weights
PSI_WEIGHTS: dict = {
    "cosine_similarity": 0.40,
    "specificity_rate":  0.30,
    "pitch_control":     0.20,   # weight applied to (1 - pitch_invasion_rate)
    "accuracy":          0.10,
}

# Clinical Risk Penalty weights
RISK_WEIGHTS: dict = {
    "pitch_invasion": 0.60,
    "non_specificity": 0.40,   # weight applied to (1 - specificity_rate)
}

# ---------------------------------------------------------------------------
# Output directories
# ---------------------------------------------------------------------------

OUTPUT_DIR_ROLE:     str = "role"
OUTPUT_DIR_ADVANCED: str = "img/advanced"
OUTPUT_DIR_IMG_ROLE: str = "img/role"
