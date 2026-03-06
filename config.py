"""
config.py
=========

Centralised configuration for analysis pipelines.

This module defines:
- Dataset paths
- Role personas
- Column naming conventions
- Visualization styling
- Statistical metric weights

Study context:
Evaluation of LLM persona effects on multidisciplinary tumour board agreement
in gastrointestinal oncology decision support.
"""

# ==========================================================
# Runtime Flags
# ==========================================================

SHOW_PLOTS: bool = False

# ==========================================================
# Data
# ==========================================================

DATA_FILE: str = "data/LLM_MDT_dataset_repository.csv"

# ==========================================================
# Role Registry
# ==========================================================

ROLE_CONFIG = {
    "Simulated Tumorboard": {
        "prefix": "F1_MDT_simulation",
        "color": "#9467bd",
    },
    "Surgeon": {
        "prefix": "F3_persona_surgeon",
        "color": "#1f77b4",
    },
    "Oncologist": {
        "prefix": "F4_persona_medical_oncologist",
        "color": "#ff7f0e",
    },
    "Radio-Oncologist": {
        "prefix": "F5_persona_radiation_oncologist",
        "color": "#2ca02c",
    },
}

ROLES = list(ROLE_CONFIG.keys())

# ==========================================================
# Column Factory Helpers
# ==========================================================

def _treatment_col(prefix: str) -> str:
    return f"{prefix}_treatment"

def _embedding_col(prefix: str) -> str:
    return f"{prefix}_embeddings"

def _domain_col(prefix: str) -> str:
    return f"{prefix}_domain_content_present"

def _boundary_col(prefix: str) -> str:
    return f"{prefix}_boundary_violation"

def _concordance_col(prefix: str) -> str:
    return f"{prefix}_treatment_concordance"

# ==========================================================
# Derived Column Mappings
# ==========================================================

REFERENCE_TREATMENT_COL = "tumorboard_primary_treatment"

SPECIALIST_COLS = [
    _treatment_col(cfg["prefix"])
    for cfg in ROLE_CONFIG.values()
]

METHOD_TREATMENT_COLS = {
    role: _treatment_col(cfg["prefix"])
    for role, cfg in ROLE_CONFIG.items()
}

ROLE_PREFIX_MAP = {
    role: cfg["prefix"]
    for role, cfg in ROLE_CONFIG.items()
}

# Specialist persona concordance columns
SPECIALIST_PERSONA_CONCORDANCE_COLS = {
    role: _concordance_col(cfg["prefix"])
    for role, cfg in ROLE_CONFIG.items()
}

# ==========================================================
# Model Output Columns
# ==========================================================

COLUMNS_ANSWER = [
    "tumorboard_treatment",
    "F1_MDT_simulation",
    "F2_multi_expert_consensus",
    "F3_persona_surgeon",
    "F4_persona_medical_oncologist",
    "F5_persona_radiation_oncologist",
    "F6_majority_vote",
]

COLUMNS_ANSWER_RENAME = [
    "Simulated Tumorboard",
    "Multi-Expert",
    "Surgeon",
    "Oncologist",
    "Radio-Oncologist",
    "Majority Vote",
]

RENAME_DICT = dict(zip(COLUMNS_ANSWER[1:], COLUMNS_ANSWER_RENAME))

# ==========================================================
# Signal Columns
# ==========================================================

SIGNAL_COLUMNS = [
    (
        prefix,
        role,
        _domain_col(prefix),
        _boundary_col(prefix),
        _concordance_col(prefix),
    )
    for role, cfg in ROLE_CONFIG.items()
    for prefix in [cfg["prefix"]]
]

# ==========================================================
# Embedding Columns
# ==========================================================

EMBEDDING_COLS = {
    role: _embedding_col(cfg["prefix"])
    for role, cfg in ROLE_CONFIG.items()
}


# Specialist persona persona embeddings
SPECIALIST_PERSONA_EMBEDDING_COLS = {
    role: f"{cfg['prefix']}_embeddings"
    for role, cfg in ROLE_CONFIG.items()
}

# Multi-expert deliberation embeddings
MULTI_EXPERT_EMBEDDING_COLS = {
    role: f"F2_multi_expert_consensus_{'tumorboard' if role == 'Simulated Tumorboard' else role.lower()}_embeddings"
    for role in ROLE_CONFIG.keys()
}


# ==========================================================
# Visualization Styling
# ==========================================================

ROLE_COLORS = {
    role: cfg["color"]
    for role, cfg in ROLE_CONFIG.items()
}

BAR_COLORS = [
    "#E63946",
    "#F4A261",
    "#E9C46A",
    "#2A9D8F",
    "#457B9D",
    "#6A4C93",
]

# ==========================================================
# Labels
# ==========================================================

TITLE_COLUMN_RENAME = {
    "tumour_type": "Tumour Type",
    "presentation": "Consultation Type",
    "tumorboard_treatment": "Tumour Board Recommendation",
    "tumorboard_primary_treatment": "Tumour Board Primary Recommendation",
}

VALUE_RENAME = {
    # Tumour types
    "Ösophagus": "Oesophagus-Ca",
    "Pankreas": "Pancreatic-Ca",
    "Magen": "Gastric-Ca",
    "Kolon": "Colorectal-Ca",
    "Leber": "Hepatobiliary-Ca",

    # Consultation type
    "1": "First Presentation",
    "2": "FUP-Consultation",

    # Treatment categories
    "lokale Therapie": "Local Therapy",
    "Diagnostik": "Further Diagnostics",
    "Endo-Resektion": "Endoscopic Resection",
    "Follow-up": "Active Surveillance",
    "Systematic Therapy": "Systemic Therapy",
}

# ==========================================================
# Framework Mapping
# ==========================================================

FRAMEWORK_PREFIX_MAP = {
    "single": "F1_MDT_simulation",
    "multi_expert": "F2_multi_expert_consensus",
}

# ==========================================================
# Experimental Metrics Weights
# ==========================================================

CRI_WEIGHTS = {
    "cosine_similarity": 0.35,
    "specificity_rate": 0.25,
    "pitch_control": 0.20,
    "accuracy": 0.15,
    "entropy_stability": 0.05,
}

PSI_WEIGHTS = {
    "cosine_similarity": 0.40,
    "specificity_rate": 0.30,
    "pitch_control": 0.20,
    "accuracy": 0.10,
}

RISK_WEIGHTS = {
    "boundary_violation": 0.60,
    "non_specificity": 0.40,
}

# ==========================================================
# Output Paths
# ==========================================================

OUTPUT_DIR_ROLE: str = "output"
OUTPUT_DIR_ADVANCED: str = "output/advanced"
OUTPUT_DIR_IMG_ROLE: str = "output/img"