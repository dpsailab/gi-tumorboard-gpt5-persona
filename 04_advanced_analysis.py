"""
04_advanced_analysis.py
Refactored version using config-driven mappings only.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scipy.stats import pointbiserialr, spearmanr
from sklearn.metrics import cohen_kappa_score
import seaborn as sns

from config import (
    DATA_FILE,
    OUTPUT_DIR_ROLE,
    SHOW_PLOTS,
    METHOD_TREATMENT_COLS,
    SIGNAL_COLUMNS,
    REFERENCE_TREATMENT_COL,
)

# -----------------------------------------------------------
# Setup
# -----------------------------------------------------------

TABLE_DIR = os.path.join(OUTPUT_DIR_ROLE, "advanced_analysis")
IMG_DIR = os.path.join(OUTPUT_DIR_ROLE, "advanced_analysis/img")
os.makedirs(TABLE_DIR, exist_ok=True)
os.makedirs(IMG_DIR, exist_ok=True)

# -----------------------------------------------------------
# Load Dataset
# -----------------------------------------------------------
df = pd.read_csv(DATA_FILE)

# Parse tumorboard treatment list
from utils import parse_treatment_list_column

df = parse_treatment_list_column(
    df,
    column="tumorboard_treatment",
    primary_output_col="tumorboard_primary_treatment"
)

# -----------------------------------------------------------
# Helpers
# -----------------------------------------------------------

def _save_or_show(path):
    plt.savefig(path, dpi=300, bbox_inches="tight")
    print(f"Saved → {path}")
    if SHOW_PLOTS:
        plt.show()
    plt.close()


def _print_and_save(df_obj: pd.DataFrame, path: str, title: str):
    print(f"\n{title}")
    print(df_obj.to_string(index=False))
    df_obj.to_excel(path, index=False)
    print(f"----> saved to {path}")

def _print_and_save_cm(cm: pd.DataFrame, path: str, title: str):
    print(f"\n{title}")
    print(cm.to_string())
    cm.to_excel(path, index=True)
    print(f"----> saved to {path}")

def _pointbiserial_safe(x, y):
    mask = x.notna() & y.notna()

    x = x[mask].astype(float)
    y = y[mask].astype(float)

    if len(x) <= 5 or x.nunique() < 2 or y.nunique() < 2:
        return np.nan, np.nan, len(x)

    return pointbiserialr(x, y)[0], pointbiserialr(x, y)[1], len(x)


# ===========================================================
# A. Confusion matrices
# ===========================================================

print("\n=== A. Confusion matrices ===")

for diag in df["tumour_type"].dropna().unique():

    sub = df[df["tumour_type"] == diag]

    ref_series = sub[f"{REFERENCE_TREATMENT_COL}"].fillna("MISSING")

    for role, col in METHOD_TREATMENT_COLS.items():

        if col not in df.columns:
            continue

        y_pred = sub[col].fillna("MISSING")

        cm = pd.crosstab(
            ref_series,
            y_pred,
            rownames=["Tumor Board"],
            colnames=[role]
        )

        fname = f"{TABLE_DIR}/confusion_{diag}_{role}.xlsx".replace("/", "-")

        _print_and_save_cm(
            cm,
            fname,
            f"Confusion Matrix | Tumour={diag} | Role={role}"
        )

print("Confusion matrices complete.")

# ===========================================================
# B. Cohen kappa
# ===========================================================

kappa_rows = []

for role, col in METHOD_TREATMENT_COLS.items():

    if col not in df.columns:
        continue

    tb_labels = []
    role_labels = []

    for _, row in df.iterrows():

        tb_list = row["tumorboard_treatment"]
        role_val = row[col]

        if isinstance(tb_list, list) and not pd.isna(role_val):

            if role_val in tb_list:
                tb_labels.append(role_val)
                role_labels.append(role_val)
            else:
                tb_labels.append("DISAGREE")
                role_labels.append(role_val)

    if len(tb_labels) > 0:
        k = cohen_kappa_score(tb_labels, role_labels)
    else:
        k = np.nan

    kappa_rows.append({
        "role": role,
        "kappa": k,
        "n": len(tb_labels)
    })

kappa_df = pd.DataFrame(kappa_rows)

_print_and_save(
    kappa_df,
    f"{TABLE_DIR}/kappa_vs_tumorboard.xlsx",
    "=== Cohen Kappa Results ==="
)

# ===========================================================
# C. Frequency analysis
# ===========================================================

print("\n=== C. Frequency distributions ===")

METHODS = (
    [REFERENCE_TREATMENT_COL]
    + list(METHOD_TREATMENT_COLS.values())
    + ["F6_majority_vote_treatment"]
)

def _get_series(method):

    if method == REFERENCE_TREATMENT_COL:
        return df[method]

    if method in df.columns:
        return df[method]

    return pd.Series([])

freqs = {
    m: _get_series(m).fillna("MISSING").value_counts()
    for m in METHODS
}

freq_df = pd.DataFrame(freqs).fillna(0)

if "MISSING" in freq_df.index:
    freq_df = freq_df.drop("MISSING")

_print_and_save(
    freq_df,
    f"{TABLE_DIR}/treatment_frequencies_by_method.xlsx",
    "=== Treatment Frequency Distribution ==="
)

# ===========================================================
# D. Point-biserial correlations
# ===========================================================

print("\n=== D. Point-biserial correlations ===")

corr_rows = []

for fw_prefix, role, spec_col, pitch_col, correct_col in SIGNAL_COLUMNS:

    if correct_col not in df.columns:
        continue

    if spec_col in df.columns:
        r, p, n = _pointbiserial_safe(df[spec_col], df[correct_col])

        corr_rows.append({
            "framework": fw_prefix,
            "role": role,
            "comparison": "specific_vs_correct",
            "r_pb": r,
            "pvalue": p,
            "n": n
        })

    if pitch_col in df.columns:
        r, p, n = _pointbiserial_safe(df[pitch_col], df[correct_col])

        corr_rows.append({
            "framework": fw_prefix,
            "role": role,
            "comparison": "pitch_vs_correct",
            "r_pb": r,
            "pvalue": p,
            "n": n
        })

corr_df = pd.DataFrame(corr_rows)

_print_and_save(
    corr_df,
    f"{TABLE_DIR}/pointbiserial_correlations.xlsx",
    "=== Point-biserial Correlations ==="
)
# ===========================================================
# E. Frequencies signals
# ===========================================================

print("\n=== E. Signal frequencies ===")

freq_rows = []

for fw_name, role, spec_col, pitch_col, _ in SIGNAL_COLUMNS:

    if spec_col in df.columns:
        freq_rows.append({
            "framework": fw_name,
            "role": role,
            "metric": "specific_rate",
            "percentage": df[spec_col].mean() * 100
        })

    if pitch_col in df.columns:
        freq_rows.append({
            "framework": fw_name,
            "role": role,
            "metric": "pitch_rate",
            "percentage": df[pitch_col].mean() * 100
        })

freq_signal_df = pd.DataFrame(freq_rows)

_print_and_save(
    freq_signal_df,
    f"{TABLE_DIR}/frequency_domain_content_pitch.xlsx",
    "=== Signal Frequencies ==="



print("\n=== Advanced Analysis Complete ===")