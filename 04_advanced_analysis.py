"""
04_advanced_analysis.py
Refactored version using config-driven mappings only.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import itertools
from scipy.stats import pointbiserialr
from sklearn.metrics import cohen_kappa_score
import seaborn as sns

from config import (
    DATA_FILE,
    OUTPUT_DIR_ROLE,
    SHOW_PLOTS,
    METHOD_TREATMENT_COLS,
    SIGNAL_COLUMNS,
    REFERENCE_TREATMENT_COL,
    KAPPA_METHODS
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

output_path = f"{TABLE_DIR}/confusion_matrices_all.xlsx"

with pd.ExcelWriter(output_path, engine="openpyxl") as writer:

    for diag in df["tumour_type"].dropna().unique():

        sub = df[df["tumour_type"] == diag]
        ref_series = sub[REFERENCE_TREATMENT_COL].fillna("MISSING")

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

            # Sheet names: max 31 chars (Excel limit)
            # e.g. "Colorectal_Surgeon"
            sheet_name = f"{diag}_{role}"[:31]

            print(f"\nConfusion Matrix - {diag}_{role}")
            print(cm.to_string())

            cm.to_excel(writer, sheet_name=sheet_name)

            print(f"  Sheet written: {sheet_name}")

print(f"\nAll confusion matrices saved → {output_path}")
print("Confusion matrices complete.")

# ===========================================================================
# B. Cohen's kappa - full pairwise matrix
# ===========================================================================

labels = list(KAPPA_METHODS.keys())
cols   = list(KAPPA_METHODS.values())

# Build matrix
kappa_matrix = pd.DataFrame(np.nan, index=labels, columns=labels)

for (l1, c1), (l2, c2) in itertools.combinations(zip(labels, cols), 2):

    if c1 not in df.columns or c2 not in df.columns:
        continue

    mask   = df[c1].notna() & df[c2].notna()
    y1     = df.loc[mask, c1].astype(str)
    y2     = df.loc[mask, c2].astype(str)

    if len(y1) == 0 or y1.nunique() < 2 or y2.nunique() < 2:
        k = np.nan
    else:
        k = cohen_kappa_score(y1, y2)

    kappa_matrix.loc[l1, l2] = round(k, 3)
    kappa_matrix.loc[l2, l1] = round(k, 3)

# Diagonal = 1
for l in labels:
    kappa_matrix.loc[l, l] = 1.000

print("\n=== Pairwise Cohen's Kappa Matrix ===")
print(kappa_matrix.to_string())

kappa_matrix.to_excel(f"{TABLE_DIR}/kappa_pairwise_matrix.xlsx")

# ---------------------------------------------------------------------------
# Heatmap - rescaled to actual data range
# ---------------------------------------------------------------------------

matrix_float = kappa_matrix.astype(float)
vmin = round(matrix_float.values[~np.eye(len(labels), dtype=bool)].min() - 0.05, 2)

fig, ax = plt.subplots(figsize=(10, 8))

sns.heatmap(
    matrix_float,
    annot=True,
    fmt=".3f",
    cmap="RdYlGn",       # diverging: red=low, green=high
    vmin=vmin,
    vmax=1.0,
    linewidths=0.5,
    ax=ax,
    square=True,
    mask=np.eye(len(labels), dtype=bool),  # hide diagonal
)

# Draw diagonal separately as grey
for i in range(len(labels)):
    ax.add_patch(plt.Rectangle((i, i), 1, 1, fill=True, color='lightgrey'))
    ax.text(i + 0.5, i + 0.5, "1.000", ha='center', va='center',
            fontsize=9, color='black')

ax.set_title("Pairwise Cohen's Kappa Matrix", fontsize=13)
plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)
plt.tight_layout()

_save_or_show(f"{IMG_DIR}/kappa_pairwise_heatmap.png")

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
)


print("\n=== Advanced Analysis Complete ===")