"""
04_advanced_analysis.py
=======================
Advanced inter-role analyses for the supplementary materials.

This module contains analyses that probe the structural relationships between
specialist role personas and the reference standard at a deeper statistical
level.  All methods are described in the supplementary methods section of the
manuscript.

Analyses
--------
A. Per-diagnosis confusion matrices (LLM role vs tumour board)
B. Cohen's kappa (treatment category agreement)
C. Treatment frequency comparison (over/under-use per role)
D. Generalised Estimating Equations (GEE) for correlated binary outcomes
E. Point-biserial correlations (role-specific content × correctness)
F. Role-specific content and pitch-invasion frequencies
G. Combination analysis (specific × pitch × correctness)
H. Majority / unanimity analysis

Outputs are written to ``output/`` (Excel) and ``output/advanced/`` (PNG).
"""

import itertools
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import pointbiserialr
from sklearn.metrics import cohen_kappa_score


from config import (
    DATA_FILE, OUTPUT_DIR_ADVANCED, OUTPUT_DIR_ROLE, SHOW_PLOTS, SPECIALIST_COLS,
    ROLE_PREFIX_MAP, COLUMNS_ANSWER, METHOD_TREATMENT_COLS, SIGNAL_COLUMNS
)
from utils import parse_treatment_list_column

# ---------------------------------------------------------------------------
# Directory setup
# ---------------------------------------------------------------------------
os.makedirs(OUTPUT_DIR_ROLE, exist_ok=True)
os.makedirs(OUTPUT_DIR_ADVANCED, exist_ok=True)

# ---------------------------------------------------------------------------
# Load and prepare data
# ---------------------------------------------------------------------------
df = pd.read_csv(DATA_FILE)

# ----------------------------------------------------------
# Ensure tumorboard_treatment is parsed as list
# and derive primary treatment (first element)
# ----------------------------------------------------------
df = parse_treatment_list_column(
    df,
    column="tumorboard_treatment",
    primary_output_col="tumorboard_primary_treatment"
)

# Columns to compare against the reference (excluding the reference itself)
comparison_cols = COLUMNS_ANSWER[1:]

print(df.columns)

REFERENCE_COL = "tumorboard_primary"

ROLE_TREATMENT_COLS = SPECIALIST_COLS


METHODS = (
    [REFERENCE_COL] +
    [c.replace("_treatment", "") for c in ROLE_TREATMENT_COLS] +
    ["F6_majority_vote"]
)

def _get_treatment_series(method: str) -> pd.Series:
    """Return the treatment series for a given method name."""
    if method == REFERENCE_COL:
        return df[f"{REFERENCE_COL}_treatment"]
    if method == "majority":
        return df["majority_treatment"]
    return df[f"{method}_treatment"]


# Ensure comparison columns exist
comparison_cols_base = [c.replace("_treatment", "") for c in ROLE_TREATMENT_COLS] + ["F6_majority_vote_treatment"]

comp_binary_cols = [f"{c}_concordance" for c in comparison_cols_base]


def _save_or_show(path: str) -> None:
    plt.savefig(path, dpi=300, bbox_inches="tight")
    print(f"Figure saved → {path}")
    if SHOW_PLOTS:
        plt.show()
    plt.close()


# ===========================================================================
# A. Per-diagnosis confusion matrices
# ===========================================================================

print("\n=== A. Per-diagnosis confusion matrices ===")

# Build role treatment column mapping using ROLE_PREFIX_MAP
# Example:
# Surgeon -> F3_persona_surgeon
# Oncologist -> F4_persona_medical_oncologist
# Radio-Oncologist -> F5_persona_radiation_oncologist

role_treatment_map = {
    role: f"{prefix}_treatment"
    for role, prefix in ROLE_PREFIX_MAP.items()
}

for diag in df["tumour_type"].unique():

    sub = df[df["tumour_type"] == diag]

    reference_series = sub[f"{REFERENCE_COL}_treatment"].fillna("MISSING")

    # Role-specific models
    for role_name, treatment_col in role_treatment_map.items():

        if treatment_col not in df.columns:
            continue

        y_true = reference_series
        y_pred = sub[treatment_col].fillna("MISSING")

        cm = pd.crosstab(
            y_true,
            y_pred,
            rownames=["Tumour Board"],
            colnames=[f"LLM ({role_name})"]
        )

        fname = f"confusion_{diag}_{role_name}.xlsx".replace("/", "-")
        cm.to_excel(f"{OUTPUT_DIR_ADVANCED}/{fname}")

        print(f"  Saved: {fname}")

# Majority vote (if exists)
if "F6_majority_vote_treatment" in df.columns:

    for diag in df["tumour_type"].unique():

        sub = df[df["tumour_type"] == diag]

        y_true = sub[f"{REFERENCE_COL}_treatment"].fillna("MISSING")
        y_pred = sub["F6_majority_vote_treatment"].fillna("MISSING")

        cm = pd.crosstab(
            y_true,
            y_pred,
            rownames=["Tumour Board"],
            colnames=["Majority Vote"]
        )

        fname = f"confusion_{diag}_majority_vote.xlsx".replace("/", "-")
        cm.to_excel(f"{OUTPUT_DIR_ADVANCED}/{fname}")

        print(f"  Saved: {fname}")

print("Confusion matrices complete.")


# ===========================================================================
# B. Cohen's kappa (method vs tumor board)
# ===========================================================================

print("\n=== B. Cohen's kappa ===")

kappa_rows = []

for method_name, role_col in METHOD_TREATMENT_COLS.items():

    if role_col not in df.columns:
        continue

    tb_labels = []
    role_labels = []

    for _, row in df.iterrows():

        tb_list = row["tumorboard_treatment"]
        role_val = row[role_col]

        if pd.isna(role_val) or not isinstance(tb_list, list):
            continue

        if role_val in tb_list:
            tb_labels.append(role_val)
            role_labels.append(role_val)
        else:
            tb_labels.append("DISAGREEMENT")
            role_labels.append(role_val)

    if len(tb_labels) == 0:
        k = np.nan
        n = 0
    else:
        k = cohen_kappa_score(tb_labels, role_labels)
        n = len(tb_labels)

    kappa_rows.append({
        "method": method_name,
        "kappa": k,
        "n": n
    })

kappa_df = pd.DataFrame(kappa_rows)

kappa_df.to_excel(
    f"{OUTPUT_DIR_ROLE}/kappa_vs_tumorboard.xlsx",
    index=False
)

print(kappa_df.to_string(index=False))

# ===========================================================================
# B2. Inter-method Cohen's kappa
# ===========================================================================

print("\n=== B2. Inter-method Cohen's kappa ===")

inter_rows = []

for (m1, c1), (m2, c2) in itertools.combinations(METHOD_TREATMENT_COLS.items(), 2):

    if c1 not in df.columns or c2 not in df.columns:
        continue

    s1 = df[c1].fillna("MISSING")
    s2 = df[c2].fillna("MISSING")

    mask = s1.notna() & s2.notna()

    if mask.sum() == 0:
        k = np.nan
    else:
        k = cohen_kappa_score(s1[mask], s2[mask])

    inter_rows.append({
        "method_A": m1,
        "method_B": m2,
        "kappa": k,
        "n": int(mask.sum())
    })

inter_df = pd.DataFrame(inter_rows)

inter_df.to_excel(
    f"{OUTPUT_DIR_ROLE}/inter_method_kappa.xlsx",
    index=False
)

print(inter_df.to_string(index=False))


# ===========================================================================
# C. Treatment frequency comparison (ordered distribution)
# ===========================================================================

print("\n=== C. Treatment frequency comparison ===")

# Frequencies
freqs = {
    m: _get_treatment_series(m).fillna("MISSING").value_counts()
    for m in METHODS
}

freqs_df = pd.DataFrame(freqs).fillna(0).astype(int)

# Remove MISSING category
if "MISSING" in freqs_df.index:
    freqs_df = freqs_df.drop(index="MISSING")

# ----------------------------------------
# Use ranking only to define category order
# ----------------------------------------
top_k = len(freqs_df)

top_treatments = (
    freqs_df.sum(axis=1)
    .sort_values(ascending=False)
    .head(top_k)
    .index.tolist()
)

freqs_df = freqs_df.loc[top_treatments]

# Save table
freqs_df.to_excel(
    f"{OUTPUT_DIR_ROLE}/treatment_frequencies_by_method.xlsx"
)

# Plot full distribution
freqs_df.plot(kind="bar", figsize=(12, 6))

plt.ylabel("Count")
plt.title("Treatment Distribution by Method")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()

_save_or_show(f"{OUTPUT_DIR_ADVANCED}/treatment_freqs_distribution.png")


# ===========================================================================
# D. Point-biserial correlations (correctness vs clinical behaviour signals)
# ===========================================================================

# ===========================================================================
# D. Point-biserial correlations (correctness vs clinical behaviour signals)
# ===========================================================================

# ===========================================================================
# D. Point-biserial correlations (correctness vs clinical behaviour signals)
# ===========================================================================

print("\n=== D. Point-biserial correlations ===")

corr_rows = []

FRAMEWORK_ROLE_SIGNAL_MAP = {
    "F3_persona_surgeon": {
        "role":        "Surgeon",
        "correct_col": "F3_persona_surgeon_treatment_concordance",
        "spec_col":    "F3_persona_surgeon_domain_content_present",
        "pitch_col":   "F3_persona_surgeon_boundary_violation",
    },
    "F4_persona_medical_oncologist": {
        "role":        "Oncologist",
        "correct_col": "F4_persona_medical_oncologist_treatment_concordance",
        "spec_col":    "F4_persona_medical_oncologist_domain_content_present",
        "pitch_col":   "F4_persona_medical_oncologist_boundary_violation",
    },
    "F5_persona_radiation_oncologist": {
        "role":        "Radio-Oncologist",
        "correct_col": "F5_persona_radiation_oncologist_treatment_concordance",
        "spec_col":    "F5_persona_radiation_oncologist_domain_content_present",
        "pitch_col":   "F5_persona_radiation_oncologist_boundary_violation",
    },
}

def _pointbiserial_safe(x: pd.Series, y: pd.Series):
    """Return (r, p, n) or (NaN, NaN, n) if computation is not possible."""
    mask = x.notna() & y.notna()
    x_clean = x[mask].astype(float)
    y_clean = y[mask].astype(float)
    n = len(x_clean)
    if n <= 5 or x_clean.nunique() < 2 or y_clean.nunique() < 2:
        return np.nan, np.nan, n
    r, p = pointbiserialr(x_clean, y_clean)
    return r, p, n

for fw_prefix, cfg in FRAMEWORK_ROLE_SIGNAL_MAP.items():

    role_name   = cfg["role"]
    correct_col = cfg["correct_col"]
    spec_col    = cfg["spec_col"]
    pitch_col   = cfg["pitch_col"]

    if correct_col not in df.columns:
        print(f"  Skipping {fw_prefix}: correctness column '{correct_col}' not found.")
        continue

    # ---- domain content present vs correctness ----
    if spec_col in df.columns:
        r, p, n = _pointbiserial_safe(df[spec_col], df[correct_col])
        print(f"  {fw_prefix} | {role_name} | specific_vs_correct → r={r:.4f}, p={p:.4f}, n={n}"
              if not np.isnan(r) else
              f"  {fw_prefix} | {role_name} | specific_vs_correct → NaN (constant or insufficient data), n={n}")
        corr_rows.append({
            "framework": fw_prefix,
            "role": role_name,
            "comparison": "specific_vs_correct",
            "r_pb": r,
            "pvalue": p,
            "n": n
        })
    else:
        print(f"  Column not found: {spec_col}")

    # ---- pitch invasion vs correctness ----
    if pitch_col in df.columns:
        r, p, n = _pointbiserial_safe(df[pitch_col], df[correct_col])
        print(f"  {fw_prefix} | {role_name} | pitch_vs_correct → r={r:.4f}, p={p:.4f}, n={n}"
              if not np.isnan(r) else
              f"  {fw_prefix} | {role_name} | pitch_vs_correct → NaN (constant or insufficient data), n={n}")
        corr_rows.append({
            "framework": fw_prefix,
            "role": role_name,
            "comparison": "pitch_vs_correct",
            "r_pb": r,
            "pvalue": p,
            "n": n
        })
    else:
        print(f"  Column not found: {pitch_col}")

corr_df = pd.DataFrame(corr_rows)

corr_df.to_excel(
    f"{OUTPUT_DIR_ROLE}/pointbiserial_correlations.xlsx",
    index=False
)

print("\nPoint-biserial correlation results:")
print(corr_df.to_string(index=False))

# ===========================================================================
# E. Role-specific content and pitch-invasion frequencies
# ===========================================================================

print("\n=== E. Role-specific content & pitch-invasion frequencies ===")



freq_rows = []

for fw_name, role_name, spec_col, pitch_col, _ in SIGNAL_COLUMNS:

    if spec_col in df.columns:
        freq_rows.append({
            "framework": fw_name,
            "role": role_name,
            "metric": "specific_rate",
            "percentage": float(df[spec_col].mean() * 100),
        })
    else:
        print(f"  Column not found: {spec_col}")

    if pitch_col in df.columns:
        freq_rows.append({
            "framework": fw_name,
            "role": role_name,
            "metric": "pitch_invasion_rate",
            "percentage": float(df[pitch_col].mean() * 100),
        })
    else:
        print(f"  Column not found: {pitch_col}")

freq_df = pd.DataFrame(freq_rows)
print(freq_df.to_string(index=False))

freq_df.to_excel(
    f"{OUTPUT_DIR_ROLE}/frequency_domain_content_present_pitch.xlsx",
    index=False
)

# ---------------------------------------------------------------------------
# Summary barplot
# ---------------------------------------------------------------------------

fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

for ax, metric in zip(axes, ["specific_rate", "pitch_invasion_rate"]):

    sub = freq_df[freq_df["metric"] == metric]

    sns.barplot(
        data=sub,
        x="role",
        y="percentage",
        hue="framework",
        palette="Set2",
        ax=ax
    )

    ax.set_xlabel("Role")
    ax.set_ylabel("Percentage (%)")
    ax.set_title(metric.replace("_", " ").title())
    ax.set_ylim(0, 100)
    ax.tick_params(axis="x", rotation=25)

plt.suptitle(
    "Role-specific Content and Boundary Violation Frequencies",
    fontsize=14
)

plt.tight_layout()

_save_or_show(
    f"{OUTPUT_DIR_ROLE}/role_domain_content_present_pitch_frequencies.png"
)


# ===========================================================================
# F. Combination analysis (specific × pitch × correctness)
# ===========================================================================

print("\n=== F. Combination analysis (specific × pitch × correctness) ===")


CATEGORIES = {
    "Role-specific only": lambda rs, pi: (rs == 1) & (pi == 0),
    "Pitch invasion only": lambda rs, pi: (rs == 0) & (pi == 1),
    "Both":               lambda rs, pi: (rs == 1) & (pi == 1),
    "Neither":            lambda rs, pi: (rs == 0) & (pi == 0),
}

combo_rows = []

for fw_name, role_name, spec_col, pitch_col, correct_col in SIGNAL_COLUMNS:

    missing = [c for c in [spec_col, pitch_col, correct_col] if c not in df.columns]
    if missing:
        print(f"  Skipping {fw_name} | {role_name}: columns not found: {missing}")
        continue

    rs  = df[spec_col].fillna(0).astype(int)
    pi  = df[pitch_col].fillna(0).astype(int)
    cor = df[correct_col].fillna(0).astype(int)

    for cat, mask_fn in CATEGORIES.items():
        mask = mask_fn(rs, pi)
        n    = int(mask.sum())
        acc  = float(cor[mask].mean() * 100) if n > 0 else np.nan

        combo_rows.append({
            "framework":    fw_name,
            "role":         role_name,
            "category":     cat,
            "n":            n,
            "accuracy_pct": acc,
        })

combo_df = pd.DataFrame(combo_rows)
combo_df.to_excel(f"{OUTPUT_DIR_ROLE}/combo_analysis.xlsx", index=False)
print(combo_df.to_string(index=False))


# ===========================================================================
# G. Majority / unanimity analysis
# ===========================================================================

print("\n=== G. Majority / unanimity analysis ===")

if "Unique_Answers_Count" not in df.columns:
    df["Unique_Answers_Count"] = df[ROLE_TREATMENT_COLS].apply(
        lambda row: len(set(row.dropna())), axis=1
    )

df["unanimous"]  = df["Unique_Answers_Count"] == 1
unanimity_rate   = float(df["unanimous"].mean() * 100)
print(f"Unanimity rate: {unanimity_rate:.1f}% (N = {len(df)})")

if "F6_majority_vote_treatment_concordance" in df.columns:
    maj_correct = float(df["F6_majority_vote_treatment_concordance"].mean() * 100)
    print(f"Majority vote correctness: {maj_correct:.1f}%")

    # Cases where majority is wrong but at least one role is correct
    mask_wrong_maj = (df["F6_majority_vote_treatment_concordance"] == 0)
    mask_any_right = (
        (df.get("F3_persona_surgeon_treatment_concordance",       pd.Series(0, index=df.index)) == 1) |
        (df.get("F4_persona_oncologist_treatment_concordance",    pd.Series(0, index=df.index)) == 1) |
        (df.get("F5_persona_radio-oncologist_treatment_concordance", pd.Series(0, index=df.index)) == 1)
    )
    special_cases = df[mask_wrong_maj & mask_any_right]
    print(f"Cases where majority wrong but ≥1 role correct: {len(special_cases)}")
    special_cases.to_excel(
        f"{OUTPUT_DIR_ROLE}/majority_wrong_but_one_correct.xlsx", index=False
    )
else:
    print("F6_majority_vote_concordance column not found — skipping majority accuracy.")

# Save unanimity summary
pd.DataFrame({
    "metric": ["unanimity_rate_pct", "n_patients"],
    "value":  [unanimity_rate, len(df)],
}).to_excel(f"{OUTPUT_DIR_ROLE}/unanimity_summary.xlsx", index=False)

print("\n=== Advanced Analysis Complete ===")
