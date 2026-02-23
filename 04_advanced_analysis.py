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

Outputs are written to ``role/`` (Excel) and ``img/advanced/`` (PNG).
"""

import itertools
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import pointbiserialr
from sklearn.metrics import cohen_kappa_score

import statsmodels.formula.api as smf
from statsmodels.genmod.families import Binomial
from statsmodels.genmod.generalized_estimating_equations import GEE
from statsmodels.genmod.cov_struct import Exchangeable

from config import (
    DATA_FILE, OUTPUT_DIR_ADVANCED, OUTPUT_DIR_ROLE, SHOW_PLOTS, SPECIALIST_COLS,
    VALUE_RENAME,
)
from utils import compare_treatments, compute_majority_treatment

# ---------------------------------------------------------------------------
# Directory setup
# ---------------------------------------------------------------------------
os.makedirs(OUTPUT_DIR_ROLE, exist_ok=True)
os.makedirs(OUTPUT_DIR_ADVANCED, exist_ok=True)

# ---------------------------------------------------------------------------
# Load and prepare data
# ---------------------------------------------------------------------------
df = pd.read_excel(DATA_FILE)
df["majority"] = df.apply(
    lambda row: compute_majority_treatment(row, SPECIALIST_COLS), axis=1
)
df["majority_treatment"] = df["majority"]

REFERENCE_COL = "Konferenzbeschluss"
ROLES         = ["surgeon", "oncologist", "radio-oncologist"]

ROLE_TREATMENT_COLS = [
    "ChatGPT_single_request_5_surgeon_treatment",
    "ChatGPT_single_request_5_oncologist_treatment",
    "ChatGPT_single_request_5_radio-oncologist_treatment",
]

METHODS = (
    [REFERENCE_COL] +
    [c.replace("_treatment", "") for c in ROLE_TREATMENT_COLS] +
    ["majority"]
)

def _get_treatment_series(method: str) -> pd.Series:
    """Return the treatment series for a given method name."""
    if method == REFERENCE_COL:
        return df[f"{REFERENCE_COL}_treatment"]
    if method == "majority":
        return df["majority_treatment"]
    return df[f"{method}_treatment"]


# Ensure comparison columns exist
comparison_cols_base = [c.replace("_treatment", "") for c in ROLE_TREATMENT_COLS] + ["majority"]
df = compare_treatments(df, REFERENCE_COL, comparison_cols_base)

comp_binary_cols = [f"{c}_comparison" for c in comparison_cols_base]


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

for diag in df["Anmeldediagnose"].unique():
    sub = df[df["Anmeldediagnose"] == diag]
    for role_col in ROLE_TREATMENT_COLS + ["majority_treatment"]:
        role_label = role_col.replace("_treatment", "").replace("ChatGPT_single_request_5_", "")
        y_true = sub[f"{REFERENCE_COL}_treatment"].fillna("MISSING")
        y_pred = sub[role_col].fillna("MISSING")
        cm = pd.crosstab(y_true, y_pred,
                         rownames=["Tumour Board"], colnames=[f"LLM ({role_label})"])
        fname = f"confusion_{diag}_{role_label}.xlsx".replace("/", "-")
        cm.to_excel(f"{OUTPUT_DIR_ADVANCED}/{fname}")
        print(f"  Saved: {fname}")

print("Confusion matrices complete.")


# ===========================================================================
# B. Cohen's kappa
# ===========================================================================

print("\n=== B. Cohen's kappa ===")

kappa_rows = []
for m1, m2 in itertools.combinations(METHODS, 2):
    s1 = _get_treatment_series(m1).fillna("MISSING")
    s2 = _get_treatment_series(m2).fillna("MISSING")
    mask = s1.notna() & s2.notna()
    try:
        k = cohen_kappa_score(s1[mask], s2[mask])
    except Exception:
        k = np.nan
    kappa_rows.append({"method_A": m1, "method_B": m2, "kappa": k, "n": int(mask.sum())})

kappa_df = pd.DataFrame(kappa_rows)
kappa_df.to_excel(f"{OUTPUT_DIR_ROLE}/kappa_results.xlsx", index=False)
print(kappa_df.to_string(index=False))


# ===========================================================================
# C. Treatment frequency comparison
# ===========================================================================

print("\n=== C. Treatment frequency comparison ===")

freqs = {m: _get_treatment_series(m).fillna("MISSING").value_counts().sort_index()
         for m in METHODS}
freqs_df = pd.DataFrame(freqs).fillna(0).astype(int)
freqs_df.to_excel(f"{OUTPUT_DIR_ROLE}/treatment_frequencies_by_method.xlsx")

# Bar chart of top K treatments
top_k = 10
top_treatments = (
    freqs_df.sum(axis=1).sort_values(ascending=False).head(top_k).index.tolist()
)
freqs_df.loc[top_treatments].plot(kind="bar", figsize=(12, 6))
plt.ylabel("Count")
plt.title(f"Top {top_k} Treatment Frequencies by Method")
plt.tight_layout()
_save_or_show(f"{OUTPUT_DIR_ADVANCED}/treatment_freqs_topk.png")


# ===========================================================================
# D. Generalised Estimating Equations (GEE)
# ===========================================================================

print("\n=== D. GEE — role effect on treatment correctness ===")

# Reshape to long format: one row per patient × method
long_rows = []
for role_col in ROLE_TREATMENT_COLS:
    role = role_col.replace("ChatGPT_single_request_5_", "").replace("_treatment", "")
    comp_col = f"ChatGPT_single_request_5_{role}_comparison"
    spec_col = f"ChatGPT_single_request_5_{role}_specific"
    if comp_col not in df.columns:
        continue
    for pid, row in df.iterrows():
        long_rows.append({
            "patient_id": pid,
            "role":        role,
            "correct":     int(row[comp_col]),
            "specific":    int(row[spec_col]) if spec_col in df.columns and pd.notna(row.get(spec_col)) else 0,
        })

long_df = pd.DataFrame(long_rows)

if not long_df.empty:
    try:
        model = GEE.from_formula(
            "correct ~ C(role)",
            groups="patient_id",
            data=long_df,
            family=Binomial(),
            cov_struct=Exchangeable(),
        )
        result = model.fit()
        print(result.summary())
        gee_params = result.params.reset_index()
        gee_params.columns = ["term", "coefficient"]
        gee_params["pvalue"] = result.pvalues.values
        gee_params.to_excel(f"{OUTPUT_DIR_ROLE}/gee_role_effect.xlsx", index=False)
    except Exception as e:
        print(f"  GEE failed: {e}")
else:
    print("  Long-format dataset is empty — skipping GEE.")


# ===========================================================================
# E. Point-biserial correlations (role-specific content × correctness)
# ===========================================================================

print("\n=== E. Point-biserial correlations ===")

FRAMEWORKS = {
    "single":           "ChatGPT_single_request_5_{}",
    "self_consistency": "ChatGPT_single_request_5_self-consistency_{}",
}

corr_rows = []
for fw_name, fw_pattern in FRAMEWORKS.items():
    for role in ROLES:
        base = fw_pattern.format(role)

        # Role-specific content × correctness
        spec_col    = f"{base}_specific"
        correct_col = f"{base}_comparison"
        if spec_col in df.columns and correct_col in df.columns:
            mask = df[[spec_col, correct_col]].dropna().index
            if len(mask) > 5:
                r, p = pointbiserialr(
                    df.loc[mask, spec_col].astype(float),
                    df.loc[mask, correct_col].astype(float),
                )
            else:
                r, p = np.nan, np.nan
            corr_rows.append(dict(framework=fw_name, role=role,
                                  comparison="specific_vs_correct",
                                  r_pb=r, pvalue=p, n=len(mask)))

        # Pitch invasion × correctness
        pitch_col = f"{base}_pitch_invasion"
        if pitch_col in df.columns and correct_col in df.columns:
            mask = df[[pitch_col, correct_col]].dropna().index
            if len(mask) > 5:
                r, p = pointbiserialr(
                    df.loc[mask, pitch_col].astype(float),
                    df.loc[mask, correct_col].astype(float),
                )
            else:
                r, p = np.nan, np.nan
            corr_rows.append(dict(framework=fw_name, role=role,
                                  comparison="pitch_vs_correct",
                                  r_pb=r, pvalue=p, n=len(mask)))

corr_df = pd.DataFrame(corr_rows)
corr_df.to_excel(f"{OUTPUT_DIR_ROLE}/pointbiserial_correlations.xlsx", index=False)
print(corr_df.to_string(index=False))


# ===========================================================================
# F. Role-specific content and pitch-invasion frequencies
# ===========================================================================

print("\n=== F. Role-specific content & pitch-invasion frequencies ===")

role_specific_cols  = [f"ChatGPT_single_request_5_{r}_specific"        for r in ROLES]
role_pitch_inv_cols = [f"ChatGPT_single_request_5_{r}_pitch_invasion"   for r in ROLES]

freq_rows = []
for fw_name, fw_pattern in FRAMEWORKS.items():
    for role in ROLES:
        base = fw_pattern.format(role)
        for metric, col_suffix in [("specific_rate", "_specific"),
                                   ("pitch_invasion_rate", "_pitch_invasion")]:
            col = f"{base}{col_suffix}"
            if col in df.columns:
                freq_rows.append({
                    "framework": fw_name, "role": role,
                    "metric": metric,
                    "percentage": float(df[col].mean() * 100),
                })

freq_df = pd.DataFrame(freq_rows)
freq_df.to_excel(f"{OUTPUT_DIR_ROLE}/frequency_specific_pitch.xlsx", index=False)

# Summary barplot
fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
for ax, metric in zip(axes, ["specific_rate", "pitch_invasion_rate"]):
    sub = freq_df[freq_df["metric"] == metric]
    sns.barplot(data=sub, x="role", y="percentage", hue="framework",
                palette="Set2", ax=ax)
    ax.set_xlabel("Role")
    ax.set_ylabel("Percentage (%)")
    ax.set_title(metric.replace("_", " ").title())
    ax.set_ylim(0, 100)
    ax.tick_params(axis="x", rotation=25)

plt.suptitle("Role-specific Content and Pitch-invasion Frequencies", fontsize=14)
plt.tight_layout()
_save_or_show(f"{OUTPUT_DIR_ROLE}/role_specific_pitch_frequencies.png")


# ===========================================================================
# G. Combination analysis (specific × pitch × correctness)
# ===========================================================================

print("\n=== G. Combination analysis (specific × pitch × correctness) ===")

combo_rows = []
CATEGORIES = [
    "Role-specific only",
    "Pitch invasion only",
    "Both",
    "Neither",
]
for fw_name, fw_pattern in FRAMEWORKS.items():
    for role in ROLES:
        base = fw_pattern.format(role)
        rs_col  = f"{base}_specific"
        pi_col  = f"{base}_pitch_invasion"
        cor_col = f"{base}_comparison"
        if not all(c in df.columns for c in [rs_col, pi_col, cor_col]):
            continue
        rs  = df[rs_col].fillna(0).astype(int)
        pi  = df[pi_col].fillna(0).astype(int)
        cor = df[cor_col].fillna(0).astype(int)
        masks = {
            "Role-specific only": (rs == 1) & (pi == 0),
            "Pitch invasion only": (rs == 0) & (pi == 1),
            "Both":    (rs == 1) & (pi == 1),
            "Neither": (rs == 0) & (pi == 0),
        }
        for cat, mask in masks.items():
            n   = int(mask.sum())
            acc = float(cor[mask].mean() * 100) if n > 0 else np.nan
            combo_rows.append(dict(framework=fw_name, role=role,
                                   category=cat, n=n, accuracy_pct=acc))

combo_df = pd.DataFrame(combo_rows)
combo_df.to_excel(f"{OUTPUT_DIR_ROLE}/combo_analysis.xlsx", index=False)
print(combo_df.to_string(index=False))


# ===========================================================================
# H. Majority / unanimity analysis
# ===========================================================================

print("\n=== H. Majority / unanimity analysis ===")

if "Unique_Answers_Count" not in df.columns:
    df["Unique_Answers_Count"] = df[ROLE_TREATMENT_COLS].apply(
        lambda row: len(set(row.dropna())), axis=1
    )

df["unanimous"]  = df["Unique_Answers_Count"] == 1
unanimity_rate   = float(df["unanimous"].mean() * 100)
print(f"Unanimity rate: {unanimity_rate:.1f}% (N = {len(df)})")

if "majority_comparison" in df.columns:
    maj_correct = float(df["majority_comparison"].mean() * 100)
    print(f"Majority vote correctness: {maj_correct:.1f}%")

    # Cases where majority is wrong but at least one role is correct
    mask_wrong_maj = (df["majority_comparison"] == 0)
    mask_any_right = (
        (df.get("ChatGPT_single_request_5_surgeon_comparison",       pd.Series(0, index=df.index)) == 1) |
        (df.get("ChatGPT_single_request_5_oncologist_comparison",    pd.Series(0, index=df.index)) == 1) |
        (df.get("ChatGPT_single_request_5_radio-oncologist_comparison", pd.Series(0, index=df.index)) == 1)
    )
    special_cases = df[mask_wrong_maj & mask_any_right]
    print(f"Cases where majority wrong but ≥1 role correct: {len(special_cases)}")
    special_cases.to_excel(
        f"{OUTPUT_DIR_ROLE}/majority_wrong_but_one_correct.xlsx", index=False
    )
else:
    print("majority_comparison column not found — skipping majority accuracy.")

# Save unanimity summary
pd.DataFrame({
    "metric": ["unanimity_rate_pct", "n_patients"],
    "value":  [unanimity_rate, len(df)],
}).to_excel(f"{OUTPUT_DIR_ROLE}/unanimity_summary.xlsx", index=False)

print("\n=== Advanced Analysis Complete ===")
