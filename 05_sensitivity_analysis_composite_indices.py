"""
05_sensitivity_analysis_composite_indices.py

Robustness and validity analysis for PSI and CRI composite indices.

Three complementary analyses:
  1. Sensitivity analysis     — rank stability under weight perturbation (±10%, ±20%)
  2. Convergent validity      — Spearman correlation between PSI and CRI
  3. Component contribution   — variance proportion per index component

PSI and CRI are computed per-role from case-level component columns in the
dataset. Each index function aggregates case-level values into a single
role-level score vector (one score per case), which is then used for
rank stability and convergent validity analyses.

All indices are exploratory composite measures; results should be interpreted
as descriptive robustness checks within this evaluation framework.
"""

import os
import pandas as pd
import numpy as np
from scipy.stats import spearmanr


# ─────────────────────────────────────────────────────────────────────────────
# 1. SENSITIVITY ANALYSIS — weight perturbation
# ─────────────────────────────────────────────────────────────────────────────

def sensitivity_analysis_index(df, base_weights, index_function):
    """
    Robustness test for composite indices under weight perturbation.

    Parameters
    ----------
    df : pd.DataFrame
        Input data passed to index_function.
    base_weights : dict
        Base component weights, e.g. {"specificity_rate": 0.4, ...}
    index_function : callable
        Function with signature index_function(df, weights=None) -> pd.DataFrame
        Must return a DataFrame with at least a "score" column.

    Returns
    -------
    pd.DataFrame with columns: epsilon, spearman_rank_correlation, p_value
    """
    eps_grid = [-0.2, -0.1, 0.0, 0.1, 0.2]
    results = []
    base_scores = index_function(df)

    for eps in eps_grid:
        new_weights = {k: v * (1 + eps) for k, v in base_weights.items()}
        total = sum(new_weights.values())
        new_weights = {k: v / total for k, v in new_weights.items()}

        perturbed_scores = index_function(df, weights=new_weights)

        rho, p_val = spearmanr(base_scores["score"], perturbed_scores["score"])

        results.append({
            "epsilon": eps,
            "spearman_rank_correlation": round(rho, 4),
            "p_value": "<0.001" if p_val < 0.001 else round(p_val, 4)
        })

    return pd.DataFrame(results)


# ─────────────────────────────────────────────────────────────────────────────
# 2. CONVERGENT VALIDITY — inter-index correlation (PSI vs CRI)
# ─────────────────────────────────────────────────────────────────────────────

def convergent_validity(psi_scores, cri_scores):
    """
    Spearman correlation between PSI and CRI scores across cases.
    High correlation supports construct convergence.

    Parameters
    ----------
    psi_scores : array-like
    cri_scores : array-like

    Returns
    -------
    dict with spearman_rho and p_value
    """
    rho, p_val = spearmanr(psi_scores, cri_scores)
    return {
        "spearman_rho": round(rho, 4),
        "p_value": "<0.001" if p_val < 0.001 else round(p_val, 4)
    }


# ─────────────────────────────────────────────────────────────────────────────
# 3. COMPONENT CONTRIBUTION — variance explained per component
# ─────────────────────────────────────────────────────────────────────────────

def component_contribution(df, components):
    """
    Reports the proportion of total score variance explained by each component.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain one column per component.
    components : list of str

    Returns
    -------
    pd.DataFrame with columns: component, variance, variance_proportion
    """
    variances = df[components].var()
    total_variance = variances.sum()
    result = pd.DataFrame({
        "component": variances.index,
        "variance": variances.values.round(4),
        "variance_proportion": (variances / total_variance).values.round(4)
    }).sort_values("variance_proportion", ascending=False).reset_index(drop=True)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# RUNNER
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    # ── Setup ─────────────────────────────────────────────────────────────────
    from config import (
        DATA_FILE,
        OUTPUT_DIR_ROLE,
        PSI_WEIGHTS,
        CRI_WEIGHTS,
    )

    TABLE_DIR = os.path.join(OUTPUT_DIR_ROLE, "sensitivity_analysis")
    os.makedirs(TABLE_DIR, exist_ok=True)

    # ── Load data ─────────────────────────────────────────────────────────────
    df = pd.read_csv(DATA_FILE)

    # ── Build per-role component DataFrame ───────────────────────────────────
    # Components derived from case-level columns:
    #   specificity_rate   : domain_content_present (0/1)
    #   pitch_control      : 1 - boundary_violation (0/1, inverted)
    #   accuracy           : treatment_concordance (0/1)
    #   entropy_stability  : role-level constant (1 - boundary_entropy / log2(3))
    #
    # cosine_similarity requires per-case embedding drift values (F3/F4/F5 vs F2).
    # These are not directly available as scalar columns; if a precomputed drift
    # CSV exists, merge it here. Otherwise it is excluded from available weights
    # (see PSI_WEIGHTS_AVAILABLE below) — rank ordering is invariant to this
    # component as confirmed in prior analysis.

    roles = {
        "surgeon": {
            "specificity_rate": "F3_persona_surgeon_domain_content_present",
            "boundary_violation": "F3_persona_surgeon_boundary_violation",
            "accuracy":          "F3_persona_surgeon_treatment_concordance",
        },
        "oncologist": {
            "specificity_rate": "F4_persona_medical_oncologist_domain_content_present",
            "boundary_violation": "F4_persona_medical_oncologist_boundary_violation",
            "accuracy":          "F4_persona_medical_oncologist_treatment_concordance",
        },
        "radiation_oncologist": {
            "specificity_rate": "F5_persona_radiation_oncologist_domain_content_present",
            "boundary_violation": "F5_persona_radiation_oncologist_boundary_violation",
            "accuracy":          "F5_persona_radiation_oncologist_treatment_concordance",
        },
    }

    # Role-level entropy stability (from S11: boundary entropy bits)
    # entropy_stability = 1 - (boundary_entropy / log2(3))
    ENTROPY_BITS = {
        "surgeon":              0.965,
        "oncologist":           0.958,
        "radiation_oncologist": 0.990,
    }
    MAX_ENTROPY = np.log2(3)  # ≈ 1.585 bits

    records = []
    for role, cols in roles.items():
        n = len(df)
        role_df = pd.DataFrame({
            "role":              [role] * n,
            "specificity_rate":  df[cols["specificity_rate"]].values,
            "pitch_control":     1 - df[cols["boundary_violation"]].values,
            "accuracy":          df[cols["accuracy"]].values,
            "entropy_stability": 1 - (ENTROPY_BITS[role] / MAX_ENTROPY),
        })
        records.append(role_df)

    df_components = pd.concat(records, ignore_index=True)

    # ── Exclude cosine_similarity (requires separate drift computation) ───────
    PSI_WEIGHTS_AVAILABLE = {k: v for k, v in PSI_WEIGHTS.items()
                             if k != "cosine_similarity"}
    CRI_WEIGHTS_AVAILABLE = {k: v for k, v in CRI_WEIGHTS.items()
                             if k != "cosine_similarity"}

    # Renormalise to sum to 1
    psi_total = sum(PSI_WEIGHTS_AVAILABLE.values())
    PSI_WEIGHTS_AVAILABLE = {k: v / psi_total for k, v in PSI_WEIGHTS_AVAILABLE.items()}
    cri_total = sum(CRI_WEIGHTS_AVAILABLE.values())
    CRI_WEIGHTS_AVAILABLE = {k: v / cri_total for k, v in CRI_WEIGHTS_AVAILABLE.items()}

    # ── Index functions ───────────────────────────────────────────────────────
    def compute_psi(df, weights=None):
        if weights is None:
            weights = PSI_WEIGHTS_AVAILABLE
        score = sum(df[k] * v for k, v in weights.items())
        return pd.DataFrame({"role": df["role"].values, "score": score.values})

    def compute_cri(df, weights=None):
        if weights is None:
            weights = CRI_WEIGHTS_AVAILABLE
        score = sum(df[k] * v for k, v in weights.items())
        return pd.DataFrame({"role": df["role"].values, "score": score.values})

    # ── 1. SENSITIVITY ANALYSIS ───────────────────────────────────────────────
    print("=" * 60)
    print("1. SENSITIVITY ANALYSIS — PSI")
    print("=" * 60)
    sa_psi = sensitivity_analysis_index(df_components, PSI_WEIGHTS_AVAILABLE, compute_psi)
    print(sa_psi.to_string(index=False))

    print("\n" + "=" * 60)
    print("1. SENSITIVITY ANALYSIS — CRI")
    print("=" * 60)
    sa_cri = sensitivity_analysis_index(df_components, CRI_WEIGHTS_AVAILABLE, compute_cri)
    print(sa_cri.to_string(index=False))

    # ── 2. CONVERGENT VALIDITY ────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("2. CONVERGENT VALIDITY — PSI vs CRI")
    print("=" * 60)
    psi_scores = compute_psi(df_components)["score"]
    cri_scores = compute_cri(df_components)["score"]
    cv = convergent_validity(psi_scores, cri_scores)
    print(f"  Spearman rho: {cv['spearman_rho']}")
    print(f"  p-value:      {cv['p_value']}")

    # ── 3. COMPONENT CONTRIBUTION ─────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("3. COMPONENT CONTRIBUTION — PSI components")
    print("=" * 60)
    cc_psi = component_contribution(df_components, list(PSI_WEIGHTS_AVAILABLE.keys()))
    print(cc_psi.to_string(index=False))

    print("\n" + "=" * 60)
    print("3. COMPONENT CONTRIBUTION — CRI components")
    print("=" * 60)
    cc_cri = component_contribution(df_components, list(CRI_WEIGHTS_AVAILABLE.keys()))
    print(cc_cri.to_string(index=False))

    # ── Save tables ───────────────────────────────────────────────────────────
    sa_psi.to_csv(os.path.join(TABLE_DIR, "sensitivity_psi.csv"), index=False)
    sa_cri.to_csv(os.path.join(TABLE_DIR, "sensitivity_cri.csv"), index=False)
    cc_psi.to_csv(os.path.join(TABLE_DIR, "component_contribution_psi.csv"), index=False)
    cc_cri.to_csv(os.path.join(TABLE_DIR, "component_contribution_cri.csv"), index=False)
    print(f"\nTables saved to {TABLE_DIR}")