"""
05_sensitivity_analysis_composite_indices.py

Robustness and validity analysis for PSI and CRI composite indices.

Two complementary analyses:
  1. Sensitivity analysis — rank stability under weight perturbation (±20%)
  2. Permutation null — observed index vs. null distribution under label shuffle

Requires:
  - A DataFrame `df` with per-case role-level scores for each PSI/CRI component
  - An `index_function(df, weights=None)` that returns a DataFrame with a "score" column
    and a "role" column (one row per role per case, or one row per case)

Author: [Author]
"""

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import matplotlib.pyplot as plt


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
        Base component weights, e.g. {"semantic_sim": 0.4, "role_specificity": 0.4, ...}
    index_function : callable
        Function with signature index_function(df, weights=None) -> pd.DataFrame
        Must return a DataFrame with at least a "score" column.

    Returns
    -------
    pd.DataFrame with columns: epsilon, spearman_rank_correlation
    """
    eps_grid = [-0.2, -0.1, 0.0, 0.1, 0.2]
    results = []
    base_scores = index_function(df)

    for eps in eps_grid:
        # Perturb weights
        new_weights = {k: v * (1 + eps) for k, v in base_weights.items()}

        # Renormalise to sum to 1
        total = sum(new_weights.values())
        new_weights = {k: v / total for k, v in new_weights.items()}

        # Compute perturbed index values
        perturbed_scores = index_function(df, weights=new_weights)

        # Rank stability via Spearman correlation
        rho, p_val = spearmanr(base_scores["score"], perturbed_scores["score"])

        results.append({
            "epsilon": eps,
            "spearman_rank_correlation": round(rho, 4),
            "p_value": round(p_val, 4)
        })

    return pd.DataFrame(results)


# ─────────────────────────────────────────────────────────────────────────────
# 2. CONVERGENT VALIDITY — inter-index correlation (PSI vs CRI)
# ─────────────────────────────────────────────────────────────────────────────

def convergent_validity(psi_scores, cri_scores):
    """
    Spearman correlation between PSI and CRI scores across cases/roles.
    High correlation supports construct convergence.

    Parameters
    ----------
    psi_scores : array-like
        PSI scores per case or per role.
    cri_scores : array-like
        CRI scores, same ordering as psi_scores.

    Returns
    -------
    dict with rho and p_value
    """
    rho, p_val = spearmanr(psi_scores, cri_scores)
    return {
        "spearman_rho": round(rho, 4),
        "p_value": round(p_val, 4)
    }


# ─────────────────────────────────────────────────────────────────────────────
# 3. COMPONENT CONTRIBUTION — variance explained per component
# ─────────────────────────────────────────────────────────────────────────────

def component_contribution(df, components):
    """
    Reports the proportion of total index score variance explained by each
    component, to detect dominant components that reduce composite value.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain one column per component named as in `components`.
    components : list of str
        Column names corresponding to index components.

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
# RUNNER — replace mock data with your actual df and index functions
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    # ── MOCK DATA (replace with actual loaded DataFrame) ──────────────────────
    np.random.seed(42)
    n = 100
    roles = np.tile(["surgeon", "oncologist", "radiation_oncologist"], n // 3 + 1)[:n]
    df_mock = pd.DataFrame({
        "role": roles,
        "semantic_sim": np.random.beta(8, 2, n),
        "role_specificity": np.random.beta(7, 2, n),
        "boundary_control": np.random.beta(6, 3, n),
        "concordance": np.random.beta(5, 2, n),
    })

    # ── MOCK INDEX FUNCTIONS (replace with your actual PSI / CRI functions) ───
    PSI_BASE_WEIGHTS = {"semantic_sim": 0.4, "role_specificity": 0.4, "boundary_control": 0.2}
    CRI_BASE_WEIGHTS = {"semantic_sim": 0.35, "role_specificity": 0.35,
                        "boundary_control": 0.15, "concordance": 0.15}

    def compute_psi(df, weights=None):
        if weights is None:
            weights = PSI_BASE_WEIGHTS
        score = sum(df[k] * v for k, v in weights.items())
        return pd.DataFrame({"role": df["role"], "score": score})

    def compute_cri(df, weights=None):
        if weights is None:
            weights = CRI_BASE_WEIGHTS
        score = sum(df[k] * v for k, v in weights.items())
        return pd.DataFrame({"role": df["role"], "score": score})

    # ── 1. SENSITIVITY ANALYSIS ───────────────────────────────────────────────
    print("=" * 60)
    print("1. SENSITIVITY ANALYSIS — PSI")
    print("=" * 60)
    sa_psi = sensitivity_analysis_index(df_mock, PSI_BASE_WEIGHTS, compute_psi)
    print(sa_psi.to_string(index=False))

    print("\n" + "=" * 60)
    print("1. SENSITIVITY ANALYSIS — CRI")
    print("=" * 60)
    sa_cri = sensitivity_analysis_index(df_mock, CRI_BASE_WEIGHTS, compute_cri)
    print(sa_cri.to_string(index=False))

    # ── 2. CONVERGENT VALIDITY ────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("2. CONVERGENT VALIDITY — PSI vs CRI")
    print("=" * 60)
    psi_scores = compute_psi(df_mock)["score"]
    cri_scores = compute_cri(df_mock)["score"]
    cv = convergent_validity(psi_scores, cri_scores)
    print(f"  Spearman rho: {cv['spearman_rho']}")
    print(f"  p-value:      {cv['p_value']}")

    # ── 3. COMPONENT CONTRIBUTION ─────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("3. COMPONENT CONTRIBUTION — PSI components")
    print("=" * 60)
    cc_psi = component_contribution(df_mock, list(PSI_BASE_WEIGHTS.keys()))
    print(cc_psi.to_string(index=False))

    print("\n" + "=" * 60)
    print("3. COMPONENT CONTRIBUTION — CRI components")
    print("=" * 60)
    cc_cri = component_contribution(df_mock, list(CRI_BASE_WEIGHTS.keys()))
    print(cc_cri.to_string(index=False))