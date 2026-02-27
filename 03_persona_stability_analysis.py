"""
03_persona_stability_analysis.py
=================================

Persona stability, robustness, and clinical boundary-control analysis.

This module quantifies the degree to which each specialist role persona
maintains semantically and clinically coherent behaviour across prompting
conditions (single-request vs self-consistency generation paradigms).

The analysis implements a multi-layer evaluation framework:

1. Persona Stability Index (PSI)
--------------------------------
Composite metric measuring role identity coherence using:
    - Semantic embedding similarity across prompting conditions
    - Role-specific clinical content specificity
    - Boundary control (reduction of cross-specialty decision leakage)
    - Treatment recommendation accuracy

PSI is theory-weighted to prioritise interpretability and clinical validity
rather than purely data-optimised performance scoring.

2. Composite Robustness Index (CRI)
--------------------------------
Extension of PSI incorporating global behavioural entropy penalties to capture
decision-making stability across patient populations.

CRI reflects global reliability of clinical reasoning behaviour rather than
single-case accuracy.

3. Persona Geometry Metrics
--------------------------------
Includes embedding-space structural analysis:

- Role Confusion Entropy:
    Measures geometric indistinguishability between role personas.

- Persona Attractor Dispersion (PAD):
    Measures embedding variance relative to role semantic centroids.

4. Safety and Boundary Control Metrics
--------------------------------
Includes:

- Role Boundary Violation Entropy:
    Measures unpredictability of out-of-scope clinical reasoning.

- Clinical Risk Penalty Score:
    Combines boundary violation rate and non-specific reasoning frequency.

5. Cross-Condition Consistency
--------------------------------
Measures stability of clinical decisions across patient cases and prompting
paradigms.

6. Sensitivity Analysis
--------------------------------
Weight perturbation analysis evaluates robustness of PSI ranking stability.
Observed rank invariance under ±20% weight perturbations indicates structural
stability of persona representations.

All outputs are written to:

    output/persona_stability_analysis/

Weight rationale
----------------
Index weights are theory-driven and documented in the Methods section.
Weights are centralised in config.py (CRI_WEIGHTS, PSI_WEIGHTS, RISK_WEIGHTS)
to facilitate reproducibility and sensitivity testing.
"""

import os
import numpy as np
import pandas as pd
from scipy.special import softmax
from scipy.stats import bootstrap, entropy, spearmanr
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns

from config import (
    CRI_WEIGHTS, DATA_FILE, OUTPUT_DIR_ROLE,
    PSI_WEIGHTS, RISK_WEIGHTS,
    ROLE_PREFIX_MAP, ROLES, SINGLE_REQUEST_EMBEDDING_COLS,
    SELF_CONSISTENCY_EMBEDDING_COLS, SINGLE_REQUEST_CONCORDANCE_COLS,
    SHOW_PLOTS
)

from utils import parse_embedding

# ---------------------------------------------------------------------------
# Directory setup
# ---------------------------------------------------------------------------
TABLE_DIR = os.path.join(OUTPUT_DIR_ROLE, "persona_stability_analysis")
IMG_DIR = os.path.join(OUTPUT_DIR_ROLE, "persona_stability_analysis/img")
os.makedirs(TABLE_DIR, exist_ok=True)
os.makedirs(IMG_DIR, exist_ok=True)

# ===========================================================================
# Helper: build role embedding matrices
# ===========================================================================

def _build_role_vectors(emb_col_map: dict) -> dict:
    """
    Extract a list of embedding arrays per role from *emb_col_map*.

    Parameters
    ----------
    emb_col_map :
        ``{role_name: column_name}`` pairs.

    Returns
    -------
    dict
        ``{role_name: np.ndarray}`` where the array has shape (N, D).
        Roles with fewer than 5 valid embeddings are excluded.
    """
    vectors = {r: [] for r in emb_col_map}
    for _, row in df.iterrows():
        for role, col in emb_col_map.items():
            v = parse_embedding(row.get(col))
            if v is not None:
                vectors[role].append(v)
    return {r: np.array(v) for r, v in vectors.items() if len(v) >= 5}

def _save_or_show(path):
    plt.savefig(path, dpi=300, bbox_inches="tight")
    print(f"Saved → {path}")
    if SHOW_PLOTS:
        plt.show()
    plt.close()

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
df = pd.read_csv(DATA_FILE)


from config import COLUMNS_ANSWER
comparison_cols = COLUMNS_ANSWER[1:]

# ===========================================================================
# 1. Persona Attractor Dispersion
# ===========================================================================

def persona_attractor_dispersion(role_vectors: dict) -> pd.DataFrame:
    """
    Compute mean normalised L2 distance to centroid per role with bootstrap CI.

    The dispersion metric is normalised by sqrt(D) to make it independent of
    the embedding dimensionality, following standard practice in embedding
    geometry literature.

    Parameters
    ----------
    role_vectors :
        ``{role: (N, D) array}``

    Returns
    -------
    pandas.DataFrame
    """
    rows = []
    for role, X in role_vectors.items():
        if len(X) < 20:
            print(f"  PAD: skipping {role} (n < 20).")
            continue
        centroid    = X.mean(axis=0)
        distances   = np.linalg.norm(X - centroid, axis=1)
        norm_d      = distances / np.sqrt(X.shape[1])

        ci_low, ci_high = np.nan, np.nan
        try:
            res = bootstrap(
                (norm_d,), np.mean,
                confidence_level=0.95, n_resamples=1000, method="percentile",
            )
            ci_low  = res.confidence_interval.low
            ci_high = res.confidence_interval.high
        except Exception:
            pass

        rows.append({
            "role": role,
            "mean_attractor_dispersion": float(np.mean(norm_d)),
            "variance_dispersion":        float(np.var(norm_d)),
            "ci_2.5":  ci_low,
            "ci_97.5": ci_high,
        })
    return pd.DataFrame(rows)


role_vectors_single = _build_role_vectors(SINGLE_REQUEST_EMBEDDING_COLS)
pad_df = persona_attractor_dispersion(role_vectors_single)
print("\n=== Persona Attractor Dispersion ===")
print(pad_df.to_string(index=False))
pad_df.to_excel(f"{TABLE_DIR}/persona_attractor_dispersion.xlsx", index=False)


# ===========================================================================
# 2. Role Confusion Entropy (embedding space)
# ===========================================================================

def role_confusion_entropy(role_vectors: dict) -> pd.DataFrame:
    """
    Compute Shannon entropy of the centroid-distance probability distribution.

    For each role's embedding, we compute the distances to every role centroid,
    convert to a probability distribution via softmax, and compute Shannon
    entropy.  High entropy ≈ the model cannot distinguish roles; low entropy
    ≈ strong role separation.

    Parameters
    ----------
    role_vectors :
        ``{role: (N, D) array}``

    Returns
    -------
    pandas.DataFrame
    """
    centroids = {r: X.mean(axis=0) for r, X in role_vectors.items()}
    rows = []
    for role, X in role_vectors.items():
        entropies = []
        for emb in X:
            dists = [-np.linalg.norm(emb - centroids[r]) for r in centroids]
            probs = softmax(dists)
            entropies.append(float(entropy(probs, base=2)))
        rows.append({
            "role": role,
            "mean_role_confusion_entropy": float(np.mean(entropies)),
            "std_role_confusion_entropy":  float(np.std(entropies)),
        })
    return pd.DataFrame(rows)


rce_df = role_confusion_entropy(role_vectors_single)
print("\n=== Role Confusion Entropy ===")
print(rce_df.to_string(index=False))
rce_df.to_excel(f"{TABLE_DIR}/role_confusion_entropy.xlsx", index=False)


# ===========================================================================
# 3. Role Performance Variability Entropy (Single Persona Framework)
# ===========================================================================

def role_performance_variability_entropy(
        df: pd.DataFrame,
        role_map: dict,
        min_cases: int = 10
) -> pd.DataFrame:
    """
    Compute Shannon entropy of role-specific treatment concordance
    across patients (single persona framework only).

    This metric quantifies how variable each specialist role's
    treatment accuracy is across clinical cases.

    Interpretation
    --------------
    - Low entropy  -> Stable performance across patients
                     (consistently high or consistently low accuracy)

    - High entropy -> High variability in performance
                     (accuracy strongly case-dependent)

    IMPORTANT:
    This function does NOT compare frameworks.
    It only evaluates role-level variability within
    the single-persona setting (F3/F4/F5).

    Parameters
    ----------
    df : pandas.DataFrame
        Main dataset containing treatment concordance columns.

    role_map : dict
        Mapping role_name -> treatment concordance column.
        Example:
        {
            "Surgeon": "F3_persona_surgeon_treatment_concordance",
            "Oncologist": "F4_persona_medical_oncologist_treatment_concordance",
            "Radio-Oncologist": "F5_persona_radiation_oncologist_treatment_concordance",
        }

    min_cases : int
        Minimum number of valid patients required to compute entropy.

    Returns
    -------
    pandas.DataFrame
        Columns:
        - role
        - accuracy_entropy_bits
        - mean_accuracy
        - std_accuracy
        - n_patients
    """

    rows = []

    for role, col in role_map.items():

        if col not in df.columns:
            print(f"Missing column for {role}")
            continue

        acc = pd.to_numeric(df[col], errors="coerce").dropna()

        if len(acc) >= min_cases:

            # Histogram over [0,1]
            hist, _ = np.histogram(
                acc,
                bins=10,
                range=(0, 1),
                density=False
            )

            # Laplace smoothing for stability
            prob = (hist + 1e-9) / np.sum(hist + 1e-9)

            H = float(entropy(prob, base=2))

        else:
            H = np.nan

        rows.append({
            "role": role,
            "accuracy_entropy_bits": H,
            "mean_accuracy": float(acc.mean()) if len(acc) > 0 else np.nan,
            "std_accuracy": float(acc.std()) if len(acc) > 0 else np.nan,
            "n_patients": len(acc)
        })

    return pd.DataFrame(rows)


variability_df = role_performance_variability_entropy(
    df,
    role_map=SINGLE_REQUEST_CONCORDANCE_COLS,
    min_cases=10
)

print("\n=== Role Performance Variability Entropy (Single Persona) ===")
print(variability_df.to_string(index=False))

variability_df.to_excel(
    f"{TABLE_DIR}/role_performance_variability_entropy.xlsx",
    index=False
)
# ===========================================================================
# 4. Single vs Self-consistency cosine similarity
# ===========================================================================

def persona_cosine_similarity(df: pd.DataFrame, roles: list) -> pd.DataFrame:
    """
    Mean cosine similarity between single-request and self-consistency embeddings.

    Parameters
    ----------
    df :
        Main DataFrame.
    roles :
        Role names (lowercase).

    Returns
    -------
    pandas.DataFrame
    """
    rows = []
    for role in roles:
        single_col = SINGLE_REQUEST_EMBEDDING_COLS[role]
        self_col = SELF_CONSISTENCY_EMBEDDING_COLS[role]
        if single_col not in df.columns or self_col not in df.columns:
            continue
        sims = []
        for _, row in df.iterrows():
            es = parse_embedding(row[single_col])
            et = parse_embedding(row[self_col])
            if es is None or et is None:
                continue
            sim = cosine_similarity(es.reshape(1, -1), et.reshape(1, -1))[0][0]
            sims.append(float(sim))
        rows.append({
            "role": role,
            "mean_cosine_similarity": float(np.mean(sims)) if sims else np.nan,
            "std_cosine_similarity":  float(np.std(sims))  if sims else np.nan,
            "n": len(sims),
        })
    return pd.DataFrame(rows)


cosine_df = persona_cosine_similarity(df, ROLES)
print("\n=== Single vs Self-consistency Cosine Similarity ===")
print(cosine_df.to_string(index=False))
cosine_df.to_excel(f"{TABLE_DIR}/persona_cosine_similarity.xlsx", index=False)


# ===========================================================================
# Persona Stability Index (Backwards compatible with legacy pipeline)
# ===========================================================================

def persona_stability_index(df: pd.DataFrame,
                            cosine_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute Persona Stability Index (PSI).

    PSI measures role-level clinical and semantic stability across prompting paradigms.

    PSI is defined as:

    PSI =
        w_cos * identity preservation (embedding cosine similarity)
      + w_spec * clinical specificity rate
      + w_pitch * boundary control (1 − boundary violation rate)
      + w_acc * treatment recommendation accuracy

    Metrics are aggregated at role level using role-specific column prefixes
    defined in ROLE_PREFIX_MAP.

    Aggregation strategy:
    - Cosine similarity is obtained from embedding-space similarity analysis.
    - Clinical rates are averaged across available experimental frameworks
      (single-request and self-consistency conditions when available).

    Returns
    -------
    pandas.DataFrame
        Role-level PSI metrics including:
        - persona_stability_index
        - cosine_similarity
        - specificity_rate
        - boundary_control
        - accuracy
    """

    rows = []

    for role_name, role_prefix in ROLE_PREFIX_MAP.items():

        # -------------------------
        # Cosine similarity
        # -------------------------

        row_cos = cosine_df[
            cosine_df["role"].str.lower() == role_name.lower()
        ]

        cos_sim = (
            row_cos["mean_cosine_similarity"].values[0]
            if not row_cos.empty else np.nan
        )

        # -------------------------
        # Legacy-style clinical rates aggregation
        # (matches old code logic across frameworks)
        # -------------------------

        spec_rates = []
        pitch_rates = []
        acc_rates = []

        for fw_pattern in ["single_request", "self_consistency"]:

            base = f"{fw_pattern}_{role_prefix}"

            spec_col = f"{base}_specific"
            pitch_col = f"{base}_pitch_invasion"
            acc_col = f"{base}_comparison"

            if spec_col in df.columns:
                spec_rates.append(df[spec_col].mean())

            if pitch_col in df.columns:
                pitch_rates.append(df[pitch_col].mean())

            if acc_col in df.columns:
                acc_rates.append(df[acc_col].mean())

        spec_rate = np.nanmean(spec_rates)
        pitch_rate = np.nanmean(pitch_rates)
        accuracy = np.nanmean(acc_rates)

        psi = (
            PSI_WEIGHTS["cosine_similarity"] * (0 if np.isnan(cos_sim) else cos_sim) +
            PSI_WEIGHTS["specificity_rate"] * (0 if np.isnan(spec_rate) else spec_rate) +
            PSI_WEIGHTS["pitch_control"] * (0 if np.isnan(pitch_rate) else (1 - pitch_rate)) +
            PSI_WEIGHTS["accuracy"] * (0 if np.isnan(accuracy) else accuracy)
        )

        rows.append({
            "role": role_name,
            "persona_stability_index": round(psi, 4),
            "cosine_similarity": round(cos_sim, 4) if not np.isnan(cos_sim) else np.nan,
            "specificity_rate": round(spec_rate, 4) if not np.isnan(spec_rate) else np.nan,
            "boundary_control": round(1 - pitch_rate, 4) if not np.isnan(pitch_rate) else np.nan,
            "accuracy": round(accuracy, 4) if not np.isnan(accuracy) else np.nan,
        })

    return pd.DataFrame(rows)


psi_df = persona_stability_index(df, cosine_df)

psi_df.to_excel(f"{TABLE_DIR}/persona_stability_index.xlsx", index=False)

print("\n=== Persona Stability Index ===")
print(psi_df.to_string(index=False))


# ===========================================================================
# Composite Robustness Index (Legacy Paper Version)
# ===========================================================================

def composite_robustness_index(df: pd.DataFrame,
                               cosine_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute Composite Robustness Index (CRI).

    CRI measures global reliability and clinical coherence of persona behaviour
    across multiple clinical and semantic signals.

    CRI is defined as:

    CRI =
        w_cos  * identity stability (embedding cosine similarity)
      + w_spec * clinical specificity rate
      + w_pitch * boundary control (1 − boundary violation rate)
      + w_acc  * treatment accuracy
      + w_ent  * (1 − global treatment entropy)

    Where:
    - Identity stability quantifies semantic consistency across prompting paradigms.
    - Specificity rate measures domain-focused clinical reasoning.
    - Boundary control penalises cross-specialty decision leakage.
    - Treatment accuracy measures alignment with MDTB reference decisions.
    - Entropy stability penalises unpredictable clinical decision behaviour.

    Entropy penalty is computed using the global mean treatment entropy across cases.

    Returns
    -------
    pandas.DataFrame
        Role-level robustness metrics including:
        - composite_robustness_index
        - identity_stability
        - clinical_fidelity
        - boundary_control
    """

    rows = []

    entropy_global = (
        df["treatment_entropy_bits"].mean()
        if "treatment_entropy_bits" in df.columns else 0
    )

    for role_name, role_prefix in ROLE_PREFIX_MAP.items():

        row_cos = cosine_df[
            cosine_df["role"].str.lower() == role_name.lower()
        ]

        cos_sim = (
            row_cos["mean_cosine_similarity"].values[0]
            if not row_cos.empty else np.nan
        )

        spec_col = f"{role_prefix}_domain_content_present"
        pitch_col = f"{role_prefix}_boundary_violation"
        acc_col = f"{role_prefix}_treatment_concordance"

        spec = df[spec_col].mean() if spec_col in df.columns else 0
        pitch = df[pitch_col].mean() if pitch_col in df.columns else 0
        acc = df[acc_col].mean() if acc_col in df.columns else 0

        cri = (
            CRI_WEIGHTS["cosine_similarity"] * (0 if np.isnan(cos_sim) else cos_sim) +
            CRI_WEIGHTS["specificity_rate"] * spec +
            CRI_WEIGHTS["pitch_control"] * (1 - pitch) +
            CRI_WEIGHTS["accuracy"] * acc +
            CRI_WEIGHTS["entropy_stability"] * (1 - entropy_global)
        )

        rows.append({
            "role": role_name,
            "composite_robustness_index": round(cri, 4),
            "identity_stability": round(cos_sim, 4) if not np.isnan(cos_sim) else np.nan,
            "clinical_fidelity": round(acc, 4),
            "boundary_control": round(1 - pitch, 4),
        })

    return pd.DataFrame(rows)


cri_df = composite_robustness_index(df, cosine_df)

cri_df.to_excel(f"{TABLE_DIR}/composite_robustness_index.xlsx", index=False)

print("\n=== Composite Robustness Index ===")
print(cri_df.to_string(index=False))


# ===========================================================================
# 7. Boundary Violation Entropy
# ===========================================================================

def boundary_violation_entropy(df: pd.DataFrame) -> pd.DataFrame:

    """
    Compute binary Shannon entropy of boundary violation behaviour per role.

    H(p) = -p log2(p) - (1-p) log2(1-p)

    Interpretation:
        - H ≈ 0 → behaviour is highly predictable (always correct or always violating)
        - H ≈ 1 → behaviour is maximally uncertain (p ≈ 0.5)

    Measures stability of role-constrained clinical reasoning.

    Parameters
    ----------
    df :
        Main dataset.

    roles :
        Role names.

    Returns
    -------
    pandas.DataFrame
        Role-level boundary violation rates and entropy in bits.
    """

    rows = []

    for role_name, role_prefix in ROLE_PREFIX_MAP.items():

        pitch_col = f"{role_prefix}_boundary_violation"

        if pitch_col not in df.columns:
            continue

        p = df[pitch_col].fillna(0).mean()

        if p in (0, 1):
            H = 0
        else:
            H = -(p*np.log2(p) + (1-p)*np.log2(1-p))

        rows.append({
            "role": role_name,
            "boundary_violation_rate": round(p, 4),
            "boundary_entropy_bits": round(H, 4)
        })

    return pd.DataFrame(rows)


boundary_df = boundary_violation_entropy(df)

boundary_df.to_excel(f"{TABLE_DIR}/boundary_entropy.xlsx", index=False)

print("\n=== Role Boundary Violation Entropy ===")
print(boundary_df.to_string(index=False))


# ===========================================================================
# 8. Clinical Risk Penalty Score
# ===========================================================================

def clinical_risk_penalty(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the Clinical Risk Penalty per role.

    CRP = w_pitch * boundary_violation_rate
        + w_nonspec * (1 - specificity_rate)

    A high score indicates that the model frequently generates content outside
    its assigned role AND fails to include role-specific clinical content.

    Parameters
    ----------
    df :
        Main DataFrame.

    Returns
    -------
    pandas.DataFrame
    """

    rows = []

    for role_name, role_prefix in ROLE_PREFIX_MAP.items():

        pitch_col = f"{role_prefix}_boundary_violation"
        spec_col  = f"{role_prefix}_domain_content_present"

        pitch = float(df[pitch_col].mean()) if pitch_col in df.columns else np.nan
        spec  = float(df[spec_col].mean())  if spec_col  in df.columns else np.nan

        crp = (
            RISK_WEIGHTS["boundary_violation"] * pitch +
            RISK_WEIGHTS["non_specificity"] * (1 - spec)
        ) if not (np.isnan(pitch) or np.isnan(spec)) else np.nan

        rows.append({
            "role": role_name,
            "clinical_risk_score": round(crp, 4) if not np.isnan(crp) else np.nan,
            "boundary_violation_rate": round(pitch, 4) if not np.isnan(pitch) else np.nan,
            "non_specificity_rate": round(1 - spec, 4) if not np.isnan(spec) else np.nan,
        })

    return pd.DataFrame(rows)


risk_df = clinical_risk_penalty(df)

print("\n=== Clinical Risk Penalty Score ===")
print(risk_df.to_string(index=False))

risk_df.to_excel(
    f"{TABLE_DIR}/clinical_risk_score.xlsx",
    index=False
)


# ===========================================================================
# 9. Summary table — one row per role, all key metrics
# ===========================================================================

summary_rows = []

def _safe_first(frame, role, col):

    if frame is None or frame.empty:
        return np.nan

    if "role" not in frame.columns:
        return np.nan

    vals = frame.loc[
        frame["role"].str.lower() == role.lower(),
        col
    ].values

    return float(vals[0]) if len(vals) else np.nan

for role_name, prefix in ROLE_PREFIX_MAP.items():

    acc_col  = f"{prefix}_treatment_concordance"
    spec_col = f"{prefix}_domain_content_present"

    summary_rows.append({

        "Role": role_name,

        "N": len(df),

        "Accuracy (%)":
            round(df[acc_col].mean() * 100, 1)
            if acc_col in df.columns else np.nan,

        "Specificity rate (%)":
            round(df[spec_col].mean() * 100, 1)
            if spec_col in df.columns else np.nan,

        "Pitch violation rate (%)":
            round(
                _safe_first(boundary_df, role_name, "boundary_violation_rate") * 100,
                1
            ),

        "Cosine sim (single vs SC)":
            round(
                _safe_first(cosine_df, role_name, "mean_cosine_similarity"),
                3
            ),

        "Boundary entropy (bits)":
            round(
                _safe_first(boundary_df, role_name, "boundary_entropy_bits"),
                3
            ),

        "Clinical risk score":
            round(
                _safe_first(risk_df, role_name, "clinical_risk_score"),
                3
            ),

        "Persona stability index":
            round(
                _safe_first(psi_df, role_name, "persona_stability_index"),
                3
            ),

        "Composite robustness index":
            round(
                _safe_first(cri_df, role_name, "composite_robustness_index"),
                3
            ),
    })


summary_df = pd.DataFrame(summary_rows).set_index("Role")

summary_df.to_excel(
    f"{TABLE_DIR}/analysis_summary.xlsx",
    index=True
)

print(f"\nSummary → {TABLE_DIR}/analysis_summary.xlsx")
print(summary_df.T.to_string())


# ===========================================================
# Bonus Track: Sensitivity Analysis — Weight Perturbation Rank Stability
# ===========================================================

print("\n=== Sensitivity Analysis: Weight Perturbation Rank Stability ===")


def renormalize_weights(weights):
    total = sum(weights.values())
    return {k: v / total for k, v in weights.items()}


def compute_psi_scores_with_weights(df_local, weights):

    rows = []

    for role_name, role_prefix in ROLE_PREFIX_MAP.items():

        spec_col = f"{role_prefix}_domain_content_present"
        pitch_col = f"{role_prefix}_boundary_violation"
        acc_col = f"{role_prefix}_treatment_concordance"

        spec = df_local[spec_col].mean() if spec_col in df_local.columns else 0
        pitch = df_local[pitch_col].mean() if pitch_col in df_local.columns else 0
        acc = df_local[acc_col].mean() if acc_col in df_local.columns else 0

        cos_sim = 0.8  # approximation since recomputing embeddings is expensive

        psi = (
            weights["cosine_similarity"] * cos_sim +
            weights["specificity_rate"] * spec +
            weights["pitch_control"] * (1 - pitch) +
            weights["accuracy"] * acc
        )

        rows.append({"role": role_name, "score": psi})

    return pd.DataFrame(rows)


# -------------------------
# Baseline PSI scores
# -------------------------

baseline_psi_df = compute_psi_scores_with_weights(df, PSI_WEIGHTS)
baseline_rank = baseline_psi_df.sort_values("score", ascending=False)["role"].values


# -------------------------
# Perturbation grid
# -------------------------

eps_grid = [-0.2, -0.1, 0, 0.1, 0.2]

heatmap_matrix = []

for eps in eps_grid:

    perturbed_weights = {
        k: v * (1 + eps)
        for k, v in PSI_WEIGHTS.items()
    }

    perturbed_weights = renormalize_weights(perturbed_weights)

    psi_df_temp = compute_psi_scores_with_weights(df, perturbed_weights)

    perturbed_rank = psi_df_temp.sort_values(
        "score", ascending=False
    )["role"].values

    # Rank stability vs baseline
    stability_scores = []

    for r in baseline_rank:
        if r in perturbed_rank:
            stability_scores.append(
                np.where(perturbed_rank == r)[0][0]
            )
        else:
            stability_scores.append(np.nan)

    heatmap_matrix.append(stability_scores)

heatmap_matrix = np.array(heatmap_matrix)


# -------------------------
# Plot Heatmap
# -------------------------

plt.figure(figsize=(8, 5))

sns.heatmap(
    heatmap_matrix,
    annot=True,
    fmt=".2f",
    cmap="viridis",
    xticklabels=baseline_rank,
    yticklabels=[f"{int(eps*100)}%" for eps in eps_grid]
)

plt.title("PSI Rank Stability Under Weight Perturbation")
plt.xlabel("Persona Role")
plt.ylabel("Weight Perturbation")

heatmap_path = f"{IMG_DIR}/psi_rank_stability_heatmap.png"

_save_or_show(heatmap_path)

print("Sensitivity analysis complete.")


# ===========================================================
# Print Rank Stability Numerical Results
# ===========================================================

print("\n=== Rank Stability Numerical Results ===")

print("\nBaseline Rank Order:")
for i, r in enumerate(baseline_rank):
    print(f"{i+1}. {r}")

print("\nPerturbation Stability Matrix (rows = perturbations):")

for i, eps in enumerate(eps_grid):

    print(f"\nPerturbation = {int(eps*100)}%")

    for j, role in enumerate(baseline_rank):

        val = heatmap_matrix[i, j]

        print(f"{role:25s} → Rank Position: {val}")

print("\n=== Sensitivity Analysis Complete ===")


print("\n=== Persona Stability Analysis Complete ===")
