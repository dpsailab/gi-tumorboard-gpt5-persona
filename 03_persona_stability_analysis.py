"""
03_persona_stability_analysis.py
=================================
Persona stability, robustness, and boundary-control analysis.

This module quantifies the degree to which each specialist role persona
maintains a coherent, bounded clinical identity across prompting conditions
(single-request vs self-consistency).  The following composite indices are
computed and saved for inclusion in the supplementary materials:

  1. **Persona Stability Index (PSI)**
     Weighted combination of cosine identity preservation, role-specific
     content rate, pitch-invasion control, and treatment accuracy.

  2. **Composite Robustness Index (CRI)**
     Extends the PSI with a global entropy stability term.

  3. **Role Confusion Entropy**
     Shannon entropy of each role's embedding-distance distribution relative
     to the centroids of all other roles.  High entropy indicates that a
     role's responses are geometrically indistinguishable from those of other
     roles.

  4. **Persona Attractor Dispersion (PAD)**
     Mean and variance of the L2 distance from each embedding to its role
     centroid, normalised by the square root of the embedding dimension.
     Bootstrap 95 % CIs are provided.

  5. **Role Boundary Violation Entropy**
     Binary entropy of the pitch-invasion rate per role; measures
     unpredictability of out-of-scope content generation.

  6. **Clinical Risk Penalty Score**
     Weighted combination of pitch-invasion rate and non-specificity rate;
     provides a clinically interpretable risk summary.

  7. **Role Consistency Entropy Across Patients**
     Per-patient accuracy distribution entropy per role.

All outputs are written to ``role/``.

Weight rationale
----------------
The composite index weights are theory-driven and documented in the Methods
section.  They are centralised in ``config.py`` (CRI_WEIGHTS, PSI_WEIGHTS,
RISK_WEIGHTS) to facilitate sensitivity analyses.
"""

import os
from ast import literal_eval

import numpy as np
import pandas as pd
from scipy.special import softmax
from scipy.stats import bootstrap, entropy
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.utils import resample

from config import (
    CRI_WEIGHTS, DATA_FILE, OUTPUT_DIR_ROLE,
    PSI_WEIGHTS, RISK_WEIGHTS, SPECIALIST_COLS,
)

from utils import compare_treatments, compute_majority_treatment, parse_embedding

# ---------------------------------------------------------------------------
# Directory setup
# ---------------------------------------------------------------------------
os.makedirs(OUTPUT_DIR_ROLE, exist_ok=True)

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
df = pd.read_excel(DATA_FILE)
df["majority"] = df.apply(
    lambda row: compute_majority_treatment(row, SPECIALIST_COLS), axis=1
)

from config import COLUMNS_ANSWER
comparison_cols = COLUMNS_ANSWER[1:]
df = compare_treatments(df, "Konferenzbeschluss", comparison_cols)

ROLES = ["surgeon", "oncologist", "radio-oncologist"]

# ---------------------------------------------------------------------------
# Single-request embedding column map
# ---------------------------------------------------------------------------
SINGLE_EMB = {r: f"ChatGPT_single_request_5_{r}_embeddings" for r in ROLES}
SELF_EMB   = {r: f"ChatGPT_single_request_5_self-consistency_{r}_embeddings"
              for r in ROLES}


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


role_vectors_single = _build_role_vectors(SINGLE_EMB)
pad_df = persona_attractor_dispersion(role_vectors_single)
print("\n=== Persona Attractor Dispersion ===")
print(pad_df.to_string(index=False))
pad_df.to_excel(f"{OUTPUT_DIR_ROLE}/persona_attractor_dispersion.xlsx", index=False)


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
rce_df.to_excel(f"{OUTPUT_DIR_ROLE}/role_confusion_entropy.xlsx", index=False)


# ===========================================================================
# 3. Role Consistency Entropy Across Patients
# ===========================================================================

def role_consistency_entropy_across_patients(df: pd.DataFrame,
                                              roles: list) -> pd.DataFrame:
    """
    Shannon entropy of per-patient accuracy distribution per role.

    A uniform distribution (all patients equally likely to be correct)
    produces maximum entropy.  A peaked distribution (model is consistently
    right or wrong for the same patients) produces low entropy.

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
        single_col = f"ChatGPT_single_request_5_{role}_comparison"
        self_col   = "ChatGPT_single_request_5_self-consistency_comparison"

        if single_col not in df.columns or self_col not in df.columns:
            print(f"  Consistency entropy: missing columns for {role}.")
            continue

        tmp = df[[single_col, self_col]].apply(pd.to_numeric, errors="coerce")
        patient_acc = tmp.mean(axis=1).dropna()

        if len(patient_acc) > 10:
            hist, _ = np.histogram(patient_acc, bins=10, range=(0, 1))
            prob = (hist + 1e-9) / (hist + 1e-9).sum()
            H = float(entropy(prob, base=2))
        else:
            H = np.nan

        rows.append({
            "role": role,
            "consistency_entropy_bits": H,
            "mean_patient_accuracy": float(patient_acc.mean()),
            "n_valid": len(patient_acc),
        })
    return pd.DataFrame(rows)


cons_df = role_consistency_entropy_across_patients(df, ROLES)
print("\n=== Role Consistency Entropy Across Patients ===")
print(cons_df.to_string(index=False))
cons_df.to_excel(f"{OUTPUT_DIR_ROLE}/role_consistency_entropy.xlsx", index=False)


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
        single_col = SINGLE_EMB[role]
        self_col   = SELF_EMB[role]
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
cosine_df.to_excel(f"{OUTPUT_DIR_ROLE}/persona_cosine_similarity.xlsx", index=False)


# ===========================================================================
# 5. Persona Stability Index (PSI)
# ===========================================================================

def persona_stability_index(df: pd.DataFrame, cosine_df: pd.DataFrame,
                             roles: list, frameworks: dict) -> pd.DataFrame:
    """
    Compute the Persona Stability Index for each role.

    PSI = w1 * cosine_sim + w2 * specificity_rate
          + w3 * (1 - pitch_invasion_rate) + w4 * accuracy

    Weights are defined in ``config.PSI_WEIGHTS``.

    Parameters
    ----------
    df :
        Main DataFrame.
    cosine_df :
        Output of ``persona_cosine_similarity``.
    roles :
        Role names (lowercase).
    frameworks :
        ``{framework_name: column_pattern}`` where ``{}`` is the role placeholder.

    Returns
    -------
    pandas.DataFrame
    """
    rows = []
    for role in roles:
        row_cos = cosine_df[cosine_df["role"] == role]
        cos_sim = row_cos["mean_cosine_similarity"].values[0] \
                  if not row_cos.empty else np.nan

        spec_rates, pitch_rates, acc_rates = [], [], []
        for fw_pattern in frameworks.values():
            base = fw_pattern.format(role)
            for col, store in [
                (f"{base}_specific",        spec_rates),
                (f"{base}_pitch_invasion",  pitch_rates),
                (f"{base}_comparison",      acc_rates),
            ]:
                if col in df.columns:
                    store.append(float(df[col].mean()))

        psi = (
            PSI_WEIGHTS["cosine_similarity"] * (cos_sim or 0) +
            PSI_WEIGHTS["specificity_rate"]  * (np.nanmean(spec_rates) if spec_rates else 0) +
            PSI_WEIGHTS["pitch_control"]     * (1 - np.nanmean(pitch_rates) if pitch_rates else 0) +
            PSI_WEIGHTS["accuracy"]          * (np.nanmean(acc_rates) if acc_rates else 0)
        )
        rows.append({
            "role": role,
            "persona_stability_index": round(psi, 4),
            "cosine_similarity": round(cos_sim, 4) if not np.isnan(cos_sim) else np.nan,
            "specificity_rate":  round(np.nanmean(spec_rates), 4) if spec_rates else np.nan,
            "pitch_control":     round(1 - np.nanmean(pitch_rates), 4) if pitch_rates else np.nan,
            "accuracy":          round(np.nanmean(acc_rates), 4) if acc_rates else np.nan,
        })
    return pd.DataFrame(rows)


FRAMEWORKS = {
    "single":           "ChatGPT_single_request_5_{}",
    "self_consistency": "ChatGPT_single_request_5_self-consistency_{}",
}

psi_df = persona_stability_index(df, cosine_df, ROLES, FRAMEWORKS)
print("\n=== Persona Stability Index ===")
print(psi_df.to_string(index=False))
psi_df.to_excel(f"{OUTPUT_DIR_ROLE}/persona_stability_index.xlsx", index=False)


# ===========================================================================
# 6. Composite Robustness Index (CRI) with bootstrap CIs
# ===========================================================================

def composite_robustness_index(df: pd.DataFrame, cosine_df: pd.DataFrame,
                                roles: list) -> pd.DataFrame:
    """
    Compute the Composite Robustness Index per role.

    CRI = w_cos  * cosine_sim
        + w_spec * specificity_rate
        + w_pitch * (1 - pitch_invasion_rate)
        + w_acc  * accuracy
        + w_ent  * (1 - global_entropy)

    Weights are defined in ``config.CRI_WEIGHTS``.

    Parameters
    ----------
    df :
        Main DataFrame.
    cosine_df :
        Output of ``persona_cosine_similarity``.
    roles :
        Role names (lowercase).

    Returns
    -------
    pandas.DataFrame
    """
    rows = []
    entropy_global = float(df["treatment_entropy_bits"].mean()) \
                     if "treatment_entropy_bits" in df.columns else 0.0

    for role in roles:
        row_cos = cosine_df[cosine_df["role"] == role]
        cos_sim = row_cos["mean_cosine_similarity"].values[0] \
                  if not row_cos.empty else np.nan

        spec_col  = f"ChatGPT_single_request_5_{role}_specific"
        pitch_col = f"ChatGPT_single_request_5_{role}_pitch_invasion"
        acc_col   = f"ChatGPT_single_request_5_{role}_comparison"

        spec  = float(df[spec_col].mean())  if spec_col  in df.columns else 0.0
        pitch = float(df[pitch_col].mean()) if pitch_col in df.columns else 0.0
        acc   = float(df[acc_col].mean())   if acc_col   in df.columns else 0.0

        cri = (
            CRI_WEIGHTS["cosine_similarity"] * (cos_sim or 0) +
            CRI_WEIGHTS["specificity_rate"]  * spec +
            CRI_WEIGHTS["pitch_control"]     * (1 - pitch) +
            CRI_WEIGHTS["accuracy"]          * acc +
            CRI_WEIGHTS["entropy_stability"] * (1 - entropy_global)
        )
        rows.append({
            "role": role,
            "composite_robustness_index": round(cri, 4),
            "identity_stability":    round(cos_sim, 4) if not np.isnan(cos_sim) else np.nan,
            "clinical_fidelity":     round(acc, 4),
            "role_boundary_control": round(1 - pitch, 4),
        })
    return pd.DataFrame(rows)


cri_df = composite_robustness_index(df, cosine_df, ROLES)
print("\n=== Composite Robustness Index ===")
print(cri_df.to_string(index=False))
cri_df.to_excel(f"{OUTPUT_DIR_ROLE}/composite_robustness_index.xlsx", index=False)

# Bootstrap CIs for CRI
print("\n=== Bootstrap 95 % CIs for Composite Robustness Index ===")
bootstrap_rows = []
entropy_global = float(df["treatment_entropy_bits"].mean()) \
                 if "treatment_entropy_bits" in df.columns else 0.0

for role in ROLES:
    boot_scores = []
    for _ in range(500):
        sample = resample(df, random_state=None)

        single_emb_col = SINGLE_EMB[role]
        self_emb_col   = SELF_EMB[role]

        single_sample = sample[single_emb_col].dropna()
        self_sample   = sample[self_emb_col].dropna()
        if single_sample.empty or self_sample.empty:
            continue

        es = parse_embedding(single_sample.iloc[0])
        et = parse_embedding(self_sample.iloc[0])
        if es is None or et is None:
            continue

        cos_sim = float(cosine_similarity(es.reshape(1, -1), et.reshape(1, -1))[0][0])

        spec_col  = f"ChatGPT_single_request_5_{role}_specific"
        pitch_col = f"ChatGPT_single_request_5_{role}_pitch_invasion"
        acc_col   = f"ChatGPT_single_request_5_{role}_comparison"

        spec  = float(sample[spec_col].mean())  if spec_col  in sample.columns else 0.0
        pitch = float(sample[pitch_col].mean()) if pitch_col in sample.columns else 0.0
        acc   = float(sample[acc_col].mean())   if acc_col   in sample.columns else 0.0
        ent   = float(sample["treatment_entropy_bits"].mean()) \
                if "treatment_entropy_bits" in sample.columns else 0.0

        score = (
            CRI_WEIGHTS["cosine_similarity"] * cos_sim +
            CRI_WEIGHTS["specificity_rate"]  * spec +
            CRI_WEIGHTS["pitch_control"]     * (1 - pitch) +
            CRI_WEIGHTS["accuracy"]          * acc +
            CRI_WEIGHTS["entropy_stability"] * (1 - ent)
        )
        boot_scores.append(score)

    if boot_scores:
        bootstrap_rows.append({
            "role": role,
            "ci_2.5":  round(float(np.percentile(boot_scores, 2.5)), 4),
            "ci_97.5": round(float(np.percentile(boot_scores, 97.5)), 4),
            "n_bootstrap": len(boot_scores),
        })

bootstrap_df = pd.DataFrame(bootstrap_rows)
print(bootstrap_df.to_string(index=False))
bootstrap_df.to_excel(
    f"{OUTPUT_DIR_ROLE}/composite_robustness_bootstrap_ci.xlsx", index=False
)


# ===========================================================================
# 7. Role Boundary Violation Entropy
# ===========================================================================

def boundary_violation_entropy(df: pd.DataFrame, roles: list) -> pd.DataFrame:
    """
    Compute the binary entropy of the pitch-invasion rate per role.

    H(p) = -p*log2(p) - (1-p)*log2(1-p)

    A rate of 0 or 1 produces entropy = 0 (fully predictable).
    A rate of 0.5 produces maximum entropy = 1 bit (maximally unpredictable).

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
        pitch_col = f"ChatGPT_single_request_5_{role}_pitch_invasion"
        if pitch_col not in df.columns:
            continue
        p = float(df[pitch_col].fillna(0).mean())
        H = 0.0 if p in (0, 1) else -(p * np.log2(p) + (1 - p) * np.log2(1 - p))
        rows.append({
            "role": role,
            "pitch_invasion_rate": round(p, 4),
            "boundary_entropy_bits": round(H, 4),
        })
    return pd.DataFrame(rows)


boundary_df = boundary_violation_entropy(df, ROLES)
print("\n=== Role Boundary Violation Entropy ===")
print(boundary_df.to_string(index=False))
boundary_df.to_excel(f"{OUTPUT_DIR_ROLE}/boundary_entropy.xlsx", index=False)


# ===========================================================================
# 8. Clinical Risk Penalty Score
# ===========================================================================

def clinical_risk_penalty(df: pd.DataFrame, roles: list) -> pd.DataFrame:
    """
    Compute the Clinical Risk Penalty per role.

    CRP = w_pitch * pitch_invasion_rate + w_nonspec * (1 - specificity_rate)

    A high score indicates that the model frequently generates content outside
    its assigned role AND fails to include role-specific clinical content.

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
        pitch_col = f"ChatGPT_single_request_5_{role}_pitch_invasion"
        spec_col  = f"ChatGPT_single_request_5_{role}_specific"
        pitch = float(df[pitch_col].mean()) if pitch_col in df.columns else np.nan
        spec  = float(df[spec_col].mean())  if spec_col  in df.columns else np.nan
        crp   = RISK_WEIGHTS["pitch_invasion"] * pitch + \
                RISK_WEIGHTS["non_specificity"] * (1 - spec) \
                if not (np.isnan(pitch) or np.isnan(spec)) else np.nan
        rows.append({
            "role": role,
            "clinical_risk_score": round(crp, 4) if crp is not np.nan else np.nan,
            "pitch_invasion_rate": round(pitch, 4),
            "non_specificity_rate": round(1 - spec, 4),
        })
    return pd.DataFrame(rows)


risk_df = clinical_risk_penalty(df, ROLES)
print("\n=== Clinical Risk Penalty Score ===")
print(risk_df.to_string(index=False))
risk_df.to_excel(f"{OUTPUT_DIR_ROLE}/clinical_risk_score.xlsx", index=False)


# ===========================================================================
# 9. Summary table — one row per role, all key metrics
# ===========================================================================

summary_rows = []

for role in ROLES:
    acc_col  = f"ChatGPT_single_request_5_{role}_comparison"
    spec_col = f"ChatGPT_single_request_5_{role}_specific"

    def _first(frame, col):
        """Safely extract the first value of *col* from *frame*, or NaN."""
        vals = frame.loc[frame["role"] == role, col].values
        return float(vals[0]) if len(vals) else np.nan

    summary_rows.append({
        "Role":                          role.title(),
        "N":                             len(df),
        "Accuracy (%)":                  round(df[acc_col].mean() * 100, 1)
                                         if acc_col in df.columns else np.nan,
        "Specificity rate (%)":          round(df[spec_col].mean() * 100, 1)
                                         if spec_col in df.columns else np.nan,
        "Pitch invasion rate (%)":       round(_first(boundary_df, "pitch_invasion_rate") * 100, 1),
        "Cosine sim (single vs SC)":     round(_first(cosine_df,   "mean_cosine_similarity"), 3),
        "Attractor dispersion":          round(_first(pad_df,      "mean_attractor_dispersion"), 4),
        "Role confusion entropy (bits)": round(_first(rce_df,      "mean_role_confusion_entropy"), 3),
        "Boundary entropy (bits)":       round(_first(boundary_df, "boundary_entropy_bits"), 3),
        "Clinical risk score":           round(_first(risk_df,     "clinical_risk_score"), 3),
        "Persona Stability Index":       round(_first(psi_df,      "persona_stability_index"), 3),
        "Composite Robustness Index":    round(_first(cri_df,      "composite_robustness_index"), 3),
    })

summary_df = pd.DataFrame(summary_rows).set_index("Role")
summary_df.to_excel(f"{OUTPUT_DIR_ROLE}/analysis_summary.xlsx")

print(f"\nSummary → {OUTPUT_DIR_ROLE}/analysis_summary.xlsx")
print(summary_df.T.to_string())

print("\n=== Persona Stability Analysis Complete ===")
