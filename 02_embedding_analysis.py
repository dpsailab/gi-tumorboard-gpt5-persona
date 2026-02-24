"""
02_embedding_analysis.py
========================
Embedding-space analysis of specialist role personas.

Quantifies whether assigning different clinical specialist role personas to an
LLM produces meaningfully distinct response embeddings.  Analyses include:

  1. Mean pairwise cosine similarity matrix across roles.
  2. PCA of concatenated role embeddings, coloured by tumour type.
  3. (Optional) UMAP dimensionality reduction with KDE density overlays.
  4. Jensen–Shannon divergence between role embedding distributions.
  5. Inter-role centroid separation matrix.
  6. Persona Drift Score: cosine distance between single-request and
     self-consistency embeddings, per role and per case.
  7. Kruskal–Wallis test for embedding-mean separation across roles.

All results are written to ``output/`` (Excel) and ``output/advanced/`` (PNG).

Requirements
------------
  - ``umap-learn``  (only if ``DO_UMAP = True`` in config.py)
  - ``scikit-learn >= 1.0``
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.spatial.distance import cdist, jensenshannon
from scipy.stats import entropy, f_oneway, kruskal
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans


from config import (
    DATA_FILE, DO_UMAP, OUTPUT_DIR_ADVANCED, OUTPUT_DIR_ROLE,
    ROLE_COLORS, SHOW_PLOTS, SPECIALIST_COLS
)
from utils import parse_embedding, safe_cosine

# ---------------------------------------------------------------------------
# Directory setup
# ---------------------------------------------------------------------------
os.makedirs(OUTPUT_DIR_ROLE, exist_ok=True)
os.makedirs(OUTPUT_DIR_ADVANCED, exist_ok=True)

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
df = pd.read_csv(DATA_FILE)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _save_or_show(path: str) -> None:
    plt.savefig(path, dpi=300, bbox_inches="tight")
    print(f"Figure saved → {path}")
    if SHOW_PLOTS:
        plt.show()
    plt.close()


# ===========================================================================
# 1. Mean pairwise cosine similarity matrix
# ===========================================================================

role_names       = [c.replace("_treatment", "") for c in SPECIALIST_COLS]
embedding_cols   = [f"{r}_embeddings" for r in role_names]
extra_col        = "F1_MDTB_simulation_embeddings"
if extra_col not in embedding_cols:
    embedding_cols.append(extra_col)

labels = [c.replace("_embeddings", "") for c in embedding_cols]

print("\n=== Computing pairwise cosine similarity matrix ===")
sim_matrix = pd.DataFrame(np.nan, index=labels, columns=labels)

for i, col1 in enumerate(embedding_cols):
    for j, col2 in enumerate(embedding_cols):
        if col1 not in df.columns or col2 not in df.columns:
            continue
        sims = []
        for raw1, raw2 in zip(df[col1], df[col2]):
            s = safe_cosine(parse_embedding(raw1), parse_embedding(raw2))
            if s is not None:
                sims.append(s)
        sim_matrix.loc[labels[i], labels[j]] = np.mean(sims) if sims else np.nan

sim_matrix.to_excel(f"{OUTPUT_DIR_ROLE}/similarity_matrix_roles.xlsx")
print(sim_matrix.round(4))

# Pretty-label the matrix for the heatmap
pretty_map = {
    "F1_MDTB_simulation":                  "Simulated Tumorboard",
    "F3_persona_surgeon":                  "Surgeon",
    "F4_persona_medical_oncologist":       "Oncologist",
    "F5_persona_radiation_oncologist":     "Radio-Oncologist",
}
sim_display = sim_matrix.rename(index=pretty_map, columns=pretty_map)

plt.figure(figsize=(8, 6))
sns.heatmap(sim_display.astype(float), annot=True, fmt=".3f",
            cmap="viridis", vmin=0.85, vmax=1.0)
plt.title("Mean Pairwise Cosine Similarity — Role Embeddings")
plt.xlabel("Role")
plt.ylabel("Role")
_save_or_show(f"{OUTPUT_DIR_ROLE}/embedding_similarity_heatmap.png")


# ===========================================================================
# 2. PCA of concatenated role embeddings
# ===========================================================================

role_emb_colnames = [c.replace("_treatment", "") + "_embeddings" for c in SPECIALIST_COLS]

print("\n=== Running PCA on concatenated role embeddings ===")
combined_vectors, valid_idx = [], []

for idx, row in df.iterrows():
    parts, ok = [], True
    for emb_col in role_emb_colnames:
        v = parse_embedding(row.get(emb_col))
        if v is None:
            ok = False
            break
        parts.append(v)
    if ok:
        combined_vectors.append(np.concatenate(parts))
        valid_idx.append(idx)

if len(combined_vectors) >= 5:
    X   = np.vstack(combined_vectors)
    pca = PCA(n_components=2, random_state=42)
    X2  = pca.fit_transform(X)

    k  = min(5, max(2, int(len(X2) ** 0.5)))
    km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(X2)

    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        x=X2[:, 0], y=X2[:, 1],
        hue=[df.loc[i, "tumour_type"] for i in valid_idx],
        style=km.labels_,
        palette="tab10",
        alpha=0.75,
    )
    plt.title("PCA — Concatenated Role Embeddings (coloured by tumour type)")
    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f} %)")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f} %)")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    _save_or_show(f"{OUTPUT_DIR_ADVANCED}/pca_concat_role_embeddings.png")

    # ------------------------------------------------------------------
    # PCA: per-role and per-tumour centroid / dispersion statistics
    # ------------------------------------------------------------------
    # Build a flat record set: one row per embedding (role × case)
    # so we can compute role centroids and per-tumour separation in 2-D.
    # Note: these embeddings are NOT concatenated — we use individual role
    # columns to keep role identity intact for centroid computation.

    fw_cols = {
        "Tumorboard":      "ChatGPT_single_request_5_embeddings",
        "Surgeon":         "ChatGPT_single_request_5_surgeon_embeddings",
        "Oncologist":      "ChatGPT_single_request_5_oncologist_embeddings",
        "Radio-oncologist":"ChatGPT_single_request_5_radio-oncologist_embeddings",
    }

    records = []
    for idx, row in df.iterrows():
        for role, col in fw_cols.items():
            v = parse_embedding(row.get(col))
            if v is not None:
                records.append({
                    "role":  role,
                    "tumor": str(row["tumour_type"]),
                    "embedding": v,
                })

    if len(records) >= 20:
        rec_df   = pd.DataFrame(records)
        all_vecs = np.vstack(rec_df["embedding"].values)

        pca_role          = PCA(n_components=2, random_state=42)
        coords            = pca_role.fit_transform(all_vecs)
        rec_df["PC1"]     = coords[:, 0]
        rec_df["PC2"]     = coords[:, 1]

        ev = pca_role.explained_variance_ratio_
        print(f"\nPCA explained variance: PC1={ev[0]*100:.1f}%, PC2={ev[1]*100:.1f}%")

        # ---- Global role centroids and pairwise distances ----
        from scipy.spatial.distance import pdist, squareform

        centroids  = rec_df.groupby("role")[["PC1", "PC2"]].mean()
        dispersion = rec_df.groupby("role")[["PC1", "PC2"]].std()

        sep_matrix = pd.DataFrame(
            squareform(pdist(centroids.values)),
            index=centroids.index,
            columns=centroids.index,
        )

        print("\n=== PCA Role Centroids (PC1, PC2) ===")
        print(centroids.round(4))
        print("\n=== PCA Pairwise Centroid Distances ===")
        print(sep_matrix.round(4))
        print("\n=== PCA Within-Role Dispersion (SD along PC1, PC2) ===")
        print(dispersion.round(4))

        # ---- Per-tumour centroid distances ----
        tumor_stats_rows = []
        for tumor_de in df["tumour_type"].unique():
            sub = rec_df[rec_df["tumor"] == tumor_de]
            if len(sub) < 8:
                continue
            sub_centroids = sub.groupby("role")[["PC1", "PC2"]].mean()
            sub_disp      = sub.groupby("role")[["PC1", "PC2"]].std()
            sub_sep       = pd.DataFrame(
                squareform(pdist(sub_centroids.values)),
                index=sub_centroids.index,
                columns=sub_centroids.index,
            )
            # Distance from each role centroid to the Tumorboard centroid
            if "Tumorboard" in sub_centroids.index:
                tb = sub_centroids.loc["Tumorboard"].values
                for role in sub_centroids.index:
                    if role == "Tumorboard":
                        continue
                    d = float(np.linalg.norm(sub_centroids.loc[role].values - tb))
                    tumor_stats_rows.append({
                        "tumour":        tumor_de,
                        "role":          role,
                        "dist_to_TB":    round(d, 4),
                        "disp_PC1":      round(float(sub_disp.loc[role, "PC1"]), 4)
                                         if role in sub_disp.index else np.nan,
                        "disp_PC2":      round(float(sub_disp.loc[role, "PC2"]), 4)
                                         if role in sub_disp.index else np.nan,
                    })

            print(f"\n  --- {tumor_de} ---")
            print(f"  Centroids:\n{sub_centroids.round(4)}")
            print(f"  Separation:\n{sub_sep.round(4)}")
            print(f"  Dispersion:\n{sub_disp.round(4)}")

        tumor_pca_df = pd.DataFrame(tumor_stats_rows)
        tumor_pca_df.to_excel(
            f"{OUTPUT_DIR_ADVANCED}/pca_per_tumour_centroid_distances.xlsx",
            index=False,
        )
        print(f"\nPer-tumour PCA stats saved → "
              f"{OUTPUT_DIR_ADVANCED}/pca_per_tumour_centroid_distances.xlsx")

        # Save global stats
        centroids.to_excel(f"{OUTPUT_DIR_ADVANCED}/pca_global_role_centroids.xlsx")
        sep_matrix.to_excel(f"{OUTPUT_DIR_ADVANCED}/pca_global_centroid_separation.xlsx")
        dispersion.to_excel(f"{OUTPUT_DIR_ADVANCED}/pca_global_role_dispersion.xlsx")

else:
    print("Insufficient complete rows for PCA — skipping.")


# ===========================================================================
# 3. UMAP (optional, controlled by DO_UMAP flag)
# ===========================================================================

if DO_UMAP:
    try:
        from umap import UMAP
    except ImportError:
        print("umap-learn not installed; skipping UMAP.  Install with: pip install umap-learn")
        DO_UMAP = False

if DO_UMAP:
    print("\n=== Running UMAP persona landscape ===")

    framework_sets = {
        "Single_request": {
            "Tumorboard":      "ChatGPT_single_request_5_embeddings",
            "Surgeon":         "ChatGPT_single_request_5_surgeon_embeddings",
            "Oncologist":      "ChatGPT_single_request_5_oncologist_embeddings",
            "Radio-oncologist":"ChatGPT_single_request_5_radio-oncologist_embeddings",
        },
        "Self_consistency": {
            "Tumorboard":      "ChatGPT_single_request_5_self-consistency_tumorboard_embeddings",
            "Surgeon":         "ChatGPT_single_request_5_self-consistency_surgeon_embeddings",
            "Oncologist":      "ChatGPT_single_request_5_self-consistency_oncologist_embeddings",
            "Radio-oncologist":"ChatGPT_single_request_5_self-consistency_radio-oncologist_embeddings",
        },
    }

    tumor_map_de = {en: de for en, de in TUMOR_MAP.items()}
    tumor_list_en = TUMOR_LIST

    for fw_name, fw_cols in framework_sets.items():
        records = []
        for idx, row in df.iterrows():
            for role, col in fw_cols.items():
                emb = parse_embedding(row.get(col))
                if emb is not None:
                    records.append({"idx": idx, "role": role,
                                    "tumor": str(row["tumour_type"]),
                                    "embedding": emb})
        if len(records) < 50:
            print(f"{fw_name}: too few samples for UMAP — skipping.")
            continue

        master = pd.DataFrame(records)
        vectors   = np.vstack(master["embedding"].values)
        roles_arr = master["role"].values
        tumor_arr = master["tumor"].values

        umap_model = UMAP(n_components=2, n_neighbors=30,
                          min_dist=0.1, random_state=42)
        X2 = umap_model.fit_transform(vectors)

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()

        ax0 = axes[0]
        for role, color in ROLE_COLORS.items():
            mask = roles_arr == role
            if not mask.any():
                continue
            ax0.scatter(X2[mask, 0], X2[mask, 1], s=35, alpha=0.6,
                        color=color, label=role)
            if mask.sum() > 15:
                try:
                    sns.kdeplot(x=X2[mask, 0], y=X2[mask, 1], fill=True,
                                alpha=0.2, levels=[0.25, 1], color=color, ax=ax0)
                except Exception:
                    pass
        ax0.set_title(f"{fw_name} — Global Persona Space")
        ax0.grid(alpha=0.3)

        for pos_i, tumor_en in enumerate(tumor_list_en, start=1):
            if pos_i >= len(axes):
                break
            ax = axes[pos_i]
            tumor_de = tumor_map_de.get(tumor_en, tumor_en)
            t_mask = np.char.find(
                np.char.lower(tumor_arr.astype(str)), tumor_de.lower()
            ) >= 0
            plotted = False
            for role, color in ROLE_COLORS.items():
                final_mask = (roles_arr == role) & t_mask
                if final_mask.sum() < 5:
                    continue
                plotted = True
                ax.scatter(X2[final_mask, 0], X2[final_mask, 1],
                           s=40, alpha=0.7, color=color)
                if final_mask.sum() > 15:
                    try:
                        sns.kdeplot(x=X2[final_mask, 0], y=X2[final_mask, 1],
                                    fill=True, alpha=0.25, levels=[0.25, 1],
                                    color=color, ax=ax)
                    except Exception:
                        pass
            if plotted:
                ax.set_title(tumor_en)
                ax.grid(alpha=0.3)

        handles = [plt.scatter([], [], color=c, label=r)
                   for r, c in ROLE_COLORS.items()]
        fig.legend(handles=handles, loc="upper right",
                   bbox_to_anchor=(0.98, 0.98))
        plt.suptitle(f"{fw_name} — UMAP Clinical Persona Landscape", fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        _save_or_show(f"{OUTPUT_DIR_ADVANCED}/umap_{fw_name}.png")

    print("UMAP complete.")


# ===========================================================================
# 4. Jensen–Shannon divergence between role distributions
# ===========================================================================

print("\n=== Jensen–Shannon divergence between role embedding distributions ===")

# Explicit role → embedding column mapping (MOST ROBUST APPROACH)
role_embedding_map = {
    "Surgeon":          "F3_persona_surgeon_embeddings",
    "Oncologist":       "F4_persona_medical_oncologist_embeddings",
    "Radio-Oncologist": "F5_persona_radiation_oncologist_embeddings",
}

role_vectors = {role: [] for role in role_embedding_map}

# Collect embeddings
for _, row in df.iterrows():
    for role, col in role_embedding_map.items():
        v = parse_embedding(row.get(col))
        if v is not None:
            role_vectors[role].append(v)

# Convert to numpy arrays
for role in role_vectors:
    if role_vectors[role]:
        role_vectors[role] = np.array(role_vectors[role])

roles_list = [r for r in role_vectors if isinstance(role_vectors[r], np.ndarray)]

# Compute JSD pairwise
js_rows = []

for i in range(len(roles_list)):
    for j in range(i + 1, len(roles_list)):
        r1, r2 = roles_list[i], roles_list[j]

        try:
            hist1 = np.histogram(
                role_vectors[r1].flatten(),
                bins=50,
                density=True
            )[0] + 1e-9

            hist2 = np.histogram(
                role_vectors[r2].flatten(),
                bins=50,
                density=True
            )[0] + 1e-9

            d = jensenshannon(hist1, hist2)

        except Exception:
            d = np.nan

        js_rows.append({
            "role_A": r1,
            "role_B": r2,
            "jensen_shannon_divergence": d
        })

        print(f"  {r1} vs {r2}: JSD = {d:.4f}")

js_df = pd.DataFrame(js_rows)

js_df.to_excel(
    f"{OUTPUT_DIR_ROLE}/role_distribution_divergence.xlsx",
    index=False
)


# ===========================================================================
# 5. Inter-role centroid separation
# ===========================================================================

print("\n=== Inter-role centroid separation ===")

role_centroids = {r: np.mean(role_vectors[r], axis=0)
                  for r in roles_list if len(role_vectors[r]) >= 5}

centroid_sep = pd.DataFrame(
    cdist(list(role_centroids.values()), list(role_centroids.values())),
    index=role_centroids.keys(),
    columns=role_centroids.keys(),
)
print(centroid_sep.round(4))
centroid_sep.to_excel(f"{OUTPUT_DIR_ROLE}/role_centroid_separation.xlsx")


# ===========================================================================
# 6. Persona Drift Score (single-request vs self-consistency)
# ===========================================================================

print("\n=== Persona Drift Score ===")

# Explicit mappings (ROBUST APPROACH)
single_request_map = {
    "Surgeon": "F3_persona_surgeon_embeddings",
    "Oncologist": "F4_persona_medical_oncologist_embeddings",
    "Radio-Oncologist": "F5_persona_radiation_oncologist_embeddings",
}

selfconsistency_map = {
    "Surgeon": "F2_multi_expert_consensus_surgeon_embeddings",
    "Oncologist": "F2_multi_expert_consensus_oncologist_embeddings",
    "Radio-Oncologist": "F2_multi_expert_consensus_radio-oncologist_embeddings",
}

case_rows = []

for role in single_request_map.keys():

    single_col = single_request_map[role]
    self_col   = selfconsistency_map[role]

    if single_col not in df.columns or self_col not in df.columns:
        print(f"Missing embedding columns for {role}")
        continue

    for idx, row in df.iterrows():

        s = parse_embedding(row.get(single_col))
        t = parse_embedding(row.get(self_col))

        drift = safe_cosine(s, t)

        if drift is not None:
            case_rows.append({
                "case_id": idx,
                "role": role,
                "tumor": row["tumour_type"],
                "cosine_drift": drift,
            })

drift_df = pd.DataFrame(case_rows)


# ---------------------------------------------------------------------------
# Drift summary per role
# ---------------------------------------------------------------------------

role_drift_summary = (
    drift_df.groupby("role")["cosine_drift"]
    .agg(["mean", "std", "count"])
    .reset_index()
    .rename(columns={
        "mean": "mean_drift",
        "std": "std_drift",
        "count": "n"
    })
)

print(role_drift_summary.to_string(index=False))


# ---------------------------------------------------------------------------
# Drift by tumour type
# ---------------------------------------------------------------------------

role_tumor_drift = (
    drift_df.groupby(["role", "tumor"])["cosine_drift"]
    .agg(["mean", "std", "count"])
    .reset_index()
)


# ---------------------------------------------------------------------------
# Kruskal-Wallis test across roles
# ---------------------------------------------------------------------------

groups = [
    drift_df[drift_df["role"] == r]["cosine_drift"].values
    for r in drift_df["role"].unique()
    if len(drift_df[drift_df["role"] == r]) > 5
]

if len(groups) >= 2:
    stat, pval = kruskal(*groups)
    print(f"\nKruskal–Wallis across roles: H = {stat:.4f}, p = {pval:.6f}")


# ---------------------------------------------------------------------------
# Centroid-level drift
# ---------------------------------------------------------------------------

print("\nCentroid-level drift:")

centroid_drift_rows = []

for role in single_request_map.keys():

    single_col = single_request_map[role]
    self_col   = selfconsistency_map[role]

    sv, tv = [], []

    for _, row in df.iterrows():

        s = parse_embedding(row.get(single_col))
        t = parse_embedding(row.get(self_col))

        if s is not None and t is not None:
            sv.append(s)
            tv.append(t)

    if len(sv) > 0:

        centroid_drift_rows.append({
            "role": role,
            "centroid_cosine_drift": safe_cosine(
                np.mean(np.vstack(sv), axis=0),
                np.mean(np.vstack(tv), axis=0),
            ),
            "n_cases": len(sv),
        })

centroid_drift_df = pd.DataFrame(centroid_drift_rows)

print(centroid_drift_df.to_string(index=False))


# ---------------------------------------------------------------------------
# Save results
# ---------------------------------------------------------------------------

with pd.ExcelWriter(f"{OUTPUT_DIR_ROLE}/persona_drift_full_analysis.xlsx") as w:
    drift_df.to_excel(w, sheet_name="case_level", index=False)
    role_drift_summary.to_excel(w, sheet_name="role_summary", index=False)
    role_tumor_drift.to_excel(w, sheet_name="role_x_tumor", index=False)
    centroid_drift_df.to_excel(w, sheet_name="centroid_drift", index=False)

print(f"Persona drift saved → {OUTPUT_DIR_ROLE}/persona_drift_full_analysis.xlsx")


# ===========================================================================
# 7. Statistical separation test between roles (embedding means)
# ===========================================================================

print("\n=== Statistical separation test between roles ===")

role_means = {}

for role in single_request_map.keys():

    col = selfconsistency_map[role]

    vals = []

    if col in df.columns:
        for _, row in df.iterrows():
            emb = parse_embedding(row.get(col))
            if emb is not None:
                vals.append(float(np.mean(emb)))

    role_means[role] = vals


groups = [v for v in role_means.values() if len(v) > 5]

if len(groups) >= 2:
    stat, pval = f_oneway(*groups)
    print(f"One-way ANOVA (embedding means): F = {stat:.4f}, p = {pval:.6f}")
else:
    print("Insufficient groups for ANOVA.")

print("\n=== Embedding Analysis Complete ===")