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

All results are written to ``role/`` (Excel) and ``img/advanced/`` (PNG).

Requirements
------------
  - ``umap-learn``  (only if ``DO_UMAP = True`` in config.py)
  - ``scikit-learn >= 1.0``
"""

import os
from ast import literal_eval

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.spatial.distance import cdist, jensenshannon
from scipy.stats import entropy, f_oneway, kruskal
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

from config import (
    DATA_FILE, DO_UMAP, OUTPUT_DIR_ADVANCED, OUTPUT_DIR_ROLE,
    ROLE_COLORS, SHOW_PLOTS, SPECIALIST_COLS, TUMOR_LIST, TUMOR_MAP,
)
from utils import compute_majority_treatment, parse_embedding, safe_cosine

# ---------------------------------------------------------------------------
# Directory setup
# ---------------------------------------------------------------------------
os.makedirs(OUTPUT_DIR_ROLE, exist_ok=True)
os.makedirs(OUTPUT_DIR_ADVANCED, exist_ok=True)

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
df = pd.read_excel(DATA_FILE)

df["majority"] = df.apply(
    lambda row: compute_majority_treatment(row, SPECIALIST_COLS), axis=1
)

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
extra_col        = "ChatGPT_single_request_5_embeddings"
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
    "ChatGPT_single_request_5_surgeon":          "Surgeon",
    "ChatGPT_single_request_5_oncologist":       "Oncologist",
    "ChatGPT_single_request_5_radio-oncologist": "Radio-Oncologist",
    "ChatGPT_single_request_5":                  "Simulated Tumorboard",
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
        hue=[df.loc[i, "Anmeldediagnose"] for i in valid_idx],
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
                                    "tumor": str(row["Anmeldediagnose"]),
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

role_vectors: dict = {r: [] for r in ["surgeon", "oncologist", "radio-oncologist"]}

for _, row in df.iterrows():
    for role in role_vectors:
        v = parse_embedding(row.get(f"ChatGPT_single_request_5_{role}_embeddings"))
        if v is not None:
            role_vectors[role].append(v)

for role in role_vectors:
    if role_vectors[role]:
        role_vectors[role] = np.array(role_vectors[role])

roles_list = [r for r in role_vectors if isinstance(role_vectors[r], np.ndarray)]

js_rows = []
for i in range(len(roles_list)):
    for j in range(i + 1, len(roles_list)):
        r1, r2 = roles_list[i], roles_list[j]
        try:
            hist1 = np.histogram(role_vectors[r1].flatten(), bins=50, density=True)[0] + 1e-9
            hist2 = np.histogram(role_vectors[r2].flatten(), bins=50, density=True)[0] + 1e-9
            d = jensenshannon(hist1, hist2)
        except Exception:
            d = np.nan
        js_rows.append({"role_A": r1, "role_B": r2, "jensen_shannon_divergence": d})
        print(f"  {r1} vs {r2}: JSD = {d:.4f}")

js_df = pd.DataFrame(js_rows)
js_df.to_excel(f"{OUTPUT_DIR_ROLE}/role_distribution_divergence.xlsx", index=False)


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

roles_title = ["Surgeon", "Oncologist", "Radio-oncologist"]
case_rows   = []

for role in roles_title:
    single_col = f"ChatGPT_single_request_5_{role.lower()}_embeddings"
    self_col   = f"ChatGPT_single_request_5_self-consistency_{role.lower()}_embeddings"

    if single_col not in df.columns or self_col not in df.columns:
        print(f"  Missing embedding columns for {role} — skipping.")
        continue

    for idx, row in df.iterrows():
        s = parse_embedding(row[single_col])
        t = parse_embedding(row[self_col])
        drift = safe_cosine(s, t)
        if drift is not None:
            case_rows.append({
                "case_id":      idx,
                "role":         role,
                "tumor_raw":    str(row["Anmeldediagnose"]),
                "cosine_drift": drift,
            })

drift_df = pd.DataFrame(case_rows)


def _match_tumor(raw: str) -> str:
    for en, de in TUMOR_MAP.items():
        if de.lower() in raw.lower():
            return en
    return "Other"


drift_df["tumor"] = drift_df["tumor_raw"].apply(_match_tumor)

role_drift_summary = (
    drift_df.groupby("role")["cosine_drift"]
    .agg(["mean", "std", "count"])
    .reset_index()
    .rename(columns={"mean": "mean_drift", "std": "std_drift", "count": "n"})
)
print(role_drift_summary.to_string(index=False))

role_tumor_drift = (
    drift_df.groupby(["role", "tumor"])["cosine_drift"]
    .agg(["mean", "std", "count"])
    .reset_index()
)

# Statistical test between roles (Kruskal–Wallis, non-parametric)
groups = [drift_df[drift_df["role"] == r]["cosine_drift"].values
          for r in roles_title if r in drift_df["role"].values]
if len(groups) >= 2:
    stat, pval = kruskal(*groups)
    print(f"\nKruskal–Wallis across roles: H = {stat:.4f}, p = {pval:.6f}")

# Centroid-level drift
centroid_drift_rows = []
for role in roles_title:
    single_col = f"ChatGPT_single_request_5_{role.lower()}_embeddings"
    self_col   = f"ChatGPT_single_request_5_self-consistency_{role.lower()}_embeddings"
    if single_col not in df.columns or self_col not in df.columns:
        continue
    sv, tv = [], []
    for _, row in df.iterrows():
        s = parse_embedding(row[single_col])
        t = parse_embedding(row[self_col])
        if s is not None and t is not None:
            sv.append(s); tv.append(t)
    if sv:
        centroid_drift_rows.append({
            "role": role,
            "centroid_cosine_drift": safe_cosine(
                np.mean(np.vstack(sv), axis=0),
                np.mean(np.vstack(tv), axis=0),
            ),
            "n_cases": len(sv),
        })

centroid_drift_df = pd.DataFrame(centroid_drift_rows)
print("\nCentroid-level drift:")
print(centroid_drift_df.to_string(index=False))

with pd.ExcelWriter(f"{OUTPUT_DIR_ROLE}/persona_drift_full_analysis.xlsx") as w:
    drift_df.to_excel(w, sheet_name="case_level", index=False)
    role_drift_summary.to_excel(w, sheet_name="role_summary", index=False)
    role_tumor_drift.to_excel(w, sheet_name="role_x_tumor", index=False)
    centroid_drift_df.to_excel(w, sheet_name="centroid_drift", index=False)
print(f"Persona drift saved → {OUTPUT_DIR_ROLE}/persona_drift_full_analysis.xlsx")


# ===========================================================================
# 7. Kruskal–Wallis / ANOVA on embedding mean values
# ===========================================================================

print("\n=== Statistical separation test between roles (embedding means) ===")

role_means: dict = {}
for role in ["surgeon", "oncologist", "radio-oncologist"]:
    col = f"ChatGPT_single_request_5_self-consistency_{role}_embeddings"
    vals = []
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
