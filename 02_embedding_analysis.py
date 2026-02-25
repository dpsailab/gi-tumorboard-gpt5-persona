"""
02_embedding_analysis.py
========================

Embedding-space analysis (Single Request vs Self-Consistency Personas).

This pipeline evaluates whether different clinical persona prompting
strategies produce systematically distinct embedding representations.

Analyses include:

1. Pairwise cosine similarity between persona roles.
2. PCA structural analysis of embedding manifolds:
   - Global role centroids
   - Within-role dispersion
   - Tumour-specific centroid separation
3. Jensen–Shannon divergence between role embedding distributions.
4. Centroid separation matrices in latent space.
5. Statistical robustness testing using:
   - Kruskal–Wallis tests on PCA projections.
   - Effect size estimation via variance decomposition.
6. Persona drift quantification between:
   - Single-request persona responses
   - Self-consistency persona responses

Outputs are stored in:
``output/embedding_analysis/``

Requirements:
------------
- scikit-learn >= 1.0
- scipy >= 1.7
- pandas >= 1.3
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.spatial.distance import cdist, jensenshannon
from scipy.stats import kruskal
from sklearn.decomposition import PCA

from config import (
    DATA_FILE,
    OUTPUT_DIR_ROLE,
    SHOW_PLOTS,
    SINGLE_REQUEST_EMBEDDING_COLS,
    SELF_CONSISTENCY_EMBEDDING_COLS,
    ROLE_CONFIG
)

from utils import parse_embedding, safe_cosine

# ------------------------------------------------------
# Setup
# ------------------------------------------------------

TABLE_DIR = os.path.join(OUTPUT_DIR_ROLE, "embedding_analysis")
IMG_DIR = os.path.join(OUTPUT_DIR_ROLE, "embedding_analysis/img")

os.makedirs(TABLE_DIR, exist_ok=True)
os.makedirs(IMG_DIR, exist_ok=True)

df = pd.read_csv(DATA_FILE)

role_colors = {
    role: ROLE_CONFIG[role]["color"]
    for role in ROLE_CONFIG
}

# ------------------------------------------------------
# Helpers
# ------------------------------------------------------

def _save_or_show(path):
    plt.savefig(path, dpi=300, bbox_inches="tight")
    print(f"Saved → {path}")
    if SHOW_PLOTS:
        plt.show()
    plt.close()


def plot_persona_pca_space(
        df,
        embedding_map: dict,
        title: str,
        output_path: str,
        include_tumorboard: bool = False
):

    print(f"\n=== PCA {title} ===")

    tumor_list = df["tumour_type"].dropna().unique()

    vectors, roles, tumors = [], [], []

    # -----------------------------
    # Build dataset
    # -----------------------------

    for idx, row in df.iterrows():

        tumor_val = str(row["tumour_type"])

        for role, col in embedding_map.items():
            # Optionally skip tumorboard if not needed
            if not include_tumorboard and role == "Simulated Tumorboard":
                continue

            emb = parse_embedding(row.get(col))

            if emb is None:
                continue

            vectors.append(emb)
            roles.append(role)
            tumors.append(tumor_val)

    if len(vectors) < 20:
        print("Not enough samples for PCA")
        return

    vectors = np.array(vectors)
    roles = np.array(roles)
    tumors = np.array(tumors)

    # -----------------------------
    # PCA
    # -----------------------------

    pca = PCA(n_components=2, random_state=42)
    X2 = pca.fit_transform(vectors)

    print(
        f"PCA variance explained: "
        f"PC1={pca.explained_variance_ratio_[0]*100:.1f}% "
        f"PC2={pca.explained_variance_ratio_[1]*100:.1f}%"
    )

    # -----------------------------
    # Plot
    # -----------------------------

    fig = plt.figure(figsize=(18,12))
    gs = plt.GridSpec(2,3, figure=fig, wspace=0.35, hspace=0.35)

    ax0 = fig.add_subplot(gs[0,0])

    # Overall plot
    for role, color in role_colors.items():
        mask = roles == role

        if mask.sum() == 0:
            continue

        ax0.scatter(
            X2[mask,0],
            X2[mask,1],
            alpha=0.6,
            s=35,
            color=color,
            label=role
        )

        if mask.sum() > 15:
            sns.kdeplot(
                x=X2[mask,0],
                y=X2[mask,1],
                fill=True,
                levels=[0.25,1],
                alpha=0.25,
                color=color,
                ax=ax0
            )

    ax0.set_title(f"Overall Persona Space")
    ax0.grid()

    # -----------------------------
    # Tumor panels
    # -----------------------------

    pos_map = [(0,1),(0,2),(1,0),(1,1),(1,2)]

    for i, tumor in enumerate(tumor_list):

        if i >= len(pos_map):
            break

        ax = fig.add_subplot(gs[pos_map[i]])

        tumor_mask_global = tumors == tumor

        for role, color in role_colors.items():

            final_mask = (roles == role) & tumor_mask_global

            if final_mask.sum() < 5:
                continue

            ax.scatter(
                X2[final_mask,0],
                X2[final_mask,1],
                s=35,
                alpha=0.6,
                color=color
            )

            if final_mask.sum() > 15:
                sns.kdeplot(
                    x=X2[final_mask,0],
                    y=X2[final_mask,1],
                    fill=True,
                    levels=[0.25, 1],
                    alpha=0.25,
                    color=color,
                    ax=ax
                )



        ax.set_title(tumor)
        ax.grid()

    handles, labels = ax0.get_legend_handles_labels()

    fig.legend(
        handles,
        labels,
        loc="upper right",
        bbox_to_anchor=(0.98,0.98)
    )

    plt.suptitle(title, fontsize=16)

    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"----> saved {output_path}")

    plt.close()

# ======================================================
# Generic Embedding Analysis Pipeline
# ======================================================

def run_embedding_analysis(df, embedding_dict, title_prefix, save_prefix):

    print(f"\n=== {title_prefix} Embedding Analysis ===")

    roles = list(embedding_dict.keys())
    emb_cols = list(embedding_dict.values())

    # ==================================================
    # 1. Pairwise cosine similarity matrix
    # ==================================================

    print("\n--- Pairwise cosine similarity ---")

    sim_matrix = pd.DataFrame(np.nan, index=roles, columns=roles)

    for i, r1 in enumerate(roles):
        for j, r2 in enumerate(roles):

            col1 = embedding_dict[r1]
            col2 = embedding_dict[r2]

            if col1 not in df.columns or col2 not in df.columns:
                continue

            sims = []

            for a, b in zip(df[col1], df[col2]):
                s = safe_cosine(parse_embedding(a), parse_embedding(b))
                if s is not None:
                    sims.append(s)

            if sims:
                sim_matrix.loc[r1, r2] = np.mean(sims)

    print(sim_matrix.round(4))

    sim_matrix.to_excel(
        f"{TABLE_DIR}/{save_prefix}_similarity_matrix.xlsx"
    )

    plt.figure(figsize=(8, 6))
    sns.heatmap(sim_matrix.astype(float), annot=True, fmt=".3f",
                cmap="viridis", vmin=0.85, vmax=1.0)
    plt.title(f"Mean Pairwise Cosine Similarity — {save_prefix} Role Embeddings")
    plt.xlabel("Role")
    plt.ylabel("Role")
    _save_or_show(f"{IMG_DIR}/{save_prefix}embedding_similarity_heatmap.png")

    # ==================================================
    # 2. Advanced PCA structural analysis (per-role)
    # ==================================================

    print("\n--- Advanced PCA structural analysis ---")

    records = []

    # --------------------------------------------------
    # Build flat record set (role × case)
    # --------------------------------------------------

    for _, row in df.iterrows():

        tumor_val = str(row["tumour_type"])

        for role in roles:

            emb = parse_embedding(row.get(embedding_dict[role]))

            if emb is not None:
                records.append({
                    "role": role,
                    "tumor": tumor_val,
                    "embedding": emb
                })

    if len(records) < 20:
        print("Not enough samples for PCA — skipping.")
    else:

        rec_df = pd.DataFrame(records)

        all_vecs = np.vstack(rec_df["embedding"].values)

        # --------------------------------------------------
        # PCA projection
        # --------------------------------------------------

        pca = PCA(n_components=2, random_state=42)
        coords = pca.fit_transform(all_vecs)

        rec_df["PC1"] = coords[:, 0]
        rec_df["PC2"] = coords[:, 1]

        ev = pca.explained_variance_ratio_
        print(f"PCA explained variance: PC1={ev[0] * 100:.1f}%, PC2={ev[1] * 100:.1f}%")

        # --------------------------------------------------
        # Global role centroids
        # --------------------------------------------------

        centroids = rec_df.groupby("role")[["PC1", "PC2"]].mean()
        dispersion = rec_df.groupby("role")[["PC1", "PC2"]].std()

        from scipy.spatial.distance import pdist, squareform

        sep_matrix = pd.DataFrame(
            squareform(pdist(centroids.values)),
            index=centroids.index,
            columns=centroids.index,
        )

        print("\n=== PCA Global Role Centroids ===")
        print(centroids.round(4))

        print("\n=== PCA Pairwise Centroid Distances ===")
        print(sep_matrix.round(4))

        print("\n=== PCA Within-Role Dispersion (SD PC1, PC2) ===")
        print(dispersion.round(4))

        # Save global stats
        centroids.to_excel(
            f"{TABLE_DIR}/{save_prefix}_pca_global_role_centroids.xlsx"
        )
        sep_matrix.to_excel(
            f"{TABLE_DIR}/{save_prefix}_pca_global_centroid_separation.xlsx"
        )
        dispersion.to_excel(
            f"{TABLE_DIR}/{save_prefix}_pca_global_role_dispersion.xlsx"
        )

        # --------------------------------------------------
        # Per-tumour centroid distances
        # --------------------------------------------------

        tumor_stats_rows = []

        for tumor_val in rec_df["tumor"].unique():

            sub = rec_df[rec_df["tumor"] == tumor_val]

            if len(sub) < 8:
                continue

            sub_centroids = sub.groupby("role")[["PC1", "PC2"]].mean()
            sub_disp = sub.groupby("role")[["PC1", "PC2"]].std()

            sub_sep = pd.DataFrame(
                squareform(pdist(sub_centroids.values)),
                index=sub_centroids.index,
                columns=sub_centroids.index,
            )

            print(f"\n--- {tumor_val} ---")
            print("Centroids:")
            print(sub_centroids.round(4))
            print("Separation:")
            print(sub_sep.round(4))
            print("Dispersion:")
            print(sub_disp.round(4))

            # Optional: distance to Tumorboard if present
            if "Simulated Tumorboard" in sub_centroids.index:

                tb = sub_centroids.loc["Simulated Tumorboard"].values

                for role in sub_centroids.index:

                    if role == "Simulated Tumorboard":
                        continue

                    d = float(
                        np.linalg.norm(
                            sub_centroids.loc[role].values - tb
                        )
                    )

                    tumor_stats_rows.append({
                        "tumour": tumor_val,
                        "role": role,
                        "dist_to_TB": round(d, 4),
                        "disp_PC1": round(float(sub_disp.loc[role, "PC1"]), 4)
                        if role in sub_disp.index else np.nan,
                        "disp_PC2": round(float(sub_disp.loc[role, "PC2"]), 4)
                        if role in sub_disp.index else np.nan,
                    })

        tumor_pca_df = pd.DataFrame(tumor_stats_rows)

        tumor_pca_df.to_excel(
            f"{TABLE_DIR}/{save_prefix}_pca_per_tumour_centroid_distances.xlsx",
            index=False
        )

        print(f"\nPer-tumour PCA stats saved → "
              f"{TABLE_DIR}/{save_prefix}_pca_per_tumour_centroid_distances.xlsx")

    # =================================================
    # 2 bis. PCA PLOT
    # =================================================

    plot_persona_pca_space(
        df=df,
        embedding_map=embedding_dict,
        title=f"{save_prefix} — PCA Persona Space",
        output_path=f"{IMG_DIR}/pca_{save_prefix}_persona_space.png",
        include_tumorboard=True
    )

    # ==================================================
    # 3. Jensen-Shannon divergence
    # ==================================================

    print("\n--- Jensen-Shannon divergence ---")

    role_vectors = {r: [] for r in roles}

    for _, row in df.iterrows():
        for r in roles:
            v = parse_embedding(row.get(embedding_dict[r]))
            if v is not None:
                role_vectors[r].append(v)

    for r in role_vectors:
        if len(role_vectors[r]) > 0:
            role_vectors[r] = np.array(role_vectors[r])

    js_rows = []

    for i in range(len(roles)):
        for j in range(i+1, len(roles)):

            r1, r2 = roles[i], roles[j]

            try:
                h1 = np.histogram(
                    role_vectors[r1].flatten(),
                    bins=50,
                    density=True
                )[0] + 1e-9

                h2 = np.histogram(
                    role_vectors[r2].flatten(),
                    bins=50,
                    density=True
                )[0] + 1e-9

                d = jensenshannon(h1, h2)

            except:
                d = np.nan

            js_rows.append({
                "role_A": r1,
                "role_B": r2,
                "jsd": d
            })

    pd.DataFrame(js_rows).to_excel(
        f"{TABLE_DIR}/{save_prefix}_jsd.xlsx",
        index=False
    )

    # ==================================================
    # 4. Centroid separation
    # ==================================================

    print("\n--- Centroid separation ---")

    centroids = {
        r: np.mean(role_vectors[r], axis=0)
        for r in roles
        if len(role_vectors[r]) >= 5
    }

    if len(centroids) > 1:
        sep_matrix = pd.DataFrame(
            cdist(list(centroids.values()), list(centroids.values())),
            index=centroids.keys(),
            columns=centroids.keys()
        )

        print(sep_matrix.round(4))

        sep_matrix.to_excel(
            f"{TABLE_DIR}/{save_prefix}_centroid_separation.xlsx"
        )

    # ==================================================
    # 5. Statistical robustness analysis (Reviewer-proof)
    # ==================================================

    print("\n=== Statistical robustness tests ===")


    # --------------------------------------------------
    # Convert embeddings → PCA scalar projections
    # --------------------------------------------------

    print("\n--- PCA-based statistical separation ---")

    pca_scores = {r: [] for r in roles}

    # Project embeddings onto PCA space for scalar testing
    all_vectors = []

    for _, row in df.iterrows():

        for r in roles:
            emb = parse_embedding(row.get(embedding_dict[r]))

            if emb is not None:
                all_vectors.append(emb)

    if len(all_vectors) > 50:

        pca_global = PCA(n_components=2, random_state=42)

        X_all = pca_global.fit_transform(np.vstack(all_vectors))

        print(
            f"PCA variance explained: "
            f"PC1={pca_global.explained_variance_ratio_[0] * 100:.1f}% "
            f"PC2={pca_global.explained_variance_ratio_[1] * 100:.1f}%"
        )

        # Rebuild labeled dataset
        rows = []

        for _, row in df.iterrows():

            for r in roles:

                emb = parse_embedding(row.get(embedding_dict[r]))

                if emb is not None:
                    rows.append({
                        "role": r,
                        "pc1": pca_global.transform([emb])[0][0],
                        "pc2": pca_global.transform([emb])[0][1]
                    })

        stat_df = pd.DataFrame(rows)

        # --------------------------------------------------
        # Kruskal-Wallis on PC1 and PC2
        # --------------------------------------------------

        print("\n--- Kruskal-Wallis separation tests ---")

        for pc in ["pc1", "pc2"]:

            groups = [
                stat_df[stat_df["role"] == r][pc].values
                for r in roles
                if len(stat_df[stat_df["role"] == r]) > 5
            ]

            if len(groups) >= 2:

                H, p = kruskal(*groups)

                print(f"{pc.upper()} → H={H:.4f}, p={p:.6f}")

            else:
                print(f"{pc.upper()} → insufficient samples")

        # --------------------------------------------------
        # Effect size (η² approximation)
        # --------------------------------------------------

        print("\n--- Effect size estimation ---")

        grand_mean = stat_df[["pc1", "pc2"]].mean()

        ss_between = 0
        ss_total = 0

        for pc in ["pc1", "pc2"]:

            overall_mean = stat_df[pc].mean()

            ss_total = np.sum((stat_df[pc] - overall_mean) ** 2)

            for r in roles:

                grp = stat_df[stat_df["role"] == r][pc]

                if len(grp) < 5:
                    continue

                ss_between += len(grp) * (grp.mean() - overall_mean) ** 2

            eta_sq = ss_between / ss_total if ss_total > 0 else np.nan

            print(f"{pc} η² = {eta_sq:.4f}")

        # --------------------------------------------------
        # Save statistical PCA dataframe
        # --------------------------------------------------

        stat_df.to_excel(
            f"{TABLE_DIR}/{save_prefix}_pca_statistical_projection.xlsx",
            index=False
        )

        print(f"Saved → {TABLE_DIR}/{save_prefix}_pca_statistical_projection.xlsx")

    else:
        print("Not enough data for statistical PCA testing")


# ======================================================
# Run pipeline
# ======================================================

run_embedding_analysis(
    df,
    SINGLE_REQUEST_EMBEDDING_COLS,
    "Single Request",
    "single_request"
)

run_embedding_analysis(
    df,
    SELF_CONSISTENCY_EMBEDDING_COLS,
    "Self Consistency",
    "self_consistency"
)


# ======================================================
# PERSONA DRIFT SCORE (Single Request vs Self-Consistency)
# ======================================================

print("\n=== Persona Drift Score ===")

case_rows = []

# ------------------------------------------------------
# Case-level drift
# ------------------------------------------------------

for role in SINGLE_REQUEST_EMBEDDING_COLS.keys():

    single_col = SINGLE_REQUEST_EMBEDDING_COLS.get(role)
    self_col   = SELF_CONSISTENCY_EMBEDDING_COLS.get(role)

    if single_col is None or self_col is None:
        continue

    if single_col not in df.columns or self_col not in df.columns:
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

# ------------------------------------------------------
# Drift summary per role
# ------------------------------------------------------

if len(drift_df) > 0:

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

    print("\n--- Drift summary per role ---")
    print(role_drift_summary.to_string(index=False))

    # ------------------------------------------------------
    # Drift by tumour type
    # ------------------------------------------------------

    role_tumor_drift = (
        drift_df.groupby(["role", "tumor"])["cosine_drift"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )

    # ------------------------------------------------------
    # Centroid-level drift
    # ------------------------------------------------------

    print("\n--- Centroid-level drift ---")

    centroid_drift_rows = []

    for role, single_col in SINGLE_REQUEST_EMBEDDING_COLS.items():

        self_col = SELF_CONSISTENCY_EMBEDDING_COLS.get(role)

        if self_col is None:
            continue

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

    # ------------------------------------------------------
    # Save drift results
    # ------------------------------------------------------

    with pd.ExcelWriter(f"{TABLE_DIR}/persona_drift_full_analysis.xlsx") as w:
        drift_df.to_excel(w, sheet_name="case_level", index=False)
        role_drift_summary.to_excel(w, sheet_name="role_summary", index=False)
        role_tumor_drift.to_excel(w, sheet_name="role_x_tumor", index=False)
        centroid_drift_df.to_excel(w, sheet_name="centroid_drift", index=False)

    print(f"Persona drift saved → {TABLE_DIR}/persona_drift_full_analysis.xlsx")

else:
    print("No drift data available")


print("\n=== Embedding Analysis Complete ===")