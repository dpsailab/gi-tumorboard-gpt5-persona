"""
06_caselevel_crossdisciplinary.py
=================================
Case-level re-analysis of the association between cross-disciplinary content
in multi-expert reasoning paths and consensus concordance.

Motivation
----------
In Table 2 the content categories are classified per reasoning PATH
(n = 300 = 100 cases x 3 roles), but the concordance outcome is the single
CASE-level consensus (F2_..._concordance), identical across a case's three
paths. Pooling to 300 therefore treats non-independent observations as
independent, and the marginal 246/300 (82%) is just the 82/100 case accuracy
counted three times. The stratified 93.9% vs 80.5% contrast is a real
per-path signal but is inflated in apparent N and clustered by case.

This script re-runs the analysis at the CASE level (100 independent consensus
outcomes), grouping each case by how many of its three reasoning paths
contained cross-disciplinary content (0/1/2/3), which restores independence.

Run
---
From the repository root (same directory as config.py):

    python 06_caselevel_crossdisciplinary.py
"""

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact, norm

from config import (
    DATA_FILE,
    MULTI_EXPERT_BOUNDARY_COLS,
)

# Same single consensus concordance column the original combo analysis maps to
# all three roles (see 03_persona_stability_analysis.py, Section 11).
MULTI_EXPERT_CORRECT_COL = (
    "F2_multi_expert_consensus_tumorboard_treatment_concordance"
)


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------
df = pd.read_csv(DATA_FILE)

# The specialist personas that produce the three reasoning paths per case.
roles = [r for r in MULTI_EXPERT_BOUNDARY_COLS if r != "Simulated Tumorboard"]
boundary_cols = [MULTI_EXPERT_BOUNDARY_COLS[r] for r in roles]

missing = [c for c in boundary_cols + [MULTI_EXPERT_CORRECT_COL]
           if c not in df.columns]
if missing:
    raise SystemExit(f"Missing expected columns: {missing}")

print(f"Roles used ({len(roles)} paths/case): {roles}\n")


# ---------------------------------------------------------------------------
# Case-level construction
# ---------------------------------------------------------------------------
# Per case: how many of the 3 paths contain cross-disciplinary content
# (boundary flag == 1), and the single consensus concordance outcome.
bnd = df[boundary_cols].apply(pd.to_numeric, errors="coerce")
n_crossdisc = (bnd == 1).sum(axis=1)  # 0..3 per case

concordant = pd.to_numeric(df[MULTI_EXPERT_CORRECT_COL], errors="coerce")

case = pd.DataFrame(
    {"n_crossdisc_paths": n_crossdisc, "concordant": concordant}
).dropna(subset=["concordant"])
case["concordant"] = (case["concordant"] == 1).astype(int)

N = len(case)


# ---------------------------------------------------------------------------
# Sanity checks against the published Table 2
# ---------------------------------------------------------------------------
print("=== Sanity checks (should match the paper) ===")
print(f"Cases analysed:                 {N}")
print(f"Total cross-disciplinary PATHS: {int((bnd == 1).sum().sum())}"
      f"   (paper: 33)")
print(f"Overall consensus accuracy:     {case['concordant'].mean() * 100:.1f}%"
      f"   (paper: 82.0% = 246/300)\n")


# ---------------------------------------------------------------------------
# Case-level distribution by number of cross-disciplinary paths (0/1/2/3)
# ---------------------------------------------------------------------------
print("=== Consensus concordance by number of cross-disciplinary paths "
      "(case level) ===")
grp = (case.groupby("n_crossdisc_paths")["concordant"]
           .agg(n="size", concordant="sum"))
grp["rate_%"] = (grp["concordant"] / grp["n"] * 100).round(1)
print(grp.to_string())
n_any = int((case["n_crossdisc_paths"] >= 1).sum())
print(f"\nCases with >=1 cross-disciplinary path: {n_any}\n")


# ---------------------------------------------------------------------------
# Binary contrast: >=1 vs 0 cross-disciplinary paths (independent cases)
# ---------------------------------------------------------------------------
case["any_crossdisc"] = (case["n_crossdisc_paths"] >= 1).astype(int)
ct = (pd.crosstab(case["any_crossdisc"], case["concordant"])
        .reindex(index=[0, 1], columns=[0, 1], fill_value=0))

a = ct.loc[1, 1]  # >=1 & concordant
b = ct.loc[1, 0]  # >=1 & discordant
c = ct.loc[0, 1]  # 0   & concordant
d = ct.loc[0, 0]  # 0   & discordant
rate_any = a / (a + b) * 100 if (a + b) else float("nan")
rate_none = c / (c + d) * 100 if (c + d) else float("nan")
or_, p_fisher = fisher_exact([[a, b], [c, d]])

print("=== Binary contrast (case level, independent) ===")
print(f"Concordance  >=1 cross-disc: {rate_any:.1f}%  ({a}/{a + b})")
print(f"Concordance   0 cross-disc: {rate_none:.1f}%  ({c}/{c + d})")
print(f"Fisher exact: OR = {or_:.2f}, p = {p_fisher:.4f}\n")


# ---------------------------------------------------------------------------
# Trend across 0/1/2/3 (Cochran-Armitage)
# ---------------------------------------------------------------------------
def cochran_armitage(counts_n, counts_r, scores):
    n = np.asarray(counts_n, float)
    r = np.asarray(counts_r, float)
    t = np.asarray(scores, float)
    Ntot, Rtot = n.sum(), r.sum()
    p = Rtot / Ntot
    tbar = (n * t).sum() / Ntot
    num = (r * (t - tbar)).sum()
    den = np.sqrt(p * (1 - p) * (n * (t - tbar) ** 2).sum())
    if den == 0:
        return float("nan"), float("nan")
    z = num / den
    return z, 2 * (1 - norm.cdf(abs(z)))

full = grp.reindex([0, 1, 2, 3], fill_value=0)
z, p_trend = cochran_armitage(full["n"].values, full["concordant"].values,
                              scores=[0, 1, 2, 3])
print("=== Cochran-Armitage trend (scores 0/1/2/3) ===")
print(f"Z = {z:.3f}, p = {p_trend:.4f}")
