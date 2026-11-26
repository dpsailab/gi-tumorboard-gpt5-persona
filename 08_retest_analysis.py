"""
08_retest_analysis.py
======================
Step 2 of 2. Test-retest analysis for the Replies to the two Matters Arising.

Inputs
    REPO_FILE   : published repository dataset (run 1, MDT reference, tumour type)
    RUN2_FILE   : data/LLM_MDT_persona_retest_run2.csv (second run, categorised)

Pre-specified questions (fixed before looking at the run-2 results)
    A. Concordance with the MDT, run 1 vs run 2, per role (and majority vote)
    B. Test-retest agreement per role: % same category, Cohen's kappa
    C. Intra-role vs inter-role agreement. Criterion: if kappa(intra) - kappa(inter)
       is below 0.10, role differences are of the same order as the model's own
       run-to-run variability.
    D. Equivalence between roles: paired difference in concordance with
       Newcombe (method 10) 90% CI, flagged against +/-5, 7.5, 10, 15 points,
       for run 1, run 2 and the per-case mean of both runs.
    E. Tumour-type subgroups: concordance run 1 / run 2 and test-retest agreement.

Output: retest_results.xlsx (one sheet per block) + printed summary.

Run:
    python 08_retest_analysis.py
"""

import ast
import os
import itertools
import math
from collections import Counter

import numpy as np
import pandas as pd
from scipy.stats import binomtest, norm
from sklearn.metrics import cohen_kappa_score

# --------------------------------------------------------------------------
# Settings
# --------------------------------------------------------------------------
REPO_FILE = "data/LLM_MDT_dataset_repository.csv"
RUN2_FILE = "data/LLM_MDT_persona_retest_run2.csv"
OUT_FILE = "output/retest_results.xlsx"

REF_COL = "tumorboard_treatment"      # list of acceptable MDT options
TUMOUR_COL = "tumour_type"
MARGINS = [5, 7.5, 10, 15]            # percentage points
N_BOOT = 2000
SEED = 20261119

ROLES = {
    "Surgeon": "F3_persona_surgeon",
    "Medical oncologist": "F4_persona_medical_oncologist",
    "Radiation oncologist": "F5_persona_radiation_oncologist",
}
PUBLISHED_CONCORDANCE = {"Surgeon": 84, "Medical oncologist": 84, "Radiation oncologist": 78}


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def parse_list(x):
    if isinstance(x, list):
        return x
    if isinstance(x, str):
        try:
            v = ast.literal_eval(x)
            return v if isinstance(v, list) else [v]
        except (ValueError, SyntaxError):
            return [x]
    return []


def concordant(treatment, options):
    return int(pd.notna(treatment) and treatment in options)


def wilson(x, n, z):
    p = x / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (c - h) / d, (c + h) / d


def newcombe_paired(y1, y2, conf=0.90):
    """Newcombe (1998) method 10 CI for p1 - p2 with paired binary data."""
    y1, y2 = np.asarray(y1), np.asarray(y2)
    a = int(((y1 == 1) & (y2 == 1)).sum())
    b = int(((y1 == 1) & (y2 == 0)).sum())
    c = int(((y1 == 0) & (y2 == 1)).sum())
    d = int(((y1 == 0) & (y2 == 0)).sum())
    n = a + b + c + d
    z = norm.ppf(1 - (1 - conf) / 2)
    p1, p2 = (a + b) / n, (a + c) / n
    diff = p1 - p2
    l1, u1 = wilson(a + b, n, z)
    l2, u2 = wilson(a + c, n, z)
    den = math.sqrt((a + b) * (c + d) * (a + c) * (b + d))
    if den == 0:
        phi = 0.0
    else:
        num = a * d - b * c
        phi = (num - n / 2) / den if num > 0 else num / den
    dl = math.sqrt(max((p1 - l1) ** 2 - 2 * phi * (p1 - l1) * (u2 - p2) + (u2 - p2) ** 2, 0))
    du = math.sqrt(max((u1 - p1) ** 2 - 2 * phi * (u1 - p1) * (p2 - l2) + (p2 - l2) ** 2, 0))
    return diff, diff - dl, diff + du, b, c


def paired_mean_ci(x1, x2, conf=0.90, n_boot=N_BOOT, seed=SEED):
    """Bootstrap CI (cases resampled) for the mean paired difference of per-case scores."""
    rng = np.random.default_rng(seed)
    d = np.asarray(x1, float) - np.asarray(x2, float)
    boots = [rng.choice(d, len(d), replace=True).mean() for _ in range(n_boot)]
    lo, hi = np.percentile(boots, [(1 - conf) / 2 * 100, (1 + conf) / 2 * 100])
    return d.mean(), lo, hi


def kappa(a, b):
    mask = pd.notna(a) & pd.notna(b)
    a, b = pd.Series(a)[mask], pd.Series(b)[mask]
    if len(a) == 0:
        return np.nan
    if a.nunique() == 1 and b.nunique() == 1:
        return 1.0 if a.iloc[0] == b.iloc[0] else 0.0
    return cohen_kappa_score(a, b)


def pct_same(a, b):
    mask = pd.notna(a) & pd.notna(b)
    return 100 * (pd.Series(a)[mask].values == pd.Series(b)[mask].values).mean()


def boot_kappa_diff(df, intra_pairs, inter_pairs, n_boot=N_BOOT, seed=SEED):
    """Bootstrap CI for mean kappa(intra) - mean kappa(inter), resampling cases."""
    rng = np.random.default_rng(seed)
    idx = np.arange(len(df))
    out = []
    for _ in range(n_boot):
        s = df.iloc[rng.choice(idx, len(idx), replace=True)]
        ki = np.nanmean([kappa(s[x], s[y]) for x, y in intra_pairs])
        ke = np.nanmean([kappa(s[x], s[y]) for x, y in inter_pairs])
        out.append(ki - ke)
    return np.percentile(out, [5, 95])


def majority(row, cols):
    vals = [row[c] for c in cols if pd.notna(row[c])]
    return Counter(vals).most_common(1)[0][0] if vals else np.nan


def margin_flags(lo, hi):
    return {f"within_±{m}": (lo > -m and hi < m) for m in MARGINS}


# --------------------------------------------------------------------------
# Load and merge
# --------------------------------------------------------------------------
repo = pd.read_csv(REPO_FILE)
repo["case_nr"] = repo["case_nr"].astype(str).str.strip()
repo[REF_COL] = repo[REF_COL].apply(parse_list)

run2 = pd.read_csv(RUN2_FILE)
run2["case_nr"] = run2["case_nr"].astype(str).str.strip()

keep = ["case_nr", TUMOUR_COL, REF_COL] + [f"{p}_treatment" for p in ROLES.values()]
df = repo[keep].merge(run2, on="case_nr", how="inner", validate="one_to_one")
print(f"Merged cases: {len(df)} (repo {len(repo)}, run 2 {len(run2)})")

# Category columns and concordance per run
R1 = {r: f"{p}_treatment" for r, p in ROLES.items()}
R2 = {r: f"{p}_run2_treatment" for r, p in ROLES.items()}
for r in ROLES:
    df[f"{r}_c1"] = [concordant(t, o) for t, o in zip(df[R1[r]], df[REF_COL])]
    df[f"{r}_c2"] = [concordant(t, o) for t, o in zip(df[R2[r]], df[REF_COL])]
    df[f"{r}_cmean"] = (df[f"{r}_c1"] + df[f"{r}_c2"]) / 2

df["MV_r1"] = df.apply(lambda row: majority(row, list(R1.values())), axis=1)
df["MV_r2"] = df.apply(lambda row: majority(row, list(R2.values())), axis=1)
df["MV_c1"] = [concordant(t, o) for t, o in zip(df["MV_r1"], df[REF_COL])]
df["MV_c2"] = [concordant(t, o) for t, o in zip(df["MV_r2"], df[REF_COL])]

# Sanity check against the published figures
print("\nSanity check, run-1 concordance (published 84 / 84 / 78):")
for r in ROLES:
    obs = 100 * df[f"{r}_c1"].mean()
    flag = "" if abs(obs - PUBLISHED_CONCORDANCE[r]) < 1 else "   [!] MISMATCH: check REPO_FILE / column names"
    print(f"  {r}: {obs:.0f}%  (published {PUBLISHED_CONCORDANCE[r]}%){flag}")

labels1 = set(pd.concat([df[c] for c in R1.values()]).dropna())
labels2 = set(pd.concat([df[c] for c in R2.values()]).dropna())
if labels2 - labels1:
    print(f"[!] Run-2 labels not used in run 1: {sorted(labels2 - labels1)}  -> check spelling")

# --------------------------------------------------------------------------
# A. Concordance run 1 vs run 2
# --------------------------------------------------------------------------
rows = []
for r in list(ROLES) + ["Majority vote"]:
    c1 = df[f"{r}_c1"] if r in ROLES else df["MV_c1"]
    c2 = df[f"{r}_c2"] if r in ROLES else df["MV_c2"]
    diff, lo, hi, b, c = newcombe_paired(c1, c2)
    p_exact = binomtest(min(b, c), b + c, 0.5).pvalue if b + c > 0 else 1.0
    rows.append({
        "role": r,
        "concordance_run1_%": round(100 * c1.mean(), 1),
        "concordance_run2_%": round(100 * c2.mean(), 1),
        "diff_run1-run2_pp": round(100 * diff, 1),
        "90%CI_low": round(100 * lo, 1), "90%CI_high": round(100 * hi, 1),
        "discordant_run1_only": b, "discordant_run2_only": c,
        "McNemar_exact_p": round(p_exact, 3),
    })
A = pd.DataFrame(rows)

# --------------------------------------------------------------------------
# B. Test-retest agreement per role
# --------------------------------------------------------------------------
B = pd.DataFrame([{
    "role": r,
    "same_category_%": round(pct_same(df[R1[r]], df[R2[r]]), 1),
    "kappa_run1_vs_run2": round(kappa(df[R1[r]], df[R2[r]]), 3),
    "cases_changed": int((df[R1[r]] != df[R2[r]]).sum()),
    "concordance_flips": int((df[f"{r}_c1"] != df[f"{r}_c2"]).sum()),
} for r in ROLES])

# --------------------------------------------------------------------------
# C. Intra-role vs inter-role agreement
# --------------------------------------------------------------------------
crow = []
for x, y in itertools.combinations(ROLES, 2):
    for run, R in (("run1", R1), ("run2", R2)):
        crow.append({"pair": f"{x} vs {y}", "run": run,
                     "same_category_%": round(pct_same(df[R[x]], df[R[y]]), 1),
                     "kappa": round(kappa(df[R[x]], df[R[y]]), 3)})
C_inter = pd.DataFrame(crow)

intra_pairs = [(R1[r], R2[r]) for r in ROLES]
inter_pairs = [(R[x], R[y]) for R in (R1, R2) for x, y in itertools.combinations(ROLES, 2)]
k_intra = np.nanmean([kappa(df[a], df[b]) for a, b in intra_pairs])
k_inter = np.nanmean([kappa(df[a], df[b]) for a, b in inter_pairs])
lo, hi = boot_kappa_diff(df, intra_pairs, inter_pairs)
C_summary = pd.DataFrame([{
    "mean_kappa_intra_role": round(k_intra, 3),
    "mean_kappa_inter_role": round(k_inter, 3),
    "difference": round(k_intra - k_inter, 3),
    "90%CI_low": round(lo, 3), "90%CI_high": round(hi, 3),
    "criterion_diff_<_0.10": bool(k_intra - k_inter < 0.10),
}])

# Unanimity among the three roles
U = pd.DataFrame([{
    "run": run,
    "unanimous_cases": int((df[list(R.values())].nunique(axis=1) == 1).sum()),
} for run, R in (("run1", R1), ("run2", R2))])

# --------------------------------------------------------------------------
# D. Equivalence between roles
# --------------------------------------------------------------------------
drow = []
for x, y in itertools.combinations(ROLES, 2):
    for run in ("c1", "c2"):
        diff, lo, hi, b, c = newcombe_paired(df[f"{x}_{run}"], df[f"{y}_{run}"])
        drow.append({"pair": f"{x} vs {y}", "data": "run1" if run == "c1" else "run2",
                     "method": "Newcombe 10",
                     "diff_pp": round(100 * diff, 1),
                     "90%CI_low": round(100 * lo, 1), "90%CI_high": round(100 * hi, 1),
                     **margin_flags(100 * lo, 100 * hi)})
    diff, lo, hi = paired_mean_ci(df[f"{x}_cmean"], df[f"{y}_cmean"])
    drow.append({"pair": f"{x} vs {y}", "data": "mean of both runs",
                 "method": "bootstrap (cases)",
                 "diff_pp": round(100 * diff, 1),
                 "90%CI_low": round(100 * lo, 1), "90%CI_high": round(100 * hi, 1),
                 **margin_flags(100 * lo, 100 * hi)})
D = pd.DataFrame(drow)

# --------------------------------------------------------------------------
# E. Tumour-type subgroups
# --------------------------------------------------------------------------
erow = []
for t, sub in df.groupby(TUMOUR_COL):
    for r in ROLES:
        erow.append({
            "tumour_type": t, "role": r, "n": len(sub),
            "concordance_run1_%": round(100 * sub[f"{r}_c1"].mean(), 1),
            "concordance_run2_%": round(100 * sub[f"{r}_c2"].mean(), 1),
            "same_category_%": round(pct_same(sub[R1[r]], sub[R2[r]]), 1),
        })
E = pd.DataFrame(erow)
E_tumour = (E.groupby("tumour_type")
              .agg(n=("n", "first"),
                   mean_same_category_pct=("same_category_%", "mean"),
                   mean_concordance_run1=("concordance_run1_%", "mean"),
                   mean_concordance_run2=("concordance_run2_%", "mean"))
              .round(1).sort_values("mean_same_category_pct"))

# --------------------------------------------------------------------------
# Save and print
# --------------------------------------------------------------------------
os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)
with pd.ExcelWriter(OUT_FILE) as w:
    A.to_excel(w, sheet_name="A_concordance_runs", index=False)
    B.to_excel(w, sheet_name="B_test_retest", index=False)
    C_inter.to_excel(w, sheet_name="C_inter_role", index=False)
    C_summary.to_excel(w, sheet_name="C_intra_vs_inter", index=False)
    U.to_excel(w, sheet_name="C_unanimity", index=False)
    D.to_excel(w, sheet_name="D_equivalence", index=False)
    E.to_excel(w, sheet_name="E_tumour_by_role", index=False)
    E_tumour.to_excel(w, sheet_name="E_tumour_summary")
    df.drop(columns=[REF_COL]).to_excel(w, sheet_name="case_level", index=False)

pd.set_option("display.width", 200)
for title, tab in [("A. Concordance with MDT, run 1 vs run 2", A),
                   ("B. Test-retest agreement per role", B),
                   ("C. Inter-role agreement", C_inter),
                   ("C. Intra- vs inter-role kappa (criterion < 0.10)", C_summary),
                   ("C. Unanimity of the three roles", U),
                   ("D. Equivalence between roles (90% CI, pp)", D),
                   ("E. Tumour types, least to most stable", E_tumour)]:
    print(f"\n=== {title} ===")
    print(tab.to_string(index=title.startswith("E.")))

print(f"\nSaved -> {OUT_FILE}")
