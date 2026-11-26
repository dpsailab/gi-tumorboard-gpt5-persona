"""
09_retest_extra.py
=======================
Follow-up checks on the test-retest data (same inputs as 08_retest_analysis.py).

1. Which cases changed category between runs, per role, and how much the
   three roles' changed cases overlap (same unstable cases, or different ones?)
2. Concordance by treatment domain ("Bereich") of the MDT decision, per role
   and per run, plus majority vote. Does each persona do best in its own domain?
3. Concordance by tumour type, per role and per run, plus majority vote.

Output: retest_extra.xlsx + printed summary.

Run:
    python 09_retest_extra.py
"""

import ast
import os
from collections import Counter

import numpy as np
import pandas as pd

REPO_FILE = "data/LLM_MDT_dataset_repository.csv"
RUN2_FILE = "data/LLM_MDT_persona_retest_run2.csv"
OUT_FILE = "output/retest_extra.xlsx"
REF_COL = "tumorboard_treatment"
TUMOUR_COL = "tumour_type"

ROLES = {
    "Surgeon": "F3_persona_surgeon",
    "Medical oncologist": "F4_persona_medical_oncologist",
    "Radiation oncologist": "F5_persona_radiation_oncologist",
}
HOME = {"Surgeon": "Surgery", "Medical oncologist": "Systemic Therapy",
        "Radiation oncologist": "Local Therapy"}


def parse_list(x):
    if isinstance(x, list):
        return x
    try:
        v = ast.literal_eval(x)
        return v if isinstance(v, list) else [v]
    except (ValueError, SyntaxError):
        return [x]


def majority(vals):
    vals = [v for v in vals if pd.notna(v)]
    return Counter(vals).most_common(1)[0][0] if vals else np.nan


repo = pd.read_csv(REPO_FILE)
repo["case_nr"] = repo["case_nr"].astype(str).str.strip()
repo[REF_COL] = repo[REF_COL].apply(parse_list)
run2 = pd.read_csv(RUN2_FILE)
run2["case_nr"] = run2["case_nr"].astype(str).str.strip()

keep = ["case_nr", TUMOUR_COL, REF_COL] + [f"{p}_treatment" for p in ROLES.values()]
df = repo[keep].merge(run2, on="case_nr", how="inner", validate="one_to_one")
df["mdt_domain"] = df[REF_COL].apply(lambda l: l[0] if l else np.nan)   # primary MDT option
df["mdt_n_options"] = df[REF_COL].apply(len)

R1 = {r: f"{p}_treatment" for r, p in ROLES.items()}
R2 = {r: f"{p}_run2_treatment" for r, p in ROLES.items()}
for r in ROLES:
    df[f"{r}_c1"] = [int(t in o) for t, o in zip(df[R1[r]], df[REF_COL])]
    df[f"{r}_c2"] = [int(t in o) for t, o in zip(df[R2[r]], df[REF_COL])]
    df[f"{r}_changed"] = (df[R1[r]] != df[R2[r]]).astype(int)
df["MV_c1"] = [int(majority([row[c] for c in R1.values()]) in row[REF_COL]) for _, row in df.iterrows()]
df["MV_c2"] = [int(majority([row[c] for c in R2.values()]) in row[REF_COL]) for _, row in df.iterrows()]

# ---------------------------------------------------------------- 1. overlap
changed = {r: set(df.loc[df[f"{r}_changed"] == 1, "case_nr"]) for r in ROLES}
s, m, ro = changed["Surgeon"], changed["Medical oncologist"], changed["Radiation oncologist"]
overlap = pd.DataFrame([
    {"set": "changed, surgeon", "n": len(s)},
    {"set": "changed, medical oncologist", "n": len(m)},
    {"set": "changed, radiation oncologist", "n": len(ro)},
    {"set": "surgeon & medical oncologist", "n": len(s & m)},
    {"set": "surgeon & radiation oncologist", "n": len(s & ro)},
    {"set": "medical & radiation oncologist", "n": len(m & ro)},
    {"set": "all three", "n": len(s & m & ro)},
    {"set": "at least one role", "n": len(s | m | ro)},
])
changed_cases = df.loc[df[[f"{r}_changed" for r in ROLES]].sum(axis=1) > 0,
                       ["case_nr", TUMOUR_COL, "mdt_domain"]
                       + [c for r in ROLES for c in (R1[r], R2[r])]]


# ---------------------------------------------------------------- 2. by domain
def by_group(col):
    rows = []
    for g, sub in df.groupby(col):
        row = {col: g, "n": len(sub)}
        for r in ROLES:
            row[f"{r} run1 %"] = round(100 * sub[f"{r}_c1"].mean(), 1)
            row[f"{r} run2 %"] = round(100 * sub[f"{r}_c2"].mean(), 1)
        row["Majority vote run1 %"] = round(100 * sub["MV_c1"].mean(), 1)
        row["Majority vote run2 %"] = round(100 * sub["MV_c2"].mean(), 1)
        rows.append(row)
    return pd.DataFrame(rows).sort_values("n", ascending=False)


domain = by_group("mdt_domain")
tumour = by_group(TUMOUR_COL)

home = pd.DataFrame([{
    "role": r,
    "home_domain": HOME[r],
    "n_home_cases": int((df["mdt_domain"] == HOME[r]).sum()),
    "concordance_home_run1 %": round(100 * df.loc[df["mdt_domain"] == HOME[r], f"{r}_c1"].mean(), 1),
    "concordance_home_run2 %": round(100 * df.loc[df["mdt_domain"] == HOME[r], f"{r}_c2"].mean(), 1),
    "concordance_other_run1 %": round(100 * df.loc[df["mdt_domain"] != HOME[r], f"{r}_c1"].mean(), 1),
    "concordance_other_run2 %": round(100 * df.loc[df["mdt_domain"] != HOME[r], f"{r}_c2"].mean(), 1),
} for r in ROLES])

# ---------------------------------------------------------------- 4. profiles
# How often each role recommends each category, vs how often the MDT chose it
cats = sorted(set(pd.concat([df[c] for c in list(R1.values()) + list(R2.values())]).dropna())
              | set(df["mdt_domain"].dropna()))
prof_rows = []
for cat in cats:
    row = {"category": cat,
           "MDT (primary option) %": round(100 * (df["mdt_domain"] == cat).mean(), 1)}
    for r in ROLES:
        row[f"{r} run1 %"] = round(100 * (df[R1[r]] == cat).mean(), 1)
        row[f"{r} run2 %"] = round(100 * (df[R2[r]] == cat).mean(), 1)
    prof_rows.append(row)
profile = pd.DataFrame(prof_rows).sort_values("MDT (primary option) %", ascending=False)

# What each role recommends when it is NOT concordant (pooled over both runs)
err_rows = []
for r in ROLES:
    for run, R, c in (("run1", R1, "_c1"), ("run2", R2, "_c2")):
        wrong = df[df[f"{r}{c}"] == 0]
        for (mdt, rec), n in wrong.groupby(["mdt_domain", R[r]]).size().items():
            err_rows.append({"role": r, "run": run, "MDT primary": mdt,
                             "role recommended": rec, "n": n})
errors = (pd.DataFrame(err_rows)
          .groupby(["role", "MDT primary", "role recommended"], as_index=False)["n"].sum()
          .sort_values(["role", "n"], ascending=[True, False]))

# ---------------------------------------------------------------- save / print
os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)
with pd.ExcelWriter(OUT_FILE) as w:
    overlap.to_excel(w, sheet_name="1_changed_overlap", index=False)
    changed_cases.to_excel(w, sheet_name="1_changed_cases", index=False)
    domain.to_excel(w, sheet_name="2_by_MDT_domain", index=False)
    home.to_excel(w, sheet_name="2_home_vs_other", index=False)
    tumour.to_excel(w, sheet_name="3_by_tumour", index=False)
    profile.to_excel(w, sheet_name="4_recommendation_profile", index=False)
    errors.to_excel(w, sheet_name="4_errors_by_role", index=False)

pd.set_option("display.width", 250)
for title, tab in [("1. Changed cases and overlap", overlap),
                   ("2. Concordance by MDT treatment domain (primary option)", domain),
                   ("2. Own domain vs other domains", home),
                   ("3. Concordance by tumour type", tumour),
                   ("4. Recommendation profile: how often each category is chosen", profile),
                   ("4. Discordant recommendations (both runs pooled)", errors)]:
    print(f"\n=== {title} ===")
    print(tab.to_string(index=False))
print("\nNote: domain = first option listed by the MDT; cases with several "
      f"options: {int((df['mdt_n_options'] > 1).sum())}")
print(f"Saved -> {OUT_FILE}")
