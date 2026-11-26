"""
07_role_boundary_structure.py
=============================
STRUCTURE axis (form of the response), not reasoning.

boundary_violation = the output introduces out-of-domain specialist
INFORMATION (a specific regimen, fractionation, operative technique) that a
real single specialist would defer to a colleague. It is NOT the decision.
A surgeon who says "not surgery but chemotherapy; refer to oncology for the
regimen" names an out-of-domain DECISION but introduces no oncology detail
-> boundary = 0. A surgeon who spells out the chemo regimen -> boundary = 1.

Key question: when the role's own recommended category is OUTSIDE its
competence, does it still spill the full cross-disciplinary information
(behaving like a tumour board) or does it defer (behaving like a siloed
specialist)? We stratify the information outcome (boundary) by whether the
role's decision is in-home or out-of-home. The decision is only a stratifying
variable here; it is not fused into the boundary measure.

    boundary_% | in-home   -> when in its lane, does it spill? (baseline)
    boundary_% | out-home  -> when out of its lane, does it still spill? (the point)

The DECISION axis (MDT concordance) is reported separately below.

Run from the repository root (same directory as config.py):

    python 07_role_boundary_structure.py
"""

import numpy as np
import pandas as pd

from config import (
    DATA_FILE,
    METHOD_TREATMENT_COLS,
    SPECIALIST_DOMAIN_COLS,
    SPECIALIST_BOUNDARY_COLS,
    SPECIALIST_PERSONA_CONCORDANCE_COLS,
)

# Home category per role, matched case-insensitively by substring.
# Verify the resolved mapping printed below against the real labels.
HOME_PATTERNS = {
    "Surgeon": "surg",
    "Oncologist": "systemic",
    "Radio-Oncologist": "local",   # real label is "Local Therapy"
}
SPECIALIST_ROLES = ["Surgeon", "Oncologist", "Radio-Oncologist"]


df = pd.read_csv(DATA_FILE)


def _num(s):
    return pd.to_numeric(s, errors="coerce")


print("=== Resolved home categories (verify these) ===")
for role in SPECIALIST_ROLES:
    tcol = METHOD_TREATMENT_COLS[role]
    hits = [v for v in pd.Series(df[tcol]).dropna().unique()
            if HOME_PATTERNS[role].lower() in str(v).lower()]
    print(f"{role:16s} pattern '{HOME_PATTERNS[role]}' -> {hits}")
print()


struct_rows, dec_rows = [], []
for role in SPECIALIST_ROLES:
    tcol = METHOD_TREATMENT_COLS[role]
    dcol = SPECIALIST_DOMAIN_COLS[role]
    bcol = SPECIALIST_BOUNDARY_COLS[role]
    ccol = SPECIALIST_PERSONA_CONCORDANCE_COLS[role]
    for c in [tcol, dcol, bcol, ccol]:
        if c not in df.columns:
            raise SystemExit(f"Missing column: {c}")

    treat = df[tcol].astype(str)
    dom = _num(df[dcol])
    bnd = _num(df[bcol])
    con = _num(df[ccol])

    is_home = treat.str.lower().apply(lambda x: HOME_PATTERNS[role].lower() in x)
    n = len(df)
    n_home = int(is_home.sum())
    n_out = int((~is_home).sum())

    struct_rows.append({
        "role": role,
        "n": n,
        "domain_%": round(dom.eq(1).mean() * 100, 1),
        "boundary_%": round(bnd.eq(1).mean() * 100, 1),
        "home_reco_%": round(n_home / n * 100, 1),
        "bnd|in-home_%": round(bnd[is_home].eq(1).mean() * 100, 1) if n_home else np.nan,
        "n_in": n_home,
        "bnd|out-home_%": round(bnd[~is_home].eq(1).mean() * 100, 1) if n_out else np.nan,
        "n_out": n_out,
    })
    dec_rows.append({"role": role, "MDT_concord_%": round(con.eq(1).mean() * 100, 1)})

struct = pd.DataFrame(struct_rows).set_index("role")
dec = pd.DataFrame(dec_rows).set_index("role")

print("=== STRUCTURE axis: out-of-domain INFORMATION, by whether the role's "
      "own decision is in/out of its competence ===")
print(struct.to_string())
print("\nRead: bnd|in-home low (in its lane it need not spill) vs bnd|out-home")
print("high (out of its lane it still produces the full cross-disciplinary")
print("information instead of deferring) = tumour-board response structure,")
print("not a siloed specialist. This is about the form of the output, not reasoning.")

# in-home reliability guard
for role in SPECIALIST_ROLES:
    n_in = int(struct.loc[role, "n_in"])
    if n_in < 10:
        print(f"\n[!] {role}: only {n_in} in-home cases; its bnd|in-home is "
              f"unreliable. Its bnd|out-home (n_out large) is the usable number.")

print("\n=== DECISION axis (separate; do NOT fuse with structure) ===")
print(dec.to_string())
print("MDT_concord_% should reproduce the published per-role concordance "
      "(surgeon 84 / medical 84 / radiation 78).")