"""
00_demographics.py
==================
Patient demographics and cohort characteristics.

This script produces the descriptive statistics reported in Table 1 of the
manuscript.  It reads directly from the main study dataset (Excel format) and
maps German column names / values to English labels via the shared config.

Analyses
--------
- Age: median and IQR, overall and by tumour type
- Sex: counts and percentages, overall and by tumour type
- Consultation type (First Presentation vs FUP): overall and by tumour type
- Tumour type distribution: counts and percentages
- Treatment recommendation distribution (Tumour Board): overall and by tumour type

Outputs (saved to ``role/tables/``)
-------------------------------------
- table_age_overall.csv
- table_age_by_tumour.csv
- table_sex_overall.csv
- table_sex_by_tumour.csv
- table_consultation_overall.csv
- table_consultation_by_tumour.csv
- table_tumour_distribution.csv
- table_treatment_by_tumour.csv
- Table1_demographics.xlsx  ← combined, publication-ready

Column mapping (dataset → analysis)
-------------------------------------
The anonymised dataset uses the following columns:
  - ``Anmeldediagnose``        : tumour type (German)
  - ``EV/WV``                  : consultation type (EV = first, WV = follow-up)
  - ``Konferenzbeschluss_treatment`` : tumour board treatment decision
  - ``age``                    : patient age in years (if present)
  - ``gender`` / ``sex``       : patient sex (if present)

If age or sex columns are absent the corresponding tables are skipped with
an informative message rather than raising an error.
"""

import os

import numpy as np
import pandas as pd

from config import (
    DATA_FILE,
    OUTPUT_DIR_ROLE,
    VALUE_RENAME,
)

# ---------------------------------------------------------------------------
# Output directory
# ---------------------------------------------------------------------------
TABLE_DIR = os.path.join(OUTPUT_DIR_ROLE, "tables")
os.makedirs(TABLE_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Column name aliases
# (adjust here if your dataset uses different names)
# ---------------------------------------------------------------------------
COL_TUMOUR       = "Anmeldediagnose"
COL_CONSULTATION = "EV/WV"
COL_TREATMENT    = "Konferenzbeschluss_treatment"
COL_AGE          = "age"           # set to None if not present
COL_SEX          = "gender"        # set to None if not present; also tries "sex"


# ===========================================================================
# Load data
# ===========================================================================

def load_data(path: str) -> pd.DataFrame:
    """Load the main study dataset from Excel."""
    df = pd.read_excel(path)

    # Normalise tumour type and consultation labels to English
    df[COL_TUMOUR]       = df[COL_TUMOUR].replace(VALUE_RENAME)
    df[COL_CONSULTATION] = df[COL_CONSULTATION].replace(VALUE_RENAME)
    if COL_TREATMENT in df.columns:
        df[COL_TREATMENT] = df[COL_TREATMENT].replace(VALUE_RENAME)

    return df


# ===========================================================================
# Utility formatters
# ===========================================================================

def _n_pct(count: int, total: int) -> str:
    pct = count / total * 100 if total else 0.0
    return f"{count} ({pct:.1f}%)"


def _median_iqr(series: pd.Series) -> str:
    med = series.median()
    q1  = series.quantile(0.25)
    q3  = series.quantile(0.75)
    return f"{med:.1f} ({q1:.0f}–{q3:.0f})"


# ===========================================================================
# Age statistics
# ===========================================================================

def age_statistics(df: pd.DataFrame,
                   group_col: str = None) -> pd.DataFrame:
    """
    Compute median age and IQR, overall or stratified by *group_col*.

    Returns a DataFrame with columns: median, q1, q3, IQR, median_IQR.
    """
    def _agg(series: pd.Series) -> dict:
        return {
            "N":          int(series.count()),
            "median":     series.median(),
            "q1":         series.quantile(0.25),
            "q3":         series.quantile(0.75),
            "IQR":        series.quantile(0.75) - series.quantile(0.25),
            "median_IQR": _median_iqr(series),
        }

    if group_col:
        rows = {grp: _agg(sub[COL_AGE]) for grp, sub in df.groupby(group_col)}
        result = pd.DataFrame(rows).T
    else:
        result = pd.DataFrame([_agg(df[COL_AGE])], index=["Overall"])

    return result


# ===========================================================================
# Sex / gender distribution
# ===========================================================================

def sex_distribution(df: pd.DataFrame,
                     sex_col: str,
                     group_col: str = None) -> pd.DataFrame:
    """
    Counts and percentages for each sex category.

    Parameters
    ----------
    df :
        Main DataFrame.
    sex_col :
        Column containing sex/gender values.
    group_col :
        Optional stratification column.

    Returns
    -------
    pandas.DataFrame
        Columns: N, percent, n_pct
    """
    def _counts(subset: pd.DataFrame) -> pd.DataFrame:
        total  = len(subset)
        counts = subset[sex_col].value_counts()
        perc   = subset[sex_col].value_counts(normalize=True) * 100
        out    = pd.concat([counts, perc.round(1)], axis=1)
        out.columns = ["N", "percent"]
        out["n_pct"] = out.apply(lambda r: _n_pct(int(r["N"]), total), axis=1)
        return out

    if group_col:
        frames = {grp: _counts(sub) for grp, sub in df.groupby(group_col)}
        return pd.concat(frames, axis=0)
    return _counts(df)


# ===========================================================================
# Consultation type (First Presentation vs FUP)
# ===========================================================================

def consultation_distribution(df: pd.DataFrame,
                               group_col: str = None) -> pd.DataFrame:
    """
    Counts and percentages for First Presentation vs FUP consultations.

    Parameters
    ----------
    df :
        Main DataFrame (consultation column already mapped to English).
    group_col :
        Optional stratification column.

    Returns
    -------
    pandas.DataFrame
    """
    def _counts(subset: pd.DataFrame) -> pd.DataFrame:
        total  = len(subset)
        counts = subset[COL_CONSULTATION].value_counts()
        perc   = subset[COL_CONSULTATION].value_counts(normalize=True) * 100
        out    = pd.concat([counts, perc.round(1)], axis=1)
        out.columns = ["N", "percent"]
        out["n_pct"] = out.apply(lambda r: _n_pct(int(r["N"]), total), axis=1)
        return out

    if group_col:
        frames = {grp: _counts(sub) for grp, sub in df.groupby(group_col)}
        return pd.concat(frames, axis=0)
    return _counts(df)


# ===========================================================================
# Tumour type distribution
# ===========================================================================

def tumour_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """Overall counts and percentages per tumour type."""
    total  = len(df)
    counts = df[COL_TUMOUR].value_counts()
    perc   = df[COL_TUMOUR].value_counts(normalize=True) * 100
    out    = pd.concat([counts, perc.round(1)], axis=1)
    out.columns = ["N", "percent"]
    out["n_pct"] = out.apply(lambda r: _n_pct(int(r["N"]), total), axis=1)
    return out


# ===========================================================================
# Treatment recommendation distribution
# ===========================================================================

def treatment_distribution(df: pd.DataFrame,
                            group_col: str = None) -> pd.DataFrame:
    """
    Counts and percentages of each tumour board treatment decision.

    Parameters
    ----------
    df :
        Main DataFrame.
    group_col :
        Optional stratification column (e.g. tumour type).

    Returns
    -------
    pandas.DataFrame
    """
    if COL_TREATMENT not in df.columns:
        print(f"  Column '{COL_TREATMENT}' not found — skipping treatment distribution.")
        return pd.DataFrame()

    def _counts(subset: pd.DataFrame) -> pd.DataFrame:
        total  = len(subset)
        counts = subset[COL_TREATMENT].value_counts()
        perc   = subset[COL_TREATMENT].value_counts(normalize=True) * 100
        out    = pd.concat([counts, perc.round(1)], axis=1)
        out.columns = ["N", "percent"]
        out["n_pct"] = out.apply(lambda r: _n_pct(int(r["N"]), total), axis=1)
        return out

    if group_col:
        frames = {grp: _counts(sub) for grp, sub in df.groupby(group_col)}
        return pd.concat(frames, axis=0)
    return _counts(df)


# ===========================================================================
# Combined Table 1 (publication-ready)
# ===========================================================================

def build_table1(df: pd.DataFrame, sex_col: str = None) -> pd.DataFrame:
    """
    Assemble a single publication-ready Table 1.

    Rows: demographic / clinical variable
    Columns: Overall + one column per tumour type

    Parameters
    ----------
    df :
        Main DataFrame.
    sex_col :
        Column for sex/gender, or None if not available.

    Returns
    -------
    pandas.DataFrame
    """
    tumour_types = sorted(df[COL_TUMOUR].unique())
    columns      = ["Overall"] + list(tumour_types)
    rows         = {}

    # ---- N ----
    rows["N"] = {
        "Overall": str(len(df)),
        **{t: str(len(df[df[COL_TUMOUR] == t])) for t in tumour_types},
    }

    # ---- Age ----
    if COL_AGE in df.columns:
        rows["Age — median (IQR)"] = {
            "Overall": _median_iqr(df[COL_AGE]),
            **{t: _median_iqr(df.loc[df[COL_TUMOUR] == t, COL_AGE])
               for t in tumour_types},
        }

    # ---- Sex ----
    if sex_col and sex_col in df.columns:
        for sex_val in sorted(df[sex_col].dropna().unique()):
            n_overall = int((df[sex_col] == sex_val).sum())
            rows[f"Sex — {sex_val}"] = {
                "Overall": _n_pct(n_overall, len(df)),
                **{t: _n_pct(
                        int((df.loc[df[COL_TUMOUR] == t, sex_col] == sex_val).sum()),
                        len(df[df[COL_TUMOUR] == t]),
                   )
                   for t in tumour_types},
            }

    # ---- Consultation type ----
    for cons_val in sorted(df[COL_CONSULTATION].dropna().unique()):
        n_overall = int((df[COL_CONSULTATION] == cons_val).sum())
        rows[f"Consultation — {cons_val}"] = {
            "Overall": _n_pct(n_overall, len(df)),
            **{t: _n_pct(
                    int((df.loc[df[COL_TUMOUR] == t, COL_CONSULTATION] == cons_val).sum()),
                    len(df[df[COL_TUMOUR] == t]),
               )
               for t in tumour_types},
        }

    # ---- Treatment recommendation ----
    if COL_TREATMENT in df.columns:
        for treat_val in sorted(df[COL_TREATMENT].dropna().unique()):
            n_overall = int((df[COL_TREATMENT] == treat_val).sum())
            rows[f"Treatment — {treat_val}"] = {
                "Overall": _n_pct(n_overall, len(df)),
                **{t: _n_pct(
                        int((df.loc[df[COL_TUMOUR] == t, COL_TREATMENT] == treat_val).sum()),
                        len(df[df[COL_TUMOUR] == t]),
                   )
                   for t in tumour_types},
            }

    return pd.DataFrame.from_dict(rows, orient="index", columns=columns)


# ===========================================================================
# Main
# ===========================================================================

def main() -> None:
    df = load_data(DATA_FILE)

    # Detect sex column (tries "gender" then "sex")
    sex_col = None
    for candidate in [COL_SEX, "sex"]:
        if candidate in df.columns:
            sex_col = candidate
            break
    if sex_col is None:
        print("  Note: no age/sex column found — corresponding tables will be skipped.")

    # ------------------------------------------------------------------
    # Individual tables
    # ------------------------------------------------------------------

    # Tumour distribution
    tum_dist = tumour_distribution(df)
    print("\n=== Tumour Type Distribution ===")
    print(tum_dist[["N", "n_pct"]].to_string())
    tum_dist.to_csv(f"{TABLE_DIR}/table_tumour_distribution.csv")

    # Consultation type
    cons_overall   = consultation_distribution(df)
    cons_by_tumour = consultation_distribution(df, group_col=COL_TUMOUR)
    print("\n=== Consultation Type (Overall) ===")
    print(cons_overall[["N", "n_pct"]].to_string())
    print("\n=== Consultation Type by Tumour Type ===")
    print(cons_by_tumour[["N", "n_pct"]].to_string())
    cons_overall.to_csv(f"{TABLE_DIR}/table_consultation_overall.csv")
    cons_by_tumour.to_csv(f"{TABLE_DIR}/table_consultation_by_tumour.csv")

    # Treatment distribution
    treat_by_tumour = treatment_distribution(df, group_col=COL_TUMOUR)
    if not treat_by_tumour.empty:
        print("\n=== Tumour Board Treatment by Tumour Type ===")
        print(treat_by_tumour[["N", "n_pct"]].to_string())
        treat_by_tumour.to_csv(f"{TABLE_DIR}/table_treatment_by_tumour.csv")

    # Age (only if column present)
    if COL_AGE in df.columns:
        age_overall   = age_statistics(df)
        age_by_tumour = age_statistics(df, group_col=COL_TUMOUR)
        print("\n=== Age (Overall) ===")
        print(age_overall[["N", "median_IQR"]].to_string())
        print("\n=== Age by Tumour Type ===")
        print(age_by_tumour[["N", "median_IQR"]].to_string())
        age_overall.to_csv(f"{TABLE_DIR}/table_age_overall.csv")
        age_by_tumour.to_csv(f"{TABLE_DIR}/table_age_by_tumour.csv")
    else:
        print(f"\n  Note: column '{COL_AGE}' not found — age tables skipped.")

    # Sex (only if column present)
    if sex_col:
        sex_overall   = sex_distribution(df, sex_col)
        sex_by_tumour = sex_distribution(df, sex_col, group_col=COL_TUMOUR)
        print(f"\n=== Sex / Gender (Overall) ===")
        print(sex_overall[["N", "n_pct"]].to_string())
        print(f"\n=== Sex / Gender by Tumour Type ===")
        print(sex_by_tumour[["N", "n_pct"]].to_string())
        sex_overall.to_csv(f"{TABLE_DIR}/table_sex_overall.csv")
        sex_by_tumour.to_csv(f"{TABLE_DIR}/table_sex_by_tumour.csv")

    # ------------------------------------------------------------------
    # Combined Table 1 (Excel)
    # ------------------------------------------------------------------
    table1 = build_table1(df, sex_col=sex_col)
    print("\n=== Table 1 — Combined Demographics ===")
    print(table1.to_string())
    table1.to_excel(f"{TABLE_DIR}/Table1_demographics.xlsx")
    print(f"\nTable 1 saved → {TABLE_DIR}/Table1_demographics.xlsx")

    print("\n=== Demographics Analysis Complete ===")


if __name__ == "__main__":
    main()
