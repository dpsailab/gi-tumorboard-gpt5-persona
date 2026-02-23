"""
utils.py
========
Shared utility functions used across all analysis modules.

Covers:
  - Embedding parsing (string / list / ndarray → numpy array)
  - Cosine similarity (robust, handles zero-norm vectors)
  - Majority-treatment computation
  - Treatment comparison (with alternative-treatment support)
  - Confidence-interval computation (Wilson method)
  - Cochran's Q and pairwise McNemar statistical tests
  - Proportion z-test with Holm–Bonferroni correction
"""

from __future__ import annotations

import ast
import itertools
from collections import Counter
from typing import Optional

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import chi2
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.proportion import proportion_confint


# ===========================================================================
# Embedding utilities
# ===========================================================================

def parse_embedding(raw) -> Optional[np.ndarray]:
    """
    Convert a stored embedding to a 1-D float numpy array.

    The embedding may be serialised in several formats depending on how the
    dataset was assembled:
      - Already a list or numpy array  →  cast to float64 array.
      - A string representation of a list / tuple  →  evaluated with
        ``ast.literal_eval`` and cast.
      - A comma-separated flat string  →  split and cast.
      - ``NaN`` / ``None`` / un-parseable  →  return ``None``.

    Parameters
    ----------
    raw :
        Raw cell value from the DataFrame.

    Returns
    -------
    numpy.ndarray or None
    """
    if raw is None:
        return None
    if isinstance(raw, float) and np.isnan(raw):
        return None
    if isinstance(raw, (list, np.ndarray)):
        return np.array(raw, dtype=float)
    if isinstance(raw, str):
        # Try structured literal (list / tuple)
        try:
            parsed = ast.literal_eval(raw)
            if isinstance(parsed, (list, tuple, np.ndarray)):
                return np.array(parsed, dtype=float)
        except Exception:
            pass
        # Fallback: comma-separated values
        try:
            parts = [float(v) for v in raw.strip("[]() ").split(",") if v.strip()]
            return np.array(parts, dtype=float)
        except Exception:
            pass
    return None


def safe_cosine(u: Optional[np.ndarray], v: Optional[np.ndarray]) -> Optional[float]:
    """
    Compute cosine similarity between two vectors.

    Returns ``None`` when either vector is ``None`` or has zero norm, avoiding
    division-by-zero errors that would otherwise silently produce ``NaN``.

    Parameters
    ----------
    u, v :
        1-D numpy arrays of equal length, or ``None``.

    Returns
    -------
    float or None
        Cosine similarity in [−1, 1], or ``None`` on failure.
    """
    if u is None or v is None:
        return None
    norm_u = np.linalg.norm(u)
    norm_v = np.linalg.norm(v)
    if norm_u == 0 or norm_v == 0:
        return None
    return float(np.dot(u, v) / (norm_u * norm_v))


# ===========================================================================
# Treatment-comparison utilities
# ===========================================================================

def compute_majority_treatment(row: pd.Series, specialist_cols: list) -> Optional[str]:
    """
    Determine the majority treatment recommendation across specialist roles.

    For each patient row, the function collects non-null treatment values from
    the specified specialist columns and returns the most frequent one.  Ties
    are broken by ``Counter.most_common`` (arbitrary but reproducible).

    Parameters
    ----------
    row :
        A single DataFrame row (from ``df.apply(..., axis=1)``).
    specialist_cols :
        Column names containing the specialist treatment predictions.

    Returns
    -------
    str or None
        Most frequent treatment string, or ``None`` if all values are missing.
    """
    treatments = [row[col] for col in specialist_cols if pd.notna(row[col])]
    if not treatments:
        return None
    counts = Counter(treatments)
    return counts.most_common(1)[0][0]


def compare_treatments(df: pd.DataFrame, reference_col: str,
                       comparison_cols: list) -> pd.DataFrame:
    """
    Append binary comparison columns to *df* for each model in *comparison_cols*.

    For every patient × model pair a new column ``{col}_comparison`` is
    created with value 1 if the model's treatment matches the reference
    (or its documented alternative) and 0 otherwise.

    Parameters
    ----------
    df :
        Input DataFrame.  Must contain ``{reference_col}_treatment``,
        ``{reference_col}_treatment_alternativ``, and
        ``{col}_treatment`` for each ``col`` in *comparison_cols*.
    reference_col :
        Base name of the reference standard column (e.g.
        ``"Konferenzbeschluss"``).
    comparison_cols :
        List of model base-name columns to evaluate.

    Returns
    -------
    pandas.DataFrame
        The input DataFrame with ``{col}_comparison`` columns appended.
    """
    ref_treatment_col = f"{reference_col}_treatment"
    ref_alt_col       = f"{reference_col}_treatment_alternativ"

    for col in comparison_cols:
        comp_col = f"{col}_comparison"
        results  = []

        for _, row in df.iterrows():
            ref     = row.get(ref_treatment_col)
            ref_alt = row.get(ref_alt_col)
            pred    = row.get(f"{col}_treatment")

            if pd.isna(ref) or pd.isna(pred):
                results.append(0)
            else:
                match = (ref == pred) or (pd.notna(ref_alt) and ref_alt == pred)
                results.append(int(match))

        df[comp_col] = results

    return df


# ===========================================================================
# Accuracy & confidence-interval helpers
# ===========================================================================

def calculate_correct_counts(df: pd.DataFrame, comparison_cols: list) -> dict:
    """
    Return the absolute number of correct predictions per model column.

    Parameters
    ----------
    df :
        DataFrame containing ``{col}_comparison`` columns.
    comparison_cols :
        List of model base-name columns.

    Returns
    -------
    dict
        ``{column_name: int}``
    """
    return {col: int(df[f"{col}_comparison"].sum()) for col in comparison_cols}


def calculate_correct_percentages(df: pd.DataFrame, comparison_cols: list,
                                  rename_dict: Optional[dict] = None) -> dict:
    """
    Return the percentage accuracy per model column.

    Parameters
    ----------
    df :
        DataFrame containing ``{col}_comparison`` columns.
    comparison_cols :
        List of model base-name columns.
    rename_dict :
        Optional mapping from raw column names to display labels.

    Returns
    -------
    dict
        ``{column_name_or_label: float}``
    """
    percentages = {col: df[f"{col}_comparison"].mean() * 100
                   for col in comparison_cols}
    if rename_dict:
        percentages = {rename_dict.get(k, k): v for k, v in percentages.items()}
    return percentages


def wilson_ci(df: pd.DataFrame, comparison_cols: list,
              rename_dict: Optional[dict] = None,
              alpha: float = 0.05) -> pd.DataFrame:
    """
    Compute Wilson (score) confidence intervals for each model's accuracy.

    The Wilson interval is preferred over the normal approximation for small
    sample sizes and proportions near 0 or 1.

    Parameters
    ----------
    df :
        DataFrame with ``{col}_comparison`` binary columns.
    comparison_cols :
        Model base-name columns to evaluate.
    rename_dict :
        Optional label mapping.
    alpha :
        Significance level for the CI (default 0.05 → 95 % CI).

    Returns
    -------
    pandas.DataFrame
        Columns: model, n, correct, proportion, ci_low, ci_high.
    """
    rows = []
    n = len(df)
    for col in comparison_cols:
        correct = int(df[f"{col}_comparison"].sum())
        p = correct / n if n > 0 else 0.0
        ci_low, ci_high = proportion_confint(count=correct, nobs=n,
                                             alpha=alpha, method="wilson")
        label = rename_dict.get(col, col) if rename_dict else col
        rows.append(dict(model=label, n=n, correct=correct,
                         proportion=p, ci_low=ci_low, ci_high=ci_high))
    return pd.DataFrame(rows)


# ===========================================================================
# Statistical tests
# ===========================================================================

def run_mcnemar(df: pd.DataFrame, col1: str, col2: str,
                alpha: float = 0.05) -> dict:
    """
    Run a McNemar test for two matched binary outcome columns.

    Uses the continuity-corrected chi-square statistic recommended for
    sample sizes where b + c < 25 (Altman & Bland, 1994).

    Parameters
    ----------
    df :
        DataFrame with binary (0/1) columns *col1* and *col2*.
    col1, col2 :
        Column names of the two methods being compared.
    alpha :
        Significance threshold.

    Returns
    -------
    dict
        Keys: b, c, statistic (χ²), pvalue, reject.
    """
    b = int(np.sum((df[col1] == 1) & (df[col2] == 0)))
    c = int(np.sum((df[col1] == 0) & (df[col2] == 1)))
    stat = ((abs(b - c) - 1) ** 2 / (b + c)) if (b + c) > 0 else 0.0
    p    = chi2.sf(stat, df=1)
    return {"b": b, "c": c, "statistic": stat, "pvalue": p, "reject": p < alpha}


def run_cochran_q(df: pd.DataFrame, comparison_cols: list,
                  alpha: float = 0.05) -> dict:
    """
    Run Cochran's Q test across *k* matched binary outcomes.

    Cochran's Q is the omnibus test for equality of proportions in paired
    (repeated-measures) designs.  It is the standard precursor to post-hoc
    pairwise McNemar tests.

    Parameters
    ----------
    df :
        DataFrame with binary (0/1) columns listed in *comparison_cols*.
    comparison_cols :
        Exactly the comparison columns (``{col}_comparison``) to test.
    alpha :
        Significance threshold.

    Returns
    -------
    dict
        Keys: Q, df, pvalue, reject.
    """
    k = len(comparison_cols)
    X = df[comparison_cols].values.astype(float)
    row_sums = X.sum(axis=1)
    col_sums = X.sum(axis=0)
    T = X.sum()

    term1 = k * (k - 1) * np.sum((col_sums - T / k) ** 2)
    term2 = np.sum(row_sums * (k - row_sums))
    Q = term1 / term2 if term2 > 0 else np.nan
    p = chi2.sf(Q, df=k - 1) if not np.isnan(Q) else np.nan

    return {"Q": Q, "df": k - 1, "pvalue": p, "reject": (p < alpha) if not np.isnan(p) else False}


def cochran_and_mcnemar(df: pd.DataFrame, comparison_cols: list,
                        alpha: float = 0.05,
                        return_statistic: bool = True) -> pd.DataFrame:
    """
    Run Cochran's Q, then produce a pairwise McNemar matrix.

    Parameters
    ----------
    df :
        DataFrame with binary (0/1) comparison columns.
    comparison_cols :
        Exactly the ``{col}_comparison`` columns to test.
    alpha :
        Significance threshold.
    return_statistic :
        If True, the matrix cells contain the McNemar χ² statistic.
        If False, the matrix cells contain the raw p-values.

    Returns
    -------
    pandas.DataFrame
        Symmetric N×N matrix (NaN on diagonal = self-comparison not meaningful).
    """
    q_res = run_cochran_q(df, comparison_cols, alpha=alpha)
    print("=== Cochran's Q Test ===")
    print(f"Q = {q_res['Q']:.3f}, df = {q_res['df']}, "
          f"p = {q_res['pvalue']:.4f}, reject = {q_res['reject']}\n")

    matrix = pd.DataFrame(np.nan, index=comparison_cols, columns=comparison_cols)
    print("=== Pairwise McNemar Tests ===")
    for c1, c2 in itertools.combinations(comparison_cols, 2):
        res = run_mcnemar(df, c1, c2, alpha=alpha)
        val = res["statistic"] if return_statistic else res["pvalue"]
        matrix.loc[c1, c2] = val
        matrix.loc[c2, c1] = val
        print(f"{c1} vs {c2}: b={res['b']}, c={res['c']}, "
              f"χ²={res['statistic']:.2f}, p={res['pvalue']:.4f}, "
              f"reject={res['reject']}")

    vals = matrix.values.copy()
    np.fill_diagonal(vals, 0.0)
    matrix = pd.DataFrame(vals, index=matrix.index, columns=matrix.columns)
    return matrix


def proportions_ztest_holm(percentages: dict, total: int = 100) -> pd.DataFrame:
    """
    Pairwise proportion z-tests with Holm–Bonferroni correction.

    Parameters
    ----------
    percentages :
        Dict of ``{label: percentage_correct}``.  Percentage values are
        internally converted to integer counts assuming *total* observations.
    total :
        Number of observations per condition.

    Returns
    -------
    pandas.DataFrame
        Columns: label, raw_p, adjusted_p, reject.
    """
    keys  = list(percentages.keys())
    pairs = list(itertools.combinations(keys, 2))
    pvals, labels = [], []

    for k1, k2 in pairs:
        counts = [int(percentages[k1]), int(percentages[k2])]
        nobs   = [total, total]
        _, p   = sm.stats.proportions_ztest(counts, nobs, alternative="two-sided")
        pvals.append(p)
        labels.append(f"{k1} vs {k2}")

    reject, pvals_adj, _, _ = multipletests(pvals, alpha=0.05, method="holm")

    return pd.DataFrame({
        "comparison": labels,
        "raw_p": pvals,
        "holm_adjusted_p": pvals_adj,
        "reject": reject,
    })
