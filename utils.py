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
  - Proportion z-test
  - Post-hoc power for a McNemar test from observed discordant pairs.
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


def compare_treatments(df, reference_col, comparison_cols):
    """
    Compute binary treatment concordance between a reference standard and one or
    more model outputs.

    For each model listed in *comparison_cols*, a new column
    ``{col}_treatment_concordance`` is appended to the DataFrame. The value is:

        1 → if the model’s predicted treatment matches either:
              (a) the primary reference treatment, or
              (b) the documented alternative reference treatment (if present)
        0 → otherwise

    Concordance is evaluated per patient (row-wise comparison).

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame containing:
            - ``{reference_col}_treatment`` (primary reference decision)
            - ``{reference_col}_treatment_alternativ`` (optional alternative decision)
            - ``{col}_treatment`` for each model in *comparison_cols*

    reference_col : str
        Base name of the reference standard (e.g. "tumorboard").
        The function expects the corresponding treatment columns to follow
        the naming convention described above.

    comparison_cols : list of str
        List of model base names for which concordance with the reference
        standard will be evaluated.

    Returns
    -------
    pandas.DataFrame
        A copy of the input DataFrame with one additional binary column per
        model: ``{col}_treatment_concordance``.
    """

    df = df.copy()

    ref = df[f"{reference_col}_treatment"]
    ref_alt = df[f"{reference_col}_treatment_alternativ"]

    for col in comparison_cols:
        pred = df[f"{col}_treatment"]

        match = (
            (pred == ref) |
            ((pred == ref_alt) & ref_alt.notna())
        )

        df[f"{col}_treatment_concordance"] = match.astype(int)

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
        DataFrame containing ``{col}_treatment_concordance`` columns.
    comparison_cols :
        List of model base-name columns.

    Returns
    -------
    dict
        ``{column_name: int}``
    """
    return {col: int(df[f"{col}_treatment_concordance"].sum()) for col in comparison_cols}


def calculate_correct_percentages(df: pd.DataFrame, comparison_cols: list,
                                  rename_dict: Optional[dict] = None) -> dict:
    """
    Return the percentage accuracy per model column.

    Parameters
    ----------
    df :
        DataFrame containing ``{col}_treatment_concordance`` columns.
    comparison_cols :
        List of model base-name columns.
    rename_dict :
        Optional mapping from raw column names to display labels.

    Returns
    -------
    dict
        ``{column_name_or_label: float}``
    """
    percentages = {col: df[f"{col}_treatment_concordance"].mean() * 100
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
        DataFrame with ``{col}_treatment_concordance`` binary columns.
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
        correct = int(df[f"{col}_treatment_concordance"].sum())
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
        Exactly the comparison columns (``{col}_treatment_concordance``) to test.
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
                        return_statistic: bool = True) -> dict:
    """
    Run Cochran's Q and pairwise McNemar tests.

    Parameters
    ----------
    df :
        DataFrame with binary (0/1) comparison columns.
    comparison_cols :
        List of ``{col}_treatment_concordance`` columns.
    alpha :
        Significance threshold.
    return_statistic :
        If True → return χ² statistic matrix.
        If False → return p-value matrix.

    Returns
    -------
    dict
        {
            "cochran": dict,
            "pairwise_matrix": pandas.DataFrame
        }
    """

    q_res = run_cochran_q(df, comparison_cols, alpha=alpha)

    matrix = pd.DataFrame(
        np.nan,
        index=comparison_cols,
        columns=comparison_cols
    )

    for c1, c2 in itertools.combinations(comparison_cols, 2):
        res = run_mcnemar(df, c1, c2, alpha=alpha)

        val = res["statistic"] if return_statistic else res["pvalue"]

        matrix.loc[c1, c2] = val
        matrix.loc[c2, c1] = val

    np.fill_diagonal(matrix.to_numpy(copy=True), 0.0)

    return {
        "cochran": q_res,
        "pairwise_matrix": matrix
    }


def mcnemar_power_from_df(df: pd.DataFrame, col1: str, col2: str,
                           alpha: float = 0.05) -> dict:
    """
    Estimate post-hoc power for a McNemar test from observed discordant pairs.

    Power is computed analytically from the observed b and c counts
    (discordant pairs), using the normal approximation to the McNemar
    statistic. This is a post-hoc power estimate — it reflects the power
    the study had to detect the observed effect size, not a prospective
    sample size calculation.

    Parameters
    ----------
    df :
        DataFrame with binary (0/1) columns col1 and col2.
    col1, col2 :
        Column names of the two methods being compared.
    alpha :
        Two-sided significance level (default 0.05).

    Returns
    -------
    dict
        Keys: b, c, n_discordant, effect_size, power.
    """
    from scipy.stats import norm

    b = int(np.sum((df[col1] == 1) & (df[col2] == 0)))
    c = int(np.sum((df[col1] == 0) & (df[col2] == 1)))
    n_discordant = b + c

    if n_discordant == 0:
        return {"b": b, "c": c, "n_discordant": 0,
                "effect_size": np.nan, "power": np.nan}

    # Effect size: proportion of discordant pairs favouring col1
    p_disc = b / n_discordant

    # Non-centrality parameter under H1
    z_alpha = norm.ppf(1 - alpha / 2)
    ncp = abs(2 * p_disc - 1) * np.sqrt(n_discordant)
    power = 1 - norm.cdf(z_alpha - ncp) + norm.cdf(-z_alpha - ncp)

    return {
        "b": b,
        "c": c,
        "n_discordant": n_discordant,
        "effect_size": round(p_disc, 4),
        "power": round(float(power), 4)
    }


def parse_treatment_list_column(df: pd.DataFrame,
                                column: str,
                                primary_output_col: Optional[str] = None) -> pd.DataFrame:
    """
    Parse treatment columns stored as serialized lists and optionally
    derive a primary-treatment column (first element).

    Parameters
    ----------
    df :
        Input dataframe.
    column :
        Column containing treatments stored as list-like strings or lists.
    primary_output_col :
        Name of derived column containing the primary treatment.
        If None → no primary column is created.

    Returns
    -------
    pd.DataFrame
        Modified dataframe (copy).
    """

    def parse_list(x):
        if isinstance(x, list):
            return x

        if isinstance(x, str):
            try:
                parsed = ast.literal_eval(x)
                if isinstance(parsed, list):
                    return parsed
            except:
                return [x]

        return []

    df = df.copy()

    # Parse list column
    df[column] = df[column].apply(parse_list)

    # Primary treatment
    if primary_output_col:
        df[primary_output_col] = df[column].apply(
            lambda x: x[0] if isinstance(x, list) and len(x) > 0 else np.nan
        )

    return df