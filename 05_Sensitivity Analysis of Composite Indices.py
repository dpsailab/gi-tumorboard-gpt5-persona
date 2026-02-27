import numpy as np
import pandas as pd
from scipy.stats import spearmanr

def sensitivity_analysis_index(df, base_weights, index_function):
    """
    Robustness test for composite indices under weight perturbation.
    """

    eps_grid = [-0.2, -0.1, 0, 0.1, 0.2]

    results = []

    base_scores = index_function(df)

    for eps in eps_grid:

        # Perturb weights
        new_weights = {
            k: v * (1 + eps)
            for k, v in base_weights.items()
        }

        # Renormalise
        total = sum(new_weights.values())
        new_weights = {k: v/total for k, v in new_weights.items()}

        # Compute new index values
        perturbed_scores = index_function(df, weights=new_weights)

        # Rank stability
        rho, _ = spearmanr(
            base_scores["score"],
            perturbed_scores["score"]
        )

        results.append({
            "epsilon": eps,
            "spearman_rank_correlation": rho
        })

    return pd.DataFrame(results)