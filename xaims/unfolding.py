# Binned EM-unfolding and dependence on the initial truth shape
#
# m.mieskolainen@imperial.ac.uk, 2025

import numpy as np

def iterative_em(response, observed, initial_truth, iterations=1):
    """Unfold Poisson counts by EM with efficiency correction and early stopping

    Args:
        response (array_like): Response (R, T), column efficiencies in (0, 1]
        observed (array_like): Nonnegative reconstructed counts (R,)
        initial_truth (array_like): Positive initial truth shape (T,), normalized internally
        iterations (int): EM iteration count

    Returns:
        np.ndarray: Estimated truth counts (T,)
    """

    response = np.asarray(response, dtype=float)
    observed, initial_truth = np.asarray(observed, float), np.asarray(initial_truth, float)
    
    if (response.ndim != 2 or observed.shape != (response.shape[0],)
            or initial_truth.shape != (response.shape[1],) or iterations < 1):
        raise ValueError("Incompatible response, counts, initial truth shape or iteration count.")
    
    if (np.any(~np.isfinite(response)) or np.any(response < 0)
            or np.any(~np.isfinite(observed)) or np.any(observed < 0)
            or np.any(~np.isfinite(initial_truth)) or np.any(initial_truth <= 0)):
        raise ValueError("Response/counts must be nonnegative and initial truth shape strictly positive.")
    
    efficiency = response.sum(axis=0)
    
    if np.any(efficiency <= 0) or np.any(efficiency > 1 + 1e-12):
        raise ValueError("Response columns must have efficiencies in (0, 1].")
    
    estimate = initial_truth / initial_truth.sum()
    
    for _ in range(iterations):
        folded = response @ estimate
        if np.any((folded == 0) & (observed > 0)):
            raise ValueError("The response cannot explain an occupied detector bin.")

        ratio = np.divide(observed, folded, out=np.zeros_like(observed), where=folded > 0)
        estimate = estimate * (response.T @ ratio) / efficiency
    
    return estimate
