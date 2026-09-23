# Histogram diagnostics and event-level Poisson-bootstrap covariances
#
# m.mieskolainen@imperial.ac.uk, 2025

import numpy as np
from scipy import sparse


def chi2_func(h, h_ref, type="symmetric", var=None, var_ref=None, ddof=0):
    """Compute a variance-weighted histogram discrepancy per occupied bin

    Args:
        h (array_like): Histogram values (...)
        h_ref (array_like): Reference values, matching h
        type (str): symmetric, pearson, or neyman variance rule
        var (array_like | None): Variances matching h, default h
        var_ref (array_like | None): Reference variances, default h_ref
        ddof (int): Degrees subtracted from occupied-bin count

    Returns:
        float: Reduced discrepancy, inf for unsupported differences, nan for no degrees of freedom
    """

    h, h_ref = np.broadcast_arrays(np.asarray(h, float), np.asarray(h_ref, float))
    if type == "symmetric":
        variance = (h if var is None else np.asarray(var)) + (h_ref if var_ref is None else np.asarray(var_ref))
    elif type == "pearson":
        variance = h_ref if var_ref is None else np.asarray(var_ref)
    elif type in ("neyman", "neuman"):
        variance = h if var is None else np.asarray(var)
    else:
        raise ValueError("Unknown chi-squared diagnostic.")

    variance = np.broadcast_to(variance, h.shape)
    if np.any(~np.isfinite(variance)) or np.any(variance < 0):
        raise ValueError("Variances must be finite and nonnegative.")

    difference = h - h_ref
    if np.any((variance == 0) & (difference != 0)):
        return np.inf

    occupied = variance > 0
    ndf = np.count_nonzero(occupied) - ddof
    if ndf <= 0:
        return np.nan
    
    return np.sum(difference[occupied]**2 / variance[occupied]) / ndf


def histogram_contributions(values, bins, weights=None, event_ids=None, n_events=None):
    """Accumulate parent-event bin weights for Poisson-bootstrap covariance

    Args:
        values (array_like): Sample values (M,)
        bins (array_like): Increasing bin edges (K + 1,)
        weights (array_like | None): Sample weights (M,), default one
        event_ids (array_like | None): Nonnegative parent IDs (M,), default one per sample
        n_events (int | None): Parent count N, inferred from IDs when omitted

    Returns:
        scipy.sparse.csr_matrix: Contributions A (N, K), with covariance A.T @ A
    """

    values, bins = np.asarray(values), np.asarray(bins)
    
    if values.ndim != 1 or bins.ndim != 1 or len(bins) < 2 or np.any(np.diff(bins) <= 0):
        raise ValueError("Expected 1D values and increasing bin edges.")
    
    weights = np.ones(len(values)) if weights is None else np.asarray(weights, float)
    ids = np.arange(len(values)) if event_ids is None else np.asarray(event_ids)
    
    if weights.shape != values.shape or ids.shape != values.shape:
        raise ValueError("Values, weights and event IDs must have matching shapes.")
    
    if not np.issubdtype(ids.dtype, np.integer) or np.any(ids < 0):
        raise ValueError("Event IDs must be nonnegative integers.")
    
    minimum_rows = int(ids.max()) + 1 if len(ids) else 0
    n_events = minimum_rows if n_events is None else n_events
    
    if n_events < minimum_rows:
        raise ValueError("n_events does not cover the parent event IDs.")
    
    index = np.searchsorted(bins, values, side="right") - 1
    index[values == bins[-1]] = len(bins) - 2  # numpy.histogram includes the last edge
    valid = np.isfinite(values) & (index >= 0) & (index < len(bins) - 1)
    matrix = sparse.coo_matrix((weights[valid], (ids[valid], index[valid])),
                               shape=(n_events, len(bins) - 1)).tocsr()
    matrix.sum_duplicates()
    
    return matrix


def covariance_chi2(difference, covariance):
    """Compute covariance-weighted discrepancy per supported direction

    Args:
        difference (array_like): Bin differences (K,)
        covariance (array_like): Difference covariance (K, K)

    Returns:
        float: Quadratic discrepancy per rank, inf for inconsistent null-space components
    """
    
    difference, covariance = np.asarray(difference), np.asarray(covariance)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    tolerance = max(float(np.max(np.abs(eigenvalues))), np.finfo(float).tiny) * 1e-10
    positive = eigenvalues > tolerance
    coordinates = eigenvectors.T @ difference
    
    if np.linalg.norm(coordinates[~positive]) > 1e-8 * max(np.linalg.norm(difference), 1):
        return np.inf

    if not np.any(positive):
        return 0.0 if np.allclose(difference, 0) else np.inf
    
    return np.sum(coordinates[positive]**2 / eigenvalues[positive]) / np.count_nonzero(positive)

def compute_ratio_uncertainty(hist_num, var_num, hist_den, var_den, covariance=0):
    """Propagate histogram ratio uncertainty including cross covariance

    Args:
        hist_num (array_like): Numerator values (...)
        var_num (array_like): Numerator variances (...)
        hist_den (array_like): Denominator values (...)
        var_den (array_like): Denominator variances (...)
        covariance (array_like | float): Numerator-denominator covariance, broadcastable to (...)

    Returns:
        tuple[np.ndarray, np.ndarray]: Ratio and standard error (...), NaN for nonpositive denominator
    """
    
    num, vn, den, vd, cov = np.broadcast_arrays(
        *[np.asarray(x, float) for x in (hist_num, var_num, hist_den, var_den, covariance)])
    ratio = np.full(den.shape, np.nan)
    sigma = np.full(den.shape, np.nan)
    valid = den > 0
    ratio[valid] = num[valid] / den[valid]
    variance = (vn[valid] / den[valid]**2 + num[valid]**2 * vd[valid] / den[valid]**4
                - 2 * num[valid] * cov[valid] / den[valid]**3)
    sigma[valid] = np.sqrt(np.maximum(variance, 0))
    
    return ratio, sigma
