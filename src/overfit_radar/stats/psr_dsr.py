"""Probabilistic Sharpe Ratio (PSR) and Deflated Sharpe Ratio (DSR) implementation."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform


EULER_MASCHERONI: float = 0.5772156649015329


def compute_psr(
    sharpe_hat: float,
    benchmark_sr: float,
    t_periods: int,
    skew: float = 0.0,
    kurtosis: float = 3.0,
) -> float:
    """
    Calculate the Probabilistic Sharpe Ratio (PSR).
    
    PSR(SR*) evaluates the probability that the estimated Sharpe ratio exceeds
    a benchmark threshold SR*, accounting for sample length, skewness, and fat tails (kurtosis).
    
    Formula:
    PSR(SR*) = Phi( (SR_hat - SR*) * sqrt(T-1) / sqrt(1 - skew*SR_hat + ((kurt-1)/4)*SR_hat^2) )
    
    Args:
        sharpe_hat: Estimated per-period (NON-annualized) Sharpe ratio.
        benchmark_sr: Benchmark per-period (NON-annualized) Sharpe ratio.
        t_periods: Number of observation bars (T).
        skew: Return skewness (gamma_3).
        kurtosis: Non-excess kurtosis (gamma_4, normal = 3.0).
        
    Returns:
        Probability in [0, 1] that true Sharpe exceeds benchmark_sr.
    """
    if t_periods <= 1:
        return 0.5

    # Variance adjustment term under non-normality
    var_term = 1.0 - skew * sharpe_hat + ((kurtosis - 1.0) / 4.0) * (sharpe_hat**2)
    if var_term <= 0.0 or np.isnan(var_term):
        denom = 1.0
    else:
        denom = np.sqrt(var_term)

    z = (sharpe_hat - benchmark_sr) * np.sqrt(t_periods - 1) / denom
    return float(stats.norm.cdf(z))


def compute_expected_max_sharpe(n_trials: float, var_sharpe: float) -> float:
    """
    Compute the expected maximum Sharpe ratio under the null hypothesis of multiple testing.
    
    Derived from extreme value theory (Bailey & Lopez de Prado, 2014):
    E[max_N {SR_n}] = sqrt(V) * [ (1 - gamma) * Phi^-1(1 - 1/N) + gamma * Phi^-1(1 - 1/(N*e)) ]
    
    Args:
        n_trials: Number of independent trials N.
        var_sharpe: Cross-sectional variance of per-period Sharpe ratios across trials.
    """
    if n_trials <= 1.0 or var_sharpe <= 0.0:
        return 0.0

    gamma = EULER_MASCHERONI
    e = np.e

    # Standard normal quantile approximations for upper tail
    p1 = 1.0 - 1.0 / n_trials
    p2 = 1.0 - 1.0 / (n_trials * e)

    q1 = stats.norm.ppf(p1)
    q2 = stats.norm.ppf(p2)

    e_max = np.sqrt(var_sharpe) * ((1.0 - gamma) * q1 + gamma * q2)
    return float(max(0.0, e_max))


def compute_dsr(
    sharpe_hat: float,
    var_sharpe: float,
    n_trials: float,
    t_periods: int,
    skew: float = 0.0,
    kurtosis: float = 3.0,
) -> float:
    """
    Calculate the Deflated Sharpe Ratio (DSR).
    
    DSR is the Probabilistic Sharpe Ratio evaluated at the expected maximum Sharpe
    under the multiple-testing null hypothesis:
    DSR = PSR(SR* = E[max_N {SR}])
    
    Args:
        sharpe_hat: Estimated per-period (NON-annualized) Sharpe ratio.
        var_sharpe: Cross-sectional variance of Sharpe estimates across trials.
        n_trials: Number of trials (raw N or Effective N).
        t_periods: Sample length T.
        skew: Return skewness.
        kurtosis: Return non-excess kurtosis.
    """
    sr_star = compute_expected_max_sharpe(n_trials=n_trials, var_sharpe=var_sharpe)
    return compute_psr(
        sharpe_hat=sharpe_hat,
        benchmark_sr=sr_star,
        t_periods=t_periods,
        skew=skew,
        kurtosis=kurtosis,
    )


def compute_effective_n(
    returns_matrix: pd.DataFrame,
    method: str = "eigenvalue",
    corr_cutoff: float = 0.5,
) -> float:
    """
    Estimate the number of effective independent trials (N_eff) from correlated strategy returns.
    
    Because strategies generated from a parameter grid exhibit high cross-correlation,
    raw N vastly overstates independent trials.
    
    Methods:
    - 'eigenvalue' (Participation Ratio):
        N_eff = (sum_i lambda_i)^2 / sum_i lambda_i^2
        Exact closed-form from random matrix theory: equals 1 for identical series, N for orthogonal series.
    - 'cluster' (Hierarchical Agglomerative Clustering):
        Clusters strategies on correlation distance D = sqrt(0.5 * (1 - rho)) at threshold corr_cutoff.
    """
    if method not in {"eigenvalue", "cluster"}:
        raise ValueError(f"Unknown effective N method: {method}")
    if not np.isfinite(returns_matrix.to_numpy(dtype=float)).all():
        raise ValueError("Effective N requires finite, synchronous returns")
    # Constant/flat series carry no independent return information.
    returns_matrix = returns_matrix.loc[:, returns_matrix.std(ddof=1) > 1e-12]
    n_strats = returns_matrix.shape[1]
    if n_strats == 0:
        return 1.0  # Conservative convention for a degenerate all-flat trial set.
    if n_strats <= 1:
        return float(n_strats)

    corr = returns_matrix.corr().fillna(0.0).to_numpy()
    # Ensure symmetry and 1s on diagonal
    corr = 0.5 * (corr + corr.T)
    np.fill_diagonal(corr, 1.0)

    if method == "eigenvalue":
        # Eigenvalues of correlation matrix
        eigvals = np.linalg.eigvalsh(corr)
        eigvals = np.maximum(0.0, eigvals)
        sum_eig = np.sum(eigvals)
        sum_sq_eig = np.sum(eigvals**2)
        if sum_sq_eig <= 1e-12:
            return 1.0
        n_eff = (sum_eig**2) / sum_sq_eig
        return float(np.clip(n_eff, 1.0, float(n_strats)))

    elif method == "cluster":
        # Distance metric D_ij = sqrt(0.5 * (1 - rho_ij))
        dist_mat = np.sqrt(np.maximum(0.0, 0.5 * (1.0 - corr)))
        np.fill_diagonal(dist_mat, 0.0)
        condensed_dist = squareform(dist_mat, checks=False)
        
        # Ward linkage
        z = linkage(condensed_dist, method="average")
        # Cutoff distance corresponding to correlation threshold
        dist_threshold = np.sqrt(0.5 * (1.0 - corr_cutoff))
        cluster_labels = fcluster(z, t=dist_threshold, criterion="distance")
        n_clusters = len(np.unique(cluster_labels))
        return float(n_clusters)
    else:
        raise ValueError(f"Unknown effective N method: {method}")
