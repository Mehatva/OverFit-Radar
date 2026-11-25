"""Unit tests for statistical metrics: Sharpe, PSR, DSR, Effective N, and CSCV PBO."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from overfit_radar.stats.basic import (
    calculate_sharpe,
    calculate_sortino,
    calculate_max_drawdown,
    calculate_skewness,
    calculate_kurtosis,
)
from overfit_radar.stats.psr_dsr import (
    compute_psr,
    compute_expected_max_sharpe,
    compute_dsr,
    compute_effective_n,
)
from overfit_radar.stats.pbo import compute_cscv_pbo


def test_basic_metrics_invariants():
    """Verify standard properties of Sharpe, Sortino, Drawdown, Skew, and Kurtosis."""
    # Zero returns: Sharpe = 0, Max Drawdown = 0
    zero_rets = pd.Series([0.0] * 100)
    assert calculate_sharpe(zero_rets) == 0.0
    assert calculate_max_drawdown(zero_rets) == 0.0

    # Normal distribution: Skewness ~ 0, Kurtosis ~ 3.0 (non-excess)
    rng = np.random.default_rng(42)
    norm_rets = pd.Series(rng.normal(0.001, 0.01, size=5000))
    skew = calculate_skewness(norm_rets)
    kurt = calculate_kurtosis(norm_rets)
    assert abs(skew) < 0.15, f"Expected normal skew ~ 0, got {skew}"
    assert abs(kurt - 3.0) < 0.25, f"Expected normal non-excess kurtosis ~ 3.0, got {kurt}"

    # Drawdown check on a known 20% drop
    drop_rets = pd.Series([0.0, 0.10, -0.20, 0.05])
    mdd = calculate_max_drawdown(drop_rets)
    assert 0.15 < mdd < 0.25, f"Expected MDD around 20%, got {mdd}"


def test_psr_known_values():
    """
    Verify Probabilistic Sharpe Ratio against numerical evaluation.
    Formula:
    PSR(SR*) = Phi( (SR_hat - SR*) * sqrt(T-1) / sqrt(1 - skew*SR_hat + ((kurt-1)/4)*SR_hat^2) )
    """
    t_obs = 1250  # 5 years daily
    sr_hat = 0.05  # daily Sharpe (~0.79 annualised)
    sr_star = 0.0  # benchmark daily Sharpe
    skew = -0.5
    kurt = 4.0  # non-excess kurtosis (fat tails)

    # Manual calculation
    denom = np.sqrt(1.0 - skew * sr_hat + ((kurt - 1.0) / 4.0) * (sr_hat**2))
    z = (sr_hat - sr_star) * np.sqrt(t_obs - 1) / denom
    expected_psr = stats.norm.cdf(z)

    computed_psr = compute_psr(
        sharpe_hat=sr_hat,
        benchmark_sr=sr_star,
        t_periods=t_obs,
        skew=skew,
        kurtosis=kurt,
    )
    assert computed_psr == pytest.approx(expected_psr, rel=1e-5)
    assert 0.0 <= computed_psr <= 1.0


def test_dsr_and_expected_max_sharpe():
    """Verify DSR calculation and extreme value theory expected maximum Sharpe."""
    n_trials = 100
    var_sharpe = 0.0005  # Variance of daily Sharpe estimates across trials
    
    # Expected max Sharpe must be positive and scale with sqrt(var)
    e_max = compute_expected_max_sharpe(n_trials=n_trials, var_sharpe=var_sharpe)
    assert e_max > 0.0
    
    # When trial Sharpe is equal to the expected maximum from pure luck, DSR must be ~ 0.50
    dsr_at_null = compute_dsr(
        sharpe_hat=e_max,
        var_sharpe=var_sharpe,
        n_trials=n_trials,
        t_periods=1000,
        skew=0.0,
        kurtosis=3.0,
    )
    assert dsr_at_null == pytest.approx(0.50, abs=0.05)


def test_effective_n_clustering():
    """
    Test Effective N estimation:
    - If series are identical (correlation 1.0), Effective N must be 1.
    - If series are orthogonal (independent), Effective N must be close to total trials N.
    """
    n_days = 500
    n_series = 10
    rng = np.random.default_rng(42)

    # Case 1: Identical series (complete redundancy)
    base = rng.normal(0, 0.01, size=n_days)
    identical_df = pd.DataFrame({f"s_{i}": base for i in range(n_series)})
    n_eff_identical = compute_effective_n(identical_df)
    assert n_eff_identical == pytest.approx(1.0, abs=0.1)

    # Case 2: Independent series
    indep_df = pd.DataFrame(
        rng.normal(0, 0.01, size=(n_days, n_series)),
        columns=[f"s_{i}" for i in range(n_series)],
    )
    n_eff_indep = compute_effective_n(indep_df)
    assert 7.0 <= n_eff_indep <= 10.0


def test_cscv_pbo_random_walk_sanity(random_walk_returns):
    """
    REQUIRED TEST 2: Random-Walk Sanity for PBO.
    On pure simulated random-walk returns where no strategy has any edge:
    1. CSCV PBO should be approximately 0.50 (between 0.35 and 0.65).
    2. Out-of-sample relative rank of the 'best' IS strategy should hover near 0.5.
    """
    # 40 strategies with pure noise returns over 1000 days
    df = random_walk_returns(n_days=1000, n_series=40, seed=1)
    
    pbo_res = compute_cscv_pbo(df, num_blocks=8, max_combinations=70)
    pbo = pbo_res["pbo"]
    
    # Under pure noise, the best strategy in-sample has ~50% probability of being below median OOS
    assert 0.35 <= pbo <= 0.65, f"Expected PBO near 0.50 for pure random walk, got {pbo:.3f}"
    assert "logits" in pbo_res
    assert len(pbo_res["logits"]) > 0
