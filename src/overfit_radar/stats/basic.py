"""Basic performance and risk statistics for quantitative strategies."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def calculate_sharpe(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    annualize: bool = True,
    periods_per_year: int = 252,
) -> float:
    """
    Compute daily or annualized Sharpe ratio.
    
    Args:
        returns: Daily net returns series.
        risk_free_rate: Annualized risk-free rate (e.g. 0.0).
        annualize: If True, multiply per-period Sharpe by sqrt(periods_per_year).
        periods_per_year: Number of trading days per year (default 252).
    """
    clean_rets = returns.dropna()
    if len(clean_rets) < 2:
        return 0.0

    rf_daily = risk_free_rate / periods_per_year
    excess_rets = clean_rets - rf_daily

    mean = excess_rets.mean()
    std = excess_rets.std(ddof=1)

    if std <= 1e-12 or np.isnan(std):
        return 0.0

    daily_sr = float(mean / std)
    if annualize:
        return float(daily_sr * np.sqrt(periods_per_year))
    return daily_sr


def calculate_sortino(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    annualize: bool = True,
    periods_per_year: int = 252,
) -> float:
    """
    Compute Sortino ratio (penalizes only downside variance).
    """
    clean_rets = returns.dropna()
    if len(clean_rets) < 2:
        return 0.0

    rf_daily = risk_free_rate / periods_per_year
    excess_rets = clean_rets - rf_daily
    mean = excess_rets.mean()

    # Downside deviations below target (rf_daily)
    downside_diff = np.minimum(excess_rets, 0.0)
    downside_dev = np.sqrt(np.mean(downside_diff**2))

    if downside_dev <= 1e-12 or np.isnan(downside_dev):
        return 0.0

    daily_sortino = float(mean / downside_dev)
    if annualize:
        return float(daily_sortino * np.sqrt(periods_per_year))
    return daily_sortino


def calculate_max_drawdown(returns: pd.Series) -> float:
    """
    Compute Maximum Peak-to-Trough Drawdown from daily returns.
    
    Returns:
        Max drawdown as a positive fraction (e.g. 0.25 = 25% drawdown).
    """
    clean_rets = returns.dropna()
    if clean_rets.empty:
        return 0.0

    # Compounded cumulative wealth path
    cum_wealth = (1.0 + clean_rets).cumprod()
    running_max = cum_wealth.cummax().clip(lower=1.0)
    drawdown = (running_max - cum_wealth) / running_max

    return float(drawdown.max())


def calculate_skewness(returns: pd.Series) -> float:
    """Compute sample skewness (Fisher-Pearson, unbiased)."""
    clean_rets = returns.dropna()
    if len(clean_rets) < 3:
        return 0.0
    val = float(stats.skew(clean_rets, bias=False))
    return 0.0 if np.isnan(val) else val


def calculate_kurtosis(returns: pd.Series) -> float:
    """
    Compute Pearson's non-excess kurtosis (where Normal distribution is 3.0).
    """
    clean_rets = returns.dropna()
    if len(clean_rets) < 4:
        return 3.0
    # scipy kurtosis with fisher=False yields non-excess kurtosis (normal=3)
    val = float(stats.kurtosis(clean_rets, fisher=False, bias=False))
    return 3.0 if np.isnan(val) else val


def calculate_autocorrelation(returns: pd.Series, lag: int = 1) -> float:
    """Compute return autocorrelation at specified lag."""
    clean_rets = returns.dropna()
    if len(clean_rets) <= lag:
        return 0.0
    val = clean_rets.autocorr(lag=lag)
    return 0.0 if np.isnan(val) else float(val)


def calculate_turnover(positions: pd.Series, periods_per_year: int = 252) -> float:
    """
    Calculate annualized portfolio turnover rate from position history.
    Turnover is the sum of absolute position changes scaled to 1 year.
    """
    clean_pos = positions.dropna()
    if len(clean_pos) < 2:
        return 0.0
    trades = clean_pos.diff().abs().fillna(0.0)
    avg_daily_turnover = trades.mean()
    return float(avg_daily_turnover * periods_per_year)
