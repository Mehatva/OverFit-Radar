"""Vectorized signal generation logic for all strategy families."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _apply_latch(entry_condition: pd.Series, exit_condition: pd.Series) -> pd.Series:
    """
    Fast vectorized latch for stateful entry/exit strategies.
    
    When entry_condition is True -> state becomes 1.
    When exit_condition is True -> state becomes 0.
    Otherwise -> state remains unchanged from previous bar.
    """
    n = len(entry_condition)
    state = np.zeros(n, dtype=np.float64)
    curr = 0.0
    
    ent = entry_condition.to_numpy(dtype=bool)
    ex = exit_condition.to_numpy(dtype=bool)
    
    for i in range(n):
        if ent[i]:
            curr = 1.0
        elif ex[i]:
            curr = 0.0
        state[i] = curr
        
    return pd.Series(state, index=entry_condition.index, dtype=np.float64)


def generate_sma_crossover(prices: pd.DataFrame, fast_window: int, slow_window: int) -> pd.Series:
    """SMA Crossover: Long when Fast SMA > Slow SMA, flat otherwise."""
    close = prices["adjusted_close"]
    fast = close.rolling(window=fast_window).mean()
    slow = close.rolling(window=slow_window).mean()
    
    sig = (fast > slow).astype(np.float64)
    # Fill warmup period with flat
    return sig.fillna(0.0)


def generate_momentum(prices: pd.DataFrame, lookback: int, holding_period: int) -> pd.Series:
    """
    Time-Series Momentum:
    Long if return over past lookback days is positive.
    Rebalanced every holding_period days to avoid daily churning.
    """
    close = prices["adjusted_close"]
    # Lookback return
    ret_lookback = close / close.shift(lookback) - 1.0
    raw_sig = (ret_lookback > 0.0).astype(np.float64)
    
    if holding_period <= 1:
        return raw_sig.fillna(0.0)
        
    # Sample every holding_period days and forward-fill
    mask = np.arange(len(close)) % holding_period == 0
    sampled_sig = raw_sig.where(mask)
    return sampled_sig.ffill().fillna(0.0)


def _compute_rsi(close: pd.Series, window: int) -> pd.Series:
    """Compute Relative Strength Index (RSI)."""
    delta = close.diff()
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    avg_gain = pd.Series(gain, index=close.index).rolling(window=window).mean()
    avg_loss = pd.Series(loss, index=close.index).rolling(window=window).mean()
    
    rs = avg_gain / (avg_loss + 1e-9)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi.fillna(50.0)


def generate_rsi_mean_reversion(
    prices: pd.DataFrame,
    window: int,
    oversold: float,
    overbought: float,
) -> pd.Series:
    """
    RSI Mean Reversion:
    Enter long when RSI drops below oversold threshold.
    Exit when RSI rises above overbought threshold.
    """
    close = prices["adjusted_close"]
    rsi = _compute_rsi(close, window)
    
    entry = rsi < oversold
    exit_ = rsi > overbought
    return _apply_latch(entry, exit_)


def generate_bollinger_mean_reversion(
    prices: pd.DataFrame,
    window: int,
    num_std: float,
    exit_at_mid: bool = True,
) -> pd.Series:
    """
    Bollinger Bands Mean Reversion:
    Enter long when price falls below lower band (mean - num_std * std).
    Exit when price recovers to middle band (mean) or upper band.
    """
    close = prices["adjusted_close"]
    mid = close.rolling(window=window).mean()
    std = close.rolling(window=window).std()
    
    lower = mid - num_std * std
    upper = mid if exit_at_mid else (mid + num_std * std)
    
    entry = close < lower
    exit_ = close > upper
    return _apply_latch(entry, exit_)


def generate_donchian_breakout(
    prices: pd.DataFrame,
    lookback_high: int,
    exit_fraction: float = 0.5,
) -> pd.Series:
    """
    Donchian Channel Breakout (Trend Following):
    Enter long when price closes above the N-day High.
    Exit when price closes below the (N * exit_fraction)-day Low.
    """
    adjustment = prices["adjusted_close"] / prices["close"]
    high = prices["high"] * adjustment
    low = prices["low"] * adjustment
    close = prices["adjusted_close"]
    
    lookback_low = max(5, int(lookback_high * exit_fraction))
    
    upper_channel = high.rolling(window=lookback_high).max().shift(1)
    lower_channel = low.rolling(window=lookback_low).min().shift(1)
    
    entry = close > upper_channel
    exit_ = close < lower_channel
    return _apply_latch(entry, exit_)


def generate_vol_filtered_momentum(
    prices: pd.DataFrame,
    lookback: int,
    vol_window: int,
    vol_percentile: float = 50.0,
) -> pd.Series:
    """
    Volatility-Filtered Momentum:
    Enter long only when momentum > 0 AND short-term volatility is below
    its historical rolling percentile threshold (avoids high-volatility whipsaws).
    """
    close = prices["adjusted_close"]
    ret_lookback = close / close.shift(lookback) - 1.0
    
    daily_rets = close.pct_change()
    vol = daily_rets.rolling(window=vol_window).std()
    # 252-day rolling percentile benchmark
    vol_bench = vol.rolling(window=252, min_periods=60).quantile(vol_percentile / 100.0)
    
    mom_ok = ret_lookback > 0.0
    vol_ok = vol <= vol_bench
    
    sig = (mom_ok & vol_ok).astype(np.float64)
    return sig.fillna(0.0)
