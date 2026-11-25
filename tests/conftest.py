"""Shared test fixtures for Overfit Radar."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def synthetic_gbm_prices():
    """
    Generate synthetic daily OHLCV prices using Geometric Brownian Motion (GBM).
    Deterministic via fixed seed to ensure tests are reproducible and independent of network.
    """
    def _generator(
        n_days: int = 1000,
        start_date: str = "2020-01-01",
        start_price: float = 100.0,
        mu: float = 0.08,
        sigma: float = 0.20,
        seed: int = 42,
    ) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        dt = 1.0 / 252.0
        
        # Daily drift and shock
        returns = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * rng.standard_normal(n_days)
        price_multipliers = np.exp(returns)
        
        close = np.empty(n_days, dtype=np.float64)
        close[0] = start_price
        for i in range(1, n_days):
            close[i] = close[i - 1] * price_multipliers[i]
            
        # Synthesize realistic Open, High, Low around Close
        intraday_noise = rng.uniform(0.002, 0.015, size=(n_days, 3))
        open_ = np.roll(close, 1)
        open_[0] = start_price
        high = np.maximum(open_, close) * (1.0 + intraday_noise[:, 0])
        low = np.minimum(open_, close) * (1.0 - intraday_noise[:, 1])
        volume = rng.integers(100_000, 5_000_000, size=n_days).astype(np.float64)
        
        dates = pd.bdate_range(start=start_date, periods=n_days, freq="B")
        
        df = pd.DataFrame(
            {
                "open": open_,
                "high": high,
                "low": low,
                "close": close,
                "adjusted_close": close.copy(),
                "volume": volume,
            },
            index=dates,
        )
        df.index.name = "date"
        return df

    return _generator


@pytest.fixture
def random_walk_returns():
    """Return zero-drift Gaussian daily returns (pure random walk / martingales)."""
    def _returns(n_days: int = 1000, n_series: int = 50, seed: int = 42) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        daily_vol = 0.20 / np.sqrt(252)
        rets = rng.normal(loc=0.0, scale=daily_vol, size=(n_days, n_series))
        dates = pd.bdate_range(start="2020-01-01", periods=n_days, freq="B")
        return pd.DataFrame(rets, index=dates, columns=[f"strat_{i}" for i in range(n_series)])
    return _returns
