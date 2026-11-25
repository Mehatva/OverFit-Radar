"""Unit tests for synthetic price and return fixtures."""

from __future__ import annotations

import numpy as np
import pandas as pd


def test_synthetic_gbm_prices_fixture(synthetic_gbm_prices):
    df = synthetic_gbm_prices(n_days=250, seed=42)
    assert len(df) == 250
    assert list(df.columns) == ["open", "high", "low", "close", "adjusted_close", "volume"]
    assert df.index.is_monotonic_increasing
    assert (df["high"] >= df["low"]).all()
    assert (df["volume"] > 0).all()
    
    # Reproducibility check with identical seed
    df2 = synthetic_gbm_prices(n_days=250, seed=42)
    pd.testing.assert_frame_equal(df, df2)
    
    # Different seed should yield different prices
    df3 = synthetic_gbm_prices(n_days=250, seed=99)
    assert not np.allclose(df["close"].values, df3["close"].values)


def test_random_walk_returns_fixture(random_walk_returns):
    rets = random_walk_returns(n_days=500, n_series=10, seed=42)
    assert rets.shape == (500, 10)
    assert rets.index.is_monotonic_increasing
    
    # Check that mean return is close to zero (martingale sanity)
    means = rets.mean()
    assert (np.abs(means) < 0.01).all()
