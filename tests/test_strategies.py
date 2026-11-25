"""Unit tests for strategy families, parameter grid expansion, and batch execution."""

from __future__ import annotations

import time
import numpy as np
import pandas as pd
import pytest

from overfit_radar.config import load_costs_config
from overfit_radar.costs.engine import BacktestEngine
from overfit_radar.strategies.families import (
    generate_sma_crossover,
    generate_momentum,
    generate_rsi_mean_reversion,
    generate_bollinger_mean_reversion,
    generate_donchian_breakout,
    generate_vol_filtered_momentum,
)
from overfit_radar.strategies.zoo import build_strategy_zoo, run_strategy_zoo


def test_strategy_signal_invariants(synthetic_gbm_prices):
    """Ensure signals are strictly binary {0.0, 1.0} and free of NaNs."""
    df = synthetic_gbm_prices(n_days=300, seed=42)
    
    signals = [
        generate_sma_crossover(df, fast_window=10, slow_window=30),
        generate_momentum(df, lookback=21, holding_period=5),
        generate_rsi_mean_reversion(df, window=14, oversold=30, overbought=70),
        generate_bollinger_mean_reversion(df, window=20, num_std=2.0, exit_at_mid=True),
        generate_donchian_breakout(df, lookback_high=20, exit_fraction=0.5),
        generate_vol_filtered_momentum(df, lookback=21, vol_window=21, vol_percentile=50),
    ]
    
    for sig in signals:
        assert isinstance(sig, pd.Series)
        assert len(sig) == len(df)
        assert not sig.isna().any(), "Signals must not contain NaNs"
        unique_vals = set(sig.unique())
        assert unique_vals.issubset({0.0, 1.0}), f"Signal values must be 0.0 or 1.0, got {unique_vals}"


def test_build_strategy_zoo_grid():
    """Ensure parameter grid generation produces ~500 valid, unique configurations."""
    zoo = build_strategy_zoo()
    
    # Must have ~500 configs (allow 400 - 650 range)
    assert 400 <= len(zoo) <= 650, f"Expected ~500 configs, got {len(zoo)}"
    
    # Check uniqueness of strategy IDs
    ids = [s.strategy_id for s in zoo]
    assert len(ids) == len(set(ids)), "Strategy IDs must be unique"
    
    # Verify SMA constraint: fast_window < slow_window
    sma_configs = [s for s in zoo if s.family == "sma_crossover"]
    for s in sma_configs:
        assert s.params["fast_window"] < s.params["slow_window"]


def test_batch_execution_performance(synthetic_gbm_prices):
    """
    Ensure the zoo runs efficiently on 500 strategies over 1,000 bars
    within a few seconds on CPU (vectorized float32/float64 execution).
    """
    df = synthetic_gbm_prices(n_days=1000, seed=42)
    costs_config = load_costs_config()
    engine = BacktestEngine(costs_config, cost_multiplier=1.0)
    
    zoo = build_strategy_zoo()
    
    t0 = time.time()
    return_matrix = run_strategy_zoo(df, zoo, engine)
    elapsed = time.time() - t0
    
    assert return_matrix.shape == (len(df), len(zoo))
    assert not return_matrix.isna().any().any(), "Return matrix must have zero NaNs"
    assert elapsed < 10.0, f"Batch execution too slow: {elapsed:.2f}s for {len(zoo)} strategies"
