"""Tests for dated transaction cost model, execution engine, and bias invariants."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from overfit_radar.config import load_costs_config
from overfit_radar.costs.cost_model import TransactionCostModel
from overfit_radar.costs.engine import BacktestEngine


def test_cost_regime_lookup():
    """Verify that transaction cost rates adapt accurately to trade execution date."""
    cost_config = load_costs_config()
    model = TransactionCostModel(cost_config)
    
    # 2015 trade: should use 2010 regime (NSE fee 0.00345%)
    rates_2015 = model.get_rates_for_date("2015-06-15")
    assert rates_2015.exchange_txn_fee == pytest.approx(0.0000345)
    
    # Late 2024 trade: should use revised Oct 2024 uniform regime (NSE fee 0.00297%)
    rates_2024 = model.get_rates_for_date("2024-11-01")
    assert rates_2024.exchange_txn_fee == pytest.approx(0.0000297)


def test_cost_monotonicity(synthetic_gbm_prices):
    """
    REQUIRED TEST 3: Cost Monotonicity.
    Net returns with cost_multiplier=2.0 must be <= net returns with cost_multiplier=1.0,
    and net returns must be <= gross returns on every single day.
    """
    df = synthetic_gbm_prices(n_days=300, seed=42)
    cost_config = load_costs_config()
    
    # Create alternating signal (active trading)
    # 1 for 5 days, 0 for 5 days
    signal = pd.Series(0.0, index=df.index)
    for i in range(len(signal)):
        signal.iloc[i] = 1.0 if (i // 5) % 2 == 1 else 0.0

    engine_gross = BacktestEngine(cost_config, cost_multiplier=0.0)
    engine_1x = BacktestEngine(cost_config, cost_multiplier=1.0)
    engine_2x = BacktestEngine(cost_config, cost_multiplier=2.0)
    
    res_gross = engine_gross.run(df, signal)
    res_1x = engine_1x.run(df, signal)
    res_2x = engine_2x.run(df, signal)
    
    # 1. Day-by-day return ordering
    assert (res_gross["net_return"] >= res_1x["net_return"]).all(), "Gross returns must be >= 1x net returns"
    assert (res_1x["net_return"] >= res_2x["net_return"]).all(), "1x net returns must be >= 2x net returns"
    
    # 2. Cumulative return ordering
    cum_gross = (1.0 + res_gross["net_return"]).prod()
    cum_1x = (1.0 + res_1x["net_return"]).prod()
    cum_2x = (1.0 + res_2x["net_return"]).prod()
    
    assert cum_gross >= cum_1x >= cum_2x, f"Expected cum_gross >= cum_1x >= cum_2x, got {cum_gross:.4f}, {cum_1x:.4f}, {cum_2x:.4f}"
    assert cum_1x < cum_gross, "Costs must cause a non-zero drag over 300 days of active trading"


def test_no_lookahead_bias(synthetic_gbm_prices):
    """
    REQUIRED TEST 1: No-Lookahead Bias.
    Changing future prices at t >= 50 MUST NOT change signals or returns at t < 50.
    """
    df1 = synthetic_gbm_prices(n_days=100, seed=42)
    
    # Define a simple moving average crossover signal logic
    def compute_sma_signal(prices: pd.DataFrame) -> pd.Series:
        fast_ma = prices["close"].rolling(window=5).mean()
        slow_ma = prices["close"].rolling(window=15).mean()
        return (fast_ma > slow_ma).astype(float)
        
    signal1 = compute_sma_signal(df1)
    
    cost_config = load_costs_config()
    engine = BacktestEngine(cost_config, cost_multiplier=1.0)
    res1 = engine.run(df1, signal1)
    
    # Now create df2 where future prices (bars 50 to 100) are drastically altered (e.g. 10x boom or crash)
    df2 = df1.copy()
    df2.iloc[50:, df2.columns.get_loc("close")] *= 5.0
    df2.iloc[50:, df2.columns.get_loc("open")] *= 5.0
    df2.iloc[50:, df2.columns.get_loc("high")] *= 5.0
    df2.iloc[50:, df2.columns.get_loc("low")] *= 5.0
    
    signal2 = compute_sma_signal(df2)
    res2 = engine.run(df2, signal2)
    
    # Signals and returns before day 50 must be BIT-FOR-BIT IDENTICAL
    pd.testing.assert_series_equal(
        signal1.iloc[:50],
        signal2.iloc[:50],
        check_names=False,
    )
    pd.testing.assert_series_equal(
        res1["net_return"].iloc[:50],
        res2["net_return"].iloc[:50],
        check_names=False,
    )
