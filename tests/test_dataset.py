"""Unit tests for rolling walk-forward windows, feature extraction, and leakage prevention."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from overfit_radar.config import load_costs_config, load_experiment_config
from overfit_radar.costs.engine import BacktestEngine
from overfit_radar.features.windows import generate_walk_forward_windows, WindowSlice
from overfit_radar.features.neighbourhood import build_neighbourhood_graph, compute_neighbourhood_metrics
from overfit_radar.features.extractor import extract_window_features
from overfit_radar.features.dataset import build_strategy_dataset
from overfit_radar.strategies.zoo import build_strategy_zoo, run_strategy_zoo


def test_walk_forward_window_integrity():
    """
    REQUIRED TEST 6: Split Integrity & Embargo.
    Ensure that:
    1. IS and OOS never overlap.
    2. Embargo gap between IS end and OOS start is strictly respected.
    3. Windows advance chronologically.
    """
    dates = pd.bdate_range(start="2010-01-01", end="2020-01-01", freq="B")
    windows = generate_walk_forward_windows(
        dates=dates,
        is_months=36,
        oos_months=12,
        step_months=6,
        embargo_days=21,
    )
    
    assert len(windows) >= 10, f"Expected at least 10 rolling windows, got {len(windows)}"
    
    for w in windows:
        assert isinstance(w, WindowSlice)
        # 1. No overlap
        assert w.is_end_idx < w.oos_start_idx, f"Window {w.window_id}: IS overlaps with OOS!"
        
        # 2. Embargo gap in business days
        gap_days = (dates[w.oos_start_idx] - dates[w.is_end_idx]).days
        assert gap_days >= 21, f"Window {w.window_id}: Embargo gap too small ({gap_days} calendar days)"
        
        # 3. Length sanity
        is_bars = w.is_end_idx - w.is_start_idx + 1
        oos_bars = w.oos_end_idx - w.oos_start_idx + 1
        assert 700 <= is_bars <= 800, f"Expected ~756 bars in 36m IS, got {is_bars}"
        assert 230 <= oos_bars <= 280, f"Expected ~252 bars in 12m OOS, got {oos_bars}"


def test_no_feature_leakage_from_oos(synthetic_gbm_prices):
    """
    CRITICAL LEAKAGE TEST:
    Modifying Out-of-Sample prices or returns MUST produce EXACTLY identical In-Sample features.
    """
    df1 = synthetic_gbm_prices(n_days=1100, seed=42)
    zoo = build_strategy_zoo()[:10]  # Subset of 10 strategies
    costs_config = load_costs_config()
    engine = BacktestEngine(costs_config, cost_multiplier=1.0)
    
    returns1 = run_strategy_zoo(df1, zoo, engine)
    
    # Take Window 0: IS [0..756], Embargo [757..777], OOS [778..1030]
    window = WindowSlice(
        window_id="test_win",
        is_start_idx=0,
        is_end_idx=755,
        oos_start_idx=777,
        oos_end_idx=1028,
    )
    
    features1 = extract_window_features(
        returns_is=returns1.iloc[window.is_start_idx : window.is_end_idx + 1],
        zoo=zoo,
        costs_config=costs_config,
    )
    
    # Now create returns2 where OOS returns (bars 777 to 1100) are wildly altered (10x noise)
    returns2 = returns1.copy()
    returns2.iloc[777:, :] = np.random.default_rng(999).normal(0.05, 0.10, size=returns2.iloc[777:, :].shape)
    
    features2 = extract_window_features(
        returns_is=returns2.iloc[window.is_start_idx : window.is_end_idx + 1],
        zoo=zoo,
        costs_config=costs_config,
    )
    
    # In-sample features must be bit-for-bit identical!
    pd.testing.assert_frame_equal(features1, features2)


def test_neighbourhood_stability_plateau_vs_spike():
    """
    KEY ORIGINAL FEATURE TEST:
    Verify that a plateau (high mean neighbour Sharpe, low std)
    is cleanly distinguished from a spike (low mean neighbour Sharpe, high std).
    """
    zoo = build_strategy_zoo()
    graph = build_neighbourhood_graph(zoo)
    
    # Pick a candidate strategy from SMA crossover
    cand_strat = [s for s in zoo if s.strategy_id == "sma_f10_s50"][0]
    neighbours = graph[cand_strat.strategy_id]
    assert len(neighbours) >= 2, "Candidate must have at least 2 adjacent grid neighbours"
    
    # Case A: Plateau landscape (candidate=1.5, neighbours all ~1.45 - 1.55)
    plateau_sharpes = {s_id: 1.50 for s_id in [cand_strat.strategy_id] + neighbours}
    p_mean, p_std = compute_neighbourhood_metrics(cand_strat.strategy_id, plateau_sharpes, graph)
    assert p_mean == pytest.approx(1.50)
    assert p_std == pytest.approx(0.0)
    
    # Case B: Spike landscape (candidate=1.5, neighbours all collapse to 0.1)
    spike_sharpes = {cand_strat.strategy_id: 1.50}
    for n_id in neighbours:
        spike_sharpes[n_id] = 0.10
    s_mean, s_std = compute_neighbourhood_metrics(cand_strat.strategy_id, spike_sharpes, graph)
    assert s_mean < 0.50
    assert s_std > 0.0, "Spike must exhibit high neighbourhood dispersion"


def test_build_strategy_dataset_end_to_end(synthetic_gbm_prices):
    """
    Verify end-to-end dataset generation on synthetic data:
    Check schema, columns, non-empty rows, and absence of NaNs.
    """
    df_price = synthetic_gbm_prices(n_days=1100, seed=42)
    universe_data = {"TEST_STOCK.NS": df_price}
    zoo = build_strategy_zoo()[:15]  # 15 strategies for fast unit test
    
    dataset = build_strategy_dataset(universe_data=universe_data, zoo=zoo)
    
    assert not dataset.empty
    assert "strategy_id" in dataset.columns
    assert "symbol" in dataset.columns
    assert "window_id" in dataset.columns
    assert "is_sharpe" in dataset.columns
    assert "oos_sharpe" in dataset.columns
    assert "survived" in dataset.columns
    assert "neighbourhood_sharpe_mean" in dataset.columns
    assert "neighbourhood_sharpe_std" in dataset.columns
    
    # Check binary label
    assert set(dataset["survived"].unique()).issubset({0, 1})
    # Check no NaNs in features
    assert not dataset[["is_sharpe", "is_dsr_raw", "neighbourhood_sharpe_mean", "oos_sharpe"]].isna().any().any()
