"""Strategy zoo generator and batch execution coordinator."""

from __future__ import annotations

import logging
from pathlib import Path
from overfit_radar.config import get_project_root, load_yaml
from typing import Dict, List, Optional
import numpy as np
import pandas as pd

from overfit_radar.costs.engine import BacktestEngine
from overfit_radar.strategies.base import StrategyConfig
from overfit_radar.strategies.families import (
    generate_sma_crossover,
    generate_momentum,
    generate_rsi_mean_reversion,
    generate_bollinger_mean_reversion,
    generate_donchian_breakout,
    generate_vol_filtered_momentum,
)

logger = logging.getLogger(__name__)


def build_strategy_zoo(config_dir: Optional[Path] = None) -> List[StrategyConfig]:
    """
    Build a dense, reproducible parameter grid of ~500 trading strategy configurations.
    
    The density of the parameter space is critical for measuring 'neighbourhood stability'
    (identifying whether an in-sample Sharpe ratio sits on a broad parameter plateau
    or a fragile, overfit spike).
    """
    grids = load_yaml((config_dir or get_project_root() / "configs") / "strategies.yaml")["families"]
    zoo: List[StrategyConfig] = []

    # 1. SMA Crossover Family (~50 configs)
    fast_windows = grids["sma_crossover"]["fast_windows"]
    slow_windows = grids["sma_crossover"]["slow_windows"]
    for fw in fast_windows:
        for sw in slow_windows:
            if fw < sw:
                strat_id = f"sma_f{fw}_s{sw}"
                zoo.append(
                    StrategyConfig(
                        strategy_id=strat_id,
                        family="sma_crossover",
                        params={"fast_window": fw, "slow_window": sw},
                    )
                )

    # 2. Time-Series Momentum Family (70 configs)
    mom_lookbacks = grids["time_series_momentum"]["lookbacks"]
    mom_holdings = grids["time_series_momentum"]["holding_periods"]
    for lb in mom_lookbacks:
        for h in mom_holdings:
            strat_id = f"mom_lb{lb}_h{h}"
            zoo.append(
                StrategyConfig(
                    strategy_id=strat_id,
                    family="time_series_momentum",
                    params={"lookback": lb, "holding_period": h},
                )
            )

    # 3. RSI Mean Reversion Family (80 configs)
    rsi_windows = grids["rsi_mean_reversion"]["windows"]
    rsi_oversolds = grids["rsi_mean_reversion"]["oversold_levels"]
    rsi_overboughts = grids["rsi_mean_reversion"]["overbought_levels"]
    for w in rsi_windows:
        for os in rsi_oversolds:
            for ob in rsi_overboughts:
                strat_id = f"rsi_w{w}_os{os}_ob{ob}"
                zoo.append(
                    StrategyConfig(
                        strategy_id=strat_id,
                        family="rsi_mean_reversion",
                        params={"window": w, "oversold": os, "overbought": ob},
                    )
                )

    # 4. Bollinger Bands Mean Reversion Family (84 configs)
    bb_windows = grids["bollinger_mean_reversion"]["windows"]
    bb_stds = grids["bollinger_mean_reversion"]["num_stds"]
    bb_exits = grids["bollinger_mean_reversion"]["exit_at_mid"]
    for w in bb_windows:
        for std in bb_stds:
            for exit_mid in bb_exits:
                strat_id = f"bb_w{w}_std{std}_mid{exit_mid}"
                zoo.append(
                    StrategyConfig(
                        strategy_id=strat_id,
                        family="bollinger_mean_reversion",
                        params={"window": w, "num_std": std, "exit_at_mid": exit_mid},
                    )
                )

    # 5. Donchian Breakout Family (60 configs)
    donch_highs = grids["donchian_breakout"]["lookback_high"]
    donch_fractions = grids["donchian_breakout"]["exit_fraction"]
    for h in donch_highs:
        for frac in donch_fractions:
            strat_id = f"donch_h{h}_ef{frac}"
            zoo.append(
                StrategyConfig(
                    strategy_id=strat_id,
                    family="donchian_breakout",
                    params={"lookback_high": h, "exit_fraction": frac},
                )
            )

    # 6. Volatility-Filtered Momentum Family (140 configs)
    v_lookbacks = grids["volatility_filtered_momentum"]["lookbacks"]
    v_windows = grids["volatility_filtered_momentum"]["vol_window"]
    v_percentiles = grids["volatility_filtered_momentum"]["vol_percentile_cutoff"]
    for lb in v_lookbacks:
        for vw in v_windows:
            for vp in v_percentiles:
                strat_id = f"volmom_lb{lb}_vw{vw}_vp{vp}"
                zoo.append(
                    StrategyConfig(
                        strategy_id=strat_id,
                        family="volatility_filtered_momentum",
                        params={"lookback": lb, "vol_window": vw, "vol_percentile": vp},
                    )
                )

    return zoo


def compute_strategy_signal(prices: pd.DataFrame, strat: StrategyConfig) -> pd.Series:
    """Dispatch to the corresponding strategy family signal generator."""
    fam = strat.family
    p = strat.params
    if fam == "sma_crossover":
        return generate_sma_crossover(prices, fast_window=p["fast_window"], slow_window=p["slow_window"])
    elif fam == "time_series_momentum":
        return generate_momentum(prices, lookback=p["lookback"], holding_period=p["holding_period"])
    elif fam == "rsi_mean_reversion":
        return generate_rsi_mean_reversion(prices, window=p["window"], oversold=p["oversold"], overbought=p["overbought"])
    elif fam == "bollinger_mean_reversion":
        return generate_bollinger_mean_reversion(prices, window=p["window"], num_std=p["num_std"], exit_at_mid=p["exit_at_mid"])
    elif fam == "donchian_breakout":
        return generate_donchian_breakout(prices, lookback_high=p["lookback_high"], exit_fraction=p["exit_fraction"])
    elif fam == "volatility_filtered_momentum":
        return generate_vol_filtered_momentum(prices, lookback=p["lookback"], vol_window=p["vol_window"], vol_percentile=p["vol_percentile"])
    else:
        raise ValueError(f"Unknown strategy family: {fam}")


def run_strategy_zoo(
    prices: pd.DataFrame,
    zoo: List[StrategyConfig],
    engine: BacktestEngine,
) -> pd.DataFrame:
    """
    Execute all strategies in the zoo against the given price history.
    
    Returns:
        DataFrame of shape (T, N_strategies) where each column represents
        the net daily returns of a strategy configuration.
    """
    n_days = len(prices)
    n_strats = len(zoo)
    
    returns_matrix = np.empty((n_days, n_strats), dtype=np.float64)
    strat_ids = [s.strategy_id for s in zoo]
    
    for j, strat in enumerate(zoo):
        sig = compute_strategy_signal(prices, strat)
        res = engine.run(prices, sig)
        returns_matrix[:, j] = res["net_return"].to_numpy()
        
    return pd.DataFrame(returns_matrix, index=prices.index, columns=strat_ids)
