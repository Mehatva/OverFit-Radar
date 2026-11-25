"""In-sample feature extraction module."""

from __future__ import annotations

from typing import Dict, List, Optional
import numpy as np
import pandas as pd

from overfit_radar.config import CostsConfig
from overfit_radar.features.neighbourhood import build_neighbourhood_graph, compute_neighbourhood_metrics
from overfit_radar.stats.basic import (
    calculate_sharpe,
    calculate_sortino,
    calculate_max_drawdown,
    calculate_skewness,
    calculate_kurtosis,
    calculate_autocorrelation,
)
from overfit_radar.stats.psr_dsr import (
    compute_psr,
    compute_dsr,
    compute_effective_n,
)
from overfit_radar.strategies.base import StrategyConfig


def extract_window_features(
    returns_is: pd.DataFrame,
    zoo: List[StrategyConfig],
    costs_config: CostsConfig,
    neighbourhood_graph: Optional[Dict[str, List[str]]] = None,
) -> pd.DataFrame:
    """
    Extract comprehensive In-Sample (IS) features for all strategies in the zoo.
    
    CRITICAL INVARIANT:
    All inputs and calculations use STRICTLY in-sample bars (returns_is).
    Zero future information from Out-of-Sample is accessed.
    
    Features computed:
    - In-sample Sharpe (annualized & per-period)
    - In-sample Sortino
    - Max drawdown
    - Skewness & non-excess kurtosis
    - Return autocorrelation (lag-1)
    - Probabilistic Sharpe Ratio (PSR)
    - Deflated Sharpe Ratio (DSR with raw N & effective N)
    - Parameter-Neighbourhood Stability (mean, std, and spike ratio)
    """
    t_obs, n_strats = returns_is.shape
    if t_obs < 10:
        raise ValueError(f"In-sample returns length ({t_obs}) too short for feature extraction.")

    if neighbourhood_graph is None:
        neighbourhood_graph = build_neighbourhood_graph(zoo)

    # 1. First pass: compute basic statistics and per-period Sharpes across all strategies
    strat_map = {s.strategy_id: s for s in zoo}
    sharpe_daily = {}
    sharpe_ann = {}
    sortino_ann = {}
    max_dd = {}
    skew_vals = {}
    kurt_vals = {}
    autocorr_vals = {}
    psr_vals = {}

    for strat_id in returns_is.columns:
        rets = returns_is[strat_id]
        sr_d = calculate_sharpe(rets, annualize=False)
        sr_a = calculate_sharpe(rets, annualize=True)
        so_a = calculate_sortino(rets, annualize=True)
        mdd = calculate_max_drawdown(rets)
        sk = calculate_skewness(rets)
        kt = calculate_kurtosis(rets)
        ac = calculate_autocorrelation(rets, lag=1)
        psr = compute_psr(sharpe_hat=sr_d, benchmark_sr=0.0, t_periods=t_obs, skew=sk, kurtosis=kt)

        sharpe_daily[strat_id] = sr_d
        sharpe_ann[strat_id] = sr_a
        sortino_ann[strat_id] = so_a
        max_dd[strat_id] = mdd
        skew_vals[strat_id] = sk
        kurt_vals[strat_id] = kt
        autocorr_vals[strat_id] = ac
        psr_vals[strat_id] = psr

    # 2. Cross-sectional metrics across trials (Variance of daily Sharpes, Effective N)
    daily_sr_array = np.array(list(sharpe_daily.values()))
    var_sharpe = float(np.var(daily_sr_array, ddof=1)) if len(daily_sr_array) > 1 else 0.0
    raw_n = float(n_strats)
    eff_n = compute_effective_n(returns_is, method="cluster")

    # 3. Second pass: DSR and Parameter-Neighbourhood Stability
    feature_rows = []
    family_counts = {s.family: 0 for s in zoo}
    for s in zoo:
        family_counts[s.family] += 1

    for strat in zoo:
        sid = strat.strategy_id
        if sid not in returns_is.columns:
            continue

        sr_d = sharpe_daily[sid]
        sr_a = sharpe_ann[sid]
        sk = skew_vals[sid]
        kt = kurt_vals[sid]

        dsr_raw = compute_dsr(
            sharpe_hat=sr_d,
            var_sharpe=var_sharpe,
            n_trials=raw_n,
            t_periods=t_obs,
            skew=sk,
            kurtosis=kt,
        )
        dsr_eff = compute_dsr(
            sharpe_hat=sr_d,
            var_sharpe=var_sharpe,
            n_trials=eff_n,
            t_periods=t_obs,
            skew=sk,
            kurtosis=kt,
        )

        # Neighbourhood stability
        neigh_mean, neigh_std = compute_neighbourhood_metrics(sid, sharpe_ann, neighbourhood_graph)
        # Spike indicator: excess of candidate Sharpe over local neighbourhood mean
        spike_excess = sr_a - neigh_mean

        feature_rows.append(
            {
                "strategy_id": sid,
                "family": strat.family,
                "is_sharpe": sr_a,
                "is_sharpe_per_period": sr_d,
                "is_sortino": sortino_ann[sid],
                "is_max_drawdown": max_dd[sid],
                "is_skewness": sk,
                "is_kurtosis": kt,
                "is_autocorr": autocorr_vals[sid],
                "is_psr": psr_vals[sid],
                "is_dsr_raw": dsr_raw,
                "is_dsr_eff": dsr_eff,
                "effective_n": eff_n,
                "neighbourhood_sharpe_mean": neigh_mean,
                "neighbourhood_sharpe_std": neigh_std,
                "neighbourhood_spike_excess": spike_excess,
                "family_trials_count": family_counts[strat.family],
            }
        )

    df_feats = pd.DataFrame(feature_rows)
    df_feats = df_feats.set_index("strategy_id")
    return df_feats
