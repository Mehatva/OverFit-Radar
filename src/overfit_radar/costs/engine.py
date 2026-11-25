"""Vectorized backtesting and execution engine with realistic t+1 fill timing."""

from __future__ import annotations

from typing import Optional
import numpy as np
import pandas as pd
from overfit_radar.config import CostsConfig
from overfit_radar.costs.cost_model import TransactionCostModel


class BacktestEngine:
    """
    Vectorized long-or-flat backtest execution engine.
    
    Strict execution rule:
    Signals are generated at day t Close.
    Orders execute at day t+1 Open (shift by 1 trading day).
    This strictly prevents close-to-close lookahead bias and accurately accounts for overnight gaps.
    """

    def __init__(self, costs_config: CostsConfig, cost_multiplier: float = 1.0,
                 dp_trade_notional: Optional[float] = None):
        """Exclude DP fees by default; optionally model fixed rupee notional per sell.

        This sensitivity uses fee / sell notional, not a portfolio capital or
        share-quantity simulation. It assumes one symbol/order per sell event.
        """
        if not np.isfinite(cost_multiplier) or cost_multiplier < 0:
            raise ValueError("cost_multiplier must be finite and nonnegative")
        if dp_trade_notional is not None and (not np.isfinite(dp_trade_notional) or dp_trade_notional <= 0):
            raise ValueError("dp_trade_notional must be finite and positive")
        self.dp_trade_notional = dp_trade_notional
        self.costs_config = costs_config
        self.cost_multiplier = cost_multiplier
        self.cost_model = TransactionCostModel(costs_config)
        self.slippage_bps = costs_config.default_slippage_bps

    def run(self, prices: pd.DataFrame, signal: pd.Series) -> pd.DataFrame:
        """
        Execute backtest from prices and daily binary signal.
        
        Args:
            prices: DataFrame with DatetimeIndex and columns ['open', 'close', 'adjusted_close'].
            signal: Series of 0.0 (flat) or 1.0 (long), computed at day t close.
            
        Returns:
            DataFrame with columns:
                - signal: Raw signal at day t close
                - position: Active position on day t
                - gross_return: Strategy gross daily return
                - cost: Total friction drag applied
                - net_return: Strategy net daily return after costs
                - trade_action: 1 (buy), -1 (sell), 0 (none)
        """
        # Align index
        common_idx = prices.index.intersection(signal.index)
        p = prices.loc[common_idx]
        s = signal.loc[common_idx].astype(np.float64)

        n = len(p)
        if n < 2:
            raise ValueError("Price series must have at least 2 bars to execute backtest.")

        # Compute dividend/split adjusted Open and Close
        # adj_factor scales Open proportionally to Adjusted Close
        adj_factor = p["adjusted_close"] / p["close"]
        adj_open = p["open"] * adj_factor
        adj_close = p["adjusted_close"]

        # Shift signal by 1 day: signal at day t close -> target position for day t+1
        # Position at index 0 is 0.0 (uninvested start)
        target_pos = s.shift(1).fillna(0.0)

        # Detect trades:
        # trade_action = target_pos - previous_target_pos
        # +1 = Buy at Open, -1 = Sell at Open, 0 = Hold or Flat
        prev_target_pos = target_pos.shift(1).fillna(0.0)
        trade_action = target_pos - prev_target_pos

        # Compute Gross Returns for each regime:
        # 1. New Buy (trade_action == 1): Buy at day t Open, hold to day t Close
        #    return = (Close_t - Open_t) / Open_t
        # 2. Existing Hold (target_pos == 1 and prev_target_pos == 1):
        #    return = (Close_t - Close_{t-1}) / Close_{t-1}
        # 3. Sell / Exit (trade_action == -1): Held overnight from Close_{t-1}, sell at day t Open
        #    return = (Open_t - Close_{t-1}) / Close_{t-1}
        # 4. Flat (target_pos == 0 and prev_target_pos == 0):
        #    return 0.0

        close_prev = adj_close.shift(1).fillna(adj_close.iloc[0])

        ret_buy = (adj_close - adj_open) / adj_open
        ret_hold = (adj_close - close_prev) / close_prev
        ret_sell = (adj_open - close_prev) / close_prev

        gross_return = np.where(
            trade_action == 1.0,
            ret_buy,
            np.where(
                (target_pos == 1.0) & (prev_target_pos == 1.0),
                ret_hold,
                np.where(
                    trade_action == -1.0,
                    ret_sell,
                    0.0,
                ),
            ),
        )
        gross_return = pd.Series(gross_return, index=common_idx, dtype=np.float64)

        # Calculate transaction costs on trade days
        costs = np.zeros(n, dtype=np.float64)
        if self.cost_multiplier > 0.0:
            for i in range(n):
                act = trade_action.iloc[i]
                if act == 1.0:  # Buy trade executed at open of day i
                    trade_date = common_idx[i]
                    costs[i] = self.cost_model.compute_cost_rate(
                        date=trade_date,
                        is_buy=True,
                        slippage_bps=self.slippage_bps,
                        cost_multiplier=self.cost_multiplier,
                    )
                elif act == -1.0:  # Sell trade executed at open of day i
                    trade_date = common_idx[i]
                    costs[i] = self.cost_model.compute_cost_rate(
                        date=trade_date,
                        is_buy=False,
                        slippage_bps=self.slippage_bps,
                        cost_multiplier=self.cost_multiplier,
                    )

        if self.dp_trade_notional is not None and self.cost_multiplier > 0:
            for i in np.flatnonzero(trade_action.to_numpy() == -1.0):
                fee = self.cost_model.get_rates_for_date(common_idx[i]).dp_charges_per_sell
                costs[i] += fee / self.dp_trade_notional * self.cost_multiplier

        cost_series = pd.Series(costs, index=common_idx, dtype=np.float64)
        net_return = gross_return - cost_series

        return pd.DataFrame(
            {
                "signal": s,
                "position": target_pos,
                "gross_return": gross_return,
                "cost": cost_series,
                "net_return": net_return,
                "trade_action": trade_action,
            },
            index=common_idx,
        )
