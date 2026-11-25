"""Dated Indian cash equity transaction cost model."""

from __future__ import annotations

from typing import Union
import pandas as pd
from overfit_radar.config import CostRegime, CostsConfig


class TransactionCostModel:
    """
    Computes date-dependent transaction costs based on historical Indian tax regimes.
    
    Includes STT, stamp duty, NSE turnover fee, SEBI charges, GST (18%), and slippage.
    """

    def __init__(self, config: CostsConfig):
        if not config.regimes:
            raise ValueError("At least one dated cost regime is required")
        self.config = config
        # Sort regimes chronologically
        self.regimes = sorted(config.regimes, key=lambda r: r.effective_date)
        self.regime_dates = [pd.to_datetime(r.effective_date) for r in self.regimes]

    def get_rates_for_date(self, date: Union[str, pd.Timestamp]) -> CostRegime:
        """Find the active cost regime for a given trade date."""
        ts = pd.to_datetime(date)
        if pd.isna(ts) or ts < self.regime_dates[0]:
            raise ValueError("Trade date predates the cost schedule or is missing")
        # Find the latest regime whose effective_date <= ts
        active_regime = self.regimes[0]
        for reg, r_date in zip(self.regimes, self.regime_dates):
            if ts >= r_date:
                active_regime = reg
            else:
                break
        return active_regime

    def compute_cost_rate(
        self,
        date: Union[str, pd.Timestamp],
        is_buy: bool,
        slippage_bps: float,
        cost_multiplier: float = 1.0,
    ) -> float:
        """
        Compute total fractional transaction cost (as a decimal, e.g. 0.0015 = 15 bps).
        
        Args:
            date: Trade execution date.
            is_buy: True for purchase, False for sale.
            slippage_bps: Half-spread + impact in basis points.
            cost_multiplier: Multiplier for stress testing (e.g. 1.0x, 2.0x).
            
        Returns:
            Total fractional cost rate to deduct from trade return.
        """
        regime = self.get_rates_for_date(date)
        slippage_rate = slippage_bps / 10_000.0

        if is_buy:
            stt = regime.stt_delivery_buy
            stamp_duty = regime.stamp_duty_buy
        else:
            stt = regime.stt_delivery_sell
            stamp_duty = regime.stamp_duty_sell

        # Statutory charges subject to GST (Brokerage + Exchange turnover + SEBI turnover)
        statutory_base = regime.brokerage_delivery + regime.exchange_txn_fee + regime.sebi_turnover_fee
        gst = regime.gst_rate * statutory_base

        total_statutory = stt + stamp_duty + statutory_base + gst
        total_drag = (total_statutory + slippage_rate) * cost_multiplier
        return float(total_drag)
