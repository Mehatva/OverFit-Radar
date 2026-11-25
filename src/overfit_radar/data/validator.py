"""Data validation module to ensure price series integrity."""

from __future__ import annotations

import logging
from typing import List
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS: List[str] = [
    "open",
    "high",
    "low",
    "close",
    "adjusted_close",
    "volume",
]


def validate_ohlcv(df: pd.DataFrame, symbol: str) -> None:
    """
    Validate that a price dataframe satisfies all standard quantitative integrity rules.
    
    Checks:
    1. Non-empty DataFrame.
    2. DatetimeIndex is strictly increasing and monotonic.
    3. Contains all required OHLCV columns.
    4. No NaN values in price or volume columns.
    5. High is greater than or equal to Low on every bar.
    6. Volume is non-negative.
    """
    if df is None or df.empty:
        raise ValueError(f"Price dataframe for {symbol} is empty.")

    # 1. Check required columns
    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Dataframe for {symbol} is missing required columns: {missing_cols}")

    # 2. Check DatetimeIndex
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError(f"Dataframe for {symbol} must have a DatetimeIndex.")
    if not df.index.is_monotonic_increasing:
        raise ValueError(f"Dates for {symbol} are not strictly monotonic increasing.")

    if not df.index.is_unique or df.index.hasnans:
        raise ValueError(f"Dates for {symbol} must be unique and nonmissing.")
    if not np.isfinite(df[REQUIRED_COLUMNS].to_numpy(dtype=float)).all():
        raise ValueError(f"Non-finite OHLCV values for {symbol}.")
    price_cols = [c for c in REQUIRED_COLUMNS if c != "volume"]
    if (df[price_cols] <= 0).any().any():
        raise ValueError(f"Prices for {symbol} must be positive.")

    # 3. Check for NaNs
    nan_counts = df[REQUIRED_COLUMNS].isna().sum()
    if nan_counts.any():
        raise ValueError(f"NaN values found in {symbol} columns: {nan_counts[nan_counts > 0].to_dict()}")

    # 4. Check price consistency: High >= Low
    invalid_hl = df["high"] < df["low"]
    if invalid_hl.any():
        bad_dates = df.index[invalid_hl].tolist()[:3]
        raise ValueError(f"High price is lower than Low price for {symbol} on dates: {bad_dates}")

    if ((df["open"] > df["high"]) | (df["open"] < df["low"]) |
        (df["close"] > df["high"]) | (df["close"] < df["low"])).any():
        raise ValueError(f"Open/close outside high-low range for {symbol}.")

    # 5. Check volume: Volume >= 0
    if (df["volume"] < 0).any():
        raise ValueError(f"Negative volume detected for {symbol}.")


def check_split_adjustment(
    df: pd.DataFrame,
    split_date: str,
    max_allowed_abs_return: float = 0.20,
) -> bool:
    """
    Check whether a known stock split or bonus issue has been properly adjusted in adjusted_close.
    
    If unadjusted, a 2:1 split produces a ~ -50% jump, and a 10:1 split produces a ~ -90% jump.
    If properly adjusted, the 1-day return around the split date stays within normal daily bounds.
    
    Returns:
        True if the price series looks continuous around the split date.
        False if an anomalous jump suggests the split was not adjusted.
    """
    ts = pd.to_datetime(split_date)
    # Find the nearest trading day on or after split_date
    sub = df[df.index >= ts]
    if sub.empty:
        logger.warning(f"No trading data found on or after {split_date}.")
        return False

    split_idx = sub.index[0]
    loc = df.index.get_loc(split_idx)
    if loc == 0:
        return False  # No prior bar: adjustment continuity is unverified.

    prior_close = df["adjusted_close"].iloc[loc - 1]
    split_close = df["adjusted_close"].iloc[loc]

    daily_return = (split_close - prior_close) / prior_close
    
    # If the return exceeds typical daily move threshold (e.g. 20%), flag as unadjusted
    if abs(daily_return) > max_allowed_abs_return:
        logger.warning(
            f"Anomalous return on corporate action date {split_idx.date()}: {daily_return:.2%}. "
            f"Expected <= {max_allowed_abs_return:.0%}. Corporate action may not be adjusted."
        )
        return False

    return True
