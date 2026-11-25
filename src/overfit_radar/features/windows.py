"""Walk-forward rolling window generator with embargo buffering."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional
import pandas as pd


@dataclass(frozen=True)
class WindowSlice:
    """Specification of an in-sample / out-of-sample temporal partition."""
    window_id: str
    is_start_idx: int
    is_end_idx: int
    oos_start_idx: int
    oos_end_idx: int
    is_start_date: Optional[pd.Timestamp] = None
    is_end_date: Optional[pd.Timestamp] = None
    oos_start_date: Optional[pd.Timestamp] = None
    oos_end_date: Optional[pd.Timestamp] = None


def generate_walk_forward_windows(
    dates: pd.DatetimeIndex,
    is_months: int = 36,
    oos_months: int = 12,
    step_months: int = 6,
    embargo_days: int = 21,
) -> List[WindowSlice]:
    """
    Generate chronological rolling walk-forward windows (which may overlap) with an embargo buffer.
    
    Standard convention:
    - 21 business days ~= 1 calendar month.
    - IS length: is_months * 21 (~756 trading days for 36 months).
    - Embargo gap: embargo_days (~21 trading days / 1 month buffer between IS and OOS).
    - OOS length: oos_months * 21 (~252 trading days for 12 months).
    - Step size: step_months * 21 (~126 trading days for 6 months).
    """
    for name, value in (("is_months", is_months), ("oos_months", oos_months), ("step_months", step_months)):
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if not isinstance(embargo_days, int) or isinstance(embargo_days, bool) or embargo_days < 0:
        raise ValueError("embargo_days must be a nonnegative integer")
    if not isinstance(dates, pd.DatetimeIndex) or dates.hasnans or not dates.is_unique or not dates.is_monotonic_increasing:
        raise ValueError("dates must be a unique, sorted DatetimeIndex without NaT")
    n_total = len(dates)
    is_bars = is_months * 21
    oos_bars = oos_months * 21
    step_bars = step_months * 21
    embargo_bars = embargo_days

    total_window_span = is_bars + embargo_bars + oos_bars
    if n_total < total_window_span:
        raise ValueError(
            f"Not enough trading bars ({n_total}) for even 1 window of span {total_window_span}."
        )

    windows: List[WindowSlice] = []
    current_start = 0
    window_idx = 0

    while True:
        is_start = current_start
        is_end = is_start + is_bars - 1
        
        oos_start = is_end + embargo_bars + 1
        oos_end = oos_start + oos_bars - 1

        if oos_end >= n_total:
            break

        win_id = f"w_{window_idx:02d}_{dates[is_start].strftime('%Y%m')}_{dates[oos_end].strftime('%Y%m')}"
        windows.append(
            WindowSlice(
                window_id=win_id,
                is_start_idx=is_start,
                is_end_idx=is_end,
                oos_start_idx=oos_start,
                oos_end_idx=oos_end,
                is_start_date=dates[is_start],
                is_end_date=dates[is_end],
                oos_start_date=dates[oos_start],
                oos_end_date=dates[oos_end],
            )
        )

        current_start += step_bars
        window_idx += 1

    return windows
