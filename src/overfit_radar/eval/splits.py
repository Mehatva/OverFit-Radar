"""Forward folds with label-maturity purging and optional family holdout."""
from dataclasses import dataclass
import numpy as np
import pandas as pd

@dataclass(frozen=True)
class ForwardFold:
    """Positional row indices; held-out labels never enter fitting."""
    train: np.ndarray
    test: np.ndarray
    family: str
    prediction_date: pd.Timestamp


def forward_folds(table: pd.DataFrame, calendar: pd.DatetimeIndex,
                  final_start: str, embargo_bars: int = 0, leave_one_family_out: bool = False):
    """Yield development folds only, excluding labels reaching the final period.

    Prediction happens at OOS start. Primary folds include all families. Training labels must mature strictly before
    that date minus embargo_bars actual market sessions. In secondary mode each test family is
    entirely excluded from training. Calendar must cover all observed endpoints.
    """
    if embargo_bars < 0 or not isinstance(embargo_bars, int):
        raise ValueError('embargo_bars must be a nonnegative integer')
    if not calendar.is_unique or not calendar.is_monotonic_increasing or calendar.hasnans:
        raise ValueError('calendar must contain sorted unique market sessions')
    dates = table[['is_end_date','oos_start_date','oos_end_date']].apply(pd.to_datetime)
    if dates.isna().any().any() or table.family.isna().any():
        raise ValueError('Missing dates or family')
    if not ((dates.is_end_date < dates.oos_start_date) & (dates.oos_start_date <= dates.oos_end_date)).all():
        raise ValueError('Invalid feature/label chronology')
    if not dates.isin(calendar).all().all():
        raise ValueError('Calendar must cover every endpoint')
    cutoff = pd.Timestamp(final_start)
    development = dates.oos_end_date < cutoff
    for prediction in sorted(dates.loc[development, 'oos_start_date'].unique()):
        prediction = pd.Timestamp(prediction)
        pos = calendar.get_loc(prediction) - embargo_bars
        if pos < 0:
            continue
        mature = development & (dates.oos_end_date < calendar[pos])
        testing = development & (dates.oos_start_date == prediction)
        for family in (sorted(table.loc[testing, 'family'].unique()) if leave_one_family_out else ['all']):
            train = np.flatnonzero((mature & ((table.family != family) if leave_one_family_out else True)).to_numpy())
            test = np.flatnonzero((testing & ((table.family == family) if leave_one_family_out else True)).to_numpy())
            if len(train) and len(test):
                yield ForwardFold(train, test, family, prediction)
