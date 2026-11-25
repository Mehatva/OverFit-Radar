"""Comparable candidate-set ranking metrics."""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def ranking_metrics(table: pd.DataFrame, scores: pd.Series, k: int = 10) -> pd.DataFrame:
    """Evaluate within symbol/window, reporting counts and stable top-k ties.

    Macro-average these rows; do not treat correlated strategy rows as independent.
    Undefined rank correlation is NaN, not zero.
    """
    if k < 1 or not isinstance(k, int):
        raise ValueError('k must be positive')
    if not scores.index.equals(table.index) or not np.isfinite(scores).all():
        raise ValueError('Scores must be finite and exactly aligned')
    frame = table.assign(score=scores)
    rows = []
    for (symbol, window), group in frame.groupby(['symbol','window_id'], sort=True):
        if not np.isfinite(group.oos_sharpe).all() or not group.survived.isin([0,1]).all():
            raise ValueError('Invalid evaluation labels')
        selected = group.sort_values(['score','strategy_id'], ascending=[False,True]).head(k)
        rho = float(spearmanr(group.score, group.oos_sharpe).statistic) if group.score.nunique()>1 and group.oos_sharpe.nunique()>1 else np.nan
        rows.append(dict(symbol=symbol, window_id=window, candidates=len(group), selected=len(selected),
                         survival_rate=float(selected.survived.mean()), mean_oos_sharpe=float(selected.oos_sharpe.mean()), spearman=rho))
    return pd.DataFrame(rows)
