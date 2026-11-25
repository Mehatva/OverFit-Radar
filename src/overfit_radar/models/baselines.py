"""Deterministic ranking baselines; scores are not calibrated probabilities."""
import hashlib
import pandas as pd


def baseline_scores(table: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """Return IS Sharpe, raw/effective DSR and row-order-independent random scores."""
    keys = ['symbol','window_id','strategy_id']
    if table.duplicated(keys).any() or table[keys].isna().any().any():
        raise ValueError('Candidate identifiers must be nonmissing and unique')
    scores = table[['is_sharpe','is_dsr_raw','is_dsr_eff']].copy()
    def random_score(row):
        import json
        key = json.dumps([seed] + list(row), separators=(',', ':'))
        return int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], 'big') / 2**64
    scores['random'] = [random_score(row) for row in table[keys].itertuples(index=False, name=None)]
    return scores
