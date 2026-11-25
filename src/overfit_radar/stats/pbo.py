"""CSCV PBO, following Bailey et al. (2015), Algorithm 2.3.

The selection metric is explicit. Equal-sized blocks and synchronous, finite
returns are required. Trimming incomplete blocks is a caller decision.
"""
from __future__ import annotations

import itertools
from typing import Any, Dict, Optional
import numpy as np
import pandas as pd
from scipy.stats import rankdata


def compute_cscv_pbo(
    returns_matrix: pd.DataFrame,
    num_blocks: int = 16,
    max_combinations: Optional[int] = None,
    metric: str = "sharpe",
    random_seed: int = 42,
) -> Dict[str, Any]:
    """Select the IS winner and measure its relative OOS rank for each split.

    Sharpe is per-period with sample standard deviation. Constant series receive
    score zero, matching calculate_sharpe. IS ties use first column order; OOS
    ties use average ranks. A capped run samples combinations without replacement
    using a fixed seed and is explicitly marked approximate.
    """
    if not isinstance(num_blocks, int) or isinstance(num_blocks, bool) or num_blocks < 2 or num_blocks % 2:
        raise ValueError("num_blocks must be a positive even integer >= 2")
    if metric not in {"mean", "sharpe"}:
        raise ValueError("metric must be 'mean' or 'sharpe'")
    values = returns_matrix.to_numpy(dtype=float)
    t, n = values.shape
    if n < 2 or t < 2 * num_blocks or t % num_blocks:
        raise ValueError("Require >= 2 strategies and equal blocks with >= 2 observations each")
    if not np.isfinite(values).all():
        raise ValueError("CSCV requires finite, synchronous returns")
    if max_combinations is not None and (not isinstance(max_combinations, int) or isinstance(max_combinations, bool) or max_combinations < 1):
        raise ValueError("max_combinations must be a positive integer")
    blocks = values.reshape(num_blocks, t // num_blocks, n)
    sums = blocks.sum(axis=1)
    squares = np.square(blocks).sum(axis=1)
    combos = list(itertools.combinations(range(num_blocks), num_blocks // 2))
    total = len(combos)
    approximate = max_combinations is not None and max_combinations < total
    if approximate:
        selected = np.sort(np.random.default_rng(random_seed).choice(total, max_combinations, replace=False))
        combos = [combos[i] for i in selected]
    count = t // 2

    def score(indices: list) -> np.ndarray:
        total_sum = sums[indices].sum(axis=0)
        means = total_sum / count
        if metric == "mean":
            return means
        variance = (squares[indices].sum(axis=0) - total_sum**2 / count) / (count - 1)
        std = np.sqrt(np.maximum(variance, 0))
        return np.divide(means, std, out=np.zeros(n), where=std > 1e-12)

    ranks = []
    for combo in combos:
        best = int(np.argmax(score(list(combo))))
        oos = score([i for i in range(num_blocks) if i not in combo])
        ranks.append(rankdata(oos, method="average")[best] / (n + 1))
    ranks = np.asarray(ranks)
    logits = np.log(ranks / (1 - ranks))
    return {
        "pbo": float(np.mean(logits <= 0)), "logits": logits, "ranks": ranks,
        "num_combinations": len(combos), "total_combinations": total,
        "mean_rank": float(ranks.mean()), "median_rank": float(np.median(ranks)),
        "metric": metric, "approximate": approximate,
    }
