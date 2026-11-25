"""Parameter-neighbourhood graph builder and plateau/spike stability metrics."""

from __future__ import annotations

from typing import Dict, List, Tuple
import numpy as np
from overfit_radar.strategies.base import StrategyConfig


def build_neighbourhood_graph(zoo: List[StrategyConfig]) -> Dict[str, List[str]]:
    """
    Construct a topological parameter graph mapping each strategy to its immediate neighbours.
    
    A strategy's neighbours are configurations within the same algorithmic family that differ
    by at most 1 grid step along any numeric parameter axis (Manhattan / Chebyshev distance <= 1).
    Robust strategies sit on broad parameter plateaus; overfit ones sit on isolated spikes.
    """
    # Group strategies by family
    by_family: Dict[str, List[StrategyConfig]] = {}
    for s in zoo:
        by_family.setdefault(s.family, []).append(s)

    graph: Dict[str, List[str]] = {s.strategy_id: [] for s in zoo}

    for fam, strat_list in by_family.items():
        if not strat_list:
            continue

        # Extract numeric parameter names
        first_params = strat_list[0].params
        numeric_keys = [k for k, v in first_params.items() if isinstance(v, (int, float)) and not isinstance(v, bool)]
        categorical_keys = [k for k, v in first_params.items() if k not in numeric_keys]

        # Build coordinate index for each numeric parameter
        coord_maps = {}
        for k in numeric_keys:
            unique_sorted = sorted(list({s.params[k] for s in strat_list}))
            coord_maps[k] = {val: idx for idx, val in enumerate(unique_sorted)}

        # Find neighbours for each strategy in this family
        for i, s1 in enumerate(strat_list):
            s1_id = s1.strategy_id
            s1_coords = {k: coord_maps[k][s1.params[k]] for k in numeric_keys}
            s1_cats = {k: s1.params[k] for k in categorical_keys}

            for j in range(i + 1, len(strat_list)):
                s2 = strat_list[j]
                s2_id = s2.strategy_id

                # Categorical parameters must match exactly
                if any(s1_cats[k] != s2.params[k] for k in categorical_keys):
                    continue

                # Numeric coordinate distance
                dist = max(abs(s1_coords[k] - coord_maps[k][s2.params[k]]) for k in numeric_keys)
                if dist <= 1:
                    graph[s1_id].append(s2_id)
                    graph[s2_id].append(s1_id)

    return graph


def compute_neighbourhood_metrics(
    candidate_id: str,
    sharpe_map: Dict[str, float],
    graph: Dict[str, List[str]],
) -> Tuple[float, float]:
    """
    Compute the mean and standard deviation of In-Sample Sharpe across the local neighbourhood.
    
    The local neighbourhood consists of the candidate strategy and its immediate parameter neighbours.
    On a parameter plateau, the local neighbourhood has high mean and near-zero standard deviation.
    On an overfit spike, the candidate stands alone and the local neighbourhood standard deviation is high.
    
    Returns:
        (neighbourhood_sharpe_mean, neighbourhood_sharpe_std)
    """
    neighbours = graph.get(candidate_id, [])
    cand_sharpe = sharpe_map.get(candidate_id, 0.0)

    if not neighbours:
        return cand_sharpe, 0.0

    # Local neighbourhood: candidate plus its adjacent neighbours
    local_cluster = [cand_sharpe] + [sharpe_map.get(n_id, 0.0) for n_id in neighbours]
    mean_val = float(np.mean(local_cluster))
    std_val = float(np.std(local_cluster, ddof=1)) if len(local_cluster) > 1 else 0.0

    return mean_val, std_val
