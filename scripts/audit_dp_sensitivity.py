"""Offline DP-fee sensitivity on the first IS window; never score final OOS.

Run from the repository: .venv/bin/python scripts/audit_dp_sensitivity.py --output PATH
Every strategy is equally weighted in summary statistics; this is not a portfolio.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from overfit_radar.config import get_project_root, load_costs_config, load_yaml
from overfit_radar.costs.engine import BacktestEngine
from overfit_radar.strategies.zoo import build_strategy_zoo, compute_strategy_signal


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = get_project_root()
    cfg = load_costs_config()
    sizes = load_yaml(root / 'configs/costs.yaml')['dp_sensitivity_trade_notionals']
    engine = BacktestEngine(cfg)
    zoo = build_strategy_zoo()
    rows = []
    date_ranges = {}
    for file in sorted((root / 'data/raw').glob('*.parquet')):
        prices = pd.read_parquet(file).iloc[:756]
        if len(prices) < 756:
            continue
        date_ranges[file.stem] = [str(prices.index.min().date()), str(prices.index.max().date())]
        fees = np.array([engine.cost_model.get_rates_for_date(d).dp_charges_per_sell for d in prices.index])
        for strategy in zoo:
            result = engine.run(prices, compute_strategy_signal(prices, strategy))
            base = result.net_return.to_numpy()
            sells = (result.trade_action.to_numpy() == -1)
            for size in [None] + sizes:
                net = base if size is None else base - sells * fees / size
                sd = net.std(ddof=1)
                rows.append({'symbol': file.stem, 'strategy_id': strategy.strategy_id,
                             'notional': 'excluded' if size is None else str(size),
                             'sharpe': float(net.mean()/sd*np.sqrt(252)) if sd > 1e-12 else 0.,
                             'cumulative_return': float(np.prod(1 + net) - 1),
                             'extra_cost_sum': float((base-net).sum())})
    if not rows:
        raise ValueError('No cached symbols with a complete development window')
    frame = pd.DataFrame(rows)
    summary = frame.groupby('notional', sort=False).agg(
        strategy_symbol_pairs=('sharpe', 'size'), mean_sharpe=('sharpe','mean'),
        mean_cumulative_return=('cumulative_return','mean'),
        mean_extra_cost_sum=('extra_cost_sum','mean')).reset_index().to_dict('records')
    payload = {'scope':'First 756 bars only; all cached symbols; unfiltered strategy grid; descriptive development sensitivity, not ML evaluation',
               'policy':'Headline DP excluded; sensitivity assumes fixed rupee sell notional, one DP fee per sell event, existing unverified historical fee schedule',
               'dates':date_ranges,'summary':summary}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))


if __name__ == '__main__':
    main()
