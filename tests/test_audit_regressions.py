"""Regression cases discovered during the phase-five readiness audit."""
import numpy as np
import pandas as pd
import pytest
from overfit_radar.stats.basic import calculate_max_drawdown
from overfit_radar.strategies.families import generate_donchian_breakout
from overfit_radar.features.windows import generate_walk_forward_windows
from overfit_radar.data.validator import validate_ohlcv, check_split_adjustment
from overfit_radar.strategies.zoo import build_strategy_zoo, compute_strategy_signal
from overfit_radar.config import load_costs_config
from overfit_radar.costs.cost_model import TransactionCostModel


def test_drawdown_includes_initial_capital():
    assert calculate_max_drawdown(pd.Series([-.2, 0.])) == pytest.approx(.2)
    assert calculate_max_drawdown(pd.Series([-.2])) == pytest.approx(.2)


def test_donchian_uses_consistent_adjusted_prices(synthetic_gbm_prices):
    prices = synthetic_gbm_prices(n_days=300)
    adjusted = prices.copy()
    adjusted['adjusted_close'] *= .5
    pd.testing.assert_series_equal(generate_donchian_breakout(prices, 10), generate_donchian_breakout(adjusted, 10))


@pytest.mark.parametrize('kind', ['duplicate', 'infinite', 'negative', 'outside'])
def test_invalid_market_bars_rejected(synthetic_gbm_prices, kind):
    prices = synthetic_gbm_prices(n_days=30)
    if kind == 'duplicate':
        prices = pd.concat([prices.iloc[:1], prices])
    elif kind == 'infinite':
        prices.iloc[0, prices.columns.get_loc('open')] = np.inf
    elif kind == 'negative':
        prices.iloc[0, prices.columns.get_loc('adjusted_close')] = -1
    else:
        prices.iloc[0, prices.columns.get_loc('open')] = prices.high.iloc[0] * 2
    with pytest.raises(ValueError):
        validate_ohlcv(prices, 'TEST')


def test_split_check_requires_prior_bar(synthetic_gbm_prices):
    prices = synthetic_gbm_prices(n_days=30)
    assert not check_split_adjustment(prices, str(prices.index[0].date()))


@pytest.mark.parametrize('kwargs', [{'is_months':0}, {'oos_months':0}, {'step_months':-1}, {'embargo_days':-1}])
def test_invalid_window_lengths_rejected(kwargs):
    with pytest.raises(ValueError):
        generate_walk_forward_windows(pd.bdate_range('2010', periods=1100), **kwargs)


def test_cost_schedule_cannot_backfill_unknown_history():
    with pytest.raises(ValueError):
        TransactionCostModel(load_costs_config()).get_rates_for_date('2009-12-31')


def test_future_price_changes_preserve_every_family(synthetic_gbm_prices):
    prices = synthetic_gbm_prices(n_days=400)
    changed = prices.copy()
    changed.loc[changed.index[320]:, ['open','high','low','close','adjusted_close']] *= 3
    for strat in build_strategy_zoo():
        pd.testing.assert_series_equal(compute_strategy_signal(prices, strat).iloc[:320], compute_strategy_signal(changed, strat).iloc[:320])


def test_dataset_has_execution_features_and_filters_on_is_only(synthetic_gbm_prices):
    from dataclasses import replace
    from overfit_radar.config import load_experiment_config
    from overfit_radar.features.dataset import build_strategy_dataset
    prices = synthetic_gbm_prices(n_days=1100)
    cfg = load_experiment_config()
    cfg = replace(cfg, walk_forward=replace(cfg.walk_forward, min_is_trades=0))
    zoo = build_strategy_zoo()[:3]
    first = build_strategy_dataset({'TEST': prices}, zoo=zoo, experiment_config=cfg)
    required = ['params', 'is_trade_count', 'is_annual_turnover', 'is_sharpe_2x_cost', 'is_cost_sensitivity', 'is_observations']
    assert set(required).issubset(first.columns)
    prices.loc[prices.index[777]:, ['open','high','low','close','adjusted_close']] *= 2
    second = build_strategy_dataset({'TEST': prices}, zoo=zoo, experiment_config=cfg)
    feature_cols = [c for c in first if not c.startswith(('oos_sharpe','survived'))]
    pd.testing.assert_frame_equal(first[feature_cols], second[feature_cols])
    strict = replace(cfg, walk_forward=replace(cfg.walk_forward, min_is_trades=10000))
    assert build_strategy_dataset({'TEST': prices}, zoo=zoo, experiment_config=strict).empty


def test_effective_trials_ignore_flat_strategies():
    from overfit_radar.stats.psr_dsr import compute_effective_n
    rng = np.random.default_rng(1)
    returns = pd.DataFrame({'active': rng.normal(size=100), 'flat1': np.zeros(100), 'flat2': np.zeros(100)})
    for method in ['cluster', 'eigenvalue']:
        assert compute_effective_n(returns, method=method) == pytest.approx(1.)


def test_grid_reads_configuration(tmp_path):
    import yaml
    from overfit_radar.config import get_project_root
    data = yaml.safe_load((get_project_root() / 'configs/strategies.yaml').read_text())
    data['families']['sma_crossover']['fast_windows'] = [5]
    data['families']['sma_crossover']['slow_windows'] = [20]
    (tmp_path / 'strategies.yaml').write_text(yaml.safe_dump(data))
    sma = [s for s in build_strategy_zoo(config_dir=tmp_path) if s.family == 'sma_crossover']
    assert [s.strategy_id for s in sma] == ['sma_f5_s20']


def test_pbo_matches_independent_sharpe_oracle():
    import itertools
    from scipy.stats import rankdata
    from overfit_radar.stats.pbo import compute_cscv_pbo
    matrix = np.random.default_rng(17).normal(size=(24, 5)) * np.array([1,2,3,4,5])
    blocks = np.split(np.arange(24), 4)
    expected = []
    for combo in itertools.combinations(range(4), 2):
        a = matrix[np.concatenate([blocks[i] for i in combo])]
        b = matrix[np.concatenate([blocks[i] for i in range(4) if i not in combo])]
        best = np.argmax(a.mean(axis=0) / a.std(axis=0, ddof=1))
        expected.append(rankdata(b.mean(axis=0) / b.std(axis=0, ddof=1))[best] / 6)
    result = compute_cscv_pbo(pd.DataFrame(matrix), num_blocks=4, metric='sharpe')
    np.testing.assert_allclose(result['ranks'], expected)
    assert result['pbo'] == pytest.approx(np.mean(np.array(expected) <= .5))


def test_pbo_validates_block_shape():
    from overfit_radar.stats.pbo import compute_cscv_pbo
    with pytest.raises(ValueError):
        compute_cscv_pbo(pd.DataFrame(np.ones((25, 3))), num_blocks=4)


def test_dp_costs_are_opt_in_and_scale_with_trade_size(synthetic_gbm_prices):
    from overfit_radar.costs.engine import BacktestEngine
    prices = synthetic_gbm_prices(n_days=30)
    signal = pd.Series(([1., 0.] * 15), index=prices.index)
    cfg = load_costs_config()
    headline = BacktestEngine(cfg).run(prices, signal)
    for notional in [10000., 50000., 100000.]:
        stressed = BacktestEngine(cfg, dp_trade_notional=notional).run(prices, signal)
        expected = (headline.trade_action == -1) * cfg.regimes[0].dp_charges_per_sell / notional
        np.testing.assert_allclose(headline.net_return - stressed.net_return, expected)
        doubled = BacktestEngine(cfg, cost_multiplier=2., dp_trade_notional=notional).run(prices, signal)
        np.testing.assert_allclose(doubled.cost, 2 * stressed.cost)
