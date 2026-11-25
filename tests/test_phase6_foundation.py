import numpy as np
import pandas as pd
from overfit_radar.eval.splits import forward_folds
from overfit_radar.models.baselines import baseline_scores
from overfit_radar.eval.metrics import ranking_metrics


def test_forward_splits_purge_labels_and_families():
    calendar = pd.bdate_range('2010', periods=1000)
    rows = [dict(family=f, is_end_date=calendar[i], oos_start_date=calendar[i+10], oos_end_date=calendar[i+100]) for i in range(100,800,50) for f in ['a','b','c']]
    d = pd.DataFrame(rows)
    folds = list(forward_folds(d,calendar,str(calendar[800].date()),21, leave_one_family_out=True))
    assert folds
    for fold in folds:
        train,test = d.iloc[fold.train],d.iloc[fold.test]
        assert set(train.family).isdisjoint(test.family)
        assert train.oos_end_date.max() < calendar[calendar.get_loc(fold.prediction_date)-21]
        assert test.oos_end_date.max() < calendar[800]
        assert not set(fold.train)&set(fold.test)


def test_baselines_ignore_labels_and_row_order():
    d=pd.DataFrame(dict(symbol=['X']*3,window_id=['w']*3,strategy_id=['a','b','c'],is_sharpe=[1,2,3],is_dsr_raw=[.2,.3,.4],is_dsr_eff=[.3,.4,.5],oos_sharpe=[-1,0,1],survived=[0,0,1]))
    before=baseline_scores(d)
    changed=d.copy();changed['oos_sharpe']=999;changed['survived']=1
    pd.testing.assert_frame_equal(before,baseline_scores(changed))
    pd.testing.assert_frame_equal(before,baseline_scores(d.iloc[::-1]).sort_index())
    result=ranking_metrics(d,before.is_sharpe,k=1)
    assert result.survival_rate.iloc[0]==1
    assert result.mean_oos_sharpe.iloc[0]==1


def test_primary_training_labels_end_before_test_oos_start():
    calendar=pd.bdate_range('2010',periods=1000)
    d=pd.DataFrame([dict(family=f,is_end_date=calendar[i],oos_start_date=calendar[i+10],oos_end_date=calendar[i+100]) for i in range(100,800,50) for f in ['a','b']])
    folds=list(forward_folds(d,calendar,str(calendar[900].date())))
    assert folds
    for fold in folds:
        train,test=d.iloc[fold.train],d.iloc[fold.test]
        assert train.oos_end_date.max() < test.oos_start_date.min()
        assert set(train.family) == set(test.family)


def test_window_returns_reset_and_force_exit(synthetic_gbm_prices):
    from dataclasses import replace
    from overfit_radar.config import load_experiment_config,load_costs_config
    from overfit_radar.features.dataset import build_strategy_dataset
    from overfit_radar.strategies.zoo import build_strategy_zoo,compute_strategy_signal
    from overfit_radar.costs.engine import BacktestEngine
    from overfit_radar.stats.basic import calculate_sharpe
    p=synthetic_gbm_prices(n_days=1100)
    strat=build_strategy_zoo()[0];cfg=load_experiment_config()
    cfg=replace(cfg,walk_forward=replace(cfg.walk_forward,min_is_trades=0))
    d=build_strategy_dataset({'X':p},zoo=[strat],experiment_config=cfg)
    engine=BacktestEngine(load_costs_config());signal=compute_strategy_signal(p,strat)
    for start,end,column in [(0,756,'is_sharpe'),(777,1029,'oos_sharpe')]:
        result=engine.run(p.iloc[start:end],signal.iloc[start:end])
        assert result.position.iloc[0]==0
        net=result.net_return.copy()
        if result.position.iloc[-1]:
            net.iloc[-1]-=engine.cost_model.compute_cost_rate(p.index[end-1],False,engine.slippage_bps)
        assert np.isclose(d[column].iloc[0],calculate_sharpe(net))


def test_manifest_rejects_tampering(tmp_path,monkeypatch):
    from overfit_radar.features import provenance as p
    monkeypatch.setattr(p,'fingerprint',lambda: {'config_hash':'a','cost_schedule_version':'v1'})
    path=p.save_versioned(pd.DataFrame({'a':[1]}),tmp_path)
    assert len(p.load_training_table(path))==1
    monkeypatch.setattr(p,'fingerprint',lambda: {'config_hash':'b','cost_schedule_version':'v1'})
    import pytest
    with pytest.raises(ValueError,match='config_hash'):
        p.load_training_table(path)
