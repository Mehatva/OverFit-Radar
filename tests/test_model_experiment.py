"""Leakage, calibration chronology and probability-metric regression checks."""
import importlib.util
import os
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from overfit_radar.config import get_project_root
path=Path(os.environ.get('PHASE6_MODELS_SCRIPT',get_project_root()/'scripts/phase6_models.py'))
spec=importlib.util.spec_from_file_location('phase6_models',path)
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def test_calibration_purges_overlapping_labels_and_equality():
    rows=[]
    for start,end in [('2010-01-01','2010-04-01'),('2010-03-01','2010-06-01'),('2010-05-01','2010-08-01'),('2010-08-01','2010-12-01')]:
        for y in [0,1]:rows.append(dict(window_id=start,oos_start_date=pd.Timestamp(start),oos_end_date=pd.Timestamp(end),survived=y))
    fit,cal=m.calibration_partition(pd.DataFrame(rows),dict(min_fit_windows=2,min_fit_rows=2,min_calibration_rows=2))
    assert fit.oos_end_date.max()<cal.oos_start_date.min()
    assert pd.Timestamp('2010-08-01') not in set(fit.oos_end_date)
    assert len(fit)==4 and len(cal)==2


def test_evaluation_labels_and_identifiers_never_change_features_or_predictions():
    rng=np.random.default_rng(4)
    d=pd.DataFrame(rng.normal(size=(180,len(m.NUMERIC+m.NEIGHBOUR))),columns=m.NUMERIC+m.NEIGHBOUR)
    d['family']=np.where(np.arange(180)%2,'a','b');d['survived']=np.arange(180)%2
    config=dict(seed=42,logistic=dict(C=1.,max_iter=1000,solver='liblinear'))
    pipeline,calibrator=m.fit_model(d.iloc[:100],d.iloc[100:140],'logistic',True,config)
    test=d.iloc[140:].copy();raw=pipeline.predict_proba(m.feature_frame(test,True))
    test['survived']=1-test.survived;test['oos_sharpe']=999;test['strategy_id']='future';test['symbol']='future'
    np.testing.assert_array_equal(raw,pipeline.predict_proba(m.feature_frame(test,True)))
    assert not set(m.NEIGHBOUR)&set(m.feature_frame(test,False))
    p=calibrator.predict_proba(m.logit(raw[:,1]))[:,1]
    assert np.isfinite(p).all() and ((p>=0)&(p<=1)).all()


def test_probability_metrics_known_values():
    values=m.probability_metrics([0,1],[.25,.75],bins=2)
    assert values['auc']==1
    assert values['brier']==pytest.approx(.0625)
    assert values['ece']==pytest.approx(.25)
