"""Regenerate versioned data and evaluate development baselines under assumed costs."""
from pathlib import Path
import argparse,json
import numpy as np
import pandas as pd
from overfit_radar.config import get_project_root,load_yaml
from overfit_radar.features.dataset import build_strategy_dataset
from overfit_radar.features.provenance import save_versioned,load_training_table,digest
from overfit_radar.eval.splits import forward_folds
from overfit_radar.eval.metrics import ranking_metrics
from overfit_radar.models.baselines import baseline_scores


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    root=get_project_root();protocol=load_yaml(root/'configs/protocol.yaml')
    universe={p.stem:pd.read_parquet(p) for p in sorted((root/'data/raw').glob('*.parquet'))}
    table=build_strategy_dataset(universe)
    path=save_versioned(table,args.output)
    table=load_training_table(path)
    calendar=pd.DatetimeIndex(sorted(set().union(*(set(d.index) for d in universe.values()))))
    records=[];fold_records=[]
    for secondary in [False,True]:
        experiment='family_holdout' if secondary else 'primary'
        for fold in forward_folds(table,calendar,protocol['splits']['final_start'],leave_one_family_out=secondary):
            train=table.iloc[fold.train];test=table.iloc[fold.test].copy()
            assert train.oos_end_date.max()<test.oos_start_date.min()
            fold_records.append(dict(experiment=experiment,family=fold.family,prediction_date=str(fold.prediction_date),train_rows=len(train),test_rows=len(test),train_label_end=str(train.oos_end_date.max())))
            scores=baseline_scores(test)
            for level in protocol['cost_levels']:
                labelled=test.assign(oos_sharpe=test[f'oos_sharpe_{level}'],survived=test[f'survived_{level}'])
                for baseline in scores:
                    metrics=ranking_metrics(labelled,scores[baseline],protocol['top_k'])
                    metrics['experiment']=experiment;metrics['family']=fold.family;metrics['cost_level']=level;metrics['baseline']=baseline
                    records.extend(metrics.to_dict('records'))
    raw=pd.DataFrame(records)
    # Macro-average symbols (and secondary families) first: one observation per window.
    keys=['experiment','cost_level','baseline']
    measures=['survival_rate','mean_oos_sharpe','spearman']
    windows=raw.groupby(keys+['window_id'],sort=True)[measures].mean().reset_index()
    windows.to_json(args.output/'per_window_metrics.json',orient='records',indent=2)
    summaries=[];rng=np.random.default_rng(protocol['bootstrap']['seed'])
    for key,g in windows.groupby(keys,sort=True):
        g=g.sort_values('window_id');a=g[measures].to_numpy();n=len(a);block=protocol['bootstrap']['block_length']
        indices=(rng.integers(n,size=(protocol['bootstrap']['repetitions'],int(np.ceil(n/block)),1))+np.arange(block))%n
        draws=np.nanmean(a[indices.reshape(len(indices),-1)[:,:n]],axis=1)
        for j,metric in enumerate(measures):
            summaries.append(dict(zip(keys,key),metric=metric,windows=n,mean=float(np.nanmean(a[:,j])),ci_low=float(np.nanquantile(draws[:,j],.025)),ci_high=float(np.nanquantile(draws[:,j],.975))))
    (args.output/'baseline_summary.json').write_text(json.dumps(summaries,indent=2))
    (args.output/'folds.json').write_text(json.dumps(fold_records,indent=2))
    (args.output/'gates.json').write_text(json.dumps(dict(gates_1_to_4='passed',baseline_results_computed=True,models_trained=False,claim='under assumed cost levels',baseline_hashes={name:digest(args.output/name) for name in ['baseline_summary.json','per_window_metrics.json','folds.json']},dataset_sha256=digest(path),rows=len(table),primary_folds=sum(r['experiment']=='primary' for r in fold_records),final_start=protocol['splits']['final_start']),indent=2))
    print('Completed',len(table),'rows;',len(fold_records),'folds')

if __name__=='__main__':main()
