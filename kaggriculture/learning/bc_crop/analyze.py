"""Predeclared whole-seed paired BC evaluation; no model selection."""
import argparse
from collections import Counter
from pathlib import Path
import numpy as np
from .experiment import read, save, integrity


def score(row):return 1. if row['margin']>0 else .5 if row['margin']==0 else 0.


def summary(rows):
    return dict(games=len(rows),mean_profit=float(np.mean([r['net_profit'] for r in rows])),
        mean_opponent_profit=float(np.mean([r['opponent_bank']-3000 for r in rows])),
        mean_bank=float(np.mean([r['bank'] for r in rows])),mean_margin=float(np.mean([r['margin'] for r in rows])),
        mean_score=float(np.mean([score(r) for r in rows])),
        win=sum(r['margin']>0 for r in rows),draw=sum(r['margin']==0 for r in rows),loss=sum(r['margin']<0 for r in rows))


def run(out):
    cfg=integrity(out);rows=read(out/'test-ledger.json');assert len(rows)==144
    report=read(out/'model/report.json');cert=read(out/'audit-test.json');entry=read(out/'audit-entry-test.json')
    assert cert['passed'] and entry['passed']
    policies=('teacher','initial','trained');groups={}
    for op in ('all',*cfg['opponents']):
        groups[op]={p:summary([r for r in rows if r['policy']==p and (op=='all' or r['opponent']==op)]) for p in policies}
    pairs=[];comparisons={}
    rng=np.random.default_rng(cfg['bootstrap']['seed'])
    indices=rng.integers(0,len(cfg['test_seeds']),size=(cfg['bootstrap']['draws'],len(cfg['test_seeds'])))
    for comparator in ('initial','teacher'):
        pair=[]
        for seed in cfg['test_seeds']:
            for op in cfg['opponents']:
                for side in (0,1):
                    a,b=[next(r for r in rows if r['seed']==seed and r['opponent']==op and r['side']==side and r['policy']==p)
                         for p in (comparator,'trained')]
                    pair.append(dict(seed=seed,opponent=op,side=side,comparator=comparator,
                        profit=b['net_profit']-a['net_profit'],margin=b['margin']-a['margin'],score=score(b)-score(a),
                        opponent_profit=b['opponent_bank']-a['opponent_bank']))
        pairs+=pair;comparisons[comparator]={}
        for op in ('all',*cfg['opponents']):
            comparisons[comparator][op]={}
            for metric in ('profit','opponent_profit','margin','score'):
                values=np.asarray([np.mean([p[metric] for p in pair if p['seed']==s and (op=='all' or p['opponent']==op)]) for s in cfg['test_seeds']])
                estimates=values[indices].mean(1)
                comparisons[comparator][op][metric]=dict(mean=float(values.mean()),ci95=np.quantile(estimates,[.025,.975]).tolist(),
                                                        seed_values=values.tolist())
    control={}
    for p in policies:
        audits=[r for r in cert['rows'] if r['policy']==p];count=Counter();selected=Counter();planted=Counter()
        for r in audits:count.update(r['counts']);selected.update(r['selected_crops']);planted.update(r['official_planted_crops'])
        control[p]=dict(counts=dict(count),selected_crops=dict(selected),official_planted_crops=dict(planted),
            selector_model_control_rate=count['model_calls']/max(1,count['selector_calls']),
            inference_seconds=sum(r['original_inference_seconds'] for r in audits),
            fallbacks=sum(r['fallbacks'] for r in audits),load_errors=sum(r['load_errors'] for r in audits),
            inference_errors=sum(r['inference_errors'] for r in audits))
    A=all(r['passed'] for r in rows) and cert['passed'] and entry['passed'] and all(control[p]['fallbacks']==0 for p in policies)
    A=A and control['trained']['counts']['official_successful_plants']>0 and control['trained']['counts']['buy_seed_orders_changed_vs_teacher']>0
    validation=report['reports'];imitation=validation['trained']['validation']['accuracy']>max(validation['initial']['validation']['accuracy'],report['majority_validation_accuracy'])
    B=imitation and comparisons['initial']['all']['profit']['ci95'][0]>0 and comparisons['initial']['all']['score']['mean']>=0
    save(out/'paired-results.json',pairs)
    result=dict(engineering_A_passed=A,imitation_B_passed=B,validation_imitation_gate=imitation,
        test_games=144,independent_seed_clusters=8,file_entry_repeats_excluded=2,bootstrap=cfg['bootstrap'],
        by_opponent=groups,trained_minus=comparisons,control=control,
        limitations=['Only inner crop choice learned; opportunity/economic gates and all management remain teacher code',
                     'Eight seed clusters and three fixed opponents; no broad long-term generalization claim',
                     'No DAgger/PPO/self-play or new strategy search; no promotion of market V1'])
    save(out/'summary.json',result)
    print(json.dumps(dict(engineering_A_passed=A,imitation_B_passed=B,overall=groups['all'],paired=comparisons['initial']['all'])),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();run(a.out.resolve())
