"""SELL-only paired experiment, reusing the existing official Games workers."""
import argparse, json, hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from statistics import mean
import numpy as np
from optimize_round2 import Games

ROOT=Path(__file__).resolve().parents[1]
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):
    assert not p.exists(),str(p)
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_bytes((json.dumps(x,ensure_ascii=False,indent=2)+'\n').encode())

def check_freeze(root,stage):
    f=read(root/(stage+'-freeze.json'))
    cfg=read(root/('registration.json' if stage=='test' else 'development-plan.json'))
    assert sha(root/('registration.json' if stage=='test' else 'development-plan.json'))==f['registration_sha256']
    for name,h in f['runtime_sha256'].items():assert sha(ROOT/name)==h,name
    for name,h in f['source_sha256'].items():assert sha(root/'sources'/name)==h,name
    return cfg

def aggregate(rr):
    return dict(games=len(rr),wins=sum(r['margin']>0 for r in rr),draws=sum(r['margin']==0 for r in rr),losses=sum(r['margin']<0 for r in rr),
        mean_net_profit=mean(r['net_profit'] for r in rr),mean_opponent_net_profit=mean(r['opponent_bank']-3000 for r in rr),mean_margin=mean(r['margin'] for r in rr),mean_score=mean(r['match_score'] for r in rr))

def summarize(rows,cfg,seeds,formal):
    lookup={(r['seed'],r['opponent'],r['side'],r['policy']):r for r in rows}
    assert len(lookup)==len(seeds)*len(cfg['opponents'])*4==len(rows)
    pairs=[]
    for seed in seeds:
        for op in cfg['opponents']:
            for side in (0,1):
                a,b=[lookup[seed,op,side,p] for p in cfg['policies']]
                pairs.append(dict(seed=seed,opponent=op,side=side,baseline_profit=a['net_profit'],candidate_profit=b['net_profit'],profit_delta=b['net_profit']-a['net_profit'],margin_delta=b['margin']-a['margin'],score_delta=b['match_score']-a['match_score']))
    panels={}
    for name,opponents in [('primary',cfg['primary_opponents']),('mirror',['teacher'])]+[(o,[o]) for o in cfg['primary_opponents']]:
        rr=[r for r in pairs if r['opponent'] in opponents]
        cluster=np.array([[mean(r[k] for r in rr if r['seed']==s) for k in ('profit_delta','margin_delta','score_delta')] for s in seeds])
        rng=np.random.default_rng(cfg['bootstrap']['rng_seed'])
        ci=np.quantile(cluster[rng.integers(0,len(seeds),(20000,len(seeds)))].mean(axis=1),[.025,.975],axis=0)
        panels[name]=dict(policies={p:aggregate([r for r in rows if r['policy']==p and r['opponent'] in opponents]) for p in cfg['policies']},
            contrasts={k:dict(mean=float(cluster[:,i].mean()),ci_95=ci[:,i].tolist(),seed_values=cluster[:,i].tolist()) for i,k in enumerate(('profit_delta','margin_delta','score_delta'))})
    pri=panels['primary'];market=panels['market'];v=pri['contrasts']
    gates=dict(profit_positive_lower_bound=v['profit_delta']['ci_95'][0]>0,profit_at_least_one_percent=v['profit_delta']['mean']>=.01*pri['policies']['baseline']['mean_net_profit'],
        margin_nonnegative=v['margin_delta']['mean']>=0,score_nonnegative=v['score_delta']['mean']>=0,
        market_profit_nonnegative=market['contrasts']['profit_delta']['mean']>=0,market_margin_nonnegative=market['contrasts']['margin_delta']['mean']>=0,market_score_nonnegative=market['contrasts']['score_delta']['mean']>=0)
    development=all(x>0 for x in v['profit_delta']['seed_values']) and gates['margin_nonnegative'] and gates['score_nonnegative']
    return dict(games=len(rows),seed_clusters=len(seeds),formal=formal,panels=panels,economic_gates=gates,economic_gate_passed=all(gates.values()) if formal else None,development_economic_gate_passed=development if not formal else None,interpretation='One paired primary contrast. Bootstrap resamples complete seeds; subgroup/mirror results descriptive. Economic gate does not replace execution/preservation gates.'),pairs

def run(root,stage):
    cfg=check_freeze(root,stage);seeds=cfg['seeds'] if stage=='test' else cfg['development_seeds']
    target=root/stage;target.mkdir(exist_ok=False);rows=[]
    with Games(root,cfg,stage) as games,ThreadPoolExecutor(max_workers=cfg['workers']) as pool:
        jobs=[pool.submit(games.play,root/'sources'/(p+'.py'),p,o,s,i,target/f'{s}-{o}-{i}-{p}') for s in seeds for o in cfg['opponents'] for i in (0,1) for p in cfg['policies']]
        for f in as_completed(jobs):
            r=f.result();r['match_score']=float(r['margin']>0)+.5*(r['margin']==0);rows.append(r)
            print(f'{stage} {len(rows)}/{len(jobs)}',flush=True)
        rows.sort(key=lambda r:(r['seed'],r['opponent'],r['side'],r['policy']))
        save(target/'ledger.json',rows)
        report,pairs=summarize(rows,cfg,seeds,stage=='test')
        save(target/'summary.json',report);save(target/'paired-results.json',pairs)
        if stage=='test':
            repeated=[]
            for policy in cfg['policies']:
                for side in (1,0):
                    r=games.play(root/'sources'/(policy+'.py'),policy,'market',seeds[0],side,root/'repeat'/f'{policy}-{side}')
                    old=next(x for x in rows if (x['seed'],x['opponent'],x['side'],x['policy'])==(r['seed'],r['opponent'],r['side'],r['policy']))
                    assert r['banks']==old['banks'] and r['semantic_replay_sha256']==old['semantic_replay_sha256'] and r['semantic_replay_sha256_v2']==old['semantic_replay_sha256_v2']
                    repeated.append(r)
            save(root/'reproducibility.json',dict(passed=True,rows=repeated,excluded_from_statistics=True))
    check_freeze(root,stage)
    print(json.dumps(report['panels']['primary'],indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--stage',choices=['development','test'],required=True);a=p.parse_args();run(a.root.resolve(),a.stage)
