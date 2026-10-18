"""Frozen-policy, paired attribution experiment using existing official workers."""
import argparse, json
from pathlib import Path
from statistics import mean
import numpy as np
from fertilizer_ablation import read, sha, save
from optimize_round2 import Games
from concurrent.futures import ThreadPoolExecutor, as_completed

def statistics(rows, cfg):
    lookup={(r['seed'],r['opponent'],r['side'],r['policy']):r for r in rows}
    assert len(lookup)==len(rows)==cfg['games']
    pairs=[]
    for s in cfg['seeds']:
        for o in cfg['opponents']:
            for side in (0,1):
                b,f,h=[lookup[s,o,side,p] for p in cfg['policies']]
                pairs.append(dict(seed=s,opponent=o,side=side,profit=[f['net_profit']-b['net_profit'],h['net_profit']-f['net_profit'],h['net_profit']-b['net_profit']],margin=[f['margin']-b['margin'],h['margin']-f['margin'],h['margin']-b['margin']]))
    clusters=np.array([[mean(r['profit'][i] for r in pairs if r['seed']==s) for i in range(3)] for s in cfg['seeds']])
    rng=np.random.default_rng(cfg['bootstrap']['rng_seed'])
    ci=np.quantile(clusters[rng.integers(0,len(clusters),(20000,len(clusters)))].mean(axis=1),[.005,.995],axis=0)
    def agg(rr):
        return dict(games=len(rr),wins=sum(r['margin']>0 for r in rr),draws=sum(r['margin']==0 for r in rr),losses=sum(r['margin']<0 for r in rr),mean_net_profit=mean(r['net_profit'] for r in rr),mean_margin=mean(r['margin'] for r in rr))
    return dict(games=len(rows),seed_clusters=len(clusters),contrasts={n:dict(mean=float(clusters[:,i].mean()),ci_99=ci[:,i].tolist(),seed_deltas=clusters[:,i].tolist()) for i,n in enumerate(cfg['contrasts'])},aggregate={p:agg([r for r in rows if r['policy']==p]) for p in cfg['policies']},by_opponent={o:{p:agg([r for r in rows if r['policy']==p and r['opponent']==o]) for p in cfg['policies']} for o in cfg['opponents']}),pairs

def run(out):
    cfg=read(out/'registration.json');freeze=read(out/'freeze.json')
    assert sha(out/'registration.json')==freeze['registration_sha256']
    assert sha(Path(__file__))==freeze['runner_sha256']
    for n,h in freeze['source_sha256'].items():assert sha(out/'sources'/n)==h
    target=out/'test';target.mkdir(exist_ok=False);rows=[]
    with Games(out,cfg,'test') as games,ThreadPoolExecutor(max_workers=cfg['workers']) as pool:
        jobs=[pool.submit(games.play,out/'sources'/(p+'.py'),p,o,s,i,target/f'{s}-{o}-{i}-{p}') for s in cfg['seeds'] for o in cfg['opponents'] for i in (0,1) for p in cfg['policies']]
        for f in as_completed(jobs):
            rows.append(f.result());print(f'games {len(rows)}/{len(jobs)}',flush=True)
        rows.sort(key=lambda r:(r['seed'],r['opponent'],r['side'],r['policy']))
        save(target/'ledger.json',rows);summary,pairs=statistics(rows,cfg)
        save(target/'summary.json',summary);save(target/'paired-results.json',pairs)
        repeats=[]
        for p in ('fertilizer','both'):
            for side in (1,0):
                r=games.play(out/'sources'/(p+'.py'),p,'market',cfg['seeds'][0],side,out/'repeat'/f'{p}-{side}')
                original=next(x for x in rows if all(x[k]==r[k] for k in ('seed','opponent','side','policy')))
                assert r['banks']==original['banks'] and r['semantic_replay_sha256_v2']==original['semantic_replay_sha256_v2']
                repeats.append(r)
        save(out/'reproducibility.json',dict(passed=True,rows=repeats))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('folder',type=Path);run(p.parse_args().folder.resolve())
