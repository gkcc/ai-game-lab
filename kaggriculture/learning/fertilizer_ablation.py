"""Registered surgical ablation; uses the existing official match workers."""
import argparse, hashlib, json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from statistics import mean
import numpy as np
from optimize_round2 import Games

def read(p): return json.loads(p.read_text(encoding='utf-8'))
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,obj):
    assert not p.exists(), p.name
    p.write_text(json.dumps(obj,indent=2)+'\n',encoding='utf-8')

OVERRIDE='''        if c in ("TOMATO", "STRAWBERRY") and tile["fertilized_until_day"] >= day and tile["yield_units"] >= 2:
            mature = True  # Free storage for the next fertilized production night.
'''
ANCHOR='''        mature = age >= CROPS[c][2] or (c == "MELON" and tile["yield_units"] >= 6)
'''

def variants(out):
    src=out/'sources'
    both=(src/'both.py').read_text(encoding='utf-8');base=(src/'baseline.py').read_text(encoding='utf-8')
    assert both.count(OVERRIDE)==1 and base.count(OVERRIDE)==0 and base.count(ANCHOR)==1
    for n,s in [('fertilizer.py',both.replace(OVERRIDE,'')),('harvest.py',base.replace(ANCHOR,ANCHOR+OVERRIDE))]:
        assert not (src/n).exists()
        (src/n).write_bytes(s.encode('utf-8'))
    # Reversible surgical edits, no hand-written substitutes for either parent.
    assert (src/'fertilizer.py').read_text().replace(ANCHOR,ANCHOR+OVERRIDE)==both
    assert (src/'harvest.py').read_text().replace(OVERRIDE,'')==base

def statistics(rows,cfg):
    lookup={(r['seed'],r['opponent'],r['side'],r['policy']):r for r in rows}
    assert len(lookup)==len(rows)==cfg['games']
    pairs=[]
    for s in cfg['seeds']:
        for o in cfg['opponents']:
            for i in (0,1):
                b,f,h,both=[lookup[s,o,i,p]['net_profit'] for p in cfg['policies']]
                pairs.append(dict(seed=s,opponent=o,side=i,contrasts=[f-b,both-f,both-b,both-f-h+b,2*f-both-b],harvest_minus_baseline=h-b))
    clusters=np.array([[mean(r['contrasts'][i] for r in pairs if r['seed']==s) for i in range(5)] for s in cfg['seeds']])
    rng=np.random.default_rng(cfg['bootstrap']['rng_seed']);indices=rng.integers(0,len(clusters),size=(cfg['bootstrap']['resamples'],len(clusters)))
    ci=np.quantile(clusters[indices].mean(axis=1),[.005,.995],axis=0)
    import math
    contrasts={}
    for i,name in enumerate(cfg['contrast_names']):
        x=clusters[:,i];pos=int(sum(x>0));neg=int(sum(x<0));n=pos+neg
        sign=min(1.,2*sum(math.comb(n,k) for k in range(min(pos,neg)+1))/2**n) if n else 1.
        contrasts[name]={'mean':float(x.mean()),'ci_99':ci[:,i].tolist(),'seed_deltas':x.tolist(),'sign_test_p_unadjusted':sign,'nominal_positive':bool(ci[0,i]>0),'nominal_negative':bool(ci[1,i]<0)}
    def aggregate(rr):return {'games':len(rr),'wins':sum(r['margin']>0 for r in rr),'draws':sum(r['margin']==0 for r in rr),'losses':sum(r['margin']<0 for r in rr),'mean_net_profit':mean(r['net_profit'] for r in rr),'mean_opponent_profit':mean(r['opponent_bank']-3000 for r in rr),'mean_margin':mean(r['margin'] for r in rr),'mean_score':mean(1 if r['margin']>0 else .5 if r['margin']==0 else 0 for r in rr)}
    return {'games':len(rows),'seed_clusters':len(clusters),'contrasts':contrasts,'aggregate':{p:aggregate([r for r in rows if r['policy']==p]) for p in cfg['policies']},'by_opponent':{o:{p:aggregate([r for r in rows if r['policy']==p and r['opponent']==o]) for p in cfg['policies']} for o in cfg['opponents']},'harvest_only_same_semantic_trajectory':all(lookup[s,o,i,'harvest']['semantic_replay_sha256_v2']==lookup[s,o,i,'baseline']['semantic_replay_sha256_v2'] for s in cfg['seeds'] for o in cfg['opponents'] for i in (0,1)),'scope':'Exploratory six seed clusters; five nominal Bonferroni bootstrap contrasts, not universal or per-action causality'},pairs

def run(out):
    cfg=read(out/'registration.json');freeze=read(out/'freeze.json')
    assert sha(out/'registration.json')==freeze['registration_sha256']==(out/'registration.sha256').read_text().strip()
    for n,h in freeze['source_sha256'].items():assert sha(out/'sources'/n)==h
    assert sha(Path(__file__))==freeze['runner_sha256']
    target=out/'test';target.mkdir(exist_ok=False)
    rows=[]
    with Games(out,cfg,'test') as games,ThreadPoolExecutor(max_workers=cfg['workers']) as pool:
        jobs=[pool.submit(games.play,out/'sources'/(p+'.py'),p,o,s,i,target/f'{s}-{o}-{i}-{p}') for s in cfg['seeds'] for o in cfg['opponents'] for i in (0,1) for p in cfg['policies']]
        for f in as_completed(jobs):
            rows.append(f.result());print(f'games {len(rows)}/{len(jobs)}',flush=True)
        rows.sort(key=lambda r:(r['seed'],r['opponent'],r['side'],r['policy']))
        save(target/'ledger.json',rows)
        summary,pairs=statistics(rows,cfg);save(target/'summary.json',summary);save(target/'paired-results.json',pairs)
        repeats=[]
        for p in cfg['repeat_policies']:
            for i in (1,0):
                s=cfg['seeds'][0];rr=games.play(out/'sources'/(p+'.py'),p,'market',s,i,out/'repeat'/f'{p}-{i}')
                original=next(r for r in rows if (r['policy'],r['opponent'],r['seed'],r['side'])==(p,'market',s,i))
                assert rr['banks']==original['banks'] and rr['semantic_replay_sha256_v2']==original['semantic_replay_sha256_v2']
                repeats.append(rr)
        save(out/'reproducibility.json',{'passed':True,'rows':repeats})
    print(json.dumps(summary['contrasts']),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command',choices=['variants','run']);parser.add_argument('folder',type=Path);a=parser.parse_args()
    (variants if a.command=='variants' else run)(a.folder.resolve())
