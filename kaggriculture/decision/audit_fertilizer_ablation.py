"""Process-isolated official full reexecution of the registered ablation."""
import argparse,json,gzip,hashlib,multiprocessing
from pathlib import Path
from collections import Counter
from concurrent.futures import ProcessPoolExecutor,as_completed
from decision.full_audit import reexecute

def check(job):
    row,folder=job
    raw=(Path(folder)/'replay.json.gz').read_bytes()
    assert hashlib.sha256(raw).hexdigest()==row['replay_gzip_sha256']
    replay=json.loads(gzip.decompress(raw));effects=Counter()
    def sink(r):
        if r['player']!=row['side']:return
        if r['phase']=='unit':
            op=r['effective'][0]
            if op in ('FERTILIZE','PICKUP','HARVEST'):effects[op+':'+r['status']]+=1
            tile=r.get('tile_before')
            if op=='HARVEST' and r['status']=='executed' and isinstance(tile,dict) and tile.get('crop') in ('TOMATO','STRAWBERRY'):
                effects['ongoing_harvest_executed']+=1
                effects['ongoing_harvest_units']+=tile.get('yield_units',0)
        if r['phase']=='market':
            a=r['requested']
            if isinstance(a,list) and len(a)>=2 and a[1]=='FERTILIZER':
                if a[0]=='BUY_PRODUCT':
                    effects['fertilizer_bought']+=r.get('executed_units',0);effects['fertilizer_cost']-=r.get('cash_delta',0)
                elif a[0]=='SELL':
                    effects['fertilizer_sold']+=r.get('executed_units',0);effects['fertilizer_sale_cash']+=r.get('cash_delta',0)
    audit=reexecute(replay,trace=True,label_sink=sink)
    assert audit['final_rewards']==row['banks'] and audit['semantic_sha256_v2']==row['semantic_replay_sha256_v2']
    last=replay['steps'][-1][row['side']]['observation']['private']
    effects['fertilizer_terminal_stock']=last['shed'].get('FERTILIZER',0)+sum(x.get('FERTILIZER',0) for x in last['inventories'])
    return {k:row[k] for k in ('seed','side','policy','opponent','replay_gzip_sha256')}|{'audit':audit,'effects':dict(effects)}

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('folder',type=Path);a=p.parse_args();root=a.folder.resolve()
    read=lambda p:json.loads(p.read_text(encoding='utf-8'))
    cfg=read(root/'registration.json');freeze=read(root/'freeze.json')
    assert hashlib.sha256(Path(__file__).read_bytes()).hexdigest()==freeze['audit_sha256']
    jobs=[(r,str(root/'test'/f"{r['seed']}-{r['opponent']}-{r['side']}-{r['policy']}")) for r in read(root/'test/ledger.json')]
    jobs.extend((r,str(root/'repeat'/f"{r['policy']}-{r['side']}")) for r in read(root/'reproducibility.json')['rows'])
    rows=[]
    with ProcessPoolExecutor(max_workers=cfg['workers'],mp_context=multiprocessing.get_context('spawn')) as pool:
        for f in as_completed([pool.submit(check,j) for j in jobs]):
            rows.append(f.result());print(f'official audit {len(rows)}/{len(jobs)}',flush=True)
    rows.sort(key=lambda r:(r['seed'],r['opponent'],r['side'],r['policy'],r['replay_gzip_sha256']))
    for n,h in freeze['source_sha256'].items():assert hashlib.sha256((root/'sources'/n).read_bytes()).hexdigest()==h
    report={'passed':True,'games':len(rows),'official_transitions':len(rows)*719,'process_isolated_trace':True,'rows':rows,'boundary':'Official execution and whole-game financial effects, not per-action marginal-profit attribution'}
    target=root/'official-audit.json';assert not target.exists();target.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')

if __name__=='__main__':main()
