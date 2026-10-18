"""Reexecute every registered replay in process-isolated official engines."""
import argparse,json,gzip,hashlib,multiprocessing
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
from decision.profit_attribution import Attribution,reexecute

def check(job):
    row,folder=job;raw=(Path(folder)/'replay.json.gz').read_bytes()
    assert hashlib.sha256(raw).hexdigest()==row['replay_gzip_sha256']
    replay=json.loads(gzip.decompress(raw));measure=Attribution(row['side'])
    with measure.installed():audit=reexecute(replay,trace=True,label_sink=measure.sink)
    assert audit['final_rewards']==row['banks'] and audit['semantic_sha256_v2']==row['semantic_replay_sha256_v2']
    return {k:row[k] for k in ('seed','side','policy','opponent','replay_gzip_sha256')}|dict(audit=audit,metrics=measure.result(replay))

def main():
    p=argparse.ArgumentParser();p.add_argument('folder',type=Path);root=p.parse_args().folder.resolve()
    read=lambda p:json.loads(p.read_text(encoding='utf-8'))
    cfg=read(root/'registration.json');freeze=read(root/'freeze.json')
    for name,h in freeze['instrumentation_sha256'].items():assert hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest()==h
    jobs=[(r,str(root/'test'/f"{r['seed']}-{r['opponent']}-{r['side']}-{r['policy']}")) for r in read(root/'test/ledger.json')]
    jobs.extend((r,str(root/'repeat'/f"{r['policy']}-{r['side']}")) for r in read(root/'reproducibility.json')['rows'])
    rows=[]
    with ProcessPoolExecutor(max_workers=cfg['workers'],mp_context=multiprocessing.get_context('spawn')) as pool:
        for f in as_completed([pool.submit(check,j) for j in jobs]):
            rows.append(f.result());print(f'audit {len(rows)}/{len(jobs)}',flush=True)
    rows.sort(key=lambda r:(r['seed'],r['opponent'],r['side'],r['policy'],r['replay_gzip_sha256']))
    target=root/'official-audit.json';assert not target.exists()
    target.write_text(json.dumps(dict(passed=True,games=len(rows),official_transitions=len(rows)*719,process_isolated=True,rows=rows),indent=2)+'\n',encoding='utf-8')

if __name__=='__main__':main()
