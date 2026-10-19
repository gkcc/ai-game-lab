"""New certificates for retained games; never overwrite a historical freeze/audit."""
import argparse,hashlib,json,multiprocessing
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
from decision.audit_profit_attribution import check

sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf-8'))

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--historical',required=True,type=Path);parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args();old=args.historical.resolve();out=args.output.resolve();game=Path(__file__).resolve().parents[1]
    assert old!=out
    target=out/'official-reexecution-v2.json';assert not target.exists()
    frozen=read(out/'freeze.json');preserve=read(out/'preservation.json')
    for name,digest in frozen['code_sha256'].items():assert sha(game/name)==digest,name
    for name,digest in preserve['historical_run_sha256'].items():assert sha(old/name)==digest,name
    ledger=read(old/'test/ledger.json');repeats=read(old/'reproducibility.json')['rows'];old_audit=read(old/'official-audit.json')
    assert len(ledger)==108 and len(repeats)==4 and len(old_audit['rows'])==112
    jobs=[('test',r,old/'test'/f"{r['seed']}-{r['opponent']}-{r['side']}-{r['policy']}") for r in ledger]
    jobs.extend(('repeat',r,old/'repeat'/f"{r['policy']}-{r['side']}") for r in repeats)
    rows=[]
    with ProcessPoolExecutor(max_workers=4,mp_context=multiprocessing.get_context('spawn')) as pool:
        pending={pool.submit(check,(r,str(folder))):kind for kind,r,folder in jobs}
        for f in as_completed(pending):
            row=f.result()
            previous=next(r for r in old_audit['rows'] if all(r[k]==row[k] for k in ('seed','opponent','side','policy','replay_gzip_sha256')))
            assert row==previous,'Repaired consumer changed historical effects/audit'
            assert row['audit']['official_transitions_reexecuted']==719
            assert row['metrics']['cash_residual']==0
            rows.append(row|dict(kind=pending[f],historical_record_equal=True))
            print(f'official consumer recheck {len(rows)}/112',flush=True)
    rows.sort(key=lambda r:(r['kind'],r['seed'],r['opponent'],r['side'],r['policy'],r['replay_gzip_sha256']))
    assert len({(r['kind'],r['seed'],r['opponent'],r['side'],r['policy']) for r in rows})==112
    for name,digest in frozen['code_sha256'].items():assert sha(game/name)==digest,name
    for name,digest in preserve['historical_run_sha256'].items():assert sha(old/name)==digest,name
    result=dict(passed=True,certificate_version=2,consumer_counting_version=2,games=112,test_games=108,repeats=4,official_transitions=112*719,process_isolated=True,all_historical_records_equal=True,old_certificate_sha256=sha(old/'official-audit.json'),old_freeze_sha256=sha(old/'freeze.json'),old_attribution_sha256=sha(old/'attribution.json'),old_ledger_sha256=sha(old/'test/ledger.json'),new_freeze_sha256=sha(out/'freeze.json'),code_sha256=frozen['code_sha256'],rows=rows,scope='Reexecution of retained games, not new independent matches or a strategy improvement experiment; v1/v2 semantic digests and old files preserved')
    target.write_bytes((json.dumps(result,indent=2)+'\n').encode())

if __name__=='__main__':main()
