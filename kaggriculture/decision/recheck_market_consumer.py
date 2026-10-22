"""Reexecute retained 32+1 games under a new freeze; keep all old evidence."""
import argparse,hashlib,json,multiprocessing
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
from decision.audit_market_teacher import check
from decision.full_audit import rules

sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf-8'))

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--historical',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();old=a.historical.resolve();out=a.output.resolve();game=Path(__file__).resolve().parents[1]
    assert old!=out and old not in out.parents
    target=out/'official-reexecution-v3.json';assert not target.exists()
    freeze=read(out/'freeze.json');preserve=read(out/'preservation.json')
    def integrity():
        for n,h in freeze['code_sha256'].items():assert sha(game/n)==h,n
        for n,h in preserve['historical_run_sha256'].items():assert sha(old/n)==h,n
        for n,h in preserve['policy_sha256'].items():assert sha(game/n)==h,n
        assert sha(Path(rules.__file__))==freeze['engine_sha256']
        assert not (old/'test').exists()
    integrity()
    ledger=read(old/'development/ledger.json');prior=read(old/'development-audit.json')
    entry=read(old/'file-entry-check.json')['official_repeat_certificate']
    assert len(ledger)==len(prior['rows'])==32
    jobs=[(r,str(old/'development'/f"{r['seed']}-{r['opponent']}-{r['side']}-{r['policy']}"),str(old/'sources'),'development') for r in ledger]
    r=next(r for r in ledger if all(r[k]==entry[k] for k in ('seed','opponent','side','policy')))
    repeat={**r,**{k:entry[k] for k in ('bot_sha256','rival_sha256','replay_gzip_sha256')}}
    jobs.append((repeat,str(old/'file-entry-check'),str(old/'file-entry-check/sources'),'file_entry_repeat'))
    key=lambda r:tuple(r[k] for k in ('kind','seed','opponent','side','policy'))
    previous={key(r):r for r in [*prior['rows'],entry]};assert len(previous)==33
    rows=[]
    with ProcessPoolExecutor(max_workers=4,mp_context=multiprocessing.get_context('spawn')) as pool:
        for f in as_completed([pool.submit(check,j) for j in jobs]):
            row=f.result();assert row==previous[key(row)],'repaired consumer changed historical certificate fields'
            assert row['audit']['official_transitions_reexecuted']==719 and row['metrics']['cash_residual']==0
            rows.append(row);print(f'official market consumer recheck {len(rows)}/33',flush=True)
    rows.sort(key=key);assert len({key(r) for r in rows})==33
    integrity()
    result=dict(passed=True,certificate_version=3,games=33,development_games=32,file_entry_repeats=1,
        official_transitions=23727,process_isolated=True,workers_closed=True,all_historical_records_equal=True,
        old_development_certificate_sha256=sha(old/'development-audit.json'),old_entry_certificate_sha256=sha(old/'file-entry-check.json'),
        old_development_freeze_sha256=sha(old/'development-freeze.json'),old_ledger_sha256=sha(old/'development/ledger.json'),
        repair_freeze_sha256=sha(out/'freeze.json'),preservation_sha256=sha(out/'preservation.json'),code_sha256=freeze['code_sha256'],
        rows=rows,scope='Retained official replays reexecuted with new consumers; not new matches or unseen evaluation. Historical hashes, rewards, effects, contract counts and accounting records all equal. No holdout seeds run.')
    target.write_bytes((json.dumps(result,indent=2)+'\n').encode())

if __name__=='__main__':main()
