"""Fixed recorded-action official reexecution; no policy or network calls."""
import argparse
from datetime import datetime,timezone
import gzip
import hashlib
import json
import os
from pathlib import Path
import sys
import time

os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
HERE=Path(__file__).resolve().parent
OUT=HERE.parent
ROOT=OUT.parent.parent
sys.path.insert(0,str(ROOT))
from learning.task_anchor_once_v1.storage import owned_writes


def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def now():return datetime.now(timezone.utc)
def name(key,suffix):return 'replay-'+key+'-'+suffix+'.json'


def save(key,suffix,value):
    target=name(key,suffix)
    assert sum(p.stat().st_size for p in HERE.rglob('*') if p.is_file())+32768<=268435456
    with owned_writes(HERE,32768,allowed_paths=[target]):
        with (HERE/target).open('x',encoding='utf-8',newline='\n') as stream:
            json.dump(value,stream,sort_keys=True,indent=2,allow_nan=False);stream.write('\n')


def identities(key):
    cfg=read(OUT/'registration.json');f=read(OUT/'freeze.json')
    allowed=[f"dev-{cfg['validation_seeds'][0]}-round2-0-candidate",
             f"holdout-{cfg['test_seeds'][0]}-market-0-candidate",
             f"holdout-{cfg['test_seeds'][0]}-market-0-teacher_task"]
    assert key in allowed
    prefix=OUT/'job-results'/key
    row=read(Path(str(prefix)+'.row.json'));complete=read(Path(str(prefix)+'.complete.json'))
    assert complete['passed'] and complete['row_sha256']==sha(Path(str(prefix)+'.row.json'))
    assert complete['certificate_sha256']==sha(Path(str(prefix)+'.audit.json'))
    assert row['game']==key and row['passed'] and row['action_turns']==719 and row['states']==720
    assert row['agent_errors']==0 and row['status']==['DONE','DONE']
    rp=OUT/'games'/key/'replay.json.gz'
    assert sha(rp)==row['replay_gzip_sha256']
    sources=['decision/full_audit.py','decision/verify_mechanisms.py','decision/semantics.py','decision/execution.py']
    assert all(sha(ROOT/p)==f['runtime_files'][p] for p in sources)
    from learning.task_entry_repair_v1.experiment import installed_files
    installed=installed_files()
    assert sha(installed['rules'])==cfg['engine_sha256']==row['engine_sha256']
    assert sha(installed['loader'])==cfg['official_loader_sha256']
    return row,rp,{p:f['runtime_files'][p] for p in sources}


def worker(key):
    reservation=read(HERE/name(key,'reservation'))
    assert sha(__file__)==reservation['reviewer_source_sha256']
    assert now()<datetime.fromisoformat(reservation['deadline_utc'])
    row,path,sources=identities(key)
    assert sha(path)==reservation['replay_sha256']
    from decision.full_audit import reexecute
    with gzip.open(path,'rt',encoding='utf-8') as stream:replay=json.load(stream)
    assert replay['info']['seed']==row['seed'] and len(replay['steps'])==720
    # The whole failed or successful invocation is charged before engine calls.
    save(key,'debit',dict(official_transitions_debited=719,completed_before=0,
         reservation_sha256=sha(HERE/name(key,'reservation')),created_utc=now().isoformat(),retry_allowed=False))
    started=time.monotonic()
    try:
        result=reexecute(replay,trace=False,label_sink=None)
        assert result['passed'] and result['official_transitions_reexecuted']==719
        assert result['semantic_sha256']==row['semantic_replay_sha256']
        assert result['semantic_sha256_v2']==row['semantic_replay_sha256_v2']
        assert result['final_rewards']==row['rewards']==row['banks']
        assert sha(path)==reservation['replay_sha256']
        assert now()<datetime.fromisoformat(reservation['deadline_utc'])
        save(key,'result',dict(passed=True,game=key,official=result,official_transitions_debited=719,
            official_transitions_completed=719,policy_callbacks=0,neural_rows=0,new_games=0,
            worker_seconds=time.monotonic()-started,reservation_sha256=sha(HERE/name(key,'reservation')),
            row_sha256=reservation['row_sha256'],source_sha256=sha(__file__),frozen_source_sha256=sources))
    except BaseException as exc:
        save(key,'failure',dict(passed=False,error_kind=type(exc).__name__,official_transitions_debited=719,
            official_transitions_completed=None,partial_work_not_refunded=True,retry_allowed=False))
        raise


def dispatch(key):
    from learning.task_anchor_once_v1.processes import CapturedProcess
    row,path,sources=identities(key)
    clock=read(HERE/'numeric-reservation.json')
    assert now()<datetime.fromisoformat(clock['deadline_utc'])
    previous=list(HERE.glob('replay-*-reservation.json'))
    assert len(previous)<3 and all(read(p)['official_transitions_reserved']==719 for p in previous)
    save(key,'reservation',dict(schema='independent-anchor-fixed-official-replay-v1',game=key,
        created_utc=now().isoformat(),deadline_utc=clock['deadline_utc'],official_transitions_reserved=719,
        cumulative_reserved_transitions=719*(len(previous)+1),total_limit=2157,
        recorded_actions_only=True,policy_callbacks_authorized=0,neural_rows_authorized=0,new_games_authorized=0,
        reviewer_source_sha256=sha(__file__),frozen_source_sha256=sources,replay_sha256=sha(path),
        row_sha256=sha(OUT/'job-results'/(key+'.row.json')),registration_sha256=sha(OUT/'registration.json'),
        retry_allowed=False,replacement_allowed=False,process_seconds=900,native_log_bytes=1048576))
    child=None;error=None;capture=None;started=time.monotonic()
    try:
        child=CapturedProcess([sys.executable,'-B',str(Path(__file__).resolve()),key,'--worker'],
            cwd=ROOT,log=HERE/('replay-'+key+'.local.log'),limit_bytes=1048576)
        while child.process.poll() is None:
            assert now()<datetime.fromisoformat(clock['deadline_utc']) and time.monotonic()-started<900
            child.check();time.sleep(.2)
        capture=child.finish();assert child.process.returncode==0
    except BaseException as exc:error=exc
    finally:
        if child is not None:
            try:child.stop()
            except BaseException:
                if error is None:error=RuntimeError('Independent replay child cleanup failed')
    reaped=child is not None and child.process.poll() is not None
    reader=child is not None and not child.thread.is_alive()
    passed=error is None and reaped and reader
    save(key,'dispatch',dict(passed=passed,error_kind=type(error).__name__ if error else None,
         child_reaped=reaped,reader_exited=reader,capture=capture,elapsed_seconds=time.monotonic()-started))
    if error is not None:raise error
    result=read(HERE/name(key,'result'))
    print(json.dumps(dict(passed=result['passed'],game=key,completed_transitions=719,
         result_sha256=sha(HERE/name(key,'result')),cleanup_passed=passed)),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('key');parser.add_argument('--worker',action='store_true')
    args=parser.parse_args();worker(args.key) if args.worker else dispatch(args.key)
