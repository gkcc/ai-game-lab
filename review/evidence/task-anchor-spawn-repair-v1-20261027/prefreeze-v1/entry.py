"""Auditable local lifecycle for one model-free POSIX spawn repair acceptance."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
PUBLIC=ROOT.parent/'publication/ai-game-lab'
sys.path.insert(0,str(ROOT))
from learning.task_anchor_spawn_repair_v1.probe import OUT,PREVIOUS,read,digest,identity
from learning.task_anchor_once_v1.storage import owned_writes
from learning.task_anchor_once_v1.processes import CapturedProcess
from learning.task_anchor_once_v1.experiment import verify as previous_verify


def git(*args):return subprocess.check_output(['git',*args],cwd=PUBLIC)
def size():return sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file())
def deadline():
    start=datetime.fromisoformat(read(OUT/'registration.json')['conservative_stage_start_utc'])
    seconds=(datetime.now(timezone.utc)-start).total_seconds()
    assert 0<=seconds<3600,'Registered repair wall deadline'
    return seconds


def save(name,value):
    # Exclusive JSON files are never rewritten: existing sizes count toward
    # the root's cumulative512KiB part of the registered1MiB run output cap.
    names=read(OUT/'registration.json').get('root_owned_names',[]) if (OUT/'registration.json').exists() else []
    used=sum((OUT/n).stat().st_size for n in names if (OUT/n).is_file())
    assert name in names or not (OUT/'registration.json').exists()
    budget=owned_writes(OUT,524288-used,allowed_paths=[name])
    with budget:
        with (OUT/name).open('x',encoding='utf-8',newline='\n') as stream:
            json.dump(value,stream,sort_keys=True,indent=2,allow_nan=False);stream.write('\n')
    assert size()<=1048576


def prepare():
    assert not OUT.exists() and not git('status','--porcelain').strip()
    old=read(PREVIOUS/'publication.json')
    assert git('rev-parse','HEAD').decode().strip()==old['commit']
    previous_verify(PREVIOUS,history=True)
    names=['registration.json','freeze.json','implementation-verification.json','pre-execution-commit.json',
        'dispatch-started.json','dispatch-result.json','dispatch-failed.json','root-result-acceptance.json',
        'preregistration-privacy.json','result-privacy.json','old-public-preservation.json','publication.json',
        'root-next-learning-decision.json']
    cfg=dict(experiment='task-anchor-spawn-repair-v1',baseline_head=old['commit'],
        prior_freeze_sha256=digest(PREVIOUS/'freeze.json'),decision_sha256=digest(PREVIOUS/'root-next-learning-decision.json'),
        conservative_stage_start_utc=old['created_utc'],total_stage_seconds=3600,
        real_spawn_starts=1,process_seconds=30,work_seconds=15,cleanup_seconds=15,target_seconds=5,
        total_output_bytes=1048576,parent_receipt_limit=65536,root_output_limit=524288,
        independent_output_limit=262144,captured_output_limit=65536,root_owned_names=names,
        prior_artifact_sha256={p.relative_to(PREVIOUS).as_posix():digest(p) for p in PREVIOUS.rglob('*') if p.is_file()},
        default_bot_sha256=digest(ROOT/'main.py'),model_array_decodes=0,cache_array_decodes=0,
        neural_rows=0,gpu_operations=0,training_fits=0,new_games=0,official_steps=0,
        previous_goal_turn='progress: published the preserved launch failure and independent repair decision',
        no_retry=True,local_registration_not_external_timestamp=True)
    OUT.mkdir()
    save('registration.json',cfg)
    sources={p.relative_to(ROOT).as_posix():digest(p) for p in (ROOT/'learning/task_anchor_spawn_repair_v1').iterdir() if p.is_file()}
    save('freeze.json',dict(registration_sha256=digest(OUT/'registration.json'),new_source_sha256=sources,
        entry_source_sha256=digest(__file__),previous_sha_chain=cfg['prior_freeze_sha256']))
    save('implementation-verification.json',dict(passed=True,focused_tests=7,skips=0,
        real_spawn_starts=0,neural_rows=0,gpu_operations=0,owned_test_temporary_directories_removed=True))
    identity(OUT);deadline();print(dict(prepared=True,freeze_sha256=digest(OUT/'freeze.json')))


def export_pre():
    identity(OUT);deadline();review=read(OUT/'pre-review.json')
    assert review['passed'] and review['freeze_sha256']==digest(OUT/'freeze.json')
    for rel in read(OUT/'freeze.json')['new_source_sha256']:
        dst=PUBLIC/'kaggriculture'/rel;assert not dst.exists();dst.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(ROOT/rel,dst)
    for name in ['registration.json','freeze.json','implementation-verification.json','pre-review.json']:
        dst=PUBLIC/'review/evidence'/OUT.name/name;assert not dst.exists();dst.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(OUT/name,dst)
    print(dict(exported=True))


def ready():
    identity(OUT);deadline();receipt=read(OUT/'pre-execution-commit.json')
    assert digest(__file__)==read(OUT/'freeze.json')['entry_source_sha256']
    for name in ['registration.json','freeze.json','pre-review.json']:
        assert git('show',receipt['commit']+':review/evidence/'+OUT.name+'/'+name)==(OUT/name).read_bytes()
    for rel,value in read(OUT/'freeze.json')['new_source_sha256'].items():
        assert hashlib.sha256(git('show',receipt['commit']+':kaggriculture/'+rel)).hexdigest()==value
    assert digest(ROOT/'main.py')==read(OUT/'registration.json')['default_bot_sha256']


def run():
    ready();cfg=read(OUT/'registration.json')
    assert not (OUT/'dispatch-started.json').exists()
    assert size()+262144<1048576 and shutil.disk_usage(OUT).free>4294967296+1048576
    save('dispatch-started.json',dict(attempt=1,freeze_sha256=digest(OUT/'freeze.json'),retry_allowed=False))
    launch=read(PREVIOUS/'gpu-launch.local.json');child=None;started=time.monotonic();error=None
    try:
        child=CapturedProcess(['wsl.exe','-d',launch['distro'],'--cd',launch['workdir'],'--',launch['python'],
            '-B','-m','learning.task_anchor_spawn_repair_v1.probe','guard'],cwd=ROOT,
            log=OUT/'process.local.log',limit_bytes=65536)
        child.process.wait(timeout=30);capture=child.finish()
        process_seconds=time.monotonic()-started
        assert child.process.returncode==0 and process_seconds<=30
        assert read(OUT/'guard-result.json')['completed'] and read(OUT/'probe-result.json')['passed']
        ready();previous_verify(PREVIOUS,history=True)
        save('dispatch-result.json',dict(passed=True,capture=capture,process_seconds=process_seconds,
            dispatch_with_postchecks_seconds=time.monotonic()-started,
            stage_elapsed_seconds=deadline(),guard_sha256=digest(OUT/'guard-result.json'),
            probe_sha256=digest(OUT/'probe-result.json'),owned_bytes_before_receipt=size()))
    except BaseException as exc:
        error=exc
        raise
    finally:
        if child is not None:
            # The Linux supervisor owns cleanup even if this Windows wait fails.
            # Do not kill its Windows transport and call that Linux cleanup.
            if child.process.poll() is None:child.process.wait(timeout=15)
            child.thread.join(timeout=5)
            if child.thread.is_alive():raise RuntimeError('Owned capture reader remains alive')
        if error is not None:
            save('dispatch-failed.json',dict(kind=type(error).__name__,retry_allowed=False,
                partial_work_not_refunded=True,stage_failed=True))
    print(dict(passed=True,real_spawn_starts=1,training_fits=0,neural_rows=0))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['prepare','export-pre','run'])
    args=parser.parse_args();{'prepare':prepare,'export-pre':export_pre,'run':run}[args.command]()
