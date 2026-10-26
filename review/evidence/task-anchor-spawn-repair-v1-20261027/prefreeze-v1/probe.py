"""One fixed model-free real WSL spawn through the repaired launcher."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

from learning.task_anchor_once_v1.storage import owned_writes, UnsafeWriteError
from learning.task_dagger_once_v1.gpu_guard import supervise
from .launcher import launch

ROOT=Path(__file__).resolve().parents[2]
PREVIOUS=ROOT/'artifacts/task-anchor-once-v1-20261027'
OUT=ROOT/'artifacts/task-anchor-spawn-repair-v1-20261027'
PAYLOAD={'test':'startup pipe remains available outside artifact scopes','value':17}


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8',newline='\n') as stream:
        json.dump(value,stream,sort_keys=True,indent=2,allow_nan=False);stream.write('\n')


def identity(out):
    out=Path(out).resolve()
    if out!=OUT.resolve():raise ValueError('Only the registered repair output is allowed')
    cfg=read(out/'registration.json');freeze=read(out/'freeze.json')
    assert digest(out/'registration.json')==freeze['registration_sha256']
    for rel,value in freeze['new_source_sha256'].items():assert digest(ROOT/rel)==value
    assert digest(PREVIOUS/'freeze.json')==cfg['prior_freeze_sha256']
    assert digest(PREVIOUS/'root-next-learning-decision.json')==cfg['decision_sha256']
    for rel,value in read(PREVIOUS/'freeze.json')['runtime_files'].items():assert digest(ROOT/rel)==value
    for name,value in cfg['prior_artifact_sha256'].items():assert digest(PREVIOUS/name)==value
    assert cfg['real_spawn_starts']==1 and cfg['total_output_bytes']==1048576
    assert cfg['process_seconds']==30 and cfg['work_seconds']==15 and cfg['target_seconds']==5
    return cfg


def synthetic_target(folder,payload):
    """Importable top-level target; no models, numerical libraries or engine."""
    if payload!=PAYLOAD:raise AssertionError('Startup payload differs')
    folder=Path(folder)
    with owned_writes(folder,4096,allowed_paths=['marker.json']) as budget:
        write(folder/'marker.json',dict(reached=True,payload=payload))
    with owned_writes(folder,4096,allowed_paths=['target-storage.json']):
        write(folder/'target-storage.json',dict(passed=True,output=budget.snapshot(),
            marker_sha256=digest(folder/'marker.json')))


def worker(out):
    cfg=identity(out)
    assert (out/'guard-started.json').exists()
    review=read(out/'pre-review.json');receipt=read(out/'pre-execution-commit.json')
    assert review['passed'] and review['freeze_sha256']==digest(out/'freeze.json')
    assert receipt['freeze_sha256']==digest(out/'freeze.json')
    # Global descriptor rejection is unchanged; use harmless non-opened fake
    # descriptor 987654. The path branch also proves out-of-scope rejection.
    forbidden=[lambda:open(987654,'wb',closefd=False),
               lambda:(out.parent/'not-created-by-spawn-repair.tmp').open('wb')]
    with owned_writes(out,4096,allowed_paths=['rejection-check.json']) as budget:
        caught=0
        for call in forbidden:
            try:
                value=call();value.close()
            except UnsafeWriteError:caught+=1
        assert caught==2
        write(out/'rejection-check.json',dict(passed=True,rejections=caught,
            arbitrary_descriptor_permission_added=False))
    result=launch(out/'parent',synthetic_target,(str(out/'target'),PAYLOAD),
        seconds=cfg['target_seconds'],receipt_limit=65536)
    first=read(out/'parent/launch-started.json');last=read(out/'parent/launch-result.json')
    target=read(out/'target/target-storage.json');marker=read(out/'target/marker.json')
    assert first['attempt']==1 and last['passed'] and marker['payload']==PAYLOAD and marker['reached']
    assert target['passed'] and target['marker_sha256']==digest(out/'target/marker.json')
    assert result['parent_receipt_bytes_total']==sum(p.stat().st_size for p in (out/'parent').iterdir())
    assert result['parent_receipt_bytes_total']<=65536 and not result['parent_receipt_refused']
    with owned_writes(out,8192,allowed_paths=['probe-result.json']):
        write(out/'probe-result.json',dict(passed=True,real_spawn_starts=1,real_target_reached=1,
            launch=result,target_output=target['output'],rejection_check_write_bytes=budget.used_bytes,
            parent_receipt_sha256={p.name:digest(p) for p in (out/'parent').iterdir()},
            target_marker_sha256=digest(out/'target/marker.json'),
            training_fits=0,model_array_decodes=0,cache_array_decodes=0,neural_rows=0,
            gpu_operations=0,official_games=0,official_steps=0))


def guard(out):
    started=time.monotonic();cfg=identity(out)
    review=read(out/'pre-review.json');receipt=read(out/'pre-execution-commit.json')
    assert review['passed'] and review['freeze_sha256']==digest(out/'freeze.json')
    assert receipt['freeze_sha256']==digest(out/'freeze.json')
    assert os.name=='posix' and not (out/'guard-started.json').exists()
    with owned_writes(out,8192,allowed_paths=['guard-started.json']):
        write(out/'guard-started.json',dict(real_spawn_attempt_authorized=1,process_seconds=30,work_seconds=15))
    remaining=min(15,30-15-(time.monotonic()-started));assert remaining>0
    # Supervisor control/IPC is deliberately outside artifact scopes too.
    result=supervise([sys.executable,'-B','-m',__package__+'.probe','worker','--out',str(out)],
        remaining,lambda:(out/'cancel.local.json').exists())
    result['child_supervision_seconds']=result['elapsed_seconds']
    result['elapsed_seconds']=time.monotonic()-started
    result['within_total_deadline']=result['elapsed_seconds']<=30
    result['completed']=result['completed'] and result['within_total_deadline']
    with owned_writes(out,8192,allowed_paths=['guard-result.json']):write(out/'guard-result.json',result)
    if not result['completed']:raise RuntimeError('Single registered synthetic entry failed; no retry')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['guard','worker'])
    parser.add_argument('--out',type=Path,default=OUT);args=parser.parse_args()
    (guard if args.command=='guard' else worker)(args.out.resolve())
