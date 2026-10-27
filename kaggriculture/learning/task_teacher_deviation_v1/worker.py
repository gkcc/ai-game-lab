"""CPU-only worker for one preregistered training-set comparison."""
import json
import os
from pathlib import Path
import time
from .experiment import OUT,TRAIN,ready,deadline_check,sha,write_new


def run(out=OUT):
    from learning.task_teacher_choice_bc_v1 import numeric
    from learning.task_bc_v1.model import TaskNetwork
    from .analysis import analyze
    out=Path(out).resolve(); cfg=ready(out)
    from .membership import ENV_KEY, verify_worker
    verify_worker(os.environ.get(ENV_KEY),read_runner_marker(out),sha(out/'freeze.json'))
    if any((out/n).exists() for n in ('worker-started.json','worker-result.json','worker-failure.json','progress.local.jsonl')):
        raise ValueError('Worker attempt consumed')
    last=dict(forwards_started=0,forwards_completed=0); marker=False; started=time.monotonic()
    def check():
        deadline_check()
        if time.monotonic()-started>cfg['limits']['seconds']: raise TimeoutError('Worker deadline')
    try:
        write_new(out/'worker-started.json',dict(attempt=1,freeze_sha256=sha(out/'freeze.json'),retry_allowed=False)); marker=True
        with (out/'progress.local.jsonl').open('x',encoding='utf-8',newline='\n') as stream:
            def progress(value):
                last.update(value)
                stream.write(json.dumps(last,sort_keys=True)+'\n'); stream.flush(); check()
            progress({})
            limits=cfg['limits']; cache=TRAIN/'private/train.npz'; model_path=TRAIN/'private/model-final.npz'
            arrays=numeric.read(cache,limits['max_groups'],limits['max_candidates'],limits['max_archive_bytes'],
                                limits['max_uncompressed_bytes'],cfg['cache_sha256'])
            if set(int(s) for s in arrays['seed'])!=set(cfg['train_seeds']): raise ValueError('Training seed role changed')
            old_path=(TRAIN.parents[1]/cfg['old_checkpoint']).resolve()
            if not old_path.is_relative_to(TRAIN.parents[1].resolve()): raise ValueError('Old model path outside root')
            check(); model=TaskNetwork(model_path,cfg['checkpoint_sha256']); old=TaskNetwork(old_path,cfg['old_checkpoint_sha256'])
            if sha(model_path)!=cfg['checkpoint_sha256'] or sha(old_path)!=cfg['old_checkpoint_sha256']:
                raise ValueError('Model changed after load')
            result=analyze(arrays,model,old,progress)
            if sha(model_path)!=cfg['checkpoint_sha256'] or sha(old_path)!=cfg['old_checkpoint_sha256']:
                raise ValueError('Model changed after scoring')
            check()
        write_new(out/'worker-result.json',dict(passed=True,aggregate=result,freeze_sha256=sha(out/'freeze.json'),
            cache_sha256=cfg['cache_sha256'],checkpoint_sha256=cfg['checkpoint_sha256'],old_checkpoint_sha256=cfg['old_checkpoint_sha256'],retry_allowed=False,
            identity_check_limit='Model before and after hashes cannot exclude transient replacement restored between checks',
            memory_scope='Immutable archive snapshot and decoded arrays add RAM; Job monitoring is sampled'))
    except BaseException as exc:
        if marker:
            try: write_new(out/'worker-failure.json',dict(error_kind=type(exc).__name__,progress=last,retry_allowed=False))
            except BaseException: pass
        raise


def read_runner_marker(out):
    path=Path(out)/'started.json'
    with path.open('r',encoding='utf-8') as stream:
        return json.load(stream)


if __name__=='__main__': run()
