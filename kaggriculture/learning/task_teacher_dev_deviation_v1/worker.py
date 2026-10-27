"""Owned CPU-only development trace worker."""
import json
import os
from pathlib import Path
import time
from .experiment import OUT,ROOT,TRAIN,ACTION,ready,deadline_check,sha,read,write_new


def run(out=OUT):
    from learning.task_bc_v1.model import TaskNetwork
    from .analysis import collect_two,analyze
    from .membership import ENV_KEY,verify_worker
    out=Path(out).resolve(); cfg=ready(out)
    with (out/'started.json').open('r',encoding='utf-8') as stream: marker=json.load(stream)
    verify_worker(os.environ.get(ENV_KEY),marker,sha(out/'freeze.json'))
    if any((out/n).exists() for n in ('worker-started.json','worker-result.json','worker-failure.json','progress.local.jsonl')):
        raise ValueError('Worker attempt consumed')
    last=dict(phase='initializing',forwards_started=0,forwards_completed=0,trace_reads='not_started'); started=time.monotonic(); marked=False
    def check():
        deadline_check()
        if time.monotonic()-started>cfg['limits']['seconds']: raise TimeoutError('Worker deadline')
    try:
        write_new(out/'worker-started.json',dict(attempt=1,freeze_sha256=sha(out/'freeze.json'),retry_allowed=False)); marked=True
        with (out/'progress.local.jsonl').open('x',encoding='utf-8',newline='\n') as stream:
            def progress(value):
                last.update(value); stream.write(json.dumps(last,sort_keys=True)+'\n'); stream.flush(); check()
            progress(dict(phase='reading_traces',trace_reads='started_count_unknown'))
            if sha(TRAIN/'schema.json')!=cfg['schema_sha256']: raise ValueError('Schema changed')
            samples,raw=collect_two(ACTION,cfg['sources'],cfg['sources'],read(TRAIN/'schema.json'),
                max_candidates_per_group=cfg['limits']['max_candidates_per_group'],check=check)
            progress(dict(phase='trace_reads_completed',trace_reads=2))
            new_path=TRAIN/'private/model-final.npz'; old_path=(ROOT/cfg['old_checkpoint']).resolve()
            if not old_path.is_relative_to(ROOT.resolve()): raise ValueError('Old checkpoint path outside root')
            new=TaskNetwork(new_path,cfg['checkpoint_sha256']); old=TaskNetwork(old_path,cfg['old_checkpoint_sha256'])
            if sha(new_path)!=cfg['checkpoint_sha256'] or sha(old_path)!=cfg['old_checkpoint_sha256']:
                raise ValueError('Model changed after loading')
            progress(dict(phase='scoring'))
            result=analyze(samples,new,old,progress)
            if sha(new_path)!=cfg['checkpoint_sha256'] or sha(old_path)!=cfg['old_checkpoint_sha256']:
                raise ValueError('Model changed during scoring')
            check()
        write_new(out/'worker-result.json',dict(passed=True,aggregate=result,raw_reads=raw,freeze_sha256=sha(out/'freeze.json'),
            checkpoint_sha256=cfg['checkpoint_sha256'],old_checkpoint_sha256=cfg['old_checkpoint_sha256'],retry_allowed=False,
            identity_check_limit='Before and after hashes do not exclude transient replacement restored between checks'))
    except BaseException as exc:
        if marked:
            try: write_new(out/'worker-failure.json',dict(error_kind=type(exc).__name__,progress=last,retry_allowed=False))
            except BaseException: pass
        raise


if __name__=='__main__': run()
