"""Reuse the verified Linux session supervisor for the single anchor fit."""
import argparse
from pathlib import Path
import sys
import time
from learning.task_dagger_once_v1.gpu_guard import supervise
from .experiment import training_ready,write_new
from .storage import owned_writes


def run(out):
    started=time.monotonic();out=Path(out).resolve();cfg=training_ready(out)
    limits=cfg['gpu_guard']
    assert not (out/'training-cancel.local.json').exists(),'Cancelled before fit dispatch'
    with owned_writes(out,cfg['storage']['supervisor_write_bytes'],
            allowed_paths=['gpu-guard-started.json','gpu-guard-result.json']):
        write_new(out/'gpu-guard-started.json',dict(work_seconds=limits['work_seconds'],total_seconds=limits['process_seconds']))
        remaining=min(limits['work_seconds'],limits['process_seconds']-limits['cleanup_seconds']-(time.monotonic()-started))
        assert remaining>0
        result=supervise([sys.executable,'-B','-m','learning.task_anchor_once_v1.training','--out',str(out)],
            remaining,lambda:(out/'training-cancel.local.json').exists())
        result['child_supervision_seconds']=result['elapsed_seconds']
        result['elapsed_seconds']=time.monotonic()-started
        result['within_total_deadline']=result['elapsed_seconds']<=limits['process_seconds']
        result['completed']=result['completed'] and result['within_total_deadline']
        write_new(out/'gpu-guard-result.json',result)
    if not result['completed']:raise RuntimeError('Owned GPU session did not complete in its registered boundary')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True)
    run(parser.parse_args().out)
