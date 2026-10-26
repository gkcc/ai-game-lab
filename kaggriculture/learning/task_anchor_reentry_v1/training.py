"""Use the unchanged guarded anchor worker through the proven spawn launcher."""
import argparse
from pathlib import Path
import time

from learning.task_anchor_once_v1 import training as method
from learning.task_anchor_once_v1.storage import owned_writes
from learning.task_anchor_spawn_repair_v1.launcher import launch
from .experiment import training_ready,read,sha,write_new


def fit(out):
    out=Path(out).resolve();cfg=training_ready(out)
    method.validate_config(cfg)
    assert not (out/'training-parent/launch-started.json').exists()
    started=time.monotonic()
    # The frozen worker still owns its512MiB artifact guard and16KiB receipt.
    # Only the defective globally guarded parent has been replaced.
    result=launch(out/'training-parent',method._worker,(str(out),cfg,started),
        seconds=method.TRAINING['process_seconds'],receipt_limit=cfg['storage']['training_parent_write_bytes'])
    report,gate=read(out/'fit-result.json'),read(out/'offline-gate.json')
    digest=sha(out/'model/trained.npz')
    assert report['completed'] and report['optimizer_updates']==512
    assert report['checkpoint_sha256']==gate['checkpoint_sha256']==digest
    # Charge the extra adapter receipt to the remainder of the same parent
    # allowance. The previous launcher writes are never refunded or reset.
    remaining=cfg['storage']['training_parent_write_bytes']-result['parent_receipt_bytes_total']
    assert remaining>0 and not result['parent_receipt_refused']
    with owned_writes(out,remaining,allowed_paths=['fit-adapter-result.json']) as budget:
        write_new(out/'fit-adapter-result.json',dict(passed=True,fit_count=1,checkpoint_sha256=digest,
            frozen_worker='learning.task_anchor_once_v1.training._worker',
            mathematical_method_unchanged=True,old_globally_guarded_launcher_called=False,
            launch_receipt_sha256=sha(out/'training-parent/launch-result.json'),
            parent_launch_write_bytes=result['parent_receipt_bytes_total'],
            adapter_write_allowance_bytes=remaining,
            output_schema_lineage='Frozen anchor method schema; run identity is this registration SHA',
            registration_sha256=sha(out/'registration.json')))
    assert result['parent_receipt_bytes_total']+budget.used_bytes<=cfg['storage']['training_parent_write_bytes']
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True)
    fit(parser.parse_args().out)
