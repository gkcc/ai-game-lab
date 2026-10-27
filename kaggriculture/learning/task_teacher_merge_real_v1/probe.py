"""Synthetic Job membership/venv probe; never reads training inputs."""
import os
from pathlib import Path
from learning.task_teacher_dev_deviation_v1.membership import ENV_KEY,verify_worker
from .experiment import OUT,ROOT,read,sha,ready,write_new,environment


def run(out=OUT):
    out=Path(out).resolve(); cfg=ready(out)
    marker=read(out/'probe-started.json')
    verify_worker(os.environ.get(ENV_KEY),marker,sha(out/'freeze.json'))
    if environment()!=cfg['environment']: raise ValueError('Probe environment changed')
    write_new(out/'probe-result.json',dict(passed=True,venv_verified=True,job_membership_verified=True,
        training_inputs_read=False,freeze_sha256=sha(out/'freeze.json'),marker_sha256=sha(out/'probe-started.json'),retry_allowed=False))


if __name__=='__main__': run()
