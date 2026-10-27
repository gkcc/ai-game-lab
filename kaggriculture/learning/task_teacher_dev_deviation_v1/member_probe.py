"""Synthetic venv/Job membership check; never opens training inputs."""
import json
import os
from pathlib import Path
from .experiment import OUT,ready,sha,environment,write_new
from .membership import ENV_KEY,verify_worker


def run(out=OUT):
    out=Path(out).resolve(); cfg=ready(out)
    with (out/'probe-started.json').open('r',encoding='utf-8') as stream: marker=json.load(stream)
    verify_worker(os.environ.get(ENV_KEY),marker,sha(out/'freeze.json'))
    if environment()!=cfg['environment']: raise ValueError('Probe interpreter identity changed')
    write_new(out/'probe-result.json',dict(passed=True,freeze_sha256=sha(out/'freeze.json'),
        marker_sha256=sha(out/'probe-started.json'),venv_verified=True,job_membership_verified=True,
        training_inputs_read=False,retry_allowed=False))


if __name__=='__main__': run()
