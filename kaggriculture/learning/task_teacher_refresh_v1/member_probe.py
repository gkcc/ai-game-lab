"""Synthetic interpreter and named-Job probe; never opens training inputs."""
import os
from .experiment import OUT, environment, read, ready, sha, write_new
from learning.task_teacher_dev_deviation_v1.membership import ENV_KEY, verify_worker


def run():
    cfg = ready(OUT)
    marker = read(OUT/'probe-started.json')
    verify_worker(os.environ.get(ENV_KEY),marker,sha(OUT/'freeze.json'))
    if environment() != cfg['environment']:
        raise ValueError('Probe interpreter changed')
    write_new(OUT/'probe-result.json',dict(passed=True,freeze_sha256=sha(OUT/'freeze.json'),
        marker_sha256=sha(OUT/'probe-started.json'),venv_verified=True,job_membership_verified=True,
        training_inputs_read=False,retry_allowed=False))


if __name__ == '__main__':
    run()
