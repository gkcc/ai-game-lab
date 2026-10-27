"""Scoped adapter to the accepted official-entry game and replay audit."""
import argparse
from contextlib import contextmanager
from pathlib import Path
import threading

from learning.task_anchor_once_v1 import games as core
from . import experiment


_BINDINGS = threading.RLock()


def task_identity(out, plan, cfg):
    if plan['policy'] == 'teacher_task':
        return None, None
    name = {'baseline': 'task', 'candidate': 'candidate'}[plan['policy']]
    return experiment.ROOT/cfg[name+'_checkpoint'], cfg[name+'_checkpoint_sha256']


@contextmanager
def bindings():
    """Restore both legacy globals even if a wrapper or official job fails."""
    with _BINDINGS:
        saved = core.verify, core.task_identity
        try:
            core.verify, core.task_identity = experiment.verify, task_identity
            yield
        finally:
            core.verify, core.task_identity = saved


def make_wrapper(out, plan, cfg):
    with bindings():
        return core.wrapper(out, plan, cfg)


def job(out, key):
    out = Path(out).resolve()
    experiment.ready(out)
    experiment.deadline_check()
    with bindings():
        return core.job(out, key)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--key', required=True)
    arguments = parser.parse_args()
    job(arguments.out, arguments.key)
