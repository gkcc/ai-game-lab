"""Scoped official worker adapter for the frozen teacher-task collection."""
import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import threading

from learning.task_anchor_once_v1 import games as core
from . import experiment

_BINDINGS = threading.RLock()
_PLAN = frozenset(('game', 'policy', 'split', 'seed', 'opponent', 'side', 'parity_kind'))
_JOB = _PLAN | frozenset(('bot_sha256', 'rival_sha256', 'task_checkpoint_sha256'))


def task_identity(out, plan, cfg):
    if plan.get('policy') != 'teacher_task':
        raise ValueError('Only teacher_task is permitted')
    return None, None


def validate_plan(plan, cfg, job=False):
    if not isinstance(plan, dict) or set(plan) != (_JOB if job else _PLAN):
        raise ValueError('Unexpected or missing plan fields')
    seed, opponent, side = plan['seed'], plan['opponent'], plan['side']
    if (plan['policy'] != 'teacher_task' or plan['split'] != 'holdout'
            or plan['parity_kind'] is not None or type(seed) is not int
            or type(side) is not int or side not in (0, 1)
            or not isinstance(opponent, str) or seed not in cfg['test_seeds']
            or opponent not in cfg['opponents'] or side not in cfg['seats']):
        raise ValueError('Plan outside frozen collection')
    if plan['game'] != f'holdout-{seed}-{opponent}-{side}-teacher_task':
        raise ValueError('Unexpected game key')
    if job and (plan['task_checkpoint_sha256'] is not None
                or any(not isinstance(plan[k], str) or len(plan[k]) != 64
                       or any(c not in '0123456789abcdef' for c in plan[k])
                       for k in ('bot_sha256', 'rival_sha256'))):
        raise ValueError('Invalid job identities')


def _unique_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('Duplicate plan field')
        result[key] = value
    return result


@contextmanager
def bindings(job_path=None, expected=None, cfg=None):
    with _BINDINGS:
        saved = core.verify, core.task_identity, core.read
        try:
            core.verify, core.task_identity = experiment.verify, task_identity
            if job_path is not None:
                def checked_read(path):
                    if Path(path).resolve() != job_path:
                        return saved[2](path)
                    with Path(path).open(encoding='utf-8-sig') as stream:
                        value = json.load(stream, object_pairs_hook=_unique_pairs)
                    validate_plan(value, cfg, job=True)
                    if value != expected:
                        raise ValueError('Job plan changed before core read')
                    return value
                core.read = checked_read
            yield
        finally:
            core.verify, core.task_identity, core.read = saved


def make_wrapper(out, plan, cfg):
    validate_plan(plan, cfg)
    with bindings():
        return core.wrapper(out, plan, cfg)


def job(out, key):
    out = Path(out).resolve()
    cfg = experiment.ready(out)
    experiment.deadline_check()
    if not isinstance(key, str) or Path(key).name != key:
        raise ValueError('Invalid game key')
    with (out/'job-inputs'/(key+'.json')).open(encoding='utf-8-sig') as stream:
        plan = json.load(stream, object_pairs_hook=_unique_pairs)
    validate_plan(plan, cfg, job=True)
    if plan['game'] != key:
        raise ValueError('Job key differs from plan')
    with bindings((out/'job-inputs'/(key+'.json')).resolve(), plan, cfg):
        return core.job(out, key)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--key', required=True)
    args = parser.parse_args()
    job(args.out, args.key)
