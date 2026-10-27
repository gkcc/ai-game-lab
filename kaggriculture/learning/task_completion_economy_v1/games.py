"""Bounded official worker adapter for the two-seed economy panel."""
import argparse
from contextlib import contextmanager
import json
from pathlib import Path, PurePosixPath
import threading

from learning.task_anchor_once_v1 import games as core
from . import experiment

_BINDINGS = threading.RLock()
_PLAN = frozenset(('game', 'policy', 'split', 'seed', 'opponent', 'side', 'parity_kind'))
_JOB = _PLAN | frozenset(('bot_sha256', 'rival_sha256', 'task_checkpoint_sha256'))


def _digest(value):
    return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def task_identity(out, plan, cfg):
    policy = plan.get('policy')
    if policy == 'teacher_task': return None, None
    if policy != 'candidate': raise ValueError('Unsupported policy')
    rel, digest = cfg['candidate_checkpoint'], cfg['candidate_checkpoint_sha256']
    if not isinstance(rel, str) or not rel or '\\' in rel or not _digest(digest):
        raise ValueError('Invalid checkpoint identity')
    posix = PurePosixPath(rel)
    if posix.is_absolute() or posix.as_posix() != rel or any(part in ('.', '..') or ':' in part for part in posix.parts):
        raise ValueError('Invalid checkpoint path')
    root = experiment.ROOT.resolve()
    path = (root / rel).resolve()
    if not path.is_relative_to(root) or path == root:
        raise ValueError('Checkpoint outside root')
    return path, digest


def validate_plan(plan, cfg, job=False):
    if not isinstance(plan, dict) or set(plan) != (_JOB if job else _PLAN):
        raise ValueError('Unexpected or missing plan fields')
    seeds, opponents, seats = cfg['test_seeds'], cfg['opponents'], cfg['seats']
    if (not isinstance(seeds, (list, tuple)) or len(seeds) != 2
            or any(type(s) is not int or not 0 <= s <= 0xffffffff for s in seeds)
            or len(set(seeds)) != 2 or not isinstance(opponents, (list, tuple))
            or list(opponents) != ['round2']
            or not isinstance(seats, (list, tuple)) or sorted(seats) != [0, 1]
            or any(type(s) is not int for s in seats)):
        raise ValueError('Invalid frozen panel')
    seed, opponent, side, policy = plan['seed'], plan['opponent'], plan['side'], plan['policy']
    if (policy not in ('candidate', 'teacher_task') or plan['split'] != 'holdout'
            or plan['parity_kind'] is not None or type(seed) is not int or seed not in seeds
            or type(side) is not int or side not in (0, 1) or opponent != opponents[0]):
        raise ValueError('Plan outside frozen collection')
    if plan['game'] != f'holdout-{seed}-{opponent}-{side}-{policy}':
        raise ValueError('Unexpected game key')
    _, digest = task_identity(None, plan, cfg)
    if job and (plan['task_checkpoint_sha256'] != digest
                or not _digest(plan['bot_sha256']) or not _digest(plan['rival_sha256'])):
        raise ValueError('Invalid job identities')


def _unique_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result: raise ValueError('Duplicate plan field')
        result[key] = value
    return result


def counted_reexecute(original, side, *args, **kwargs):
    """Count official actor FERTILIZE event statuses in the existing replay.

    This observes the existing event stream; it does not change rule execution.
    An executed status means the official atomic event changed resources or tile,
    not that a long-term task or economic benefit has been established.
    """
    previous = kwargs.get('label_sink')
    counts = dict(requested=0,executed_with_observed_change=0,blocked=0,no_effect=0,error=0,other=0)
    def sink(event):
        if previous is not None: previous(event)
        requested = event.get('requested')
        if (event.get('phase')!='unit' or event.get('player')!=side
                or not isinstance(requested,list) or not requested or requested[0]!='FERTILIZE'):
            return
        counts['requested'] += 1
        status = event.get('status')
        key = {'executed':'executed_with_observed_change','blocked':'blocked',
               'no_effect':'no_effect','error':'error'}.get(status,'other')
        counts[key] += 1
    kwargs['label_sink'] = sink
    result = original(*args,**kwargs)
    result['fertilizer_actor_request_status'] = counts
    return result


@contextmanager
def bindings(job_path=None, expected=None, cfg=None):
    with _BINDINGS:
        saved = core.verify, core.task_identity, core.read, core.reexecute
        try:
            core.verify, core.task_identity = experiment.verify, task_identity
            if job_path is not None:
                core.reexecute = lambda *a,**kw: counted_reexecute(saved[3], expected['side'], *a,**kw)
                def checked_read(path):
                    if Path(path).resolve() != job_path: return saved[2](path)
                    with Path(path).open(encoding='utf-8-sig') as stream:
                        value = json.load(stream, object_pairs_hook=_unique_pairs)
                    validate_plan(value, cfg, job=True)
                    if value != expected: raise ValueError('Job plan changed before core read')
                    return value
                core.read = checked_read
            yield
        finally:
            core.verify, core.task_identity, core.read, core.reexecute = saved


def make_wrapper(out, plan, cfg):
    validate_plan(plan, cfg)
    with bindings(): return core.wrapper(out, plan, cfg)


def job(out, key):
    out = Path(out).resolve()
    cfg = experiment.ready(out)
    experiment.deadline_check()
    if not isinstance(key, str) or Path(key).name != key: raise ValueError('Invalid game key')
    path = (out/'job-inputs'/(key+'.json')).resolve()
    with path.open(encoding='utf-8-sig') as stream:
        plan = json.load(stream, object_pairs_hook=_unique_pairs)
    validate_plan(plan, cfg, job=True)
    if plan['game'] != key: raise ValueError('Job key differs from plan')
    with bindings(path, plan, cfg): return core.job(out, key)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--key', required=True)
    args = parser.parse_args()
    job(args.out, args.key)
