"""One bounded WSL GPU training worker under the existing session supervisor."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

from .experiment import OUT, deadline_check, read, ready, sha, write_new


def worker(out):
    os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
    for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ[name] = '1'
    from . import numeric, training
    out = Path(out).resolve()
    cfg = ready(out)
    marker, source = read(out/'fit-started.json'), read(out/'train-data-result.json')
    if not source['passed'] or marker['train_data_sha256'] != sha(out/'train-data-result.json'):
        raise ValueError('Training input marker mismatch')
    path = out/'private/train.npz'
    if sha(path) != source['array_sha256']:
        raise ValueError('Training arrays changed')
    limits = cfg['data_limits']
    arrays = numeric.read(path, limits['max_groups_per_split'], limits['max_candidates_per_split'])
    groups = numeric.unpack(arrays, limits['max_groups_per_split'], limits['max_candidates_per_split'])
    if set(g['seed'] for g in groups) != set(cfg['train_seeds']):
        raise ValueError('Training seed role differs')
    launch = read(out/'gpu-launch.local.json')
    trained = training.fit(groups, read(out/'schema.json'), out/'private/model-final.npz',
        cfg['training']['training_seed'], launch['expected_device'], launch['expected_device_kind'])
    if set(trained['train_seeds']) != set(cfg['train_seeds']):
        raise ValueError('Training unexpectedly used another seed')
    deadline_check()
    result = dict(passed=True, checkpoint_sha256=trained['sha256'],
        baseline=trained['baseline'], train_seeds=sorted(trained['train_seeds']),
        training_summary=trained['training_summary'],
        train_data_sha256=sha(out/'train-data-result.json'), updates=128,
        only_completed_observed_used_for_gradients=True, final_update_only=True,
        no_test_data_read=True, no_official_games_during_fit=True,
        gpu_device_kind_checked_against_private_frozen_identity=True,
        gpu_speedup_measured=False, official_environment_on_gpu=False)
    write_new(out/'fit-result.json', result)


def guard(out):
    if os.name != 'posix':
        raise RuntimeError('Registered GPU process requires Linux')
    from learning.task_dagger_once_v1.gpu_guard import supervise
    out = Path(out).resolve()
    cfg = ready(out)
    if (out/'gpu-guard-started.json').exists() or (out/'gpu-cancel.local.json').exists():
        raise ValueError('GPU attempt already consumed or cancelled')
    started = time.monotonic()
    write_new(out/'gpu-guard-started.json', dict(attempt=1, freeze_sha256=sha(out/'freeze.json'),
        work_seconds=1180, process_seconds=cfg['training']['process_seconds'], retry_allowed=False))
    failure = None
    supervision = None
    try:
        supervision = supervise([sys.executable, '-B', '-m',
            'learning.task_completion_bc_v1.gpu_entry', 'worker', '--out', str(out)],
            1180, lambda:(out/'gpu-cancel.local.json').exists())
        if not supervision['completed'] or not read(out/'fit-result.json')['passed']:
            raise RuntimeError('Registered GPU worker failed')
        deadline_check()
    except BaseException as exc:
        failure = exc
    elapsed = time.monotonic()-started
    result = dict(passed=failure is None and supervision is not None
                       and supervision['completed'] and elapsed <= cfg['training']['process_seconds'],
                  error_kind=type(failure).__name__ if failure else None,
                  supervision=supervision, elapsed_seconds=elapsed,
                  within_total_deadline=elapsed <= cfg['training']['process_seconds'], retry_allowed=False)
    write_new(out/'gpu-guard-result.json', result)
    if not result['passed']:
        raise RuntimeError('GPU session did not complete and clean up; no retry')
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=('guard','worker'))
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    try:
        (guard if args.command == 'guard' else worker)(args.out)
    except BaseException as exc:
        print(json.dumps(dict(passed=False, error_kind=type(exc).__name__, retry_allowed=False)), flush=True)
        raise SystemExit(1) from None


if __name__ == '__main__':
    main()
