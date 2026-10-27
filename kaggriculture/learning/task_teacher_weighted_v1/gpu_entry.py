"""One weighted fit under a Linux session guard; Windows transport is separate."""
import argparse
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import sys
import time

from .experiment import OUT, ROOT, read, sha, ready, deadline_check, write_new

TOKEN_ENV = 'TASK_TEACHER_WEIGHTED_GUARD_TOKEN'


def token_sha(token):
    if not isinstance(token, str) or not token: raise ValueError('Guard token required')
    return hashlib.sha256(token.encode('utf-8')).hexdigest()


class ReceiptFailure(RuntimeError):
    def __init__(self, primary_error_kind, receipt_error_kind):
        super().__init__('Failure receipt could not be written')
        self.primary_error_kind, self.receipt_error_kind = primary_error_kind, receipt_error_kind


def check_software_versions(expected):
    actual = dict(python='.'.join(str(v) for v in sys.version_info[:3]),
                  **{name: importlib.metadata.version(name) for name in ('numpy', 'jax', 'optax')})
    if expected != actual: raise ValueError('WSL software versions changed')


def _worker_impl(out, progress):
    os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
    for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'): os.environ[key] = '1'
    from learning.task_teacher_choice_bc_v1 import numeric
    from learning.task_choice_v1.features import OPS
    from . import training
    from .weights import operation_weights
    cfg = ready(out)
    progress(dict(phase='checking_input'))
    check_software_versions(cfg['wsl_software_versions'])
    identity_paths = dict(freeze=out/'freeze.json', registration=out/'registration.json',
        pre_execution_commit=out/'pre-execution-commit.json', marker=out/'fit-started.json',
        guard_marker=out/'gpu-guard-started.json', launch=out/'gpu-launch.local.json', schema=ROOT/cfg['schema_path'])
    identities = {name: sha(path) for name, path in identity_paths.items()}
    def verify_identities():
        if any(sha(path) != identities[name] for name, path in identity_paths.items()):
            raise ValueError('Frozen fit identity changed')
    marker = read(identity_paths['marker'])
    guard_marker = read(identity_paths['guard_marker'])
    freeze_sha = identities['freeze']
    launch_path = out/'gpu-launch.local.json'
    if (marker['attempt'] != 1 or type(marker['attempt']) is not int or marker['retry_allowed'] is not False
        or marker['freeze_sha256'] != freeze_sha or marker['cache_sha256'] != cfg['cache_sha256']
        or identities['launch'] != cfg['gpu_launch_config_sha256']
        or guard_marker != dict(attempt=1, freeze_sha256=freeze_sha, fit_marker_sha256=identities['marker'],
            host_token_sha256=marker['host_token_sha256'], retry_allowed=False)
        or token_sha(os.environ.get(TOKEN_ENV)) != marker['host_token_sha256']
        or os.name != 'posix' or os.getsid(0) != os.getpid()):
        raise ValueError('Frozen fit identity changed')
    launch = read(launch_path)
    schema_path = ROOT/cfg['schema_path']
    if identities['schema'] != cfg['schema_sha256']: raise ValueError('Schema identity changed')
    schema = read(schema_path)
    verify_identities()
    limits = cfg['data_limits']
    progress(dict(phase='decoding_training_data'))
    arrays = numeric.read(ROOT/cfg['cache_path'], limits['max_groups'], limits['max_candidates'],
                          limits['max_archive_bytes'], limits['max_uncompressed_bytes'], cfg['cache_sha256'])
    groups = []
    counts = {op: 0 for op in OPS}
    for i in range(len(arrays['shared'])):
        lo, hi = int(arrays['offsets'][i]), int(arrays['offsets'][i+1])
        label = int(arrays['teacher_choice_label'][i])
        ops = [OPS[int(x)] for x in arrays['ops'][lo:hi]]
        counts[ops[label]] += 1
        groups.append(training.Group(arrays['shared'][i], arrays['candidates'][lo:hi], label, ops, int(arrays['seed'][i])))
    seeds = cfg['old_train_seeds']
    if (not isinstance(seeds, list) or not seeds or any(type(s) is not int or not 0 <= s <= 0xffffffff for s in seeds)
        or len(set(seeds)) != len(seeds) or set(g.seed for g in groups) != set(seeds)):
        raise ValueError('Training seed role differs')
    if counts != cfg['expected_operations']: raise ValueError('Operation counts changed')
    weights = operation_weights(groups)
    expected = cfg['expected_class_weights']
    if (not isinstance(expected, dict) or set(expected) != set(OPS)
        or any(type(expected[o]) not in (float, int) or not math.isfinite(expected[o])
               or abs(weights[o]-expected[o]) > 1e-12 for o in OPS)):
        raise ValueError('Class weights changed')
    verify_identities()
    progress(dict(phase='starting_fit'))
    checkpoint = out/'private/model-final.npz'
    trained = training.fit(groups, schema, checkpoint, cfg['training']['training_seed'],
                           launch['expected_device'], launch['expected_device_kind'], progress)
    summary = trained['training_summary']
    parity = summary['inference_parity']
    if (set(trained['train_seeds']) != set(seeds) or summary['synchronized_updates'] != 128
        or summary['gpu_verified'] is not True or summary['weights_changed'] is not True
        or summary['nonzero_gradient_updates'] <= 0 or parity['groups'] != 16 or parity['synchronized'] is not True
        or parity['requested_matmul_precision'] != 'highest' or parity['hardware_precision_measured'] is not False
        or sha(checkpoint) != trained['sha256']):
        raise ValueError('Training acceptance failed')
    verify_identities()
    deadline_check()
    write_new(out/'fit-result.json', dict(passed=True, checkpoint_sha256=trained['sha256'], freeze_sha256=freeze_sha,
        cache_sha256=cfg['cache_sha256'], schema_sha256=cfg['schema_sha256'], identities=identities,
        archive_snapshot_reads=1, operations=counts,
        class_weights=weights, updates=128, nonzero_gradient_updates=summary['nonzero_gradient_updates'],
        weights_changed=True, gpu_verified=True, inference_parity=parity, retry_allowed=False,
        gpu_memory_hard_cap_claimed=False, windows_transport_cleanup_verified_here=False))


def worker(out):
    out = Path(out).resolve()
    last = dict(phase='starting_worker', updates_started=0, updates_completed=0, parity_started=0,
                parity_completed=0, checkpoint_written=False, device_validated=False)
    try:
        with (out/'fit-progress.local.jsonl').open('x', encoding='utf-8', newline='\n') as stream:
            def progress(event):
                last.update(event)
                stream.write(json.dumps(last, sort_keys=True, allow_nan=False)+'\n'); stream.flush()
            progress({})
            _worker_impl(out, progress)
    except BaseException as exc:
        if (out/'fit-started.json').exists():
            try:
                write_new(out/'fit-worker-failure.json', dict(passed=False, error_kind=type(exc).__name__,
                    last_progress=last, checkpoint_exists=(out/'private/model-final.npz').exists(), retry_allowed=False))
            except BaseException as receipt:
                raise ReceiptFailure(type(exc).__name__, type(receipt).__name__) from None
        raise


def guard(out, host_token):
    if os.name != 'posix': raise RuntimeError('Linux guard required')
    from learning.task_dagger_once_v1.gpu_guard import supervise
    out = Path(out).resolve()
    cfg = ready(out)
    if any((out/name).exists() for name in ('gpu-guard-started.json', 'gpu-guard-result.json', 'gpu-cancel.local.json')):
        raise ValueError('GPU attempt already consumed or cancelled')
    start = time.monotonic(); marker = dict(created=False, complete=False); error = None; supervision = None
    freeze_sha = None; guard_marker_sha = None
    try:
        path = (out/'gpu-guard-started.json').resolve()
        if not path.is_relative_to(OUT.resolve()): raise ValueError('Marker outside experiment')
        freeze_sha = sha(out/'freeze.json')
        fit_path = out/'fit-started.json'
        fit_marker_sha = sha(fit_path)
        fit_marker = read(fit_path)
        token_digest = token_sha(host_token)
        if (type(fit_marker['attempt']) is not int or fit_marker['attempt'] != 1
            or fit_marker['retry_allowed'] is not False or fit_marker['freeze_sha256'] != freeze_sha
            or fit_marker['cache_sha256'] != cfg['cache_sha256'] or fit_marker['host_token_sha256'] != token_digest
            or sha(fit_path) != fit_marker_sha):
            raise ValueError('Host fit identity changed')
        with path.open('x', encoding='utf-8', newline='\n') as stream:
            marker['created'] = True
            json.dump(dict(attempt=1, freeze_sha256=freeze_sha, fit_marker_sha256=fit_marker_sha,
                           host_token_sha256=token_digest, retry_allowed=False), stream)
            stream.write('\n')
        marker['complete'] = True
        guard_marker_sha = sha(path)
        if sha(out/'freeze.json') != freeze_sha: raise ValueError('GPU guard identity changed')
        old_token = os.environ.get(TOKEN_ENV)
        try:
            os.environ[TOKEN_ENV] = host_token
            supervision = supervise([sys.executable, '-B', '-m', 'learning.task_teacher_weighted_v1.gpu_entry',
                'worker', '--out', str(out)], cfg['training']['work_seconds'], lambda: (out/'gpu-cancel.local.json').exists())
        finally:
            if old_token is None: os.environ.pop(TOKEN_ENV, None)
            else: os.environ[TOKEN_ENV] = old_token
        flags = ('owned_process_group_signalled', 'direct_child_reaped', 'no_live_owned_session_members_before_reap')
        if (not supervision['completed'] or not all(supervision.get(k) is True for k in flags)
            or not read(out/'fit-result.json')['passed']): raise RuntimeError('GPU worker or Linux cleanup failed')
        if sha(path) != guard_marker_sha or sha(out/'freeze.json') != freeze_sha or sha(fit_path) != fit_marker_sha:
            raise ValueError('GPU guard identity changed')
        deadline_check()
    except BaseException as exc:
        error = type(exc).__name__
    elapsed = time.monotonic()-start
    result = dict(passed=error is None and supervision is not None and elapsed <= cfg['training']['process_seconds'],
        marker=marker, error_kind=error, supervision=supervision, elapsed_seconds=elapsed, retry_allowed=False,
        freeze_sha256=freeze_sha, guard_marker_sha256=guard_marker_sha,
        windows_transport_cleanup_verified_here=False, gpu_memory_hard_cap_claimed=False)
    if marker['created']:
        try: write_new(out/'gpu-guard-result.json', result)
        except BaseException as exc: raise ReceiptFailure(error, type(exc).__name__) from None
    if not result['passed']: raise RuntimeError('GPU session failed; no retry')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=('guard', 'worker')); parser.add_argument('--out', required=True)
    parser.add_argument('--host-token')
    args = parser.parse_args()
    try:
        if args.command == 'guard': guard(args.out, args.host_token)
        else: worker(args.out)
    except BaseException as exc:
        print(json.dumps(dict(passed=False, error_kind=getattr(exc, 'primary_error_kind', None) or type(exc).__name__, retry_allowed=False)), flush=True)
        raise SystemExit(1) from None
