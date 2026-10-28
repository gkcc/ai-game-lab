"""One unweighted merged teacher fit under a Linux session guard."""
import argparse
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path, PurePosixPath
import sys
import time

from .experiment import OUT, ROOT, read, sha, ready, deadline_check, prepare_private_checkpoint, write_new

TOKEN_ENV = 'TASK_TEACHER_MERGE_GUARD_TOKEN'
ARCHIVE_READ_SCOPE = 'Admission ready/verify may hash archives repeatedly; worker decodes one bounded immutable snapshot per archive; not one physical file read'


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


def project_path(relative):
    if not isinstance(relative,str) or not relative or '\\' in relative or ':' in relative:
        raise ValueError('Invalid project path')
    p=PurePosixPath(relative)
    if p.is_absolute() or p.as_posix()!=relative or any(x in ('.','..') for x in p.parts):
        raise ValueError('Invalid project path')
    target=(ROOT/relative).resolve()
    if not target.is_relative_to(ROOT.resolve()) or target==ROOT.resolve(): raise ValueError('Path outside project')
    return target


def _worker_impl(out, progress):
    os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
    for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'): os.environ[key] = '1'
    from learning.task_teacher_choice_bc_v1 import numeric
    from learning.task_teacher_choice_bc_v1 import training
    from learning.task_teacher_merge_v1.merge import merge, to_groups
    from learning.task_choice_v1.features import OPS
    cfg = ready(out)
    progress(dict(phase='checking_input'))
    check_software_versions(cfg['wsl_software_versions'])
    identity_paths = dict(freeze=out/'freeze.json', registration=out/'registration.json',
        pre_execution_commit=out/'pre-execution-commit.json', marker=out/'fit-started.json',
        guard_marker=out/'gpu-guard-started.json', launch=out/'gpu-launch.local.json',
        schema=project_path(cfg['schema_path']['old']),new_schema=project_path(cfg['schema_path']['new']),
        seed_inventory=out/'seed-inventory.local.json')
    identities = {name: sha(path) for name, path in identity_paths.items()}
    def verify_identities():
        if any(sha(path) != identities[name] for name, path in identity_paths.items()):
            raise ValueError('Frozen fit identity changed')
    marker = read(identity_paths['marker'])
    guard_marker = read(identity_paths['guard_marker'])
    freeze_sha = identities['freeze']
    launch_path = out/'gpu-launch.local.json'
    if (marker['attempt'] != 1 or type(marker['attempt']) is not int or marker['retry_allowed'] is not False
        or marker['freeze_sha256'] != freeze_sha or marker['archive_sha256'] != cfg['archive_sha256']
        or identities['launch'] != cfg['gpu_launch_config_sha256']
        or guard_marker != dict(attempt=1, freeze_sha256=freeze_sha, fit_marker_sha256=identities['marker'],
            host_token_sha256=marker['host_token_sha256'], retry_allowed=False)
        or token_sha(os.environ.get(TOKEN_ENV)) != marker['host_token_sha256']
        or os.name != 'posix' or os.getsid(0) != os.getpid()):
        raise ValueError('Frozen fit identity changed')
    launch = read(launch_path)
    schema_path = project_path(cfg['schema_path']['old'])
    if identities['schema'] != cfg['schema_sha256'] or identities['new_schema'] != cfg['schema_sha256']:
        raise ValueError('Schema identity changed')
    schema = read(schema_path)
    if identities['seed_inventory']!=cfg['seed_inventory_sha256']: raise ValueError('Seed inventory changed')
    inventory=read(identity_paths['seed_inventory']); holdout=inventory['holdout_seeds']
    if (not isinstance(holdout,list) or type(cfg['holdout_seed_count']) is not int
        or len(holdout)!=cfg['holdout_seed_count'] or any(type(s) is not int or not 0<=s<2**32 for s in holdout)
        or len(set(holdout))!=len(holdout)):
        raise ValueError('Seed inventory invalid')
    verify_identities()
    limits = cfg['data_limits']
    progress(dict(phase='decoding_training_data'))
    arrays=[]
    for name in ('old','new'):
        item=numeric.read(project_path(cfg['archive_path'][name]), limits['max_groups'], limits['max_candidates'],
                          limits['max_archive_bytes'], limits['max_uncompressed_bytes'], cfg['archive_sha256'][name])
        if len(item['shared'])!=cfg['expected_groups'][name] or len(item['candidates'])!=cfg['expected_candidates'][name]:
            raise ValueError('Archive count changed')
        counts={op:0 for op in OPS}
        for i,label in enumerate(item['teacher_choice_label']):
            counts[OPS[int(item['ops'][int(item['offsets'][i])+int(label)])]]+=1
        if counts!=cfg['expected_operations'][name]: raise ValueError('Archive operation counts changed')
        arrays.append(item)
    seeds=cfg['old_train_seeds']+cfg['new_train_seeds']
    merged=merge(*arrays,old_seeds=cfg['old_train_seeds'],new_seeds=cfg['new_train_seeds'],
                 holdout_seeds=holdout,max_groups=limits['max_groups'],max_candidates=limits['max_candidates'])
    groups=to_groups(merged,train_seeds=seeds,holdout_seeds=holdout,
                     max_groups=limits['max_groups'],max_candidates=limits['max_candidates'])
    if any(not isinstance(launch.get(k),str) or not launch[k].strip()
           for k in ('expected_device','expected_device_kind')): raise ValueError('Invalid frozen GPU identity')
    verify_identities()
    checkpoint = prepare_private_checkpoint(out)
    progress(dict(phase='starting_fit'))
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
        archive_sha256=cfg['archive_sha256'], schema_sha256=cfg['schema_sha256'], identities=identities,
        archive_snapshot_reads=dict(old=1,new=1), groups=len(groups), updates=128, nonzero_gradient_updates=summary['nonzero_gradient_updates'],
        archive_read_scope=ARCHIVE_READ_SCOPE,
        sampling='Old then biased refresh groups; random permutation followed by candidate-width padding budget and up to 256 groups, so inclusion probability is not simply uniform; no added operation weights',
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
            or fit_marker['archive_sha256'] != cfg['archive_sha256'] or fit_marker['host_token_sha256'] != token_digest
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
            supervision = supervise([sys.executable, '-B', '-m', 'learning.task_teacher_merge_bc_v1.gpu_entry',
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
