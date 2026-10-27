"""One conditional-teacher GPU worker inside the existing Linux session guard.

Windows transport supervision is a separate caller responsibility; neither
Windows Job membership nor a hard GPU memory cap is asserted here.
"""
import argparse
import json
import importlib.metadata
import os
from pathlib import Path
import sys
import time
from learning.task_choice_v1.features import OPS
from .experiment import OUT, deadline_check, read, ready, sha, write_new


class ReceiptFailure(RuntimeError):
    def __init__(self, primary_error_kind, receipt_error_kind):
        super().__init__('Failure receipt could not be written')
        self.primary_error_kind,self.receipt_error_kind = primary_error_kind,receipt_error_kind


def check_software_versions(expected):
    actual = dict(python='.'.join(str(v) for v in sys.version_info[:3]),
                  numpy=importlib.metadata.version('numpy'),jax=importlib.metadata.version('jax'),
                  optax=importlib.metadata.version('optax'))
    if expected != actual: raise ValueError('WSL software versions changed')


def _worker_impl(out, progress):
    os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE','false')
    for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'): os.environ[key] = '1'
    from . import numeric, training
    cfg = ready(out)
    progress(dict(phase='checking_input'))
    check_software_versions(cfg['wsl_software_versions'])
    marker,source = read(out/'fit-started.json'),read(out/'train-data-result.json')
    cache_sha = source['array_sha256']
    if (not source['passed'] or marker['train_data_sha256'] != sha(out/'train-data-result.json')
        or marker['freeze_sha256'] != sha(out/'freeze.json') or source['freeze_sha256'] != marker['freeze_sha256']
        or marker['array_sha256'] != cache_sha or source['collection_attempt'] != 1
        or source['source_manifest_sha256'] != cfg['source_manifest_sha256']
        or source['sampling_algorithm'] != cfg['sampling_algorithm']):
        raise ValueError('Training input identity changed')
    path = out/'private/train.npz'
    if sha(path) != cache_sha or sha(out/'schema.json') != cfg['schema_sha256']:
        raise ValueError('Training input changed')
    limits = cfg['data_limits']
    progress(dict(phase='decoding_training_data'))
    arrays = numeric.read(path,limits['max_groups'],limits['max_candidates'],limits['max_archive_bytes'],limits['max_uncompressed_bytes'],cache_sha)
    if sha(path) != cache_sha or sha(out/'schema.json') != cfg['schema_sha256'] or sha(out/'train-data-result.json') != marker['train_data_sha256']:
        raise ValueError('Training input changed during reading')
    groups = []
    for i in range(len(arrays['shared'])):
        lo,hi = int(arrays['offsets'][i]),int(arrays['offsets'][i+1])
        groups.append(dict(shared=arrays['shared'][i],candidates=arrays['candidates'][lo:hi],
            teacher_choice_label=int(arrays['teacher_choice_label'][i]),
            candidate_ops=[OPS[int(x)] for x in arrays['ops'][lo:hi]],seed=int(arrays['seed'][i])))
    if set(g['seed'] for g in groups) != set(cfg['train_seeds']): raise ValueError('Training seed role differs')
    launch = read(out/'gpu-launch.local.json')
    if sha(out/'gpu-launch.local.json') != cfg['gpu_launch_config_sha256']:
        raise ValueError('GPU launch identity changed')
    progress(dict(phase='starting_fit'))
    trained = training.fit(groups,read(out/'schema.json'),out/'private/model-final.npz',
        cfg['training']['training_seed'],launch['expected_device'],launch['expected_device_kind'],progress)
    if set(trained['train_seeds']) != set(cfg['train_seeds']): raise ValueError('Training seed role changed')
    summary = trained['training_summary']
    parity = summary['inference_parity']
    if (summary['synchronized_updates'] != 128 or not summary['gpu_verified'] or not summary['weights_changed']
        or summary['nonzero_gradient_updates'] <= 0 or parity['groups'] != 16 or not parity['synchronized']
        or parity['requested_matmul_precision'] != 'highest' or parity['hardware_precision_measured'] is not False
        or sha(out/'private/model-final.npz') != trained['sha256']):
        raise ValueError('Training acceptance failed')
    progress(dict(phase='validating_fit'))
    deadline_check()
    write_new(out/'fit-result.json',dict(passed=True,checkpoint_sha256=trained['sha256'],
        baseline=trained['baseline'],train_seeds=sorted(trained['train_seeds']),training_summary=summary,
        train_data_sha256=sha(out/'train-data-result.json'),updates=128,conditional_teacher_labels=True,
        completion_status_used=False,final_update_only=True,no_test_data_read=True,no_official_games_during_fit=True,
        gpu_identity_checked_against_private_frozen_identity=True,gpu_memory_hard_cap_claimed=False,
        cache_memory_scope='Immutable archive snapshot plus ZIP/NPY decoding and arrays add memory; not a hard RAM cap'))


def worker(out):
    out = Path(out).resolve()
    last = dict(phase='starting_worker',updates_started=0,updates_completed=0,parity_started=0,parity_completed=0,
                checkpoint_written=False,device_validated=False)
    try:
        with (out/'fit-progress.local.jsonl').open('x',encoding='utf-8',newline='\n') as stream:
            def progress(event):
                last.update(event)
                stream.write(json.dumps(last,sort_keys=True,allow_nan=False)+'\n'); stream.flush()
            progress({})
            _worker_impl(out,progress)
            progress(dict(phase='fit_result_written'))
    except BaseException as exc:
        if (out/'fit-started.json').exists():
            try:
                write_new(out/'fit-worker-failure.json',dict(passed=False,error_kind=type(exc).__name__,last_progress=last,
                    checkpoint_exists=(out/'private/model-final.npz').exists(),retry_allowed=False,
                    counter_semantics='Completed means synchronized, not subsequent validation passed'))
            except BaseException as receipt:
                raise ReceiptFailure(type(exc).__name__,type(receipt).__name__) from None
        raise


def guard(out):
    if os.name != 'posix': raise RuntimeError('Linux guard required')
    from learning.task_dagger_once_v1.gpu_guard import supervise
    out = Path(out).resolve()
    cfg = ready(out)
    if (out/'gpu-guard-started.json').exists() or (out/'gpu-cancel.local.json').exists():
        raise ValueError('GPU attempt already consumed or cancelled')
    start = time.monotonic()
    marker = dict(created=False,complete=False)
    error = None; supervision = None
    try:
        path = (out/'gpu-guard-started.json').resolve()
        if not path.is_relative_to(OUT.resolve()): raise ValueError('Marker outside experiment')
        with path.open('x',encoding='utf-8',newline='\n') as stream:
            marker['created'] = True
            json.dump(dict(attempt=1,freeze_sha256=sha(out/'freeze.json'),retry_allowed=False),stream,indent=2)
            stream.write('\n')
        marker['complete'] = True
        supervision = supervise([sys.executable,'-B','-m','learning.task_teacher_choice_bc_v1.gpu_entry',
            'worker','--out',str(out)],cfg['training']['work_seconds'],lambda:(out/'gpu-cancel.local.json').exists())
        flags = ('owned_process_group_signalled','direct_child_reaped','no_live_owned_session_members_before_reap')
        if (not supervision['completed'] or not all(supervision.get(k) is True for k in flags)
            or not read(out/'fit-result.json')['passed']): raise RuntimeError('GPU worker or Linux cleanup failed')
        deadline_check()
    except BaseException as exc:
        error = type(exc).__name__
    elapsed = time.monotonic()-start
    result = dict(passed=error is None and supervision is not None and elapsed <= cfg['training']['process_seconds'],
        marker=marker,error_kind=error,supervision=supervision,elapsed_seconds=elapsed,retry_allowed=False,
        windows_transport_cleanup_verified_here=False,gpu_memory_hard_cap_claimed=False)
    if marker['created']:
        try: write_new(out/'gpu-guard-result.json',result)
        except BaseException as exc: raise ReceiptFailure(error,type(exc).__name__) from None
    if not result['passed']: raise RuntimeError('GPU session failed; no retry')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command',choices=('guard','worker')); parser.add_argument('--out',required=True)
    args = parser.parse_args()
    try: (guard if args.command=='guard' else worker)(args.out)
    except BaseException as exc:
        print(json.dumps(dict(passed=False,error_kind=getattr(exc,'primary_error_kind',None) or type(exc).__name__,retry_allowed=False)),flush=True)
        raise SystemExit(1) from None
