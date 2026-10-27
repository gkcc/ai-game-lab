"""Thin POSIX analysis entry using the two accepted process boundaries.

The fixture exercises the same entry and fixed marker only. Neither path
imports a model, a numerical package or the official game engine.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import time


MODULE = 'learning.task_action_observation_v1.analysis_entry'
FIXTURE_MANIFEST = {'schema': 'task-action-observation-entry-fixture-v1',
                    'marker': 'fixed-model-free-analysis-entry-17'}
TARGET_OUTPUTS = ['paired-rows.json', 'game-descriptors.json', 'raw-read-receipts.json',
                  'analysis-result.json', 'analysis-provenance.json', 'analysis-target-result.json',
                  'analysis-failure.local.json', 'analysis-target-failure.local.json', 'fixture-result.json']
CLEANUP_FIELDS = ('owned_process_group_signalled', 'direct_child_reaped',
                  'no_live_owned_session_members_before_reap')


def worker_audit(event, args):
    if event in {'subprocess.Popen', 'os.system', 'os.fork', 'os.forkpty',
                 'os.posix_spawn', 'os.spawn', 'os.exec'}:
        raise RuntimeError('Analysis target may not create or replace processes')
    if event != 'import':
        return
    name = args[0]
    forbidden = ('torch', 'numpy', 'jax', 'tensorflow', 'kaggle_environments', 'bots', 'models')
    if any(name == prefix or name.startswith(prefix + '.') for prefix in forbidden):
        raise RuntimeError('Numerical, model, agent and engine imports are prohibited')
    allowed = ('learning.task_action_observation_v1',
               'learning.task_action_impact_diagnostic_v1.analysis',
               'learning.task_bc_error_diagnostics_v1.shadow',
               'learning.task_anchor_once_v1.storage')
    parents = ('learning.task_action_impact_diagnostic_v1',
               'learning.task_bc_error_diagnostics_v1', 'learning.task_anchor_once_v1')
    if name.startswith('learning.') and name not in parents and not any(
            name == prefix or name.startswith(prefix + '.') for prefix in allowed):
        raise RuntimeError('Unregistered learning import prohibited')


def limits(cfg, fixture=False):
    value = dict(cfg['analysis_limits'])
    fixed = dict(work_seconds=420, process_seconds=450, target_seconds=360,
                 cpu_seconds=300, address_space_bytes=2**31, output_bytes=128 * 2**20,
                 parent_receipt_bytes=65536, compressed_bytes=2**30, decoded_bytes=12 * 2**30)
    if any(type(value.get(k)) is not int or value[k] != expected for k, expected in fixed.items()):
        raise ValueError('Registered analysis limits differ')
    if fixture:
        value.update(target_seconds=30, work_seconds=45, process_seconds=60)
    return value


def location(out, fixture=False):
    from .experiment import ROOT, OUT
    path = Path(out)
    path = (path if path.is_absolute() else ROOT / path).resolve()
    if path != OUT.resolve():
        raise ValueError('Only the registered observation output is allowed')
    return path, path / 'entry-fixture' if fixture else path


def names(fixture=False):
    return dict(started='fixture-started.json' if fixture else 'analysis-started.json',
                guard='fixture-guard-result.json' if fixture else 'guard-result.json',
                result='fixture-result.json' if fixture else 'analysis-result.json')


def resource_limits(value):
    import resource
    for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                 'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
        os.environ[name] = '1'
    applied = {}
    for key, limit in ((resource.RLIMIT_CPU, value['cpu_seconds']),
                       (resource.RLIMIT_AS, value['address_space_bytes'])):
        _, hard = resource.getrlimit(key)
        bound = limit if hard == resource.RLIM_INFINITY else min(limit, hard)
        resource.setrlimit(key, (bound, bound))
        applied['cpu_seconds' if key == resource.RLIMIT_CPU else 'address_space_bytes'] = bound
    affinity = None
    if hasattr(os, 'sched_setaffinity'):
        allowed = os.sched_getaffinity(0)
        if not allowed:
            raise RuntimeError('No available analysis CPU')
        os.sched_setaffinity(0, {min(allowed)})
        affinity = len(os.sched_getaffinity(0))
        if affinity != 1:
            raise RuntimeError('Single CPU affinity was not established')
    applied.update(thread_environment=1, cpu_affinity_count=affinity)
    return applied


def import_provenance(out):
    from .experiment import ROOT, read, sha
    frozen = read(out / 'freeze.json')['runtime_files']
    registered, actual = ROOT.resolve(), []
    for name, module in sorted(sys.modules.items()):
        filename = getattr(module, '__file__', None)
        if filename is None:
            continue
        path = Path(filename).resolve()
        if not path.is_relative_to(registered):
            continue
        relative = path.relative_to(registered).as_posix()
        value = sha(path)
        if frozen.get(relative) != value:
            raise ValueError('Actual project import is absent from the frozen source identities')
        actual.append(dict(module=name, path=relative, sha256=value))
    if not actual:
        raise ValueError('No project import provenance was captured')
    return dict(schema='task-action-observation-import-provenance-v1', modules=actual,
                python_version=sys.version.split()[0], frozen_sources_verified=True,
                numerical_model_engine_imports=0)


def target(out_text, fixture=False):
    """Spawn target: install restrictions before importing the pure analyzer."""
    from .experiment import read, write_new
    main_out, out = location(out_text, fixture)
    cfg = read(main_out / 'registration.json')
    value = limits(cfg, fixture)
    sys.addaudithook(worker_audit)
    applied = resource_limits(value)
    from learning.task_anchor_once_v1.storage import owned_writes
    from . import analysis
    budget = owned_writes(out, value['output_bytes'], allowed_paths=TARGET_OUTPUTS)
    with budget:
        try:
            write_new(out / 'analysis-provenance.json', import_provenance(main_out))
            if fixture:
                if read(out / 'input-manifest.json') != FIXTURE_MANIFEST:
                    raise ValueError('Fixed synthetic entry marker differs')
                write_new(out / 'fixture-result.json', dict(passed=True, complete=True,
                    marker_verified=True, fixture_role='positive_integration_only',
                    scientific_raw_files_read=0, model_array_decodes=0, neural_forwards=0,
                    official_transitions=0, new_games=0, training_updates=0, gpu_seconds=0))
            else:
                analysis.run(out)
            write_new(out / 'analysis-target-result.json', dict(passed=True, fixture=fixture,
                resource_limits=applied, output_bytes_before_final_receipt=budget.used_bytes,
                output_limit_bytes=value['output_bytes'], retry_allowed=False))
        except BaseException as exc:
            write_new(out / 'analysis-target-failure.local.json', dict(
                kind=type(exc).__name__, message=str(exc)[:1000], retry_allowed=False))
            # Multiprocessing would otherwise print a potentially private traceback.
            raise SystemExit(1) from None


def worker(out_text, fixture=False):
    from .experiment import read, write_new, sha, verify, deadline_check
    from learning.task_anchor_spawn_repair_v1.launcher import launch
    from learning.task_anchor_once_v1.storage import owned_writes
    main_out, out = location(out_text, fixture)
    cfg = verify(main_out)
    deadline_check()
    value, output_names = limits(cfg, fixture), names(fixture)
    begun = read(out / output_names['started'])
    if begun.get('freeze_sha256') != sha(main_out / 'freeze.json') or begun.get('fixture') is not fixture:
        raise ValueError('Worker lacks its matching exclusive start receipt')
    # No owned_writes context wraps launch, process construction or spawn IPC.
    launched = launch(out / 'analysis-parent', target, (str(main_out), fixture),
                      seconds=value['target_seconds'], receipt_limit=value['parent_receipt_bytes'])
    result = read(out / output_names['result'])
    reached = read(out / 'analysis-target-result.json')
    if not (launched['passed'] and launched['target_handle_closed'] and reached['passed']
            and result['complete'] and not launched['parent_receipt_refused']):
        raise RuntimeError('Analysis target did not complete its registered result')
    deadline_check()
    with owned_writes(out, 65536, allowed_paths=['analysis-worker-result.json']):
        write_new(out / 'analysis-worker-result.json', dict(passed=True, fixture=fixture,
            primary=dict(result_complete=True, child_exit_code=launched['child_exit_code']),
            cleanup=dict(target_handle_closed=launched['target_handle_closed']),
            parent_receipt_bytes=launched['parent_receipt_bytes_total'], retry_allowed=False))


def guard(out_text, fixture=False):
    from .experiment import read, write_new, sha, ready, deadline_check
    from learning.task_dagger_once_v1.gpu_guard import supervise
    from learning.task_anchor_once_v1.storage import owned_writes
    started = time.monotonic()
    if not sys.platform.startswith('linux'):
        raise RuntimeError('This analysis entry requires the registered POSIX route')
    main_out, out = location(out_text, fixture)
    cfg = ready(main_out)
    deadline_check()
    value, output_names = limits(cfg, fixture), names(fixture)
    if (main_out / 'cancel.local.json').exists():
        raise RuntimeError('Registered task cancellation is already present')
    with owned_writes(out, 65536, allowed_paths=[output_names['started']]) as first:
        write_new(out / output_names['started'], dict(attempt=1, fixture=fixture,
            freeze_sha256=sha(main_out / 'freeze.json'), work_seconds=value['work_seconds'],
            process_seconds=value['process_seconds'], target_seconds=value['target_seconds'],
            attempt_consumed=True, retry_allowed=False))
    supervised, failure, primary = None, None, dict(status='not_returned')
    try:
        command = [sys.executable, '-B', '-m', MODULE, 'worker', '--out', str(main_out)]
        if fixture:
            command.append('--fixture')
        supervised = supervise(command, value['work_seconds'], lambda: (main_out / 'cancel.local.json').exists())
        primary = dict(status=supervised['reason'], child_exit_code=supervised['child_exit_code'],
                       child_completed=supervised['completed'])
        if not supervised['completed']:
            raise RuntimeError('Analysis session did not complete')
        if not read(out / 'analysis-worker-result.json')['passed']:
            raise RuntimeError('Analysis worker completion receipt is missing')
        deadline_check()
    except BaseException as exc:
        failure = exc
        primary['error_kind'] = type(exc).__name__
    finally:
        elapsed = time.monotonic() - started
        cleanup = {name: None if supervised is None else supervised.get(name) for name in CLEANUP_FIELDS}
        cleanup_proven = all(flag is True for flag in cleanup.values())
        completed = failure is None and cleanup_proven and elapsed <= value['process_seconds']
        result = dict(schema='task-action-observation-guard-result-v1', completed=completed,
            fixture=fixture, primary=primary, cleanup=dict(**cleanup, proven=cleanup_proven), **cleanup,
            elapsed_seconds=elapsed, within_total_deadline=elapsed <= value['process_seconds'],
            child_supervision_seconds=None if supervised is None else supervised['elapsed_seconds'],
            cleanup_state='verified' if cleanup_proven else 'unknown', retry_allowed=False)
        with owned_writes(out, 65536 - first.used_bytes, allowed_paths=[output_names['guard']]):
            write_new(out / output_names['guard'], result)
    if failure is not None:
        raise failure
    if not result['completed']:
        raise RuntimeError('Analysis cleanup or registered total deadline was not proven')
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['guard', 'worker'])
    parser.add_argument('--out', required=True)
    parser.add_argument('--fixture', action='store_true')
    args = parser.parse_args()
    try:
        (guard if args.command == 'guard' else worker)(args.out, args.fixture)
    except BaseException as exc:
        print(json.dumps(dict(passed=False, error_kind=type(exc).__name__, retry_allowed=False)), flush=True)
        raise SystemExit(1) from None


if __name__ == '__main__':
    main()
