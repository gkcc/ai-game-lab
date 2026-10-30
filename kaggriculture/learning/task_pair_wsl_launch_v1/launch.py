"""Fail-closed, registered one-shot WSL paired-training launcher candidate.

The training registration does not exist yet. Source and runtime contracts
must be authenticated before private data, a child or an output is accessed.
"""

import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import re
import secrets
import stat
import subprocess
import sys
import time
import types


GAME_SOURCE = Path(__file__).resolve().parents[2]
REPO = GAME_SOURCE.parent
REG_REL = 'review/evidence/task-pair-wsl-training-v1-20261030/registration.json'
ENTRY_REL = 'kaggriculture/learning/task_pair_wsl_training_v1/entry.py'
LAUNCH_REL = 'kaggriculture/learning/task_pair_wsl_launch_v1/launch.py'
GATE_REL = 'kaggriculture/learning/task_pair_pidfd_guard_v1/gate.py'
SUPERVISOR_NAME = 'learning.task_pair_supervision_v1.supervisor'
GUARD_NAME = 'learning.task_pair_pidfd_guard_v1.guard'
OUTPUT_NAME = 'task-pair-wsl-training-supervisor-v1-20261030'
MAX_SOURCE = 4 * 1024 * 1024
MAX_PRIVATE = 2 * 1024 * 1024
HEX_COMMIT = re.compile(r'[0-9a-f]{40}\Z')
HEX_SHA = re.compile(r'[0-9a-f]{64}\Z')
HEX_NONCE = re.compile(r'[0-9a-f]{32}\Z')
STAGE = 'arguments'


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _stage(name):
    global STAGE
    STAGE = name


def _bounded_file(path, maximum):
    before = path.lstat()
    _require(stat.S_ISREG(before.st_mode) and not stat.S_ISLNK(before.st_mode)
             and not getattr(before, 'st_file_attributes', 0) & 0x400
             and before.st_size <= maximum, 'Unexpected source file')
    flags = os.O_RDONLY | getattr(os, 'O_BINARY', 0) | getattr(os, 'O_NOFOLLOW', 0)
    descriptor = os.open(path, flags)
    try:
        info = os.fstat(descriptor)
        _require(stat.S_ISREG(info.st_mode) and info.st_size <= maximum, 'Unexpected source file')
        with os.fdopen(descriptor, 'rb', closefd=False) as stream:
            raw = stream.read(maximum + 1)
        _require(len(raw) == info.st_size and len(raw) <= maximum, 'Source length differs')
        return raw
    finally:
        os.close(descriptor)


def _git(*args):
    return subprocess.check_output(['git', '-C', str(REPO), *args], stderr=subprocess.DEVNULL, timeout=30)


def _head():
    raw = _git('rev-parse', '--verify', 'HEAD')
    _require(len(raw) == 41 and raw[-1:] == b'\n', 'Invalid Git HEAD')
    commit = raw[:-1].decode('ascii')
    _require(HEX_COMMIT.fullmatch(commit) is not None, 'Invalid Git HEAD')
    return commit


def _blob(commit, relative):
    _require(type(commit) is str and HEX_COMMIT.fullmatch(commit)
             and relative in (REG_REL, ENTRY_REL), 'Invalid prebinding input')
    spec = commit + ':' + relative
    size = _git('cat-file', '-s', spec)
    _require(len(size) <= 32 and size.strip().isdigit() and int(size.strip()) <= MAX_SOURCE,
             'Registered blob exceeds limit')
    raw = _git('show', spec)
    _require(len(raw) == int(size.strip()), 'Registered blob length differs')
    return raw


def _registration_object(raw):
    def unique(pairs):
        out = {}
        for key, value in pairs:
            _require(key not in out, 'Duplicate registration field')
            out[key] = value
        return out
    def reject(value):
        raise ValueError('Invalid registration constant')
    try:
        data = json.loads(raw.decode('utf-8-sig'), object_pairs_hook=unique, parse_constant=reject)
    except (UnicodeError, ValueError, TypeError, RecursionError):
        raise ValueError('Invalid registration') from None
    _require(type(data) is dict and type(data.get('source_sha256')) is dict,
             'Invalid registration sources')
    digest = data['source_sha256'].get(ENTRY_REL)
    _require(type(digest) is str and HEX_SHA.fullmatch(digest), 'Entry identity absent')
    return digest


def authenticated_entry(commit):
    """Load exactly one source snapshot after a new HEAD registration gate.

    The launcher itself is already executing: the host must separately
    authenticate this source and the intended interpreter before execution.
    """
    _require(type(commit) is str and HEX_COMMIT.fullmatch(commit) and _head() == commit,
             'Registration commit differs')
    registered = _blob(commit, REG_REL)
    _require(_bounded_file(REPO / REG_REL, MAX_SOURCE) == registered, 'Registration bytes differ')
    expected = _registration_object(registered)
    raw = _bounded_file(REPO / ENTRY_REL, MAX_SOURCE)
    _require(raw == _blob(commit, ENTRY_REL) and hashlib.sha256(raw).hexdigest() == expected,
             'Registered entry differs')
    entry = types.ModuleType('registered_training_entry_snapshot')
    entry.__file__ = str(REPO / ENTRY_REL)
    exec(compile(raw, '<registered-training-entry>', 'exec', dont_inherit=True), entry.__dict__)
    recorded, registration = entry.registration_gate(commit)
    _require(recorded == registered, 'Training registration changed')
    entry.verify_registered_sources(commit, registration['source_sha256'])
    _require(_head() == commit, 'Registration commit changed')
    return entry, registration


def verify_private_outputs(entry, modules, root, commit, *, check):
    """Independently re-open private checkpoints and recompute the dev screen.

    This validates the output files and the screen arithmetic, not the child's
    source data or claimed optimizer execution. No private values are printed.
    """
    _require(callable(check), 'Output check is missing')
    output = root / 'artifacts' / entry.OUTPUT_NAME
    _require(not output.is_symlink() and output.resolve(strict=True) == output and output.is_dir(),
             'Invalid child output directory')
    manifest = entry._json(_bounded_file(REPO / entry.CACHE_REL, MAX_SOURCE), MAX_SOURCE)
    schema = manifest.get('numeric_feature_schema_sha256')
    _require(type(schema) is str and HEX_SHA.fullmatch(schema), 'Output schema identity differs')
    selected = entry._json(_bounded_file(REPO / entry.EARLY_REL, MAX_SOURCE), MAX_SOURCE)
    baseline_sha = selected.get('checkpoint_sha256')
    _require(type(baseline_sha) is str and HEX_SHA.fullmatch(baseline_sha), 'Baseline identity differs')
    snapshots, digests = {}, {}
    for arm in ('baseline', 'control', 'auxiliary'):
        check()
        path = output / (arm + '.npz')
        raw = _bounded_file(path, modules.snapshot.MAX_ARCHIVE_BYTES)
        digest = hashlib.sha256(raw).hexdigest()
        if arm == 'baseline':
            _require(digest == baseline_sha, 'Baseline output identity differs')
        item = modules.snapshot.load_checkpoint(raw, expected_sha256=digest,
                                                 expected_schema_sha256=schema)
        loaded = modules.model.TaskNetwork(path, digest)
        entry._verify_reloaded(loaded, item, modules.np)
        if arm != 'baseline':
            provenance = {'schema': 'task-pair-training-provenance-v1', 'arm': arm,
                          'training_source': 'original_teacher_train',
                          'initialization': 'frozen_early_task_head', 'registration_commit': commit,
                          'completed_updates': 64, 'worker_coefficient': 0.0 if arm == 'control' else 0.25}
            _require(entry._same(loaded.metadata.get('pair_training'), provenance),
                     'Private model provenance differs')
        _require(_bounded_file(path, modules.snapshot.MAX_ARCHIVE_BYTES) == raw,
                 'Private output changed during reload')
        snapshots[arm], digests[arm] = item, digest
    np, names = modules.np, modules.snapshot.PARAMS
    baseline, control, auxiliary = (snapshots[name] for name in ('baseline', 'control', 'auxiliary'))
    _require(all(np.array_equal(model.mean, baseline.mean) and np.array_equal(model.scale, baseline.scale)
                 for model in (control, auxiliary)), 'Output normalization differs')
    different = lambda first, second: any(not np.array_equal(first.params[name], second.params[name])
                                           for name in names)
    _require(different(baseline, control) and different(baseline, auxiliary) and different(control, auxiliary),
             'Private outputs do not have the required distinct weights')
    check()
    evaluation = entry._json(_bounded_file(output / 'evaluation.local.json', MAX_PRIVATE), MAX_PRIVATE)
    _require(set(evaluation) == {'schema', 'metrics', 'development_screen',
                                 'new_games', 'independent_game_evaluation'}
             and evaluation['schema'] == 'task-pair-wsl-private-evaluation-v1'
             and evaluation['new_games'] is False and evaluation['independent_game_evaluation'] is False,
             'Private evaluation differs')
    assessed = entry.assess_development(evaluation['metrics'])
    _require(entry._same(evaluation['development_screen'], assessed), 'Development screen differs')
    child = entry._json(_bounded_file(output / 'child-result.local.json', MAX_PRIVATE), MAX_PRIVATE)
    _require(set(child) == {'schema', 'completed_updates', 'auxiliary_nonzero_updates',
                            'development_screen_status', 'supervisor_confirmation'}
             and child['schema'] == 'task-pair-wsl-child-result-v1'
             and type(child['completed_updates']) is int and child['completed_updates'] == 64
             and type(child['auxiliary_nonzero_updates']) is int
             and 32 <= child['auxiliary_nonzero_updates'] <= 64
             and child['development_screen_status'] == assessed['status']
             and child['supervisor_confirmation'] is False, 'Private child result differs')
    check()
    return {'model_files_reloaded': True, 'weights_distinct': True,
            'development_screen_status': assessed['status'], 'model_sha256': digests,
            'child_reported_updates': child['completed_updates'],
            'child_reported_auxiliary_nonzero': child['auxiliary_nonzero_updates']}


class Prepared:
    """Keep registered private configuration out of implicit representations."""
    def __init__(self, entry, registration, private, root, artifacts, private_path,
                 expected_sha, modules, supervisor, guard, started):
        self.entry, self.registration, self.private = entry, registration, private
        self.root, self.artifacts, self.private_path = root, artifacts, private_path
        self.expected_sha, self.modules = expected_sha, modules
        self.supervisor, self.guard, self.started = supervisor, guard, started

    def __repr__(self):
        return '<PrivateRegisteredTrainingRun>'


def prepare(commit, game_root, private_path, expected_sha, *, started, clock=time.monotonic):
    """Authenticate registered sources and import guards before private reads."""
    _stage('registration')
    entry, registration = authenticated_entry(commit)
    limits = registration['limits']
    _require(clock() - started < limits['outer_total_seconds'], 'Outer deadline expired')
    _stage('imports')
    binding = entry.prebind(registration['source_sha256'])
    _stage('private_contract')
    root, artifacts = entry._private_root(game_root)
    raw = entry._private_file(private_path, artifacts, entry.MAX_PRIVATE)
    private = entry.parse_private_contract(raw, expected_sha)
    private_path = Path(private_path).resolve(strict=True)
    _stage('runtime')
    os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
    entry._runtime_pre(private)
    _stage('imports')
    modules = entry.finish_bootstrap(binding, registration['source_sha256'])
    entry._runtime_after(private)
    supervisor, guard = sys.modules.get(SUPERVISOR_NAME), sys.modules.get(GUARD_NAME)
    _require(supervisor is not None and guard is not None
             and registration['source_sha256'].get(GATE_REL) == guard.GATE_SHA256,
             'Registered supervisor or gate differs')
    _require(clock() - started < limits['outer_total_seconds'], 'Outer deadline expired')
    return Prepared(entry, registration, private, root, artifacts, private_path, expected_sha,
                    modules, supervisor, guard, started)


_GUARD_RESULT_KEYS = frozenset((
    'passed', 'child_started', 'child_exit_observed', 'child_exit_zero',
    'work_deadline_triggered', 'owned_session_empty_before_reap', 'direct_child_reaped',
    'used_pidfd', 'receipt_sink_attempted', 'receipt_sink_accepted', 'cancelled',
    'owned_session_clean', 'guard_with_cleanup_within_budget', 'canonical_token_receipt',
    'child_reported_success', 'registered_update_count_met',
    'registered_auxiliary_threshold_met', 'both_arms_changed_and_differ'))
_REQUIRED_GUARD_TRUE = _GUARD_RESULT_KEYS - {'passed', 'work_deadline_triggered', 'cancelled'}


def validate_guard_result(value):
    _require(type(value) is dict and set(value) == _GUARD_RESULT_KEYS
             and all(type(item) is bool for item in value.values()), 'Invalid supervisor result')
    clean = (value['child_started'] and value['owned_session_empty_before_reap']
             and value['direct_child_reaped'] and value['used_pidfd'])
    _require(value['owned_session_clean'] is clean, 'Supervisor cleanup result differs')
    accepted = (all(value[name] for name in _REQUIRED_GUARD_TRUE)
                and not value['work_deadline_triggered'] and not value['cancelled'])
    _require(value['passed'] is accepted, 'Supervisor acceptance differs')
    return accepted


def run_prepared(prepared, commit, *, host_nonce, clock=time.monotonic):
    """Run once; retain sensitive token, raw receipt and outputs locally only."""
    _require(type(host_nonce) is str and HEX_NONCE.fullmatch(host_nonce), 'Invalid host attempt identity')
    entry, registration, private = prepared.entry, prepared.registration, prepared.private
    modules, supervisor = prepared.modules, prepared.supervisor
    limits, sources = registration['limits'], registration['source_sha256']
    output = prepared.artifacts / OUTPUT_NAME
    _stage('output')
    _require(output.resolve(strict=False) == output
             and not os.path.lexists(prepared.artifacts / entry.OUTPUT_NAME),
             'A previous output already exists')
    output.mkdir(mode=0o700, exist_ok=False)
    # Ownership is recorded separately so an existing directory is never
    # modified when exclusive mkdir fails.
    prepared.owned_output = output
    token = secrets.token_hex(16)
    start = {'schema': 'task-pair-wsl-supervisor-start-v1', 'attempt': 1,
             'registration_commit': commit, 'token': token, 'host_nonce': host_nonce,
             'private_contract_sha256': prepared.expected_sha}
    entry._write_json(output / 'started.local.json', start)
    _require(entry._json(_bounded_file(output / 'started.local.json', entry.MAX_PRIVATE), entry.MAX_PRIVATE)
             == start, 'Private start record differs')

    def cancelled():
        return (clock() - prepared.started >= limits['outer_total_seconds']
                or os.path.lexists(output / 'cancel.request'))

    def check():
        if clock() - prepared.started >= limits['outer_total_seconds']:
            raise TimeoutError('Sampled outer deadline exceeded')
        _require(not os.path.lexists(output / 'cancel.request'), 'Cancellation requested')

    check()
    contract = supervisor.Contract(expected_updates=64, minimum_auxiliary_nonzero=32,
        work_seconds=limits['work_seconds'], guard_with_cleanup_seconds=limits['guard_with_cleanup_seconds'],
        term_seconds=limits['term_seconds'], cleanup_seconds=limits['cleanup_seconds'],
        gate_seconds=limits['gate_seconds'], expected_gate_sha256=sources[GATE_REL])
    command = [entry.child_interpreter(private), '-B', '-I', str(REPO / ENTRY_REL),
               '--registration-commit', commit, '--game-root', str(prepared.root),
               '--private-contract', str(prepared.private_path),
               '--expected-private-contract-sha256', prepared.expected_sha,
               '--receipt-token', token]

    def receipt_sink(raw):
        _require(type(raw) is bytes and all(supervisor.inspect_receipt(raw, token, contract).values()),
                 'Invalid private receipt')
        path = output / 'child-receipt.local.json'
        entry._write_new(path, raw)
        return _bounded_file(path, entry.MAX_PRIVATE) == raw

    _stage('supervision')
    guard_result = supervisor.supervise_command(command, token, contract, cancelled=cancelled,
                                                 receipt_sink=receipt_sink)
    accepted = validate_guard_result(guard_result)
    prepared.guard_result = guard_result
    entry._write_json(output / 'guard.local.json', {'schema': 'task-pair-wsl-guard-result-v1',
                                                     'gates': guard_result})
    _require(accepted, 'Owned child did not pass supervision')
    check()
    raw = _bounded_file(output / 'child-receipt.local.json', entry.MAX_PRIVATE)
    _require(all(supervisor.inspect_receipt(raw, token, contract).values()), 'Private receipt differs')
    receipt = entry._json(raw, entry.MAX_PRIVATE)

    _stage('outputs')
    outputs = verify_private_outputs(entry, modules, prepared.root, commit, check=check)
    _require(outputs['child_reported_updates'] == receipt['completed_updates']
             and outputs['child_reported_auxiliary_nonzero'] == receipt['auxiliary_nonzero_updates'],
             'Private child reports disagree')
    _stage('final_check')
    check()
    _require(entry.registration_gate(commit)[1] == registration, 'Registration changed')
    entry.verify_registered_sources(commit, sources)
    later = entry._private_file(prepared.private_path, prepared.artifacts, entry.MAX_PRIVATE)
    _require(entry.parse_private_contract(later, prepared.expected_sha) == private,
             'Private contract changed')
    # The supervisor does not itself initialize the child's JAX/GPU backend.
    entry._runtime_after(private)
    modules.extra.verify_loaded_origins(modules.smoke, modules.root,
                                        required_modules=modules.extra.TRAIN_MODULES)
    check()
    _stage('publish')
    result = {'schema': 'task-pair-wsl-supervisor-private-result-v1',
              'technical_execution_passed': True, 'owned_session_clean': True,
              'model_files_reloaded': outputs['model_files_reloaded'],
              'weights_distinct': outputs['weights_distinct'],
              'development_screen_status': outputs['development_screen_status'],
              'private_model_sha256': outputs['model_sha256'],
              'outer_completion_confirmation': False,
              'new_games': False, 'independent_competitive_improvement': False}
    entry._write_json(output / 'supervisor-result.local.json', result)
    _require(entry._json(_bounded_file(output / 'supervisor-result.local.json', MAX_PRIVATE), MAX_PRIVATE)
             == result, 'Private result readback differs')
    check()
    return {'passed': True}


@contextmanager
def _quiet_native_output():
    """Keep source/runtime diagnostics and the child command private."""
    sys.stdout.flush()
    sys.stderr.flush()
    saved_out = saved_err = sink = None
    try:
        saved_out, saved_err = os.dup(1), os.dup(2)
        sink = os.open(os.devnull, os.O_WRONLY)
        os.dup2(sink, 1)
        os.dup2(sink, 2)
        yield
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        if saved_out is not None:
            os.dup2(saved_out, 1)
        if saved_err is not None:
            os.dup2(saved_err, 2)
        for descriptor in (saved_out, saved_err, sink):
            if descriptor is not None:
                os.close(descriptor)


class QuietParser(argparse.ArgumentParser):
    def error(self, message):
        raise ValueError('Invalid launcher arguments')


def main(argv=None):
    global STAGE
    STAGE = 'arguments'
    prepared = None
    started = time.monotonic()
    try:
        parser = QuietParser(add_help=False)
        parser.add_argument('--registration-commit', required=True)
        parser.add_argument('--game-root', required=True)
        parser.add_argument('--private-contract', required=True)
        parser.add_argument('--expected-private-contract-sha256', required=True)
        parser.add_argument('--host-nonce', required=True)
        args = parser.parse_args(argv)
        _require(type(args.host_nonce) is str and HEX_NONCE.fullmatch(args.host_nonce),
                 'Invalid host attempt identity')
        _require(sys.flags.isolated and sys.flags.dont_write_bytecode and __debug__,
                 'Expected isolated non-optimized interpreter without bytecode writes')
        with _quiet_native_output():
            prepared = prepare(args.registration_commit, args.game_root, args.private_contract,
                               args.expected_private_contract_sha256, started=started)
            result = run_prepared(prepared, args.registration_commit, host_nonce=args.host_nonce)
    except BaseException as exc:
        if prepared is not None and getattr(prepared, 'owned_output', None) is not None:
            try:
                guard = getattr(prepared, 'guard_result', None)
                cleanup = ('confirmed' if type(guard) is dict and guard.get('owned_session_clean') is True
                           else 'unconfirmed')
                prepared.entry._write_json(prepared.owned_output / 'failure.local.json',
                    {'schema': 'task-pair-wsl-supervisor-failure-v1', 'stage': STAGE,
                     'error_type': type(exc).__name__, 'owned_session_cleanup': cleanup})
            except BaseException:
                pass
        name = type(exc).__name__
        result = {'passed': False, 'stage': STAGE, 'error_type': name if name in
                  {'ValueError', 'OSError', 'TimeoutError', 'ImportError', 'MemoryError',
                   'KeyboardInterrupt', 'SystemExit'} else 'OtherError'}
    print(json.dumps(result, sort_keys=True, separators=(',', ':')), flush=True)
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    sys.exit(main())
