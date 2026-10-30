"""Windows host candidate: one WSL attempt, no process termination fallback.

No production registration or private host contract is created by this file.
Run only after the caller independently authenticates the host source itself.
"""

import argparse
from contextlib import contextmanager
import hashlib
import hmac
import json
import math
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
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
LAUNCH_REL = 'kaggriculture/learning/task_pair_wsl_launch_v1/launch.py'
HOST_REL = 'kaggriculture/learning/task_pair_wsl_launch_v1/host.py'
HOST_DOC_REL = 'kaggriculture/learning/task_pair_wsl_launch_v1/HOST.zh-CN.md'
HOST_OUTPUT_NAME = 'task-pair-wsl-training-host-v1-20261030'
HOST_SCHEMA = 'task-pair-wsl-host-private-contract-v1'
MAX_SOURCE = 4 * 1024 * 1024
MAX_PRIVATE = 65536
CANCEL_GRACE_SECONDS = 60
POLL_SECONDS = 0.1
MAX_UNBOUND_CANCEL_POLLS = 600
HEX_SHA = re.compile(r'[0-9a-f]{64}\Z')
HEX_COMMIT = re.compile(r'[0-9a-f]{40}\Z')
HEX_NONCE = re.compile(r'[0-9a-f]{32}\Z')
STAGE = 'arguments'

# This literal is covered by host.py's registered SHA. The WSL interpreter
# executes the exact verified byte snapshot, not a second path read. Startup
# interpreter/site code and hostile same-user concurrency remain out of scope.
BOUND_BOOTSTRAP = '''\
import hashlib, os, pathlib, stat, sys
try:
    path = pathlib.Path(sys.argv[1])
    expected = sys.argv[2]
    if len(expected) != 64 or any(c not in '0123456789abcdef' for c in expected):
        raise ValueError('Invalid source identity')
    before = path.lstat()
    limit = 4 * 1024 * 1024
    if not stat.S_ISREG(before.st_mode) or stat.S_ISLNK(before.st_mode) or before.st_size > limit:
        raise ValueError('Unexpected source file')
    descriptor = os.open(path, os.O_RDONLY | getattr(os, 'O_BINARY', 0) | getattr(os, 'O_NOFOLLOW', 0))
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode) or info.st_size > limit:
            raise ValueError('Unexpected source file')
        with os.fdopen(descriptor, 'rb', closefd=False) as stream:
            raw = stream.read(limit + 1)
        if len(raw) != info.st_size or len(raw) > limit or hashlib.sha256(raw).hexdigest() != expected:
            raise ValueError('Source identity differs')
    finally:
        os.close(descriptor)
except BaseException:
    raise SystemExit(1)
sys.argv = [str(path)] + sys.argv[3:]
scope = {'__name__': '__main__', '__file__': str(path), '__package__': None}
exec(compile(raw, '<registered-wsl-launcher>', 'exec', dont_inherit=True), scope)
'''


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _stage(value):
    global STAGE
    STAGE = value


def _bounded(path, maximum):
    before = path.lstat()
    _require(stat.S_ISREG(before.st_mode) and not stat.S_ISLNK(before.st_mode)
             and not getattr(before, 'st_file_attributes', 0) & 0x400 and before.st_size <= maximum,
             'Unexpected input file')
    flags = os.O_RDONLY | getattr(os, 'O_BINARY', 0) | getattr(os, 'O_NOFOLLOW', 0)
    descriptor = os.open(path, flags)
    try:
        info = os.fstat(descriptor)
        _require(stat.S_ISREG(info.st_mode) and info.st_size <= maximum, 'Unexpected input file')
        with os.fdopen(descriptor, 'rb', closefd=False) as stream:
            raw = stream.read(maximum + 1)
        _require(len(raw) == info.st_size and len(raw) <= maximum, 'Input length differs')
        return raw
    finally:
        os.close(descriptor)


def _git(*args):
    return subprocess.check_output(['git', '-C', str(REPO), *args], stderr=subprocess.DEVNULL, timeout=30)


def _head():
    raw = _git('rev-parse', '--verify', 'HEAD')
    _require(len(raw) == 41 and raw[-1:] == b'\n', 'Invalid Git HEAD')
    value = raw[:-1].decode('ascii')
    _require(HEX_COMMIT.fullmatch(value), 'Invalid Git HEAD')
    return value


def _blob(commit, relative):
    _require(type(commit) is str and HEX_COMMIT.fullmatch(commit)
             and relative in (REG_REL, LAUNCH_REL, HOST_REL), 'Invalid source request')
    spec = commit + ':' + relative
    size = _git('cat-file', '-s', spec)
    _require(len(size) <= 32 and size.strip().isdigit() and int(size.strip()) <= MAX_SOURCE,
             'Registered source exceeds limit')
    raw = _git('show', spec)
    _require(len(raw) == int(size.strip()), 'Registered source length differs')
    return raw


def _json(raw, maximum):
    _require(type(raw) is bytes and 0 < len(raw) <= maximum, 'Invalid metadata size')
    def pairs(items):
        out = {}
        for key, value in items:
            _require(key not in out, 'Duplicate metadata field')
            out[key] = value
        return out
    def reject(value):
        raise ValueError('Invalid metadata constant')
    try:
        value = json.loads(raw.decode('utf-8-sig'), object_pairs_hook=pairs, parse_constant=reject)
    except (UnicodeError, ValueError, TypeError, RecursionError):
        raise ValueError('Invalid metadata') from None
    _require(type(value) is dict, 'Invalid metadata object')
    return value


def authenticate_sources(commit):
    """Bind registration and source snapshots before any private input."""
    _require(type(commit) is str and HEX_COMMIT.fullmatch(commit) and _head() == commit,
             'Registration commit differs')
    raw = _blob(commit, REG_REL)
    _require(_bounded(REPO / REG_REL, MAX_SOURCE) == raw, 'Registration differs')
    data = _json(raw, MAX_SOURCE)
    sources = data.get('source_sha256')
    _require(type(sources) is dict, 'Registration sources absent')
    _require(type(sources.get(HOST_DOC_REL)) is str and HEX_SHA.fullmatch(sources[HOST_DOC_REL]),
             'Host contract documentation is unregistered')
    snapshots = {}
    for relative in (HOST_REL, LAUNCH_REL):
        digest = sources.get(relative)
        _require(type(digest) is str and HEX_SHA.fullmatch(digest), 'Required source is unregistered')
        item = _bounded(REPO / relative, MAX_SOURCE)
        _require(item == _blob(commit, relative) and hashlib.sha256(item).hexdigest() == digest,
                 'Registered source differs')
        snapshots[relative] = item
    launch = types.ModuleType('registered_wsl_launcher_snapshot')
    launch.__file__ = str(REPO / LAUNCH_REL)
    exec(compile(snapshots[LAUNCH_REL], '<registered-wsl-launcher>', 'exec', dont_inherit=True), launch.__dict__)
    entry, registration = launch.authenticated_entry(commit)
    _require(registration['source_sha256'].get(HOST_REL) == sources[HOST_REL]
             and registration['source_sha256'].get(LAUNCH_REL) == sources[LAUNCH_REL]
             and _head() == commit, 'Registered host source differs')
    entry.prebind(registration['source_sha256'])
    return launch, entry, registration


def _posix_path(value):
    _require(type(value) is str and 1 <= len(value) <= 2048 and '\\' not in value
             and all(ord(char) >= 32 and ord(char) != 127 for char in value), 'Invalid WSL path')
    path = PurePosixPath(value)
    _require(path.is_absolute() and path != PurePosixPath('/') and '..' not in path.parts
             and str(path) == value, 'Invalid WSL path')
    return path


def parse_host_contract(raw, expected_sha):
    _require(type(expected_sha) is str and HEX_SHA.fullmatch(expected_sha)
             and hashlib.sha256(raw).hexdigest() == expected_sha, 'Private host contract differs')
    value = _json(raw, MAX_PRIVATE)
    keys = {'schema', 'distribution', 'windows_game_root', 'windows_training_contract',
            'windows_wsl_executable', 'wsl_game_root', 'wsl_training_contract',
            'wsl_repo_root', 'wsl_venv_executable', 'training_contract_sha256'}
    _require(set(value) == keys and value['schema'] == HOST_SCHEMA, 'Private host contract differs')
    name = value['distribution']
    _require(type(name) is str and 1 <= len(name) <= 128 and name == name.strip()
             and all(32 <= ord(char) != 127 for char in name), 'Invalid WSL distribution')
    for name in ('windows_game_root', 'windows_training_contract', 'windows_wsl_executable'):
        text = value[name]
        _require(type(text) is str and 1 <= len(text) <= 2048
                 and '\x00' not in text and not any(ord(char) < 32 for char in text)
                 and PureWindowsPath(text).is_absolute() and '..' not in PureWindowsPath(text).parts,
                 'Invalid Windows path')
    root = _posix_path(value['wsl_game_root'])
    contract = _posix_path(value['wsl_training_contract'])
    _posix_path(value['wsl_repo_root'])
    _posix_path(value['wsl_venv_executable'])
    _require(contract.is_relative_to(root / 'artifacts')
             and type(value['training_contract_sha256']) is str
             and HEX_SHA.fullmatch(value['training_contract_sha256']), 'Invalid WSL contract identity')
    return value


def _system_wsl():
    """Use the Windows system directory, not an arbitrary PATH executable."""
    _require(os.name == 'nt', 'Windows host required')
    import ctypes
    buffer = ctypes.create_unicode_buffer(32768)
    length = ctypes.windll.kernel32.GetSystemDirectoryW(buffer, len(buffer))
    _require(0 < length < len(buffer), 'Windows system directory unavailable')
    path = (Path(buffer.value) / 'wsl.exe').resolve(strict=True)
    _require(path.is_file(), 'Windows WSL executable unavailable')
    return path


class Prepared:
    def __init__(self, launch, entry, registration, root, artifacts, host_path, host_sha,
                 host, training_path, training, started):
        self.launch, self.entry, self.registration = launch, entry, registration
        self.root, self.artifacts, self.host_path, self.host_sha = root, artifacts, host_path, host_sha
        self.host, self.training_path, self.training, self.started = host, training_path, training, started

    def __repr__(self):
        return '<PrivateRegisteredWindowsHost>'


def prepare(commit, game_root, host_path, expected_sha, *, started, clock=time.monotonic):
    _stage('registration')
    launch, entry, registration = authenticate_sources(commit)
    _require(clock() - started < registration['limits']['outer_total_seconds'], 'Outer deadline expired')
    _stage('private_contract')
    root, artifacts = entry._private_root(game_root)
    raw = entry._private_file(host_path, artifacts, MAX_PRIVATE)
    host = parse_host_contract(raw, expected_sha)
    host_path = Path(host_path).resolve(strict=True)
    _require(Path(host['windows_game_root']).resolve(strict=True) == root
             and Path(host['windows_wsl_executable']).resolve(strict=True) == _system_wsl(),
             'Registered Windows host identity differs')
    training_path = Path(host['windows_training_contract']).resolve(strict=True)
    training_raw = entry._private_file(training_path, artifacts, MAX_PRIVATE)
    _require(hashlib.sha256(training_raw).hexdigest() == host['training_contract_sha256'],
             'Private training contract differs')
    training = _json(training_raw, MAX_PRIVATE)
    runtime = training.get('runtime_identity')
    _require(training.get('schema') == entry.PRIVATE_SCHEMA and type(runtime) is dict
             and runtime.get('venv_executable') == host['wsl_venv_executable']
             and _posix_path(host['wsl_venv_executable']).parent.is_relative_to(
                 _posix_path(runtime.get('venv_prefix'))), 'Private WSL runtime differs')
    _require(clock() - started < registration['limits']['outer_total_seconds'], 'Outer deadline expired')
    return Prepared(launch, entry, registration, root, artifacts, host_path, expected_sha,
                    host, training_path, training, started)


def bound_linux_start(prepared, commit, nonce):
    """Bind a Linux-created directory to this exact Windows attempt."""
    output = prepared.artifacts / prepared.launch.OUTPUT_NAME
    _require(not output.is_symlink() and output.resolve(strict=True) == output and output.is_dir(),
             'Linux supervision directory is unavailable')
    start = _json(_bounded(output / 'started.local.json', MAX_PRIVATE), MAX_PRIVATE)
    _require(set(start) == {'schema', 'attempt', 'registration_commit', 'token',
                            'host_nonce', 'private_contract_sha256'}
             and start['schema'] == 'task-pair-wsl-supervisor-start-v1'
             and type(start['attempt']) is int and start['attempt'] == 1
             and start['registration_commit'] == commit
             and type(start['host_nonce']) is str and hmac.compare_digest(start['host_nonce'], nonce)
             and start['private_contract_sha256'] == prepared.host['training_contract_sha256']
             and type(start['token']) is str and HEX_NONCE.fullmatch(start['token']),
             'Linux start identity differs')
    return output, start


def request_cancel(prepared, commit, nonce):
    """Write only to a directory bound by this attempt's private nonce."""
    try:
        output, _ = bound_linux_start(prepared, commit, nonce)
        marker = output / 'cancel.request'
        if os.path.lexists(marker):
            return _bounded(marker, MAX_PRIVATE) == b'cancel\n'
        prepared.entry._write_new(marker, b'cancel\n')
        return _bounded(marker, MAX_PRIVATE) == b'cancel\n'
    except (OSError, ValueError, TypeError):
        return False


def _receipt(raw, token):
    value = _json(raw, MAX_PRIVATE)
    _require(set(value) == {'schema', 'token', 'passed', 'completed_updates',
                            'auxiliary_nonzero_updates', 'both_arms_changed', 'arms_differ'}
             and value['schema'] == 'task-pair-training-supervisor-receipt-v1'
             and type(value['token']) is str and hmac.compare_digest(value['token'], token)
             and value['passed'] is True and value['both_arms_changed'] is True and value['arms_differ'] is True
             and type(value['completed_updates']) is int and value['completed_updates'] == 64
             and type(value['auxiliary_nonzero_updates']) is int
             and 32 <= value['auxiliary_nonzero_updates'] <= 64,
             'Private child receipt differs')
    canonical = (json.dumps(value, sort_keys=True, separators=(',', ':')) + '\n').encode('utf-8')
    _require(raw == canonical, 'Private child receipt is not canonical')
    return value


def observe_linux(prepared, commit, nonce):
    """Inspect only bounded private status/receipt metadata, never models."""
    try:
        output, start = bound_linux_start(prepared, commit, nonce)
        report = _json(_bounded(output / 'guard.local.json', MAX_PRIVATE), MAX_PRIVATE)
        _require(set(report) == {'schema', 'gates'}
                 and report['schema'] == 'task-pair-wsl-guard-result-v1', 'Private guard report differs')
        gates = report['gates']
        passed = prepared.launch.validate_guard_result(gates)
        if not gates['owned_session_clean']:
            return {'state': 'unknown', 'linux_cleanup_confirmed': False}
        if not passed:
            return {'state': 'failed', 'linux_cleanup_confirmed': True}
        _require(not os.path.lexists(output / 'failure.local.json'), 'Linux attempt reported failure')
        _receipt(_bounded(output / 'child-receipt.local.json', MAX_PRIVATE), start['token'])
        result = _json(_bounded(output / 'supervisor-result.local.json', MAX_PRIVATE), MAX_PRIVATE)
        keys = {'schema', 'technical_execution_passed', 'owned_session_clean', 'model_files_reloaded',
                'weights_distinct', 'development_screen_status', 'private_model_sha256',
                'outer_completion_confirmation', 'new_games', 'independent_competitive_improvement'}
        hashes = result.get('private_model_sha256')
        _require(set(result) == keys and result['schema'] == 'task-pair-wsl-supervisor-private-result-v1'
                 and all(result[key] is True for key in ('technical_execution_passed', 'owned_session_clean',
                                                          'model_files_reloaded', 'weights_distinct'))
                 and result['outer_completion_confirmation'] is False and result['new_games'] is False
                 and result['independent_competitive_improvement'] is False
                 and result['development_screen_status'] in ('passed', 'failed', 'unknown')
                 and type(hashes) is dict and set(hashes) == {'baseline', 'control', 'auxiliary'}
                 and all(type(value) is str and HEX_SHA.fullmatch(value) for value in hashes.values()),
                 'Private Linux result differs')
        return {'state': 'passed', 'linux_cleanup_confirmed': True,
                'development_screen_status': result['development_screen_status']}
    except (OSError, ValueError, TypeError, KeyError):
        return {'state': 'unknown', 'linux_cleanup_confirmed': False}


def wait_one(command, *, deadline, cancel, observe, grace=CANCEL_GRACE_SECONDS,
             spawn=subprocess.Popen, clock=time.monotonic, sleep=time.sleep):
    """Wait for a single process without terminate/kill, including on timeout.

    The callbacks must bind the private attempt before touching any marker or
    accepting evidence. A missing cleanup proof always remains unknown.
    """
    _require(type(command) is list and command and all(type(x) is str and x and '\x00' not in x for x in command)
             and callable(cancel) and callable(observe) and callable(spawn) and callable(clock) and callable(sleep)
             and type(deadline) in (int, float) and type(grace) in (int, float)
             and math.isfinite(deadline) and math.isfinite(grace)
             and 0 < grace <= CANCEL_GRACE_SECONDS, 'Invalid host supervision request')
    if clock() >= deadline:
        return {'status': 'failed', 'wsl_started': False, 'wsl_exit_observed': False,
                'wsl_exit_zero': False, 'host_deadline_reached': True,
                'host_cancelled': False, 'host_wait_error': False,
                'cancel_marker_written': False, 'linux_cleanup_confirmed': False}
    try:
        process = spawn(command, shell=False, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL, close_fds=True,
                        creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0) if os.name == 'nt' else 0)
    except BaseException as exc:
        # Popen may fail after creating a process but before returning its
        # handle. Only a nonce-bound cancellation is safe; there is no PID or
        # exit proof and this path must always remain unknown.
        cancel_written = False
        try:
            stop_at = clock() + grace
        except BaseException:
            try:
                cancel_written = cancel() is True
            except BaseException:
                pass
        else:
            for _ in range(MAX_UNBOUND_CANCEL_POLLS):
                try:
                    cancel_written = cancel() is True
                except BaseException:
                    break
                if cancel_written:
                    break
                try:
                    if clock() >= stop_at:
                        break
                    sleep(POLL_SECONDS)
                except BaseException:
                    break
        return {'status': 'unknown', 'wsl_started': None, 'wsl_exit_observed': False,
                'wsl_exit_zero': False, 'host_deadline_reached': False,
                'host_cancelled': isinstance(exc, KeyboardInterrupt), 'host_wait_error': True,
                'cancel_marker_written': cancel_written, 'linux_cleanup_confirmed': False}
    timed_out = False
    interrupted = False
    wait_error = False
    cancel_written = False
    grace_deadline = None

    def unconfirmed(*, cancellation, problem):
        nonlocal cancel_written
        if not cancel_written:
            try:
                cancel_written = cancel() is True
            except BaseException:
                pass
        return {'status': 'unknown', 'wsl_started': True, 'wsl_exit_observed': False,
                'wsl_exit_zero': False, 'host_deadline_reached': timed_out,
                'host_cancelled': cancellation, 'host_wait_error': problem,
                'cancel_marker_written': cancel_written, 'linux_cleanup_confirmed': False}

    while True:
        try:
            now = clock()
        except BaseException as exc:
            return unconfirmed(cancellation=interrupted or isinstance(exc, KeyboardInterrupt), problem=True)
        if not timed_out and now >= deadline:
            timed_out = True
            if grace_deadline is None:
                grace_deadline = now + grace
        try:
            code = process.poll()
        except KeyboardInterrupt:
            interrupted = True
            if grace_deadline is None:
                grace_deadline = now + grace
            code = None
        except BaseException:
            wait_error = True
            if grace_deadline is None:
                grace_deadline = now + grace
            code = None
        if code is not None:
            if type(code) is not int:
                return unconfirmed(cancellation=interrupted, problem=True)
            break
        if timed_out or interrupted or wait_error:
            if not cancel_written:
                try:
                    cancel_written = cancel() is True
                except BaseException:
                    cancel_written = False
            if now >= grace_deadline:
                return {'status': 'unknown', 'wsl_started': True, 'wsl_exit_observed': False,
                        'wsl_exit_zero': False, 'host_deadline_reached': timed_out, 'host_cancelled': interrupted,
                        'host_wait_error': wait_error, 'cancel_marker_written': cancel_written,
                        'linux_cleanup_confirmed': False}
        try:
            sleep(POLL_SECONDS)
        except KeyboardInterrupt:
            interrupted = True
            if grace_deadline is None:
                grace_deadline = now + grace
        except BaseException:
            return unconfirmed(cancellation=interrupted, problem=True)
    try:
        evidence = observe()
        valid = (type(evidence) is dict and evidence.get('state') in ('passed', 'failed', 'unknown')
                 and type(evidence.get('linux_cleanup_confirmed')) is bool)
        if valid and evidence['state'] == 'passed':
            valid = evidence.get('development_screen_status') in ('passed', 'failed', 'unknown')
    except BaseException:
        evidence, valid = {}, False
    clean = valid and evidence['linux_cleanup_confirmed'] is True
    if clean and code == 0 and not timed_out and not interrupted and not wait_error and evidence['state'] == 'passed':
        status = 'passed'
    elif clean and evidence['state'] in ('passed', 'failed'):
        status = 'failed'
    else:
        status = 'unknown'
    result = {'status': status, 'wsl_started': True, 'wsl_exit_observed': True,
              'wsl_exit_zero': code == 0, 'host_deadline_reached': timed_out,
              'host_cancelled': interrupted, 'host_wait_error': wait_error, 'cancel_marker_written': cancel_written,
              'linux_cleanup_confirmed': clean}
    if status == 'passed':
        result['development_screen_status'] = evidence['development_screen_status']
    return result


def run_prepared(prepared, commit, *, clock=time.monotonic, sleep=time.sleep, spawn=subprocess.Popen):
    """Start exactly one authenticated WSL launcher and keep failures local."""
    launch, entry, registration, host = prepared.launch, prepared.entry, prepared.registration, prepared.host
    output = prepared.artifacts / HOST_OUTPUT_NAME
    linux_output = prepared.artifacts / launch.OUTPUT_NAME
    child_output = prepared.artifacts / entry.OUTPUT_NAME
    _stage('output')
    _require(output.resolve(strict=False) == output and not os.path.lexists(linux_output)
             and not os.path.lexists(child_output), 'A prior attempt already exists')
    output.mkdir(mode=0o700, exist_ok=False)
    prepared.owned_output = output
    nonce = secrets.token_hex(16)
    start = {'schema': 'task-pair-wsl-host-start-v1', 'attempt': 1, 'registration_commit': commit,
             'host_nonce': nonce, 'host_contract_sha256': prepared.host_sha,
             'training_contract_sha256': host['training_contract_sha256']}
    entry._write_json(output / 'started.local.json', start)
    _require(_json(_bounded(output / 'started.local.json', MAX_PRIVATE), MAX_PRIVATE) == start,
             'Private host start record differs')
    launcher = _posix_path(host['wsl_repo_root']) / LAUNCH_REL
    command = [str(_system_wsl()), '--distribution', host['distribution'], '--exec',
               host['wsl_venv_executable'], '-B', '-I', '-c', BOUND_BOOTSTRAP, str(launcher),
               registration['source_sha256'][LAUNCH_REL],
               '--registration-commit', commit, '--game-root', host['wsl_game_root'],
               '--private-contract', host['wsl_training_contract'],
               '--expected-private-contract-sha256', host['training_contract_sha256'],
               '--host-nonce', nonce]
    deadline = prepared.started + registration['limits']['outer_total_seconds']
    _stage('supervision')
    result = wait_one(command, deadline=deadline,
        cancel=lambda: request_cancel(prepared, commit, nonce),
        observe=lambda: observe_linux(prepared, commit, nonce),
        spawn=spawn, clock=clock, sleep=sleep)
    if result['status'] == 'passed':
        _stage('final_check')
        _require(entry.registration_gate(commit)[1] == registration, 'Registration changed')
        entry.verify_registered_sources(commit, registration['source_sha256'])
        host_raw = entry._private_file(prepared.host_path, prepared.artifacts, MAX_PRIVATE)
        _require(parse_host_contract(host_raw, prepared.host_sha) == host, 'Private host contract changed')
        training_raw = entry._private_file(prepared.training_path, prepared.artifacts, MAX_PRIVATE)
        _require(hashlib.sha256(training_raw).hexdigest() == host['training_contract_sha256'],
                 'Private training contract changed')
        if clock() >= deadline:
            result['status'], result['host_deadline_reached'] = 'failed', True
            result.pop('development_screen_status', None)
    _stage('publish')
    private_result = {'schema': 'task-pair-wsl-host-private-result-v1', 'gates': result,
                      'host_exit_confirmation': False, 'new_games': False}
    entry._write_json(output / 'host-result.local.json', private_result)
    _require(_json(_bounded(output / 'host-result.local.json', MAX_PRIVATE), MAX_PRIVATE) == private_result,
             'Private host result readback differs')
    if result['status'] == 'passed' and clock() >= deadline:
        entry._write_json(output / 'host-late.local.json',
                          {'schema': 'task-pair-wsl-host-late-v1', 'deadline_reached': True})
        result['status'], result['host_deadline_reached'] = 'failed', True
        result.pop('development_screen_status', None)
    return result


@contextmanager
def _quiet_native_output():
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
        raise ValueError('Invalid host arguments')


def main(argv=None):
    global STAGE
    STAGE = 'arguments'
    prepared = None
    started = time.monotonic()
    try:
        parser = QuietParser(add_help=False)
        parser.add_argument('--registration-commit', required=True)
        parser.add_argument('--game-root', required=True)
        parser.add_argument('--host-contract', required=True)
        parser.add_argument('--expected-host-contract-sha256', required=True)
        args = parser.parse_args(argv)
        _require(os.name == 'nt' and sys.flags.isolated and sys.flags.dont_write_bytecode and __debug__,
                 'Expected isolated Windows host without bytecode writes')
        with _quiet_native_output():
            prepared = prepare(args.registration_commit, args.game_root, args.host_contract,
                               args.expected_host_contract_sha256, started=started)
            result = run_prepared(prepared, args.registration_commit)
    except BaseException as exc:
        if prepared is not None and getattr(prepared, 'owned_output', None) is not None:
            try:
                prepared.entry._write_json(prepared.owned_output / 'failure.local.json',
                    {'schema': 'task-pair-wsl-host-failure-v1', 'stage': STAGE,
                     'error_type': type(exc).__name__, 'linux_cleanup': 'unconfirmed'})
            except BaseException:
                pass
        kind = type(exc).__name__
        result = {'status': 'unknown', 'stage': STAGE,
                  'error_type': kind if kind in {'ValueError', 'OSError', 'TimeoutError', 'ImportError',
                       'MemoryError', 'KeyboardInterrupt', 'SystemExit'} else 'OtherError'}
    print(json.dumps(result, sort_keys=True, separators=(',', ':')), flush=True)
    return {'passed': 0, 'failed': 1, 'unknown': 2}[result['status']]


if __name__ == '__main__':
    sys.exit(main())
