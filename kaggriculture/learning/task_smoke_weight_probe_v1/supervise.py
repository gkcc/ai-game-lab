"""One owned, NUL-stdio Job for the preregistered offline probe.

The caller must verify this supervisor's bytes before executing it.  This file
can bind its source after startup, but cannot undo code already executed.
``verify`` checks only source and namespace and creates a scratch-local ready
receipt; ``run`` repeats that check in the same scratch before one attempt.
The standalone helper identity is checked using the prior salted commitment.
No environment or BLAS thread settings are changed by this supervisor.
"""
import argparse
import ctypes
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import runpy
import stat
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[2]
PUBLIC = ROOT.parent/'publication'/'ai-game-lab'
BASE_COMMIT = 'c60f3a68f0ced0c9034e562b68ec30623f5498a5'
SMOKE_SOURCE = 'c0ff4328ef468cc4f6b40bd213fa728c50960505'
SMOKE_REGISTRATION = '59d6f302a72e66cbd435b1933de94b723e3c05c0'
STAGE_NAME = 'task-smoke-weight-probe-v1-20261030'
SMOKE_STAGE = 'task-teacher-merged-eval-v3-smoke-20261030'
REGISTRATION = 'review/evidence/'+STAGE_NAME+'/registration.json'
PACKAGE = 'learning/task_smoke_weight_probe_v1/'
SOURCE_FILES = tuple(PACKAGE+name for name in (
    'core.py', 'run.py', 'README.zh-CN.md', 'test_agent.py', 'test_core.py',
    'test_registration.py', 'test_run.py', 'supervise.py', 'test_supervise.py'))
PUBLIC_SOURCE_CHANGES = frozenset('kaggriculture/'+name for name in SOURCE_FILES) | frozenset((
    'review/PROGRESS.md', 'review/evidence/task-smoke-weight-probe-v1-source-20261030/validation.json'))
SUPERVISION = dict(wall_seconds=600, max_job_commit_bytes=4*1024**3,
                   min_available_bytes=4*1024**3, prelaunch_memory_samples=1,
                   stdio='nul_from_creation', child_environment='inherited_unchanged')
_COMMIT = re.compile(r'[0-9a-f]{40}\Z')
_LIMIT_JOB_MEMORY = 0x200
_KILL_ON_CLOSE = 0x2000
_READY_SCOPE = 'source_and_namespace_only_no_replay_trace_or_checkpoint'


def _require(value, message):
    if not value:
        raise ValueError(message)


class SharedDeadline:
    """A single UTC and monotonic budget shared by both owned child Jobs."""
    def __init__(self):
        now = time.monotonic()
        _require(math.isfinite(now), 'Unreliable monotonic clock')
        self.last = now
        self.monotonic_end = now + SUPERVISION['wall_seconds']
        self.utc_end = datetime.now(timezone.utc) + timedelta(seconds=SUPERVISION['wall_seconds'])

    def remaining(self):
        current = time.monotonic()
        _require(math.isfinite(current) and current >= self.last, 'Unreliable monotonic clock')
        self.last = current
        return min(self.monotonic_end-current, (self.utc_end-datetime.now(timezone.utc)).total_seconds())


def _git(clock, *args):
    remaining = clock.remaining()
    _require(remaining > 0, 'Supervisor deadline reached')
    try:
        return subprocess.check_output(['git', '-C', str(PUBLIC), *args], stderr=subprocess.STDOUT,
                                       timeout=min(30, remaining))
    except (OSError, subprocess.SubprocessError):
        raise ValueError('Git identity unavailable') from None


def _read(path, maximum=1024**2):
    path = Path(path)
    for part in (path, *path.parents):
        info = part.lstat()
        _require(not stat.S_ISLNK(info.st_mode) and not getattr(info, 'st_file_attributes', 0) & 0x400,
                 'Input path contains a link')
    _require(path.is_file() and path.stat().st_size <= maximum, 'Missing or oversized input')
    with path.open('rb') as stream:
        value = stream.read(maximum+1)
    _require(len(value) <= maximum, 'Oversized input')
    return value


def _json(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            _require(key not in result, 'Duplicate metadata field')
            result[key] = value
        return result
    try:
        return json.loads(raw.decode('utf-8-sig'), object_pairs_hook=pairs,
                          parse_constant=lambda _: (_ for _ in ()).throw(ValueError('Invalid number')))
    except (UnicodeError, json.JSONDecodeError):
        raise ValueError('Invalid metadata') from None


def _same_json(left, right):
    """Do not accept JSON booleans or floats as registered integer limits."""
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(_same_json(left[k], right[k]) for k in left)
    if isinstance(left, list):
        return len(left) == len(right) and all(_same_json(a, b) for a, b in zip(left, right))
    return left == right


def _startup():
    argv = getattr(sys, 'orig_argv', ())
    _require(os.name == 'nt' and isinstance(argv, list) and len(argv) >= 4
             and argv[1:3] == ['-B', '-I'] and sys.flags.dont_write_bytecode == 1
             and sys.flags.isolated == 1 and sys.flags.optimize == 0 and 'PYTHONOPTIMIZE' not in os.environ
             and os.path.samefile(sys.executable, ROOT/'.venv'/'Scripts'/'python.exe')
             and os.path.samefile(sys.prefix, ROOT/'.venv') and not os.path.samefile(sys.prefix, sys.base_prefix),
             'Unexpected supervisor interpreter')


def _source_check(source, registration, clock):
    """Bind the direct source and registration commits before importing launcher."""
    _require(isinstance(source, str) and _COMMIT.fullmatch(source)
             and isinstance(registration, str) and _COMMIT.fullmatch(registration), 'Invalid commit')
    head = _git(clock, 'rev-parse', 'HEAD').decode('ascii').strip()
    _require(head == registration and not _git(clock, 'status', '--porcelain').strip(),
             'Repository is not clean at registration')
    parents = _git(clock, 'rev-list', '--parents', '-n', '1', source).decode('ascii').split()
    _require(parents == [source, BASE_COMMIT], 'Unexpected source parent')
    changed = {p for p in _git(clock, 'diff', '--name-only', '-z', BASE_COMMIT, source, '--').decode().split('\0') if p}
    _require(changed == PUBLIC_SOURCE_CHANGES, 'Unexpected source changes')
    parents = _git(clock, 'rev-list', '--parents', '-n', '1', registration).decode('ascii').split()
    _require(parents == [registration, source], 'Unexpected registration parent')
    changed = {p for p in _git(clock, 'diff', '--name-only', '-z', source, registration, '--').decode().split('\0') if p}
    _require(changed == {REGISTRATION}, 'Unexpected registration changes')
    for relative in SOURCE_FILES:
        committed = _git(clock, 'show', source+':kaggriculture/'+relative)
        _require(committed == _git(clock, 'show', registration+':kaggriculture/'+relative)
                 == _read(ROOT/relative) == _read(PUBLIC/'kaggriculture'/relative), 'Source bytes changed')
    raw = _git(clock, 'show', registration+':'+REGISTRATION)
    _require(raw == _read(PUBLIC/REGISTRATION), 'Registration bytes changed')
    reg = _json(raw)
    _require(isinstance(reg, dict) and reg.get('source_commit') == source
             and _same_json(reg.get('supervision'), SUPERVISION), 'Supervisor contract changed')
    launch_rel = 'learning/panel_eval_v3/launch.py'
    launch_bytes = _read(ROOT/launch_rel)
    _require(launch_bytes == _git(clock, 'show', SMOKE_SOURCE+':kaggriculture/'+launch_rel)
             == _git(clock, 'show', source+':kaggriculture/'+launch_rel)
             == _git(clock, 'show', registration+':kaggriculture/'+launch_rel), 'Launcher bytes changed')
    _require(_git(clock, 'rev-parse', 'HEAD').decode('ascii').strip() == registration
             and not _git(clock, 'status', '--porcelain').strip(), 'Repository changed during verification')
    entry = _read(ROOT/PACKAGE/'run.py')
    return reg, hashlib.sha256(entry).hexdigest()


def _load_launcher():
    return runpy.run_path(str(ROOT/'learning'/'panel_eval_v3'/'launch.py'), run_name='probe_outer_launcher')


def _load_helper(launcher, helper_path, clock):
    """Reuse the old preregistered salted identity; do not expose salt or path."""
    smoke = ROOT/'artifacts'/SMOKE_STAGE
    prefix = 'review/evidence/'+SMOKE_STAGE+'/'
    documents = []
    for name in ('registration.json', 'root-decision.json'):
        raw = _read(smoke/name)
        _require(raw == _git(clock, 'show', SMOKE_REGISTRATION+':'+prefix+name), 'Smoke metadata changed')
        documents.append(_json(raw))
    helper, _ = launcher['_helper'](helper_path, smoke, documents[0], documents[1])
    return helper


def _limited_job(launcher):
    """Use Windows' job-wide committed-memory limit and query it back."""
    base, structure = launcher['WindowsJob'], launcher['_ExtendedLimit']
    class LimitedJob(base):
        def __init__(self):
            super().__init__()
            self.limit_verified = False

        def configure(self, job):
            value = structure()
            value.BasicLimitInformation.LimitFlags = _KILL_ON_CLOSE | _LIMIT_JOB_MEMORY
            value.JobMemoryLimit = SUPERVISION['max_job_commit_bytes']
            self._require(self.k.SetInformationJobObject(job, 9, ctypes.byref(value), ctypes.sizeof(value)))
            found = structure()
            self._require(self.k.QueryInformationJobObject(job, 9, ctypes.byref(found), ctypes.sizeof(found), None))
            flags = found.BasicLimitInformation.LimitFlags
            _require(flags & (_KILL_ON_CLOSE | _LIMIT_JOB_MEMORY) == (_KILL_ON_CLOSE | _LIMIT_JOB_MEMORY)
                     and found.JobMemoryLimit == SUPERVISION['max_job_commit_bytes'], 'Job memory limit mismatch')
            self.limit_verified = True
    return LimitedJob()


def _memory_ready():
    """One current availability sample; it is not a future-memory guarantee."""
    import psutil
    available = psutil.virtual_memory().available
    _require(type(available) is int and available >= SUPERVISION['min_available_bytes'],
             'Insufficient current available memory')


_VERIFY_CODE = r'''import hashlib,json,runpy,sys
from pathlib import Path
entry,source,registration,scratch,digest=sys.argv[1:]
module=runpy.run_path(entry,run_name='probe_readiness_precheck')
module['_startup'](source,registration,scratch)
module['_namespace_and_runtime'](scratch)
if Path(sys.pycache_prefix).exists(): raise ValueError('Unexpected cache')
receipt={'source_commit':source,'registration_commit':registration,'entry_sha256':digest,
         'scope':'source_and_namespace_only_no_replay_trace_or_checkpoint'}
with (Path(scratch)/'ready.json').open('x',encoding='utf-8') as stream: json.dump(receipt,stream)
'''


def _child(launcher, helper, scratch, clock, command, prefix):
    _require(clock.remaining() > 0, 'Supervisor deadline reached')
    _memory_ready()
    api = _limited_job(launcher)
    result = launcher['_supervise'](api, command, ROOT, clock.utc_end, helper, scratch,
                                    helper.default_root(), _clock=clock)
    _require(api.limit_verified and isinstance(result, dict) and result.get('passed') is True
             and all(result.get(key) is True for key in ('child_reaped', 'job_members_zero', 'handles_closed', 'cleanup_known'))
             and type(result.get('exit_code')) is int and result['exit_code'] == 0 and not _present(prefix),
             'Owned child or cleanup failed')
    return result


def _write(path, value):
    raw = (json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2)+'\n').encode('utf-8')
    _require(len(raw) <= 16384, 'Receipt too large')
    with Path(path).open('xb') as stream:
        stream.write(raw)


def _present(path):
    try:
        path.lstat()
        return True
    except FileNotFoundError:
        return False


def _trusted_artifacts():
    artifacts = ROOT/'artifacts'
    for path in (artifacts, *artifacts.parents):
        info = path.lstat()
        _require(not stat.S_ISLNK(info.st_mode) and not getattr(info, 'st_file_attributes', 0) & 0x400,
                 'Artifact path contains a link')
    _require(artifacts.is_dir() and artifacts.resolve(strict=True) == artifacts, 'Invalid artifact directory')
    return artifacts


def supervise(mode, source, registration, helper_path):
    """Verify or perform the single offline attempt; never print private paths."""
    _require(mode in ('verify', 'run'), 'Invalid supervisor mode')
    _startup()
    _require(isinstance(source, str) and _COMMIT.fullmatch(source)
             and isinstance(registration, str) and _COMMIT.fullmatch(registration), 'Invalid commit')
    clock = SharedDeadline()
    artifacts = _trusted_artifacts()
    # Check the local verification gate *before* Git or any owned child is
    # started.  A failed or interrupted attempt is never silently adopted.
    verify_attempt = artifacts/(STAGE_NAME+'-supervision-verify')
    run_attempt = artifacts/(STAGE_NAME+'-supervision-attempt')
    _require(not _present(run_attempt) and not _present(artifacts/STAGE_NAME),
             'Attempt already used or cleanup unresolved')
    if mode == 'run':
        _require(_present(verify_attempt) and verify_attempt.is_dir(), 'Prior verification missing')
        verified = _json(_read(verify_attempt/'result.local.json', 16384))
        _require(_same_json(verified, dict(passed=True, source_commit=source, registration_commit=registration,
                                           cleanup_known=True, retry_allowed=False)), 'Prior verification not accepted')
    else:
        _require(not _present(verify_attempt), 'Verification already attempted')
    # Argument and source rejections up to this point do not consume the
    # attempt.  After the exclusive directory exists, failure blocks retry.
    _, entry_sha = _source_check(source, registration, clock)
    launcher = _load_launcher()
    attempt = verify_attempt if mode == 'verify' else run_attempt
    attempt.mkdir(exist_ok=False)
    _write(attempt/'started.json', dict(source_commit=source, registration_commit=registration,
                                        retry_allowed=False, scope=('source_namespace_only' if mode == 'verify'
                                                                     else 'one_offline_attempt')))
    scratch = None
    run_id = None
    try:
        # This validates the standalone helper's salted identity before its
        # code runs; the loader disables any cross-chat stale sweep.
        helper = _load_helper(launcher, helper_path, clock)
        with helper.scratch_directory('weight-probe-supervisor') as scratch:
            marker = helper.read_marker(scratch, root=helper.default_root())
            run_id = marker['run_id']
            prefix = scratch/'verify-cache'
            verify = [sys.executable, '-B', '-X', 'pycache_prefix='+str(prefix), '-I', '-c', _VERIFY_CODE,
                      str(ROOT/PACKAGE/'run.py'), source, registration, str(scratch), entry_sha]
            _child(launcher, helper, scratch, clock, verify, prefix)
            ready = _json(_read(scratch/'ready.json', 16384))
            _require(ready == dict(source_commit=source, registration_commit=registration, entry_sha256=entry_sha,
                                   scope=_READY_SCOPE) and clock.remaining() > 0, 'Ready receipt or deadline changed')
            if mode == 'run':
                _, current_sha = _source_check(source, registration, clock)
                _require(current_sha == entry_sha and clock.remaining() > 0, 'Source or deadline changed')
                prefix = scratch/'run-cache'
                command = [sys.executable, '-B', '-X', 'pycache_prefix='+str(prefix), '-I',
                           str(ROOT/PACKAGE/'run.py'), '--source-commit', source,
                           '--registration-commit', registration, '--scratch', str(scratch)]
                _child(launcher, helper, scratch, clock, command, prefix)
                result = _json(_read(artifacts/STAGE_NAME/'result.json', 16384))
                _require(isinstance(result, dict) and result.get('passed') is True
                         and result.get('source_commit') == source and result.get('registration_commit') == registration
                         and clock.remaining() > 0,
                         'Probe result missing or changed')
        _require(scratch is not None and not _present(scratch) and clock.remaining() > 0,
                 'Scratch cleanup or deadline unconfirmed')
        _write(attempt/'result.local.json', dict(passed=True, source_commit=source,
                                                registration_commit=registration, cleanup_known=True,
                                                retry_allowed=False))
        return dict(passed=True, mode=mode, source_ready=True, cleanup_known=True, retry_allowed=False,
                    prelaunch_memory='one conditional sample per child', job_commit_limit_verified=True)
    except BaseException as exc:
        retained = scratch is not None and _present(scratch)
        recovery = dict(passed=False, error_kind=type(exc).__name__, cleanup_known=not retained and scratch is not None,
                        retry_allowed=False, scratch_retained=retained,
                        scratch_path_if_retained=str(scratch) if retained else None,
                        scratch_run_id_if_retained=run_id if retained else None)
        # The recovery record is local-only. A failure after entering this
        # block always blocks future attempts, including failed verification.
        try:
            _write(attempt/'failure.local.json', recovery)
        except BaseException:
            pass
        return dict(passed=False, mode=mode, error_kind=type(exc).__name__,
                    cleanup_known=recovery['cleanup_known'], retry_allowed=False)


def main():
    class QuietParser(argparse.ArgumentParser):
        def error(self, _message):
            raise ValueError('Invalid arguments')
    parser = QuietParser()
    parser.add_argument('--mode', choices=('verify', 'run'), required=True)
    parser.add_argument('--source-commit', required=True)
    parser.add_argument('--registration-commit', required=True)
    parser.add_argument('--helper', required=True)
    args = parser.parse_args()
    return supervise(args.mode, args.source_commit, args.registration_commit, args.helper)


if __name__ == '__main__':
    try:
        result = main()
    except BaseException as exc:
        result = dict(passed=False, error_kind=type(exc).__name__, retry_allowed=False)
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))
    raise SystemExit(0 if result.get('passed') is True else 1)
