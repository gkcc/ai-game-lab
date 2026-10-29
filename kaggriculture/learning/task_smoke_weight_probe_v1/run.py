"""One preregistered offline probe; stdout and result contain aggregates only.

Direct startup requires an externally verified entrypoint, a clean preregistered
Git commit, and caller-owned scratch. No official game or replay is executed.
"""
import argparse
from contextlib import contextmanager, redirect_stderr, redirect_stdout
import hashlib
import importlib.machinery
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
import re
import runpy
import stat
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]
PUBLIC = ROOT.parent / 'publication' / 'ai-game-lab'
STAGE_NAME = 'task-smoke-weight-probe-v1-20261030'
SMOKE_STAGE = 'task-teacher-merged-eval-v3-smoke-20261030'
SMOKE_KEY = 'holdout-S01-round2-0-merged'
SOURCE_COMMIT = 'c0ff4328ef468cc4f6b40bd213fa728c50960505'
SOURCE_REGISTRATION = '59d6f302a72e66cbd435b1933de94b723e3c05c0'
CERTIFICATE_COMMIT = 'c60f3a68f0ced0c9034e562b68ec30623f5498a5'
CERTIFICATE = 'review/evidence/task-teacher-merged-eval-v3-smoke-result-20261030/certificate.json'
REGISTRATION = 'review/evidence/' + STAGE_NAME + '/registration.json'
RUNTIME = ('learning/task_smoke_weight_probe_v1/core.py', 'learning/task_smoke_weight_probe_v1/run.py')
SAMPLE_TURNS = tuple(i * 718 // 31 for i in range(32))
MAX_CONTROLLER_CALLS = 96
CONCLUSION_RULES = dict(task_choice_control_observed='at_least_one_same_input_first_choice_difference',
                        action_control_observed='at_least_one_complete_request_difference',
                        otherwise='not_observed_not_proof_of_no_control')
SUPERVISION = dict(wall_seconds=600, max_job_commit_bytes=4 * 1024**3,
                   min_available_bytes=4 * 1024**3, prelaunch_memory_samples=1,
                   stdio='nul_from_creation', child_environment='inherited_unchanged')
INPUT_LIMITS = dict(replay_compressed_bytes=4 * 1024**2, replay_decoded_bytes=128 * 1024**2,
                    trace_compressed_bytes=16 * 1024**2, trace_decoded_bytes=512 * 1024**2,
                    trace_line_bytes=8 * 1024**2)
_COMMIT = re.compile(r'[0-9a-f]{40}\Z')
_SHA = re.compile(r'[0-9a-f]{64}\Z')


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _git(*args):
    try:
        return subprocess.check_output(['git', '-C', str(PUBLIC), *args], stderr=subprocess.STDOUT, timeout=30)
    except (OSError, subprocess.SubprocessError):
        raise ValueError('Git identity unavailable') from None


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _json(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError('Duplicate JSON key')
            result[key] = value
        return result
    def invalid(_):
        raise ValueError('Nonfinite JSON value')
    try:
        return json.loads(raw.decode('utf-8-sig'), object_pairs_hook=pairs, parse_constant=invalid)
    except (UnicodeError, json.JSONDecodeError):
        raise ValueError('Invalid JSON') from None


def _same_json(left, right):
    """Registration limits must not accept numerically equal wrong types."""
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(_same_json(left[k], right[k]) for k in left)
    if isinstance(left, list):
        return len(left) == len(right) and all(_same_json(a, b) for a, b in zip(left, right))
    return left == right


def _read(path, maximum):
    path = Path(path)
    for part in (path, *path.parents):
        info = part.lstat()
        _require(not stat.S_ISLNK(info.st_mode) and not getattr(info, 'st_file_attributes', 0) & 0x400,
                 'Input path contains a link')
    _require(path.is_file() and path.stat().st_size <= maximum, 'Missing or oversized input')
    with path.open('rb') as stream:
        data = stream.read(maximum + 1)
    _require(len(data) <= maximum, 'Oversized input')
    return data


def _project_path(relative):
    _require(isinstance(relative, str) and relative and '\\' not in relative and ':' not in relative,
             'Invalid source path')
    p = PurePosixPath(relative)
    _require(not p.is_absolute() and p.as_posix() == relative and all(x not in ('.', '..') for x in p.parts),
             'Invalid source path')
    return ROOT.joinpath(*p.parts)


def _startup(source_commit, registration_commit, scratch):
    argv = getattr(sys, 'orig_argv', ())
    _require(isinstance(argv, list) and len(argv) >= 6 and argv[1:3] == ['-B', '-X']
             and argv[3].startswith('pycache_prefix=') and argv[4] == '-I'
             and sys.flags.dont_write_bytecode == 1 and sys.flags.isolated == 1
             and sys.flags.optimize == 0 and isinstance(sys.pycache_prefix, str), 'Untrusted Python entry')
    _require(os.name == 'nt' and os.path.samefile(sys.executable, ROOT/'.venv'/'Scripts'/'python.exe')
             and os.path.samefile(sys.prefix, ROOT/'.venv') and not os.path.samefile(sys.prefix, sys.base_prefix),
             'Unexpected project interpreter')
    _require(isinstance(source_commit, str) and _COMMIT.fullmatch(source_commit)
             and isinstance(registration_commit, str) and _COMMIT.fullmatch(registration_commit), 'Invalid commit')
    _require(Path(scratch).is_absolute(), 'Scratch must be absolute')
    _require(_git('rev-parse', 'HEAD').decode('ascii').strip() == registration_commit
             and not _git('status', '--porcelain').strip(), 'Repository must be clean and registered')
    parents = _git('rev-list', '--parents', '-n', '1', registration_commit).decode('ascii').split()
    _require(parents == [registration_commit, source_commit], 'Registration must be a direct child')
    changed = {p for p in _git('diff', '--name-only', '-z', source_commit, registration_commit, '--').decode().split('\0') if p}
    _require(changed == {REGISTRATION}, 'Registration contains unexpected changes')
    _git('merge-base', '--is-ancestor', CERTIFICATE_COMMIT, source_commit)
    reg_raw = _git('show', registration_commit + ':' + REGISTRATION)
    _require(_read(PUBLIC/REGISTRATION, 1024**2) == reg_raw, 'Registration changed')
    reg = _json(reg_raw)
    expected = dict(schema='same-observation-weight-probe-v1', status='approved_for_one_offline_probe',
        source_commit=source_commit, smoke_source_commit=SOURCE_COMMIT,
        smoke_registration_commit=SOURCE_REGISTRATION, smoke_certificate_commit=CERTIFICATE_COMMIT,
        smoke_game=SMOKE_KEY, side=0, sampled_turns=list(SAMPLE_TURNS),
        intervention='negate_task_w2_and_b2_in_memory', input_limits=INPUT_LIMITS,
        max_controller_calls=MAX_CONTROLLER_CALLS, new_games=0, official_transitions=0,
        no_economic_evaluation=True, conclusion_rules=CONCLUSION_RULES, supervision=SUPERVISION,
        retry_allowed=False, training=False, official_counterfactual_execution=False)
    _require(isinstance(reg, dict) and set(reg) == set(expected) | {'runtime_source_sha256', 'smoke_certificate_sha256'}
             and all(_same_json(reg[k], v) for k, v in expected.items())
             and all(type(x) is int for x in reg['sampled_turns']), 'Registration contract changed')
    sources = reg['runtime_source_sha256']
    _require(isinstance(sources, dict) and set(sources) == set(RUNTIME), 'Probe source set changed')
    for rel in RUNTIME:
        source = _read(_project_path(rel), 1024**2)
        _require(isinstance(sources[rel], str) and _SHA.fullmatch(sources[rel])
                 and _sha(source) == sources[rel]
                 and source == _read(PUBLIC/'kaggriculture'/rel, 1024**2)
                 and source == _git('show', source_commit + ':kaggriculture/' + rel)
                 and source == _git('show', registration_commit + ':kaggriculture/' + rel), 'Probe source changed')
    cert_raw = _git('show', CERTIFICATE_COMMIT + ':' + CERTIFICATE)
    _require(_read(PUBLIC/CERTIFICATE, 1024**2) == cert_raw
             == _git('show', source_commit + ':' + CERTIFICATE)
             == _git('show', registration_commit + ':' + CERTIFICATE)
             and _sha(cert_raw) == reg['smoke_certificate_sha256'],
             'Smoke certificate changed')
    cert = _json(cert_raw)
    launcher_rel = 'learning/panel_eval_v3/launch.py'
    launcher = _read(ROOT/launcher_rel, 1024**2)
    _require(launcher == _git('show', SOURCE_COMMIT + ':kaggriculture/' + launcher_rel)
             == _git('show', SOURCE_REGISTRATION + ':kaggriculture/' + launcher_rel), 'Frozen launcher changed')
    launch = runpy.run_path(str(ROOT/launcher_rel), run_name='probe_launcher_precheck')
    smoke = ROOT/'artifacts'/SMOKE_STAGE
    smoke_prefix = 'review/evidence/' + SMOKE_STAGE + '/'
    for name in ('root-decision.json', 'registration.json', 'freeze.json', 'pre-review.json', 'safety-registration.json'):
        data = _git('show', SOURCE_REGISTRATION + ':' + smoke_prefix + name)
        _require(_read(smoke/name, 1024**2) == data == _read(PUBLIC/smoke_prefix/name, 1024**2)
                 == _git('show', source_commit + ':' + smoke_prefix + name)
                 == _git('show', registration_commit + ':' + smoke_prefix + name),
                 'Smoke registration changed')
    config = _json(_read(smoke/'registration.json', 1024**2))
    freeze = _json(_read(smoke/'freeze.json', 1024**2))
    # Bind the previously frozen runtime to the *new* source and preregistration
    # commits too; validating only the historical smoke commits would miss a
    # later alteration in the public working tree.
    launch['_verify_frozen_runtime'](PUBLIC, ROOT, registration_commit, source_commit, freeze)
    _require(config.get('run_kind') == 'development_smoke' and config.get('selected_game_key') == SMOKE_KEY,
             'Unexpected smoke configuration')
    return reg, cert, config, smoke


def _namespace_and_runtime(scratch):
    """Reject local shadows before importing probe or frozen project modules."""
    bootstrap = runpy.run_path(str(ROOT/'learning/panel_eval_v3/bootstrap.py'), run_name='probe_bootstrap_precheck')
    bootstrap['verify_initializers'](ROOT)
    namespace = ROOT/'learning'/'task_smoke_weight_probe_v1'
    suffixes = ('.pyc', *importlib.machinery.EXTENSION_SUFFIXES)
    for suffix in ('.py', *suffixes):
        _require(not (namespace.parent/(namespace.name+suffix)).exists(), 'Probe namespace shadow')
        _require(not (namespace/('__init__'+suffix)).exists(), 'Probe namespace initializer')
    for name in ('core', 'run'):
        _require(not any((namespace/(name+suffix)).exists() for suffix in importlib.machinery.EXTENSION_SUFFIXES),
                 'Probe extension shadow')
        candidate = namespace/name
        if candidate.is_dir():
            _require(not any((candidate/('__init__'+suffix)).exists() for suffix in ('.py', *suffixes)),
                     'Probe package shadow')
    spec = importlib.machinery.PathFinder.find_spec('learning', [str(ROOT)])
    _require('learning' not in sys.modules and spec is not None and spec.loader is None
             and spec.submodule_search_locations is not None
             and [Path(p).resolve() for p in spec.submodule_search_locations] == [(ROOT/'learning').resolve()],
             'Unexpected project namespace')
    spec.submodule_search_locations = [str(ROOT/'learning')]
    package = importlib.util.module_from_spec(spec)
    package.__path__ = [str(ROOT/'learning')]
    sys.modules['learning'] = package
    sys.path.insert(0, str(ROOT))
    from learning.panel_eval_v3.isolation import validate_runtime
    validate_runtime(scratch, sys.pycache_prefix, ROOT)


def _bound_sources(cert, summary, replay_sha, trace_sha):
    """Refuse another episode even if its seat and step numbers happen to match."""
    _require(isinstance(cert, dict) and cert.get('schema') == 'bounded-development-smoke-certificate-v1'
             and cert.get('source_commit') == SOURCE_COMMIT and cert.get('registration_commit') == SOURCE_REGISTRATION
             and cert.get('stage_kind') == 'public-development-not-holdout' and cert.get('accepted_games') == 1
             and cert.get('replay_states') == 720 and cert.get('action_turns') == 719
             and cert.get('final_status') == ['DONE', 'DONE'] and cert.get('agent_errors_reported') == 0
             and cert.get('primary_replay_transitions') == 719 and cert.get('extra_replay_transitions') == 0,
             'Invalid smoke certificate')
    _require(isinstance(summary, dict) and summary.get('passed') is True and summary.get('game') == SMOKE_KEY
             and summary.get('run_kind') == 'development_smoke' and summary.get('complete_games') == 1
             and summary.get('states') == 720 and summary.get('action_turns') == 719
             and summary.get('primary_replay_transitions') == 719 and summary.get('agent_errors') == 0,
             'Invalid smoke completion')
    cert_files, summary_files = cert.get('local_artifact_sha256'), summary.get('local_artifact_sha256')
    _require(isinstance(cert_files, dict) and isinstance(summary_files, dict)
             and cert_files.get('replay') == summary_files.get('replay') == replay_sha
             and cert_files.get('trace') == summary_files.get('trace') == trace_sha
             and all(isinstance(v, str) and _SHA.fullmatch(v) for v in (replay_sha, trace_sha)),
             'Replay and trace are not the certified pair')


def _identity(path, expected, limit):
    _require(isinstance(expected, str) and _SHA.fullmatch(expected), 'Invalid source digest')
    _readable = Path(path)
    for part in (_readable, *_readable.parents):
        info = part.lstat()
        _require(not stat.S_ISLNK(info.st_mode) and not getattr(info, 'st_file_attributes', 0) & 0x400,
                 'Source path contains a link')
    size = _readable.stat().st_size
    _require(_readable.is_file() and 0 < size <= limit, 'Compressed source size exceeds limit')
    return {'path': 'private-source', 'size': size, 'sha256': expected}


def _read_replay(path, digest):
    from learning.task_action_impact_diagnostic_v1.analysis import Meter, decoded_chunks
    ident = _identity(path, digest, INPUT_LIMITS['replay_compressed_bytes'])
    meter = Meter(compressed_limit=INPUT_LIMITS['replay_compressed_bytes'],
                  decoded_limit=INPUT_LIMITS['replay_decoded_bytes'])
    data = bytearray()
    with decoded_chunks(path, ident, meter) as chunks:
        for block in chunks:
            data.extend(block)
    return _json(bytes(data))


def _trace_lines(path, digest):
    from learning.task_action_impact_diagnostic_v1.analysis import Meter, decoded_chunks
    ident = _identity(path, digest, INPUT_LIMITS['trace_compressed_bytes'])
    meter = Meter(compressed_limit=INPUT_LIMITS['trace_compressed_bytes'],
                  decoded_limit=INPUT_LIMITS['trace_decoded_bytes'])
    pending = bytearray()
    with decoded_chunks(path, ident, meter) as chunks:
        for block in chunks:
            pending.extend(block)
            start = 0
            while True:
                end = pending.find(b'\n', start)
                if end < 0:
                    break
                _require(0 < end-start <= INPUT_LIMITS['trace_line_bytes'], 'Invalid trace line size')
                yield _json(bytes(pending[start:end]))
                start = end + 1
            if start:
                del pending[:start]
            _require(len(pending) <= INPUT_LIMITS['trace_line_bytes'], 'Trace line exceeds limit')
        _require(not pending, 'Trace requires final newline')


def _write(path, value):
    data = (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode('utf-8')
    _require(len(data) <= 16384, 'Result too large')
    with path.open('xb') as stream:
        stream.write(data)


def run(source_commit, registration_commit, scratch):
    reg, cert, config, smoke = _startup(source_commit, registration_commit, scratch)
    _namespace_and_runtime(scratch)
    from learning.panel_eval_v3 import experiment as official
    from learning.task_smoke_weight_probe_v1 import core
    from learning.task_dagger_once_v1.policy import BoundedAgent
    official._check_official_runtime(config['official_loader_sha256'])
    _require(tuple(core.SAMPLE_TURNS) == SAMPLE_TURNS and core.MAX_CONTROLLER_CALLS == MAX_CONTROLLER_CALLS,
             'Sampling or call limit changed')
    summary_raw = _read(smoke/'results'/'coordinator-complete.json', 16384)
    _require(_sha(summary_raw) == cert['receipt_sha256']['coordinator'], 'Completion receipt changed')
    summary = _json(summary_raw)
    replay_sha = cert['local_artifact_sha256']['replay']
    trace_sha = cert['local_artifact_sha256']['trace']
    _bound_sources(cert, summary, replay_sha, trace_sha)
    task, crop = _project_path(config['merged_checkpoint']), _project_path(config['crop_checkpoint'])
    task_sha, crop_sha = config['merged_checkpoint_sha256'], config['crop_checkpoint_sha256']
    _require(_sha(_read(task, 128 * 1024**2)) == task_sha
             and _sha(_read(crop, 128 * 1024**2)) == crop_sha, 'Checkpoint identity changed')
    replay = smoke/'games'/SMOKE_KEY/'replay.json.gz'
    trace = smoke/'traces'/(SMOKE_KEY+'.jsonl.gz')
    _identity(replay, replay_sha, INPUT_LIMITS['replay_compressed_bytes'])
    _identity(trace, trace_sha, INPUT_LIMITS['trace_compressed_bytes'])
    output = ROOT/'artifacts'/STAGE_NAME
    output.mkdir(exist_ok=False)
    _write(output/'started.json', dict(schema='same-observation-weight-probe-v1', retry_allowed=False))
    try:
        source = _read_replay(replay, replay_sha)
        rows = _trace_lines(trace, trace_sha)
        factory = lambda: BoundedAgent(crop, crop_sha, task, task_sha, limits=config['game_limits'])
        result = core.compare(source, rows, factory, config['game_limits'], task_sha, crop_sha, side=0)
        _require(_git('rev-parse', 'HEAD').decode('ascii').strip() == registration_commit
                 and not _git('status', '--porcelain').strip(), 'Repository changed during probe')
        for rel in RUNTIME:
            local = _read(_project_path(rel), 1024**2)
            _require(_sha(local) == reg['runtime_source_sha256'][rel]
                     and local == _read(PUBLIC/'kaggriculture'/rel, 1024**2), 'Probe source changed during execution')
        _require(_sha(_read(PUBLIC/CERTIFICATE, 1024**2)) == reg['smoke_certificate_sha256']
                 and _sha(_read(task, 128 * 1024**2)) == task_sha
                 and _sha(_read(crop, 128 * 1024**2)) == crop_sha, 'Source changed during probe')
        result.update(schema='same-observation-weight-probe-v1', source_commit=source_commit,
                      registration_commit=registration_commit, smoke_certificate_sha256=reg['smoke_certificate_sha256'],
                      max_controller_calls=MAX_CONTROLLER_CALLS, new_games=0, official_transitions=0,
                      no_economic_evaluation=True, conclusion_rules=CONCLUSION_RULES, supervision=SUPERVISION,
                      retry_allowed=False, training=False, official_counterfactual_execution=False)
        _write(output/'result.json', result)
        return result
    except BaseException as exc:
        try:
            _write(output/'failure.json', dict(passed=False, error_kind=type(exc).__name__, retry_allowed=False,
                                               partial_results_published=False))
        except BaseException:
            pass
        raise


@contextmanager
def _silence():
    """Redirect current CRT streams; cached external Win32 handles remain outside this guarantee.

    The caller must start and supervise the process in an owned Job with NUL
    handles and enforce its own cleanup and resource policy.
    """
    saved = [os.dup(1), os.dup(2)]
    try:
        with open(os.devnull, 'w', encoding='utf-8') as sink:
            os.dup2(sink.fileno(), 1)
            os.dup2(sink.fileno(), 2)
            with redirect_stdout(sink), redirect_stderr(sink):
                yield
    finally:
        for fd, original in zip((1, 2), saved):
            os.dup2(original, fd)
            os.close(original)


def main():
    class QuietParser(argparse.ArgumentParser):
        def error(self, _message):
            raise ValueError('Invalid arguments')
    parser = QuietParser()
    parser.add_argument('--source-commit', required=True)
    parser.add_argument('--registration-commit', required=True)
    parser.add_argument('--scratch', required=True)
    args = parser.parse_args()
    return run(args.source_commit, args.registration_commit, args.scratch)


if __name__ == '__main__':
    try:
        with _silence():
            value = main()
    except BaseException as exc:
        print(json.dumps({'passed': False, 'error_kind': type(exc).__name__}))
        raise SystemExit(1) from None
    print(json.dumps(value, ensure_ascii=False, allow_nan=False))
