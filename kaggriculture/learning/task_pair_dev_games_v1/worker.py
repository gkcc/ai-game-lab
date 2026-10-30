"""Bind a verified worker snapshot to one historical-development job.

The caller must use a dedicated child process and independently authenticate
the private registration, sources and model paths.  This module is not a CLI,
registration authority, process supervisor, or real-game authorization gate.
"""
from contextlib import contextmanager, redirect_stderr, redirect_stdout
import hashlib
import json
import os
from pathlib import Path
import sys
import threading
from types import ModuleType

from .panel import Panel, _JOB, _PLAN
from . import capture


MAX_WORKER_BYTES = 4 * 1024 * 1024
MAX_JOB_BYTES = 16384
WORKER_RELATIVE = 'learning/task_anchor_once_v1/games.py'
NATIVE_OUTPUT_SCOPE = ('Python stdout/stderr and standard fd 1/2 redirected to NUL by worker; '
                       'other native handles not independently verified')


def _single_python_thread():
    if (threading.current_thread() is not threading.main_thread()
            or len(threading.enumerate()) != 1):
        raise RuntimeError('Worker requires one Python thread')


def _registered(check):
    if not callable(check) or check() is not True:
        raise ValueError('External registration check failed')


def _root_and_stage(game_root, stage):
    root = Path(game_root)
    output = Path(stage)
    if not root.is_absolute() or not output.is_absolute():
        raise ValueError('Absolute game and stage roots required')
    original_root, original_output = root.absolute(), output.absolute()
    root = root.resolve(strict=True)
    output = output.resolve(strict=False)
    artifacts = root / 'artifacts'
    if (original_root != root or original_output != output
            or artifacts.resolve(strict=False) != artifacts
            or output == artifacts or not output.is_relative_to(artifacts)):
        raise ValueError('Stage is outside the game artifacts root')
    return root, output


def _load_snapshot(source, expected_sha, root, check):
    _single_python_thread()
    _registered(check)
    if (type(source) is not bytes or not 0 < len(source) <= MAX_WORKER_BYTES
            or hashlib.sha256(source).hexdigest() != expected_sha):
        raise ValueError('Worker snapshot identity differs')
    path = root / WORKER_RELATIVE
    module = ModuleType('learning.task_anchor_once_v1._pair_dev_worker')
    module.__file__ = str(path)
    module.__package__ = 'learning.task_anchor_once_v1'
    # Compile and execute exactly the bounded bytes whose SHA was just checked.
    exec(compile(source, str(path), 'exec'), module.__dict__)
    if (not hasattr(module, 'ROOT') or Path(module.ROOT).resolve(strict=True) != root
            or any(not callable(getattr(module, name, None))
                   for name in ('verify', 'task_identity', 'read', 'wrapper', 'job',
                                'audit', 'reexecute', 'observation', 'owned_writes', 'stable',
                                'write_new'))):
        raise ValueError('Worker snapshot contract differs')
    _registered(check)
    return module


def _unique_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('Duplicate job key')
        result[key] = value
    return result


def _read_job(path):
    with Path(path).open('rb') as stream:
        raw = stream.read(MAX_JOB_BYTES + 1)
    if len(raw) > MAX_JOB_BYTES:
        raise ValueError('Job document is too large')
    def invalid_constant(_):
        raise ValueError('Nonfinite job value')
    return json.loads(raw.decode('utf-8-sig'), object_pairs_hook=_unique_pairs,
                      parse_constant=invalid_constant)


@contextmanager
def _bound_worker(source, panel, root, stage, expected, check):
    """Bind only a new module object, never the canonical imported worker."""
    module = _load_snapshot(source, panel.worker_sha, root, check)
    originals = module.verify, module.task_identity, module.read, module.stable, module.write_new
    expected = dict(expected)
    job_path = stage / 'job-inputs' / (expected['game'] + '.json')
    storage_path = stage / 'job-results' / (expected['game'] + '.storage.json')

    def verified(directory):
        if Path(directory).resolve(strict=False) != stage:
            raise ValueError('Unexpected development stage')
        _registered(check)
        return panel.worker_config()

    def task_identity(directory, plan, cfg):
        if Path(directory).resolve(strict=False) != stage or cfg != panel.worker_config():
            raise ValueError('Unexpected development identity context')
        if type(plan) is not dict or not _PLAN.issubset(plan):
            raise ValueError('Unexpected development game')
        extra_job_keys = _JOB - _PLAN
        present = extra_job_keys.intersection(plan)
        if present and present != extra_job_keys:
            raise ValueError('Incomplete development job identity')
        keys = _JOB if present else _PLAN
        panel.validate_plan({key: plan[key] for key in keys}, job=bool(present))
        if any(plan[key] != expected[key] for key in ('game', 'seed_alias', 'seed',
                                                      'opponent', 'side', 'policy', 'split')):
            raise ValueError('Unexpected development game')
        relative, digest = panel.task_model(plan)
        return root / relative, digest

    def checked_read(path):
        if Path(path).resolve(strict=False) == job_path:
            value = panel.validate_plan(_read_job(path), job=True)
            if value != expected:
                raise ValueError('Private job identity changed')
            return value
        return originals[2](path)

    # The old stable() deliberately removes inference_seconds.  Its remaining
    # JSON structure must compare in insertion order and preserve scalar type.
    from learning.task_bc_error_diagnostics_v1.shadow import _typed
    def ordered_stable(value):
        return _typed(originals[3](value))

    def checked_write(path, value):
        target = Path(path).resolve(strict=False)
        if target == storage_path:
            if (type(value) is not dict or value.get('game') != expected['game']
                    or value.get('native_output_accounted_by') != 'bounded parent process log'):
                raise ValueError('Unexpected worker storage receipt')
            value = dict(value, native_output_accounted_by=NATIVE_OUTPUT_SCOPE)
        elif target.parent == storage_path.parent and target.name.endswith('.storage.json'):
            raise ValueError('Storage receipt belongs to another game')
        return originals[4](path, value)

    try:
        module.verify, module.task_identity, module.read, module.stable, module.write_new = (
            verified, task_identity, checked_read, ordered_stable, checked_write)
        yield module
    finally:
        module.verify, module.task_identity, module.read, module.stable, module.write_new = originals


@contextmanager
def _quiet_output():
    """Suppress both Python streams and inherited native fd 1/2 in this child."""
    sys.stdout.flush()
    sys.stderr.flush()
    saved = []
    with open(os.devnull, 'w', encoding='utf-8') as sink:
        try:
            saved.append(os.dup(1))
            saved.append(os.dup(2))
            os.dup2(sink.fileno(), 1)
            os.dup2(sink.fileno(), 2)
            with redirect_stdout(sink), redirect_stderr(sink):
                yield
        finally:
            restore_failed = False
            for index, fd in enumerate(saved, 1):
                try:
                    os.dup2(fd, index)
                except OSError:
                    restore_failed = True
                finally:
                    os.close(fd)
            if restore_failed:
                raise RuntimeError('Output restoration failed')


def make_wrapper(*, panel, source, game_root, stage, plan, check):
    """Create one private wrapper; the outer registered coordinator owns it."""
    try:
        _single_python_thread()
        with _quiet_output():
            if type(panel) is not Panel:
                raise ValueError('Expected development panel')
            root, stage = _root_and_stage(game_root, stage)
            plan = panel.validate_plan(plan)
            with _bound_worker(source, panel, root, stage, plan, check) as module:
                result = module.wrapper(stage, plan, panel.worker_config())
                if Path(result) != stage / 'agents' / (plan['game'] + '.py'):
                    raise ValueError('Unexpected wrapper destination')
                _registered(check)
                return result
    except BaseException:
        raise RuntimeError('Development wrapper failed') from None


def run_job(*, panel, source, game_root, stage, plan, check):
    """Run one private job; success still requires outer receipt verification."""
    try:
        _single_python_thread()
        with _quiet_output():
            if type(panel) is not Panel:
                raise ValueError('Expected development panel')
            root, stage = _root_and_stage(game_root, stage)
            plan = panel.validate_plan(plan, job=True)
            with _bound_worker(source, panel, root, stage, plan, check) as module:
                # Source and import-origin authentication is the outer
                # registered caller's responsibility, including this module.
                from learning.task_pair_dev_metrics_v1.consumer import GameConsumer
                with capture.collect_one(module, stage=stage, panel=panel, expected=plan,
                                         check=check, consumer_factory=GameConsumer) as results:
                    module.job(stage, plan['game'])
                if len(results) != 1:
                    raise ValueError('Private consumer did not finish')
                _registered(check)
                capture.store_metrics(module, stage, results[0])
                _registered(check)
    except BaseException:
        raise RuntimeError('Development worker failed') from None
