"""Compare a wrapper with bytes produced by the frozen official generator.

The caller supplies an owned OS-temporary scratch directory and must serialize
other uses of the imported core module.  This checks generator equivalence,
not model, rules, checkpoint, or source identity; the coordinator must verify
those separately and recheck this result around job execution.  Existing path
checks are not a defense against same-user changes between check and use.
"""
import hashlib
import hmac
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import sys
import tempfile

from learning.panel_safety_v1 import seed_inventory as history
from learning.task_teacher_merged_eval_v1 import games as merged

MAX_WRAPPER_BYTES = 64 * 1024
_ALIAS = re.compile(r'S0[1-4]\Z')
_LIMIT_NAME = re.compile(r'[a-z][a-z0-9_]*\Z')


def _relative(value):
    if not isinstance(value, str) or not value or '\\' in value or ':' in value:
        raise ValueError('Invalid relative path')
    path = PurePosixPath(value)
    if path.is_absolute() or path.as_posix() != value or any(part in ('.', '..') for part in path.parts):
        raise ValueError('Invalid relative path')
    return path


def _stage(value):
    artifacts, stage = history._stage_path(merged.experiment.trusted_artifacts(), value)
    if not stage.is_dir() or stage.parent != artifacts:
        raise ValueError('Invalid panel stage')
    return stage


def _scratch(value):
    supplied = Path(value)
    if not supplied.is_absolute() or '..' in supplied.parts:
        raise ValueError('Invalid scratch directory')
    scratch = Path(os.path.abspath(supplied))
    for ancestor in (scratch, *scratch.parents):
        merged.experiment._reject_reparse(ancestor)
    temp_root = Path(tempfile.gettempdir()).resolve()
    if (not scratch.is_dir() or scratch.resolve(strict=True) != scratch
        or scratch == temp_root or not scratch.is_relative_to(temp_root)):
        raise ValueError('Invalid scratch directory')
    return scratch


def _item(item):
    if not isinstance(item, dict):
        raise ValueError('Invalid game identity')
    alias, opponent, side, policy = (item.get(k) for k in ('seed_alias','opponent','side','policy'))
    if (not isinstance(alias, str) or not _ALIAS.fullmatch(alias)
        or opponent not in ('round2','market','teacher')
        or type(side) is not int or side not in (0,1)
        or policy not in ('merged','unweighted','teacher_task')
        or item.get('split') != 'holdout' or item.get('parity_kind') is not None
        or item.get('game') != f'holdout-{alias}-{opponent}-{side}-{policy}'
        or 'task_checkpoint_sha256' not in item or not merged.digest(item.get('bot_sha256'))):
        raise ValueError('Invalid game identity')
    return item['game']


def _configuration(cfg, root):
    if not isinstance(cfg, dict) or not merged.digest(cfg.get('crop_checkpoint_sha256')):
        raise ValueError('Invalid wrapper configuration')
    relative = _relative(cfg.get('crop_checkpoint'))
    target = (root / relative).resolve()
    if target == root or not target.is_relative_to(root):
        raise ValueError('Invalid checkpoint path')
    limits = cfg.get('game_limits')
    if (not isinstance(limits, dict) or not limits
        or any(not isinstance(k, str) or not _LIMIT_NAME.fullmatch(k)
               or type(v) is not int or v < 0 for k, v in limits.items())):
        raise ValueError('Invalid game limits')


def _remove_owned(path, scratch):
    """Remove only the sandbox created by this call, after checking links."""
    if path.parent != scratch or path == scratch:
        raise ValueError('Scratch cleanup refused')
    for ancestor in (scratch, *scratch.parents):
        merged.experiment._reject_reparse(ancestor)
    merged.experiment._reject_reparse(path)
    if not path.is_dir() or path.resolve(strict=True) != path or not path.is_relative_to(scratch):
        raise ValueError('Scratch cleanup refused')
    def walk_error(_):
        raise ValueError('Scratch cleanup refused')
    for folder, dirs, names in os.walk(path, followlinks=False, onerror=walk_error):
        for name in (*dirs, *names):
            merged.experiment._reject_reparse(Path(folder)/name)
    shutil.rmtree(path)


def _verify(stage_value, item, cfg, scratch_value):
    if not __debug__ or sys.flags.optimize != 0:
        raise ValueError('Optimized Python is not permitted')
    stage, scratch = _stage(stage_value), _scratch(scratch_value)
    root = merged.experiment.ROOT.resolve()
    if (stage.parent != root/'artifacts' or scratch == root or scratch.is_relative_to(root)
        or root.is_relative_to(scratch)):
        raise ValueError('Invalid panel stage')
    key = _item(item)
    _configuration(cfg, root)
    task, task_sha = merged.task_identity(stage, item, cfg)
    if task_sha != item.get('task_checkpoint_sha256'):
        raise ValueError('Task identity changed')
    task_rel = None if task is None else task.relative_to(root)
    actual = history._read_regular(stage/'agents'/(key+'.py'), stage, MAX_WRAPPER_BYTES)
    actual_sha = hashlib.sha256(actual).hexdigest()
    if not hmac.compare_digest(actual_sha, item['bot_sha256']):
        raise ValueError('Wrapper hash mismatch')

    sandbox = Path(tempfile.mkdtemp(prefix='canonical-wrapper-', dir=str(scratch)))
    try:
        if sandbox.parent != scratch or sandbox.resolve(strict=True) != sandbox:
            raise ValueError('Invalid scratch sandbox')
        fake_root = sandbox/'project'
        fake_root.mkdir()
        fake_stage = fake_root/stage.relative_to(root)
        fake_task = None if task_rel is None else fake_root/task_rel
        with merged._LOCK:
            core = merged.core
            original_root, original_identity = core.ROOT, core.task_identity
            try:
                core.ROOT = fake_root
                core.task_identity = lambda _out, _plan, _cfg: (fake_task, task_sha)
                generated = core.wrapper(fake_stage, item, cfg)
            finally:
                core.ROOT, core.task_identity = original_root, original_identity
        if Path(generated) != fake_stage/'agents'/(key+'.py'):
            raise ValueError('Unexpected generator output')
        canonical = history._read_regular(generated, sandbox, MAX_WRAPPER_BYTES)
        if not hmac.compare_digest(actual, canonical):
            raise ValueError('Noncanonical wrapper')
        return actual_sha
    finally:
        _remove_owned(sandbox, scratch)


def verify_wrapper(stage, item, cfg, scratch_root):
    """Return the matched SHA, or fail without displaying paths or plan data."""
    try:
        return _verify(stage, item, cfg, scratch_root)
    except KeyboardInterrupt:
        # Preserve an explicit interruption without propagating callback text.
        raise KeyboardInterrupt() from None
    except BaseException:
        raise ValueError('Canonical wrapper verification failed') from None
