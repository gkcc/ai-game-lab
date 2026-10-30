"""Bind this development panel to already-consumed, locally recorded inputs.

Call only after an independently checked public registration.  The historical
reader is an isolated snapshot of the existing prefix diagnostic; no seed
inventory scan, game, model inference, or historical replay decoding occurs.
Returned identities and all underlying receipts are private.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import stat
from types import ModuleType

from .manifest import _relative_path, build_manifest
from learning.task_pair_dev_games_v1.panel import ENGINE_COMMIT, ENGINE_SHA


MODEL_LIMIT = 64 * 1024 * 1024
METADATA_LIMIT = 4 * 1024 * 1024
_ROLES = ('S02', 'S03')
_OPPONENTS = ('market', 'round2')
_ARMS = ('baseline', 'control', 'auxiliary')
_HISTORIC_ARMS = ('merged', 'unweighted', 'teacher_task')


def _require(value, message):
    if not value:
        raise ValueError(message)


def _checked(check):
    _require(callable(check) and check() is True, 'Registration check failed')


def _digest(value):
    _require(type(value) is str and len(value) == 64
             and all(c in '0123456789abcdef' for c in value), 'Invalid digest')
    return value


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        _require(key not in result, 'Duplicate metadata key')
        result[key] = value
    return result


def _invalid_constant(_value):
    raise ValueError('Nonfinite metadata value')


def _json(raw, expected, maximum=METADATA_LIMIT):
    _require(type(raw) is bytes and 0 < len(raw) <= maximum, 'Invalid metadata size')
    _require(hashlib.sha256(raw).hexdigest() == _digest(expected), 'Metadata identity differs')
    value = json.loads(raw.decode('utf-8-sig'), object_pairs_hook=_pairs,
                       parse_constant=_invalid_constant)
    _require(type(value) is dict, 'Metadata object required')
    return value


def _safe_relative_file(root, relative, suffix, *, artifacts=False):
    _relative_path(relative, suffix, artifacts=artifacts)
    root = Path(root)
    _require(root.is_absolute() and root.resolve(strict=True) == root, 'Invalid project root')
    current = root
    for part in relative.split('/'):
        current = current / part
        info = current.lstat()
        _require(not stat.S_ISLNK(info.st_mode)
                 and not (getattr(info, 'st_file_attributes', 0) & 0x400), 'Linked input refused')
    _require(current.is_file() and current.resolve(strict=True) == current
             and current.is_relative_to(root), 'Invalid project input')
    return current


def _file_digest(path, maximum):
    digest, total = hashlib.sha256(), 0
    with Path(path).open('rb') as stream:
        while True:
            block = stream.read(65536)
            if not block:
                break
            total += len(block)
            _require(total <= maximum, 'Input size limit exceeded')
            digest.update(block)
    _require(total > 0, 'Empty input')
    return digest.hexdigest()


def historical_roles(*, source, source_sha256, source_path, panel_path, status_path, check):
    """Recheck both consumed roles against the accepted prior panel's receipts.

    The source must be the registered snapshot of task_prefix_diagnostic_v2's
    existing read-only checker.  An isolated module selects both prior frozen
    opponents, leaving the original diagnostic and its holdout guards intact.
    Its checker verifies the prior registration, seed document, public alias
    ledger and each selected job's acceptance/supervision/row/audit receipts.
    It stats old replay/trace files but does not decode them.
    """
    _checked(check)
    _require(type(source) is bytes and 0 < len(source) <= METADATA_LIMIT
             and hashlib.sha256(source).hexdigest() == _digest(source_sha256),
             'Historical checker snapshot differs')
    source_path = Path(source_path)
    _require(source_path.is_absolute(), 'Historical checker path must be absolute')
    module = ModuleType('_task_pair_dev_historical_checker')
    module.__file__ = str(source_path)
    exec(compile(source, str(source_path), 'exec', dont_inherit=True), module.__dict__)
    _require(module.SEED_ALIASES == (('S02', 1), ('S03', 2))
             and module.OPPONENTS == ('round2',)
             and module.ARMS == _HISTORIC_ARMS, 'Historical selector changed')
    # This changes only a freshly created read-only selector, not a game guard.
    module.OPPONENTS = ('round2', 'market')
    selected = module.checked_rows(Path(panel_path), Path(status_path))
    expected = {(alias, opponent, side, arm) for alias in _ROLES
                for opponent in _OPPONENTS for side in (0, 1) for arm in _HISTORIC_ARMS}
    _require(type(selected) is dict and set(selected) == expected, 'Historical roles incomplete')
    seeds = {}
    for (alias, opponent, side, arm), item in selected.items():
        row = item.get('row') if type(item) is dict else None
        _require(type(row) is dict and row.get('seed_alias') == alias
                 and row.get('opponent') == opponent and type(row.get('side')) is int
                 and row['side'] == side and row.get('policy') == arm
                 and type(row.get('seed')) is int and 0 <= row['seed'] < 2**32,
                 'Historical role row differs')
        if alias in seeds:
            _require(seeds[alias] == row['seed'], 'Historical role is inconsistent')
        seeds[alias] = row['seed']
    _require(len(set(seeds.values())) == len(_ROLES), 'Historical roles overlap')
    _checked(check)
    return seeds


def bind_prior(*, seeds, supervisor_raw, supervisor_sha256, selected_export_raw,
               selected_export_sha256, historical_registration, historical_freeze, historical_seed_raw,
               game_root, historical_panel_relative, model_paths, check):
    """Bind three previous training outputs and old crop/opponents to a Panel.

    The caller authenticates the public registration, its historical documents,
    prior host completion and the *earlier fixed* private supervisor digest.
    The old seed-document snapshot is checked against its registered digest and
    indices to catch an accidental role swap. This verifies model/source bytes
    without loading model arrays or starting inference. It does not prove the old
    supervisor's own reports or give a third-party timestamp.
    """
    _checked(check)
    cfg, freeze = historical_registration, historical_freeze
    _require(type(cfg) is dict and cfg.get('engine_commit') == ENGINE_COMMIT
             and cfg.get('engine_sha256') == ENGINE_SHA and type(freeze) is dict,
             'Historical engine identity differs')
    # Recheck the two ordered roles against the old committed seed document.
    # This catches an accidental swap between the old checker and the manifest.
    seed_doc = _json(historical_seed_raw, cfg.get('seed_document_sha256'), 16384)
    values = seed_doc.get('seeds')
    _require(type(values) is list and len(values) == 4
             and all(type(seed) is int and 0 <= seed < 2**32 for seed in values)
             and len(set(values)) == 4
             and type(seeds) is dict and all(type(seeds.get(role)) is int for role in _ROLES)
             and seeds == {'S02': values[1], 'S03': values[2]},
             'Historical seed roles differ')
    previous = _json(supervisor_raw, supervisor_sha256, 65536)
    export = _json(selected_export_raw, selected_export_sha256, 65536)
    _require(previous.get('schema') == 'task-pair-wsl-supervisor-private-result-v1'
             and previous.get('technical_execution_passed') is True
             and previous.get('model_files_reloaded') is True
             and previous.get('weights_distinct') is True
             and previous.get('owned_session_clean') is True
             and previous.get('new_games') is False,
             'Prior training receipt not accepted')
    identities = previous.get('private_model_sha256')
    _require(type(identities) is dict and set(identities) == set(_ARMS)
             and len({_digest(identities[role]) for role in _ARMS}) == len(_ARMS),
             'Prior model identities differ')
    _require(export.get('checkpoint_sha256') == identities['baseline'],
             'Baseline differs from historical selected export')
    _require(type(model_paths) is dict and set(model_paths) == set(_ARMS), 'Invalid model roles')
    root = Path(game_root)
    models = {}
    for role in _ARMS:
        path = _safe_relative_file(root, model_paths[role], '.npz', artifacts=True)
        _require(_file_digest(path, MODEL_LIMIT) == identities[role], 'Prior model bytes differ')
        models[role] = dict(path=model_paths[role], sha256=identities[role])

    crop = dict(path=cfg.get('crop_checkpoint'), sha256=_digest(cfg.get('crop_checkpoint_sha256')))
    crop_path = _safe_relative_file(root, crop['path'], '.npz', artifacts=True)
    _require(_file_digest(crop_path, MODEL_LIMIT) == crop['sha256'], 'Crop model bytes differ')
    sources = cfg.get('source_sha256')
    _require(type(sources) is dict, 'Historical opponent identities missing')
    # Only the already accepted panel's own copied opponent sources are used.
    _relative_path(historical_panel_relative + '/placeholder.py', '.py', artifacts=True)
    opponents = {}
    for role in _OPPONENTS:
        relative = historical_panel_relative + '/sources/' + role + '.py'
        expected = _digest(sources.get(role + '.py'))
        path = _safe_relative_file(root, relative, '.py', artifacts=True)
        _require(_file_digest(path, METADATA_LIMIT) == expected, 'Opponent bytes differ')
        opponents[role] = dict(path=relative, sha256=expected)
    runtime = freeze.get('runtime_files')
    _require(type(runtime) is dict, 'Historical runtime identities missing')
    worker_rel = 'learning/task_anchor_once_v1/games.py'
    worker_sha = _digest(runtime.get(worker_rel))
    worker_path = _safe_relative_file(root, worker_rel, '.py')
    _require(_file_digest(worker_path, METADATA_LIMIT) == worker_sha, 'Worker bytes differ')
    _checked(check)
    return build_manifest(seeds=seeds, models=models, crop=crop,
                          opponents=opponents, worker_sha256=worker_sha)
