"""Bind the one-game structural aliases to authenticated historical inputs.

The returned manifest contains private values and must never be printed or
published. This is local identity verification, not an independent timestamp.
"""
import hashlib
import json
import math
from pathlib import Path
from types import ModuleType

from learning.task_pair_dev_campaign_v1.manifest import PrivateManifest


TEACHER_SHA = '0844f0152e742aba684108b5914964fb85082b3c0881d0da4d41649907a9b6ff'
TEACHER_REL = 'bots/teacher_fertilizer_terminal_v2.py'
PREFIX = 'kaggriculture/learning/task_prefix_diagnostic_v2/runner.py'


def _require(value, message):
    if not value:
        raise ValueError(message)


def context_identity(*values):
    """Digest exact authenticated in-memory metadata, never serialize it publicly."""
    def valid(value):
        if value is None or type(value) in (str, int, bool):
            return True
        if type(value) is float:
            return math.isfinite(value)
        if type(value) is list:
            return all(valid(item) for item in value)
        if type(value) is dict:
            return all(type(key) is str and valid(item) for key, item in value.items())
        return False
    _require(len(values) == 4 and all(type(value) is dict and valid(value) for value in values),
             'Invalid authenticated metadata')
    raw = json.dumps(values, ensure_ascii=True, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    return hashlib.sha256(raw).digest()


def verify_mapping(manifest, previous, seed_s01, teacher_relative):
    """Check actual roles, not just the old Panel's self-consistent labels."""
    _require(type(manifest) is PrivateManifest and type(previous) is PrivateManifest
             and type(seed_s01) is int and 0 <= seed_s01 < 2**32
             and seed_s01 not in previous.panel.seeds, 'Historical seed binding differs')
    _require(manifest.panel.seeds == (seed_s01, previous.panel.seeds[0])
             and manifest.panel.models == previous.panel.models
             and manifest.panel.crop == previous.panel.crop
             and manifest.panel.worker_sha == previous.panel.worker_sha
             and manifest.opponent('market') == (teacher_relative, TEACHER_SHA)
             and manifest.opponent('round2') == previous.opponent('round2'),
             'Single-game role mapping differs')
    return True


def _s01_rows(*, source, expected_sha, source_path, historical, status_path, check):
    _require(callable(check) and check() is True and type(source) is bytes and source
             and hashlib.sha256(source).hexdigest() == expected_sha, 'Historical checker differs')
    module = ModuleType('_task_control_once_historical_checker')
    module.__file__ = str(source_path)
    exec(compile(source, str(source_path), 'exec', dont_inherit=True), module.__dict__)
    _require(module.SEED_ALIASES == (('S02', 1), ('S03', 2))
             and module.OPPONENTS == ('round2',)
             and module.ARMS == ('merged', 'unweighted', 'teacher_task'), 'Historical selector changed')
    # Change only this isolated, read-only checker; no original game guard is modified.
    module.SEED_ALIASES = (('S01', 0),)
    module.OPPONENTS = ('teacher',)
    rows = module.checked_rows(Path(historical), Path(status_path))
    expected = {('S01', 'teacher', side, arm) for side in (0, 1) for arm in module.ARMS}
    _require(type(rows) is dict and set(rows) == expected, 'Historical S01 evidence incomplete')
    seeds = set()
    for (alias, opponent, side, arm), item in rows.items():
        row = item.get('row') if type(item) is dict else None
        _require(type(row) is dict and row.get('seed_alias') == alias and row.get('opponent') == opponent
                 and type(row.get('side')) is int and row['side'] == side and row.get('policy') == arm
                 and type(row.get('seed')) is int and 0 <= row['seed'] < 2**32,
                 'Historical S01 row differs')
        seeds.add(row['seed'])
    _require(len(seeds) == 1 and check() is True, 'Historical S01 role inconsistent')
    return next(iter(seeds))


def build(*, modules, bootstrap, identity, reg, old_reg, freeze, game_root, repo,
          status_path, prior_binding, check):
    """Rebind the old verified models and sources to one historical S01 game."""
    _require(callable(check) and check() is True, 'Registration check failed')
    root, repo = Path(game_root), Path(repo)
    history, manifests = modules['history'], modules['manifest']
    historical = root / bootstrap.OLD_PANEL
    source = identity._read(repo / PREFIX)
    seed_s01 = _s01_rows(source=source, expected_sha=reg['public_sources'][PREFIX],
        source_path=repo / PREFIX, historical=historical, status_path=status_path, check=check)
    old_seeds = history.historical_roles(source=source, source_sha256=reg['public_sources'][PREFIX],
        source_path=repo / PREFIX, panel_path=historical, status_path=status_path, check=check)
    seed_raw = bootstrap._read(historical / 'private-seeds.local.json', 16384)
    seed_document = history._json(seed_raw, old_reg['seed_document_sha256'], 16384)
    seeds = seed_document.get('seeds')
    _require(type(seeds) is list and len(seeds) == 4 and all(type(value) is int for value in seeds)
             and seed_s01 == seeds[0] and old_seeds == {'S02': seeds[1], 'S03': seeds[2]},
             'Historical seed mapping differs')
    previous = history.bind_prior(seeds=old_seeds,
        supervisor_raw=bootstrap._read(root / bootstrap.SUPERVISOR, 65536),
        supervisor_sha256=prior_binding['prior_supervisor_sha256'],
        selected_export_raw=bootstrap._read(repo / bootstrap.EXPORT, 65536),
        selected_export_sha256=reg['public_sources'][bootstrap.EXPORT], historical_registration=old_reg,
        historical_freeze=freeze, historical_seed_raw=seed_raw, game_root=root,
        historical_panel_relative=bootstrap.OLD_PANEL,
        model_paths={role: bootstrap.MODEL_DIR + '/' + role + '.npz'
                     for role in ('baseline', 'control', 'auxiliary')}, check=check)
    teacher_relative = bootstrap.OLD_PANEL + '/sources/teacher.py'
    teacher = history._safe_relative_file(root, teacher_relative, '.py', artifacts=True)
    local_teacher = history._safe_relative_file(root, TEACHER_REL, '.py')
    public_teacher = repo / 'kaggriculture' / TEACHER_REL
    _require(old_reg['source_sha256']['teacher.py'] == TEACHER_SHA
             and reg['public_sources']['kaggriculture/' + TEACHER_REL] == TEACHER_SHA
             and history._file_digest(teacher, history.METADATA_LIMIT) == TEACHER_SHA
             and history._file_digest(local_teacher, history.METADATA_LIMIT) == TEACHER_SHA
             and hashlib.sha256(identity._read(public_teacher)).hexdigest() == TEACHER_SHA,
             'Frozen teacher identity differs')
    models = {role: dict(path=previous.panel.models[i][0], sha256=previous.panel.models[i][1])
              for i, role in enumerate(('baseline', 'control', 'auxiliary'))}
    crop = dict(path=previous.panel.crop[0], sha256=previous.panel.crop[1])
    round2 = previous.opponent('round2')
    manifest = manifests.build_manifest(seeds={'S02': seed_s01, 'S03': old_seeds['S02']}, models=models, crop=crop,
        opponents={'market': dict(path=teacher_relative, sha256=TEACHER_SHA),
                   'round2': dict(path=round2[0], sha256=round2[1])}, worker_sha256=previous.panel.worker_sha)
    verify_mapping(manifest, previous, seed_s01, teacher_relative)
    _require(check() is True, 'Registration changed during identity binding')
    return manifest, previous, seed_s01, teacher_relative
