"""Pre-import verification of the fixed local development-game registration.

This is deliberately standard-library only.  It authenticates committed and
local bytes but is not an OS filesystem lock or a defence against same-user
concurrent replacement.  No model, seed, replay, trace or private status is
read by this module.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys


REGISTRATION = 'review/evidence/task-pair-dev-games-v1-20261030/registration.json'
HISTORICAL_REGISTRATION = 'review/evidence/task-teacher-merged-eval-v3-formal-20261030/registration.json'
HISTORICAL_FREEZE = 'review/evidence/task-teacher-merged-eval-v3-formal-20261030/freeze.json'
ENGINE_COMMIT = 'f994f2050d58d84f649329c1e35ce09ada7007c5'
ENGINE_SHA = 'bc8a54879ef02c7ea64b8b333d6a976f0ea65c4949149d01f463f23bccee653e'
RESULT_COMMIT = '5a66216f99a8e467818978c96e707f6045834709'
HISTORICAL_RESULT = '3a3e3172b0403a171a38f6e78e37d07f387be2cb'
_HEX = re.compile(r'[0-9a-f]{64}\Z')
_COMMIT = re.compile(r'[0-9a-f]{40}\Z')
_MAX_SOURCE = 8 * 1024 * 1024
_REQUIRED = frozenset((
    HISTORICAL_REGISTRATION, HISTORICAL_FREEZE,
    'review/evidence/task-bc-v1-20261023-run1/model/selected-export.json',
    'kaggriculture/learning/task_prefix_diagnostic_v2/runner.py',
    'kaggriculture/learning/task_single_intervention_v1/trusted_imports.py',
    'kaggriculture/learning/task_pair_dev_games_v1/__init__.py',
    'kaggriculture/learning/task_pair_dev_games_v1/panel.py',
    'kaggriculture/learning/task_pair_dev_games_v1/worker.py',
    'kaggriculture/learning/task_pair_dev_games_v1/capture.py',
    'kaggriculture/learning/task_pair_dev_metrics_v1/__init__.py',
    'kaggriculture/learning/task_pair_dev_metrics_v1/consumer.py',
    'kaggriculture/learning/task_pair_dev_metrics_v1/panel.py',
    'kaggriculture/learning/task_pair_dev_metrics_v1/replay_pair.py',
    'kaggriculture/learning/task_pair_dev_campaign_v1/identity.py',
    'kaggriculture/learning/task_pair_dev_campaign_v1/bootstrap.py',
    'kaggriculture/learning/task_pair_dev_campaign_v1/manifest.py',
    'kaggriculture/learning/task_pair_dev_campaign_v1/history.py',
    'kaggriculture/learning/task_pair_dev_campaign_v1/child.py',
    'kaggriculture/learning/task_pair_dev_campaign_v1/campaign.py',
    'kaggriculture/learning/task_pair_dev_campaign_v1/acceptance.py',
    'kaggriculture/learning/task_pair_dev_campaign_v1/driver.py',
    'kaggriculture/learning/task_pair_dev_campaign_v1/evidence.py',
    'kaggriculture/learning/task_pair_dev_campaign_v1/projection.py',
    'review/evidence/task-pair-wsl-training-v1-result-20261030/RESULTS.zh-CN.md',
    'review/evidence/task-pair-wsl-training-v1-result-20261030/certificate.json',
    'review/evidence/task-pair-wsl-training-v1-20261030/registration.json',
    'review/evidence/task-teacher-merged-eval-v3-formal-result-20261030/alias-ledger.json',
    'tools/panel_v3_alias_ledger.py',
))


def _require(value, message):
    if not value:
        raise ValueError(message)


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        _require(key not in result, 'Duplicate document key')
        result[key] = value
    return result


def _invalid_constant(_value):
    raise ValueError('Nonfinite document value')


def _json(raw):
    _require(type(raw) is bytes and 0 < len(raw) <= _MAX_SOURCE, 'Invalid document size')
    result = json.loads(raw.decode('utf-8-sig'), object_pairs_hook=_pairs,
                        parse_constant=_invalid_constant)
    _require(type(result) is dict, 'Document must be an object')
    return result


def _relative(value):
    _require(type(value) is str and value and '\\' not in value and ':' not in value
             and not any(ord(c) < 32 for c in value), 'Invalid source path')
    path = PurePosixPath(value)
    _require(not path.is_absolute() and path.as_posix() == value
             and all(p not in ('.', '..') for p in path.parts), 'Invalid source path')
    return value


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _read(path, *, allow_empty=False):
    with Path(path).open('rb') as stream:
        raw = stream.read(_MAX_SOURCE + 1)
    _require(len(raw) <= _MAX_SOURCE and (allow_empty or bool(raw)), 'Source size limit')
    return raw


def _git(repo, *args):
    proc = subprocess.run(['git', '-C', str(repo), *args], stdin=subprocess.DEVNULL,
                          stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                          check=False, timeout=30)
    _require(proc.returncode == 0 and len(proc.stdout) <= _MAX_SOURCE, 'Git verification failed')
    return proc.stdout


def _file(root, relative):
    relative = _relative(relative)
    path = root.joinpath(*relative.split('/'))
    _require(path.resolve(strict=True) == path and path.is_file() and path.is_relative_to(root),
             'Source file location differs')
    return path


def _check_digest(root, relative, expected):
    _require(type(expected) is str and _HEX.fullmatch(expected) is not None, 'Invalid source digest')
    _require(_sha(_read(_file(root, relative), allow_empty=True)) == expected, 'Source bytes changed')


def _panel(reg):
    expected = dict(seed_roles=['S02', 'S03'], opponents=['market', 'round2'],
                    policies=['baseline', 'control', 'auxiliary'], sides=[0, 1],
                    games=24, scope='consumed_development_no_promotion')
    panel = reg.get('panel')
    _require(type(panel) is dict and panel == expected and type(panel.get('games')) is int
             and all(type(side) is int for side in panel.get('sides', ())), 'Panel registration differs')
    _require(reg.get('schema') == 'task-pair-dev-games-prereg-v1'
             and reg.get('previous_training_result_commit') == RESULT_COMMIT
             and reg.get('historical_panel_result_commit') == HISTORICAL_RESULT
             and reg.get('engine_commit') == ENGINE_COMMIT
             and reg.get('engine_sha256') == ENGINE_SHA,
             'Registration identity differs')
    evaluation = reg.get('evaluation')
    expected_eval = dict(comparisons=['control_minus_baseline', 'auxiliary_minus_control'],
                         cluster='seed', min_per_seed=5, min_total=20,
                         significance='not_assessed', promotion=False, retries=False,
                         official_games=24, official_replays=24, training_updates=0,
                         new_seed_generation=False)
    _require(type(evaluation) is dict and evaluation == expected_eval
             and all(type(evaluation.get(k)) is int for k in
                     ('min_per_seed', 'min_total', 'official_games', 'official_replays', 'training_updates'))
             and all(evaluation.get(k) is False for k in ('promotion', 'retries', 'new_seed_generation')),
             'Evaluation registration differs')


def _historical_lineage(repo, registration_commit):
    _git(repo, 'merge-base', '--is-ancestor', HISTORICAL_RESULT, registration_commit)
    _git(repo, 'merge-base', '--is-ancestor', RESULT_COMMIT, registration_commit)
    historical_blobs = (HISTORICAL_REGISTRATION, HISTORICAL_FREEZE,
        'review/evidence/task-teacher-merged-eval-v3-formal-result-20261030/alias-ledger.json',
        'tools/panel_v3_alias_ledger.py')
    training_blobs = ('review/evidence/task-pair-wsl-training-v1-result-20261030/RESULTS.zh-CN.md',
        'review/evidence/task-pair-wsl-training-v1-result-20261030/certificate.json',
        'review/evidence/task-pair-wsl-training-v1-20261030/registration.json',
        'review/evidence/task-bc-v1-20261023-run1/model/selected-export.json')
    for commit, paths in ((HISTORICAL_RESULT, historical_blobs), (RESULT_COMMIT, training_blobs)):
        for rel in paths:
            _require(_git(repo, 'show', commit + ':' + rel) == _read(_file(repo, rel)),
                     'Historical evidence bytes changed')


def verify_public(*, repo, game_root, registration_commit):
    """Verify clean committed registration, frozen sources and official engine.

    Registration and source maps are returned for the caller.  This does not
    authenticate a private manifest, check process cleanup, or launch games.
    """
    _require(sys.flags.isolated and sys.flags.dont_write_bytecode and __debug__,
             'Isolated non-optimized Python without bytecode writes required')
    _require(os.name == 'nt', 'This development panel requires Windows')
    repo, game_root = Path(repo), Path(game_root)
    _require(repo.is_absolute() and repo.resolve(strict=True) == repo
             and game_root.is_absolute() and game_root.resolve(strict=True) == game_root,
             'Unexpected project root')
    _require(type(registration_commit) is str and _COMMIT.fullmatch(registration_commit),
             'Invalid registration commit')
    _require(_git(repo, 'rev-parse', 'HEAD').decode().strip() == registration_commit,
             'Registration is not current HEAD')
    _require(not _git(repo, 'status', '--porcelain', '--untracked-files=all'),
             'Public worktree is not clean')
    raw = _read(_file(repo, REGISTRATION))
    _require(_git(repo, 'show', registration_commit + ':' + REGISTRATION) == raw,
             'Registration differs from committed bytes')
    reg = _json(raw)
    _panel(reg)
    source = reg.get('public_sources')
    _require(type(source) is dict and _REQUIRED.issubset(source), 'Required sources are missing')
    for rel, digest in source.items():
        _check_digest(repo, rel, digest)
        _require(_git(repo, 'show', registration_commit + ':' + rel) == _read(_file(repo, rel), allow_empty=True),
                 'Registered source differs from committed bytes')
    _historical_lineage(repo, registration_commit)

    old_reg = _json(_read(_file(repo, HISTORICAL_REGISTRATION)))
    freeze = _json(_read(_file(repo, HISTORICAL_FREEZE)))
    _require(old_reg.get('engine_commit') == ENGINE_COMMIT
             and old_reg.get('engine_sha256') == ENGINE_SHA, 'Historical official rules differ')
    local, public = freeze.get('runtime_files'), freeze.get('public_runtime_sha256')
    supplemental = freeze.get('supplemental_runtime_sha256')
    _require(type(local) is dict and type(public) is dict and len(local) == len(public) == 297
             and set(local) == set(public) and type(supplemental) is dict and len(supplemental) == 23,
             'Historical runtime map differs')
    public_game = _file(repo, 'kaggriculture/run_match.py').parent
    for rel in sorted(local):
        _check_digest(game_root, rel, local[rel])
        _check_digest(public_game, rel, public[rel])
    for rel, pair in supplemental.items():
        _require(type(pair) is dict and set(pair) == {'local_sha256', 'public_sha256'},
                 'Historical supplemental identity differs')
        _check_digest(game_root, rel, pair['local_sha256'])
        _check_digest(public_game, rel, pair['public_sha256'])

    environment = old_reg.get('environment')
    _require(type(environment) is dict and type(environment.get('versions')) is dict,
             'Historical environment missing')
    _require(_sha(_read(Path(sys.executable))) == environment.get('interpreter_sha256'),
             'Interpreter identity differs')
    versions = environment['versions']
    _require(versions.get('python') == '.'.join(map(str, sys.version_info[:3]))
             and importlib.metadata.version('numpy') == versions.get('numpy')
             and importlib.metadata.version('kaggle-environments') == versions.get('kaggle_environments'),
             'Installed runtime versions differ')
    spec = importlib.util.find_spec('kaggle_environments')
    _require(spec is not None and spec.submodule_search_locations is not None
             and len(spec.submodule_search_locations) == 1, 'Official package location differs')
    package = Path(next(iter(spec.submodule_search_locations))).resolve(strict=True)
    rules = _read(package / 'envs/kaggriculture/kaggriculture.py')
    _require(_sha(rules) == ENGINE_SHA and _sha(_read(package/'agent.py')) == old_reg['official_loader_sha256']
             and _sha(_read(package/'core.py')) == old_reg['official_core_sha256'],
             'Installed official engine bytes differ')
    upstream = game_root / 'upstream'
    _require(_git(upstream, 'rev-parse', 'HEAD').decode().strip() == ENGINE_COMMIT
             and _git(upstream, 'show', ENGINE_COMMIT + ':kaggle_environments/envs/kaggriculture/kaggriculture.py') == rules,
             'Installed rules differ from official checkout')
    return reg, old_reg, freeze
