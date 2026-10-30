"""Bind consumed S01/S04 roles to a new market-opponent comparison.

All returned manifests and role values are private. The caller must first
authenticate the separate registration and its runtime, and retain the exact
private inputs without printing or publishing them. No game or model inference
occurs in this module; the historical receipts are not a third-party timestamp.
"""
import hashlib
from pathlib import Path
from types import ModuleType

from learning.task_control_once_v1 import bindings as control_binding
from learning.task_pair_dev_campaign_v1.manifest import PrivateManifest


def _require(value, message):
    if not value:
        raise ValueError(message)


def verify_mapping(manifest, previous, history_roles):
    """Recheck real role identities, rather than trusting internal aliases."""
    _require(type(manifest) is PrivateManifest and type(previous) is PrivateManifest
             and type(history_roles) is dict and set(history_roles) == {'S01', 'S04'}
             and all(type(value) is int and 0 <= value < 2**32 for value in history_roles.values()),
             'Historical comparison roles differ')
    s01, s04 = history_roles['S01'], history_roles['S04']
    _require(len({s01, s04, *previous.panel.seeds}) == 4
             and manifest.panel.seeds == (s01, s04)
             and manifest.panel.models == previous.panel.models
             and manifest.panel.crop == previous.panel.crop
             and manifest.panel.worker_sha == previous.panel.worker_sha
             and manifest.opponent('market') == previous.opponent('market')
             and manifest.opponent('round2') == previous.opponent('round2'),
             'Historical comparison identity differs')
    return True


def _market_roles(*, source, expected_sha, source_path, historical, status_path, check):
    """Read twelve authenticated prior S01/S04 market receipt coordinates."""
    _require(callable(check) and check() is True and type(source) is bytes
             and 0 < len(source) <= 4 * 1024 * 1024 and type(expected_sha) is str
             and hashlib.sha256(source).hexdigest() == expected_sha,
             'Historical checker identity differs')
    module = ModuleType('_task_pair_historical_market_checker')
    module.__file__ = str(source_path)
    exec(compile(source, str(source_path), 'exec', dont_inherit=True), module.__dict__)
    _require(module.SEED_ALIASES == (('S02', 1), ('S03', 2))
             and module.OPPONENTS == ('round2',)
             and module.ARMS == ('merged', 'unweighted', 'teacher_task'),
             'Historical checker roles differ')
    # This modifies only a fresh, read-only checker instance, not a game guard.
    module.SEED_ALIASES = (('S01', 0), ('S04', 3))
    module.OPPONENTS = ('market',)
    rows = module.checked_rows(Path(historical), Path(status_path))
    expected = {(alias, 'market', side, arm) for alias in ('S01', 'S04')
                for side in (0, 1) for arm in module.ARMS}
    _require(type(rows) is dict and set(rows) == expected, 'Historical market receipts incomplete')
    roles = {}
    for (alias, opponent, side, arm), item in rows.items():
        row = item.get('row') if type(item) is dict else None
        _require(type(row) is dict and row.get('seed_alias') == alias
                 and row.get('opponent') == opponent and type(row.get('side')) is int
                 and row['side'] == side and row.get('policy') == arm
                 and type(row.get('seed')) is int and 0 <= row['seed'] < 2**32,
                 'Historical market role differs')
        if alias in roles:
            _require(roles[alias] == row['seed'], 'Historical market role is inconsistent')
        roles[alias] = row['seed']
    _require(set(roles) == {'S01', 'S04'} and len(set(roles.values())) == 2
             and check() is True, 'Historical market role binding differs')
    return roles


def build(*, modules, bootstrap, identity, reg, old_reg, freeze, game_root, repo,
          status_path, prior_binding, check):
    """Return ``(manifest, previous, history_roles)`` for an authenticated caller.

    ``history_roles`` maps real S01/S04 roles to private values. It, the two
    manifests, and their representations must stay inside the local process.
    """
    _require(callable(check) and check() is True, 'Registration check failed')
    root, repo = Path(game_root), Path(repo)
    control, previous, s01, teacher = control_binding.build(
        modules=modules, bootstrap=bootstrap, identity=identity, reg=reg, old_reg=old_reg,
        freeze=freeze, game_root=root, repo=repo, status_path=status_path,
        prior_binding=prior_binding, check=check)
    _require(control_binding.verify_mapping(control, previous, s01, teacher),
             'Prior single-game identity differs')

    history, manifests = modules['history'], modules['manifest']
    source = identity._read(repo / control_binding.PREFIX)
    roles = _market_roles(source=source, expected_sha=reg['public_sources'][control_binding.PREFIX],
                          source_path=repo / control_binding.PREFIX,
                          historical=root / bootstrap.OLD_PANEL, status_path=status_path, check=check)
    seed_raw = bootstrap._read(root / bootstrap.OLD_PANEL / 'private-seeds.local.json', 16384)
    seed_document = history._json(seed_raw, old_reg['seed_document_sha256'], 16384)
    seeds = seed_document.get('seeds')
    _require(type(seeds) is list and len(seeds) == 4
             and all(type(value) is int and 0 <= value < 2**32 for value in seeds)
             and len(set(seeds)) == 4 and roles == {'S01': seeds[0], 'S04': seeds[3]}
             and s01 == seeds[0] and previous.panel.seeds == (seeds[1], seeds[2]),
             'Historical seed document differs')

    models = {role: dict(path=previous.panel.models[index][0], sha256=previous.panel.models[index][1])
              for index, role in enumerate(('baseline', 'control', 'auxiliary'))}
    crop = dict(path=previous.panel.crop[0], sha256=previous.panel.crop[1])
    opponents = {role: dict(path=previous.opponent(role)[0], sha256=previous.opponent(role)[1])
                 for role in ('market', 'round2')}
    manifest = manifests.build_manifest(seeds={'S02': roles['S01'], 'S03': roles['S04']},
        models=models, crop=crop, opponents=opponents, worker_sha256=previous.panel.worker_sha)
    _require(verify_mapping(manifest, previous, roles) and check() is True,
             'Historical comparison binding changed')
    return manifest, previous, roles
