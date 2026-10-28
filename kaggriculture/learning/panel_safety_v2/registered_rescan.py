"""Rescan a registered panel's seed history without exposing selected seeds.

The v1 registration, freeze, decision and review retain their existing fields.
One additional committed ``safety-registration.json`` binds an exclusively
created stage marker and those four exact evidence files.  This is a
cooperative integrity check, not protection against a same-user process
replacing files between checks and use.  It does not launch games.
"""
import hashlib
import json
import math
import os
from decimal import Decimal, InvalidOperation
from pathlib import Path, PurePosixPath
import re
import subprocess

from learning.panel_safety_v1 import seed_inventory as history
from learning.task_teacher_merged_eval_v1 import experiment as old
from learning.task_teacher_merged_eval_v1.evaluation import plans

PUBLIC_FILES = ('root-decision.json', 'registration.json', 'freeze.json', 'pre-review.json')
SAFETY = 'safety-registration.json'
MARKER = 'ownership-marker.json'
PRIVATE = 'private-seeds.local.json'
SNAPSHOT = 'seed-inventory.local.json'
EXECUTION = 'pre-execution-commit.json'
SCHEMA = 'registered-seed-rescan-v1'
_NAME = re.compile(r'[a-z0-9][a-z0-9-]{0,95}\Z')
ROOT_RESULTS = frozenset(('games-started.json','games-failure.local.json','games-result.json',
    'game-ledger.json','game-audit-ledger.json','independent-runner-started.json',
    'independent-attempt.local.json','independent-runner-failure.json','independent-runner-result.json',
    'independent-failure.local.json','independent-replay-verification.json','independent-result-review.json',
    'evaluation-started.json','evaluation-failure.json','evaluation-result.json'))
_SEED_ROLES = frozenset(('train','training','test','holdout','validation','eval','evaluation',
                         'development','dev','reserved','used','unused','selected','excluded','values'))
_SEED_ALIAS = re.compile(r'(?:s[0-9]+|alias(?:[_-]?[a-z0-9]+)?)\Z')
_NUMERIC_TEXT = re.compile(r'[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?\Z')
_METADATA_TOKENS = frozenset(('count','counts','size','length','bytes','hash','hashes','sha','sha256',
                              'digest','checksum','score','scores','profit','profits','reward','rewards',
                              'index','indices','min','max','range','probability','mean','median','variance',
                              'limit','limits','budget','threshold','timeout','seconds'))


def _git(*args):
    try:
        return old.git(*args)
    except subprocess.CalledProcessError:
        raise ValueError('Registered panel Git identity failed') from None


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _commit(value):
    return isinstance(value, str) and len(value) == 40 and all(c in '0123456789abcdef' for c in value)


def _digest(value):
    return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def _read(path, boundary, limit=history.MAX_FILE):
    return history._read_regular(path, boundary, limit)


def _json(raw, limit=history.MAX_FILE):
    return history._json(raw, limit)


def _name(value):
    if not isinstance(value, str) or not _NAME.fullmatch(value):
        raise ValueError('Invalid registered stage name')
    return value


def _history_seed_values(value, active=False, aliases=None):
    """Expand recognized seed roles, preserving old numeric-key metadata semantics.

    Named seed roles activate seed values and, for explicit seed maps, their
    scalar aliases. Count/hash metadata, booleans and unrelated values are not
    seeds. Non-finite or fractional numbers in an active seed slot fail closed.
    """
    if aliases is None:
        aliases = active
    found = set()
    if isinstance(value, dict):
        for key, item in value.items():
            label = re.sub(r'(?<=[a-z0-9])(?=[A-Z])', '_', str(key)).casefold()
            if (set(re.split(r'[_-]+',label)) & _METADATA_TOKENS
                or label in ('num_seeds','n_seeds','number_of_seeds')):
                continue
            if active and label.isdecimal():
                index = int(label)
                if index < 2**32:
                    found.add(index)
                found.update(_history_seed_values(item, True, True))
                continue
            if active and (_SEED_ALIAS.fullmatch(label) or aliases):
                found.update(_history_seed_values(item, True, True))
                continue
            selected = 'seed' in label or label == 'excluded_uint32' or (active and label in _SEED_ROLES)
            seed_map = (label in ('seed','seeds','excluded_uint32','seed_roles','seed_values','seed_list','seed_aliases',
                                  'seed_map','seed_mapping','seeds_by_alias','seed_by_alias')
                        or label.endswith(('_seed','_seeds')) or (active and label in _SEED_ROLES))
            found.update(_history_seed_values(item, selected, selected and seed_map))
    elif isinstance(value, list):
        for item in value:
            found.update(_history_seed_values(item, active, aliases))
    elif active and type(value) is int and 0 <= value < 2**32:
        found.add(value)
    elif active and type(value) is float:
        if not math.isfinite(value) or not value.is_integer():
            raise ValueError('Invalid numeric value in historical seed metadata')
        if 0 <= value < 2**32:
            found.add(int(value))
    elif active and isinstance(value, str):
        decimal_text = value.isdecimal()
        numeric_text = decimal_text or _NUMERIC_TEXT.fullmatch(value) is not None
        if numeric_text:
            if len(value) > 128:
                raise ValueError('Oversized numeric value in historical seed metadata')
            if decimal_text:
                number = int(value)
                if number < 2**32:
                    found.add(number)
            else:
                try:
                    number = Decimal(value)
                    if not number.is_finite():
                        raise ValueError('Invalid numeric value in historical seed metadata')
                    if 0 <= number < 2**32:
                        if number != number.to_integral_value():
                            raise ValueError('Invalid numeric value in historical seed metadata')
                        found.add(int(number))
                except InvalidOperation:
                    raise ValueError('Invalid numeric value in historical seed metadata') from None
    return found


def _expanded_history(artifacts, public, stage, tracked, own_public):
    """Scan recognized local metadata, including ignored public/private files.

    Game, replay and result directories are not entered. The five verified
    own-stage public files alone may be skipped; an extra recognized file in
    that directory fails closed even if Git ignores it.
    """
    artifacts, stage = history._stage_path(artifacts, stage)
    public = Path(os.path.abspath(public))
    for part in (public, *public.parents):
        old._reject_reparse(part)
    if not public.is_dir() or public.resolve() != public:
        raise ValueError('Untrusted public root')
    files, required_public = [], set()
    for name in tracked:
        if not isinstance(name, str):
            raise ValueError('Invalid tracked metadata path')
        path = PurePosixPath(name)
        if ('\\' in name or ':' in name or path.is_absolute() or path.as_posix() != name
            or not path.parts or any(part in ('.','..') for part in path.parts)):
            raise ValueError('Invalid tracked metadata path')
        if (len(path.parts) >= 3 and tuple(p.casefold() for p in path.parts[:2]) == ('review','evidence')
            and path.parts[2].casefold() == stage.name.casefold()):
            raise ValueError('Unexpected registered public evidence')
        if {p.casefold() for p in path.parts} & (history.NON_METADATA_DIRS | {'private'}):
            continue
        if (path.name.lower() in history.NAMES
            or (path.parent.name.casefold() == 'job-inputs' and path.name.lower().endswith('.json'))):
            required_public.add(name)
    def walk_error(exc):
        raise ValueError('Historical metadata cannot be scanned') from exc
    seen_public = set()
    for folder, dirs, names in os.walk(public, followlinks=False, onerror=walk_error):
        directory = Path(folder)
        relative = directory.relative_to(public)
        parts = tuple(part.casefold() for part in relative.parts)
        own_stage = len(parts) >= 3 and parts[:3] == ('review','evidence',stage.name.casefold())
        if own_stage and dirs:
            raise ValueError('Unexpected registered public directory')
        kept = []
        for name in dirs:
            if name.casefold() == '.git' or name.casefold() in history.NON_METADATA_DIRS:
                continue
            child = directory/name
            old._reject_reparse(child)
            kept.append(name)
        dirs[:] = kept
        for name in names:
            lower = name.lower()
            path = directory/name
            rel = path.relative_to(public).as_posix()
            if own_stage:
                if rel not in own_public:
                    raise ValueError('Unexpected registered public metadata')
                continue
            recognized = (lower in history.NAMES or (directory.name.casefold() == 'job-inputs'
                          and lower.endswith('.json')))
            if not recognized:
                continue
            seen_public.add(rel)
            files.append((path, public, 'public/'+rel))
    if not required_public <= seen_public:
        raise ValueError('Tracked historical metadata is missing')
    for folder, dirs, names in os.walk(artifacts, followlinks=False, onerror=walk_error):
        directory = Path(folder)
        kept = []
        for name in dirs:
            child = directory/name
            old._reject_reparse(child)
            if child != stage and name.casefold() not in history.NON_METADATA_DIRS:
                kept.append(name)
        dirs[:] = kept
        for name in names:
            lower = name.lower()
            if not lower.endswith('.json'):
                continue
            if lower in history.NAMES or directory.name.casefold() == 'job-inputs':
                path=directory/name
                files.append((path, artifacts, 'local/'+path.relative_to(artifacts).as_posix()))
    values, identities, total = set(), [], 0
    for path, boundary, label in files:
        raw = _read(path, boundary)
        total += len(raw)
        if total > history.MAX_TOTAL:
            raise ValueError('Historical metadata exceeds budget')
        values.update(_history_seed_values(_json(raw), path.name.lower() == 'seeds.json'))
        identities.append((label,_sha(raw)))
    return values, len(files), total, tuple(sorted(identities))


def _public_evidence(stage, commit, tracked):
    prefix = 'review/evidence/' + stage.name + '/'
    stage_names = {n for n in tracked if n.casefold().startswith(prefix.casefold())}
    expected_names = {prefix + name for name in (*PUBLIC_FILES, SAFETY)}
    if stage_names != expected_names:
        raise ValueError('Unexpected registered public evidence')
    files = {}
    for name in (*PUBLIC_FILES, SAFETY):
        path = prefix + name
        committed = _git('show', commit + ':' + path)
        local = _read(stage / name, stage)
        public = _read(old.PUBLIC.joinpath(*PurePosixPath(path).parts), old.PUBLIC)
        if committed != local or committed != public:
            raise ValueError('Registered public evidence changed')
        files[name] = committed
    return files, stage_names


def _verify_documents(stage, parent, prereg_commit, files):
    decision, cfg, freeze, review, safety = (_json(files[n]) for n in (*PUBLIC_FILES, SAFETY))
    if not all(isinstance(v, dict) for v in (decision, cfg, freeze, review, safety)):
        raise ValueError('Invalid registration documents')
    if (set(safety) != {'schema', 'stage', 'parent_commit', 'marker_sha256', 'evidence_sha256'}
        or safety['schema'] != SCHEMA or safety['stage'] != stage.name or safety['parent_commit'] != parent
        or not _digest(safety['marker_sha256']) or not isinstance(safety['evidence_sha256'], dict)
        or safety['evidence_sha256'] != {n: _sha(files[n]) for n in PUBLIC_FILES}):
        raise ValueError('Invalid safety registration')
    if (cfg.get('baseline_head') != parent or decision.get('previous_commit') != parent
        or decision.get('status') != 'approved_for_implementation_not_execution'
        or freeze.get('parent_commit') != parent or review.get('baseline_commit') != parent
        or freeze.get('root_decision_sha256') != _sha(files['root-decision.json'])
        or freeze.get('registration_sha256') != _sha(files['registration.json'])
        or review.get('root_decision_sha256') != _sha(files['root-decision.json'])
        or review.get('registration_sha256') != _sha(files['registration.json'])
        or review.get('freeze_sha256') != _sha(files['freeze.json']) or review.get('passed') is not True
        or type(cfg.get('seed_count')) is not int or cfg['seed_count'] != 4
        or cfg.get('opponents') != ['round2', 'market', 'teacher']
        or cfg.get('policies') != ['merged', 'unweighted', 'teacher_task'] or cfg.get('seats') != [0, 1]
        or any(type(seat) is not int for seat in cfg['seats'])
        or cfg.get('seed_commitment_domain') != old.SEED_DOMAIN.decode('ascii').rstrip('\x00')):
        raise ValueError('Registered panel identity changed')
    for name in ('seed_commitment', 'seed_document_sha256', 'seed_inventory_sha256'):
        if not _digest(cfg.get(name)) or review.get(name) != cfg[name]:
            raise ValueError('Registered seed identity changed')
    if (type(decision.get('seed_count')) is not int or decision['seed_count'] != 4
        or decision.get('seed_commitment') != cfg['seed_commitment']
        or decision.get('seed_document_sha256') != cfg['seed_document_sha256']):
        raise ValueError('Decision seed identity changed')
    execution = _json(_read(stage/EXECUTION, stage, 16384), 16384)
    if (not isinstance(execution, dict) or set(execution) != {'commit','registration_sha256','freeze_sha256'}
        or execution['commit'] != prereg_commit or execution['registration_sha256'] != _sha(files['registration.json'])
        or execution['freeze_sha256'] != _sha(files['freeze.json'])):
        raise ValueError('Execution commit receipt changed')
    marker_raw = _read(stage / MARKER, stage, 16384)
    marker = _json(marker_raw, 16384)
    if (not isinstance(marker, dict) or set(marker) != {'schema', 'stage', 'parent_commit', 'created_exclusively', 'nonce'}
        or marker['schema'] != SCHEMA or marker['stage'] != stage.name or marker['parent_commit'] != parent
        or marker['created_exclusively'] is not True or not _digest(marker['nonce'])
        or _sha(marker_raw) != safety['marker_sha256']):
        raise ValueError('Stage ownership marker changed')
    private_raw, snapshot_raw = _read(stage/PRIVATE, stage, 16384), _read(stage/SNAPSHOT, stage)
    if _sha(private_raw) != cfg['seed_document_sha256'] or _sha(snapshot_raw) != cfg['seed_inventory_sha256']:
        raise ValueError('Private seed snapshot changed')
    document, snapshot = _json(private_raw, 16384), _json(snapshot_raw)
    seeds, salt = old.validate_seed_document(document)
    if old.seed_commitment(document) != cfg['seed_commitment']:
        raise ValueError('Private seed commitment changed')
    if (not isinstance(snapshot, dict) or not isinstance(snapshot.get('excluded_uint32'), list)
        or any(type(x) is not int or not 0 <= x < 2**32 for x in snapshot['excluded_uint32'])
        or len(set(snapshot['excluded_uint32'])) != len(snapshot['excluded_uint32'])):
        raise ValueError('Invalid frozen historical inventory')
    for value in (decision, cfg, freeze, review, safety):
        old.ensure_redacted(value, seeds, salt)
        if salt.casefold() in json.dumps(value, ensure_ascii=False, sort_keys=True).casefold():
            raise ValueError('Private seed or salt present in public metadata')
    return cfg, seeds, frozenset(snapshot['excluded_uint32'])


def _own_metadata(stage, cfg, seeds):
    """Validate recognized stage metadata without reading game results."""
    allowed_root = {*PUBLIC_FILES, SAFETY, MARKER, PRIVATE, SNAPSHOT, EXECUTION}
    expected = {p['game']: p for p in plans(seeds)}
    seen, identities = set(), []
    def walk_error(exc):
        raise ValueError('Stage metadata cannot be scanned') from exc
    for folder, dirs, names in os.walk(stage, followlinks=False, onerror=walk_error):
        directory = Path(folder)
        kept = []
        for name in dirs:
            child = directory/name
            old._reject_reparse(child)
            if name.casefold() not in history.NON_METADATA_DIRS:
                kept.append(name)
        dirs[:] = kept
        for name in names:
            path = directory/name
            rel = path.relative_to(stage)
            lower = name.casefold()
            if not lower.endswith('.json'):
                continue
            if len(rel.parts) == 1 and name in ROOT_RESULTS:
                continue
            if len(rel.parts) == 1 and name in allowed_root:
                raw = _read(path, stage)
                identities.append((rel.as_posix(), _sha(raw)))
                continue
            if len(rel.parts) != 2 or rel.parts[0] != 'job-inputs' or not name.endswith('.json'):
                raise ValueError('Unexpected stage seed metadata')
            game = name[:-5]
            if game not in expected or game in seen:
                raise ValueError('Unregistered game input')
            raw = _read(path, stage)
            item = _json(raw)
            plan = expected[game]
            extra = {'bot_sha256', 'rival_sha256', 'task_checkpoint_sha256'}
            if (not isinstance(item, dict) or set(item) != set(plan) | extra
                or type(item.get('seed')) is not int or type(item.get('side')) is not int
                or any(item.get(k) != v for k, v in plan.items())):
                raise ValueError('Game input differs from registered plan')
            task = None if plan['policy'] == 'teacher_task' else cfg.get(plan['policy']+'_checkpoint_sha256')
            rival = cfg.get('source_sha256', {}).get(plan['opponent']+'.py')
            if (not _digest(item['bot_sha256']) or not _digest(rival) or item['rival_sha256'] != rival
                or item['task_checkpoint_sha256'] != task or (task is not None and not _digest(task))
                or _sha(_read(stage/'agents'/(game+'.py'), stage)) != item['bot_sha256']
                or _sha(_read(stage/'sources'/(plan['opponent']+'.py'), stage)) != rival):
                raise ValueError('Game input source identity changed')
            seen.add(game)
            identities.append((rel.as_posix(), _sha(raw)))
    ordered = list(expected)
    if seen != set(ordered[:len(seen)]):
        raise ValueError('Game inputs are not a registered prefix')
    return tuple(sorted(identities))


def preselection_inventory(new_out):
    """Freeze a superset of recognized history before creating a new stage.

    The v1 baseline validates the old consumed batch and repository parent;
    the expanded scans add ignored/private metadata and richer seed encodings.
    The returned private inventory must be frozen before any seed is selected.
    It cannot prevent same-user replacement between a check and later use.
    """
    baseline = history.build_inventory(new_out)
    artifacts, stage = history._stage_path(old.trusted_artifacts(), new_out)
    _name(stage.name)
    parent = baseline.parent_commit
    if (not _commit(parent) or _git('rev-parse', 'HEAD').decode('ascii').strip() != parent
        or _git('status', '--porcelain').strip()):
        raise ValueError('Inventory parent changed')
    public_stage = old.PUBLIC/'review'/'evidence'/stage.name
    for path in (public_stage, public_stage.parent, public_stage.parent.parent, old.PUBLIC):
        old._reject_reparse(path)
    if stage.exists() or public_stage.exists():
        raise ValueError('Future stage name is already occupied')
    tracked = [p for p in _git('ls-files', '-z').decode('utf-8').split('\0') if p]
    current, count, size, identities = _expanded_history(artifacts, old.PUBLIC, stage, tracked, frozenset())
    if count < baseline.metadata_files or size < baseline.metadata_bytes:
        raise ValueError('Expanded history omitted legacy metadata')
    again, count2, size2, identities2 = _expanded_history(artifacts, old.PUBLIC, stage, tracked, frozenset())
    history._stage_path(artifacts, stage)
    for path in (public_stage, public_stage.parent, public_stage.parent.parent, old.PUBLIC):
        old._reject_reparse(path)
    if (again != current or count2 != count or size2 != size or identities2 != identities
        or _git('rev-parse', 'HEAD').decode('ascii').strip() != parent or _git('status', '--porcelain').strip()
        or stage.exists() or public_stage.exists()):
        raise ValueError('History or repository changed during preselection')
    return history.SeedInventory(baseline._values | current, count, size, parent)


def rescan_registered(stage, prereg_commit):
    """Validate ownership and reject newly observed collisions before use.

    Call before the first game, before each subsequent game and before final
    evaluation.  The coordinating process must still bind source identities,
    check its deadline and ensure unique job launch.  Values stay private in
    memory and are intentionally absent from exceptions and representation.
    """
    artifacts, stage = history._stage_path(old.trusted_artifacts(), stage)
    _name(stage.name)
    if not stage.is_dir() or not _commit(prereg_commit):
        raise ValueError('Invalid registered stage or commit')
    if _git('rev-parse', 'HEAD').decode('ascii').strip() != prereg_commit or _git('status', '--porcelain').strip():
        raise ValueError('Registered repository is not clean')
    parents = _git('rev-list', '--parents', '-n', '1', prereg_commit).decode('ascii').split()
    if len(parents) != 2 or parents[0] != prereg_commit or not _commit(parents[1]):
        raise ValueError('Registered commit must have one parent')
    parent = parents[1]
    _git('merge-base', '--is-ancestor', history.FAILURE_COMMIT, parent)
    prefix = 'review/evidence/' + stage.name + '/'
    parent_paths = _git('ls-tree', '-r', '--name-only', '-z', parent).decode('utf-8').split('\0')
    if any(p.casefold() == prefix[:-1].casefold() or p.casefold().startswith(prefix.casefold())
           for p in parent_paths if p):
        raise ValueError('Stage name existed in parent commit')
    tracked = [p for p in _git('ls-files', '-z').decode('utf-8').split('\0') if p]
    files, own_public = _public_evidence(stage, prereg_commit, tracked)
    cfg, seeds, frozen = _verify_documents(stage, parent, prereg_commit, files)
    consumed = history.load_consumed_prior()
    own = _own_metadata(stage, cfg, seeds)
    other = [p for p in tracked if p not in own_public]
    current, count, size, identities = _expanded_history(artifacts, old.PUBLIC, stage, other, own_public)
    _, legacy_count, legacy_size = history._scan_metadata(artifacts, old.PUBLIC, stage, other)
    if count < legacy_count or size < legacy_size:
        raise ValueError('Expanded history omitted legacy metadata')
    all_history = frozen | consumed._values | current
    inventory = history.SeedInventory(all_history, count, size, parent)
    inventory.require_unused(seeds)
    again, count2, size2, identities2 = _expanded_history(artifacts, old.PUBLIC, stage, other, own_public)
    files2, own_public2 = _public_evidence(stage, prereg_commit, tracked)
    cfg2, seeds2, frozen2 = _verify_documents(stage, parent, prereg_commit, files2)
    consumed2 = history.load_consumed_prior()
    if (again != current or count2 != count or size2 != size or identities2 != identities
        or _own_metadata(stage, cfg, seeds) != own
        or files2 != files or own_public2 != own_public or cfg2 != cfg or seeds2 != seeds or frozen2 != frozen
        or consumed2._values != consumed._values
        or _git('rev-parse', 'HEAD').decode('ascii').strip() != prereg_commit or _git('status', '--porcelain').strip()):
        raise ValueError('History or registration changed during rescan')
    history._stage_path(artifacts, stage)
    return inventory
