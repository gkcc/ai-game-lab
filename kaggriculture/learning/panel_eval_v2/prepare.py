"""Prepare one future panel without launching games or publishing data.

Call ``prepare`` only after the entire runtime has been reviewed and committed
as a clean parent.  It snapshots recognized history before exclusively
creating a stage, then selects private seeds and writes three draft documents.
An independently obtained review is supplied to ``accept_review``.  The
caller commits exactly the five public evidence files and only then calls
``bind_execution``.  Review independence is a procedural claim, not something
this code or the local commit can cryptographically prove.

This module is not a security boundary against a same-user process replacing
files between checks.  An interrupted attempt remains in place and is never
silently adopted or deleted; its selected seeds may need to be treated as
consumed.  Exceptions and return values do not include the seed values or salt.
"""
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import secrets
import shutil

from learning.panel_safety_v1 import seed_inventory as history
from learning.panel_safety_v2 import registered_rescan as rescan
from learning.task_teacher_merged_eval_v1 import evaluation
from learning.task_teacher_merged_eval_v1 import experiment as old
from . import experiment


_DRAFT = ('root-decision.json', 'registration.json', 'freeze.json')
_PRIVATE = (rescan.MARKER, rescan.PRIVATE, rescan.SNAPSHOT)
_SCOPE = 'Recognized public and local registration, reservation, seed-list and job-plan metadata only'
_LIMITATION = 'Deleted, external, inaccessible or unrecognized seed history cannot be excluded'
_REVIEW_FIELDS = frozenset(('reviewer', 'review_basis', 'passed', 'baseline_commit',
    'root_decision_sha256', 'registration_sha256', 'freeze_sha256', 'seed_document_sha256',
    'seed_commitment', 'seed_inventory_sha256', 'marker_sha256', 'scratch_helper_commitment',
    'open_findings', 'checks'))


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _json_bytes(value):
    try:
        raw = (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode('utf-8')
    except (TypeError, ValueError, RecursionError):
        raise ValueError('Invalid evidence document') from None
    if len(raw) > history.MAX_FILE:
        raise ValueError('Evidence document exceeds budget')
    return raw


def _redacted(value, seeds, salt):
    old.ensure_redacted(value, seeds, salt)
    if salt.casefold() in json.dumps(value, ensure_ascii=False, sort_keys=True).casefold():
        raise ValueError('Private seed or salt present in public metadata')


def _stage(out, must_exist):
    # Validate the supplied component before Windows path normalization.
    try:
        supplied = Path(out)
        rescan._name(supplied.name)
        _, stage = history._stage_path(old.trusted_artifacts(), supplied)
    except (TypeError, OSError):
        raise ValueError('Invalid panel stage') from None
    rescan._name(stage.name)
    if must_exist and (not stage.is_dir() or stage.resolve(strict=True) != stage):
        raise ValueError('Panel stage is unavailable')
    if not must_exist and old._reject_reparse(stage):
        raise ValueError('Panel stage is already occupied')
    return stage


def _write(stage, name, value):
    if name not in {*rescan.PUBLIC_FILES, rescan.SAFETY, *_PRIVATE, rescan.EXECUTION}:
        raise ValueError('Unexpected preparation output')
    _stage(stage, True)
    path = stage/name
    old._reject_reparse(path)
    with path.open('xb') as stream:
        stream.write(_json_bytes(value))


def _read(stage, name, limit=history.MAX_FILE):
    return history._json(history._read_regular(stage/name, stage, limit), limit)


def _deadline(value):
    if not isinstance(value, str):
        raise ValueError('Invalid panel deadline')
    try:
        end = datetime.fromisoformat(value)
        now = datetime.now(timezone.utc)
        if end.tzinfo is None or end.utcoffset() is None:
            raise ValueError('Invalid panel deadline')
        remaining = end.astimezone(timezone.utc) - now
        if remaining <= timedelta(0) or remaining > timedelta(hours=12):
            raise ValueError('Panel deadline outside the bounded window')
    except (OverflowError, ValueError):
        raise ValueError('Invalid or expired panel deadline') from None
    return value


def _helper_source(path):
    try:
        source = Path(path)
        if not source.is_absolute() or '..' in source.parts:
            raise ValueError('Invalid standalone helper path')
        return history._read_regular(source, Path(source.anchor), 4*1024*1024)
    except (TypeError, OSError):
        raise ValueError('Standalone helper is unavailable') from None


def _supplemental(parent):
    if not rescan._commit(parent):
        raise ValueError('Invalid source parent')
    result = {}
    for rel in sorted(experiment.SUPPLEMENTAL_FILES):
        try:
            local = history._read_regular(old.ROOT/rel, old.ROOT)
            committed = old.git('show', parent+':kaggriculture/'+rel)
        except (OSError, ValueError):
            raise ValueError('Panel source is unavailable') from None
        if local != committed:
            raise ValueError('Panel source differs from clean parent')
        result[rel] = {'local_sha256': _sha(local), 'public_sha256': _sha(committed)}
    return result


def _inputs(parent):
    """Check the frozen official runtime and inputs before choosing any seed."""
    old.clean_head(parent)
    parent_cfg, paths, hashes, opponents = old.historical_inputs()
    previous, env = old.read(old.PREVIOUS/'registration.json'), old.environment()
    if env != previous.get('environment'):
        raise ValueError('Frozen Windows runtime changed')
    if (parent_cfg.get('engine_commit') != experiment.ENGINE_COMMIT
        or parent_cfg.get('engine_sha256') != experiment.ENGINE_SHA256):
        raise ValueError('Frozen official engine changed')
    experiment._check_official_runtime(parent_cfg.get('official_loader_sha256'))
    if shutil.disk_usage(old.ROOT).free < old.STORAGE['free_floor_bytes'] + old.STORAGE['game_write_bytes']:
        raise ValueError('Insufficient free disk')
    reference = experiment._reference_freeze()
    supplemental = _supplemental(parent)
    freeze = {key: reference[key] for key in
              ('runtime_files', 'public_runtime_sha256', 'new_source_sha256', 'historical_evidence_sha256')}
    freeze.update(parent_commit=parent, supplemental_runtime_sha256=supplemental)
    experiment._validate_sources(None, parent, freeze, reference)
    old.clean_head(parent)
    return parent_cfg, paths, hashes, opponents, previous, env, reference, freeze


def _seed_document(inventory):
    selected = []
    # A bounded failure does not report candidate values; the stage remains.
    for _ in range(1024):
        seed = 1_000_000_000 + secrets.randbelow(2**32 - 1_000_000_000)
        if seed != evaluation.GATES['bootstrap_seed'] and seed not in selected and not inventory.contains(seed):
            selected.append(seed)
            if len(selected) == 4:
                break
    if len(selected) != 4:
        raise ValueError('Could not select four distinct unused seeds')
    inventory.require_unused(selected)
    return {'version': 1, 'seeds': selected, 'salt': secrets.token_hex(32)}


def _copy_opponents(stage, opponents):
    directory = stage/'sources'
    directory.mkdir(exist_ok=False)
    old._reject_reparse(directory)
    hashes = {}
    for name, source in opponents.items():
        if name not in ('round2.py', 'market.py', 'teacher.py'):
            raise ValueError('Unexpected opponent source')
        raw = history._read_regular(source, old.ROOT)
        with (directory/name).open('xb') as stream:
            stream.write(raw)
        hashes[name] = _sha(raw)
    if set(hashes) != {'round2.py', 'market.py', 'teacher.py'}:
        raise ValueError('Frozen opponents are incomplete')
    return hashes


def prepare(out, deadline_utc, helper_path):
    """Freeze local preregistration drafts. Never launch a match or commit."""
    deadline = _deadline(deadline_utc)
    helper_source = _helper_source(helper_path)
    stage = _stage(out, False)
    inventory = rescan.preselection_inventory(stage)  # before mkdir or selection
    parent = inventory.parent_commit
    inputs = _inputs(parent)
    parent_cfg, paths, hashes, opponents, previous, env, reference, freeze = inputs
    _deadline(deadline)
    old.clean_head(parent)
    _stage(stage, False)
    public_stage = old.PUBLIC/'review'/'evidence'/stage.name
    if old._reject_reparse(public_stage) or public_stage.exists():
        raise ValueError('Public evidence stage already exists')
    stage.mkdir(exist_ok=False)
    _stage(stage, True)
    _write(stage, rescan.MARKER, dict(schema=rescan.SCHEMA, stage=stage.name,
        parent_commit=parent, created_exclusively=True, nonce=secrets.token_hex(32)))
    snapshot = dict(excluded_uint32=list(inventory.private_values()), metadata_files=inventory.metadata_files,
        metadata_bytes=inventory.metadata_bytes, parent_commit=parent, scope=_SCOPE, limitation=_LIMITATION)
    _write(stage, rescan.SNAPSHOT, snapshot)
    document = _seed_document(inventory)
    seeds, salt = old.validate_seed_document(document)
    _write(stage, rescan.PRIVATE, document)
    commitment = old.seed_commitment(document)
    helper_identity = experiment.helper_commitment(helper_source, salt)
    doc_sha, inv_sha = old.sha(stage/rescan.PRIVATE), old.sha(stage/rescan.SNAPSHOT)
    decision = dict(previous_commit=parent, status='approved_for_implementation_not_execution',
        created_utc=datetime.now(timezone.utc).isoformat(), deadline_utc=deadline,
        purpose='Compare three frozen task-policy arms in a new complete official panel.',
        seed_count=4, seed_commitment=commitment, seed_document_sha256=doc_sha,
        seed_selection='System-random distinct uint32 values in [1000000000, 4294967295], excluding recognized history and the fixed bootstrap seed.',
        panel='Four private seed roles, three frozen opponents, both seats, three arms; 72 games and 72 primary replays, plus one selected extra replay.',
        interpretation='Four whole-seed clusters are exploratory; no significance, generalization or automatic promotion claim.',
        stopping='The registered deadline terminates the owned outer Job and its descendants; a started failure, unknown cleanup or interruption stops the whole attempt; no retries or partial scoring.',
        memory_wait=dict(experiment.MEMORY_WAIT),
        outer_supervision=dict(experiment.OUTER_SUPERVISION),
        scratch_helper_commitment=helper_identity,
        limitations='History may be missing or unrecognized; the local commitment is neither third-party timestamp nor proof of operator blinding.',
        provenance='Independent subagent review, not WebChat.')
    _redacted(decision, seeds, salt)
    _write(stage, 'root-decision.json', decision)
    source_sha = _copy_opponents(stage, opponents)
    cfg = dict(experiment='task-teacher-merged-eval-v2', baseline_head=parent, deadline_utc=deadline,
        seed_count=4, seed_commitment=commitment, seed_document_sha256=doc_sha,
        seed_inventory_sha256=inv_sha, seed_scope=_SCOPE, seed_history_limitation=_LIMITATION,
        opponents=['round2','market','teacher'], seats=[0,1], policies=['merged','unweighted','teacher_task'],
        merged_checkpoint=paths['merged'].relative_to(old.ROOT).as_posix(), merged_checkpoint_sha256=hashes['merged'],
        unweighted_checkpoint=paths['unweighted'].relative_to(old.ROOT).as_posix(), unweighted_checkpoint_sha256=hashes['unweighted'],
        crop_checkpoint=paths['crop'].relative_to(old.ROOT).as_posix(), crop_checkpoint_sha256=hashes['crop'],
        source_sha256=source_sha, engine_commit=parent_cfg['engine_commit'], engine_sha256=parent_cfg['engine_sha256'],
        official_loader_sha256=parent_cfg['official_loader_sha256'], official_core_sha256=experiment.ENGINE_CORE_SHA256,
        environment=env,
        game_limits=previous['game_limits'], storage=old.STORAGE, official_job_supervision=old.JOB,
        independent_job_supervision=old.JOB, host_preflight=old.PREFLIGHT,
        outer_supervision=dict(experiment.OUTER_SUPERVISION), independent_limits=old.INDEPENDENT,
        scratch_helper_commitment=helper_identity,
        official_limits=old.OFFICIAL, gates=dict(evaluation.GATES), no_retraining=True, no_automatic_promotion=True,
        retry_allowed=False, seed_commitment_domain=old.SEED_DOMAIN.decode('ascii').rstrip('\x00'),
        independent_selection=experiment.INDEPENDENT_SELECTION,
        interpretation='Four seed clusters; descriptive screen, no significance or automatic promotion')
    experiment._validate_contract(cfg, decision, freeze, reference, parent_cfg, paths, hashes, opponents, previous, env)
    _redacted(cfg, seeds, salt)
    _write(stage, 'registration.json', cfg)
    freeze['registration_sha256'] = old.sha(stage/'registration.json')
    freeze['root_decision_sha256'] = old.sha(stage/'root-decision.json')
    _redacted(freeze, seeds, salt)
    _write(stage, 'freeze.json', freeze)
    old.clean_head(parent)
    _deadline(deadline)
    return _draft_receipt(stage, require_clean=True)


def _draft_receipt(stage, *, require_clean):
    """Recheck exactly the draft data without reading game results."""
    stage = _stage(stage, True)
    allowed = {*_DRAFT, *_PRIVATE, 'sources'}
    if {p.name for p in stage.iterdir()} != allowed:
        raise ValueError('Unexpected data in unreviewed stage')
    for p in stage.iterdir():
        old._reject_reparse(p)
    marker, doc, inv = (_read(stage, name) for name in _PRIVATE)
    decision, cfg, freeze = (_read(stage, name) for name in _DRAFT)
    if not all(isinstance(item, dict) for item in (marker, doc, inv, decision, cfg, freeze)):
        raise ValueError('Invalid stage draft')
    parent = marker.get('parent_commit')
    if (set(marker) != {'schema','stage','parent_commit','created_exclusively','nonce'}
        or marker.get('schema') != rescan.SCHEMA or marker.get('stage') != stage.name
        or marker.get('created_exclusively') is not True or not rescan._digest(marker.get('nonce'))
        or not rescan._commit(parent) or cfg.get('baseline_head') != parent
        or decision.get('previous_commit') != parent or freeze.get('parent_commit') != parent):
        raise ValueError('Stage ownership or parent changed')
    if require_clean:
        old.clean_head(parent)
    seeds, salt = old.validate_seed_document(doc)
    excluded = inv.get('excluded_uint32')
    if (not isinstance(excluded, list) or any(type(n) is not int or not 0 <= n < 2**32 for n in excluded)
        or len(set(excluded)) != len(excluded) or set(excluded) & set(seeds)
        or any(seed < 1_000_000_000 for seed in seeds)
        or evaluation.GATES['bootstrap_seed'] in seeds or inv.get('parent_commit') != parent
        or old.seed_commitment(doc) != cfg.get('seed_commitment')
        or cfg.get('seed_document_sha256') != old.sha(stage/rescan.PRIVATE)
        or cfg.get('seed_inventory_sha256') != old.sha(stage/rescan.SNAPSHOT)
        or decision.get('seed_commitment') != cfg['seed_commitment']
        or decision.get('seed_document_sha256') != cfg['seed_document_sha256']
        or freeze.get('root_decision_sha256') != old.sha(stage/'root-decision.json')
        or freeze.get('registration_sha256') != old.sha(stage/'registration.json')):
        raise ValueError('Private snapshot or draft identity changed')
    _deadline(decision.get('deadline_utc'))
    inputs = _inputs(parent) if require_clean else None
    if inputs:
        parent_cfg, paths, hashes, opponents, previous, env, reference, _ = inputs
        experiment._validate_contract(cfg, decision, freeze, reference, parent_cfg, paths, hashes, opponents, previous, env)
        experiment._validate_sources(stage, parent, freeze, reference)
        if {p.name for p in (stage/'sources').iterdir()} != set(opponents):
            raise ValueError('Unexpected frozen opponent source')
        for name, source in opponents.items():
            copied = history._read_regular(stage/'sources'/name, stage)
            if _sha(copied) != old.sha(source):
                raise ValueError('Frozen opponent copy changed')
    for value in (decision, cfg, freeze):
        _redacted(value, seeds, salt)
    return dict(baseline_commit=parent, root_decision_sha256=old.sha(stage/'root-decision.json'),
        registration_sha256=old.sha(stage/'registration.json'), freeze_sha256=old.sha(stage/'freeze.json'),
        seed_document_sha256=cfg['seed_document_sha256'], seed_commitment=cfg['seed_commitment'],
        seed_inventory_sha256=cfg['seed_inventory_sha256'], marker_sha256=old.sha(stage/rescan.MARKER),
        scratch_helper_commitment=cfg['scratch_helper_commitment'])


def accept_review(out, review, helper_path):
    """Record an independently supplied review, with exact draft identities.

    The caller must obtain the review from a real separate agent and verify its
    provenance; a supplied object is not by itself proof of independence.
    """
    stage = _stage(out, True)
    expected = _draft_receipt(stage, require_clean=True)
    if (not isinstance(review, dict) or set(review) != _REVIEW_FIELDS
        or review.get('passed') is not True or review.get('review_basis') != 'independent_subagent'
        or not isinstance(review.get('reviewer'), str) or not review['reviewer'].startswith('/root/')
        or not review['reviewer'][6:] or review.get('open_findings') != []
        or not isinstance(review.get('checks'), list) or not review['checks']
        or any(not isinstance(item, str) or not item.strip() for item in review['checks'])
        or any(review.get(key) != value for key, value in expected.items())):
        raise ValueError('Independent review does not match the frozen draft')
    document = _read(stage, rescan.PRIVATE)
    seeds, salt = old.validate_seed_document(document)
    if experiment.helper_commitment(_helper_source(helper_path), salt) != expected['scratch_helper_commitment']:
        raise ValueError('Standalone helper changed after preparation')
    _redacted(review, seeds, salt)
    _write(stage, 'pre-review.json', review)
    files = {name: history._read_regular(stage/name, stage) for name in rescan.PUBLIC_FILES}
    safety = dict(schema=rescan.SCHEMA, stage=stage.name, parent_commit=expected['baseline_commit'],
        marker_sha256=old.sha(stage/rescan.MARKER), evidence_sha256={name:_sha(raw) for name,raw in files.items()})
    _redacted(safety, seeds, salt)
    _write(stage, rescan.SAFETY, safety)
    old.clean_head(expected['baseline_commit'])
    return {'baseline_commit':expected['baseline_commit'], 'reviewer':review['reviewer'],
            'public_files':list((*rescan.PUBLIC_FILES, rescan.SAFETY))}


def bind_execution(out, prereg_commit):
    """Bind the clean direct-child preregistration; do not start any game."""
    stage = _stage(out, True)
    if not rescan._commit(prereg_commit):
        raise ValueError('Invalid preregistration commit')
    if {p.name for p in stage.iterdir()} != {*_PRIVATE, *rescan.PUBLIC_FILES, rescan.SAFETY, 'sources'}:
        raise ValueError('Unexpected data in reviewed stage')
    old.clean_head(prereg_commit)
    parents = old.git('rev-list', '--parents', '-n', '1', prereg_commit).decode('ascii').split()
    marker = _read(stage, rescan.MARKER)
    if (len(parents) != 2 or parents[0] != prereg_commit or parents[1] != marker.get('parent_commit')):
        raise ValueError('Preregistration is not a direct child of the frozen parent')
    prefix = 'review/evidence/'+stage.name+'/'
    expected_paths = {prefix+name for name in (*rescan.PUBLIC_FILES, rescan.SAFETY)}
    changed = {p for p in old.git('diff', '--name-only', '-z', parents[1], prereg_commit, '--').decode('utf-8').split('\0') if p}
    if changed != expected_paths:
        raise ValueError('Preregistration commit must contain exactly the five evidence files')
    tracked = [p for p in old.git('ls-files', '-z').decode('utf-8').split('\0') if p]
    files, _ = rescan._public_evidence(stage, prereg_commit, tracked)
    registration = _sha(files['registration.json'])
    freeze = _sha(files['freeze.json'])
    _write(stage, rescan.EXECUTION, dict(commit=prereg_commit, registration_sha256=registration, freeze_sha256=freeze))
    # A failed post-bind check leaves the exclusive receipt; no silent retry.
    experiment.Context(stage, prereg_commit).ready()
    return {'commit':prereg_commit, 'registration_sha256':registration, 'freeze_sha256':freeze}
