"""Bind the failed v2 panel's four private roles to committed evidence.

Call this before selecting a new panel's seeds and again at each registered
history rescan.  Returned values exist only in the redacted-repr SeedInventory;
callers must not log its private values or publish the source document.

This is a cooperative consistency check, not a defense against another process
of the same user replacing and restoring data between checks.  It reads no
game output, replay, per-turn state, job input, or economic result.
"""
import hashlib
from pathlib import Path
import subprocess

from learning.panel_safety_v1 import seed_inventory as history
from learning.task_teacher_merged_eval_v1 import experiment as old


REGISTRATION_COMMIT = 'e2d96abe357dd2a59f9cc23a23953bca8e967e7c'
SOURCE_COMMIT = '69c01ec9b93a88ae7046dc1060611e19463e4a96'
STAGE_NAME = 'task-teacher-merged-eval-v2-20261029'
EVIDENCE_PREFIX = 'review/evidence/' + STAGE_NAME + '/'
FIRST_GAME = 'holdout-S01-round2-0-merged'
RECEIPT_PATHS = {
    'binding': 'pre-execution-commit.json',
    'campaign_failure': 'games-failure.local.json',
    'canonical': 'results/canonical/' + FIRST_GAME + '.json',
    'coordinator_failure': 'results/coordinator-failure.json',
    'freeze': 'freeze.json',
    'job_start': 'results/job-start/' + FIRST_GAME + '.json',
    'job_supervision': 'job-supervision/' + FIRST_GAME + '.json',
    'launcher_result': 'replays/launcher-result.json',
    'registration': 'registration.json',
    'worker_storage': 'job-results/' + FIRST_GAME + '.storage.json',
}


def _git(*args):
    """Capture errors without opening a writable null device under a budget."""
    try:
        return subprocess.check_output(['git', *args], cwd=old.PUBLIC,
                                       stderr=subprocess.STDOUT, timeout=15)
    except (OSError, subprocess.SubprocessError):
        raise ValueError('Prior v2 Git identity unavailable') from None


def _digest(data):
    return hashlib.sha256(data).hexdigest()


def _commit(value):
    return isinstance(value,str) and len(value) == 40 and all(c in '0123456789abcdef' for c in value)


def _number(obj, name, expected):
    return type(obj.get(name)) is int and obj[name] == expected


def _verify(registration, freeze, certificate, private_document, receipts):
    """Pure validation of bounded byte snapshots; private roles never logged."""
    if (not isinstance(receipts,dict) or set(receipts) != set(RECEIPT_PATHS)
        or any(not isinstance(raw,bytes) for raw in
               (registration,freeze,certificate,private_document,*receipts.values()))):
        raise ValueError('Prior v2 evidence incomplete')
    cfg, frozen, cert = (history._json(raw) for raw in (registration,freeze,certificate))
    if not all(isinstance(value,dict) for value in (cfg,frozen,cert)):
        raise ValueError('Prior v2 evidence invalid')
    hashes = cert.get('local_receipt_sha256')
    if (cert.get('schema') != 'bounded-panel-failure-v1'
        or cert.get('registration_commit') != REGISTRATION_COMMIT or cert.get('source_commit') != SOURCE_COMMIT
        or cert.get('status') != 'failed-no-partial-economic-evaluation'
        or not _number(cert,'seed_roles_reserved_no_reuse',4)
        or not _number(cert,'attempted_games',1) or not _number(cert,'accepted_games',0)
        or cert.get('retry_allowed') is not False or cert.get('economic_evaluation_started') is not False
        or cert.get('partial_results_used_for_selection') is not False
        or not isinstance(hashes,dict) or set(hashes) != set(RECEIPT_PATHS)
        or any(not old.digest(hashes[key]) or hashes[key] != _digest(raw) for key,raw in receipts.items())):
        raise ValueError('Prior v2 failure certificate invalid')
    reg_sha, freeze_sha = _digest(registration), _digest(freeze)
    if (receipts['registration'] != registration or receipts['freeze'] != freeze
        or hashes['registration'] != reg_sha or hashes['freeze'] != freeze_sha
        or frozen.get('registration_sha256') != reg_sha
        or not _number(cfg,'seed_count',4)
        or cfg.get('seed_document_sha256') != _digest(private_document)
        or cfg.get('seed_commitment_domain') != old.SEED_DOMAIN.decode('ascii').rstrip('\x00')):
        raise ValueError('Prior v2 registration or private document changed')
    binding, failure, coordinator, launcher, supervision, storage, canonical, started = (
        history._json(receipts[key]) for key in ('binding','campaign_failure','coordinator_failure',
            'launcher_result','job_supervision','worker_storage','canonical','job_start'))
    if not all(isinstance(value,dict) for value in
               (binding,failure,coordinator,launcher,supervision,storage,canonical,started)):
        raise ValueError('Prior v2 local receipts invalid')
    if binding != dict(commit=REGISTRATION_COMMIT,registration_sha256=reg_sha,freeze_sha256=freeze_sha):
        raise ValueError('Prior v2 execution binding changed')
    job = supervision.get('supervision')
    budget = storage.get('output')
    if (not _number(failure,'attempted_games',1) or not _number(failure,'completed_games',0)
        or failure.get('current_game_additional_transitions') != 'unknown' or failure.get('retry_allowed') is not False
        or coordinator.get('error_kind') != 'CampaignFailure' or coordinator.get('cleanup_state') != 'confirmed_empty'
        or coordinator.get('retry_allowed') is not False
        or launcher.get('passed') is not False or not _number(launcher,'exit_code',1)
        or launcher.get('child_reaped') is not True or launcher.get('job_members_zero') is not True
        or launcher.get('handles_closed') is not True or launcher.get('cleanup_known') is not True
        or launcher.get('scratch_retained') is not False or launcher.get('retry_allowed') is not False
        or supervision.get('game') != FIRST_GAME or supervision.get('error_kind') is not None
        or supervision.get('cleanup_state') != 'recorded' or supervision.get('retry_allowed') is not False
        or not isinstance(job,dict) or job.get('passed') is not False or not _number(job,'exit_code',1)
        or job.get('launcher_reaped') is not True or job.get('job_members_zero') is not True
        or job.get('cleanup_error_kind') is not None or job.get('stop_reason') is not None
        or storage.get('passed') is not False or storage.get('error_kind') != 'ValueError'
        or not isinstance(budget,dict) or not _number(budget,'used_bytes',0)
        or not _number(budget,'created_count',0) or not _number(budget,'refused',0)):
        raise ValueError('Prior v2 failure or cleanup receipt changed')
    if (canonical.get('game') != FIRST_GAME or canonical.get('passed') is not True
        or canonical.get('launcher_reaped') is not True or canonical.get('job_members_zero') is not True
        or canonical.get('isolated_cache_prefix_absent') is not True or canonical.get('retry_allowed') is not False
        or not old.digest(canonical.get('wrapper_sha256')) or started.get('game') != FIRST_GAME
        or started.get('wrapper_sha256') != canonical['wrapper_sha256'] or started.get('retry_allowed') is not False):
        raise ValueError('Prior v2 wrapper receipts changed')
    document = history._json(private_document,16384)
    seeds, _ = old.validate_seed_document(document)
    if old.seed_commitment(document) != cfg.get('seed_commitment'):
        raise ValueError('Prior v2 private commitment changed')
    return history.SeedInventory(seeds,0,0)


def _snapshot(stage, failure_commit):
    public = old.PUBLIC
    raw = {key:history._read_regular(stage/rel,stage) for key,rel in RECEIPT_PATHS.items()}
    raw['private_document'] = history._read_regular(stage/'private-seeds.local.json',stage,16384)
    raw['registered'] = _git('show',REGISTRATION_COMMIT+':'+EVIDENCE_PREFIX+'registration.json')
    raw['frozen'] = _git('show',REGISTRATION_COMMIT+':'+EVIDENCE_PREFIX+'freeze.json')
    raw['certificate'] = _git('show',failure_commit+':'+EVIDENCE_PREFIX+'failure-certificate.json')
    for name,key in (('registration.json','registered'),('freeze.json','frozen'),
                     ('failure-certificate.json','certificate')):
        if history._read_regular(public/EVIDENCE_PREFIX/name,public) != raw[key]:
            raise ValueError('Prior v2 public evidence differs from committed bytes')
    return raw


def load_failed_v2(failure_commit, descendant_commit):
    """Return four reserved private roles after binding Git and local receipts.

    ``failure_commit`` must contain the reviewed v2 failure certificate and be
    an ancestor of ``descendant_commit``, which must be the current clean HEAD.
    The caller must verify these values appear in its historical exclusion set
    before selecting seeds, and reject collisions again after registration.
    """
    if not _commit(failure_commit) or not _commit(descendant_commit):
        raise ValueError('Prior v2 failure or descendant commit is not pinned')
    try:
        artifacts = old.trusted_artifacts()
        _, stage = history._stage_path(artifacts,artifacts/STAGE_NAME)
        if not stage.is_dir() or stage.name != STAGE_NAME:
            raise ValueError('Prior v2 stage unavailable')
        parents = _git('rev-list','--parents','-n','1',REGISTRATION_COMMIT).decode('ascii').split()
        if parents != [REGISTRATION_COMMIT,SOURCE_COMMIT]:
            raise ValueError('Prior v2 registration ancestry changed')
        _git('merge-base','--is-ancestor',REGISTRATION_COMMIT,failure_commit)
        _git('merge-base','--is-ancestor',failure_commit,descendant_commit)
        if _git('rev-parse','HEAD').decode('ascii').strip() != descendant_commit or _git('status','--porcelain').strip():
            raise ValueError('Prior v2 descendant repository is not clean')
        before = _snapshot(stage,failure_commit)
        inventory = _verify(before['registered'],before['frozen'],before['certificate'],
                            before['private_document'],{key:before[key] for key in RECEIPT_PATHS})
        after = _snapshot(stage,failure_commit)
        if (before != after or _git('rev-parse','HEAD').decode('ascii').strip() != descendant_commit
            or _git('status','--porcelain').strip()):
            raise ValueError('Prior v2 evidence changed during validation')
        return inventory
    except (OSError, ValueError, TypeError, KeyError, UnicodeError, subprocess.SubprocessError):
        raise ValueError('Prior v2 reserved-seed evidence is invalid or unavailable') from None
