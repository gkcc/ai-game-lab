"""Stage-bound, fail-closed identity and budget checks for a future panel.

This module neither chooses seeds nor starts a game.  It verifies a registered
stage before an adapter may hand it to the existing official worker.  The
public seed-rescan code checks the stage ownership and selected-seed history;
the additional checks here bind the frozen official runtime and model inputs.

``freeze.json`` keeps the v1 fields and adds ``supplemental_runtime_sha256``:
an exact mapping from the paths below to ``{local_sha256, public_sha256}``.
``root-decision.json`` adds the exact ``memory_wait`` mapping below.  Both
files must be part of the same preregistration commit as the safety evidence.
"""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess

from learning.panel_safety_v1 import memory_gate, seed_inventory as history
from learning.panel_safety_v2 import registered_rescan
from learning.task_teacher_merged_eval_v1 import evaluation
from learning.task_teacher_merged_eval_v1 import experiment as old


ROOT = old.ROOT
REFERENCE_COMMIT = '692b9c7cd640c31bfb319e7ae434c6fa7176ba43'
REFERENCE_STAGE = 'task-teacher-merged-eval-v1-20261028'
ENGINE_COMMIT = 'f994f2050d58d84f649329c1e35ce09ada7007c5'
ENGINE_SHA256 = 'bc8a54879ef02c7ea64b8b333d6a976f0ea65c4949149d01f463f23bccee653e'
ENGINE_CORE_SHA256 = '0922c4599a1b6e0d8c3dadf06ae5297f98138d859d686f00daee6f36e6d45d0e'

SUPPLEMENTAL_FILES = frozenset(
    'learning/task_teacher_merged_eval_v1/' + name + '.py'
    for name in ('experiment', 'evaluation', 'games', 'campaign', 'independent_method', 'independent_runner')
) | frozenset((
    'learning/panel_safety_v1/memory_gate.py',
    'learning/panel_safety_v1/seed_inventory.py',
    'learning/panel_safety_v2/registered_rescan.py',
    'learning/panel_eval_v2/experiment.py',
    'learning/panel_eval_v2/bootstrap.py',
    'learning/panel_eval_v2/launch.py',
    'learning/panel_eval_v2/epoch.py',
    'learning/panel_eval_v2/isolation.py',
    'learning/panel_eval_v2/adapter.py',
    'learning/panel_eval_v2/wrapper_verify.py',
    'learning/panel_eval_v2/runner.py',
    'learning/panel_eval_v2/worker.py',
    'learning/panel_eval_v2/independent_worker.py',
    'learning/panel_eval_v2/prepare.py',
))

MEMORY_WAIT = dict(
    min_available_bytes=memory_gate.MIN_AVAILABLE_BYTES,
    samples=memory_gate.SAMPLES,
    sample_interval_seconds=int(memory_gate.SAMPLE_INTERVAL_SECONDS),
    retry_interval_seconds=int(memory_gate.RETRY_INTERVAL_SECONDS),
    single_wait_seconds=int(memory_gate.SINGLE_WAIT_SECONDS),
    total_wait_seconds=int(memory_gate.TOTAL_WAIT_SECONDS),
)
OUTER_SUPERVISION = dict(
    hard_deadline='registered_deadline_utc',
    scope='owned_outer_job_including_descendants',
    termination='terminate_owned_job_and_wait_for_empty',
    cleanup_unknown='retain_scratch_marker',
)
INDEPENDENT_SELECTION = 'S01, merged, round2, seat 0; deterministic 20 callback sample'
HELPER_DOMAIN = b'panel-scratch-helper-v1\x00'


def helper_commitment(raw, salt):
    """Bind helper bytes without publishing their raw hash or local path."""
    if not isinstance(raw, bytes) or not raw or not old.digest(salt):
        raise ValueError('Invalid private helper identity')
    return hashlib.sha256(HELPER_DOMAIN + bytes.fromhex(salt) + hashlib.sha256(raw).digest()).hexdigest()


def _validate_helper_review(cfg, review):
    if (not isinstance(cfg, dict) or not isinstance(review, dict)
        or not old.digest(cfg.get('scratch_helper_commitment'))
        or review.get('scratch_helper_commitment') != cfg['scratch_helper_commitment']
        or review.get('passed') is not True or review.get('review_basis') != 'independent_subagent'
        or not isinstance(review.get('reviewer'), str) or not review['reviewer'].startswith('/root/')
        or not review['reviewer'][6:] or review.get('open_findings') != []
        or not isinstance(review.get('checks'), list) or not review['checks']
        or any(not isinstance(item, str) or not item.strip() for item in review['checks'])):
        raise ValueError('Reviewed helper identity changed')


def _same(a, b):
    """JSON equality without accepting booleans for numeric budget fields."""
    if type(a) is not type(b):
        return False
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(_same(a[key], b[key]) for key in a)
    if isinstance(a, list):
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    return a == b


def _read(stage, name, limit=history.MAX_FILE):
    return history._json(history._read_regular(stage/name, stage, limit), limit)


def _digest(raw):
    return hashlib.sha256(raw).hexdigest()


def _reference_freeze():
    """Read the earlier committed freeze, never the failed panel's results."""
    rel = 'review/evidence/' + REFERENCE_STAGE + '/freeze.json'
    committed = old.git('show', REFERENCE_COMMIT + ':' + rel)
    local = history._read_regular(old.OUT/'freeze.json', old.OUT)
    if local != committed:
        raise ValueError('Reference freeze differs from committed evidence')
    ref = history._json(committed)
    required = ('runtime_files', 'public_runtime_sha256', 'new_source_sha256', 'historical_evidence_sha256')
    if not isinstance(ref, dict) or any(not isinstance(ref.get(k), dict) or not ref[k] for k in required):
        raise ValueError('Invalid reference freeze')
    return ref


def _validate_contract(cfg, decision, freeze, reference, parent_cfg, paths, model_sha, opponents, previous, env):
    """Check immutable v1 configuration plus the registered bounded wait."""
    if not all(isinstance(v, dict) for v in (cfg, decision, freeze, reference, parent_cfg, paths,
                                              model_sha, opponents, previous, env)):
        raise ValueError('Invalid panel contract')
    fixed = {
        'seed_count': 4,
        'opponents': ['round2', 'market', 'teacher'],
        'seats': [0, 1],
        'policies': ['merged', 'unweighted', 'teacher_task'],
        'engine_commit': ENGINE_COMMIT,
        'engine_sha256': ENGINE_SHA256,
        'official_core_sha256': ENGINE_CORE_SHA256,
        'official_loader_sha256': parent_cfg.get('official_loader_sha256'),
        'environment': env,
        'game_limits': previous.get('game_limits'),
        'storage': old.STORAGE,
        'official_job_supervision': old.JOB,
        'independent_job_supervision': old.JOB,
        'host_preflight': old.PREFLIGHT,
        'outer_supervision': OUTER_SUPERVISION,
        'independent_limits': old.INDEPENDENT,
        'official_limits': old.OFFICIAL,
        'gates': evaluation.GATES,
        'no_retraining': True,
        'no_automatic_promotion': True,
        'retry_allowed': False,
        'independent_selection': INDEPENDENT_SELECTION,
    }
    if (parent_cfg.get('engine_commit') != ENGINE_COMMIT or parent_cfg.get('engine_sha256') != ENGINE_SHA256
        or not old.digest(parent_cfg.get('official_loader_sha256'))
        or not isinstance(previous.get('game_limits'), dict) or not previous['game_limits']
        or not isinstance(previous.get('environment'), dict) or not previous['environment']
        or not _same(env, previous.get('environment'))
        or any(key not in cfg or not _same(cfg[key], value) for key, value in fixed.items())
        or not _same(decision.get('memory_wait'), MEMORY_WAIT)
        or not _same(decision.get('outer_supervision'), OUTER_SUPERVISION)
        or not old.digest(cfg.get('scratch_helper_commitment'))
        or decision.get('scratch_helper_commitment') != cfg['scratch_helper_commitment']
        or decision.get('status') != 'approved_for_implementation_not_execution'
        or not isinstance(decision.get('deadline_utc'), str)
        or cfg.get('deadline_utc') != decision['deadline_utc']):
        raise ValueError('Registered engine, policy or resource contract changed')
    if set(paths) != {'merged', 'unweighted', 'crop'} or set(model_sha) != set(paths):
        raise ValueError('Invalid frozen model identities')
    for name in paths:
        try:
            rel = Path(paths[name]).relative_to(ROOT).as_posix()
        except (ValueError, TypeError):
            raise ValueError('Frozen model path outside project') from None
        if cfg.get(name+'_checkpoint') != rel or cfg.get(name+'_checkpoint_sha256') != model_sha[name]:
            raise ValueError('Frozen model identity changed')
    if set(opponents) != {'round2.py', 'market.py', 'teacher.py'}:
        raise ValueError('Invalid frozen opponents')
    try:
        sources = {name: old.sha(path) for name, path in opponents.items()}
    except OSError:
        raise ValueError('Frozen opponent is unavailable') from None
    if not _same(cfg.get('source_sha256'), sources):
        raise ValueError('Frozen opponent identities changed')
    for key in ('runtime_files', 'public_runtime_sha256', 'new_source_sha256', 'historical_evidence_sha256'):
        if key not in reference or not _same(freeze.get(key), reference[key]):
            raise ValueError('Reference runtime or historical evidence changed')
    return sources


def _validate_sources(stage, prereg_commit, freeze, reference):
    """Bind old source snapshots plus a fixed set of new runtime files."""
    supplemental = freeze.get('supplemental_runtime_sha256')
    parent = freeze.get('parent_commit')
    if (not registered_rescan._commit(parent) or not isinstance(supplemental, dict)
        or set(supplemental) != SUPPLEMENTAL_FILES):
        raise ValueError('Supplemental source set is incomplete')
    for rel in sorted(SUPPLEMENTAL_FILES):
        entry = supplemental[rel]
        if (not isinstance(entry, dict) or set(entry) != {'local_sha256', 'public_sha256'}
            or not history.old.digest(entry['local_sha256']) or not history.old.digest(entry['public_sha256'])):
            raise ValueError('Invalid supplemental source identity')
        try:
            local = history._read_regular(ROOT/rel, ROOT)
            published = old.git('show', prereg_commit + ':kaggriculture/' + rel)
            before_registration = old.git('show', parent + ':kaggriculture/' + rel)
        except (OSError, ValueError):
            raise ValueError('Supplemental source is unavailable') from None
        if (local != published or before_registration != published or _digest(local) != entry['local_sha256']
            or _digest(published) != entry['public_sha256']):
            raise ValueError('Supplemental source changed')
    try:
        for rel, expected in {**reference['runtime_files'], **reference['new_source_sha256']}.items():
            if old.sha(ROOT/rel) != expected:
                raise ValueError('Frozen runtime source changed')
    except OSError:
        raise ValueError('Frozen runtime source is unavailable') from None
    protected = {'kaggriculture/'+rel for rel in reference['public_runtime_sha256']}
    protected.update('kaggriculture/'+rel for rel in reference['new_source_sha256'])
    protected.update('review/evidence/'+rel for rel in reference['historical_evidence_sha256'])
    protected.add('kaggriculture/bots/teacher_fertilizer_terminal_v2.py')
    changed = set(old.git('diff', '--name-only', '-z', REFERENCE_COMMIT, prereg_commit, '--').decode('utf-8').split('\0'))
    if protected & changed:
        raise ValueError('Published historical runtime or evidence changed')


def _installed_files():
    from learning.task_entry_repair_v1.experiment import installed_files
    files = installed_files()
    return {**files, 'core': files['loader'].with_name('core.py')}


def _check_official_runtime(loader_sha256):
    """Verify the installed rules, entry loader, core and pinned upstream HEAD."""
    try:
        installed = _installed_files()
        head = subprocess.check_output(['git', '-C', str(ROOT/'upstream'), 'rev-parse', 'HEAD'],
                                       stderr=subprocess.DEVNULL, timeout=15).decode('ascii').strip()
        if (head != ENGINE_COMMIT or old.sha(installed['rules']) != ENGINE_SHA256
            or old.sha(installed['loader']) != loader_sha256
            or old.sha(installed['core']) != ENGINE_CORE_SHA256):
            raise ValueError('Official runtime identity changed')
    except (OSError, KeyError, UnicodeError, subprocess.SubprocessError):
        raise ValueError('Official runtime identity unavailable') from None


class Context:
    """Bind worker-facing checks to one registered stage and preregistration."""

    ROOT = ROOT

    def __init__(self, stage, prereg_commit):
        _, out = history._stage_path(old.trusted_artifacts(), stage)
        registered_rescan._name(out.name)
        if not out.is_dir() or not registered_rescan._commit(prereg_commit):
            raise ValueError('Invalid registered stage or commit')
        self.OUT, self.prereg_commit = out, prereg_commit

    @staticmethod
    def trusted_artifacts():
        return old.trusted_artifacts()

    @staticmethod
    def _reject_reparse(path):
        return old._reject_reparse(path)

    def trusted_out(self, out):
        _, requested = history._stage_path(old.trusted_artifacts(), out)
        if requested != self.OUT or not requested.is_dir():
            raise ValueError('Unexpected registered stage')
        return requested

    def deadline_check(self):
        out = self.trusted_out(self.OUT)
        decision, cfg = _read(out, 'root-decision.json'), _read(out, 'registration.json')
        if (not isinstance(decision, dict) or not isinstance(cfg, dict)
            or not isinstance(decision.get('deadline_utc'), str)
            or cfg.get('deadline_utc') != decision['deadline_utc']):
            raise ValueError('Invalid registered deadline')
        try:
            deadline = datetime.fromisoformat(decision['deadline_utc'])
        except ValueError:
            raise ValueError('Invalid registered deadline') from None
        if deadline.tzinfo is None or deadline.utcoffset() is None:
            raise ValueError('Registered deadline has no timezone')
        if datetime.now(timezone.utc) >= deadline.astimezone(timezone.utc):
            raise TimeoutError('Registered deadline reached')

    def ready(self, out=None):
        out = self.trusted_out(self.OUT if out is None else out)
        self.deadline_check()
        registered_rescan.rescan_registered(out, self.prereg_commit)
        cfg, decision, freeze = (_read(out, name) for name in ('registration.json', 'root-decision.json', 'freeze.json'))
        review = _read(out, 'pre-review.json')
        reference = _reference_freeze()
        try:
            parent_cfg, paths, hashes, opponents = old.historical_inputs()
            previous, env = old.read(old.PREVIOUS/'registration.json'), old.environment()
            _validate_contract(cfg, decision, freeze, reference, parent_cfg, paths, hashes, opponents, previous, env)
            _validate_helper_review(cfg, review)
            for name, path in opponents.items():
                if old.sha(out/'sources'/name) != old.sha(path):
                    raise ValueError('Registered opponent copy changed')
            _check_official_runtime(parent_cfg['official_loader_sha256'])
            _validate_sources(out, self.prereg_commit, freeze, reference)
        except OSError:
            raise ValueError('Required frozen input is unavailable') from None
        self.deadline_check()
        registered_rescan.rescan_registered(out, self.prereg_commit)
        if not _same(cfg, _read(out, 'registration.json')):
            raise ValueError('Registration changed during readiness check')
        return cfg

    def verify(self, out=None):
        return self.ready(out)

    def verify_commit(self, out=None):
        return self.ready(out)

    def load_seeds(self, out, cfg):
        """Return seed values only to the local worker after identity checks."""
        stage = self.trusted_out(out)
        registered = _read(stage, 'registration.json')
        if not _same(cfg, registered):
            raise ValueError('Registration changed before seed access')
        raw = history._read_regular(stage/'private-seeds.local.json', stage, 16384)
        if _digest(raw) != cfg.get('seed_document_sha256'):
            raise ValueError('Private seed document changed')
        document = history._json(raw, 16384)
        seeds, _ = old.validate_seed_document(document)
        if old.seed_commitment(document) != cfg.get('seed_commitment'):
            raise ValueError('Private seed commitment changed')
        return seeds

    def write_new(self, path, value):
        """Exclusive stage-local JSON output with static reparse checks."""
        stage = self.trusted_out(self.OUT)
        supplied = Path(path)
        if '..' in supplied.parts:
            raise ValueError('Output traversal refused')
        if os.name == 'nt':
            # GetFullPathName can strip trailing dots or spaces, so inspect
            # the original components before normalizing the path.
            for part in supplied.parts:
                if part != supplied.anchor and (':' in part or part.rstrip(' .') != part or Path(part).is_reserved()):
                    raise ValueError('Unsupported Windows output path')
        target = Path(os.path.abspath(supplied))
        if target == stage or not target.is_relative_to(stage):
            raise ValueError('Output outside registered stage')
        if os.name == 'nt':
            for part in target.relative_to(stage).parts:
                if ':' in part or part.rstrip(' .') != part or Path(part).is_reserved():
                    raise ValueError('Unsupported Windows output path')
        try:
            payload = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
        except (ValueError, TypeError, RecursionError):
            raise ValueError('Invalid evidence document') from None
        if len(payload.encode('utf-8')) > history.MAX_FILE:
            raise ValueError('Evidence document exceeds budget')
        parent = stage
        for part in target.relative_to(stage).parts[:-1]:
            parent = parent/part
            if not old._reject_reparse(parent):
                parent.mkdir()
            if not parent.is_dir() or parent.resolve(strict=True) != parent:
                raise ValueError('Untrusted evidence directory')
        old._reject_reparse(target)
        with target.open('x', encoding='utf-8', newline='\n') as stream:
            stream.write(payload)
