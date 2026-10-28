"""Prospective, private-seed registration for a bounded official panel.

This module never generates seeds.  The caller supplies a private document,
and the public record binds it using a domain-separated salted commitment.
"""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import stat
import struct
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
PUBLIC = ROOT.parent / 'publication/ai-game-lab'
STAGE_NAME = 'task-teacher-merged-eval-v1-20261028'
OUT = ROOT / 'artifacts' / STAGE_NAME
SOURCE = Path(__file__).resolve().parent
MERGE = ROOT / 'artifacts/task-teacher-merge-bc-v1-20261028'
BC = ROOT / 'artifacts/task-teacher-choice-bc-v1-20261027'
ACTION = ROOT / 'artifacts/task-action-observation-v1-20261027'
PREVIOUS = ROOT / 'artifacts/task-teacher-weighted-eval-v2-20261028'
BASE = '1df5de875dfa4354dfe03248cbb110d4cfc9570a'
PRIVATE_SEEDS = 'private-seeds.local.json'
SEED_DOMAIN = b'ai-game-lab:kaggriculture:merged-eval:v1:seed-commit\x00'
MIB, GIB = 1024**2, 1024**3
JOB = dict(max_rss_bytes=2*GIB, max_commit_bytes=3*GIB, min_available_bytes=2*GIB,
           wall_seconds=900, poll_seconds=.25, cleanup_seconds=10)
PREFLIGHT = dict(min_available_bytes=4*GIB, samples=3, interval_seconds=10)
# The stage checks current owned bytes before each job, with room for its
# bounded Python writes and parent metadata.  This is not an OS disk quota.
STORAGE = dict(total_bytes=10*GIB, game_write_bytes=128*MIB, free_floor_bytes=4*GIB,
               process_log_bytes=MIB, small_receipt_bytes=16384)
INDEPENDENT = dict(compressed_bytes=128*MIB, decoded_bytes=GIB, seconds=600, sample_count=20)
OFFICIAL = dict(new_games=72, live_transitions=72*719, primary_replay_transitions=72*719,
                independent_replay_transitions=719, total_transitions=145*719)
REUSED = (
    'learning/task_teacher_weighted_eval_v2/resources.py',
    'learning/task_teacher_choice_eval_v2/campaign.py',
    'learning/task_anchor_once_v1/campaign.py',
    'learning/task_anchor_once_v1/games.py',
    'learning/task_anchor_once_v1/storage.py',
    'learning/task_job_empty_exit_v1/sampling.py',
    'learning/task_job_sampling_race_v1/sampling.py',
    'learning/task_training_job_supervision_v1/job.py',
)
MERGE_EVIDENCE = ('registration.json','freeze.json','root-result-acceptance.json','fit-result.json','model-seal.json',
                  'fit-transport-result.json','gpu-guard-result.json','independent-result-assessment.json',
                  'pre-execution-commit.json','pre-review.json','privacy-revision.json')
OLD_EVIDENCE = ('registration.json','model-seal.json','root-result-acceptance.json')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda: stream.read(MIB), b''):
            digest.update(part)
    return digest.hexdigest()


def git(*args):
    return subprocess.check_output(['git', *args], cwd=PUBLIC)


def _absolute(path):
    return Path(os.path.abspath(os.fspath(path)))


def _reject_reparse(path):
    """Reject links/junctions/reparse points without following them."""
    try:
        info = Path(path).lstat()
    except FileNotFoundError:
        return False
    if (stat.S_ISLNK(info.st_mode) or getattr(info, 'st_file_attributes', 0) & 0x400
        or getattr(Path(path), 'is_junction', lambda: False)()):
        raise ValueError('Output path contains a link or reparse point')
    return True


def trusted_artifacts():
    """Require the artifacts ancestor under the actual, non-linked project root."""
    root = _absolute(ROOT)
    for ancestor in (root, *root.parents):
        _reject_reparse(ancestor)
    if not root.is_dir() or root.resolve(strict=True) != root:
        raise ValueError('Project root is not a trusted directory')
    artifacts = root/'artifacts'
    if not _reject_reparse(artifacts) or not artifacts.is_dir() or artifacts.resolve(strict=True) != artifacts:
        raise ValueError('Artifacts ancestor is not a trusted directory')
    return artifacts


def trusted_out(out):
    """Validate the fixed, already-created stage before any resolution or I/O.

    This rejects existing links and reparse points but is not a security
    boundary against same-user replacement between validation and use.
    """
    artifacts = trusted_artifacts(); expected = artifacts/STAGE_NAME
    if '..' in Path(out).parts or _absolute(OUT) != expected or _absolute(out) != expected:
        raise ValueError('Unexpected experiment output')
    if not _reject_reparse(expected) or not expected.is_dir() or expected.resolve(strict=True) != expected:
        raise ValueError('Experiment output is not a trusted directory')
    return expected


def create_out():
    """Create the fixed stage exclusively after checking its real parent.

    An existing entry is never removed or adopted. The second check detects a
    replaced or redirected stage but does not prevent same-user TOCTOU races.
    """
    artifacts = trusted_artifacts(); stage = artifacts/STAGE_NAME
    if _absolute(OUT) != stage:
        raise ValueError('Unexpected experiment output')
    stage.mkdir(exist_ok=False)
    return trusted_out(stage)


def write_new(path, value):
    stage = trusted_out(OUT); supplied = Path(path)
    if '..' in supplied.parts:
        raise ValueError('Output traversal refused')
    path = _absolute(supplied)
    if path == stage or not path.is_relative_to(stage):
        raise ValueError('Output outside owned experiment')
    current = stage
    for part in path.relative_to(stage).parts[:-1]:
        current = current/part
        if not _reject_reparse(current):
            current.mkdir()
        if not current.is_dir() or current.resolve(strict=True) != current:
            raise ValueError('Output parent is not a trusted directory')
    _reject_reparse(path)
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def deadline_check():
    stage = trusted_out(OUT)
    end = datetime.fromisoformat(read(stage/'root-decision.json')['deadline_utc'])
    if end.tzinfo is None or datetime.now(timezone.utc) >= end:
        raise TimeoutError('Panel deadline')


def clean_head(expected):
    if git('rev-parse', 'HEAD').decode().strip() != expected or git('status', '--porcelain').strip():
        raise ValueError('Unexpected or dirty public HEAD')


def digest(value):
    return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def validate_seed_document(value):
    if not isinstance(value, dict) or set(value) != {'version', 'seeds', 'salt'} or value['version'] != 1 or type(value['version']) is not int:
        raise ValueError('Invalid private seed document')
    seeds, salt = value['seeds'], value['salt']
    if (not isinstance(seeds, list) or len(seeds) != 4
        or any(type(seed) is not int or not 0 <= seed < 2**32 for seed in seeds) or len(set(seeds)) != 4
        or not digest(salt)):
        raise ValueError('Invalid private seed roles or salt')
    return seeds, salt


def seed_commitment(value):
    seeds, salt = validate_seed_document(value)
    return hashlib.sha256(SEED_DOMAIN + bytes.fromhex(salt) + struct.pack('>4I', *seeds)).hexdigest()


def _private_file(out, name):
    out = trusted_out(out)
    if not isinstance(name,str) or not name or Path(name).name != name:
        raise ValueError('Invalid private metadata filename')
    path = out / name
    _reject_reparse(path)
    if path.resolve().parent != out:
        raise ValueError('Private metadata outside experiment')
    return path


def _bounded_json(path, expected, max_bytes):
    with Path(path).open('rb') as stream:
        snapshot = stream.read(max_bytes+1)
    if len(snapshot) > max_bytes or hashlib.sha256(snapshot).hexdigest() != expected:
        raise ValueError('Private metadata identity changed')
    def unique(pairs):
        result = {}
        for key,value in pairs:
            if key in result: raise ValueError('Duplicate private metadata field')
            result[key] = value
        return result
    return json.loads(snapshot.decode('utf-8-sig'),object_pairs_hook=unique)


def ensure_redacted(value, seeds, salt):
    """Refuse seed/salt bytes in any public registration or decision value."""
    serialized = json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)
    if salt in serialized or any(re.search(r'(?<![0-9])'+str(seed)+r'(?![0-9])', serialized) for seed in seeds):
        raise ValueError('Private seed or salt present in public metadata')


def load_seeds(out, cfg):
    """Return seeds only after document, commitment and role-inventory checks."""
    out = trusted_out(out)
    path = _private_file(out, PRIVATE_SEEDS)
    inventory_path = _private_file(out, 'seed-inventory.local.json')
    if (type(cfg.get('seed_count')) is not int or cfg['seed_count'] != 4 or not digest(cfg.get('seed_commitment'))
        or not digest(cfg.get('seed_document_sha256')) or not digest(cfg.get('seed_inventory_sha256'))):
        raise ValueError('Private seed identity changed')
    document = _bounded_json(path, cfg['seed_document_sha256'], 16384)
    inv = _bounded_json(inventory_path, cfg['seed_inventory_sha256'], 16*MIB)
    seeds, _ = validate_seed_document(document)
    if (seed_commitment(document) != cfg['seed_commitment']
        or not isinstance(inv.get('excluded_uint32'), list)
        or set(seeds) & set(inv['excluded_uint32'])):
        raise ValueError('Seed commitment or historical role changed')
    return seeds


def _seed_values(value, active=False):
    """Read explicitly named seed roles, never treat scores or counts as seeds."""
    found = set()
    roles = {'train', 'training', 'test', 'holdout', 'validation', 'eval', 'evaluation',
             'development', 'dev', 'reserved', 'used', 'unused', 'selected', 'excluded', 'values'}
    if isinstance(value, dict):
        for key, item in value.items():
            label = str(key).lower()
            if label in ('seed_count','seeds_count') or label.endswith(('_seed_count', '_seeds_count', '_seed_sha256', '_seeds_sha256')):
                continue
            if active and label.isdigit():
                if int(label) < 2**32: found.add(int(label))
                found.update(_seed_values(item, False))
                continue
            selected = 'seed' in label or label == 'excluded_uint32' or (active and label in roles)
            found.update(_seed_values(item, selected))
    elif isinstance(value, list):
        for item in value:
            found.update(_seed_values(item, False if isinstance(item, dict) else active))
    elif active and type(value) is int and 0 <= value < 2**32:
        found.add(value)
    elif active and isinstance(value, str) and value.isdecimal() and int(value) < 2**32:
        found.add(int(value))
    return found


def inventory(*, check=deadline_check):
    """Scan recognized metadata without results; pre-decision callers may supply a check.

    A caller can use a no-op or its own time check before a formal decision
    exists.  Preparation and execution always call the default deadline gate.
    """
    if not callable(check): raise ValueError('Inventory check must be callable')
    names = {'registration.json', 'input-manifest.json', 'root-decision.json', 'seeds.json',
             'seed-inventory.json', 'seed-inventory.local.json', 'seed-roles.local.json',
             'seeds.local.json', PRIVATE_SEEDS}
    private_names = {'seeds.json', 'seeds.local.json', PRIVATE_SEEDS, 'seed-roles.local.json'}
    paths = []
    tracked = [n for n in git('ls-files', '-z').decode('utf-8').split('\0') if n]
    for name in tracked:
        path = Path(name)
        if OUT.name in path.parts or 'private' in {p.casefold() for p in path.parts}:
            continue
        if path.name.lower() in names or path.parent.name == 'job-inputs':
            paths.append(('public/'+name, PUBLIC/path))
    artifacts = trusted_artifacts()
    for path in sorted(artifacts.rglob('*.json')):
        if path.is_relative_to(OUT):
            continue
        rel = path.relative_to(artifacts)
        private = 'private' in {p.casefold() for p in rel.parts}
        if private and path.name.lower() not in private_names:
            continue
        if path.name.lower() in names or (not private and path.parent.name == 'job-inputs'):
            paths.append(('local/'+rel.as_posix(), path))
    seen, identities, total = set(), [], 0
    for label, path in paths:
        check()
        data = path.read_bytes(); total += len(data)
        if total > 256*MIB: raise ValueError('Seed metadata budget exceeded')
        seen.update(_seed_values(json.loads(data.decode('utf-8-sig')), path.name.lower() == 'seeds.json'))
        identities.append([label, hashlib.sha256(data).hexdigest()])
    return dict(excluded_uint32=sorted(seen), identities=identities, metadata_files=len(paths), metadata_bytes=total,
        scope='Recognized public and local registration, reservation, seed-list and job-plan metadata only',
        limitation='Deleted, external, inaccessible or unrecognized seed history cannot be excluded')


def public_equal(folder, names):
    for name in names:
        if git('show', BASE+':review/evidence/'+folder.name+'/'+name) != (folder/name).read_bytes():
            raise ValueError('Published predecessor differs')


def environment():
    venv, executable = ROOT/'.venv', ROOT/'.venv/Scripts/python.exe'
    if (os.name != 'nt' or not os.path.samefile(sys.executable, executable) or not os.path.samefile(sys.prefix, venv)
        or os.path.samefile(sys.prefix, sys.base_prefix) or sys.flags.optimize != 0 or 'PYTHONOPTIMIZE' in os.environ):
        raise ValueError('Expected project Windows interpreter')
    from learning.task_reward_gate_v1.experiment import versions
    return dict(interpreter_sha256=sha(executable), versions=versions(),
                optimization_level=0, python_optimize_env_absent=True)


def validate_parent_evidence(parent, seal, fit, transport, guard, old_seal, old_acceptance, hashes, precommit):
    """Bind existing receipts; this does not independently recompute training."""
    if (parent.get('passed') is not True or parent.get('competitive_gain_proven') is not False
        or parent.get('automatic_promotion') is not False or seal.get('passed') is not True
        or seal.get('checkpoint_sha256') != parent.get('checkpoint_sha256')
        or fit.get('checkpoint_sha256') != seal.get('checkpoint_sha256')
        or parent.get('registration_sha256') != hashes['registration.json']
        or parent.get('freeze_sha256') != hashes['freeze.json']
        or seal.get('freeze_sha256') != hashes['freeze.json'] or fit.get('freeze_sha256') != hashes['freeze.json']
        or guard.get('freeze_sha256') != hashes['freeze.json']
        or parent.get('pre_review_sha256') != hashes['pre-review.json']
        or parent.get('privacy_revision_sha256') != hashes['privacy-revision.json']
        or parent.get('model_seal_sha256') != hashes['model-seal.json']
        or parent.get('fit_result_sha256') != hashes['fit-result.json'] or seal.get('fit_result_sha256') != hashes['fit-result.json']
        or parent.get('fit_transport_sha256') != hashes['fit-transport-result.json'] or seal.get('fit_transport_sha256') != hashes['fit-transport-result.json']
        or parent.get('gpu_guard_sha256') != hashes['gpu-guard-result.json'] or seal.get('linux_guard_sha256') != hashes['gpu-guard-result.json']
        or parent.get('independent_assessment_sha256') != hashes['independent-result-assessment.json']
        or parent.get('pre_execution_commit') != precommit
        or fit.get('passed') is not True or transport.get('passed') is not True or guard.get('passed') is not True
        or old_seal.get('passed') is not True or old_acceptance.get('passed') is not True
        or old_acceptance.get('evidence_sha256',{}).get('model-seal.json') != hashes['old-model-seal.json']
        or old_acceptance.get('evidence_sha256',{}).get('registration.json') != hashes['old-registration.json']):
        raise ValueError('Historical model admission failed')


def historical_inputs():
    public_equal(MERGE, MERGE_EVIDENCE)
    public_equal(BC, OLD_EVIDENCE)
    public_equal(ACTION, ('registration.json',))
    public_equal(PREVIOUS, ('registration.json',))
    parent, parent_cfg, seal = (read(MERGE/name) for name in ('root-result-acceptance.json', 'registration.json', 'model-seal.json'))
    old_seal = read(BC/'model-seal.json'); old_acceptance = read(BC/'root-result-acceptance.json')
    fit, transport, guard = (read(MERGE/name) for name in ('fit-result.json','fit-transport-result.json','gpu-guard-result.json'))
    evidence_hashes = {name:sha(MERGE/name) for name in MERGE_EVIDENCE}
    evidence_hashes.update({'old-model-seal.json':sha(BC/'model-seal.json'), 'old-registration.json':sha(BC/'registration.json')})
    validate_parent_evidence(parent,seal,fit,transport,guard,old_seal,old_acceptance,evidence_hashes,
                             read(MERGE/'pre-execution-commit.json').get('commit'))
    paths = dict(merged=MERGE/'private/model-final.npz',
                 unweighted=ROOT/parent_cfg['protected_paths']['unweighted'],
                 crop=ROOT/parent_cfg['protected_paths']['crop'])
    hashes = dict(merged=seal['checkpoint_sha256'], unweighted=old_seal['checkpoint_sha256'],
                  crop=parent_cfg['protected_sha256']['crop'])
    if (hashes['unweighted'] != parent_cfg['protected_sha256']['unweighted']
        or any(sha(paths[name]) != hashes[name] for name in paths)
        or hashes['merged'] == hashes['unweighted']):
        raise ValueError('Historical checkpoint identity changed')
    opponent_cfg = read(ACTION/'registration.json')
    opponent_paths = {'round2.py': ACTION/'sources/round2.py', 'market.py': ACTION/'sources/market.py',
                      'teacher.py': ROOT/'bots/teacher_fertilizer_terminal_v2.py'}
    if (any(sha(opponent_paths[name]) != opponent_cfg['source_sha256'][name] for name in ('round2.py', 'market.py'))
        or git('show', BASE+':kaggriculture/bots/teacher_fertilizer_terminal_v2.py') != opponent_paths['teacher.py'].read_bytes()):
        raise ValueError('Frozen opponent identity changed')
    return parent_cfg, paths, hashes, opponent_paths


def runtime_files():
    previous = read(MERGE/'freeze.json')
    expected = dict(previous['runtime_files'])
    expected.update(previous['new_source_sha256'])
    local, public = {}, {}
    allowed_lines = {'decision/execution.py', 'decision/full_audit.py', 'decision/verify_audit.py', 'learning/round2_worker.py'}
    for rel in sorted(set(expected) | set(REUSED)):
        source = (ROOT/rel).read_bytes(); published = git('show', BASE+':kaggriculture/'+rel)
        local_digest, public_digest = hashlib.sha256(source).hexdigest(), hashlib.sha256(published).hexdigest()
        if rel in expected and local_digest != expected[rel]: raise ValueError('Historical runtime changed')
        if source != published and (rel not in allowed_lines or rel not in expected
            or source.replace(b'\r\n', b'\n') != published.replace(b'\r\n', b'\n')):
            raise ValueError('Published runtime differs')
        if rel in previous['public_runtime_sha256'] and public_digest != previous['public_runtime_sha256'][rel]:
            raise ValueError('Published historical runtime changed')
        local[rel], public[rel] = local_digest, public_digest
    return local, public


def prepare(out=OUT):
    """Freeze an externally supplied private seed document; never generate seeds."""
    out = trusted_out(out)
    if (out/'registration.json').exists(): raise ValueError('Panel already registered')
    deadline_check(); clean_head(BASE); decision = read(out/'root-decision.json')
    if (decision.get('previous_commit') != BASE or decision.get('status') != 'approved_for_implementation_not_execution'
        or type(decision.get('seed_count')) is not int or decision['seed_count'] != 4 or not digest(decision.get('seed_commitment'))
        or not digest(decision.get('seed_document_sha256')) or 'test_seeds' in decision or 'salt' in decision):
        raise ValueError('Missing private-seed prospective decision')
    seed_path = _private_file(out, PRIVATE_SEEDS)
    document = _bounded_json(seed_path, decision['seed_document_sha256'], 16384)
    seeds, salt = validate_seed_document(document)
    if seed_commitment(document) != decision['seed_commitment']: raise ValueError('Seed commitment changed')
    ensure_redacted(decision, seeds, salt)
    inv = inventory()
    if set(seeds) & set(inv['excluded_uint32']): raise ValueError('Seed appears in recognized history')
    parent_cfg, paths, hashes, opponents = historical_inputs()
    previous = read(PREVIOUS/'registration.json')
    env = environment()
    if env != previous['environment']: raise ValueError('Windows runtime changed')
    from learning.task_entry_repair_v1.experiment import installed_files
    installed = installed_files()
    if (sha(installed['rules']) != parent_cfg['engine_sha256'] or sha(installed['loader']) != parent_cfg['official_loader_sha256']
        or parent_cfg['engine_commit'] != 'f994f2050d58d84f649329c1e35ce09ada7007c5'):
        raise ValueError('Official engine changed')
    if shutil.disk_usage(ROOT).free < STORAGE['free_floor_bytes'] + STORAGE['game_write_bytes']:
        raise ValueError('Insufficient free disk')
    runtime, public_runtime = runtime_files()
    source = {p.relative_to(ROOT).as_posix(): sha(p) for p in sorted(SOURCE.glob('*.py'))}
    if not source: raise ValueError('Missing panel source')
    (out/'sources').mkdir(exist_ok=False)
    source_sha = {}
    for name, path in opponents.items():
        with (out/'sources'/name).open('xb') as stream: stream.write(path.read_bytes())
        source_sha[name] = sha(path)
    write_new(out/'seed-inventory.local.json', inv)
    from .evaluation import GATES
    cfg = dict(experiment='task-teacher-merged-eval-v1', baseline_head=BASE, deadline_utc=decision['deadline_utc'],
        seed_count=4, seed_commitment=decision['seed_commitment'], seed_document_sha256=decision['seed_document_sha256'],
        seed_inventory_sha256=sha(out/'seed-inventory.local.json'), seed_scope=inv['scope'], seed_history_limitation=inv['limitation'],
        opponents=['round2','market','teacher'], seats=[0,1], policies=['merged','unweighted','teacher_task'],
        merged_checkpoint=paths['merged'].relative_to(ROOT).as_posix(), merged_checkpoint_sha256=hashes['merged'],
        unweighted_checkpoint=paths['unweighted'].relative_to(ROOT).as_posix(), unweighted_checkpoint_sha256=hashes['unweighted'],
        crop_checkpoint=paths['crop'].relative_to(ROOT).as_posix(), crop_checkpoint_sha256=hashes['crop'], source_sha256=source_sha,
        engine_commit=parent_cfg['engine_commit'], engine_sha256=parent_cfg['engine_sha256'],
        official_loader_sha256=parent_cfg['official_loader_sha256'], environment=env,
        game_limits=previous['game_limits'], storage=STORAGE, official_job_supervision=JOB,
        independent_job_supervision=JOB, host_preflight=PREFLIGHT, independent_limits=INDEPENDENT,
        official_limits=OFFICIAL, gates=dict(GATES), no_retraining=True, no_automatic_promotion=True,
        retry_allowed=False, seed_commitment_domain=SEED_DOMAIN.decode('ascii').rstrip('\x00'),
        independent_selection='S01, merged, round2, seat 0; deterministic 20 callback sample',
        interpretation='Four seed clusters; descriptive screen, no significance or automatic promotion')
    ensure_redacted(cfg, seeds, salt)
    write_new(out/'registration.json', cfg)
    evidence = {folder.name+'/'+name: sha(folder/name) for folder,names in (
        (MERGE, MERGE_EVIDENCE),
        (BC, OLD_EVIDENCE),
        (ACTION, ('registration.json',)), (PREVIOUS, ('registration.json',))) for name in names}
    write_new(out/'freeze.json', dict(registration_sha256=sha(out/'registration.json'), root_decision_sha256=sha(out/'root-decision.json'),
        parent_commit=BASE, runtime_files=runtime, public_runtime_sha256=public_runtime,
        new_source_sha256=source, historical_evidence_sha256=evidence))
    return verify(out)


def verify(out=OUT):
    out = trusted_out(out)
    deadline_check(); cfg, freeze, decision = read(out/'registration.json'), read(out/'freeze.json'), read(out/'root-decision.json')
    from .evaluation import GATES
    if (freeze.get('parent_commit') != BASE or cfg.get('baseline_head') != BASE
        or sha(out/'registration.json') != freeze.get('registration_sha256')
        or sha(out/'root-decision.json') != freeze.get('root_decision_sha256')
        or type(cfg.get('seed_count')) is not int or cfg['seed_count'] != 4 or cfg.get('seed_commitment') != decision.get('seed_commitment')
        or cfg.get('seed_document_sha256') != decision.get('seed_document_sha256')
        or 'test_seeds' in cfg or 'salt' in cfg or 'test_seeds' in decision or 'salt' in decision
        or cfg.get('opponents') != ['round2','market','teacher'] or cfg.get('seats') != [0,1]
        or cfg.get('policies') != ['merged','unweighted','teacher_task'] or cfg.get('storage') != STORAGE
        or cfg.get('official_job_supervision') != JOB or cfg.get('independent_job_supervision') != JOB
        or cfg.get('host_preflight') != PREFLIGHT or cfg.get('independent_limits') != INDEPENDENT
        or cfg.get('official_limits') != OFFICIAL or cfg.get('gates') != GATES
        or cfg.get('no_retraining') is not True or cfg.get('no_automatic_promotion') is not True or cfg.get('retry_allowed') is not False):
        raise ValueError('Frozen panel or budget changed')
    for name, expected in freeze['historical_evidence_sha256'].items():
        if sha(ROOT/'artifacts'/name) != expected: raise ValueError('Historical evidence changed')
    parent_cfg, paths, hashes, opponents = historical_inputs()
    previous = read(PREVIOUS/'registration.json')
    if (decision.get('previous_commit') != BASE or decision.get('status') != 'approved_for_implementation_not_execution'
        or decision.get('deadline_utc') != cfg.get('deadline_utc') or decision.get('seed_count') != cfg['seed_count']
        or cfg.get('seed_commitment_domain') != SEED_DOMAIN.decode('ascii').rstrip('\x00')
        or cfg.get('engine_commit') != parent_cfg['engine_commit'] or cfg.get('engine_sha256') != parent_cfg['engine_sha256']
        or cfg.get('official_loader_sha256') != parent_cfg['official_loader_sha256']
        or cfg.get('game_limits') != previous['game_limits'] or cfg.get('environment') != previous['environment']
        or set(cfg.get('source_sha256',{})) != {'round2.py','market.py','teacher.py'}):
        raise ValueError('Frozen predecessor parameters changed')
    for name in ('merged','unweighted','crop'):
        if (cfg[name+'_checkpoint'] != paths[name].relative_to(ROOT).as_posix()
            or cfg[name+'_checkpoint_sha256'] != hashes[name]): raise ValueError('Model identity changed')
    for name, path in opponents.items():
        if sha(path) != cfg['source_sha256'][name] or sha(out/'sources'/name) != cfg['source_sha256'][name]:
            raise ValueError('Opponent identity changed')
    seeds = load_seeds(out, cfg)
    private_document = _bounded_json(_private_file(out, PRIVATE_SEEDS), cfg['seed_document_sha256'], 16384)
    ensure_redacted(decision, seeds, private_document['salt'])
    ensure_redacted(cfg, seeds, private_document['salt'])
    current = inventory()
    if set(seeds) & set(current['excluded_uint32']): raise ValueError('New historical seed collision')
    if {p.relative_to(ROOT).as_posix() for p in SOURCE.glob('*.py')} != set(freeze['new_source_sha256']):
        raise ValueError('Stage source set changed')
    for rel, expected in {**freeze['runtime_files'], **freeze['new_source_sha256']}.items():
        if sha(ROOT/rel) != expected: raise ValueError('Runtime source changed')
    if os.name == 'nt':
        if environment() != cfg['environment']: raise ValueError('Windows environment changed')
        from learning.task_entry_repair_v1.experiment import installed_files
        installed = installed_files()
        if sha(installed['rules']) != cfg['engine_sha256'] or sha(installed['loader']) != cfg['official_loader_sha256']:
            raise ValueError('Official engine changed')
    return cfg


def verify_commit(out=OUT):
    out = trusted_out(out); cfg = verify(out); receipt = read(out/'pre-execution-commit.json'); commit = receipt['commit']
    clean_head(commit); git('merge-base', '--is-ancestor', BASE, commit)
    for name in ('root-decision.json','registration.json','freeze.json','pre-review.json'):
        if git('show', commit+':review/evidence/'+out.name+'/'+name) != (out/name).read_bytes():
            raise ValueError('Committed evidence changed')
    for rel in read(out/'freeze.json')['new_source_sha256']:
        if git('show', commit+':kaggriculture/'+rel) != (ROOT/rel).read_bytes(): raise ValueError('Committed source changed')
    # Previous sources/evidence must remain byte-identical at the execution
    # commit.  A single diff checks this without hundreds of per-game git calls.
    freeze = read(out/'freeze.json')
    protected = {'kaggriculture/'+rel for rel in freeze['public_runtime_sha256']}
    protected.update('review/evidence/'+rel for rel in freeze['historical_evidence_sha256'])
    protected.add('kaggriculture/bots/teacher_fertilizer_terminal_v2.py')
    changed = set(git('diff','--name-only','-z',BASE,commit,'--').decode('utf-8').split('\0'))
    if protected & changed: raise ValueError('Published historical sources or evidence changed')
    return cfg


def ready(out=OUT):
    out = trusted_out(out); cfg = verify_commit(out); review = read(out/'pre-review.json'); receipt = read(out/'pre-execution-commit.json')
    if (review.get('passed') is not True or review.get('freeze_sha256') != sha(Path(out)/'freeze.json')
        or receipt.get('freeze_sha256') != sha(Path(out)/'freeze.json')
        or receipt.get('registration_sha256') != sha(Path(out)/'registration.json')):
        raise ValueError('Review or commitment changed')
    return cfg
