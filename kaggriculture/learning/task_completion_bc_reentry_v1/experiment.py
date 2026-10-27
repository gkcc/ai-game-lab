"""Prospective registration for one bounded completion BC reentry."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[2]
PUBLIC = ROOT.parent / 'publication/ai-game-lab'
OUT = ROOT / 'artifacts/task-completion-bc-reentry-v1-20261027'
SOURCE = Path(__file__).resolve().parent
PRECISION = ROOT / 'artifacts/task-completion-precision-v1-20261027'
PREVIOUS = ROOT / 'artifacts/task-completion-bc-v1-20261027'
BASE = 'fb2879ede4eba6e5ee66156860420b0d08385395'
EXCLUDED = {1210575027, 441357442, 2026102708, 3911611436, 2328454240}
MIB = 1024**2


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(MIB), b''): h.update(block)
    return h.hexdigest()


def git(*args):
    return subprocess.check_output(['git', *args], cwd=PUBLIC)


def write_new(path, value):
    path = Path(path).resolve()
    if not path.is_relative_to(OUT): raise ValueError('Output outside experiment')
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8', newline='\n') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')


def deadline_check():
    value = read(OUT/'root-decision.json')['deadline_utc']
    deadline = datetime.fromisoformat(value)
    if deadline.tzinfo is None or datetime.now(timezone.utc) >= deadline:
        raise TimeoutError('Stage deadline reached or invalid')


def clean_head(expected):
    if git('rev-parse', 'HEAD').decode().strip() != expected or git('status', '--porcelain').strip():
        raise ValueError('Unexpected or dirty public HEAD')


def select_seeds(seen):
    excluded = set(seen) | EXCLUDED
    selected = []
    index = 0
    while len(selected) < 2:
        candidate = int.from_bytes(hashlib.sha256(f'task-completion-bc-reentry-v1:20261027:{index}'.encode()).digest()[:4], 'big')
        index += 1
        if candidate not in excluded and candidate not in selected: selected.append(candidate)
    return selected


def seed_inventory(out=OUT):
    from learning.bc_crop.prepare import TEXT_SUFFIXES
    names = {'registration.json', 'input-manifest.json', 'root-decision.json',
             'seed-inventory.local.json', 'seed-inventory.json', 'seeds.json'}
    paths = [('public/'+n, PUBLIC/n) for n in git('ls-files', '-z').decode().split('\0') if n and Path(n).suffix.lower() in TEXT_SUFFIXES]
    paths += [('local/'+p.relative_to(ROOT/'artifacts').as_posix(), p)
              for p in sorted((ROOT/'artifacts').rglob('*.json'))
              if not p.is_relative_to(out) and (p.name.lower() in names or p.parent.name == 'job-inputs')]
    seen, identities, total = set(), [], 0
    pattern = re.compile(r'(?<!\d)\d{1,10}(?!\d)')
    for label, path in paths:
        deadline_check()
        total += path.stat().st_size
        if total > 256*MIB: raise ValueError('Seed inventory budget exceeded')
        data = path.read_bytes()
        seen.update(int(m.group()) for m in pattern.finditer(data.decode('utf-8-sig')) if int(m.group()) < 2**32)
        identities.append([label, hashlib.sha256(data).hexdigest()])
    return dict(test_seeds=select_seeds(seen), excluded_uint32=sorted(seen | EXCLUDED), identities=identities,
                input_bytes=total, scope='Tracked public text and local registration, input manifests, seed inventories and job-input metadata',
                limitation='Deleted, unavailable, external and unlisted history cannot be searched')


def _copy_new(source, target, expected):
    if sha(source) != expected: raise ValueError('Source identity changed')
    target.parent.mkdir(parents=True, exist_ok=True)
    with source.open('rb') as src, target.open('xb') as dst:
        for block in iter(lambda: src.read(MIB), b''): dst.write(block)
    if sha(target) != expected: raise ValueError('Copy identity changed')


def prepare(out=OUT):
    out = Path(out).resolve()
    if os.name != 'nt': raise ValueError('Registration requires the Windows official runtime')
    if out != OUT or (out/'registration.json').exists() or (out/'freeze.json').exists():
        raise ValueError('Registration is exclusive')
    deadline_check()
    decision = read(out/'root-decision.json')
    if decision.get('previous_commit') != BASE or decision.get('status') != 'approved_for_implementation_not_execution':
        raise ValueError('Invalid root decision')
    clean_head(BASE)
    names = ('root-result-acceptance.json', 'registration.json', 'freeze.json', 'run-started.json',
             'worker-result.json', 'run-result.json', 'independent-result-review.json')
    for name in names:
        if git('show', BASE+':review/evidence/'+PRECISION.name+'/'+name) != (PRECISION/name).read_bytes():
            raise ValueError('Published precision evidence changed')
    precision = read(PRECISION/'root-result-acceptance.json')
    precision_cfg = read(PRECISION/'registration.json')
    precision_freeze = read(PRECISION/'freeze.json')
    if (not precision['verification_passed'] or not precision['diagnostic_execution_completed']
            or not precision['highest']['parity_passed'] or precision['default']['parity_passed']
            or precision['highest']['candidate_rows'] != 1741
            or sha(PRECISION/'freeze.json') != precision['freeze_sha256']
            or sha(PRECISION/'registration.json') != precision_freeze['registration_sha256']):
        raise ValueError('Precision evidence invalid')
    previous_cfg, previous_freeze = read(PREVIOUS/'registration.json'), read(PREVIOUS/'freeze.json')
    if (sha(PREVIOUS/'registration.json') != previous_freeze['registration_sha256']
            or sha(PREVIOUS/'registration.json') != precision_cfg['prior_registration_sha256']
            or sha(PREVIOUS/'freeze.json') != precision_cfg['prior_freeze_sha256']
            or sha(PREVIOUS/'root-result-acceptance.json') != precision_cfg['prior_acceptance_sha256']):
        raise ValueError('Previous registration changed')
    model, cache = PREVIOUS/'private/model-final.npz', PREVIOUS/'private/train.npz'
    if (sha(model) != precision['checkpoint_sha256'] or sha(cache) != precision['training_cache_sha256']
            or sha(model) != precision_cfg['checkpoint_sha256'] or sha(cache) != precision_cfg['training_cache_sha256']):
        raise ValueError('Previous model or cache identity changed')
    bindings = {'train-data-result.json': precision_cfg['prior_training_data_result_sha256'],
                'schema.json': precision_cfg['schema_sha256'],
                'gpu-launch.local.json': precision_cfg['private_gpu_launch_sha256']}
    if (read(PREVIOUS/'train-data-result.json')['array_sha256'] != precision['training_cache_sha256']
            or not set(previous_cfg['test_seeds']).issubset(EXCLUDED)):
        raise ValueError('Prior training data or reserved test seeds differ')
    runtime = dict(precision_freeze['runtime_files'])
    runtime.update(precision_freeze['protected_source_sha256'])
    source = {p.relative_to(ROOT).as_posix(): sha(p) for p in sorted(SOURCE.glob('*.py'))}
    runtime.update(source)
    for rel, digest in runtime.items():
        if sha(ROOT/rel) != digest: raise ValueError('Runtime identity changed')
    from learning.task_entry_repair_v1.experiment import installed_files
    from learning.task_reward_gate_v1.experiment import versions
    installed = installed_files()
    if (versions() != previous_cfg['runtime_versions'] or sha(installed['rules']) != previous_cfg['engine_sha256']
        or sha(installed['loader']) != previous_cfg['official_loader_sha256']
        or sha(ROOT/previous_cfg['crop_checkpoint']) != previous_cfg['crop_checkpoint_sha256']):
        raise ValueError('Environment changed')
    inventory = seed_inventory(out)
    if set(inventory['test_seeds']) & (set(previous_cfg['train_seeds']) | EXCLUDED):
        raise ValueError('Seed isolation failed')
    for name, digest in bindings.items(): _copy_new(PREVIOUS/name, out/name, digest)
    for name, digest in previous_cfg['source_sha256'].items():
        _copy_new(PREVIOUS/'sources'/name, out/'sources'/name, digest)
    (out/'private').mkdir(parents=True, exist_ok=True)
    write_new(out/'seed-inventory.local.json', inventory)
    cfg = dict(previous_cfg)
    cfg.update(experiment='task-completion-bc-reentry-v1', baseline_head=BASE,
        created_utc=datetime.now(timezone.utc).isoformat(), task_started_utc=decision['started_utc'],
        task_deadline_utc=decision['deadline_utc'], total_stage_seconds=decision['total_stage_seconds'],
        seed_generation='SHA256 task-completion-bc-reentry-v1:20261027:<index>, first uint32 not in inventory',
        deadline_utc=decision['deadline_utc'], train_seeds=previous_cfg['train_seeds'], test_seeds=inventory['test_seeds'],
        seed_inventory_sha256=sha(out/'seed-inventory.local.json'), seed_scope=inventory['scope'], seed_limitation=inventory['limitation'],
        training=previous_cfg['training'], evaluation=previous_cfg['evaluation'], data_limits=previous_cfg['data_limits'],
        game_limits=previous_cfg['game_limits'], storage=previous_cfg['storage'], source_sha256=previous_cfg['source_sha256'],
        opponents=previous_cfg['opponents'], seats=previous_cfg['seats'], policy=previous_cfg['policy'],
        engine_sha256=previous_cfg['engine_sha256'], official_loader_sha256=previous_cfg['official_loader_sha256'],
        runtime_versions=previous_cfg['runtime_versions'], crop_checkpoint=previous_cfg['crop_checkpoint'],
        crop_checkpoint_sha256=previous_cfg['crop_checkpoint_sha256'],
        training_cache='artifacts/task-completion-bc-v1-20261027/private/train.npz', training_cache_sha256=sha(cache),
        old_model_sha256=sha(model), copied_files=bindings, precision_evidence={n:sha(PRECISION/n) for n in names},
        prior_registration_sha256=sha(PREVIOUS/'registration.json'), prior_freeze_sha256=sha(PREVIOUS/'freeze.json'),
        max_fits=1, max_updates=128, inference_matmul_precision='highest', training_precision='default',
        checkpoint_selection='final update only', old_failure_not_refunded=True, old_holdouts_not_reused=True,
        model_must_be_sealed_before_test_read=True, failure_progress_receipt=True,
        official_limits=previous_cfg['official_limits'],
        retry_allowed=False)
    write_new(out/'registration.json', cfg)
    write_new(out/'freeze.json', dict(registration_sha256=sha(out/'registration.json'),
        root_decision_sha256=sha(out/'root-decision.json'), runtime_files=runtime, new_source_sha256=source,
        precision_commit=BASE))
    return verify(out)


def verify(out=OUT):
    out = Path(out).resolve()
    if out != OUT: raise ValueError('Unexpected experiment root')
    deadline_check()
    receipt = out/'pre-execution-commit.json'
    clean_head(read(receipt)['commit'] if receipt.exists() else BASE)
    cfg, freeze = read(out/'registration.json'), read(out/'freeze.json')
    if sha(out/'registration.json') != freeze['registration_sha256'] or sha(out/'root-decision.json') != freeze['root_decision_sha256']:
        raise ValueError('Registration or decision changed')
    if {p.relative_to(ROOT).as_posix() for p in SOURCE.glob('*.py')} != set(freeze['new_source_sha256']):
        raise ValueError('Source set changed')
    for rel, digest in freeze['runtime_files'].items():
        if sha(ROOT/rel) != digest: raise ValueError('Runtime changed')
    for name, digest in cfg['precision_evidence'].items():
        if sha(PRECISION/name) != digest: raise ValueError('Precision evidence changed')
    for name, digest in cfg['copied_files'].items():
        if sha(out/name) != digest or sha(PREVIOUS/name) != digest: raise ValueError('Copied input changed')
    for name, digest in cfg['source_sha256'].items():
        if sha(out/'sources'/name) != digest: raise ValueError('Rival source changed')
    for path, digest in ((PREVIOUS/'registration.json', cfg['prior_registration_sha256']),
        (PREVIOUS/'freeze.json', cfg['prior_freeze_sha256']), (PREVIOUS/'private/model-final.npz', cfg['old_model_sha256']),
        (ROOT/cfg['training_cache'], cfg['training_cache_sha256']), (out/'seed-inventory.local.json', cfg['seed_inventory_sha256'])):
        if sha(path) != digest: raise ValueError('Bound input changed')
    seeds = cfg['test_seeds']
    if len(seeds) != 2 or len(set(seeds)) != 2 or set(seeds) & (set(cfg['train_seeds']) | EXCLUDED):
        raise ValueError('Seed isolation changed')
    if sha(ROOT/cfg['crop_checkpoint']) != cfg['crop_checkpoint_sha256']:
        raise ValueError('Frozen crop checkpoint changed')
    if os.name == 'nt':
        from learning.task_entry_repair_v1.experiment import installed_files
        from learning.task_reward_gate_v1.experiment import versions
        installed = installed_files()
        if (versions() != cfg['runtime_versions'] or sha(installed['rules']) != cfg['engine_sha256']
                or sha(installed['loader']) != cfg['official_loader_sha256']):
            raise ValueError('Windows official environment changed')
    return cfg


def ready(out=OUT):
    cfg = verify(out)
    review, receipt = read(Path(out)/'pre-review.json'), read(Path(out)/'pre-execution-commit.json')
    if (not review['passed'] or review['freeze_sha256'] != sha(Path(out)/'freeze.json')
        or receipt['freeze_sha256'] != sha(Path(out)/'freeze.json')
        or receipt['registration_sha256'] != sha(Path(out)/'registration.json')):
        raise ValueError('Review or commitment changed')
    verify_commit(out)
    return cfg


def verify_commit(out=OUT):
    out = Path(out).resolve()
    cfg = verify(out)
    receipt = read(out/'pre-execution-commit.json')
    commit = receipt['commit']
    clean_head(commit)
    if git('merge-base', '--is-ancestor', BASE, commit) != b'': raise ValueError('Invalid commit ancestry')
    for name in ('root-decision.json', 'registration.json', 'freeze.json', 'pre-review.json', 'schema.json'):
        if git('show', commit+':review/evidence/'+out.name+'/'+name) != (out/name).read_bytes():
            raise ValueError('Committed registration changed')
    for rel in read(out/'freeze.json')['new_source_sha256']:
        if git('show', commit+':kaggriculture/'+rel) != (ROOT/rel).read_bytes():
            raise ValueError('Committed source changed')
    return cfg


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=('prepare', 'verify', 'ready', 'verify_commit'))
    args = parser.parse_args()
    result = {'prepare': prepare, 'verify': verify, 'ready': ready, 'verify_commit': verify_commit}[args.command]()
    print(json.dumps(dict(experiment=result['experiment'], passed=True)))
