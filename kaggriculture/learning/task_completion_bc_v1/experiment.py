"""Prospective, seed-isolated completion-filtered task-choice probe."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[2]
PUBLIC = ROOT.parent / 'publication/ai-game-lab'
OUT = ROOT / 'artifacts/task-completion-bc-v1-20261027'
SOURCE = Path(__file__).resolve().parent
PARENT = ROOT / 'artifacts/task-completion-evidence-v1-20261027'
ACTION = ROOT / 'artifacts/task-action-observation-v1-20261027'
ANCHOR = ROOT / 'artifacts/task-anchor-reentry-v1-20261027'
BASE = 'e79a53c57dd929a19180692100d2eef41ad21f1b'
START = '2026-10-27T05:25:00+00:00'
DEADLINE = '2026-10-27T08:25:00+00:00'
MIB = 1024**2
GIB = 1024**3
TRAIN_SEEDS = (3911611436, 2328454240)
SCHEMA_PATH = ROOT / 'artifacts/task-dagger-once-v1-20261027/cache/schema.json'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(MIB), b''):
            h.update(block)
    return h.hexdigest()


def write_new(path, value):
    path = Path(path).resolve()
    if not path.is_relative_to(OUT):
        raise ValueError('Output outside owned experiment')
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8', newline='\n') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')


def deadline_check():
    if datetime.now(timezone.utc) >= datetime.fromisoformat(DEADLINE):
        raise TimeoutError('Stage deadline reached')


def git(*args):
    return subprocess.check_output(['git', *args], cwd=PUBLIC)


def seed_inventory():
    """Read metadata only. Existing raw replays/traces and tensors are excluded."""
    from learning.bc_crop.prepare import TEXT_SUFFIXES
    paths = [('public/' + name, PUBLIC / name) for name in
             git('ls-files', '-z').decode().split('\0')[:-1]
             if Path(name).suffix.lower() in TEXT_SUFFIXES]
    allowed_names = {'registration.json', 'input-manifest.json', 'root-decision.json',
                     'seed-inventory.local.json', 'seed-inventory.json', 'seeds.json'}
    paths += [('local/' + p.relative_to(ROOT/'artifacts').as_posix(), p)
              for p in sorted((ROOT/'artifacts').rglob('*.json'))
              if not p.is_relative_to(OUT) and
              (p.name.lower() in allowed_names or p.parent.name == 'job-inputs')]
    seen, identities, used = set(), [], 0
    pattern = re.compile(r'(?<!\d)\d{1,10}(?!\d)')
    for label, path in paths:
        deadline_check()
        used += path.stat().st_size
        if used > 256*MIB:
            raise ValueError('Metadata-only seed inventory exceeds budget')
        data = path.read_bytes()
        seen.update(int(m.group()) for m in pattern.finditer(data.decode('utf-8-sig'))
                    if int(m.group()) < 2**32)
        identities.append([label, hashlib.sha256(data).hexdigest()])
    selected, draws = [], 0
    while len(selected) < 2:
        digest = hashlib.sha256(('task-completion-bc-v1:20261027:' + str(draws)).encode('ascii')).digest()
        value = int.from_bytes(digest[:4], 'big')
        draws += 1
        if value not in seen and value not in selected and value not in TRAIN_SEEDS:
            selected.append(value)
    return dict(test_seeds=selected, draws=draws, input_bytes=used, identities=identities,
                excluded_uint32=sorted(seen),
                scope='Tracked public text plus local preregistration, seed inventory, input manifest and job-input metadata; no generic configuration, result, replay, trace or tensors',
                limitation='Cannot exclude deleted, unavailable, external or locally unlisted seed history')


def validate_historical_report(game, part, report):
    if (report['game'] != game['key'] or report['side'] != game['side']
            or report['labels_sha256'] != part['sha256'] or report['rows'] != part['rows']
            or report['replay_sha256'] != game['replay']['sha256']
            or report['trace_sha256'] != game['trace']['sha256']):
        raise ValueError('Historical labels differ from the published game commitment')


def prepare(out=OUT):
    out = Path(out).resolve()
    deadline_check()
    if out != OUT or (out/'registration.json').exists():
        raise ValueError('Registration is exclusive')
    if git('rev-parse', 'HEAD').decode().strip() != BASE or git('status', '--porcelain').strip():
        raise ValueError('Unexpected public baseline or existing changes')
    decision = read(out/'root-decision.json')
    if decision.get('status') != 'approved_for_implementation_not_execution':
        raise ValueError('Missing prospective root decision')
    parent = read(PARENT/'root-result-acceptance.json')
    if not parent['passed'] or not parent['feasibility_gate_met']:
        raise ValueError('Predecessor evidence not accepted')
    evidence_prefix = 'review/evidence/' + PARENT.name + '/'
    for name in ('root-result-acceptance.json', 'freeze.json', 'input-manifest.json',
                 'analysis-result.json', 'independent-result-review.json', 'game-ledger.json'):
        if git('show', BASE + ':' + evidence_prefix + name) != (PARENT/name).read_bytes():
            raise ValueError('Accepted predecessor evidence differs from public commit')
    old_freeze = read(PARENT/'freeze.json')
    if sha(PARENT/'freeze.json') != parent['freeze_sha256']:
        raise ValueError('Predecessor frozen source identity changed')
    old_cfg = read(ACTION/'registration.json')
    previous_manifest = read(PARENT/'input-manifest.json')
    previous_dataset = read(PARENT/'dataset-manifest.json')
    action_manifest = read(ACTION/'input-manifest.json')
    if (sha(ACTION/'input-manifest.json') != old_freeze['parent_manifest_sha256']
            or sha(ACTION/'ledger.json') != action_manifest['ledger_sha256']
            or sha(ACTION/'audit.json') != action_manifest['audit_sha256']):
        raise ValueError('Actor reconstruction evidence changed')
    action_audits = {row['game']: row for row in read(ACTION/'audit.json')}
    if set(g['seed'] for g in previous_manifest['games']) != set(TRAIN_SEEDS):
        raise ValueError('Unexpected development seed source')
    if len(previous_manifest['games']) != 8 or len(previous_dataset['games']) != 8 or not previous_dataset['complete']:
        raise ValueError('Unexpected accepted source population')
    by_game = {p['game']: p for p in previous_dataset['games']}
    source_groups = []
    for game in previous_manifest['games']:
        part = by_game[game['key']]
        report_name = 'game-results/' + game['key'] + '.json'
        if git('show', BASE + ':' + evidence_prefix + report_name) != (PARENT/report_name).read_bytes():
            raise ValueError('Accepted historical game report changed')
        report = read(PARENT/report_name)
        validate_historical_report(game, part, report)
        audit = action_audits[game['key']]
        if (not audit['passed'] or audit['replay_sha256'] != game['replay']['sha256']
                or audit['trace_sha256'] != game['trace']['sha256']
                or audit['actor_actions_reproduced'] != 719 or audit['rival_actions_reproduced'] != 719
                or audit['trace_turns_reproduced'] != 719 or audit['errors'] != 0):
            raise ValueError('Historical actor trace reconstruction is not bound')
        # Hashing the already-created label file binds it without decoding rows.
        label_path = (PARENT/part['path']).resolve()
        if not label_path.is_relative_to(PARENT/'private/games') or sha(label_path) != part['sha256']:
            raise ValueError('Historical label identity changed')
        source_groups.append(dict(key=game['key'], seed=game['seed'], side=game['side'],
                                  opponent=game['opponent'], arm=game['arm'],
                                  trace=game['trace'], replay=game['replay'],
                                  labels_path=part['path'], labels_sha256=part['sha256'],
                                  labels_rows=part['rows'], accepted_report_sha256=sha(PARENT/report_name)))
    if len({g['key'] for g in source_groups}) != 8:
        raise ValueError('Aliased historical inputs')
    schema = read(SCHEMA_PATH)
    other = read(ROOT/'artifacts/task-bc-v1-20261023-run1/cache/schema.json')
    if schema != other or schema['schema_sha256'] != '235eef551a799e7dc61e8b25448f9b7f6215bf6f97d044bf84b6edf9bd91ea56':
        raise ValueError('Trusted task schema changed')
    runtime = dict(old_freeze['runtime_files'])
    runtime.update(old_freeze['new_source_sha256'])
    new_source = {p.relative_to(ROOT).as_posix(): sha(p) for p in sorted(SOURCE.glob('*.py'))}
    runtime.update(new_source)
    for rel, expected in runtime.items():
        if sha(ROOT/rel) != expected:
            raise ValueError('Frozen runtime changed: ' + rel)
    for rel, expected in old_freeze['protected_source_sha256'].items():
        if sha(ROOT/rel) != expected:
            raise ValueError('Protected source changed')
    from learning.task_entry_repair_v1.experiment import installed_files
    from learning.task_reward_gate_v1.experiment import versions
    installed = installed_files()
    if (versions() != old_cfg['runtime_versions'] or sha(installed['rules']) != old_cfg['engine_sha256']
            or sha(installed['loader']) != old_cfg['official_loader_sha256']):
        raise ValueError('Official environment changed')
    crop = ROOT/old_cfg['crop_checkpoint']
    if sha(crop) != old_cfg['crop_checkpoint_sha256']:
        raise ValueError('Frozen crop checkpoint changed')
    if shutil.disk_usage(ROOT).free < 8*GIB:
        raise ValueError('Insufficient reserved space')
    inventory = seed_inventory()
    write_new(out/'seed-inventory.local.json', inventory)
    for name, expected in old_cfg['source_sha256'].items():
        source = ACTION/'sources'/name
        if sha(source) != expected:
            raise ValueError('Frozen rival changed')
        target = out/'sources'/name
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open('xb') as f:
            f.write(source.read_bytes())
    launch = read(ANCHOR/'gpu-launch.local.json')
    previous_device = read(ANCHOR/'model/device-local.json')
    if (previous_device['platform'] != 'gpu'
            or launch['expected_device_substring'] not in previous_device['device_kind']):
        raise ValueError('Previous GPU identity evidence is inconsistent')
    launch['expected_device'] = previous_device['device']
    launch['expected_device_kind'] = previous_device['device_kind']
    write_new(out/'gpu-launch.local.json', launch)
    write_new(out/'schema.json', schema)
    cfg = dict(experiment='task-completion-bc-v1', baseline_head=BASE,
        created_utc=datetime.now(timezone.utc).isoformat(), task_started_utc=START,
        task_deadline_utc=DEADLINE, total_stage_seconds=10800,
        train_seeds=list(TRAIN_SEEDS), test_seeds=inventory['test_seeds'],
        seed_generation='First four SHA256 bytes of task-completion-bc-v1:20261027:index; reject existing visible metadata and duplicates',
        seed_inventory_sha256=sha(out/'seed-inventory.local.json'), seed_inventory_bytes=inventory['input_bytes'],
        seed_scope=inventory['scope'], seed_history_limitation=inventory['limitation'],
        source_groups=source_groups, parent_acceptance_sha256=sha(PARENT/'root-result-acceptance.json'),
        train_population='Eight prior games: two baseline, four candidate, two teacher_task; actual chosen from each, with a common frozen opportunity controller and crop model',
        parent_manifest_sha256=sha(PARENT/'input-manifest.json'), parent_dataset_sha256=sha(PARENT/'dataset-manifest.json'),
        parent_game_ledger_sha256=sha(PARENT/'game-ledger.json'),
        previous_freeze_sha256=sha(PARENT/'freeze.json'), trusted_schema_sha256=sha(out/'schema.json'),
        action_manifest_sha256=sha(ACTION/'input-manifest.json'),
        action_audit_sha256=sha(ACTION/'audit.json'), action_ledger_sha256=sha(ACTION/'ledger.json'),
        gpu_launch_config_sha256=sha(out/'gpu-launch.local.json'),
        engine_commit=old_cfg['engine_commit'], engine_sha256=old_cfg['engine_sha256'],
        official_loader_sha256=old_cfg['official_loader_sha256'], runtime_versions=old_cfg['runtime_versions'],
        crop_checkpoint=old_cfg['crop_checkpoint'], crop_checkpoint_sha256=old_cfg['crop_checkpoint_sha256'],
        source_sha256=old_cfg['source_sha256'], game_limits=old_cfg['game_limits'],
        opponents=['round2', 'market'], seats=[0, 1], policy='teacher_task',
        teacher_task_note='Frozen task teacher with fixed crop model, not the untouched complete original teacher',
        training=dict(updates=128, batch_groups_max=256, padded_candidates_max=65536,
                      learning_rate=0.0003, gradient_norm_clip=1.0, training_seed=2026102708,
                      synchronized_seconds=600, process_seconds=1200, initialization='fixed random',
                      checkpoint_selection='final update only', train_status='completed_observed',
                      label='actual chosen at episode origin', test_gradients=False),
        data_limits=dict(train_games=8, test_games=8, max_groups_per_split=16000,
                         max_candidates_per_split=1200000, train_compressed_bytes=512*MIB,
                         train_decoded_bytes=4*GIB, test_compressed_bytes=GIB,
                         test_decoded_bytes=8*GIB, episode_file_bytes=64*MIB),
        official_limits=dict(new_games=8, live_transitions=8*719, primary_replay_transitions=8*719,
                             separate_label_replay_transitions=8*719, independent_replay_transitions=719,
                             total_transitions=25*719, max_workers=1, per_job_seconds=900),
        rationale_extra_label_replays='Separate label replay preserves unchanged accepted official worker; disclosed before new data',
        storage=dict(total_bytes=4*GIB, game_write_bytes=128*MIB, free_floor_bytes=4*GIB,
                     process_log_bytes=MIB, small_receipt_bytes=16384),
        evaluation=dict(baseline='train completed-operation frequency with Laplace smoothing, renormalized over present ops, uniform within op; first canonical argmax',
            scope='all holdout episode origins, with completed and unknown reported separately',
            top1_gain_min=0.05, each_seed_top1_gain_positive=True, ce_not_worse=True,
            min_operations=3, min_examples_per_operation=20, operation_macro_recall_gain_min=0.05,
            cross_operation_top1_not_worse=True, multi_candidate_top1_not_worse=True,
            significance_claim=False, candidate_match_is_not_competitive_performance=True),
        independent_sample='20 lowest SHA256 identities from first test seed, round2, seat0; not selected by status',
        independent_limits=dict(compressed_bytes=128*MIB, decoded_bytes=GIB, raw_files=2,
                                replay_transitions=719, sample_count=20),
        output_privacy='Raw replays, traces, individual labels, features, logits, checkpoints, logs and local launch metadata stay local',
        no_default_or_frozen_model_changes=True, no_ppo=True, local_registration_not_third_party_timestamp=True)
    write_new(out/'registration.json', cfg)
    write_new(out/'freeze.json', dict(registration_sha256=sha(out/'registration.json'),
        root_decision_sha256=sha(out/'root-decision.json'), runtime_files=runtime,
        new_source_sha256=new_source, protected_source_sha256=old_freeze['protected_source_sha256'],
        previous_acceptance_sha256=sha(PARENT/'root-result-acceptance.json'),
        previous_independent_review_sha256=sha(PARENT/'independent-result-review.json'),
        previous_public_commit=BASE, schema_sha256=sha(out/'schema.json')))
    return verify(out)


def verify(out=OUT):
    out = Path(out).resolve()
    deadline_check()
    if out != OUT:
        raise ValueError('Unexpected experiment root')
    cfg, freeze = read(out/'registration.json'), read(out/'freeze.json')
    if sha(out/'registration.json') != freeze['registration_sha256'] or sha(out/'root-decision.json') != freeze['root_decision_sha256']:
        raise ValueError('Registration or decision changed')
    if {p.relative_to(ROOT).as_posix() for p in SOURCE.glob('*.py')} != set(freeze['new_source_sha256']):
        raise ValueError('Frozen source set changed')
    for rel, expected in freeze['runtime_files'].items():
        if sha(ROOT/rel) != expected:
            raise ValueError('Runtime changed: ' + rel)
    for rel, expected in freeze['protected_source_sha256'].items():
        if sha(ROOT/rel) != expected:
            raise ValueError('Protected source changed')
    for name, key in (('root-result-acceptance.json', 'previous_acceptance_sha256'),
                      ('independent-result-review.json', 'previous_independent_review_sha256')):
        if sha(PARENT/name) != freeze[key]:
            raise ValueError('Predecessor evidence changed')
    for path, expected in ((out/'schema.json', freeze['schema_sha256']),
                           (out/'seed-inventory.local.json', cfg['seed_inventory_sha256']),
                           (out/'gpu-launch.local.json', cfg['gpu_launch_config_sha256']),
                           (PARENT/'input-manifest.json', cfg['parent_manifest_sha256']),
                           (PARENT/'dataset-manifest.json', cfg['parent_dataset_sha256']),
                           (PARENT/'game-ledger.json', cfg['parent_game_ledger_sha256']),
                           (PARENT/'freeze.json', cfg['previous_freeze_sha256']),
                           (ACTION/'input-manifest.json', cfg['action_manifest_sha256']),
                           (ACTION/'audit.json', cfg['action_audit_sha256']),
                           (ACTION/'ledger.json', cfg['action_ledger_sha256'])):
        if sha(path) != expected:
            raise ValueError('Bound metadata changed')
    for game in cfg['source_groups']:
        if sha(PARENT/'game-results'/(game['key']+'.json')) != game['accepted_report_sha256']:
            raise ValueError('Accepted historical game report changed')
    for name, expected in cfg['source_sha256'].items():
        if sha(out/'sources'/name) != expected:
            raise ValueError('Rival source changed')
    if set(cfg['train_seeds']) & set(cfg['test_seeds']) or len(set(cfg['test_seeds'])) != 2:
        raise ValueError('Seed partition changed')
    if os.name == 'nt':
        from learning.task_entry_repair_v1.experiment import installed_files
        from learning.task_reward_gate_v1.experiment import versions
        installed = installed_files()
        if (versions() != cfg['runtime_versions'] or sha(installed['rules']) != cfg['engine_sha256']
                or sha(installed['loader']) != cfg['official_loader_sha256']
                or sha(ROOT/cfg['crop_checkpoint']) != cfg['crop_checkpoint_sha256']):
            raise ValueError('Official runtime or checkpoint changed')
    return cfg


def ready(out=OUT):
    out = Path(out).resolve()
    cfg = verify(out)
    review, receipt = read(out/'pre-review.json'), read(out/'pre-execution-commit.json')
    if not review['passed'] or review['freeze_sha256'] != sha(out/'freeze.json'):
        raise ValueError('Exact freeze lacks independent pre-review')
    if receipt['freeze_sha256'] != sha(out/'freeze.json') or receipt['registration_sha256'] != sha(out/'registration.json'):
        raise ValueError('Pre-execution commitment changed')
    return cfg


def verify_commit(out=OUT):
    out = Path(out).resolve()
    receipt = read(out/'pre-execution-commit.json')
    for name in ('registration.json', 'freeze.json', 'pre-review.json', 'root-decision.json', 'schema.json'):
        if git('show', receipt['commit'] + ':review/evidence/' + out.name + '/' + name) != (out/name).read_bytes():
            raise ValueError('Committed preregistration differs')
    for rel in read(out/'freeze.json')['new_source_sha256']:
        if git('show', receipt['commit'] + ':kaggriculture/' + rel) != (ROOT/rel).read_bytes():
            raise ValueError('Committed source differs')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=('prepare', 'verify'))
    args = parser.parse_args()
    cfg = prepare() if args.command == 'prepare' else verify()
    print(json.dumps(dict(experiment=cfg['experiment'], passed=True, test_seeds=cfg['test_seeds'])))
