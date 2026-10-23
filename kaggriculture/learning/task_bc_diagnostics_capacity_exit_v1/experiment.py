"""Register, freeze and bind the one-shot exit repair before any execution."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
from pathlib import Path
import subprocess
import sys
import urllib.request

from learning.bc_crop.agent import ROOT, sha
from learning.bc_crop.experiment import read
from learning.task_entry_repair_v1.experiment import installed_files
from learning.task_bc_diagnostics_capacity_v1.experiment import public_sources, verify as verify_old
from learning.task_bc_diagnostics_capacity_v1.runner import budget_rows, new_json

PUBLIC = ROOT.parent / 'publication/ai-game-lab'
OLD = ROOT / 'artifacts/task-bc-diagnostics-capacity-v1-20261023'
PACKAGE = Path(__file__).resolve().parent
BASELINE = '754d8ffcd50954c3a6862a7b1c001077cab78ce6'


def git(*args):
    return subprocess.check_output(['git', '-C', str(PUBLIC), *args])


def prepare(out):
    out = Path(out).resolve()
    assert not out.exists(), 'A stage may not replace or resume another attempt'
    assert git('rev-parse', 'HEAD').decode().strip() == BASELINE
    assert not git('status', '--porcelain').strip()
    old_cfg = verify_old(OLD, 'preparation')
    previous = read(OLD / 'preparation-summary.json')
    assert previous['passed'] is False and previous['owned_trees_closed'] is True
    assert not (OLD / 'formal-started.json').exists()
    rows = budget_rows(OLD)
    assert len(rows) == 7 and sum(r['groups'] for r in rows) == 1536
    assert sum(r['candidates'] for r in rows) == 88500
    assert sha(OLD / 'budget-reservations.jsonl') == previous['budget_ledger_sha256']
    manifest = read(OLD / 'sample-manifest.json')
    assert manifest['rows'] == 1280 and manifest['candidates'] == 73750
    assert manifest['train_rows'] == 1024 and manifest['validation_rows'] == 256
    assert len(manifest['groups']) == 10 and all(g['rows'] == 128 for g in manifest['groups'])
    protection = read(OLD / 'protection-before.json')['runtime'].copy()
    for rel, digest in protection.items():
        assert sha(ROOT / rel) == digest, rel
    for base in (OLD, ROOT / 'learning/task_bc_diagnostics_capacity_v1'):
        for path in base.rglob('*'):
            if path.is_file() and '__pycache__' not in path.parts:
                protection[path.relative_to(ROOT).as_posix()] = sha(path)
    report = ROOT / 'TASK_DIAGNOSTICS_CAPACITY.zh-CN.md'
    protection[report.relative_to(ROOT).as_posix()] = sha(report)
    files = [v for v in git('ls-files', '-z').decode().split('\0') if v]
    public_protection = {rel: sha(PUBLIC / rel) for rel in files}
    important = ['preparation-registration.json', 'preparation-freeze.json', 'preparation-summary.json',
        'budget-reservations.jsonl', 'sample-manifest.json', 'root-next-stage-decision.json',
        'next-stage-advice.json', 'evidence-acceptance.json', 'publication-verification.json',
        'preparation/reference/reference-receipt.json', 'preparation/reference/reference.jsonl.gz',
        'preparation/reference/summary.json']
    official = installed_files()
    old_holdout = read(ROOT / 'artifacts/task-bc-holdout-v1-20261023/registration.json')
    assert sha(official['rules']) == old_holdout['engine_sha256']
    assert sha(official['loader']) == old_holdout['official_loader_sha256']
    pinned_rule_commit = old_holdout['engine_commit']
    upstream_url = ('https://raw.githubusercontent.com/Kaggle/kaggle-environments/' +
        pinned_rule_commit + '/kaggle_environments/envs/kaggriculture/kaggriculture.py')
    with urllib.request.urlopen(upstream_url, timeout=30) as response:
        upstream_sha = hashlib.sha256(response.read()).hexdigest()
    assert upstream_sha == sha(official['rules']), 'Installed rules differ from the pinned official source'
    out.mkdir(parents=True)
    new_json(out / 'protection-before.json', dict(runtime=protection, public=public_protection))
    cfg = dict(schema='task-capacity-exit-registration-v1', stage='task-bc-diagnostics-capacity-exit-v1',
        created_utc=datetime.now(timezone.utc).isoformat(), baseline_head=BASELINE,
        local_registration_not_external_timestamp=True, sample_only=True,
        historical_reuse_not_new_holdout=True, old_stage=OLD.relative_to(ROOT).as_posix(),
        old_artifacts_sha256={rel: sha(OLD / rel) for rel in important},
        fixed_runtime_sources_sha256=read(OLD / 'preparation-freeze.json')['runtime_files'],
        protection_sha256=sha(out / 'protection-before.json'),
        sample_groups=1280, sample_candidates=73750, sample_train_groups=1024, sample_validation_groups=256,
        sample_groups_in_original_manifest_order=[{k: g[k] for k in ('game','split','rows','candidates','sha256')}
                                                for g in manifest['groups']],
        selection='Reuse exactly the immutable old10 groups, physical rows0..127; no recollection, replacement or skipping.',
        old_budget_groups=1536, old_budget_candidates=88500, old_budget_lines=7,
        new_evaluated_groups_max=1280, new_evaluated_candidates_max=73750,
        shared_evaluated_groups_max=5120, shared_evaluated_candidates_max=320000,
        planned_cumulative_groups=2816, planned_cumulative_candidates=162250,
        budget='New append-only ledger hash-binds immutable predecessor; full reservation before dispatch, failures never refunded.',
        task_model=old_cfg['task_model'], task_model_sha256=old_cfg['task_model_sha256'],
        engine_commit=pinned_rule_commit, engine_sha256=sha(official['rules']), official_loader_sha256=sha(official['loader']),
        engine_commit_evidence='Installed distribution has no Git metadata; installed rule bytes were compared with the official pinned commit over anonymous HTTPS.',
        official_rule_source_url=upstream_url, official_rule_source_sha256=upstream_sha,
        python_version=sys.version.split()[0], dependency_versions={k: importlib.metadata.version(k) for k in ('numpy','kaggle-environments')},
        normal_zero_exit_grace_seconds_max=2.0, close_seconds_max_including_grace=10.0,
        stage_wall_seconds_max=900.0, prior_plus_new_wall_seconds_max=3600.0,
        old_benchmark_wall_seconds=previous['elapsed_seconds'], numeric_job_work_seconds_max=120.0,
        lifecycle_matrix_work_seconds_max=120.0, max_workers=1, blas_threads=1,
        lifecycle_case_count=7, attempts_per_case_max=1,
        lifecycle_cases=['normal_exit','cooperative_deadline','noncooperative_descendants','controller_hard_interrupt',
                         'root_zero_natural_descendants','root_zero_stubborn_descendants','root_zero_original_deadline'],
        resource_limits=old_cfg['resource_limits'],
        acceptance=['Exactly seven registered real Windows cases meet expected outcome and prove owned-tree closure.',
                    'No successful capacity worker needs forced termination; zero root exit is insufficient alone.',
                    'Complete all sample train support before scoring the ten new shards in original manifest order.',
                    'All1280 compact rows and whole split summaries equal the immutable prior pure-function reference exactly.',
                    'Budget/source/history/privacy and independent review pass; old false outcomes stay false.'],
        stop='Any runtime, fixture, budget, hash, numeric or closure failure stops dispatch; no retry, grace tuning or new registration to relabel this run.',
        clock='Starts immediately before first real fixture dispatch; includes all waits and cleanup until terminal. Authoring, static review and exact commit occur before that marker.',
        reference_rescoring=0, extra_formal_or_independent_scoring_passes=0,
        policy_calls=0, training_runs=0, new_games=0, official_transitions=0,
        full_historical_diagnostics_prohibited=True, models_and_default_unchanged=True,
        old_capacity_stage_acceptance=False, original_diagnostic_stage_acceptance=False, original_policy_admission=False,
        publication='Source, aggregate lifecycle/measurement, relative hashes and reviews only; raw sample, weights, pickle, compact rows, logs and identities stay local.')
    new_json(out / 'registration.json', cfg)
    return cfg


def freeze(out):
    out = Path(out).resolve(); cfg = read(out / 'registration.json')
    assert not (out / 'freeze.json').exists()
    files = cfg['fixed_runtime_sources_sha256'].copy()
    for path in PACKAGE.iterdir():
        if path.is_file() and path.suffix in ('.py','.md'):
            files[path.relative_to(ROOT).as_posix()] = sha(path)
    public, mapping = public_sources(cfg, files)
    new_json(out / 'freeze.json', dict(runtime_files=files, public_source_sha256=public,
        line_ending_mappings=mapping, registration_sha256=sha(out / 'registration.json'),
        protection_sha256=sha(out / 'protection-before.json')))
    return verify(out)


def verify(out):
    out = Path(out).resolve(); cfg = read(out / 'registration.json'); f = read(out / 'freeze.json')
    assert sha(out / 'registration.json') == f['registration_sha256']
    assert sha(out / 'protection-before.json') == f['protection_sha256'] == cfg['protection_sha256']
    assert cfg['sample_only'] and cfg['historical_reuse_not_new_holdout']
    assert cfg['training_runs'] == cfg['policy_calls'] == cfg['new_games'] == cfg['official_transitions'] == 0
    for rel,digest in f['runtime_files'].items(): assert sha(ROOT / rel) == digest, rel
    for rel,digest in cfg['old_artifacts_sha256'].items(): assert sha(OLD / rel) == digest, rel
    assert sha(ROOT / cfg['task_model']) == cfg['task_model_sha256']
    official = installed_files()
    assert sha(official['rules']) == cfg['engine_sha256'] and sha(official['loader']) == cfg['official_loader_sha256']
    return cfg


def ready(out):
    out = Path(out).resolve(); cfg = verify(out)
    review = read(out / 'pre-review.json')
    assert review['passed'] is True and review['freeze_sha256'] == sha(out / 'freeze.json')
    commit = read(out / 'pre-execution-commit.json')['commit']
    for name in ('registration.json','freeze.json','pre-review.json'):
        assert git('show',commit+':review/evidence/'+out.name+'/'+name) == (out/name).read_bytes(), name
    for rel,digest in read(out / 'freeze.json')['public_source_sha256'].items():
        assert hashlib.sha256(git('show',commit+':kaggriculture/'+rel)).hexdigest() == digest, rel
    return cfg


if __name__ == '__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('mode',choices=('prepare','freeze','verify'))
    parser.add_argument('--out',type=Path,required=True); args=parser.parse_args()
    result={'prepare':prepare,'freeze':freeze,'verify':verify}[args.mode](args.out)
    print(dict(passed=True,mode=args.mode,stage=result['stage']))
