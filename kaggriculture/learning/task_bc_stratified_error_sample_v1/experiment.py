"""Pre-register source selection without reading any original data stream."""
import argparse
from datetime import datetime, timezone
import hashlib
from pathlib import Path
import subprocess
import sys

from learning.bc_crop.agent import ROOT, sha
from learning.bc_crop.experiment import read
from learning.task_entry_repair_v1.experiment import installed_files
from learning.task_bc_diagnostics_capacity_v1.experiment import public_sources
from learning.task_bc_diagnostics_capacity_v1.runner import new_json
from learning.task_bc_diagnostics_capacity_exit_v1.runner import budget_rows as old_budget

PUBLIC = ROOT.parent / 'publication/ai-game-lab'
OLD = ROOT / 'artifacts/task-bc-diagnostics-capacity-exit-v1-20261023'
PACKAGE = Path(__file__).resolve().parent
BASELINE = 'e417a4f3861f45000e92e67a08d77ab448dfed7c'
STAGE = 'task-bc-stratified-error-sample-v1'


def git(*args):
    return subprocess.check_output(['git', '-C', str(PUBLIC), *args])


def stat(path):
    value = Path(path).stat()
    return dict(size=value.st_size, mtime_ns=value.st_mtime_ns)


def prepare(out):
    out = Path(out).resolve()
    assert out.parent == ROOT / 'artifacts' and not out.exists()
    assert git('rev-parse','HEAD').decode().strip() == BASELINE and not git('status','--porcelain').strip()
    prev_cfg = read(OLD / 'registration.json'); prev = read(OLD / 'summary.json')
    release = read(OLD / 'publication-verification.json'); decision = read(OLD / 'root-next-stage-decision.json')
    assert prev['passed'] and release['passed'] and release['commit'] == BASELINE
    assert decision['recommended_stage'] == STAGE and decision['status'] == 'planned_not_started'
    assert read(OLD / 'evidence-review.json')['passed'] and read(OLD / 'root-evidence-acceptance.json')['passed']
    assert sha(OLD / 'budget-reservations.jsonl') == prev['budget_ledger_sha256']
    prior_rows = old_budget(OLD, prev_cfg)
    assert len(prior_rows) == 14 and sum(r['groups'] for r in prior_rows) + prev_cfg['old_budget_groups'] == 2816
    assert sum(r['candidates'] for r in prior_rows) + prev_cfg['old_budget_candidates'] == 162250
    fixed = read(OLD / 'freeze.json')['runtime_files']
    for rel,digest in fixed.items(): assert sha(ROOT / rel) == digest, rel
    selection = decision['source_selection']; source = ROOT / selection['source_root']
    source_cfg = read(source / 'registration.json'); manifest = read(source / 'dataset-manifest.json')
    ledger = read(source / 'reconstruction-ledger.json')
    assert sha(source / 'dataset-manifest.json') == selection['source_manifest_sha256']
    assert sha(source / 'reconstruction-ledger.json') == selection['source_reconstruction_ledger_sha256']
    chosen = {}
    for split,key in (('train','train_seeds'),('validation','validation_seeds')):
        for i,seed in enumerate(source_cfg[key]):
            for side in (0,1):
                opponent = ['v1','market','round2'][(i+side)%3]
                matches = [r for r in ledger['rows'] if (r['split'],r['seed'],r['side'],r['opponent']) == (split,seed,side,opponent)]
                assert len(matches) == 1
                row = matches[0]; chosen[row['game']] = row
    groups = []
    for entry in manifest['files']:
        if entry['game'] not in chosen: continue
        row = chosen[entry['game']]
        assert entry['files'] == row['files'] and entry['split'] == row['split']
        group = {k:row[k] for k in ('game','split','seed','side','opponent')}
        group['source_files'] = {}
        for kind,info in entry['files'].items():
            path = (source / info['path']).resolve()
            assert path.is_relative_to(source / 'data') and info['rows'] == row['decisions']
            group['source_files'][kind] = dict(info,bytes=path.stat().st_size)
        group['source_sha256'] = {k:v['sha256'] for k,v in group['source_files'].items()}
        group['source_rows'] = row['decisions']; groups.append(group)
    assert len(groups) == 20 and sum(g['source_rows'] for g in groups) == 45121
    assert sum(f['bytes'] for g in groups for f in g['source_files'].values()) == 121152135

    # Preserve the inherited hash ledger without re-reading disallowed originals.
    prior_protection = read(OLD / 'protection-before.json')
    inherited = prior_protection['runtime'].copy()
    for base in (OLD, ROOT / 'learning/task_bc_diagnostics_capacity_exit_v1'):
        for path in base.rglob('*'):
            if path.is_file() and '__pycache__' not in path.parts:
                inherited[path.relative_to(ROOT).as_posix()] = sha(path)
    old_report = ROOT / 'TASK_DIAGNOSTICS_CAPACITY_EXIT.zh-CN.md'
    inherited[old_report.relative_to(ROOT).as_posix()] = sha(old_report)
    protected = dict(inherited_runtime_sha256=inherited,
        inherited_protection_sha256=sha(OLD / 'protection-before.json'),
        inherited_evidence_review_sha256=sha(OLD / 'evidence-review.json'),
        stat_before={rel:stat(ROOT/rel) for rel in inherited},
        public={rel:sha(PUBLIC/rel) for rel in git('ls-files','-z').decode().split('\0') if rel},
        method='Inherited historical hashes retained. Current source/metadata/model allowlist and the predecessor exit stage derived artifacts are freshly byte-hashed; its compact/pickle/logs are not decoded. Other inherited historical files use before/after stat, not fresh byte-proof. Selected60 original streams are hashed once only inside charged collection.',
        predecessor_derived_artifacts_hashed_without_decoding=True)
    important = ['registration.json','freeze.json','summary.json','budget-reservations.jsonl','evidence-review.json',
        'root-evidence-acceptance.json','next-stage-advice.json','root-next-stage-decision.json','publication-review.json',
        'publication-verification.json']
    official = installed_files()
    assert sha(official['rules']) == prev_cfg['engine_sha256'] and sha(official['loader']) == prev_cfg['official_loader_sha256']
    assert sha(ROOT / prev_cfg['task_model']) == prev_cfg['task_model_sha256']
    out.mkdir(parents=True)
    new_json(out / 'protection-before.json',protected)
    cfg = dict(schema='task-stratified-diagnostic-registration-v1',stage=STAGE,
        created_utc=datetime.now(timezone.utc).isoformat(),baseline_head=BASELINE,
        local_registration_not_external_timestamp=True,sample_only=True,historical_reuse_not_new_holdout=True,
        old_stage=OLD.relative_to(ROOT).as_posix(),old_artifacts_sha256={f:sha(OLD/f) for f in important},
        protection_sha256=sha(out/'protection-before.json'),fixed_runtime_sources_sha256=fixed,
        source_root=selection['source_root'],source_metadata_sha256={f:sha(source/f) for f in ('registration.json','dataset-manifest.json','reconstruction-ledger.json')},
        selected_groups=groups,registered_groups=[{k:g[k] for k in ('game','split','seed','side','opponent','source_sha256')} for g in groups],
        source_selection=selection,old_budget_groups=2816,old_budget_candidates=162250,
        new_evaluated_groups_max=1200,new_evaluated_candidates_max=120000,
        shared_evaluated_groups_max=5120,shared_evaluated_candidates_max=320000,
        max_cumulative_groups=4016,max_cumulative_candidates=282250,
        sample_groups_max=600,sample_train_groups_max=480,sample_validation_groups_max=120,sample_candidates_max=60000,
        source_scan_limits=dict(compressed_source_bytes_hashed=121152135,index_metadata_rows=45121,label_rows=45121,
            input_physical_lines_read=45121,all_original_gzip_physical_lines=135363,input_json_rows_decoded=600,
            input_json_candidates_decoded=145200,decompressed_source_bytes=4294967296),
        maximum_source_json_line_bytes=16777216,maximum_original_candidates_per_row=242,
        max_derived_decoded_groups=2160,max_derived_decoded_candidates=240000,
        synthetic_limits=dict(cases=12,required_cases=12,rows=64,candidates=256,
            counting='Count every numerical input JSON decode, including test readback, plus every pipeline evaluate conservatively again. Fixture construction and metadata-only operations are reported separately, not model evaluations.'),
        task_model=prev_cfg['task_model'],task_model_sha256=prev_cfg['task_model_sha256'],
        engine_commit=prev_cfg['engine_commit'],engine_sha256=prev_cfg['engine_sha256'],official_loader_sha256=prev_cfg['official_loader_sha256'],
        engine_provenance='Pinned official source exact rule hash verified in predecessor; installed rule and loader bytes rechecked now.',
        python_version=sys.version.split()[0],dependency_versions=prev_cfg['dependency_versions'],
        stage_wall_seconds_max=900.0,numeric_job_work_seconds_max=120.0,close_seconds_max_including_grace=10.0,
        normal_zero_exit_grace_seconds_max=2.0,max_workers=1,blas_threads=1,resource_limits=prev_cfg['resource_limits'],
        clock='One continuous clock starts before the first tiny synthetic job; includes all subsequent source hash/read, extraction, both score paths, reporting, waiting and cleanup. Static authoring/metadata preregistration precedes it.',
        acceptance=['All focused synthetic checks pass within the shared tiny-fixture allowance.',
            'Exact twenty registered source identities, once per original hash/stream; full original-order checks and immutable sparse membership before predictions.',
            'One reference and one shard pass only, all selected train support complete, all compact rows and split summaries exact; every budget/receipt reconciles.',
            'Conditional coverage and disagreement statistics report zero cells, sample denominators and sample-relative support; informativeness may be inconclusive.',
            'No forced successful worker, deadline or resource breach; independent review and privacy pass.'],
        stop='Any failure stops dispatch. No retry/resume/refill, sample truncation, extra model call or deadline extension.',
        reporting=dict(marginals=['split','seed','phase','teacher_class','opponent','side'],interaction=['phase','teacher_class'],
            witness_max=5,witness_selection='Per teacher class, first candidate mismatch in original manifest then physical-row order; stored compact only.',
            no_population_estimate=True,no_reward_causal_claim=True,support_is_sample_relative=True),
        policy_calls=0,training_runs=0,new_games=0,official_transitions=0,full_history_processing=False,
        original_diagnostic_stage_acceptance=False,original_policy_admission=False,
        publication='Only source, relative hashes, aggregate counts/statistics and reviews. Raw samples, compact rows, witnesses, indexes, model weights, native process identities and logs stay local.')
    new_json(out/'registration.json',cfg)
    return cfg


def freeze(out):
    out = Path(out).resolve(); cfg = read(out/'registration.json'); files = cfg['fixed_runtime_sources_sha256'].copy()
    for path in PACKAGE.iterdir():
        if path.is_file() and path.suffix in ('.py','.md'): files[path.relative_to(ROOT).as_posix()] = sha(path)
    published,mapping = public_sources(cfg,files)
    new_json(out/'freeze.json',dict(runtime_files=files,public_source_sha256=published,line_ending_mappings=mapping,
        registration_sha256=sha(out/'registration.json'),protection_sha256=sha(out/'protection-before.json')))
    return verify(out)


def verify(out):
    out=Path(out).resolve(); cfg=read(out/'registration.json'); f=read(out/'freeze.json')
    assert sha(out/'registration.json')==f['registration_sha256']
    assert sha(out/'protection-before.json')==f['protection_sha256']==cfg['protection_sha256']
    for rel,digest in f['runtime_files'].items(): assert sha(ROOT/rel)==digest,rel
    for rel,digest in cfg['old_artifacts_sha256'].items(): assert sha(OLD/rel)==digest,rel
    for rel,digest in cfg['source_metadata_sha256'].items(): assert sha(ROOT/cfg['source_root']/rel)==digest,rel
    assert sha(ROOT/cfg['task_model'])==cfg['task_model_sha256']
    official=installed_files()
    assert sha(official['rules'])==cfg['engine_sha256'] and sha(official['loader'])==cfg['official_loader_sha256']
    assert cfg['policy_calls']==cfg['training_runs']==cfg['new_games']==cfg['official_transitions']==0
    return cfg


def ready(out):
    out=Path(out).resolve(); cfg=verify(out); r=read(out/'pre-review.json')
    assert r['passed'] and r['freeze_sha256']==sha(out/'freeze.json')
    commit=read(out/'pre-execution-commit.json')['commit']
    for name in ('registration.json','freeze.json','pre-review.json'):
        assert git('show',commit+':review/evidence/'+out.name+'/'+name)==(out/name).read_bytes(),name
    for rel,digest in read(out/'freeze.json')['public_source_sha256'].items():
        assert hashlib.sha256(git('show',commit+':kaggriculture/'+rel)).hexdigest()==digest,rel
    return cfg


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('mode',choices=('prepare','freeze','verify'))
    parser.add_argument('--out',type=Path,required=True); args=parser.parse_args()
    result={'prepare':prepare,'freeze':freeze,'verify':verify}[args.mode](args.out)
    print(dict(passed=True,mode=args.mode,stage=result['stage']))
