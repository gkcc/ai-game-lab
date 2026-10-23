"""Freeze exploratory definitions and preserved inputs before private analysis."""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import subprocess

from learning.bc_crop.agent import ROOT, sha
from learning.bc_crop.experiment import read, save
from learning.task_bc_holdout_v1.experiment import verify as verify_panel, PUBLIC, TASK_SHA, CROP_SHA

PANEL = ROOT/'artifacts/task-bc-holdout-v1-20261023'
DATA = ROOT/'artifacts/task-choice-v1-20261023'
PACKAGE = Path(__file__).resolve().parent


def git(*args):
    return subprocess.check_output(['git','-C',str(PUBLIC),*args])


def prepare(out):
    out=Path(out).resolve()
    if out.exists():
        assert {p.name for p in out.iterdir()} <= {'preparation'}, 'Refuse existing experiment'
    else: out.mkdir(parents=True)
    assert not git('status','--porcelain').strip()
    cfg=verify_panel(PANEL,history=True)
    assert read(PANEL/'final-review.json')['passed']
    assert read(PANEL/'acceptance.json')['execution_acceptance_passed']
    assert read(PANEL/'acceptance.json')['statistical_admission_passed'] is False
    regressions={p.name:sha(p) for p in (out/'preparation').glob('*.json')}
    assert {'shadow-regressions.json','numeric-regressions.json','numeric-regressions-v2.json',
            'analysis-regressions-v2.json','runner-regressions-v2.json'} <= set(regressions)
    assert read(out/'preparation/shadow-regressions.json')['passed']
    assert read(out/'preparation/numeric-regressions.json')['passed']
    assert read(out/'preparation/numeric-regressions-v2.json')['passed']
    assert read(out/'preparation/analysis-regressions-v2.json')['passed']
    assert read(out/'preparation/runner-regressions-v2.json')['passed']
    files=read(PANEL/'freeze.json')['runtime_files'].copy()
    for p in PACKAGE.iterdir():
        if p.is_file() and p.suffix in ('.py','.md'): files[p.relative_to(ROOT).as_posix()]=sha(p)
    protected=read(PANEL/'protection-before.json')['runtime'].copy()
    for base in (PANEL, ROOT/'learning/task_bc_holdout_v1'):
        for p in base.rglob('*'):
            if p.is_file() and '__pycache__' not in p.parts: protected[p.relative_to(ROOT).as_posix()]=sha(p)
    p=ROOT/'TASK_BC_HOLDOUT_V1.zh-CN.md';protected[p.relative_to(ROOT).as_posix()]=sha(p)
    tracked=[s for s in git('ls-files','-z').decode().split('\0') if s]
    save(out/'protection-before.json',dict(runtime=protected,public={s:sha(PUBLIC/s) for s in tracked}))
    names=['registration.json','ledger.json','audit.json','statistics.json','run-summary.json','acceptance.json',
           'independent-review.json','final-review.json','next-stage-advice.json','root-next-stage-decision.json']
    inputs={str((PANEL/n).relative_to(ROOT)).replace('\\','/'):sha(PANEL/n) for n in names}
    for n in ['dataset-manifest.json','reconstruction-ledger.json','coverage.json','registration.json']:
        inputs[(DATA/n).relative_to(ROOT).as_posix()]=sha(DATA/n)
    for row in read(PANEL/'ledger.json'):
        for p,h in [(PANEL/'games'/row['game']/'replay.json.gz',row['replay_gzip_sha256']),
                    (PANEL/'traces'/(row['game']+'.jsonl.gz'),row['trace_sha256'])]:
            assert sha(p)==h;inputs[p.relative_to(ROOT).as_posix()]=h
    for row in read(DATA/'dataset-manifest.json')['files']:
        for item in row['files'].values():
            p=DATA/item['path'];assert sha(p)==item['sha256'];inputs[p.relative_to(ROOT).as_posix()]=item['sha256']
    registration=dict(stage='task-bc-error-diagnostics-v1',created_utc=datetime.now(timezone.utc).isoformat(),
        baseline_head=git('rev-parse','HEAD').decode().strip(),post_outcome_exploratory=True,
        local_registration_not_external_timestamp=True,panel_root=PANEL.relative_to(ROOT).as_posix(),
        dataset_root=DATA.relative_to(ROOT).as_posix(),task_checkpoint=(PANEL/'model/trained.npz').relative_to(ROOT).as_posix(),
        crop_checkpoint=cfg['crop_checkpoint'],task_checkpoint_sha256=TASK_SHA,crop_checkpoint_sha256=CROP_SHA,
        old_statistical_admission_passed=False,default_changed=False,training_runs=0,new_games=0,official_transitions=0,
        seeds=cfg['test_seeds'],opponents=cfg['opponents'],origins=cfg['arms'],seats=[0,1],
        limits=dict(wall_seconds=14400,per_job_seconds=900,max_workers=1,max_source_games=96,max_source_turns=69024,
            max_shadow_policy_calls=69024,max_factual_groups=600000,max_factual_candidates=40000000,
            max_shadow_groups=600000,max_shadow_candidates=40000000,max_witnesses=12,independent_witness_calls=24,
            full_raw_processing_passes=2),
        concurrency='One owned subprocess at a time (within the approved maximum of two), so remaining global budgets are exact without shared simulation state.',
        passes='Historical train is read twice: complete train-only index first, then score all train and validation with that final index; validation once. Panel raw traces twice: inventory then factual scoring and opposite-policy shadows. Compact summaries and at-most12 targeted witness reads are not extra full raw passes.',
        precision='Original JSON numeric values to float64 for actual deployed scores and identity; separate float32 training-cache keys; preserve ordered candidate count and semantic IDs; normalize signed zero only in numeric identity. Never infer deployment alias from float32 equality alone.',
        comparison='Conditional teacher label versus fixed model on each factual callback; opposite fixed policy on identical legal observation, own within-turn reservations. Stop matching callback ordinals after first full-context divergence. Requests are not observed execution or causal reward effects.',
        summaries='Overall; origin/opponent; each seed; seat; fixed step thirds 0-239,240-479,480-718; teacher task class; singleton versus multiple candidates; cross-operation competition; train-seen64 versus unseen. No full Cartesian product. Counts and explicit denominators; null for empty. No new inferential tests.',
        witnesses='At most2 per opponent/origin stratum. First prefer earliest deployment64 alias/label-conflict witness, otherwise earliest conditional mismatch with changed full action. Second earliest distinct context/reservation divergence. Order registered seed, seat, step, decision. Missing categories remain empty; no largest-loss selection.',
        stop='Any input/schema/hash/nonfinite/factual-choice mismatch or budget/time failure stops all further dispatch; preserve failure and owned process exit; no retries, new seeds or budget expansion.',
        success='Account for all96 source games/69024 turns; both origins conditional and shadow analyses; original135406 source groups and6900116 candidates preserved; bounded reproducible witnesses; correct descriptive denominators and history/privacy. Complete inconclusive is valid; no strategy or model admission.',
        next_decision='At most one evidence-supported later correction, representation or weighting research question. Novelty, label gaps or first divergence alone do not identify profit causation or justify retraining.',
        inherited_advice_sha256=sha(PANEL/'next-stage-advice.json'),root_refinement_sha256=sha(PANEL/'root-next-stage-decision.json'),
        preparation_regressions_sha256=regressions)
    save(out/'registration.json',registration)
    save(out/'freeze.json',dict(runtime_files=files,input_files=inputs,registration_sha256=sha(out/'registration.json'),
        protection_sha256=sha(out/'protection-before.json'),new_private_analysis_absent=True))
    return verify(out,history=True)


def verify(out,history=False):
    out=Path(out);cfg=read(out/'registration.json');f=read(out/'freeze.json')
    assert cfg['post_outcome_exploratory'] and cfg['training_runs']==cfg['new_games']==cfg['official_transitions']==0
    assert cfg['old_statistical_admission_passed'] is False and not cfg['default_changed']
    assert sha(out/'registration.json')==f['registration_sha256'] and sha(out/'protection-before.json')==f['protection_sha256']
    for rel,h in f['runtime_files'].items(): assert sha(ROOT/rel)==h,rel
    for field,key in [('task_checkpoint',TASK_SHA),('crop_checkpoint',CROP_SHA)]: assert sha(ROOT/cfg[field])==key
    for n,h in cfg['preparation_regressions_sha256'].items(): assert sha(out/'preparation'/n)==h,n
    if history:
        for rel,h in read(out/'protection-before.json')['runtime'].items(): assert sha(ROOT/rel)==h,rel
        for rel,h in f['input_files'].items(): assert sha(ROOT/rel)==h,rel
    return cfg


def ready(out):
    out=Path(out);cfg=verify(out);r=read(out/'pre-review.json');receipt=read(out/'pre-execution-commit.json')
    assert r['passed'] and r['freeze_sha256']==sha(out/'freeze.json')
    for n in ('registration.json','freeze.json','pre-review.json'):
        assert git('show',receipt['commit']+':review/evidence/'+out.name+'/'+n)==(out/n).read_bytes()
    return cfg


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('prepare','verify'));p.add_argument('--out',required=True,type=Path);a=p.parse_args()
    print(prepare(a.out) if a.mode=='prepare' else verify(a.out,history=True))
