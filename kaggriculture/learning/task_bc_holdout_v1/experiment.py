"""Preregister explicit new seeds and immutable two-head comparison."""
import argparse
from datetime import datetime, timezone
from importlib.metadata import version
import platform
import random
from pathlib import Path
import shutil
import subprocess

from learning.bc_crop.agent import ROOT, sha
from learning.bc_crop.experiment import read, save
from learning.bc_crop.prepare import novelty
from learning.task_entry_repair_v1.experiment import verify as verify_previous, installed_files

PREVIOUS=ROOT/'artifacts/task-entry-repair-v1-20261023'
TASK=ROOT/'artifacts/task-bc-v1-20261023-run1'
PUBLIC=ROOT.parent/'publication/ai-game-lab'
TASK_SHA='9253bc05e632eec2d5a813557272655ab97e023c09843c60a80781b6cc2ab143'
CROP_SHA='d497e8284dde1f7b836b1cf88915822f82a96be1f99e7b126d06ea263e281ffd'


def git(*args):
    return subprocess.check_output(['git','-C',str(PUBLIC),*args])


def seed_config(value):
    if set(value)!={'test_seeds','generation'}:raise ValueError('Explicit seeds and generation contract required')
    seeds=value['test_seeds'];g=value['generation']
    if len(seeds)!=8 or len(set(seeds))!=8 or any(type(s)is not int or not 0<=s<2**32 for s in seeds):
        raise ValueError('Eight distinct uint32 seeds required')
    assert g['rng_seed']==2026102340 and g['minimum']==0 and g['maximum_exclusive']==2**32
    assert g['count']==8 and g['draws']==1
    assert seeds==random.Random(g['rng_seed']).sample(range(g['minimum'],g['maximum_exclusive']),8)
    return seeds


def admission(cfg):
    assert cfg['evaluation_role']=='independent-holdout-fixed-model'
    assert cfg['game_budget']==96 and cfg['max_workers']==2
    assert cfg['training_runs']==cfg['parameter_updates']==cfg['checkpoint_reselections']==0
    assert cfg['task_checkpoint_sha256']==TASK_SHA and cfg['crop_checkpoint_sha256']==CROP_SHA
    assert cfg['prior_task_stage_acceptance_passed'] is False and cfg['default_changed'] is False
    assert cfg['arms']==['trained','teacher_task'] and cfg['opponents']==['v1','market','round2']
    assert cfg['seats']==[0,1]
    assert cfg['test_seeds']==seed_config(cfg['explicit_seed_config'])
    assert cfg['bootstrap']==dict(replicates=20000,seed=2026102341,quantile_method='linear')
    assert cfg['gates']==dict(relative_ci_lower_gt=-.02,opponent_relative_mean_min=-.05,opponent_score_delta_min=-.05)


def prepare(out,seeds_path):
    out=Path(out).resolve();seeds_path=Path(seeds_path).resolve()
    if out.exists():raise FileExistsError('Preserve all experiments, including failed attempts')
    assert not git('status','--porcelain').strip()
    old=verify_previous(PREVIOUS,history=True)
    assert read(PREVIOUS/'acceptance.json')['passed'] and read(PREVIOUS/'independent-review.json')['passed']
    assert read(TASK/'stage-outcome.json')['stage_acceptance_passed'] is False
    explicit=read(seeds_path);seeds=seed_config(explicit)
    tracked=[n for n in git('ls-files','-z').decode().split('\0') if n]
    scan=novelty(dict(train_seeds=[],validation_seeds=[],test_seeds=seeds,parity_seed=seeds[0]),PUBLIC,tracked,ROOT/'artifacts')
    # Include failed dispatch records even when no result or replay was produced.
    import re,hashlib,json
    pattern=re.compile(r'(?<!\d)(?:'+'|'.join(map(str,seeds))+r')(?!\d)')
    attempted=[]
    for p in sorted((ROOT/'artifacts').rglob('*attempt*.json')):
        raw=p.read_bytes();assert not pattern.search(raw.decode('utf-8-sig')),p.name
        attempted.append((p.relative_to(ROOT).as_posix(),sha(p)))
    scan['failed_attempt_files']=len(attempted)
    scan['failed_attempt_manifest_sha256']=hashlib.sha256(json.dumps(attempted,separators=(',',':')).encode()).hexdigest()
    demo=ROOT/'artifacts/bc-crop-v1-20261023-run1';original=read(demo/'registration.json')
    sources={name:demo/'sources'/name for name in ('v1.py','market.py','round2.py')}
    for name,p in sources.items():assert sha(p)==original['source_sha256'][name]
    cfg=dict(experiment='task-bc-holdout-v1',created_utc=datetime.now(timezone.utc).isoformat(),
        baseline_head=git('rev-parse','HEAD').decode().strip(),initial_worktree_clean=True,
        runtime_versions=dict(python=platform.python_version(),numpy=version('numpy'),kaggle_environments=version('kaggle-environments')),
        evaluation_role='independent-holdout-fixed-model',previous_root=PREVIOUS.relative_to(ROOT).as_posix(),
        prior_task_stage_acceptance_passed=False,default_changed=False,
        task_checkpoint_sha256=TASK_SHA,crop_checkpoint=old['crop_checkpoint'],crop_checkpoint_sha256=CROP_SHA,
        engine_commit=old['engine_commit'],engine_sha256=old['engine_sha256'],official_loader_sha256=old['official_loader_sha256'],
        explicit_seed_config=explicit,seed_config_sha256=sha(seeds_path),test_seeds=seeds,seed_novelty=scan,
        source_sha256={n:sha(p) for n,p in sources.items()},arms=['trained','teacher_task'],opponents=['v1','market','round2'],seats=[0,1],
        game_budget=96,max_workers=2,per_game_timeout_seconds=900,
        training_runs=0,parameter_updates=0,checkpoint_reselections=0,
        bootstrap=dict(replicates=20000,seed=2026102341,quantile_method='linear'),
        gates=dict(relative_ci_lower_gt=-.02,opponent_relative_mean_min=-.05,opponent_score_delta_min=-.05),
        generation_population='Eight uniform sampled uint32 seeds without replacement; no outcome-dependent redraw',
        execution_order='Seed configuration order; within seed opponent order, then arm order, then seat order. Maximum two independent game processes; no next-seed prefetch.',
        statistic='Own net=actual final cash minus own initial Replay cash; pair trained-teacher_task at identical seed/opponent/seat; relative=mean paired difference/reference mean. 20k PCG64 seed2026102341 resamples of eight whole seeds, retaining all six pairs per seed; re-estimate numerator and denominator; linear 2.5/97.5 percentiles.',
        denominator='Every full-sample, per-opponent and bootstrap reference mean net denominator must be positive; otherwise undefined relative gate and no admission.',
        interpretation='Engineering retention tolerance of2percent, not natural threshold or superiority premise. Eight independent seed clusters, not96independent games. Per-opponent observed guards are not statistical noninferiority guarantees. Known opponents only; teacher_task is hybrid rule task plus same improved crop head.',
        stop='Any error, timeout, source/model/privacy/audit mismatch stops new dispatch; bounded in-flight work finishes or is killed and waited at timeout. No retries/replacements. After complete seed stop if mean student final cash <50percent reference; no truncated confirmatory inference. Never stop early for favorable results.',
        acceptance='All96 real official filename games720states719actionsDONEboth0errors; official ordered replay and both actor/model traces regenerated; all frozen identity/history/privacy checks; complete paired statistics and gates.',
        inference_scope='If gates pass only retain fixed task student as research baseline for this known-opponent/new-seed panel. Do not replace default or claim full mechanism freedom/teacher superiority. Only primary lower>0 permits panel profit-improvement claim.',
        boundaries='No new training/data/parameter search/checkpoint choice/DAgger/PPO. Failure consumes all attempted seeds; no adaptive extension. Raw Replay, traces, weights and local logs remain private.')
    admission(cfg)
    protected=read(PREVIOUS/'protection-before.json')['runtime'].copy()
    for base in (PREVIOUS,ROOT/'learning/task_entry_repair_v1'):
        for p in base.rglob('*'):
            if p.is_file() and '__pycache__' not in p.parts:protected[p.relative_to(ROOT).as_posix()]=sha(p)
    protected['TASK_ENTRY_REPAIR_V1.zh-CN.md']=sha(ROOT/'TASK_ENTRY_REPAIR_V1.zh-CN.md')
    files=read(PREVIOUS/'freeze.json')['runtime_files'].copy()
    for p in Path(__file__).parent.iterdir():
        if p.is_file() and p.suffix in ('.py','.json','.md'):files[p.relative_to(ROOT).as_posix()]=sha(p)
    out.mkdir(parents=True)
    for sub in ('sources','model','agents','traces','games','jobs'): (out/sub).mkdir()
    for name,p in sources.items():shutil.copyfile(p,out/'sources'/name)
    shutil.copyfile(PREVIOUS/'model/trained.npz',out/'model/trained.npz')
    shutil.copyfile(seeds_path,out/'seeds.json')
    save(out/'registration.json',cfg)
    save(out/'protection-before.json',dict(runtime=protected,public={n:sha(PUBLIC/n) for n in tracked}))
    save(out/'freeze.json',dict(runtime_files=files,registration_sha256=sha(out/'registration.json'),
        protection_sha256=sha(out/'protection-before.json'),seed_config_sha256=sha(out/'seeds.json'),
        prior_acceptance_sha256=sha(PREVIOUS/'acceptance.json'),prior_review_sha256=sha(PREVIOUS/'independent-review.json'),
        prior_admission_sha256=sha(TASK/'admission.json'),new_game_results_absent=True,local_freeze_not_external_timestamp=True))
    return verify(out,history=True)


def verify(out,history=False):
    out=Path(out);cfg=read(out/'registration.json');f=read(out/'freeze.json');admission(cfg)
    assert sha(out/'registration.json')==f['registration_sha256']
    assert cfg['runtime_versions']==dict(python=platform.python_version(),numpy=version('numpy'),kaggle_environments=version('kaggle-environments'))
    assert sha(out/'protection-before.json')==f['protection_sha256']
    assert sha(out/'seeds.json')==f['seed_config_sha256']==cfg['seed_config_sha256']
    for rel,h in f['runtime_files'].items():assert sha(ROOT/rel)==h,rel
    for name,h in cfg['source_sha256'].items():assert sha(out/'sources'/name)==h,name
    assert sha(out/'model/trained.npz')==TASK_SHA and sha(ROOT/cfg['crop_checkpoint'])==CROP_SHA
    official=installed_files();assert sha(official['rules'])==cfg['engine_sha256'];assert sha(official['loader'])==cfg['official_loader_sha256']
    assert sha(PREVIOUS/'acceptance.json')==f['prior_acceptance_sha256']
    assert sha(PREVIOUS/'independent-review.json')==f['prior_review_sha256']
    assert sha(TASK/'admission.json')==f['prior_admission_sha256']
    if history:
        for rel,h in read(out/'protection-before.json')['runtime'].items():assert sha(ROOT/rel)==h,rel
    return cfg


def ready(out):
    out=Path(out);cfg=verify(out);receipt=read(out/'pre-execution-commit.json');review=read(out/'pre-review.json')
    assert review['passed'] and review['freeze_sha256']==sha(out/'freeze.json')
    assert review['registration_sha256']==sha(out/'registration.json')
    for name in ('registration.json','freeze.json','pre-review.json'):
        assert git('show',receipt['commit']+':review/evidence/'+out.name+'/'+name)==(out/name).read_bytes()
    assert not (out/'failure.json').exists()
    return cfg


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('prepare','verify'));p.add_argument('--out',type=Path,required=True)
    p.add_argument('--seed-config',type=Path);a=p.parse_args()
    if a.mode=='prepare':
        if a.seed_config is None:p.error('prepare requires explicit --seed-config')
        prepare(a.out,a.seed_config)
    else:verify(a.out,history=True)
