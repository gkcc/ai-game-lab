"""Frozen eight-game development study of selected-task outcome evidence."""
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess

ROOT=Path(__file__).resolve().parents[2]
PUBLIC=ROOT.parent/'publication/ai-game-lab'
OUT=ROOT/'artifacts/task-completion-evidence-v1-20261027'
SOURCE=Path(__file__).resolve().parent
SOURCE_OUT=ROOT/'artifacts/task-action-observation-v1-20261027'
BASE='4592327c396b3f71607dec5e9fbddd0da8d72d15'
PARENT_MANIFEST='b433ee9652f38bbb0cb39afb1e02dcdb3ad84183beabd678f3ecbc8c0570f4fa'
START='2026-10-27T03:22:00+00:00'
DEADLINE='2026-10-27T06:10:00+00:00'
SEEDS=(3911611436,2328454240)
MIB=1024**2;GIB=1024**3


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for data in iter(lambda:stream.read(MIB),b''):h.update(data)
    return h.hexdigest()


def write_new(path,value):
    path=Path(path).resolve()
    if not path.is_relative_to(OUT):raise ValueError('Output must stay inside the registered study')
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8',newline='\n') as stream:
        json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False);stream.write('\n')


def remaining_seconds():
    return (datetime.fromisoformat(DEADLINE)-datetime.now(timezone.utc)).total_seconds()


def deadline_check():
    if remaining_seconds()<=0:raise TimeoutError('Registered total-stage deadline expired')


def git(*args):
    return subprocess.check_output(['git',*args],cwd=PUBLIC)


def selected_games(source):
    games=source['games'];result=[]
    for seed in SEEDS:
        for arm in ('baseline','candidate','teacher_task'):
            found=[g for g in games if (g['seed'],g['opponent'],g['side'],g['arm'])==(seed,'round2',0,arm)]
            if len(found)!=1:raise ValueError('Missing or duplicate registered game')
            result.extend(found)
        found=[g for g in games if (g['seed'],g['opponent'],g['side'],g['arm'])==(seed,'market',1,'candidate')]
        if len(found)!=1:raise ValueError('Missing or duplicate registered game')
        result.extend(found)
    if len(result)!=8 or len({g['key'] for g in result})!=8:raise ValueError('Eight distinct games required')
    return result


def prepare(out=OUT):
    out=Path(out).resolve();deadline_check()
    if out!=OUT or (out/'registration.json').exists():raise ValueError('Registration is exclusive')
    if git('status','--porcelain').strip() or git('rev-parse','HEAD').decode().strip()!=BASE:
        raise ValueError('Public baseline or worktree changed')
    decision=read(out/'root-decision.json')
    if decision['status']!='approved_for_implementation_not_execution':raise ValueError('Missing phase decision')
    if sha(out/'schedule-amendment.json')!=decision['schedule_amendment_sha256']:
        raise ValueError('Pre-freeze schedule amendment changed')
    if sha(out/'planning-initial/root-decision.json')!=decision['planning_initial_decision_sha256']:
        raise ValueError('Initial planning decision changed')
    if sha(out/'planning-initial/experiment.py')!=decision['planning_initial_source_sha256']:
        raise ValueError('Initial planning source changed')
    if sha(SOURCE_OUT/'input-manifest.json')!=PARENT_MANIFEST:raise ValueError('Prior manifest changed')
    parent=read(SOURCE_OUT/'root-result-acceptance.json')
    if not (parent['campaign_accepted'] and parent['offline_analysis_complete'] and not parent['candidate_admitted']):
        raise ValueError('Prior development study is not accepted within its stated scope')
    source=read(SOURCE_OUT/'input-manifest.json');games=selected_games(source)
    if sum(g[k]['size'] for g in games for k in ('replay','trace'))>256*MIB:
        raise ValueError('Registered compressed input limit exceeded by source metadata')
    parent_cfg=read(SOURCE_OUT/'registration.json');parent_freeze=read(SOURCE_OUT/'freeze.json')
    runtime=dict(parent_freeze['runtime_files'])
    for rel,digest in runtime.items():
        if sha(ROOT/rel)!=digest:raise ValueError('Previously frozen runtime source changed: '+rel)
    new={p.relative_to(ROOT).as_posix():sha(p) for p in SOURCE.glob('*.py')};runtime.update(new)
    from learning.task_entry_repair_v1.experiment import installed_files
    files=installed_files()
    if sha(files['rules'])!=parent_cfg['engine_sha256'] or sha(files['loader'])!=parent_cfg['official_loader_sha256']:
        raise ValueError('Official engine or loader changed')
    if shutil.disk_usage(ROOT).free<5*GIB:raise ValueError('Insufficient declared storage reserve')
    write_new(out/'input-manifest.json',dict(schema='task-completion-evidence-input-v1',
        source_manifest_sha256=PARENT_MANIFEST,games=games,games_count=8,raw_files=16))
    cfg=dict(experiment='task-completion-evidence-v1',created_utc=datetime.now(timezone.utc).isoformat(),
        baseline_head=BASE,task_started_utc=START,task_deadline_utc=DEADLINE,total_stage_seconds=10080,
        source_manifest_sha256=PARENT_MANIFEST,input_manifest_sha256=sha(out/'input-manifest.json'),
        data_role='Eight already observed development games; never a new holdout or performance evaluation',
        fixed_selection='Each of two prior development seeds: round2 seat0 S/C/T and market seat1 C',
        games=8,raw_files=16,new_games=0,training_updates=0,neural_forwards=0,gpu_operations=0,
        main_official_transitions=5752,independent_official_transitions=719,total_official_transitions=6471,
        independent_selection='First seed, round2, seat0, candidate; no replacement or rerun',
        independent_sample_rule='Twenty smallest SHA256 of ASCII json.dumps([game,first_step,first_ordinal],ensure_ascii=True,separators=(comma,colon)), without looking at label',
        engine_sha256=parent_cfg['engine_sha256'],engine_commit=parent_cfg['engine_commit'],
        official_loader_sha256=parent_cfg['official_loader_sha256'],
        runtime_versions=dict(python=platform.python_version(),numpy=version('numpy'),
                              kaggle_environments=version('kaggle-environments')),
        analysis_limits=dict(compressed_bytes=256*MIB,decoded_bytes=4*GIB,output_bytes=GIB,
            work_seconds=1200,cpu_seconds=900,sampled_memory_bytes=2*GIB,
            log_bytes=2*MIB,cleanup_seconds=20),
        review_limits=dict(source_metadata_bytes=20*MIB,compressed_bytes=128*MIB,decoded_bytes=GIB),
        artifact_scope='Source trajectory and both player states are privileged offline input. Local rows only identify a selected task and its evidence status; no observation, features, teacher label or opponent private data are exported.',
        positive_definition='The same continuously selected unit/target/op issues the matching official primitive at the target and the official atom shows its operation-specific effect.',
        unknown_definition='Movement, unresolved interruptions, blocked/no-effect actions, missing or ambiguous evidence are not failed tasks; retain unknown.',
        learning_scope='Episode labels describe observed selected actions, not counterfactual action values or causal profitability.',
        feasibility_gate=dict(min_observed=100,min_coverage=0.5,min_distinct_ops=3,min_observed_per_seed=10,
            meaning='Eligibility for a separately preregistered learning probe only, not a promotion or statistical superiority claim'),
        acceptance='Eight semantic replays/16 source streams checked; 8x719 steps; identity and legal observation checks; independent one-game replay plus 20 fixed examples; no privacy leak or source change.',
        default_changed=False,prior_results_changed=False,old_192_raw_reads=0,
        local_registration_not_external_timestamp=True)
    write_new(out/'registration.json',cfg)
    protected={n:sha(ROOT/n) for n in ('main.py','bots/teacher_fertilizer_terminal_v2.py')}
    write_new(out/'freeze.json',dict(registration_sha256=sha(out/'registration.json'),
        input_manifest_sha256=sha(out/'input-manifest.json'),runtime_files=runtime,new_source_sha256=new,
        protected_source_sha256=protected,root_decision_sha256=sha(out/'root-decision.json'),
        schedule_amendment_sha256=sha(out/'schedule-amendment.json'),
        parent_manifest_sha256=PARENT_MANIFEST,parent_acceptance_sha256=sha(SOURCE_OUT/'root-result-acceptance.json'),
        parent_correction_sha256=sha(SOURCE_OUT/'independent-observation-scope-correction.json'),
        parent_publication_sha256=sha(SOURCE_OUT/'publication.json')))
    return verify(out)


def verify(out=OUT):
    out=Path(out).resolve();deadline_check()
    if out!=OUT:raise ValueError('Unexpected study output')
    cfg=read(out/'registration.json');frozen=read(out/'freeze.json')
    for name,key in (('registration.json','registration_sha256'),('input-manifest.json','input_manifest_sha256'),
                     ('root-decision.json','root_decision_sha256'),('schedule-amendment.json','schedule_amendment_sha256')):
        if sha(out/name)!=frozen[key]:raise ValueError('Frozen metadata changed: '+name)
    for name,key in (('input-manifest.json','parent_manifest_sha256'),('root-result-acceptance.json','parent_acceptance_sha256'),
                     ('independent-observation-scope-correction.json','parent_correction_sha256'),('publication.json','parent_publication_sha256')):
        if sha(SOURCE_OUT/name)!=frozen[key]:raise ValueError('Prior evidence changed: '+name)
    for rel,digest in {**frozen['runtime_files'],**frozen['protected_source_sha256']}.items():
        if sha(ROOT/rel)!=digest:raise ValueError('Frozen or protected source changed: '+rel)
    actual={p.relative_to(ROOT).as_posix() for p in SOURCE.glob('*.py')}
    if actual!=set(frozen['new_source_sha256']):raise ValueError('New source set changed')
    from learning.task_entry_repair_v1.experiment import installed_files
    files=installed_files()
    if sha(files['rules'])!=cfg['engine_sha256'] or sha(files['loader'])!=cfg['official_loader_sha256']:
        raise ValueError('Official engine changed')
    if dict(python=platform.python_version(),numpy=version('numpy'),kaggle_environments=version('kaggle-environments'))!=cfg['runtime_versions']:
        raise ValueError('Runtime dependency changed')
    return cfg


def ready(out=OUT):
    cfg=verify(out);out=Path(out)
    review=read(out/'pre-review.json');receipt=read(out/'pre-execution-commit.json')
    if not review['passed'] or review['freeze_sha256']!=sha(out/'freeze.json'):
        raise ValueError('Independent pre-review does not cover this exact freeze')
    if receipt['freeze_sha256']!=sha(out/'freeze.json') or receipt['registration_sha256']!=sha(out/'registration.json'):
        raise ValueError('Local pre-execution commit binding changed')
    return cfg


def verify_commit(out=OUT):
    out=Path(out);receipt=read(out/'pre-execution-commit.json')
    for name in ('registration.json','freeze.json','pre-review.json','input-manifest.json'):
        if git('show',receipt['commit']+':review/evidence/'+out.name+'/'+name)!=(out/name).read_bytes():
            raise ValueError('Pre-execution commit bytes differ')


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=('prepare','verify'));args=parser.parse_args()
    c=prepare() if args.command=='prepare' else verify()
    print(json.dumps(dict(experiment=c['experiment'],games=c['games'],passed=True)))
