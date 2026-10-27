"""Preregister the two-reserved-seed official comparison, before games."""
from datetime import datetime, timezone
import hashlib
import json
import os
import re
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
PUBLIC = ROOT.parent/'publication/ai-game-lab'
OUT = ROOT/'artifacts/task-teacher-choice-eval-v2-20261027'
PARENT = ROOT/'artifacts/task-teacher-choice-bc-v1-20261027'
ECONOMY = ROOT/'artifacts/task-completion-economy-v1-20261027'
JOB = ROOT/'artifacts/task-training-job-supervision-v1-20261027'
SOURCE = Path(__file__).resolve().parent
FAILED = ROOT/'artifacts/task-teacher-choice-eval-v1-20261027'
RACE = ROOT/'artifacts/task-job-sampling-race-v1-20261027'
MIB,GIB = 1024**2,1024**3
CONSUMED = {2941010634,3774915423}
OFFICIAL = dict(new_games=8,live_transitions=8*719,primary_replay_transitions=8*719,
                independent_replay_transitions=719,total_transitions=17*719,per_job_seconds=900)
GATES = dict(seed_mean_own_min=-1000,seed_mean_margin_min=-1000,seat_own_min=-2000,seat_margin_min=-2000)
INDEPENDENT = dict(compressed_bytes=128*MIB,decoded_bytes=GIB,seconds=600,sample_count=20)
JOB_LIMITS = dict(max_rss_bytes=2*GIB,max_commit_bytes=3*GIB,min_available_bytes=2*GIB,
                  wall_seconds=900,poll_seconds=.25,cleanup_seconds=10)


def read(path): return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def git(*args): return subprocess.check_output(['git',*args],cwd=PUBLIC)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(MIB),b''): h.update(block)
    return h.hexdigest()


def write_new(path,value):
    path=Path(path).resolve()
    if not path.is_relative_to(OUT.resolve()): raise ValueError('Output outside experiment')
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8',newline='\n') as stream:
        json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False); stream.write('\n')


def deadline_check():
    deadline=datetime.fromisoformat(read(OUT/'root-decision.json')['deadline_utc'])
    if deadline.tzinfo is None or datetime.now(timezone.utc)>=deadline: raise TimeoutError('Invalid or expired deadline')


def clean_head(expected):
    if git('rev-parse','HEAD').decode().strip()!=expected or git('status','--porcelain').strip():
        raise ValueError('Unexpected or dirty public HEAD')


def public_evidence(folder,names,baseline):
    for name in names:
        if git('show',baseline+':review/evidence/'+folder.name+'/'+name)!=(folder/name).read_bytes():
            raise ValueError('Published evidence changed')


def windows_venv_sha():
    venv=ROOT/'.venv'; executable=venv/'Scripts/python.exe'
    try:
        if (not os.path.samefile(sys.executable,executable) or not os.path.samefile(sys.prefix,venv)
            or os.path.samefile(sys.prefix,sys.base_prefix)):
            raise ValueError('Project Windows virtual environment required')
    except OSError as exc:
        raise ValueError('Project Windows virtual environment identity unavailable') from exc
    return sha(executable)


def prior_plan(parent):
    plan=parent['evaluation_plan']
    expected=dict(OFFICIAL); expected.pop('per_job_seconds')
    expected.update(GATES); expected.update(opponents=['round2'],seats=[0,1])
    if plan!=expected or set(parent['reserved_test_seeds'])!=CONSUMED:
        raise ValueError('Previously frozen evaluation plan changed')
    return plan


def seed_inventory():
    """Only registered seed/input metadata; no results, replays or raw data."""
    allowed={'registration.json','input-manifest.json','root-decision.json','seeds.json','seed-inventory.json','seed-inventory.local.json'}
    tracked=[n for n in git('ls-files','-z').decode().split('\0') if n and 'private' not in {part.lower() for part in Path(n).parts}
             and (Path(n).name in allowed or Path(n).parent.name=='job-inputs')]
    paths=[('public/'+n,PUBLIC/n) for n in tracked]
    paths += [('local/'+p.relative_to(ROOT/'artifacts').as_posix(),p) for p in sorted((ROOT/'artifacts').rglob('*.json'))
              if not p.is_relative_to(OUT) and 'private' not in {part.lower() for part in p.relative_to(ROOT/'artifacts').parts}
              and (p.name in allowed or p.parent.name=='job-inputs')]
    seen=set(CONSUMED); identities=[]; total=0; pattern=re.compile(r'(?<!\d)\d{1,10}(?!\d)')
    for label,path in paths:
        deadline_check(); total+=path.stat().st_size
        if total>256*MIB: raise ValueError('Seed metadata budget exceeded')
        data=path.read_bytes()
        seen.update(int(m.group()) for m in pattern.finditer(data.decode('utf-8-sig')) if int(m.group())<2**32)
        identities.append([label,hashlib.sha256(data).hexdigest()])
    return dict(excluded_uint32=sorted(seen),identities=identities,input_bytes=total,
        scope='Tracked and local non-private registration, seed, root-decision, input-manifest and job-input metadata',
        limitation='Private, deleted, unavailable, external and unlisted history cannot be excluded')


def validate_seeds(seeds,excluded):
    if (not isinstance(seeds,list) or len(seeds)!=2 or any(type(x) is not int or not 0<=x<2**32 for x in seeds)
        or len(set(seeds))!=2 or set(seeds)&(set(excluded)|CONSUMED)):
        raise ValueError('Fresh explicit seed selection required')


def prepare(out=OUT):
    out=Path(out).resolve()
    if os.name!='nt' or out!=OUT.resolve(): raise ValueError('Registration requires Windows project environment')
    if any((out/name).exists() for name in ('registration.json','freeze.json','games-started.json')):
        raise ValueError('Registration or panel already exists')
    deadline_check()
    decision=read(out/'root-decision.json')
    baseline=decision.get('previous_commit')
    if (not isinstance(baseline,str) or len(baseline)!=40 or any(c not in '0123456789abcdef' for c in baseline)
        or decision.get('status')!='approved_for_implementation_not_execution'):
        raise ValueError('Invalid decision')
    clean_head(baseline)
    names=('root-result-acceptance.json','registration.json','freeze.json','fit-result.json','model-seal.json',
           'fit-transport-result.json','independent-result-assessment.json','root-next-learning-decision.json','privacy-projection.json')
    public_evidence(PARENT,names,baseline)
    public_evidence(ECONOMY,('registration.json',),baseline)
    public_evidence(JOB,('registration.json','freeze.json','root-result-acceptance.json'),baseline)
    failure_names=('root-result-acceptance.json','registration.json','freeze.json','failure-projection.json','independent-failure-assessment.json')
    race_names=('root-result-acceptance.json','registration.json','freeze.json','result.json')
    public_evidence(FAILED,failure_names,baseline)
    public_evidence(RACE,race_names,baseline)
    failed=read(FAILED/'root-result-acceptance.json'); race=read(RACE/'root-result-acceptance.json')
    if (failed.get('passed') is not False or failed.get('panel_completed') is not False
        or failed.get('no_retries') is not True or race.get('passed') is not True):
        raise ValueError('Prior failure or synthetic sampling acceptance missing')
    for name in failure_names[1:]:
        if sha(FAILED/name)!=failed['evidence_sha256'][name]: raise ValueError('Failed panel evidence changed')
    for name in race_names[1:]:
        if sha(RACE/name)!=race['evidence_sha256'][name]: raise ValueError('Sampling evidence changed')
    race_freeze=read(RACE/'freeze.json')
    for rel,field in (('learning/task_job_sampling_race_v1/sampling.py','new_source_sha256'),
                      ('learning/task_training_job_supervision_v1/job.py','shared_source_sha256')):
        if sha(ROOT/rel)!=race_freeze[field][rel]: raise ValueError('Accepted sampling source changed')
    inventory=seed_inventory()
    seeds=decision.get('test_seeds'); validate_seeds(seeds,inventory['excluded_uint32'])
    accepted,parent,prior_freeze=(read(PARENT/n) for n in names[:3])
    if not accepted['passed'] or not accepted['training_pipeline_accepted'] or accepted['competitive_gain_proven']:
        raise ValueError('Prior training status differs')
    for name in names[1:-1]:
        if name in accepted['evidence_sha256'] and sha(PARENT/name)!=accepted['evidence_sha256'][name]:
            raise ValueError('Prior evidence chain changed')
    if sha(PARENT/'registration.json')!=prior_freeze['registration_sha256']:
        raise ValueError('Prior registration changed')
    projection=read(PARENT/'privacy-projection.json')
    if not projection['passed'] or projection['public_projection_sha256']!=sha(PARENT/'root-result-acceptance.json'):
        raise ValueError('Prior public privacy projection changed')
    prior_plan(parent)
    seal,fit=read(PARENT/'model-seal.json'),read(PARENT/'fit-result.json')
    if (not seal['passed'] or seal['checkpoint_sha256']!=fit['checkpoint_sha256']
        or seal['fit_result_sha256']!=sha(PARENT/'fit-result.json')):
        raise ValueError('Candidate seal changed')
    economy=read(ECONOMY/'registration.json')
    if (economy['candidate_checkpoint_sha256']!=parent['reference_checkpoint_sha256']
        or economy['crop_checkpoint_sha256']!=parent['crop_checkpoint_sha256']):
        raise ValueError('Reference checkpoint identities changed')
    candidate=(PARENT/'private/model-final.npz').resolve()
    reference=(ROOT/parent['reference_checkpoint']).resolve()
    crop=(ROOT/parent['crop_checkpoint']).resolve()
    for path,digest in ((candidate,seal['checkpoint_sha256']),(reference,parent['reference_checkpoint_sha256']),
                        (crop,parent['crop_checkpoint_sha256'])):
        if not path.is_relative_to(ROOT.resolve()) or sha(path)!=digest: raise ValueError('Checkpoint identity changed')
    if seal['checkpoint_sha256']==parent['reference_checkpoint_sha256']: raise ValueError('Models unexpectedly identical')
    if windows_venv_sha()!=read(JOB/'registration.json')['environment']['interpreter_sha256']:
        raise ValueError('Windows Job interpreter changed')
    from learning.task_entry_repair_v1.experiment import installed_files
    from learning.task_reward_gate_v1.experiment import versions
    installed=installed_files()
    if (versions()!=parent['runtime_versions'] or sha(installed['rules'])!=parent['engine_sha256']
        or sha(installed['loader'])!=parent['official_loader_sha256']): raise ValueError('Official runtime changed')
    if shutil.disk_usage(ROOT).free < economy['storage']['free_floor_bytes']+4*GIB:
        raise ValueError('Insufficient reserved disk space')
    rival=ECONOMY/'sources/round2.py'
    if sha(rival)!=economy['source_sha256']['round2.py']: raise ValueError('Opponent identity changed')
    runtime=dict(prior_freeze['runtime_files'])
    for rel in ('learning/task_job_sampling_race_v1/sampling.py','learning/task_training_job_supervision_v1/job.py'):
        runtime[rel]=sha(ROOT/rel)
    source={p.relative_to(ROOT).as_posix():sha(p) for p in sorted(SOURCE.glob('*.py'))}
    runtime.update(source)
    for rel,digest in runtime.items():
        if sha(ROOT/rel)!=digest: raise ValueError('Runtime changed')
    target=out/'sources/round2.py'; target.parent.mkdir(parents=True,exist_ok=True)
    with target.open('xb') as stream: stream.write(rival.read_bytes())
    write_new(out/'seed-inventory.local.json',inventory)
    cfg=dict(experiment='task-teacher-choice-eval-v2',baseline_head=baseline,deadline_utc=decision['deadline_utc'],
        test_seeds=list(seeds),seed_inventory_sha256=sha(out/'seed-inventory.local.json'),
        seed_scope=inventory['scope'],seed_history_limitation=inventory['limitation'],
        opponents=['round2'],seats=[0,1],policies=['candidate','baseline'],
        candidate_checkpoint=candidate.relative_to(ROOT.resolve()).as_posix(),candidate_checkpoint_sha256=seal['checkpoint_sha256'],
        baseline_checkpoint=parent['reference_checkpoint'],baseline_checkpoint_sha256=parent['reference_checkpoint_sha256'],
        crop_checkpoint=parent['crop_checkpoint'],crop_checkpoint_sha256=parent['crop_checkpoint_sha256'],
        source_sha256=dict(economy['source_sha256']),engine_commit=parent['engine_commit'],engine_sha256=parent['engine_sha256'],
        official_loader_sha256=parent['official_loader_sha256'],runtime_versions=parent['runtime_versions'],
        windows_venv_interpreter_sha256=parent['windows_venv_interpreter_sha256'],
        game_limits=dict(economy['game_limits']),storage=dict(economy['storage']),official_limits=dict(OFFICIAL),
        official_job_supervision=dict(JOB_LIMITS),independent_job_supervision=dict(JOB_LIMITS),
        independent_limits=dict(INDEPENDENT),independent_selection='first reserved seed, candidate, round2, seat 0',
        acceptance=dict(GATES),no_retraining=True,retries=False,
        interpretation='Two seed clusters, one opponent; exploratory screen, not significance or automatic promotion')
    write_new(out/'registration.json',cfg)
    write_new(out/'freeze.json',dict(registration_sha256=sha(out/'registration.json'),root_decision_sha256=sha(out/'root-decision.json'),
        runtime_files=runtime,new_source_sha256=source,parent_commit=baseline,
        parent_result_sha256=sha(PARENT/'root-result-acceptance.json'),parent_seal_sha256=sha(PARENT/'model-seal.json'),
        failure_evidence={n:sha(FAILED/n) for n in failure_names},race_evidence={n:sha(RACE/n) for n in race_names}))
    return verify(out)


def verify(out=OUT):
    out=Path(out).resolve()
    if out!=OUT.resolve(): raise ValueError('Unexpected output')
    deadline_check()
    receipt=out/'pre-execution-commit.json'
    cfg,freeze=read(out/'registration.json'),read(out/'freeze.json')
    clean_head(read(receipt)['commit'] if receipt.exists() else cfg['baseline_head'])
    if sha(out/'registration.json')!=freeze['registration_sha256'] or sha(out/'root-decision.json')!=freeze['root_decision_sha256']:
        raise ValueError('Registration changed')
    if (cfg['opponents']!=['round2'] or cfg['seats']!=[0,1]
        or cfg['policies']!=['candidate','baseline'] or cfg['official_limits']!=OFFICIAL
        or cfg['acceptance']!=GATES or cfg['independent_limits']!=INDEPENDENT
        or cfg['official_job_supervision']!=JOB_LIMITS or cfg['independent_job_supervision']!=JOB_LIMITS):
        raise ValueError('Panel or budget changed')
    if sha(out/'seed-inventory.local.json')!=cfg['seed_inventory_sha256']: raise ValueError('Seed inventory changed')
    validate_seeds(cfg['test_seeds'],read(out/'seed-inventory.local.json')['excluded_uint32'])
    for folder,key in ((FAILED,'failure_evidence'),(RACE,'race_evidence')):
        for name,digest in freeze[key].items():
            if sha(folder/name)!=digest: raise ValueError('Prior evidence changed')
    if sha(PARENT/'root-result-acceptance.json')!=freeze['parent_result_sha256'] or sha(PARENT/'model-seal.json')!=freeze['parent_seal_sha256']:
        raise ValueError('Parent identity changed')
    if sha(out/'sources/round2.py')!=cfg['source_sha256']['round2.py']:
        raise ValueError('Opponent changed')
    if {p.relative_to(ROOT).as_posix() for p in SOURCE.glob('*.py')}!=set(freeze['new_source_sha256']):
        raise ValueError('Source set changed')
    for rel,digest in freeze['runtime_files'].items():
        if sha(ROOT/rel)!=digest: raise ValueError('Runtime changed')
    for key in ('candidate_checkpoint','baseline_checkpoint','crop_checkpoint'):
        path=(ROOT/cfg[key]).resolve()
        if not path.is_relative_to(ROOT.resolve()) or sha(path)!=cfg[key+'_sha256']:
            raise ValueError('Checkpoint changed')
    if os.name=='nt':
        if windows_venv_sha()!=cfg['windows_venv_interpreter_sha256']: raise ValueError('Windows virtual environment changed')
        from learning.task_entry_repair_v1.experiment import installed_files
        from learning.task_reward_gate_v1.experiment import versions
        installed=installed_files()
        if (versions()!=cfg['runtime_versions'] or sha(installed['rules'])!=cfg['engine_sha256']
            or sha(installed['loader'])!=cfg['official_loader_sha256']): raise ValueError('Official runtime changed')
    return cfg


def ready(out=OUT):
    cfg=verify(out)
    review,receipt=read(Path(out)/'pre-review.json'),read(Path(out)/'pre-execution-commit.json')
    if (not review['passed'] or review['freeze_sha256']!=sha(Path(out)/'freeze.json')
        or receipt['freeze_sha256']!=sha(Path(out)/'freeze.json') or receipt['registration_sha256']!=sha(Path(out)/'registration.json')):
        raise ValueError('Review or commitment changed')
    verify_commit(out)
    return cfg


def verify_commit(out=OUT):
    out=Path(out).resolve(); cfg=verify(out)
    commit=read(out/'pre-execution-commit.json')['commit']
    clean_head(commit)
    if git('merge-base','--is-ancestor',cfg['baseline_head'],commit)!=b'': raise ValueError('Invalid ancestry')
    for name in ('root-decision.json','registration.json','freeze.json','pre-review.json'):
        if git('show',commit+':review/evidence/'+out.name+'/'+name)!=(out/name).read_bytes(): raise ValueError('Committed evidence changed')
    for rel in read(out/'freeze.json')['new_source_sha256']:
        if git('show',commit+':kaggriculture/'+rel)!=(ROOT/rel).read_bytes(): raise ValueError('Committed source changed')
    return cfg
