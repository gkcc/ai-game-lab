"""Freeze a small fresh-seed paired official screen before any game."""
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2]
PUBLIC=ROOT.parent/'publication/ai-game-lab'
OUT=ROOT/'artifacts/task-teacher-weighted-eval-v2-20261028'
SOURCE=Path(__file__).resolve().parent
PARENT=ROOT/'artifacts/task-teacher-weighted-v1-20261027'
BC=ROOT/'artifacts/task-teacher-choice-bc-v1-20261027'
EVAL=ROOT/'artifacts/task-teacher-choice-eval-v2-20261027'
ACTION=ROOT/'artifacts/task-action-observation-v1-20261027'
BASE='b9d3da6a7b786c62d66d802dc885f1dae09d5c34'
FAILED=ROOT/'artifacts/task-teacher-weighted-eval-v1-20261027'
JOB_TEST=ROOT/'artifacts/task-job-empty-exit-v1-r2-20261028'
JOB_EVIDENCE=('root-result-acceptance.json','result.json','independent-result-assessment.json',
              'registration.json','freeze.json','pre-execution-commit.json')
MIB,GIB=1024**2,1024**3
JOB=dict(max_rss_bytes=2*GIB,max_commit_bytes=3*GIB,min_available_bytes=2*GIB,
         wall_seconds=900,poll_seconds=.25,cleanup_seconds=10)
PREFLIGHT=dict(min_available_bytes=4*GIB,samples=3,interval_seconds=10)
INDEPENDENT=dict(compressed_bytes=128*MIB,decoded_bytes=GIB,seconds=600,sample_count=20)
OFFICIAL=dict(new_games=24,live_transitions=24*719,primary_replay_transitions=24*719,
              independent_replay_transitions=719,total_transitions=49*719)
GATES=dict(pair_own_min=-2000,pair_margin_min=-2000,seed_opponent_own_min=-1000,seed_opponent_margin_min=-1000,
           bootstrap_seed=2026102733,bootstrap_replicates=10000,bootstrap_cluster='whole seed',
           bootstrap_interpretation='descriptive only; no significance or promotion')


def read(path): return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda:f.read(MIB),b''): h.update(part)
    return h.hexdigest()


def git(*args): return subprocess.check_output(['git',*args],cwd=PUBLIC)


def write_new(path,value):
    path=Path(path).resolve()
    if not path.is_relative_to(OUT.resolve()): raise ValueError('Output outside owned experiment')
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8',newline='\n') as f:
        json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False); f.write('\n')


def deadline_check():
    end=datetime.fromisoformat(read(OUT/'root-decision.json')['deadline_utc'])
    if end.tzinfo is None or datetime.now(timezone.utc)>=end: raise TimeoutError('Stage deadline')


def environment():
    venv=ROOT/'.venv'; python=venv/'Scripts/python.exe'
    if (os.name!='nt' or not os.path.samefile(sys.executable,python) or not os.path.samefile(sys.prefix,venv)
        or os.path.samefile(sys.prefix,sys.base_prefix) or sys.flags.optimize!=0 or 'PYTHONOPTIMIZE' in os.environ):
        raise ValueError('Expected non-optimized project Windows interpreter')
    from learning.task_reward_gate_v1.experiment import versions
    return dict(interpreter_sha256=sha(python),versions=versions(),optimization_level=0,python_optimize_env_absent=True)


def clean_head(expected):
    if git('rev-parse','HEAD').decode().strip()!=expected or git('status','--porcelain').strip():
        raise ValueError('Unexpected or dirty public HEAD')


def public_equal(folder,names):
    for name in names:
        if git('show',BASE+':review/evidence/'+folder.name+'/'+name)!=(folder/name).read_bytes():
            raise ValueError('Published predecessor differs')


def inventory():
    """Metadata-only collision scan; existing reservations are excluded."""
    from learning.task_teacher_choice_bc_v1.experiment import seed_values
    names={'registration.json','input-manifest.json','root-decision.json','seeds.json',
           'seed-inventory.json','seed-inventory.local.json'}
    def allowed(path): return 'private' not in {x.casefold() for x in Path(path).parts}
    tracked=[n for n in git('ls-files','-z').decode('utf-8').split('\0') if n]
    public=[('public/'+n,PUBLIC/n) for n in tracked if allowed(n) and OUT.name not in Path(n).parts
            and (Path(n).name.lower() in names or Path(n).parent.name=='job-inputs')]
    artifacts=ROOT/'artifacts'
    local=[('local/'+p.relative_to(artifacts).as_posix(),p) for p in sorted(artifacts.rglob('*.json'))
           if not p.is_relative_to(OUT) and allowed(p.relative_to(artifacts))
           and (p.name.lower() in names or p.parent.name=='job-inputs')]
    seen=set(); identities=[]; used=0
    for label,path in public+local:
        deadline_check(); used+=path.stat().st_size
        if used>256*MIB: raise ValueError('Seed metadata budget exceeded')
        data=path.read_bytes(); obj=json.loads(data.decode('utf-8-sig'))
        seen.update(seed_values(obj,path.name.lower()=='seeds.json'))
        identities.append([label,hashlib.sha256(data).hexdigest()])
    return dict(excluded_uint32=sorted(seen),identities=identities,metadata_files=len(identities),metadata_bytes=used,
        scope='Recognized public and local non-private seed metadata, including existing reservations',
        limitation='Private, deleted, unavailable, external or unlisted history cannot be excluded')


def validate_new_seeds(seeds,excluded,reservations,failed_seeds):
    from .evaluation import plans
    plans(seeds)
    if set(seeds)&(set(excluded)|set(reservations)|set(failed_seeds)):
        raise ValueError('Proposed seed appears in recognized history')


def prepare(out=OUT):
    out=Path(out).resolve()
    if out!=OUT.resolve() or (out/'registration.json').exists(): raise ValueError('Panel already registered')
    deadline_check(); clean_head(BASE)
    decision=read(out/'root-decision.json')
    if decision.get('previous_commit')!=BASE or decision.get('status')!='approved_for_implementation_not_execution':
        raise ValueError('Missing prospective decision')
    public_equal(PARENT,('registration.json','freeze.json','root-result-acceptance.json','fit-result.json','model-seal.json'))
    public_equal(BC,('registration.json','model-seal.json','root-result-acceptance.json'))
    public_equal(EVAL,('registration.json','freeze.json','root-result-acceptance.json'))
    public_equal(ACTION,('registration.json',))
    parent,parent_cfg,seal=(read(PARENT/name) for name in ('root-result-acceptance.json','registration.json','model-seal.json'))
    if (parent.get('passed') is not True or parent.get('competitive_gain_proven') is not False
        or parent.get('automatic_promotion') is not False or seal.get('passed') is not True
        or seal.get('checkpoint_sha256')!=parent.get('checkpoint_sha256')):
        raise ValueError('Weighted fit not technically sealed')
    reservations=parent_cfg['reserved_eval_seeds']
    if (not isinstance(reservations,list) or len(reservations)!=6 or len(set(reservations))!=6
        or any(type(s) is not int or not 0<=s<2**32 for s in reservations)):
        raise ValueError('Invalid reserved seeds')
    prior_inventory=read(PARENT/'seed-inventory.local.json')
    if (sha(PARENT/'seed-inventory.local.json')!=parent_cfg['seed_inventory_sha256']
        or reservations!=prior_inventory['reserved_eval_seeds']): raise ValueError('Original reservation changed')
    public_equal(FAILED,('registration.json','freeze.json','root-failure-acceptance.json'))
    public_equal(JOB_TEST,JOB_EVIDENCE)
    job_result=read(JOB_TEST/'root-result-acceptance.json')
    if (job_result.get('passed') is not True or job_result.get('confirmed_empty_exits')!=1
        or job_result.get('launcher_reaped') is not True or job_result.get('job_members_zero') is not True
        or job_result.get('exit_code')!=0 or job_result.get('previous_failed_panel_reaccepted') is not False
        or job_result.get('result_sha256')!=sha(JOB_TEST/'result.json')
        or job_result.get('independent_assessment_sha256')!=sha(JOB_TEST/'independent-result-assessment.json')
        or job_result.get('registration_sha256')!=sha(JOB_TEST/'registration.json')
        or job_result.get('freeze_sha256')!=sha(JOB_TEST/'freeze.json')
        or job_result.get('pre_execution_commit')!=read(JOB_TEST/'pre-execution-commit.json')['commit']):
        raise ValueError('Synthetic Job evidence invalid')
    failed=read(FAILED/'root-failure-acceptance.json')
    if (failed.get('passed') is not True or failed.get('partial_panel_scored') is not False
        or failed.get('error_kind')!='EmptySnapshot' or failed.get('freeze_sha256')!=sha(FAILED/'freeze.json')):
        raise ValueError('Previous failure evidence changed')
    seeds=decision.get('test_seeds')
    inv=inventory(); excluded=set(inv['excluded_uint32']) | set(reservations) | set(read(FAILED/'registration.json')['test_seeds'])
    validate_new_seeds(seeds,excluded,reservations,read(FAILED/'registration.json')['test_seeds'])
    inv['excluded_uint32']=sorted(excluded)
    weighted=PARENT/'private/model-final.npz'; unweighted=ROOT/parent_cfg['reference_checkpoint']; crop=ROOT/parent_cfg['crop_checkpoint']
    if (sha(weighted)!=seal['checkpoint_sha256'] or sha(unweighted)!=parent_cfg['reference_checkpoint_sha256']
        or sha(crop)!=parent_cfg['crop_checkpoint_sha256'] or sha(weighted)==sha(unweighted)):
        raise ValueError('Model identities changed or coincide')
    env=environment(); old_eval=read(EVAL/'registration.json')
    if env['interpreter_sha256']!=old_eval['windows_venv_interpreter_sha256'] or env['versions']!=old_eval['runtime_versions']:
        raise ValueError('Windows runtime changed')
    from learning.task_entry_repair_v1.experiment import installed_files
    installed=installed_files()
    if sha(installed['rules'])!=old_eval['engine_sha256'] or sha(installed['loader'])!=old_eval['official_loader_sha256']:
        raise ValueError('Official engine changed')
    if shutil.disk_usage(ROOT).free<8*GIB: raise ValueError('Insufficient free disk')
    opponent_cfg=read(ACTION/'registration.json'); sources={}
    for name in ('round2.py','market.py'):
        original=ACTION/'sources'/name; digest=sha(original)
        if digest!=opponent_cfg['source_sha256'][name]: raise ValueError('Opponent source changed')
        sources[name]=digest
    from learning.task_teacher_merge_real_v1.experiment import trusted_runtime
    runtime=trusted_runtime((EVAL/'freeze.json').read_bytes(),
        git('show',BASE+':review/evidence/'+EVAL.name+'/freeze.json'),read(EVAL/'root-result-acceptance.json'))
    for rel in ('learning/task_teacher_merge_real_v1/experiment.py','learning/task_job_sampling_race_v1/sampling.py',
                'learning/task_job_empty_exit_v1/sampling.py','learning/task_training_job_supervision_v1/job.py'):
        data=git('show',BASE+':kaggriculture/'+rel)
        if data!=(ROOT/rel).read_bytes(): raise ValueError('Published dependency changed')
        runtime[rel]=hashlib.sha256(data).hexdigest()
    for rel,digest in runtime.items():
        if sha(ROOT/rel)!=digest: raise ValueError('Runtime source changed')
    source={p.relative_to(ROOT).as_posix():sha(p) for p in sorted(SOURCE.glob('*.py'))}
    if not source: raise ValueError('Missing panel source')
    (out/'sources').mkdir(exist_ok=False)
    for name in sources:
        with (out/'sources'/name).open('xb') as f: f.write((ACTION/'sources'/name).read_bytes())
    write_new(out/'seed-inventory.local.json',inv)
    cfg=dict(experiment='task-teacher-weighted-eval-v2',baseline_head=BASE,deadline_utc=decision['deadline_utc'],
        test_seeds=seeds,excluded_previous_reservations=reservations,seed_inventory_sha256=sha(out/'seed-inventory.local.json'),
        seed_scope=inv['scope'],seed_history_limitation=inv['limitation'],opponents=['round2','market'],seats=[0,1],
        policies=['weighted','unweighted'],weighted_checkpoint=weighted.relative_to(ROOT).as_posix(),
        weighted_checkpoint_sha256=seal['checkpoint_sha256'],unweighted_checkpoint=parent_cfg['reference_checkpoint'],
        unweighted_checkpoint_sha256=parent_cfg['reference_checkpoint_sha256'],crop_checkpoint=parent_cfg['crop_checkpoint'],
        crop_checkpoint_sha256=parent_cfg['crop_checkpoint_sha256'],source_sha256=sources,
        engine_commit=old_eval['engine_commit'],engine_sha256=old_eval['engine_sha256'],
        official_loader_sha256=old_eval['official_loader_sha256'],environment=env,game_limits=old_eval['game_limits'],
        storage=old_eval['storage'],official_job_supervision=JOB,independent_job_supervision=JOB,host_preflight=PREFLIGHT,
        independent_limits=INDEPENDENT,official_limits=OFFICIAL,gates=GATES,
        independent_selection='first new seed, weighted, round2, seat 0; deterministic 20 callback sample',
        no_retraining=True,no_automatic_promotion=True,retry_allowed=False,
        interpretation='Three seed clusters; descriptive screen, no significance or promotion')
    write_new(out/'registration.json',cfg)
    write_new(out/'freeze.json',dict(registration_sha256=sha(out/'registration.json'),
        root_decision_sha256=sha(out/'root-decision.json'),parent_commit=BASE,runtime_files=runtime,new_source_sha256=source,
        weighted_acceptance_sha256=sha(PARENT/'root-result-acceptance.json'),weighted_seal_sha256=sha(PARENT/'model-seal.json'),
        weighted_registration_sha256=sha(PARENT/'registration.json'),old_model_seal_sha256=sha(BC/'model-seal.json'),
        eval_freeze_sha256=sha(EVAL/'freeze.json'),eval_acceptance_sha256=sha(EVAL/'root-result-acceptance.json'),
        action_registration_sha256=sha(ACTION/'registration.json'),failure_acceptance_sha256=sha(FAILED/'root-failure-acceptance.json'),
        failure_registration_sha256=sha(FAILED/'registration.json'),failure_freeze_sha256=sha(FAILED/'freeze.json'),
        job_evidence_sha256={name:sha(JOB_TEST/name) for name in JOB_EVIDENCE}))
    return verify(out)


def verify(out=OUT):
    out=Path(out).resolve()
    if out!=OUT.resolve(): raise ValueError('Unexpected output')
    deadline_check(); cfg,freeze=read(out/'registration.json'),read(out/'freeze.json')
    if (sha(out/'registration.json')!=freeze['registration_sha256'] or sha(out/'root-decision.json')!=freeze['root_decision_sha256']
        or sha(PARENT/'root-result-acceptance.json')!=freeze['weighted_acceptance_sha256']
        or sha(PARENT/'model-seal.json')!=freeze['weighted_seal_sha256'] or sha(PARENT/'registration.json')!=freeze['weighted_registration_sha256']
        or sha(BC/'model-seal.json')!=freeze['old_model_seal_sha256'] or sha(EVAL/'freeze.json')!=freeze['eval_freeze_sha256']
        or sha(EVAL/'root-result-acceptance.json')!=freeze['eval_acceptance_sha256']
        or sha(ACTION/'registration.json')!=freeze['action_registration_sha256']
        or sha(FAILED/'root-failure-acceptance.json')!=freeze['failure_acceptance_sha256']
        or sha(FAILED/'registration.json')!=freeze['failure_registration_sha256']
        or sha(FAILED/'freeze.json')!=freeze['failure_freeze_sha256']): raise ValueError('Predecessor identity changed')
    if (cfg['opponents']!=['round2','market'] or cfg['seats']!=[0,1] or cfg['policies']!=['weighted','unweighted']
        or cfg['official_job_supervision']!=JOB or cfg['independent_job_supervision']!=JOB or cfg['host_preflight']!=PREFLIGHT
        or cfg['independent_limits']!=INDEPENDENT or cfg['official_limits']!=OFFICIAL or cfg['gates']!=GATES
        or cfg['retry_allowed'] is not False or cfg['no_retraining'] is not True or cfg['no_automatic_promotion'] is not True):
        raise ValueError('Panel or budget changed')
    if (set(freeze['job_evidence_sha256'])!=set(JOB_EVIDENCE)
        or any(sha(JOB_TEST/name)!=digest for name,digest in freeze['job_evidence_sha256'].items())):
        raise ValueError('Synthetic Job evidence changed')
    reservations=read(PARENT/'registration.json')['reserved_eval_seeds']; inv=read(out/'seed-inventory.local.json')
    from .evaluation import plans
    plans(cfg['test_seeds'])
    if (sha(out/'seed-inventory.local.json')!=cfg['seed_inventory_sha256'] or cfg['test_seeds']!=read(out/'root-decision.json')['test_seeds']
        or cfg['excluded_previous_reservations']!=reservations or len(reservations)!=6 or len(set(reservations))!=6
        or set(cfg['test_seeds'])&set(inv['excluded_uint32']) or set(cfg['test_seeds'])&set(reservations)
        or set(cfg['test_seeds'])&set(read(FAILED/'registration.json')['test_seeds'])): raise ValueError('Seed identity changed')
    for name,digest in cfg['source_sha256'].items():
        if sha(out/'sources'/name)!=digest: raise ValueError('Opponent changed')
    for name in ('weighted','unweighted','crop'):
        if sha(ROOT/cfg[name+'_checkpoint'])!=cfg[name+'_checkpoint_sha256']: raise ValueError('Model changed')
    if {p.relative_to(ROOT).as_posix() for p in SOURCE.glob('*.py')}!=set(freeze['new_source_sha256']):
        raise ValueError('Stage source set changed')
    for rel,digest in {**freeze['runtime_files'],**freeze['new_source_sha256']}.items():
        if sha(ROOT/rel)!=digest: raise ValueError('Runtime source changed')
    if os.name=='nt':
        if environment()!=cfg['environment']: raise ValueError('Windows runtime changed')
        from learning.task_entry_repair_v1.experiment import installed_files
        installed=installed_files()
        if sha(installed['rules'])!=cfg['engine_sha256'] or sha(installed['loader'])!=cfg['official_loader_sha256']:
            raise ValueError('Official runtime changed')
    return cfg


def verify_commit(out=OUT):
    out=Path(out).resolve(); cfg=verify(out); receipt=read(out/'pre-execution-commit.json'); commit=receipt['commit']
    clean_head(commit); git('merge-base','--is-ancestor',cfg['baseline_head'],commit)
    for name in ('root-decision.json','registration.json','freeze.json','pre-review.json'):
        if git('show',commit+':review/evidence/'+out.name+'/'+name)!=(out/name).read_bytes():
            raise ValueError('Committed evidence changed')
    for rel in read(out/'freeze.json')['new_source_sha256']:
        if git('show',commit+':kaggriculture/'+rel)!=(ROOT/rel).read_bytes(): raise ValueError('Committed source changed')
    return cfg


def ready(out=OUT):
    cfg=verify_commit(out); review=read(Path(out)/'pre-review.json'); receipt=read(Path(out)/'pre-execution-commit.json')
    if (review.get('passed') is not True or review.get('freeze_sha256')!=sha(Path(out)/'freeze.json')
        or receipt.get('freeze_sha256')!=sha(Path(out)/'freeze.json')
        or receipt.get('registration_sha256')!=sha(Path(out)/'registration.json')):
        raise ValueError('Review or commitment changed')
    return cfg
