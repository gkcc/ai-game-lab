"""Frozen prospective observation; historical attempts remain closed."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
PUBLIC = ROOT.parent / 'publication/ai-game-lab'
OUT = ROOT / 'artifacts/task-action-observation-v1-20261027'
SOURCE = Path(__file__).resolve().parent
PREVIOUS = ROOT / 'artifacts/task-anchor-reentry-v1-20261027'
REPAIR = ROOT / 'artifacts/task-anchor-spawn-repair-v1-20261027'
CLOSED = ROOT / 'artifacts/task-launcher-evidence-repair-v1-20261027'
BASE = '20726f44279e9a46d363ff74ba0bc9102be51ad5'
START = '2026-10-27T01:40:49+00:00'
DEADLINE = '2026-10-27T03:10:49+00:00'
MIB = 1024**2
GIB = 1024**3
FIXTURE = {'schema':'task-action-observation-entry-fixture-v1',
           'marker':'fixed-model-free-analysis-entry-17'}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    result=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for data in iter(lambda:stream.read(MIB),b''):result.update(data)
    return result.hexdigest()


def write_new(path,value):
    path=Path(path).resolve()
    if not path.is_relative_to(OUT):raise ValueError('Write outside new observation directory')
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8',newline='\n') as stream:
        json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False);stream.write('\n')


def deadline_check():
    if datetime.now(timezone.utc)>=datetime.fromisoformat(DEADLINE):
        raise TimeoutError('Observation total stage deadline reached')


def git(*args):
    return subprocess.check_output(['git',*args],cwd=PUBLIC)


def source_inventory():
    """Conservative seed exclusion: metadata only, never replay/trace arrays."""
    from learning.bc_crop.prepare import TEXT_SUFFIXES
    tracked=git('ls-files','-z').decode().split('\0')[:-1]
    paths=[('public/'+n,PUBLIC/n) for n in tracked if Path(n).suffix.lower() in TEXT_SUFFIXES]
    paths += [('local/'+p.relative_to(ROOT/'artifacts').as_posix(),p)
        for p in sorted((ROOT/'artifacts').rglob('*.json'))
        if not p.is_relative_to(OUT) and not p.name.lower().startswith('replay')
        and (re.search(r'(registration|manifest|ledger|config|result|summary|seed|attempt)',p.name,re.I)
             or 'job-inputs' in p.parts)]
    found=set();identities=[];used=0;pattern=re.compile(r'(?<!\d)\d{1,10}(?!\d)')
    for label,path in paths:
        deadline_check();used+=path.stat().st_size
        if used>256*MIB:raise ValueError('Seed metadata inventory exceeds declared 256 MiB')
        data=path.read_bytes()
        found.update(int(m.group()) for m in pattern.finditer(data.decode('utf-8-sig')) if int(m.group())<2**32)
        identities.append([label,hashlib.sha256(data).hexdigest()])
    selected=[];draws=0
    while len(selected)<2:
        data=hashlib.sha256(('task-action-observation-v1:20261027:'+str(draws)).encode()).digest()
        value=int.from_bytes(data[:4],'big');draws+=1
        if value not in found and value not in selected:selected.append(value)
    return dict(seeds=selected,draws=draws,input_bytes=used,files=identities,
        excluded_uint32=sorted(found),scope='Tracked public text and seed-bearing local metadata; no raw Replay, trace or tensor reads',
        limitation='No assertion about deleted, unavailable or external seed history')


def prepare(out=OUT):
    out=Path(out).resolve();deadline_check()
    if out!=OUT or (out/'registration.json').exists():raise ValueError('Registration is exclusive')
    if git('status','--porcelain').strip() or git('rev-parse','HEAD').decode().strip()!=BASE:
        raise ValueError('Unexpected public baseline or uncommitted work')
    advice=read(out/'independent-next-step.json');decision=read(out/'root-decision.json')
    if decision['status']!='approved_for_implementation_not_execution':raise ValueError('Root decision missing')
    if decision['advice_sha256']!=sha(out/'independent-next-step.json'):raise ValueError('Advice changed')
    old=read(PREVIOUS/'registration.json');oldfreeze=read(PREVIOUS/'freeze.json')
    if not read(REPAIR/'root-result-acceptance.json')['passed']:raise ValueError('Accepted POSIX launcher evidence missing')
    from learning.task_entry_repair_v1.experiment import installed_files
    from learning.task_reward_gate_v1.experiment import versions
    installed=installed_files()
    if sha(installed['rules'])!=old['engine_sha256'] or sha(installed['loader'])!=old['official_loader_sha256']:
        raise ValueError('Official Windows engine changed')
    if versions()!=old['runtime_versions']:raise ValueError('Official job runtime changed')
    runtime=dict(oldfreeze['runtime_files'])
    for rel,digest in runtime.items():
        if sha(ROOT/rel)!=digest:raise ValueError('Existing frozen runtime changed: '+rel)
    extra=['learning/task_action_impact_diagnostic_v1/analysis.py',
           'learning/task_action_impact_diagnostic_v1/__init__.py',
           'learning/task_launcher_contract_repair_v1/observer.py',
           'learning/task_launcher_contract_repair_v1/experiment.py',
           'learning/task_launcher_contract_repair_v1/__init__.py']
    for rel in extra:runtime[rel]=sha(ROOT/rel)
    new={p.relative_to(ROOT).as_posix():sha(p) for p in SOURCE.glob('*.py')}
    runtime.update(new)
    models={old['task_checkpoint']:old['task_checkpoint_sha256'],
            old['crop_checkpoint']:old['crop_checkpoint_sha256'],
            (PREVIOUS/'model/trained.npz').relative_to(ROOT).as_posix():
                read(PREVIOUS/'model-freeze.json')['checkpoint_sha256']}
    for rel,digest in models.items():
        if sha(ROOT/rel)!=digest:raise ValueError('Frozen model identity changed')
    if shutil.disk_usage(ROOT).free<8*GIB:raise ValueError('Insufficient declared output reserve')
    inventory=source_inventory();write_new(out/'seed-inventory.local.json',inventory)
    for name,digest in old['source_sha256'].items():
        source=PREVIOUS/'sources'/name
        if sha(source)!=digest:raise ValueError('Frozen opponent changed')
        target=out/'sources'/name;target.parent.mkdir(parents=True,exist_ok=True)
        with target.open('xb') as stream:stream.write(source.read_bytes())
    launch=read(PREVIOUS/'gpu-launch.local.json')
    write_new(out/'wsl-launch.local.json',launch)
    write_new(out/'entry-fixture/input-manifest.json',FIXTURE)
    candidate=(PREVIOUS/'model/trained.npz').relative_to(ROOT).as_posix()
    cfg=dict(experiment='task-action-observation-v1',created_utc=datetime.now(timezone.utc).isoformat(),
        baseline_head=BASE,task_started_utc=START,task_deadline_utc=DEADLINE,total_stage_seconds=5400,
        data_role='Fresh development observation only; no promotion, tuning, confidence interval or independent holdout claim',
        validation_seeds=inventory['seeds'],test_seeds=[],opponents=['round2','market'],seats=[0,1],
        arms=['baseline','candidate','teacher_task'],
        seed_generation=dict(method='First four SHA256 bytes big-endian of task-action-observation-v1:20261027:index',
                             draws=inventory['draws'],rejection='Only prior metadata matches and duplicate values'),
        seed_inventory_sha256=sha(out/'seed-inventory.local.json'),
        seed_inventory_bytes=inventory['input_bytes'],seed_inventory_scope=inventory['scope'],
        task_checkpoint=old['task_checkpoint'],task_checkpoint_sha256=old['task_checkpoint_sha256'],
        crop_checkpoint=old['crop_checkpoint'],crop_checkpoint_sha256=old['crop_checkpoint_sha256'],
        candidate_checkpoint=candidate,candidate_checkpoint_sha256=models[candidate],
        source_sha256=old['source_sha256'],runtime_versions=old['runtime_versions'],
        engine_commit=old['engine_commit'],engine_sha256=old['engine_sha256'],official_loader_sha256=old['official_loader_sha256'],
        analysis_python=old['gpu_environment']['python'],wsl_config_sha256=sha(out/'wsl-launch.local.json'),
        game_limits=old['game_limits'],max_games=24,max_workers=2,per_job_timeout_seconds=900,campaign_seconds=2700,
        task_forward_budget=128000,crop_forward_budget=384000,neural_input_row_budget=7680000,policy_turn_budget=69024,
        live_transition_limit=17256,primary_reexecution_limit=17256,independent_reexecution_limit=1438,
        total_official_transition_limit=35950,
        storage=dict(total_bytes=4*GIB,game_write_bytes=128*MIB,primary_parent_write_bytes=64*MIB,
                     independent_reserve_bytes=64*MIB,free_floor_bytes=4*GIB,process_log_bytes=MIB,small_receipt_bytes=16384),
        analysis_limits=dict(work_seconds=420,process_seconds=450,target_seconds=360,cpu_seconds=300,
            address_space_bytes=2*GIB,output_bytes=128*MIB,parent_receipt_bytes=65536,
            compressed_bytes=GIB,decoded_bytes=12*GIB),
        fixture_limits=dict(work_seconds=45,process_seconds=60,target_seconds=30),
        transport_limits=dict(fixture_seconds=90,analysis_seconds=480,cleanup_reserve_seconds=20),
        acceptance=dict(required_games=24,required_states=720,required_action_turns=719,required_triads=8,
            required_comparisons=16,required_raw_files=48,official_errors=0,source_or_identity_errors=0,
            extra_unplanned_games=0,scientific_inference='Descriptive mechanism localization; economic causality remains unproven'),
        independent_selection='First new development seed, market, seat0, candidate and teacher_task',
        research_question='Where do frozen task-choice differences first reach complete requests and public observable state; what are the paired final cash accounts?',
        closed_windows_branch_reopened=False,old_192_raw_reads=0,training_updates=0,policy_changes=False,
        prior_attempt_refunded=False,default_changed=False,automatic_admission=False,
        review_source_metadata_budget_bytes=20*MIB,official_workflow='Existing Windows official jobs plus accepted POSIX pure-analysis supervision',
        local_registration_not_third_party_timestamp=True)
    write_new(out/'registration.json',cfg)
    protected={n:sha(ROOT/n) for n in ('main.py','bots/teacher_fertilizer_terminal_v2.py')}
    write_new(out/'freeze.json',dict(registration_sha256=sha(out/'registration.json'),runtime_files=runtime,
        new_source_sha256=new,model_sha256=models,protected_source_sha256=protected,
        root_decision_sha256=sha(out/'root-decision.json'),advice_sha256=sha(out/'independent-next-step.json'),
        accepted_launcher_sha256=sha(REPAIR/'root-result-acceptance.json'),
        fixture_input_sha256=sha(out/'entry-fixture/input-manifest.json')))
    return verify(out)


def verify(out=OUT):
    out=Path(out).resolve();deadline_check()
    if out!=OUT:raise ValueError('Unexpected observation root')
    cfg,f=read(out/'registration.json'),read(out/'freeze.json')
    if sha(out/'registration.json')!=f['registration_sha256']:raise ValueError('Registration changed')
    for rel,digest in f['runtime_files'].items():
        if sha(ROOT/rel)!=digest:raise ValueError('Frozen source changed: '+rel)
    for rel,digest in f['protected_source_sha256'].items():
        if sha(ROOT/rel)!=digest:raise ValueError('Protected Bot changed')
    for name,digest in cfg['source_sha256'].items():
        if sha(out/'sources'/name)!=digest:raise ValueError('Opponent source changed')
    for name,key in [('root-decision.json','root_decision_sha256'),('independent-next-step.json','advice_sha256')]:
        if sha(out/name)!=f[key]:raise ValueError('Frozen decision changed')
    if sha(out/'seed-inventory.local.json')!=cfg['seed_inventory_sha256']:raise ValueError('Seed inventory changed')
    if sha(out/'wsl-launch.local.json')!=cfg['wsl_config_sha256']:raise ValueError('WSL transport identity changed')
    if sha(out/'entry-fixture/input-manifest.json')!=f['fixture_input_sha256']:raise ValueError('Fixture changed')
    if os.name=='nt':
        for rel,digest in f['model_sha256'].items():
            if sha(ROOT/rel)!=digest:raise ValueError('Model changed')
        from learning.task_entry_repair_v1.experiment import installed_files
        from learning.task_reward_gate_v1.experiment import versions
        actual=installed_files()
        if versions()!=cfg['runtime_versions'] or sha(actual['rules'])!=cfg['engine_sha256'] or sha(actual['loader'])!=cfg['official_loader_sha256']:
            raise ValueError('Official runtime changed')
    else:
        import platform
        if platform.python_version()!=cfg['analysis_python']:raise ValueError('Accepted analysis interpreter changed')
    return cfg


def ready(out=OUT):
    out=Path(out).resolve();cfg=verify(out)
    review,receipt=read(out/'pre-review.json'),read(out/'pre-execution-commit.json')
    if not review['passed'] or review['freeze_sha256']!=sha(out/'freeze.json'):
        raise ValueError('Exact source has no independent pre-review')
    if receipt['freeze_sha256']!=sha(out/'freeze.json') or receipt['registration_sha256']!=sha(out/'registration.json'):
        raise ValueError('Pre-execution binding changed')
    # The Windows preregistration dispatcher verifies exact committed bytes.
    # POSIX payload never imports or reads models merely to prove registration.
    if os.name=='nt':
        for name in ('registration.json','freeze.json','pre-review.json'):
            if git('show',receipt['commit']+':review/evidence/'+out.name+'/'+name)!=(out/name).read_bytes():
                raise ValueError('Local preregistration commit differs')
    return cfg


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['prepare','verify']);args=parser.parse_args()
    cfg=prepare() if args.command=='prepare' else verify()
    print(json.dumps(dict(experiment=cfg['experiment'],seeds=cfg['validation_seeds'],passed=True)))
