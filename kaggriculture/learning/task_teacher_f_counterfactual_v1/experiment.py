"""First-stage screening only; completed candidates need a separate pair freeze.

Use the game venv with ``python -I -B``. No import reads private data. This
coordinator reuses the fixed runtime verifier, resource gate and owned-Job
backend; it never calls the old Anchor experiment or its failed attempt.
"""
import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import secrets
import shutil
import subprocess
import sys
import time
from types import ModuleType

REPO = Path(__file__).resolve().parents[3]
GAME = REPO.parent.parent/'kaggriculture'
BASE = 'e9901942781d074e45f16bde3c290d8c13c6495c'
PREFIX = 'kaggriculture/learning/task_teacher_f_counterfactual_v1/'
EVIDENCE = 'review/evidence/task-teacher-f-counterfactual-screen-v1-20261106'
REGISTRATION = EVIDENCE+'/registration.json'
PRIVATE = REPO.parent/'private/task-teacher-f-counterfactual-screen-v1-20261106'
STAGE = GAME/'artifacts/task-teacher-f-counterfactual-screen-v1-20261106'
INFRA = 'kaggriculture/learning/task_teacher_fresh_panel_v1/runner.py'
TEMPLATES = (
    ('2073577b50f0dbeece60b30a819b49cc9ba9051d',
     'review/evidence/task-teacher-fresh-panel-v1-20261105/registration.json','public_sources'),
    ('85391a37ee23f34d5a063a5bf3117818a21dd844',
     'kaggriculture/learning/task_teacher_counterfactual_pair_v1/registration.json','source_sha256'))
MAX_METADATA = 16*1024**2


def require(value):
    if not value: raise ValueError('F screening registration or evidence differs')


def sha(raw): return hashlib.sha256(raw).hexdigest()


def git(*args):
    result=subprocess.run(['git','-C',str(REPO),*args],stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,stderr=subprocess.DEVNULL,timeout=45,
        creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    require(result.returncode==0 and len(result.stdout)<=MAX_METADATA)
    return result.stdout


def infrastructure():
    """Reuse a previously published implementation, without altering its file."""
    raw=(REPO/INFRA).read_bytes()
    require(raw==git('show',BASE+':'+INFRA))
    module=ModuleType('_f_screen_infrastructure')
    module.__file__=str(REPO/INFRA)
    exec(compile(raw,module.__file__,'exec',dont_inherit=True),module.__dict__)
    return module


def required_sources(infra):
    paths={INFRA}
    for commit,relative,key in TEMPLATES:
        value=infra.decode(git('show',commit+':'+relative))
        require(type(value.get(key)) is dict)
        paths.update(value[key])
    paths.update(p.relative_to(REPO).as_posix() for p in (REPO/PREFIX).glob('*.py'))
    paths.add(EVIDENCE+'/REGISTRATION.zh-CN.md')
    return paths


def authenticate(commit):
    infra=infrastructure()
    infra.REGISTRATION,infra.PREFIX,infra.BASE=REGISTRATION,PREFIX,BASE
    infra._required_sources=lambda _old:required_sources(infra)
    infra.validate_registration=lambda value,contract:contract.validate_registration(value)
    reg,contract,identity,bootstrap,base=infra.authenticate(commit)
    require(not os.path.lexists(GAME/'learning/task_teacher_f_counterfactual_v1'))
    return infra,reg,contract,identity,bootstrap,base


def runtime(infra,bootstrap,reg):
    modules=infra._runtime(bootstrap,reg)
    for key,name in (('f_sources','sources'),('f_worker','worker')):
        module=importlib.import_module('learning.task_teacher_f_counterfactual_v1.'+name)
        require(Path(module.__file__).resolve(strict=True)==(REPO/PREFIX/(name+'.py')).resolve(strict=True))
        modules[key]=module
    from learning.task_teacher_fresh_panel_v1 import contract as readiness
    modules['readiness']=readiness
    return modules


def prepare():
    require(os.name=='nt' and sys.flags.isolated and sys.flags.dont_write_bytecode and __debug__)
    infra=infrastructure()
    require(git('rev-parse','HEAD').decode().strip()==BASE
        and not os.path.lexists(PRIVATE) and not os.path.lexists(STAGE)
        and not os.path.lexists(REPO/REGISTRATION))
    names=required_sources(infra)
    identities={name:sha(infra.read(REPO/name)) for name in sorted(names)}
    contract=infra.snapshot(PREFIX+'contract.py',identities[PREFIX+'contract.py'])
    base=infra.snapshot(infra.BASE_ENTRY,identities[infra.BASE_ENTRY])
    # No runtime or private source is loaded before stable readiness.
    old_contract='kaggriculture/learning/task_teacher_fresh_panel_v1/contract.py'
    readiness=infra.snapshot(old_contract,identities[old_contract])
    readiness.await_memory(base.physical_available)
    deadline=time.monotonic()+900
    def guard():
        if base.physical_available()<contract.LIMITS['running_available_bytes']:
            raise MemoryError('Preparation memory floor')
        if time.monotonic()>=deadline: raise TimeoutError('Preparation deadline')
        return True
    identity=infra.snapshot(infra.IDENTITY,identities[infra.IDENTITY])
    infra.preparation_runtime(identity,BASE)
    reg=dict(schema='teacher-f-counterfactual-screen-prereg-v1',base_commit=BASE,
        engine_commit=contract.ENGINE_COMMIT,engine_sha256=contract.ENGINE_SHA,
        public_sources=identities,versions=infra._versions(),limits=contract.LIMITS,
        screening=contract.screening_contract(),phase_write_bytes=contract.PHASE_BYTES,
        pair_execution_authorized=False,training=False,gpu=False,
        independent_evaluation=False,promotion=False)
    bootstrap=infra.snapshot(infra.BOOTSTRAP,identities[infra.BOOTSTRAP])
    modules=runtime(infra,bootstrap,reg)
    context=modules['f_sources'].open_context(guard=guard)
    try:
        plans=context.private_source_plans()
        require(len(plans)==8)
        plans=[p.document() for p in plans]
        binding=dict(schema='teacher-f-screen-binding-v1',base_commit=BASE,salt=secrets.token_hex(32),
            completed_sources=context.identities,source_plans=plans,
            default_sha256=sha(infra.read(GAME/'main.py')))
    finally:
        context.close()
    guard()
    require(all(sha(infra.read(REPO/p))==h for p,h in identities.items()))
    PRIVATE.mkdir(exist_ok=False)
    reg['binding_commitment_sha256']=infra.write(PRIVATE/'binding.local.json',binding)
    contract.validate_registration(reg)
    (REPO/EVIDENCE).mkdir(parents=True,exist_ok=True)
    infra.write(REPO/REGISTRATION,reg)
    return dict(status='prepared_no_screening',authenticated_old_games=32,fixed_sources=8,
        model_forwards=0,official_transitions=0,new_complete_games=0)


def marker(kind,commit,binding_sha,index=None):
    require(kind in ('attempt','permit','started','consumed'))
    value=dict(schema='teacher-f-screen-marker-v1',kind=kind,registration_commit=commit,
        binding_sha256=binding_sha,retry_allowed=False)
    if kind!='attempt':
        require(type(index) is int and 0<=index<8)
        value['source_index']=index
    return value


def name(index):
    require(type(index) is int and 0<=index<8)
    return f'source-{index:02d}'


def validate_result(value,source,plan,contract,worker,context):
    require(type(value) is dict and set(value)=={'source','source_plan','coverage','candidate'}
        and contract.source_index(value['source'])==contract.source_index(source)
        and contract.same(value['source_plan'],plan.document())
        and type(value['coverage']) is dict
        and value['coverage']['source_identity_sha256']==plan.identity_sha256)
    if value['candidate'] is not None:
        worker.validate_case(value['candidate'],context)
        candidate=value['candidate']
        coverage=value['coverage']
        window,rank=candidate['window'],candidate['rank']
        chosen=coverage['selection_plan'][window][rank]
        require(chosen==dict(step=candidate['step'],ordinal=candidate['ordinal'])
            and candidate['source_identity_sha256']==plan.identity_sha256)
    return value


def run(commit,*,preflight=False,source_index=None):
    infra,reg,contract,identity,bootstrap,base=authenticate(commit)
    limits=contract.LIMITS
    raw=infra.read(PRIVATE/'binding.local.json');binding=infra.decode(raw)
    require(sha(raw)==reg['binding_commitment_sha256']
        and binding.get('schema')=='teacher-f-screen-binding-v1' and binding.get('base_commit')==BASE)
    child=source_index is not None
    require(not (preflight and child))
    if child: name(source_index)
    started=time.monotonic()
    latest=dict(available_bytes=None,sample_monotonic=None)
    def available():
        value=base.physical_available()
        latest.update(available_bytes=value,sample_monotonic=time.monotonic())
        return value
    import psutil
    owner=(infra.decode(infra.read(PRIVATE/'owner.local.json')) if child else
        dict(schema='task-pair-fresh-multi-owner-v1',registration_commit=commit,
            binding_sha256=sha(raw),pid=os.getpid(),created_us=int(psutil.Process().create_time()*1_000_000),
            deadline=started+limits['screen_wall_seconds']))
    def guard():
        if available()<limits['running_available_bytes']: raise MemoryError('Screening memory floor')
        if time.monotonic()>=owner['deadline']: raise TimeoutError('Screening deadline')
        return True
    old_contract='kaggriculture/learning/task_teacher_fresh_panel_v1/contract.py'
    readiness=infra.snapshot(old_contract,reg['public_sources'][old_contract],commit)
    def ready():
        return readiness.await_memory(available,timeout=min(limits['readiness_timeout_seconds'],owner['deadline']-time.monotonic()))
    owned=False;results=[];accepted=[];jobs=[];active_index=None;context=None;again=None
    try:
        if not child:
            require(not os.path.lexists(PRIVATE/'attempt.local.json')
                and not os.path.lexists(PRIVATE/'owner.local.json')
                and not os.path.lexists(PRIVATE/'closed.local.json') and not os.path.lexists(STAGE))
            if not preflight:
                infra.write(PRIVATE/'attempt.local.json',marker('attempt',commit,sha(raw)))
                owned=True
                infra.write(PRIVATE/'owner.local.json',owner)
            ready()
        else:
            require(not os.path.lexists(PRIVATE/'closed.local.json'))
            guard()
        modules=runtime(infra,bootstrap,reg)
        if child:
            modules['fresh'].verify_live_owner(owner,commit=commit,binding_sha=sha(raw),
                current_pid=os.getpid(),process=psutil.Process,executable=sys.executable,
                samefile=os.path.samefile,now=time.monotonic())
            require(infra.decode(infra.read(PRIVATE/'attempt.local.json'))==marker('attempt',commit,sha(raw)))
        context=modules['f_sources'].open_context(guard=guard,expected_identities=binding['completed_sources'],
            verify_game_files=not child)
        plans=context.private_source_plans()
        require([p.document() for p in plans]==binding['source_plans']
            and sha(infra.read(GAME/'main.py'))==binding['default_sha256'])
        guard()
        require(shutil.disk_usage(GAME/'artifacts').free>=limits['disk_floor_bytes']+limits['screen_output_bytes'])
        if preflight:
            return dict(status='ready',screened_sources=0,new_complete_games=0)
        if child:
            folder=STAGE/name(source_index)
            permit=infra.decode(infra.read(folder/'permit.local.json'))
            require(permit==marker('permit',commit,sha(raw),source_index))
            for kind in ('started','consumed'):
                infra.write(folder/(kind+'.local.json'),marker(kind,commit,sha(raw),source_index))
            meter=modules['f_sources'].screening_module().Meter()
            try:
                coverage,candidate,plan=modules['f_sources'].screen_source(context,
                    dict(contract.SOURCES[source_index]),guard,meter=meter)
                result=dict(source=dict(contract.SOURCES[source_index]),source_plan=plan.document(),
                    coverage=coverage,candidate=candidate)
                validate_result(result,contract.SOURCES[source_index],plans[source_index],contract,
                    modules['f_worker'],context)
                guard()
                identity.verify_public(repo=REPO,game_root=GAME,registration_commit=commit)
                require(infra.read(PRIVATE/'binding.local.json')==raw
                    and sha(infra.read(GAME/'main.py'))==binding['default_sha256'])
                infra.write(folder/'result.local.json',result,512*1024)
            except BaseException as exc:
                infra.write(folder/'failed.local.json',dict(error_kind=type(exc).__name__,
                    meter=meter.snapshot(),retry_allowed=False,
                    failure_observation=dict(latest)),512*1024)
                raise
            return None
        STAGE.mkdir(exist_ok=False)
        for index,source in enumerate(contract.SOURCES):
            gate=ready();guard()
            require(owner['deadline']-time.monotonic()>limits['job_wall_seconds']+limits['cleanup_seconds']
                and modules['driver']._owned_size(STAGE)+1024**2<=limits['screen_output_bytes']
                and shutil.disk_usage(STAGE).free>=limits['disk_floor_bytes']+limits['screen_output_bytes'])
            folder=STAGE/name(index);folder.mkdir(exist_ok=False)
            infra.write(folder/'readiness.local.json',gate)
            infra.write(folder/'permit.local.json',marker('permit',commit,sha(raw),index))
            active_index=index
            command=[sys.executable,'-I','-B',str(Path(__file__).resolve()),
                     '--registration-commit',commit,'--source-index',str(index)]
            job=infra.supervise_recorded(modules['campaign'],command,GAME,
                dict(max_rss_bytes=limits['max_rss_bytes'],max_commit_bytes=limits['max_commit_bytes'],
                    min_available_bytes=limits['running_available_bytes'],wall_seconds=limits['job_wall_seconds'],
                    cleanup_seconds=limits['cleanup_seconds'],poll_seconds=limits['poll_seconds']))
            job_sha=infra.write(folder/'supervision.local.json',job,512*1024)
            jobs.append(dict(source_index=index,sha256=job_sha,launcher_reaped=job.get('launcher_reaped'),
                job_members_zero=job.get('job_members_zero'),cleanup_error_kind=job.get('cleanup_error_kind')))
            require(modules['acceptance'].supervision_ok(job));guard()
            data=infra.read(folder/'result.local.json',512*1024)
            result=validate_result(infra.decode(data),source,plans[index],contract,modules['f_worker'],context)
            for kind in ('permit','started','consumed'):
                require(infra.decode(infra.read(folder/(kind+'.local.json')))==marker(kind,commit,sha(raw),index))
            proof=dict(schema='teacher-f-screen-accepted-v1',source_index=index,passed=True,
                registration_commit=commit,binding_sha256=sha(raw),result_sha256=sha(data),
                supervision_sha256=job_sha,retry_allowed=False)
            accepted.append(dict(source_index=index,sha256=infra.write(folder/'accepted.local.json',proof)))
            results.append(result);active_index=None
        projection=contract.screen_projection(results)
        again=modules['f_sources'].open_context(guard=guard,expected_identities=binding['completed_sources'])
        require([p.document() for p in again.private_source_plans()]==binding['source_plans']
            and sha(infra.read(GAME/'main.py'))==binding['default_sha256']
            and infra.read(PRIVATE/'binding.local.json')==raw)
        identity.verify_public(repo=REPO,game_root=GAME,registration_commit=commit)
        guard()
        result_sha=infra.write(PRIVATE/'screen-result.local.json',dict(schema='teacher-f-screen-result-v1',
            registration_commit=commit,binding_sha256=sha(raw),results=results,accepted=accepted,
            pair_execution_authorized=False))
        projection_sha=infra.write(PRIVATE/'projected-summary.local.json',projection)
        infra.write(PRIVATE/'screen-seal.local.json',dict(schema='teacher-f-screen-seal-v1',
            registration_commit=commit,binding_sha256=sha(raw),screen_result_sha256=result_sha,
            projection_sha256=projection_sha,accepted=accepted,pair_execution_authorized=False))
        require(modules['driver']._owned_size(STAGE)<=limits['screen_output_bytes']
            and modules['driver']._owned_size(PRIVATE)<=limits['private_result_bytes'])
        return dict(status='passed',screened_sources=8,candidate_count=projection['candidate_count'],
            official_transitions=projection['counters']['official_transitions_confirmed_completed'],
            new_complete_games=0,pair_execution_authorized=False,training=False)
    finally:
        for current in (context,again):
            if current is not None:current.close()
        if owned:
            error=sys.exc_info()[1]
            memory=infra.closing_memory(error,latest,available)
            infra.write(PRIVATE/'closed.local.json',dict(schema='teacher-f-screen-closed-v1',
                registration_commit=commit,binding_sha256=sha(raw),
                outcome='completed' if error is None else 'failed',
                error_kind=None if error is None else type(error).__name__,accepted_sources=len(accepted),
                active_index=active_index,elapsed_seconds=time.monotonic()-started,
                child_cleanup_receipts=jobs,retry_allowed=False,**memory))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare',action='store_true')
    parser.add_argument('--preflight',action='store_true')
    parser.add_argument('--registration-commit')
    parser.add_argument('--source-index',type=int)
    args=parser.parse_args()
    try:
        require(not(args.prepare and (args.preflight or args.registration_commit or args.source_index is not None)))
        value=prepare() if args.prepare else run(args.registration_commit,
            preflight=args.preflight,source_index=args.source_index)
        if value is not None:print(json.dumps(value))
        return 0
    except BaseException as exc:
        kind='memory_gate' if isinstance(exc,MemoryError) else 'time_gate' if isinstance(exc,TimeoutError) else 'invalid_or_runtime'
        print(json.dumps(dict(status='failed',category=kind,retry_allowed=False)),file=sys.stderr)
        return 1


if __name__=='__main__':raise SystemExit(main())
