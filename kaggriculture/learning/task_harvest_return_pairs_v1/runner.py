"""One registered four-pair campaign; no training or fresh seeds."""
import argparse
import importlib
import json
import os
from pathlib import Path
import re
import secrets
import shutil
import subprocess
import sys
import time
from types import ModuleType


REPO=Path(__file__).resolve().parents[3]
GAME=REPO.parent.parent/'kaggriculture'
BASE='317914328a81d5d26581791481140d9eea6438ea'
PRIOR_REG_COMMIT='5f5d6bb8e37df9210d1f623ceb5c0c48f71ce2a6'
PREFIX='kaggriculture/learning/task_harvest_return_pairs_v1/'
EVIDENCE='review/evidence/task-harvest-return-pairs-v1-20261106'
REGISTRATION,NOTE=EVIDENCE+'/registration.json',EVIDENCE+'/REGISTRATION.zh-CN.md'
PRIVATE=REPO.parent/'private/task-harvest-return-pairs-v1-20261106'
STAGE=GAME/'artifacts/task-harvest-return-pairs-v1-20261106'
SCHEMA='task-harvest-return-pairs-v1'
PRIOR_REG='review/evidence/task-harvest-model-dev-v1-20261106/registration.json'
PRIOR_RESULTS='review/evidence/task-harvest-model-dev-v1-result-20261106'
_SHARED=None


def require(value):
    if not value:raise ValueError('Registered terminal return-pair campaign differs')


def git(*args):
    result=subprocess.run(['git','-C',str(REPO),*args],stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,stderr=subprocess.DEVNULL,timeout=45,
        creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    require(result.returncode==0 and len(result.stdout)<=16*1024**2)
    return result.stdout


def shared():
    """Reuse the frozen storage, identity and resource helpers verbatim."""
    global _SHARED
    if _SHARED is None:
        path='kaggriculture/learning/task_harvest_model_dev_v1/runner.py'
        raw=(REPO/path).read_bytes();require(raw==git('show',BASE+':'+path))
        module=ModuleType('_return_pairs_frozen_helpers');module.__file__=str(REPO/path)
        exec(compile(raw,module.__file__,'exec',dont_inherit=True),module.__dict__)
        module.BASE=BASE;module.PRIVATE=PRIVATE
        _SHARED=module
    return _SHARED


def sha(raw):
    import hashlib
    return hashlib.sha256(raw).hexdigest()


def canonical(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=True,allow_nan=False).encode()


def source_map(infra):
    old=infra.decode(git('show',PRIOR_REG_COMMIT+':'+PRIOR_REG))
    require(len(old['public_sources'])==551)
    extras={PRIOR_REG,NOTE,PREFIX+'README.zh-CN.md'}
    extras.update(PRIOR_RESULTS+'/'+name for name in ('summary.json','certificate.json',
        'RESULTS.zh-CN.md','REVIEW.zh-CN.md','NEXT.zh-CN.md'))
    paths=set(old['public_sources'])|extras
    paths.update(p.relative_to(REPO).as_posix() for p in (REPO/PREFIX).glob('*.py'))
    required=('__init__.py','contract.py','selection.py','policy.py','core.py','sources.py','worker.py','runner.py',
        'test_contract.py','test_selection.py','test_policy.py','test_core.py','test_sources.py','test_worker.py','test_runner.py')
    require({PREFIX+n for n in required}.issubset(paths))
    for path,expected in old['public_sources'].items():
        raw=infra.read(REPO/path)
        if path=='review/PROGRESS.md':
            original=git('show',PRIOR_REG_COMMIT+':'+path)
            require(sha(original)==expected and raw==git('show',BASE+':'+path) and raw.startswith(original))
        else:require(sha(raw)==expected)
    for path in extras-{NOTE,PREFIX+'README.zh-CN.md'}:
        require(infra.read(REPO/path)==git('show',BASE+':'+path))
    return {path:sha(infra.read(REPO/path)) for path in sorted(paths)}


def validate_registration(reg,contract):
    require(reg.get('schema')==SCHEMA+'-prereg' and reg.get('base_commit')==BASE
        and canonical(reg.get('limits'))==canonical(contract.LIMITS)
        and canonical(reg.get('specification'))==canonical(contract.specification())
        and reg.get('engine_commit')==contract.ENGINE_COMMIT and reg.get('engine_sha256')==contract.ENGINE_SHA
        and type(reg.get('binding_commitment_sha256')) is str
        and re.fullmatch('[0-9a-f]{64}',reg['binding_commitment_sha256']))


def authenticate(commit):
    infra=shared().infrastructure()
    infra.REGISTRATION,infra.PREFIX,infra.BASE=REGISTRATION,PREFIX,BASE
    infra._required_sources=lambda _old:set(source_map(infra))
    infra.validate_registration=validate_registration
    reg,contract,identity,bootstrap,base=infra.authenticate(commit)
    require(not os.path.lexists(GAME/'learning/task_harvest_return_pairs_v1'))
    return infra,reg,contract,identity,bootstrap,base


def metadata_modules():
    require('numpy' not in sys.modules and 'kaggle_environments' not in sys.modules)
    os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
    sys.path.insert(0,str(REPO/'kaggriculture'))
    result={key:importlib.import_module('learning.'+name) for key,name in (
        ('source','task_harvest_return_pairs_v1.sources'),('driver','task_pair_dev_campaign_v1.driver'))}
    require('kaggle_environments' not in sys.modules and 'torch' not in sys.modules)
    return result


def runtime(infra,bootstrap,reg):
    result=infra._runtime(bootstrap,reg)
    for key,name in (('source','sources'),('return_worker','worker')):
        module=importlib.import_module('learning.task_harvest_return_pairs_v1.'+name)
        require(Path(module.__file__).resolve(strict=True)==(REPO/PREFIX/(name+'.py')).resolve(strict=True))
        result[key]=module
    return result


def private_write(infra,name,value,*,closure=False):
    return shared().private_write(infra,name,value,closure=closure)


def close(infra,name,value,primary):return shared().close(infra,name,value,primary)


def prepare():
    helpers=shared();infra=helpers.infrastructure()
    require(os.name=='nt' and sys.flags.isolated and sys.flags.dont_write_bytecode and __debug__
        and git('rev-parse','HEAD').decode().strip()==BASE
        and all(not os.path.lexists(p) for p in (PRIVATE,STAGE,REPO/REGISTRATION)))
    sources=source_map(infra)
    contract=infra.snapshot(PREFIX+'contract.py',sources[PREFIX+'contract.py'])
    base=infra.snapshot(infra.BASE_ENTRY,sources[infra.BASE_ENTRY])
    readiness=infra.snapshot('kaggriculture/learning/task_teacher_fresh_panel_v1/contract.py',
        sources['kaggriculture/learning/task_teacher_fresh_panel_v1/contract.py'])
    deadline=time.monotonic()+contract.LIMITS['preparation_wall_seconds']
    def check():
        if base.physical_available()<contract.LIMITS['running_available_bytes']:raise MemoryError('Preparation floor')
        if time.monotonic()>=deadline:raise TimeoutError('Preparation deadline')
        return True
    readiness.await_memory(base.physical_available,floor=contract.LIMITS['initial_available_bytes'])
    identity=infra.snapshot(infra.IDENTITY,sources[infra.IDENTITY])
    infra.preparation_runtime(identity,BASE)
    rules=helpers.official_identity(infra,contract)
    PRIVATE.mkdir(exist_ok=False)
    private_write(infra,'prepare-started.local.json',dict(base_commit=BASE,public_sources=sources,
        specification=contract.specification()))
    binding_sha=None
    try:
        modules=metadata_modules();source,dm=modules['source'],modules['driver']
        archive=source.archive(infra,dm,check,full=True);source.assets_verify(dm,archive,check)
        selections=source.select_cases(dm,archive,check,lambda n,v:private_write(infra,n,v))
        require(len(selections)==4)
        selection_hashes={s['role']:sha(infra.read(PRIVATE/(s['role']+'.selection.local.json'))) for s in selections}
        binding=dict(schema=SCHEMA+'-binding',base_commit=BASE,salt=secrets.token_hex(32),
            archive=archive,selections=selection_hashes,default_sha256=archive['default_sha256'])
        binding_sha=private_write(infra,'binding.local.json',binding)
        reg=dict(schema=SCHEMA+'-prereg',base_commit=BASE,engine_commit=contract.ENGINE_COMMIT,
            engine_sha256=contract.ENGINE_SHA,installed_source_sha256=rules,public_sources=sources,
            versions=infra._versions(),limits=contract.LIMITS,specification=contract.specification(),
            binding_commitment_sha256=binding_sha)
        validate_registration(reg,contract)
        require(check() and all(sha(infra.read(REPO/p))==h for p,h in sources.items()))
        infra.write(REPO/REGISTRATION,reg)
    finally:
        error=sys.exc_info()[1]
        close(infra,'prepare-closed.local.json',dict(outcome='prepared' if error is None else 'failed',
            error_kind=None if error is None else type(error).__name__,binding_sha256=binding_sha,
            source_trace_scans=4 if error is None else None,source_replay_decodes=0,
            official_constructions=0,official_actions=0,model_forwards=0,retry_allowed=False),error)
    return dict(status='prepared',sealed_old_files=contract.PRIOR_FILES,source_trace_scans=4,
        source_replay_decodes=0,model_forwards=0,official_actions=0,frozen_sources=len(sources),cases_registered=4)


def marker(kind,commit,binding_sha,role=None):
    require(kind in ('attempt','permit','started','consumed'))
    value=dict(schema=SCHEMA+'-marker',kind=kind,registration_commit=commit,
        binding_sha256=binding_sha,retry_allowed=False)
    if kind!='attempt':
        require(role in ('F01','F02','F03','F04'));value['role']=role
    return value


def binding_read(infra,reg):
    raw=infra.read(PRIVATE/'binding.local.json');binding=infra.decode(raw)
    ready=infra.decode(infra.read(PRIVATE/'prepare-closed.local.json'))
    require(sha(raw)==reg['binding_commitment_sha256']==ready['binding_sha256']
        and binding['schema']==SCHEMA+'-binding' and binding['base_commit']==BASE
        and ready['outcome']=='prepared' and ready['error_kind'] is None
        and set(binding['selections'])=={'F01','F02','F03','F04'})
    return raw,binding


def selected_read(infra,binding,source):
    raw=infra.read(PRIVATE/(source['role']+'.selection.local.json'));value=infra.decode(raw)
    require(sha(raw)==binding['selections'][source['role']]
        and value['role']==source['role'] and canonical(value['source'])==canonical(source))
    return value


def aggregate_costs(records,contract):
    require(len(records)==4 and [r['row']['role'] for r in records]==list(contract.ROLES))
    keys=('task_forwards','task_rows','crop_forwards','crop_rows','gate_evaluations','controller_calls','teacher_proposals')
    total={k:0 for k in keys};paired=0;steps=0
    for record in records:
        cost=record['model_costs'];require(type(cost) is dict and set(cost)==set(keys)
            and all(type(v) is int and v>=0 for v in cost.values()))
        active=record['row']['status']=='paired';require(record['row']['status'] in ('paired','no_eligible_target'))
        require(cost['controller_calls']==1438*active)
        if not active:require(not any(cost.values()) and record['official_actions']==0)
        else:require(type(record['official_actions']) is int and 0<record['official_actions']<=1438
            and record['official_actions']%2==0)
        for k,v in cost.items():total[k]+=v
        paired+=active;steps+=record['official_actions']
    require(total['controller_calls']==1438*paired<=contract.LIMITS['max_controller_calls']
        and total['gate_evaluations']==total['teacher_proposals']==0
        and total['task_forwards']<=contract.LIMITS['max_task_forwards']
        and total['crop_forwards']<=contract.LIMITS['max_crop_forwards']
        and total['task_rows']+total['crop_rows']<=contract.LIMITS['max_neural_input_rows']
        and steps<=contract.LIMITS['max_official_action_transitions'])
    return dict(model_costs=total,paired=paired,terminal_suffixes=2*paired,official_actions=steps,
        environment_constructions=2*paired,restores=2*paired,explicit_resets=0,rival_calls=1438*paired,
        source_trace_scans=2*paired,execution_source_replay_decodes=paired,
        verification_source_replay_decodes=paired,reconstructed_replays=2*paired,
        verification_reconstructed_replay_decodes=2*paired,training=False,gpu_training=False)


def budget_check(dm,contract,*,reserve=0):
    total=dm._owned_size(PRIVATE)+(dm._owned_size(STAGE) if STAGE.exists() else 0)
    require(dm._owned_size(PRIVATE)<=contract.LIMITS['private_result_bytes']
        and total+reserve<=contract.LIMITS['total_output_bytes']
        and shutil.disk_usage(GAME/'artifacts').free>=contract.LIMITS['disk_floor_bytes']+reserve)


def run(commit,*,role=None):
    infra,reg,contract,identity,bootstrap,base=authenticate(commit)
    raw,binding=binding_read(infra,reg);binding_sha=sha(raw)
    child=role is not None;require(not child or role in contract.ROLES)
    import psutil
    started=time.monotonic();limits=contract.LIMITS;latest={'available_bytes':None,'sample_monotonic':None}
    def available():
        n=base.physical_available();latest.update(available_bytes=n,sample_monotonic=time.monotonic());return n
    owner=(infra.decode(infra.read(PRIVATE/'owner.local.json')) if child else
        dict(schema='task-pair-fresh-multi-owner-v1',registration_commit=commit,binding_sha256=binding_sha,
            pid=os.getpid(),created_us=int(psutil.Process().create_time()*1000000),
            deadline=started+limits['total_wall_seconds']))
    process=psutil.Process();owned=False;accepted=[];cleanup=[]
    def check():
        if available()<limits['running_available_bytes'] or process.memory_info().rss>limits['max_rss_bytes']:
            raise MemoryError('Runtime resource floor')
        if time.monotonic()>=owner['deadline']:raise TimeoutError('Campaign deadline')
        require(infra.read(PRIVATE/'binding.local.json')==raw and not os.path.lexists(PRIVATE/'closed.local.json'))
        return True
    readiness=infra.snapshot('kaggriculture/learning/task_teacher_fresh_panel_v1/contract.py',
        reg['public_sources']['kaggriculture/learning/task_teacher_fresh_panel_v1/contract.py'])
    def ready():
        return readiness.await_memory(available,floor=limits['between_available_bytes'],
            timeout=min(limits['readiness_timeout_seconds'],owner['deadline']-time.monotonic()))
    try:
        if not child:
            require(all(not os.path.lexists(p) for p in (PRIVATE/'attempt.local.json',PRIVATE/'owner.local.json',
                PRIVATE/'closed.local.json',STAGE)))
            private_write(infra,'attempt.local.json',marker('attempt',commit,binding_sha));owned=True
            private_write(infra,'owner.local.json',owner);ready()
        else:check()
        modules=runtime(infra,bootstrap,reg);dm,source=modules['driver'],modules['source']
        if child:
            modules['fresh'].verify_live_owner(owner,commit=commit,binding_sha=binding_sha,current_pid=os.getpid(),
                process=psutil.Process,executable=sys.executable,samefile=os.path.samefile,now=time.monotonic())
            require(infra.decode(infra.read(PRIVATE/'attempt.local.json'))==marker('attempt',commit,binding_sha))
        archive=source.archive(infra,dm,check,full=not child)
        require(canonical(archive)==canonical(binding['archive'])
            and source.assets_verify(dm,archive,check)
            and shared().official_identity(infra,contract)==reg['installed_source_sha256'])
        budget_check(dm,contract)
        if child:
            case=next(s for s in archive['cases'] if s['role']==role);out=STAGE/role
            selected=selected_read(infra,binding,case)
            require(infra.read(STAGE/'binding.local.json')==raw
                and infra.decode(infra.read(out/'permit.local.json'))==marker('permit',commit,binding_sha,role))
            for name in ('started','consumed'):infra.write(out/(name+'.local.json'),marker(name,commit,binding_sha,role))
            modules['return_worker'].execute_case(modules,case,selected,out,check)
            require(check() and source.assets_verify(dm,archive,check))
            identity.verify_public(repo=REPO,game_root=GAME,registration_commit=commit)
            return None
        STAGE.mkdir(exist_ok=False);infra.write(STAGE/'binding.local.json',binding)
        require(infra.read(STAGE/'binding.local.json')==raw)
        for case in archive['cases']:
            current=case['role'];selected=selected_read(infra,binding,case)
            gate=ready();budget_check(dm,contract,reserve=limits['per_case_output_bytes'])
            require(len(accepted)<4 and owner['deadline']-time.monotonic()>limits['job_wall_seconds']+limits['cleanup_seconds'])
            out=STAGE/current;out.mkdir(exist_ok=False)
            infra.write(out/'readiness.local.json',gate)
            infra.write(out/'permit.local.json',marker('permit',commit,binding_sha,current))
            command=[sys.executable,'-I','-B',str(Path(__file__).resolve()),'run','--registration-commit',commit,'--role',current]
            supervision=infra.supervise_recorded(modules['campaign'],command,GAME,
                dict(max_rss_bytes=limits['max_rss_bytes'],max_commit_bytes=limits['max_commit_bytes'],
                    min_available_bytes=limits['running_available_bytes'],wall_seconds=limits['job_wall_seconds'],
                    cleanup_seconds=limits['cleanup_seconds'],poll_seconds=limits['poll_seconds']))
            sh=infra.write(out/'supervision.local.json',dict(supervision=supervision,retry_allowed=False))
            cleanup.append(dict(role=current,sha256=sh,supervision=supervision))
            require(modules['acceptance'].supervision_ok(supervision) and check())
            for kind in ('permit','started','consumed'):
                require(infra.decode(infra.read(out/(kind+'.local.json')))==marker(kind,commit,binding_sha,current))
            record=modules['return_worker'].verify_case(modules,case,selected,out,check)
            infra.write(out/'accepted.local.json',dict(passed=True,role=current,registration_commit=commit,
                case_sha256=record['document_sha256'],supervision_sha256=sh,retry_allowed=False))
            accepted.append(record);private_write(infra,current+'.result.local.json',record)
        require(len(accepted)==4 and check() and source.assets_verify(dm,archive,check))
        identity.verify_public(repo=REPO,game_root=GAME,registration_commit=commit)
        projection=contract.projection([r['row'] for r in accepted]);costs=aggregate_costs(accepted,contract)
        private_write(infra,'projected-summary.local.json',projection)
        private_write(infra,'result.local.json',dict(schema=SCHEMA+'-result',registration_commit=commit,
            binding_sha256=binding_sha,accepted=accepted,execution=costs,
            projection_sha256=sha(infra.read(PRIVATE/'projected-summary.local.json'))))
        budget_check(dm,contract)
        return dict(status='completed',paired=costs['paired'],terminal_suffixes=costs['terminal_suffixes'],
            complete_from_zero_games=0,training=False,gpu=False)
    finally:
        if owned:
            error=sys.exc_info()[1];memory=infra.closing_memory(error,latest,available)
            close(infra,'closed.local.json',dict(schema=SCHEMA+'-closed',registration_commit=commit,
                binding_sha256=binding_sha,outcome='completed' if error is None else 'failed',
                error_kind=None if error is None else type(error).__name__,accepted_cases=len(accepted),
                child_cleanup_receipts=cleanup,**memory,elapsed_seconds=time.monotonic()-started,retry_allowed=False),error)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);commands=parser.add_subparsers(dest='command',required=True)
    commands.add_parser('prepare');run_parser=commands.add_parser('run')
    run_parser.add_argument('--registration-commit',required=True);run_parser.add_argument('--role')
    args=parser.parse_args(argv)
    try:
        result=prepare() if args.command=='prepare' else run(args.registration_commit,role=args.role)
        if result is not None:print(json.dumps(result,separators=(',',':')))
        return 0
    except BaseException as error:
        print(json.dumps(dict(status='failed',error_kind=type(error).__name__,retry_allowed=False)))
        return 1


if __name__=='__main__':raise SystemExit(main())
