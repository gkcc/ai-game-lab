"""One separately registered seven-phase campaign for all eight F hits.

Only the parameterized phase/output checks and NUL-compatible write adapter
are reused from the historical pair coordinator. None of its experiment
entrypoints or private-source loaders is called. Seed-bearing data stay local.
"""
import argparse
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import re
import secrets
import shutil
import subprocess
import sys
import time
from types import ModuleType

REPO = Path(__file__).resolve().parents[3]
GAME = REPO.parent.parent/'kaggriculture'
BASE = '5a0dd353034889940d4215def53b5ddffd3f32c9'
SCREEN_COMMIT = '7c8841019ebb5265ee23aa5e48b7598aaaeb6dc6'
PREFIX = 'kaggriculture/learning/task_teacher_f_pair_v1/'
F_PREFIX = 'kaggriculture/learning/task_teacher_f_counterfactual_v1/'
SCREEN_EVIDENCE = 'review/evidence/task-teacher-f-counterfactual-screen-v1-20261106'
SCREEN_RESULTS = 'review/evidence/task-teacher-f-counterfactual-screen-v1-result-20261106'
EVIDENCE = 'review/evidence/task-teacher-f-pair-v1-20261106'
REGISTRATION = EVIDENCE+'/registration.json'
PRIVATE = REPO.parent/'private/task-teacher-f-pair-v1-20261106'
STAGE = GAME/'artifacts/task-teacher-f-pair-v1-20261106'
INFRA = 'kaggriculture/learning/task_teacher_fresh_panel_v1/runner.py'
MAX_RECEIPT = 32*1024
PARENT_RESERVE = 16*1024**2
CHILD_RESERVE = 56*3*MAX_RECEIPT
MAX_BINDING = 512*1024


def require(value):
    if not value: raise ValueError('F pair registration or phase chain differs')


def sha(raw): return hashlib.sha256(raw).hexdigest()


def finite(value): return type(value) in (int,float) and math.isfinite(value)


def encoded(value): return (json.dumps(value,ensure_ascii=True,separators=(',',':'),allow_nan=False)+'\n').encode()


def git(*args):
    result=subprocess.run(['git','-C',str(REPO),*args],stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,stderr=subprocess.DEVNULL,timeout=45,
        creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    require(result.returncode==0 and len(result.stdout)<=16*1024**2)
    return result.stdout


def infrastructure():
    raw=(REPO/INFRA).read_bytes();require(raw==git('show',BASE+':'+INFRA))
    module=ModuleType('_f_pair_infrastructure');module.__file__=str(REPO/INFRA)
    exec(compile(raw,module.__file__,'exec',dont_inherit=True),module.__dict__)
    return module


def required_sources(infra):
    old=infra.decode(git('show',SCREEN_COMMIT+':'+SCREEN_EVIDENCE+'/registration.json'))
    paths=set(old['public_sources'])
    paths.update((SCREEN_EVIDENCE+'/registration.json',SCREEN_RESULTS+'/certificate.json',
        SCREEN_RESULTS+'/summary.json',EVIDENCE+'/REGISTRATION.zh-CN.md'))
    paths.update(p.relative_to(REPO).as_posix() for p in (REPO/PREFIX).glob('*.py'))
    return paths


def specification(contract):
    return dict(sources=list(contract.SOURCES),cases=8,phases=list(contract.PHASES),phase_count=56,
        complete_games=16,official_main_replays=16,pair_official_transitions=23008,
        screen_official_transitions=16,combined_official_transitions=23024,
        case_selection='all_screen_hits_fixed_source_order',scope='consumed_development',
        income='terminal_cash_not_transaction_reconciliation',weighting=['case_equal','seed_equal'])


def validate_registration(reg,contract):
    require(type(reg) is dict and reg.get('schema')=='teacher-f-pair-prereg-v1'
        and reg.get('base_commit')==BASE and reg.get('screen_registration_commit')==SCREEN_COMMIT
        and reg.get('engine_commit')==contract.ENGINE_COMMIT and reg.get('engine_sha256')==contract.ENGINE_SHA
        and contract.same(reg.get('limits'),contract.LIMITS)
        and contract.same(reg.get('phase_write_bytes'),contract.PHASE_BYTES)
        and contract.same(reg.get('pairing'),specification(contract))
        and reg.get('evaluation')==dict(training=False,gpu=False,new_seeds=False,
            independent_evaluation=False,statistical_significance=False,promotion=False,retries=False)
        and reg.get('metadata_reserve_bytes')==PARENT_RESERVE
        and reg.get('child_metadata_max_bytes')==3*MAX_RECEIPT
        and type(reg.get('binding_commitment_sha256')) is str
        and re.fullmatch(r'[0-9a-f]{64}',reg['binding_commitment_sha256']) is not None)


def authenticate(commit):
    infra=infrastructure()
    # The old pure F contract remains authoritative for limits and projection;
    # the new registration validator is explicit, not the screen-only schema.
    infra.REGISTRATION,infra.PREFIX,infra.BASE=REGISTRATION,F_PREFIX,BASE
    infra._required_sources=lambda _old:required_sources(infra)
    infra.validate_registration=validate_registration
    reg,contract,identity,bootstrap,base=infra.authenticate(commit)
    require(not os.path.lexists(GAME/'learning/task_teacher_f_pair_v1')
        and not os.path.lexists(GAME/'learning/task_teacher_f_counterfactual_v1'))
    sources=infra.snapshot(PREFIX+'sources.py',reg['public_sources'][PREFIX+'sources.py'],commit)
    proof=sources.public(infra,git,contract)
    return infra,reg,contract,identity,bootstrap,base,sources,proof


def runtime(infra,bootstrap,reg):
    modules=infra._runtime(bootstrap,reg)
    names={'f_sources':'task_teacher_f_counterfactual_v1.sources',
        'f_worker':'task_teacher_f_counterfactual_v1.worker',
        'pair':'task_teacher_counterfactual_pair_v1.experiment'}
    for key,name in names.items():
        module=importlib.import_module('learning.'+name)
        relative='kaggriculture/learning/'+name.replace('.','/')+'.py'
        require(Path(module.__file__).resolve(strict=True)==(REPO/relative).resolve(strict=True))
        modules[key]=module
    return modules


def units(contract):
    return [dict(case_index=i,phase=phase) for i in range(8) for phase in contract.PHASES]


def binding_valid(binding,contract):
    require(type(binding) is dict and set(binding)=={'schema','base_commit','salt','screen','execution_order'}
        and binding['schema']=='teacher-f-pair-binding-v1' and binding['base_commit']==BASE
        and type(binding['salt']) is str and re.fullmatch(r'[0-9a-f]{64}',binding['salt']) is not None
        and binding['execution_order']==units(contract) and type(binding['screen']) is dict)
    cases=binding['screen']['cases']
    require(type(cases) is list and len(cases)==8)
    for index,case in enumerate(cases):
        require(type(case) is dict and contract.source_index(dict(role=case.get('role'),
            opponent=case.get('opponent'),side=case.get('side'),policy='baseline'))==index)


class Metadata:
    """Charge each attempted exclusive JSON write; never refund an error.

    Child metadata reserves attempt, result and possible failure per phase.
    A killed child has unknown attempted bytes, bounded by that allowance, not
    an invented zero. Case data uses the original cumulative owned_writes.
    """
    def __init__(self,infra,limit,initial=0): self.infra,self.limit,self.used=infra,limit,initial
    def write(self,path,value,maximum=MAX_RECEIPT):
        size=len(encoded(value));require(size<=maximum and self.used+size<=self.limit)
        self.used+=size
        return self.infra.write(path,value,maximum)


def prepare():
    require(os.name=='nt' and sys.flags.isolated and sys.flags.dont_write_bytecode and __debug__)
    infra=infrastructure()
    require(git('rev-parse','HEAD').decode().strip()==BASE and not os.path.lexists(PRIVATE)
        and not os.path.lexists(STAGE) and not os.path.lexists(REPO/REGISTRATION))
    identities={p:sha(infra.read(REPO/p)) for p in sorted(required_sources(infra))}
    contract=infra.snapshot(F_PREFIX+'contract.py',identities[F_PREFIX+'contract.py'])
    sources=infra.snapshot(PREFIX+'sources.py',identities[PREFIX+'sources.py'])
    proof=sources.public(infra,git,contract)
    base=infra.snapshot(infra.BASE_ENTRY,identities[infra.BASE_ENTRY])
    ready_path='kaggriculture/learning/task_teacher_fresh_panel_v1/contract.py'
    ready=infra.snapshot(ready_path,identities[ready_path])
    ready.await_memory(base.physical_available)
    deadline=time.monotonic()+900
    def guard():
        if base.physical_available()<contract.LIMITS['running_available_bytes']: raise MemoryError('Preparation memory floor')
        if time.monotonic()>=deadline: raise TimeoutError('Preparation deadline')
        return True
    identity=infra.snapshot(infra.IDENTITY,identities[infra.IDENTITY])
    infra.preparation_runtime(identity,BASE)
    reg=dict(schema='teacher-f-pair-prereg-v1',base_commit=BASE,screen_registration_commit=SCREEN_COMMIT,
        engine_commit=contract.ENGINE_COMMIT,engine_sha256=contract.ENGINE_SHA,public_sources=identities,
        versions=infra._versions(),limits=contract.LIMITS,phase_write_bytes=contract.PHASE_BYTES,
        pairing=specification(contract),metadata_reserve_bytes=PARENT_RESERVE,
        child_metadata_max_bytes=3*MAX_RECEIPT,evaluation=dict(training=False,gpu=False,new_seeds=False,
            independent_evaluation=False,statistical_significance=False,promotion=False,retries=False))
    bootstrap=infra.snapshot(infra.BOOTSTRAP,identities[infra.BOOTSTRAP]);modules=runtime(infra,bootstrap,reg)
    evidence=sources.load(infra,modules,contract,proof,guard)
    try:
        binding=dict(schema='teacher-f-pair-binding-v1',base_commit=BASE,salt=secrets.token_hex(32),
            screen=evidence.identity,execution_order=units(contract))
        binding_valid(binding,contract);evidence.verify_files(infra,guard)
    finally:evidence.close()
    guard();require(all(sha(infra.read(REPO/p))==h for p,h in identities.items()))
    PRIVATE.mkdir(exist_ok=False)
    reg['binding_commitment_sha256']=infra.write(PRIVATE/'binding.local.json',binding,MAX_BINDING)
    validate_registration(reg,contract)
    (REPO/EVIDENCE).mkdir(parents=True,exist_ok=True)
    infra.write(REPO/REGISTRATION,reg)
    return dict(status='prepared_not_executed',frozen_cases=8,phases=56,new_complete_games=0)


def numbered(kind,sequence):
    require(type(sequence) is int and 0<=sequence<56)
    require(kind in ('readiness','claim','child-attempt','child-result','child-failed','job','accepted'))
    return f'{kind}-{sequence:03d}.local.json'


def attempt(commit,binding_sha):
    return dict(schema='teacher-f-pair-attempt-v1',registration_commit=commit,binding_sha256=binding_sha,retry_allowed=False)


def state(): return dict(previous_sha256=None,charged_bytes=0,accounts=[],outputs={})


def readiness_valid(value,contract):
    require(type(value) is dict and value.get('consecutive_samples')==3
        and type(value.get('last_available_bytes')) is int
        and value['last_available_bytes']>=contract.LIMITS['between_available_bytes']
        and finite(value.get('elapsed_seconds')) and 20<=value['elapsed_seconds']<=300)


def validate_claim(claim,sequence,previous,contract,commit,binding_sha,owner,ready_raw):
    unit=units(contract)[sequence]
    limit=min(contract.PHASE_BYTES[unit['phase']],contract.LIMITS['pair_output_bytes']-PARENT_RESERVE-previous['charged_bytes'])
    require(type(claim) is dict and set(claim)=={'schema','registration_commit','binding_sha256','sequence',
        'unit','attempt','retry_allowed','previous_sha256','limit_bytes','deadline','readiness_sha256'}
        and claim['schema']=='teacher-f-pair-claim-v1' and claim['registration_commit']==commit
        and claim['binding_sha256']==binding_sha and type(claim['sequence']) is int and claim['sequence']==sequence
        and claim['unit']==unit and type(claim['attempt']) is int and claim['attempt']==1
        and claim['retry_allowed'] is False and claim['previous_sha256']==previous['previous_sha256']
        and type(claim['limit_bytes']) is int and claim['limit_bytes']==limit and limit>0
        and finite(claim['deadline']) and claim['deadline']<=owner['deadline']
        and claim['readiness_sha256']==sha(ready_raw))
    return unit,limit


def storage_valid(storage,limit):
    require(type(storage) is dict and set(storage)=={'limit_bytes','used_bytes','remaining_bytes','refused','created_count'}
        and all(type(storage[k]) is int for k in storage) and storage['limit_bytes']==limit
        and 0<=storage['used_bytes']<=limit and storage['remaining_bytes']==limit-storage['used_bytes']
        and storage['refused']==0 and storage['created_count']>=0)


def validate_phase(raws,sequence,previous,*,infra,contract,pair,job_ok,commit,binding_sha,owner):
    """Pure receipt validation shared by acceptance, child and completed reader."""
    require(set(raws)=={'readiness','claim','child-attempt','child-result','job'})
    values={key:infra.decode(raw) for key,raw in raws.items()}
    readiness_valid(values['readiness'],contract)
    unit,limit=validate_claim(values['claim'],sequence,previous,contract,commit,binding_sha,owner,raws['readiness'])
    marker=dict(schema='teacher-f-pair-child-attempt-v1',registration_commit=commit,binding_sha256=binding_sha,
        sequence=sequence,unit=unit,claim_sha256=sha(raws['claim']),retry_allowed=False)
    require(values['child-attempt']==marker and job_ok(values['job']))
    receipt=values['child-result']
    require(set(receipt)=={'schema','registration_commit','binding_sha256','sequence','unit','claim_sha256',
        'result','storage','retry_allowed'} and receipt['schema']=='teacher-f-pair-child-result-v1'
        and receipt['registration_commit']==commit and receipt['binding_sha256']==binding_sha
        and type(receipt['sequence']) is int and receipt['sequence']==sequence and receipt['unit']==unit
        and receipt['claim_sha256']==sha(raws['claim']) and receipt['retry_allowed'] is False)
    result=pair._phase_result(receipt['result'],unit['phase']);storage_valid(receipt['storage'],limit)
    updated=dict(previous_sha256=previous['previous_sha256'],charged_bytes=previous['charged_bytes']+receipt['storage']['used_bytes'],
        accounts=list(previous['accounts']),outputs={i:dict(v) for i,v in previous['outputs'].items()})
    index=unit['case_index']
    if unit['phase']=='prepare_case':require(index not in updated['outputs']);updated['outputs'][index]={}
    require(index in updated['outputs'])
    for path,digest in result['outputs'].items():
        require(path not in updated['outputs'][index] or updated['outputs'][index][path]==digest)
        updated['outputs'][index][path]=digest
    if unit['phase']=='pair_verify':updated['accounts'].append(result['details']['account'])
    accepted=dict(schema='teacher-f-pair-accepted-v1',passed=True,registration_commit=commit,binding_sha256=binding_sha,
        sequence=sequence,unit=unit,previous_sha256=previous['previous_sha256'],
        claim_sha256=sha(raws['claim']),child_attempt_sha256=sha(raws['child-attempt']),
        child_result_sha256=sha(raws['child-result']),job_sha256=sha(raws['job']),
        charged_bytes=updated['charged_bytes'],retry_allowed=False)
    return updated,accepted


def read_chain(count,read,*,infra,contract,pair,job_ok,commit,binding_sha,owner):
    require(type(count) is int and 0<=count<=56)
    current=state()
    for sequence in range(count):
        raws={key:read(numbered(key,sequence)) for key in ('readiness','claim','child-attempt','child-result','job')}
        current,expected=validate_phase(raws,sequence,current,infra=infra,contract=contract,pair=pair,
            job_ok=job_ok,commit=commit,binding_sha=binding_sha,owner=owner)
        raw=read(numbered('accepted',sequence));require(infra.decode(raw)==expected)
        current['previous_sha256']=sha(raw)
    return current


def verify_outputs(pair,current,guard):
    for index,outputs in current['outputs'].items():
        pair._verify_outputs(STAGE/f'case-{index:02d}',outputs,guard)


def disk_boundary(modules,limits,guard):
    """Detect native/unmetered files at every phase boundary, not just closure."""
    guard()
    private=modules['driver']._owned_size(PRIVATE)
    stage=modules['driver']._owned_size(STAGE)
    require(private<=limits['private_result_bytes'] and private+stage<=limits['pair_output_bytes'])
    return private+stage


def owner_valid(owner,commit,binding_sha):
    require(type(owner) is dict and set(owner)=={'schema','registration_commit','binding_sha256','pid','created_us','deadline'}
        and owner['schema']=='task-pair-fresh-multi-owner-v1' and owner['registration_commit']==commit
        and owner['binding_sha256']==binding_sha and type(owner['pid']) is int and owner['pid']>0
        and type(owner['created_us']) is int and owner['created_us']>0 and finite(owner['deadline']))


def run(commit,*,preflight=False,child_sequence=None,results=False):
    infra,reg,contract,identity,bootstrap,base,sources,proof=authenticate(commit)
    raw=infra.read(PRIVATE/'binding.local.json',MAX_BINDING);binding=infra.decode(raw);binding_sha=sha(raw)
    require(binding_sha==reg['binding_commitment_sha256']);binding_valid(binding,contract)
    child=child_sequence is not None
    require(sum((preflight,child,results))<=1)
    if child:numbered('claim',child_sequence)
    limits=contract.LIMITS;started=time.monotonic();latest=dict(available_bytes=None,sample_monotonic=None)
    def available():
        value=base.physical_available();latest.update(available_bytes=value,sample_monotonic=time.monotonic());return value
    import psutil
    owner=(infra.decode(infra.read(PRIVATE/'owner.local.json')) if child or results else
        dict(schema='task-pair-fresh-multi-owner-v1',registration_commit=commit,binding_sha256=binding_sha,
            pid=os.getpid(),created_us=int(psutil.Process().create_time()*1_000_000),deadline=started+limits['pair_wall_seconds']))
    owner_valid(owner,commit,binding_sha)
    deadline=(started+limits['job_wall_seconds'] if results else owner['deadline'])
    def guard():
        if available()<limits['running_available_bytes']: raise MemoryError('Pair memory floor')
        if time.monotonic()>=deadline: raise TimeoutError('Pair deadline')
        if shutil.disk_usage(GAME/'artifacts').free<limits['disk_floor_bytes']: raise OSError('Disk floor')
        return True
    ready_path='kaggriculture/learning/task_teacher_fresh_panel_v1/contract.py'
    readiness=infra.snapshot(ready_path,reg['public_sources'][ready_path],commit)
    def ready():
        return readiness.await_memory(available,timeout=min(limits['readiness_timeout_seconds'],deadline-time.monotonic()))
    parent_meta=Metadata(infra,PARENT_RESERVE-CHILD_RESERVE-MAX_RECEIPT,len(raw))
    owned=False;evidence=None;current=state();jobs=[];active=None;result_sha=None;seal_sha=None
    budget=None;child_meta=Metadata(infra,3*MAX_RECEIPT);child_attempted=False
    def read(name):return infra.read(PRIVATE/name,MAX_RECEIPT)
    try:
        if not child and not results:
            require(not os.path.lexists(PRIVATE/'attempt.local.json') and not os.path.lexists(PRIVATE/'owner.local.json')
                and not os.path.lexists(PRIVATE/'closed.local.json') and not os.path.lexists(STAGE))
            if not preflight:
                parent_meta.write(PRIVATE/'attempt.local.json',attempt(commit,binding_sha));owned=True
                parent_meta.write(PRIVATE/'owner.local.json',owner)
            ready()
        else:guard()
        modules=runtime(infra,bootstrap,reg);pair=modules['pair'];job_ok=modules['acceptance'].supervision_ok
        common=dict(infra=infra,contract=contract,pair=pair,job_ok=job_ok,commit=commit,binding_sha=binding_sha,owner=owner)
        if child:
            require(not os.path.lexists(PRIVATE/'closed.local.json'))
            modules['fresh'].verify_live_owner(owner,commit=commit,binding_sha=binding_sha,current_pid=os.getpid(),
                process=psutil.Process,executable=sys.executable,samefile=os.path.samefile,now=time.monotonic())
        if child or results:require(infra.decode(read('attempt.local.json'))==attempt(commit,binding_sha))
        evidence=sources.load(infra,modules,contract,proof,guard,verify_game_files=not child)
        require(contract.same(evidence.identity,binding['screen']))
        if preflight:
            require(shutil.disk_usage(GAME/'artifacts').free>=limits['disk_floor_bytes']+limits['pair_output_bytes'])
            return dict(status='ready',frozen_cases=8,phase_count=56,new_complete_games=0)
        if results:
            current=read_chain(56,read,**common);verify_outputs(pair,current,guard)
            closed=infra.decode(read('closed.local.json'));result_raw=infra.read(PRIVATE/'pair-result.local.json',MAX_BINDING)
            seal=infra.decode(read('pair-seal.local.json'))
            projection=contract.pair_projection(current['accounts'],list(contract.SOURCES))
            expected=pair_result(commit,binding_sha,current,projection)
            require(infra.decode(result_raw)==expected and closed.get('outcome')=='completed'
                and closed.get('result_sha256')==sha(result_raw) and closed.get('accepted_phases')==56
                and closed.get('seal_sha256')==sha(read('pair-seal.local.json'))
                and seal==pair_seal(commit,binding_sha,current,sha(result_raw)))
            return dict(status='completed',cases=8,complete_games=16,official_main_replays=16,summary=projection)
        if child:
            current=read_chain(child_sequence,read,**common);verify_outputs(pair,current,guard)
            disk_boundary(modules,limits,guard)
            claim_raw=read(numbered('claim',child_sequence));claim=infra.decode(claim_raw)
            ready_raw=read(numbered('readiness',child_sequence));readiness_valid(infra.decode(ready_raw),contract)
            unit,limit=validate_claim(claim,child_sequence,current,contract,commit,binding_sha,owner,ready_raw)
            deadline=min(deadline,claim['deadline']);guard()
            child_meta.write(PRIVATE/numbered('child-attempt',child_sequence),dict(schema='teacher-f-pair-child-attempt-v1',
                registration_commit=commit,binding_sha256=binding_sha,sequence=child_sequence,unit=unit,
                claim_sha256=sha(claim_raw),retry_allowed=False));child_attempted=True
            directory=STAGE/f"case-{unit['case_index']:02d}"
            with pair._owned_writes_with_subprocess_null(directory,limit) as budget:
                result=modules['f_worker'].run_phase(context=evidence.context,stage=directory,phase=unit['phase'],
                    case=evidence.cases[unit['case_index']],guard=guard)
            storage=budget.snapshot();storage_valid(storage,limit);pair._phase_result(result,unit['phase'])
            disk_boundary(modules,limits,guard)
            verify_outputs(pair,current,guard);pair._verify_outputs(directory,result['outputs'],guard)
            evidence.verify_files(infra,guard);identity.verify_public(repo=REPO,game_root=GAME,registration_commit=commit)
            require(infra.read(PRIVATE/'binding.local.json',MAX_BINDING)==raw);guard()
            child_meta.write(PRIVATE/numbered('child-result',child_sequence),dict(schema='teacher-f-pair-child-result-v1',
                registration_commit=commit,binding_sha256=binding_sha,sequence=child_sequence,unit=unit,
                claim_sha256=sha(claim_raw),result=result,storage=storage,retry_allowed=False))
            return None
        STAGE.mkdir(exist_ok=False)
        for sequence,unit in enumerate(binding['execution_order']):
            gate=ready();guard()
            disk_boundary(modules,limits,guard)
            current=read_chain(sequence,read,**common);verify_outputs(pair,current,guard)
            evidence.verify_files(infra,guard)
            require(infra.read(PRIVATE/'binding.local.json',MAX_BINDING)==raw
                and deadline-time.monotonic()>limits['job_wall_seconds']+limits['cleanup_seconds'])
            remaining=limits['pair_output_bytes']-current['charged_bytes']
            require(shutil.disk_usage(STAGE).free>=limits['disk_floor_bytes']+remaining)
            if unit['phase']=='prepare_case':(STAGE/f"case-{unit['case_index']:02d}").mkdir(exist_ok=False)
            ready_sha=parent_meta.write(PRIVATE/numbered('readiness',sequence),gate)
            limit=min(contract.PHASE_BYTES[unit['phase']],remaining-PARENT_RESERVE);require(limit>0)
            claim=dict(schema='teacher-f-pair-claim-v1',registration_commit=commit,binding_sha256=binding_sha,
                sequence=sequence,unit=unit,attempt=1,retry_allowed=False,previous_sha256=current['previous_sha256'],
                limit_bytes=limit,deadline=time.monotonic()+limits['job_wall_seconds'],readiness_sha256=ready_sha)
            parent_meta.write(PRIVATE/numbered('claim',sequence),claim);active=sequence
            command=[sys.executable,'-I','-B',str(Path(__file__).resolve()),'--registration-commit',commit,'--child',str(sequence)]
            job=infra.supervise_recorded(modules['campaign'],command,GAME,dict(max_rss_bytes=limits['max_rss_bytes'],
                max_commit_bytes=limits['max_commit_bytes'],min_available_bytes=limits['running_available_bytes'],
                wall_seconds=limits['job_wall_seconds'],cleanup_seconds=limits['cleanup_seconds'],poll_seconds=limits['poll_seconds']))
            job_sha=parent_meta.write(PRIVATE/numbered('job',sequence),job)
            jobs.append(dict(sequence=sequence,sha256=job_sha,launcher_reaped=job.get('launcher_reaped'),
                job_members_zero=job.get('job_members_zero'),cleanup_error_kind=job.get('cleanup_error_kind')))
            require(job_ok(job));disk_boundary(modules,limits,guard);verify_outputs(pair,current,guard)
            raws={key:read(numbered(key,sequence)) for key in ('readiness','claim','child-attempt','child-result','job')}
            updated,accepted=validate_phase(raws,sequence,current,**common)
            verify_outputs(pair,updated,guard);evidence.verify_files(infra,guard)
            current=updated
            current['previous_sha256']=parent_meta.write(PRIVATE/numbered('accepted',sequence),accepted)
            active=None
        current=read_chain(56,read,**common);verify_outputs(pair,current,guard)
        require(len(current['accounts'])==8 and len(jobs)==56)
        projection=contract.pair_projection(current['accounts'],list(contract.SOURCES))
        evidence.verify_files(infra,guard);identity.verify_public(repo=REPO,game_root=GAME,registration_commit=commit)
        require(infra.read(PRIVATE/'binding.local.json',MAX_BINDING)==raw
            and modules['driver']._owned_size(STAGE)+modules['driver']._owned_size(PRIVATE)<=limits['pair_output_bytes'])
        result_sha=parent_meta.write(PRIVATE/'pair-result.local.json',pair_result(commit,binding_sha,current,projection),MAX_BINDING)
        seal_sha=parent_meta.write(PRIVATE/'pair-seal.local.json',pair_seal(commit,binding_sha,current,result_sha))
        return dict(status='completed',cases=8,complete_games=16,official_main_replays=16,training=False,promotion=False)
    finally:
        error=sys.exc_info()[1];close_error=None
        if evidence is not None:
            try:evidence.close()
            except BaseException as exc:close_error=exc
        effective=error if error is not None else close_error
        if child and child_attempted and effective is not None:
            try:
                child_meta.write(PRIVATE/numbered('child-failed',child_sequence),dict(schema='teacher-f-pair-child-failed-v1',
                    registration_commit=commit,binding_sha256=binding_sha,sequence=child_sequence,
                    error_kind=type(effective).__name__,context_close_error_kind=None if close_error is None else type(close_error).__name__,
                    storage=None if budget is None else budget.snapshot(),retry_allowed=False,
                    **infra.closing_memory(effective,latest,available)))
            except BaseException:
                # No substitute success or retry: the parent's rejected Job
                # records missing failure metadata as unknown progress.
                pass
        if owned:
            memory=infra.closing_memory(effective,latest,available)
            closed=dict(schema='teacher-f-pair-closed-v1',registration_commit=commit,binding_sha256=binding_sha,
                outcome='completed' if effective is None else 'failed',error_kind=None if effective is None else type(effective).__name__,
                context_close_error_kind=None if close_error is None else type(close_error).__name__,
                accepted_phases=(56 if effective is None else (active if active is not None else len(jobs))),
                active_sequence=active,active_phase_work='unknown' if active is not None else 'none',
                active_job_cleanup=('unconfirmed' if active is not None and (not jobs or jobs[-1]['sequence']!=active)
                    else 'see_child_cleanup_receipts' if active is not None else 'all_recorded'),
                confirmed_case_write_bytes=current['charged_bytes'],unconfirmed_case_write_bytes='unknown' if active is not None else None,
                parent_metadata_charged_bytes=parent_meta.used,child_metadata_reserved_bytes=CHILD_RESERVE,
                result_sha256=result_sha,seal_sha256=seal_sha,elapsed_seconds=time.monotonic()-started,
                child_cleanup_receipts=jobs,retry_allowed=False,**memory)
            # The final receipt has its own reserve even if an earlier metadata
            # write was refused; context-close failures never erase run failure.
            try:infra.write(PRIVATE/'closed.local.json',closed,MAX_RECEIPT)
            except BaseException:
                # Missing closure is never success. If work already failed,
                # preserve that original exception instead of replacing it
                # with a secondary receipt-write failure.
                if effective is None:raise
        if error is None and close_error is not None:raise close_error


def pair_result(commit,binding_sha,current,projection):
    return dict(schema='teacher-f-pair-result-v1',registration_commit=commit,binding_sha256=binding_sha,
        accounts=current['accounts'],projection=projection,technical_pass=True,terminal_cash_only=True,
        complete_cases=8,complete_games=16,official_main_replays=16,pair_official_transitions=23008,
        combined_official_transitions=23024,training=False,gpu=False,new_seeds=False,
        independent_evaluation=False,statistical_significance=False,promotion=False)


def pair_seal(commit,binding_sha,current,result_sha):
    return dict(schema='teacher-f-pair-seal-v1',registration_commit=commit,binding_sha256=binding_sha,
        result_sha256=result_sha,final_accepted_sha256=current['previous_sha256'],accepted_phases=56,
        charged_bytes=current['charged_bytes'],retry_allowed=False)


class Parser(argparse.ArgumentParser):
    def error(self,_message):raise ValueError('Invalid private campaign command')


def main():
    try:
        parser=Parser(description=__doc__)
        parser.add_argument('--prepare',action='store_true');parser.add_argument('--preflight',action='store_true')
        parser.add_argument('--registration-commit');parser.add_argument('--child',type=int)
        parser.add_argument('--results',action='store_true')
        try:args=parser.parse_args()
        except SystemExit as exc:
            if exc.code==0:return 0  # argparse help only, before any execution.
            raise
        require(not(args.prepare and (args.registration_commit or args.preflight or args.child is not None or args.results)))
        result=prepare() if args.prepare else run(args.registration_commit,preflight=args.preflight,
            child_sequence=args.child,results=args.results)
        if result is not None:print(json.dumps(result,allow_nan=False))
        return 0
    except BaseException as exc:
        category='memory_gate' if isinstance(exc,MemoryError) else 'time_gate' if isinstance(exc,TimeoutError) else 'invalid_or_runtime'
        print(json.dumps(dict(status='failed',category=category,retry_allowed=False)),file=sys.stderr);return 1


if __name__=='__main__':raise SystemExit(main())
