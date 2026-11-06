"""Twelve new independent point pairs; reuse four sealed first points."""
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
BASE='8b7949b4ca29e3c2492cc01216d8391ecc7a2a1d'
PRIOR_REG_COMMIT='f3a52cd96f23b514625767dfbfd9d27175f870ae'
PREFIX='kaggriculture/learning/task_harvest_quantile_pairs_v1/'
EVIDENCE='review/evidence/task-harvest-quantile-pairs-v1-20261106'
REGISTRATION,NOTE=EVIDENCE+'/registration.json',EVIDENCE+'/REGISTRATION.zh-CN.md'
PRIVATE=REPO.parent/'private/task-harvest-quantile-pairs-v1-20261106'
STAGE=GAME/'artifacts/task-harvest-quantile-pairs-v1-20261106'
SCHEMA='task-harvest-quantile-pairs-v1'
PRIOR_REG='review/evidence/task-harvest-return-pairs-v1-20261106/registration.json'
PRIOR_RESULTS='review/evidence/task-harvest-return-pairs-v1-result-20261106'
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
    require(len(old['public_sources'])==574)
    extras={PRIOR_REG,NOTE,PREFIX+'README.zh-CN.md'}
    extras.update(PRIOR_RESULTS+'/'+name for name in ('summary.json','certificate.json',
        'RESULTS.zh-CN.md','REVIEW.zh-CN.md','NEXT.zh-CN.md'))
    paths=set(old['public_sources'])|extras
    paths.update(p.relative_to(REPO).as_posix() for p in (REPO/PREFIX).glob('*.py'))
    required=('__init__.py','contract.py','selection.py','sources.py','analysis.py','runner.py',
        'test_selection.py','test_analysis.py','test_contract.py','test_sources.py','test_runner.py')
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
    require(not os.path.lexists(GAME/'learning/task_harvest_quantile_pairs_v1'))
    return infra,reg,contract,identity,bootstrap,base


def metadata_modules():
    require('numpy' not in sys.modules and 'kaggle_environments' not in sys.modules)
    os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
    sys.path.insert(0,str(REPO/'kaggriculture'))
    result={key:importlib.import_module('learning.'+name) for key,name in (
        ('source','task_harvest_quantile_pairs_v1.sources'),('driver','task_pair_dev_campaign_v1.driver'))}
    require('kaggle_environments' not in sys.modules and 'torch' not in sys.modules)
    return result


def runtime(infra,bootstrap,reg):
    result=infra._runtime(bootstrap,reg)
    for key,name in (('source','sources'),('analysis','analysis')):
        module=importlib.import_module('learning.task_harvest_quantile_pairs_v1.'+name)
        require(Path(module.__file__).resolve(strict=True)==(REPO/PREFIX/(name+'.py')).resolve(strict=True))
        result[key]=module
    worker=importlib.import_module('learning.task_harvest_return_pairs_v1.worker')
    require(Path(worker.__file__).resolve(strict=True)==
        (REPO/'kaggriculture/learning/task_harvest_return_pairs_v1/worker.py').resolve(strict=True))
    result['return_worker']=worker
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
        selections=source.select_sources(dm,archive,check,lambda n,v:private_write(infra,n,v))
        require(len(selections)==4)
        selection_hashes={s['origin']:sha(infra.read(PRIVATE/(s['origin']+'.selection.local.json'))) for s in selections}
        binding=dict(schema=SCHEMA+'-binding',base_commit=BASE,salt=secrets.token_hex(32),
            archive=archive,selections=selection_hashes,plans=source.plans(selections),
            default_sha256=archive['default_sha256'])
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
    return dict(status='prepared',sealed_old_files=145+contract.PAIR_FILES,source_trace_scans=4,
        source_replay_decodes=0,model_forwards=0,official_actions=0,normalization_loads=0,
        frozen_sources=len(sources),new_slots_registered=12,old_pairs_reused=4)


def marker(kind,commit,binding_sha,role=None):
    require(kind in ('attempt','permit','started','consumed'))
    value=dict(schema=SCHEMA+'-marker',kind=kind,registration_commit=commit,
        binding_sha256=binding_sha,retry_allowed=False)
    if kind!='attempt':
        require(type(role) is str and re.fullmatch('F0[1-4]-Q[1-3]',role));value['role']=role
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
        and value['origin']==source['role'] and canonical(value['source'])==canonical(source))
    return value


def aggregate_costs(records,contract,analysis_rows):
    require(len(records)==12 and [r['plan']['role'] for r in records]==list(contract.ROLES)
        and type(analysis_rows) is int and 0<=analysis_rows<=contract.LIMITS['max_analysis_feature_rows'])
    keys=('task_forwards','task_rows','crop_forwards','crop_rows','gate_evaluations','controller_calls','teacher_proposals')
    total={k:0 for k in keys};paired=0;steps=0
    for record in records:
        cost=record['model_costs'];require(type(cost) is dict and set(cost)==set(keys)
            and all(type(v) is int and v>=0 for v in cost.values()))
        active=record['row']['status']=='paired';require(record['row']['status'] in ('paired','no_eligible_target'))
        require(record['row']['role']==record['plan']['origin'] and record['official_actions']==record['plan']['official_actions'])
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
        and total['task_rows']+total['crop_rows']+analysis_rows<=contract.LIMITS['max_neural_input_rows']
        and steps<=contract.LIMITS['max_official_action_transitions'])
    return dict(model_costs=total,analysis_feature_rows=analysis_rows,
        total_input_rows_including_analysis=total['task_rows']+total['crop_rows']+analysis_rows,
        paired=paired,terminal_suffixes=2*paired,official_actions=steps,
        environment_constructions=2*paired,restores=2*paired,explicit_resets=0,rival_calls=1438*paired,
        source_trace_scans=2*paired,execution_source_replay_decodes=paired,
        verification_source_replay_decodes=paired,reconstructed_replays=2*paired,
        verification_reconstructed_replay_decodes=2*paired,training=False,gpu_training=False)


def budget_check(dm,contract,*,reserve=0):
    total=dm._owned_size(PRIVATE)+(dm._owned_size(STAGE) if STAGE.exists() else 0)
    require(dm._owned_size(PRIVATE)<=contract.LIMITS['private_result_bytes']
        and total+reserve<=contract.LIMITS['total_output_bytes']
        and shutil.disk_usage(GAME/'artifacts').free>=contract.LIMITS['disk_floor_bytes']+reserve)


def point_selection(source,archive,documents,plan):
    """Bind a unique job to one point without changing original source identity."""
    case=next(c for c in archive['cases'] if c['role']==plan['origin'])
    document=documents[plan['origin']]
    require(canonical(case)==canonical(document['source']))
    selected=source.selected_point(document,plan['point']);target=selected['selection']['target']
    require(selected['role']==plan['role'] and selected['split']==plan['split']
        and selected['selection']['index']==plan['index'] and selected['selection']['reason']==plan['reason']
        and (None if target is None else source.ordered_hash(target))==plan['target_sha256']
        and plan['official_actions']==(0 if target is None else 2*(719-target['step'])))
    return case,selected


def ledger_rows(archive,records,contract):
    require(len(records)==12 and [r['plan']['role'] for r in records]==list(contract.ROLES))
    output=[]
    for source,reference in zip(archive['cases'],archive['references']):
        origin=source['role'];old=reference['row']
        require(old['role']==reference['origin']==origin and old['status']=='paired' and old['technical_passed'] is True)
        output.append(dict(origin=origin,point='Q0',split=contract.split(origin),status='paired',
            technical_passed=True,H=old['H'],W=old['W']))
        for point in contract.POINTS:
            record=next(r for r in records if r['plan']['role']==origin+'-'+point);row=record['row'];plan=record['plan']
            require(row['role']==plan['origin']==origin and plan['point']==point and plan['split']==contract.split(origin)
                and all(row[k]==source[k] for k in ('cohort','opponent','side')))
            item=dict(origin=origin,point=point,split=plan['split'],status=row['status'])
            if row['status']=='no_eligible_target':item['reason']=plan['reason']
            else:item.update(technical_passed=row['technical_passed'],H=row['H'],W=row['W'])
            output.append(item)
    return output


def analyze_completed(infra,modules,archive,documents,records,contract,check):
    """Read bound normalization once; transform <=32 rows, with no score call."""
    dm,source=modules['driver'],modules['source']
    meter=dict(normalization_loads_attempted=0,normalization_loads_returned=0,
        feature_rows_requested=0,feature_rows_completed=None,model_forwards=0,model_constructions=0,
        reference_samples_read=0,new_samples_read=0)
    private_write(infra,'analysis-started.local.json',dict(max_feature_rows=32,model_forwards=0,training=False))
    try:
        samples=source.reference_samples(dm,archive,check);meter['reference_samples_read']=len(samples)
        require(len(samples)==4)
        for record in records:
            if record['row']['status']=='no_eligible_target':continue
            plan=record['plan'];case,selected=point_selection(source,archive,documents,plan)
            out=STAGE/plan['role'];raw=dm._read(out,out/'case.local.json',contract.LIMITS['max_receipt_bytes'])
            require(sha(raw)==record['document_sha256']);doc=infra.decode(raw)
            sample_raw=dm._read(out,out/'sample.local.json',contract.LIMITS['max_receipt_bytes'])
            require(sha(sample_raw)==doc['sample']['sha256'] and len(sample_raw)==doc['sample']['bytes'])
            samples.append(dict(origin=plan['origin'],point=plan['point'],split=plan['split'],
                seed=case['row']['seed'],target=selected['selection']['target'],sample=infra.decode(sample_raw)))
            meter['new_samples_read']+=1;check()
        require(4<=len(samples)<=contract.LIMITS['max_analysis_samples'])
        meter['feature_rows_requested']=2*len(samples)
        costs=aggregate_costs(records,contract,meter['feature_rows_requested'])  # Reserve before transforming.
        models={tuple(c['task']) for c in archive['cases']};require(len(models)==1)
        relative,expected=next(iter(models));path=GAME/relative
        require(dm._hash(GAME,path,dm.MAX_MODEL)==expected and check())
        import numpy as np
        from learning.task_bc_v1.model import DIMENSIONS
        from learning.task_choice_v1.features import OPS
        meter['normalization_loads_attempted']+=1
        with np.load(path,allow_pickle=False) as data:
            metadata=json.loads(str(data['metadata']))
            mean,scale=data['mean'].astype(np.float64),data['scale'].astype(np.float64)
        meter['normalization_loads_returned']+=1
        inputs=samples[0]['sample']['inputs']
        require(metadata['dimensions']==list(DIMENSIONS) and metadata['ops']==list(OPS)
            and metadata['schema_version']==inputs['schema_version'] and metadata['schema_sha256']==inputs['schema_sha256']
            and metadata['feature_names']==inputs['feature_names']
            and metadata['shared_feature_names']==inputs['shared_feature_names'])
        diagnostic=modules['analysis'].analyze(samples,mean,scale)
        rows=diagnostic['public']['counts']['encoded_hw_rows']
        require(rows==meter['feature_rows_requested'] and diagnostic['public']['counts']['model_forwards']==0
            and diagnostic['public']['counts']['model_constructions']==0)
        meter['feature_rows_completed']=rows
        require(dm._hash(GAME,path,dm.MAX_MODEL)==expected and check())
        private_write(infra,'feature-analysis.private.local.json',diagnostic['private'])
        private_write(infra,'feature-analysis.public.local.json',diagnostic['public'])
        return diagnostic['public'],costs,dict(meter)
    finally:
        error=sys.exc_info()[1]
        close(infra,'analysis-closed.local.json',dict(outcome='completed' if error is None else 'failed',
            error_kind=None if error is None else type(error).__name__,meter=meter,
            feature_rows_actual_unknown=meter['feature_rows_completed'] is None,
            counter_scope='requested rows reserved before normalization; completed rows known only on success',
            training=False,retry_allowed=False),error)


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
        documents={c['role']:selected_read(infra,binding,c) for c in archive['cases']}
        plans=source.plans([documents[o] for o in contract.ORIGINS])
        require(canonical(plans)==canonical(binding['plans']) and len(plans)==12)
        budget_check(dm,contract)
        if child:
            plan=next(p for p in plans if p['role']==role);out=STAGE/role
            case,selected=point_selection(source,archive,documents,plan)
            require(infra.read(STAGE/'binding.local.json')==raw
                and infra.decode(infra.read(out/'permit.local.json'))==marker('permit',commit,binding_sha,role))
            for name in ('started','consumed'):infra.write(out/(name+'.local.json'),marker(name,commit,binding_sha,role))
            modules['return_worker'].execute_case(modules,case,selected,out,check)
            require(check() and source.assets_verify(dm,archive,check))
            identity.verify_public(repo=REPO,game_root=GAME,registration_commit=commit)
            return None
        STAGE.mkdir(exist_ok=False);infra.write(STAGE/'binding.local.json',binding)
        require(infra.read(STAGE/'binding.local.json')==raw)
        for plan in plans:
            current=plan['role'];case,selected=point_selection(source,archive,documents,plan)
            gate=ready();budget_check(dm,contract,reserve=limits['per_case_output_bytes'])
            require(len(accepted)<12 and owner['deadline']-time.monotonic()>limits['job_wall_seconds']+limits['cleanup_seconds'])
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
            require(record['official_actions']==plan['official_actions']);record['plan']=plan
            infra.write(out/'accepted.local.json',dict(passed=True,role=current,registration_commit=commit,
                case_sha256=record['document_sha256'],supervision_sha256=sh,retry_allowed=False))
            accepted.append(record);private_write(infra,current+'.result.local.json',record)
        require(len(accepted)==12 and check() and source.assets_verify(dm,archive,check))
        identity.verify_public(repo=REPO,game_root=GAME,registration_commit=commit)
        diagnostic,costs,analysis_meter=analyze_completed(infra,modules,archive,documents,accepted,contract,check)
        rows=ledger_rows(archive,accepted,contract);projection=contract.projection(rows)
        require(check() and source.assets_verify(dm,archive,check))
        identity.verify_public(repo=REPO,game_root=GAME,registration_commit=commit)
        private_write(infra,'projected-summary.local.json',projection)
        private_write(infra,'result.local.json',dict(schema=SCHEMA+'-result',registration_commit=commit,
            binding_sha256=binding_sha,accepted=accepted,execution=costs,analysis_meter=analysis_meter,
            analysis_projection_sha256=sha(infra.read(PRIVATE/'feature-analysis.public.local.json')),
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
