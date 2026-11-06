"""One frozen four-game dual-model development campaign and its main replays.

Preparation authenticates completed data but executes no model or environment.
All inference and official execution are inside the single registered run.
"""
import argparse
import hashlib
import importlib
import importlib.util
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
BASE='bca1eb4b844cef2b407cbb15dff7e546b5ae1920'
PREFIX='kaggriculture/learning/task_harvest_model_dev_v1/'
EVIDENCE='review/evidence/task-harvest-model-dev-v1-20261106'
REGISTRATION,NOTE=EVIDENCE+'/registration.json',EVIDENCE+'/REGISTRATION.zh-CN.md'
PRIVATE=REPO.parent/'private/task-harvest-model-dev-v1-20261106'
STAGES={c:GAME/('artifacts/task-harvest-model-dev-'+c.lower()+'-v1-20261106') for c in ('C01','C02')}
INFRA='kaggriculture/learning/task_teacher_fresh_panel_v1/runner.py'
CHOICE_REG='review/evidence/task-harvest-choice-v1-20261106/registration.json'
CHOICE_RESULTS='review/evidence/task-harvest-choice-v1-result-20261106'
SCHEMA='task-harvest-model-dev-v1'


def require(value):
    if not value:raise ValueError('Registered harvest model development differs')


def sha(raw):return hashlib.sha256(raw).hexdigest()


def canonical(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=True,allow_nan=False).encode()


def git(*args):
    result=subprocess.run(['git','-C',str(REPO),*args],stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,stderr=subprocess.DEVNULL,timeout=45,
        creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    require(result.returncode==0 and len(result.stdout)<=16*1024**2)
    return result.stdout


def infrastructure():
    raw=(REPO/INFRA).read_bytes();require(raw==git('show',BASE+':'+INFRA))
    module=ModuleType('_harvest_model_frozen_infrastructure');module.__file__=str(REPO/INFRA)
    exec(compile(raw,module.__file__,'exec',dont_inherit=True),module.__dict__)
    return module


def source_map(infra):
    old=infra.decode(git('show','c4b60e5aad32406755c7570f4dee576ade16743c:'+CHOICE_REG))
    require(len(old['public_sources'])==530)
    extras={CHOICE_REG,NOTE,PREFIX+'README.zh-CN.md'}
    extras.update(CHOICE_RESULTS+'/'+name for name in ('summary.json','certificate.json',
        'RESULTS.zh-CN.md','REVIEW.zh-CN.md','NEXT.zh-CN.md'))
    paths=set(old['public_sources'])|extras
    paths.update(p.relative_to(REPO).as_posix() for p in (REPO/PREFIX).glob('*.py'))
    require({PREFIX+n for n in ('__init__.py','contract.py','policy.py','precheck.py','sources.py',
        'worker.py','runner.py','test_policy.py','test_precheck.py','test_worker.py','test_sources.py',
        'test_contract.py','test_runner.py')}.issubset(paths))
    for path,expected in old['public_sources'].items():
        raw=infra.read(REPO/path)
        if path=='review/PROGRESS.md':
            original=git('show','c4b60e5aad32406755c7570f4dee576ade16743c:'+path)
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
    infra=infrastructure()
    infra.REGISTRATION,infra.PREFIX,infra.BASE=REGISTRATION,PREFIX,BASE
    infra._required_sources=lambda _old:set(source_map(infra))
    infra.validate_registration=validate_registration
    reg,contract,identity,bootstrap,base=infra.authenticate(commit)
    require(not os.path.lexists(GAME/'learning/task_harvest_model_dev_v1'))
    return infra,reg,contract,identity,bootstrap,base


def metadata_modules():
    require('numpy' not in sys.modules and 'kaggle_environments' not in sys.modules)
    os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
    sys.path.insert(0,str(REPO/'kaggriculture'))
    result={key:importlib.import_module('learning.'+name) for key,name in (
        ('source','task_harvest_model_dev_v1.sources'),('driver','task_pair_dev_campaign_v1.driver'))}
    require('kaggle_environments' not in sys.modules and 'torch' not in sys.modules)
    return result


def runtime(infra,bootstrap,reg):
    result=infra._runtime(bootstrap,reg)
    for key,name in (('source','sources'),('harvest_worker','worker'),('precheck','precheck')):
        module=importlib.import_module('learning.task_harvest_model_dev_v1.'+name)
        require(Path(module.__file__).resolve(strict=True)==(REPO/PREFIX/(name+'.py')).resolve(strict=True))
        result[key]=module
    return result


def official_identity(infra,contract):
    spec=importlib.util.find_spec('kaggle_environments')
    require(spec is not None and len(spec.submodule_search_locations)==1)
    root=Path(next(iter(spec.submodule_search_locations)))
    expected={'envs/kaggriculture/kaggriculture.py':contract.ENGINE_SHA,
        'envs/kaggriculture/kaggriculture.json':contract.SPEC_SHA,'core.py':contract.CORE_SHA,
        'utils.py':contract.UTILS_SHA,'agent.py':contract.AGENT_SHA}
    require(all(sha(infra.read(root/p))==h for p,h in expected.items())
        and sha(infra.read(REPO/'kaggriculture/bots/teacher_fertilizer_terminal_v2.py'))==contract.TEACHER_SHA)
    return expected


def private_write(infra,name,value,*,closure=False):
    # Reuse the already exercised exclusive writer and closure reserve.
    path='kaggriculture/learning/task_harvest_trajectory_v1/runner.py'
    shared=infra.snapshot(path,sha(infra.read(REPO/path)))
    shared.PRIVATE=PRIVATE
    limits=dict(max_private_bytes=16*1024**2,closure_reserve_bytes=1024**2,max_output_file_bytes=8*1024**2)
    return shared.private_write(infra,limits,name,value,closure=closure)


def close(infra,name,value,primary):
    try:return private_write(infra,name,value,closure=True)
    except BaseException:
        if primary is None:raise
        return None


def prepare():
    infra=infrastructure()
    require(os.name=='nt' and sys.flags.isolated and sys.flags.dont_write_bytecode and __debug__
        and git('rev-parse','HEAD').decode().strip()==BASE and not os.path.lexists(PRIVATE)
        and not os.path.lexists(REPO/REGISTRATION) and all(not os.path.lexists(s) for s in STAGES.values()))
    sources=source_map(infra)
    contract=infra.snapshot(PREFIX+'contract.py',sources[PREFIX+'contract.py'])
    base=infra.snapshot(infra.BASE_ENTRY,sources[infra.BASE_ENTRY])
    ready=infra.snapshot('kaggriculture/learning/task_teacher_fresh_panel_v1/contract.py',
        sources['kaggriculture/learning/task_teacher_fresh_panel_v1/contract.py'])
    deadline=time.monotonic()+900
    def check():
        if base.physical_available()<contract.LIMITS['running_available_bytes']:raise MemoryError('Preparation floor')
        if time.monotonic()>=deadline:raise TimeoutError('Preparation deadline')
        return True
    ready.await_memory(base.physical_available,floor=contract.LIMITS['initial_available_bytes'])
    identity=infra.snapshot(infra.IDENTITY,sources[infra.IDENTITY])
    infra.preparation_runtime(identity,BASE)
    rules=official_identity(infra,contract)
    PRIVATE.mkdir(exist_ok=False)
    private_write(infra,'prepare-started.local.json',dict(base_commit=BASE,public_sources=sources,
        specification=contract.specification()))
    binding_sha=None
    try:
        modules=metadata_modules();source,dm=modules['source'],modules['driver']
        context=source.source_context(check,full=True)
        archive=source.archive_metadata(infra,check)
        assets=source.required_assets(context,dm,GAME,check)
        plans={c:list(contract.HarvestPanel(context.manifests[c].panel,c).plans()) for c in contract.COHORTS}
        require(sum(len(v) for v in plans.values())==4 and check())
        binding=dict(schema=SCHEMA+'-binding',base_commit=BASE,salt=secrets.token_hex(32),
            source_parent_commit=BASE,completed_sources=context.identities,assets=assets,plans=plans,
            default_sha256=archive['default_sha256'],archival_witness_sha256=archive['witness_sha256'],
            archival_commitment=archive['prior_commitment'])
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
            official_constructions=0,official_actions=0,model_forwards=0,retry_allowed=False),error)
    return dict(status='prepared',old_games_authenticated=32,old_streams_decoded=0,
        model_forwards=0,official_actions=0,frozen_sources=len(sources),games_registered=4)


def marker(kind,commit,binding_sha,cohort=None,game=None):
    require(kind in ('attempt','permit','started','consumed'))
    result=dict(schema=SCHEMA+'-marker',kind=kind,registration_commit=commit,
        binding_sha256=binding_sha,retry_allowed=False)
    if kind!='attempt':
        require(cohort in STAGES and type(game) is str)
        result.update(cohort=cohort,game=game)
    return result


def binding_read(infra,reg):
    raw=infra.read(PRIVATE/'binding.local.json');binding=infra.decode(raw)
    ready=infra.decode(infra.read(PRIVATE/'prepare-closed.local.json'))
    require(sha(raw)==reg['binding_commitment_sha256']==ready['binding_sha256']
        and binding['schema']==SCHEMA+'-binding' and binding['base_commit']==BASE
        and ready['outcome']=='prepared' and ready['error_kind'] is None)
    return raw,binding


def operation_check(value):
    expected={'live':dict(environment_constructions=1,resets=2,official_steps=719,actor_calls=719,rival_calls=719),
        'reconstruction':dict(actor_calls=719,rival_calls=719),
        'replay':dict(environment_constructions=1,resets=1,official_steps=719)}
    require(type(value) is dict and set(value)==set(expected))
    for phase,items in expected.items():
        wanted={k+suffix:v for k,v in items.items() for suffix in ('_attempted','_returned')}
        require(canonical(value[phase])==canonical(wanted))


def precheck_check(value,meter,contract):
    require(value.get('passed') is True and canonical(value.get('meter'))==canonical(meter)
        and meter.get('status')=='passed' and meter.get('scope')=='precheck'
        and canonical(value.get('tolerances'))==canonical({k:contract.PRECHECK_LIMITS[k] for k in ('atol','rtol')}))
    for key,n in (('point',4),('actor_construction',8),('controller',8),('score',8)):
        require(all(type(meter.get(key+suffix)) is int and meter[key+suffix]==n
            for suffix in ('_attempts','_returns')))
    require(type(meter.get('accounting_failures')) is int and meter['accounting_failures']==0
        and type(meter.get('emission_failures')) is int and meter['emission_failures']==0
        and [r['role'] for r in value['points']]==list(contract.ROLES)
        and all(r['passed'] is True for r in value['points']))


def aggregate_costs(pre_meter,rows,contract):
    names={'task_forwards','task_rows','crop_forwards','crop_rows','gate_evaluations','controller_calls','teacher_proposals'}
    require(len(rows)==4 and set(pre_meter['model_costs'])==names)
    result={key:0 for key in names}
    for cost in [pre_meter['model_costs'],*(r['model_counts'] for r in rows)]:
        require(type(cost) is dict and set(cost)==names and all(type(v) is int and v>=0 for v in cost.values()))
        for key,value in cost.items():result[key]+=value
    require(result['controller_calls']==contract.LIMITS['max_controller_calls']==5760
        and result['gate_evaluations']==result['teacher_proposals']==0
        and result['task_forwards']<=contract.LIMITS['max_task_forwards']
        and result['crop_forwards']<=contract.LIMITS['max_crop_forwards']
        and result['task_rows']+result['crop_rows']<=contract.LIMITS['max_neural_input_rows'])
    return result


def budget_check(dm,contract,*,reserve=0):
    total=dm._owned_size(PRIVATE)+sum(dm._owned_size(s) for s in STAGES.values() if s.exists())
    require(dm._owned_size(PRIVATE)<=contract.LIMITS['private_result_bytes']
        and total+reserve<=contract.LIMITS['total_output_bytes']
        and shutil.disk_usage(GAME/'artifacts').free>=contract.LIMITS['disk_floor_bytes']+reserve)


def run(commit,*,cohort=None,game=None):
    infra,reg,contract,identity,bootstrap,base=authenticate(commit)
    raw,binding=binding_read(infra,reg);binding_sha=sha(raw)
    child=game is not None
    require((cohort in STAGES if child else cohort is None))
    import psutil
    started=time.monotonic();limits=contract.LIMITS;latest={'available_bytes':None,'sample_monotonic':None}
    def available():
        n=base.physical_available();latest.update(available_bytes=n,sample_monotonic=time.monotonic());return n
    owner=(infra.decode(infra.read(PRIVATE/'owner.local.json')) if child else
        dict(schema='task-pair-fresh-multi-owner-v1',registration_commit=commit,binding_sha256=binding_sha,
            pid=os.getpid(),created_us=int(psutil.Process().create_time()*1000000),
            deadline=started+limits['total_wall_seconds']))
    process=psutil.Process();owned=False;accepted=[];cleanup=[];pre_meter={}
    def check():
        if available()<limits['running_available_bytes'] or process.memory_info().rss>limits['max_rss_bytes']:
            raise MemoryError('Runtime resource floor')
        if time.monotonic()>=owner['deadline']:raise TimeoutError('Campaign deadline')
        require(infra.read(PRIVATE/'binding.local.json')==raw and not os.path.lexists(PRIVATE/'closed.local.json'))
        return True
    readiness=infra.snapshot('kaggriculture/learning/task_teacher_fresh_panel_v1/contract.py',
        reg['public_sources']['kaggriculture/learning/task_teacher_fresh_panel_v1/contract.py'])
    def ready():
        return readiness.await_memory(available,timeout=min(limits['readiness_timeout_seconds'],owner['deadline']-time.monotonic()))
    try:
        if not child:
            require(all(not os.path.lexists(p) for p in (PRIVATE/'attempt.local.json',PRIVATE/'owner.local.json',
                PRIVATE/'closed.local.json',*STAGES.values())))
            private_write(infra,'attempt.local.json',marker('attempt',commit,binding_sha));owned=True
            private_write(infra,'owner.local.json',owner)
            ready()
        else:check()
        modules=runtime(infra,bootstrap,reg)
        dm,source=modules['driver'],modules['source']
        if child:
            modules['fresh'].verify_live_owner(owner,commit=commit,binding_sha=binding_sha,current_pid=os.getpid(),
                process=psutil.Process,executable=sys.executable,samefile=os.path.samefile,now=time.monotonic())
            require(infra.decode(infra.read(PRIVATE/'attempt.local.json'))==marker('attempt',commit,binding_sha))
        context=source.source_context(check,full=not child)
        require(context.identities==binding['completed_sources']
            and source.required_assets(context,dm,GAME,check)==binding['assets']
            and sha(infra.read(GAME/'main.py'))==binding['default_sha256']
            and official_identity(infra,contract)==reg['installed_source_sha256'])
        panels={c:contract.HarvestPanel(context.manifests[c].panel,c) for c in contract.COHORTS}
        require({c:list(p.plans()) for c,p in panels.items()}==binding['plans'])
        budget_check(dm,contract)
        if child:
            stage=STAGES[cohort];manifest=context.manifests[cohort];panel=panels[cohort]
            selected=[p for p in panel.plans() if p['game']==game];require(len(selected)==1)
            require(infra.read(stage/'manifest.local.json')==raw
                and infra.decode(infra.read(stage/'job-results'/(game+'.permit.local.json')))
                    ==marker('permit',commit,binding_sha,cohort,game))
            for name in ('started','consumed'):
                infra.write(stage/'job-results'/(game+'.'+name+'.local.json'),marker(name,commit,binding_sha,cohort,game))
            counters={}
            try:
                modules['harvest_worker'].execute_job(modules=modules,manifest=manifest,panel=panel,
                    stage=stage,plan=selected[0],check=check,counters=counters)
                operation_check(counters)
                require(check() and sha(infra.read(GAME/'main.py'))==binding['default_sha256'])
                identity.verify_public(repo=REPO,game_root=GAME,registration_commit=commit)
            finally:
                error=sys.exc_info()[1]
                try:
                    infra.write(stage/'job-results'/(game+'.operations.local.json'),dict(counts=counters,
                        outcome='completed' if error is None else 'failed',
                        error_kind=None if error is None else type(error).__name__,retry_allowed=False))
                except BaseException:
                    if error is None:raise
            return None
        archive=source.archive_metadata(infra,check)
        require(archive['witness_sha256']==binding['archival_witness_sha256']
            and archive['prior_commitment']==binding['archival_commitment'])
        emit=lambda name,value:private_write(infra,name,value)
        points=source.points(infra,archive,context,check,emit)
        def emit_precheck(value):
            require(type(value) is dict and value.get('scope')=='precheck'
                and value.get('event') in ('precheck_controller','precheck_score','precheck_point','precheck_failure')
                and value.get('role') in (None,*contract.ROLES)
                and value.get('branch') in (None,'original','disabled','original_pool','expanded_pool'))
            name=(value.get('role') or 'panel')+'.'+value['event']
            if value.get('branch'):name+='.'+value['branch']
            private_write(infra,name+'.local.json',value)
        pre=modules['precheck'].run(points,game_root=GAME,guard=check,emit=emit_precheck,meter=pre_meter)
        precheck_check(pre,pre_meter,contract)
        private_write(infra,'precheck-result.local.json',pre)
        require(check())
        for current,stage in STAGES.items():
            manifest=context.manifests[current];panel=panels[current]
            driver=dm.LocalDriver(manifest=manifest,game_root=GAME,stage=stage,commit=commit,
                command_for=lambda _name:[],check=check)
            require(driver.claim());infra.write(stage/'manifest.local.json',binding)
            for plan in panel.plans():
                gate=ready();budget_check(dm,contract,reserve=limits['per_game_reserve_bytes'])
                require(len(accepted)<4 and owner['deadline']-time.monotonic()>limits['job_wall_seconds']+limits['cleanup_seconds'])
                name=plan['game'];where=stage/'job-results'
                infra.write(where/(name+'.readiness.local.json'),gate)
                infra.write(where/(name+'.permit.local.json'),marker('permit',commit,binding_sha,current,name))
                command=[sys.executable,'-I','-B',str(Path(__file__).resolve()),'run',
                    '--registration-commit',commit,'--cohort',current,'--game',name]
                supervision=infra.supervise_recorded(modules['campaign'],command,GAME,
                    dict(max_rss_bytes=limits['max_rss_bytes'],max_commit_bytes=limits['max_commit_bytes'],
                        min_available_bytes=limits['running_available_bytes'],wall_seconds=limits['job_wall_seconds'],
                        cleanup_seconds=limits['cleanup_seconds'],poll_seconds=limits['poll_seconds']))
                sh=infra.write(where/(name+'.supervision.local.json'),dict(supervision=supervision,retry_allowed=False))
                cleanup.append(dict(cohort=current,game=name,sha256=sh,supervision=supervision))
                acceptance=modules['acceptance'];require(acceptance.supervision_ok(supervision) and check())
                bundle={k:dm._read(stage,where/(name+suffix),8*1024**2 if k=='metrics' else acceptance.MAX_RECEIPT_BYTES)
                    for k,suffix in dm.RECEIPTS.items()}
                metrics=acceptance.accept(panel,plan,bundle,supervision,lambda p,r:
                    modules['harvest_worker'].verify_files(modules=modules,manifest=manifest,panel=panel,
                        stage=stage,plan=p,row=r))
                row,audit=(acceptance._json(bundle[k],acceptance.MAX_RECEIPT_BYTES) for k in ('row','audit'))
                operations=infra.decode(infra.read(where/(name+'.operations.local.json')))
                require(operations['outcome']=='completed' and operations['error_kind'] is None
                    and operations['counts']==audit['harvest_model_operation_counts']
                    and audit['official']['shed_order_frames_checked']==720
                    and audit['parity_kind'] is None and audit['original_parity_model_counts']=={}
                    and type(audit['original_controller_parity_turns']) is int and audit['original_controller_parity_turns']==0)
                require(canonical(audit['total_model_counts'])==canonical({k:2*v for k,v in audit['primary_model_counts'].items()}))
                operation_check(operations['counts'])
                for kind in ('permit','started','consumed'):
                    require(infra.decode(infra.read(where/(name+'.'+kind+'.local.json')))==marker(kind,commit,binding_sha,current,name))
                src=next(s for s in contract.sources() if s['cohort']==current
                    and all(s[k]==plan[k] for k in ('seed_alias','opponent','side')))
                baseline=context.rows[(current,src['game'])]
                require(row['initial_banks']==baseline['initial_banks'])
                record=dict(role=src['role'],cohort=current,opponent=src['opponent'],side=src['side'],
                    baseline=baseline,candidate=row,harvest_choice=metrics['result']['harvest_choice'],
                    model_counts=audit['total_model_counts'],operations=operations['counts'])
                receipt=dict(passed=True,role=src['role'],registration_commit=commit,
                    bundle_sha256={k:sha(v) for k,v in bundle.items()},supervision_sha256=sh,
                    operations_sha256=sha(infra.read(where/(name+'.operations.local.json'))),retry_allowed=False)
                infra.write(where/(name+'.accepted.local.json'),receipt)
                accepted.append(record)
                private_write(infra,src['role']+'.result.local.json',record)
        require(len(accepted)==4 and check())
        # Source files, default and used weights are rechecked, not reexecuted.
        require(source.required_assets(context,dm,GAME,check)==binding['assets']
            and sha(infra.read(GAME/'main.py'))==binding['default_sha256'])
        identity.verify_public(repo=REPO,game_root=GAME,registration_commit=commit)
        projection=contract.projection(accepted)
        model_costs=aggregate_costs(pre_meter,accepted,contract)
        private_write(infra,'projected-summary.local.json',projection)
        private_write(infra,'result.local.json',dict(schema=SCHEMA+'-result',registration_commit=commit,
            binding_sha256=binding_sha,accepted=accepted,precheck=pre,
            total_model_costs_including_precheck=model_costs,
            projection_sha256=sha(infra.read(PRIVATE/'projected-summary.local.json'))))
        budget_check(dm,contract)
        return dict(status='completed',games=4,official_replays=4,training=False,gpu=False)
    finally:
        if owned:
            error=sys.exc_info()[1]
            memory=infra.closing_memory(error,latest,available)
            close(infra,'closed.local.json',dict(schema=SCHEMA+'-closed',registration_commit=commit,
                binding_sha256=binding_sha,outcome='completed' if error is None else 'failed',
                error_kind=None if error is None else type(error).__name__,accepted_games=len(accepted),
                precheck_counts=pre_meter,child_cleanup_receipts=cleanup,**memory,
                elapsed_seconds=time.monotonic()-started,retry_allowed=False),error)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);commands=parser.add_subparsers(dest='command',required=True)
    commands.add_parser('prepare');run_parser=commands.add_parser('run')
    run_parser.add_argument('--registration-commit',required=True)
    run_parser.add_argument('--cohort');run_parser.add_argument('--game')
    args=parser.parse_args(argv)
    try:
        result=prepare() if args.command=='prepare' else run(args.registration_commit,cohort=args.cohort,game=args.game)
        if result is not None:print(json.dumps(result,separators=(',',':')))
        return 0
    except BaseException as error:
        print(json.dumps(dict(status='failed',error_kind=type(error).__name__,retry_allowed=False)))
        return 1


if __name__=='__main__':raise SystemExit(main())
