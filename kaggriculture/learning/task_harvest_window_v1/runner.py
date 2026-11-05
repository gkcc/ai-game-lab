"""Registered four-point official continuation, with no training or new seeds.

The old source authenticator, runtime binding, memory gates and Windows Job
supervisor are reused. This file is the coordinator, not a second game engine.
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
GAME = REPO.parent.parent / 'kaggriculture'
BASE = 'b58fda94abc06b377ca341bf4d17fead2eb6dff2'
PRIOR_COMMIT = '0ec1175a446fa44eeffb5bd8c4b8cf43ae473499'
PRIOR_REG = 'review/evidence/task-joint-decoder-v1-20261106/registration.json'
PRIOR_RESULTS = 'review/evidence/task-joint-decoder-v1-result-20261106'
PREFIX = 'kaggriculture/learning/task_harvest_window_v1/'
EVIDENCE = 'review/evidence/task-harvest-window-v1-20261106'
REGISTRATION = EVIDENCE + '/registration.json'
NOTE = EVIDENCE + '/REGISTRATION.zh-CN.md'
PRIVATE = REPO.parent / 'private/task-harvest-window-v1-20261106'
STAGE = GAME / 'artifacts/task-harvest-window-v1-20261106'
INFRA = 'kaggriculture/learning/task_teacher_fresh_panel_v1/runner.py'
READY = 'kaggriculture/learning/task_teacher_fresh_panel_v1/contract.py'
SCHEMA = 'task-harvest-window-prereg-v1'
MAX_SMALL = 1024*1024


def require(value):
    if not value:
        raise ValueError('Registered harvest window differs')


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def git(*args):
    result = subprocess.run(['git', '-C', str(REPO), *args], stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, timeout=45,
        creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
    require(result.returncode == 0 and len(result.stdout) <= 16*1024**2)
    return result.stdout


def infrastructure():
    raw = (REPO/INFRA).read_bytes()
    require(raw == git('show', BASE+':'+INFRA))
    module = ModuleType('_harvest_window_infrastructure')
    module.__file__ = str(REPO/INFRA)
    exec(compile(raw, module.__file__, 'exec', dont_inherit=True), module.__dict__)
    return module


def required_sources(infra):
    previous = infra.decode(git('show', PRIOR_COMMIT+':'+PRIOR_REG))
    require(len(previous['public_sources']) == 401)
    paths = set(previous['public_sources']) | {PRIOR_REG, NOTE, PREFIX+'README.zh-CN.md'}
    evidence = {PRIOR_RESULTS+'/'+name for name in
        ('summary.json','certificate.json','RESULTS.zh-CN.md','REVIEW.zh-CN.md','NEXT.zh-CN.md')}
    paths.update(evidence)
    paths.update(p.relative_to(REPO).as_posix() for p in (REPO/PREFIX).glob('*.py'))
    require({PREFIX+name for name in ('__init__.py','contract.py','selection.py','core.py',
        'gate.py','worker.py','runner.py')}.issubset(paths))
    for path, expected in previous['public_sources'].items():
        raw = infra.read(REPO/path)
        if path == 'review/PROGRESS.md':
            old = git('show', PRIOR_COMMIT+':'+path)
            published = git('show', BASE+':'+path)
            require(sha(old) == expected and published.startswith(old) and raw == published)
        else:
            require(sha(raw) == expected)
    for path in evidence | {PRIOR_REG}:
        require(infra.read(REPO/path) == git('show', BASE+':'+path))
    return paths


def validate_registration(reg, contract):
    require(type(reg) is dict and reg.get('schema') == SCHEMA and reg.get('base_commit') == BASE
        and reg.get('engine_commit') == contract.ENGINE_COMMIT
        and reg.get('engine_sha256') == contract.ENGINE_SHA256
        and json.dumps(reg.get('limits'),sort_keys=True,allow_nan=False)
            == json.dumps(contract.LIMITS,sort_keys=True,allow_nan=False)
        and reg.get('panel') == contract.specification()
        and reg.get('evaluation') == dict(training=False,gpu=False,new_seeds=False,
            independent_evaluation=False,statistical_significance=False,promotion=False,
            terminal_profit=False,retries=False)
        and type(reg.get('binding_commitment_sha256')) is str
        and re.fullmatch('[0-9a-f]{64}',reg['binding_commitment_sha256']) is not None)


def authenticate(commit):
    infra = infrastructure()
    infra.REGISTRATION,infra.PREFIX,infra.BASE = REGISTRATION,PREFIX,BASE
    infra._required_sources = lambda _old: required_sources(infra)
    infra.validate_registration = validate_registration
    reg,contract,identity,bootstrap,base = infra.authenticate(commit)
    require(not os.path.lexists(GAME/'learning/task_harvest_window_v1'))
    return infra,reg,contract,identity,bootstrap,base


def runtime(infra,bootstrap,reg):
    modules = infra._runtime(bootstrap,reg)
    for key,name in (('selection','selection'),('window_worker','worker')):
        module = importlib.import_module('learning.task_harvest_window_v1.'+name)
        require(Path(module.__file__).resolve(strict=True) == (REPO/PREFIX/(name+'.py')).resolve(strict=True))
        modules[key] = module
    return modules


def close_preserving_primary(writer,value,error):
    try:
        writer(value)
    except BaseException:
        if error is None:
            raise
        return False
    return True


def source_context(infra,reg,modules,base,contract,binding,check,*,child=False):
    context = modules['sources'].load(guard=check,verify_game_files=not child)
    require(context.identities == binding['completed_sources']
        and sha(infra.read(GAME/'main.py')) == binding['default_sha256']
        and all(base.validate_assets(context.manifests[c],modules['driver']) for c in ('C01','C02'))
        and [s['source'] for s in binding['selections']] == contract.sources())
    for case in binding['selections']:
        target = case['selection']['target']
        if target is not None:
            require(target['side'] == case['source']['side']
                    and contract.WINDOW[0] <= target['step'] <= contract.WINDOW[1])
    return context


def prepare():
    require(os.name == 'nt' and sys.flags.isolated and sys.flags.dont_write_bytecode and __debug__)
    infra = infrastructure()
    require(git('rev-parse','HEAD').decode().strip() == BASE
        and all(not os.path.lexists(p) for p in (PRIVATE,STAGE,REPO/REGISTRATION)))
    identities = {p:sha(infra.read(REPO/p)) for p in sorted(required_sources(infra))}
    contract = infra.snapshot(PREFIX+'contract.py',identities[PREFIX+'contract.py'])
    base = infra.snapshot(infra.BASE_ENTRY,identities[infra.BASE_ENTRY])
    ready = infra.snapshot(READY,identities[READY])
    ready.await_memory(base.physical_available)
    deadline = time.monotonic()+900
    def check():
        if base.physical_available() < contract.LIMITS['running_available_bytes']:
            raise MemoryError('Preparation memory floor')
        if time.monotonic() >= deadline:
            raise TimeoutError('Preparation deadline')
        return True
    identity = infra.snapshot(infra.IDENTITY,identities[infra.IDENTITY])
    infra.preparation_runtime(identity,BASE)
    reg = dict(schema=SCHEMA,base_commit=BASE,engine_commit=contract.ENGINE_COMMIT,
        engine_sha256=contract.ENGINE_SHA256,public_sources=identities,versions=infra._versions(),
        limits=contract.LIMITS,panel=contract.specification(),evaluation=dict(training=False,gpu=False,
            new_seeds=False,independent_evaluation=False,statistical_significance=False,
            promotion=False,terminal_profit=False,retries=False))
    modules = runtime(infra,infra.snapshot(infra.BOOTSTRAP,identities[infra.BOOTSTRAP]),reg)
    require(not os.path.lexists(GAME/'learning/task_harvest_window_v1'))
    PRIVATE.mkdir(exist_ok=False)
    start_sha = infra.write(PRIVATE/'selection-started.local.json',dict(schema='task-harvest-selection-started-v1',
        base_commit=BASE,public_sources=identities,specification=contract.specification(),
        no_counterfactual_outcomes_read=True,local_registration_not_third_party_timestamp=True))
    binding_sha = None
    selected = []
    try:
        context = modules['sources'].load(guard=check)
        require(all(base.validate_assets(context.manifests[c],modules['driver']) for c in ('C01','C02')))
        rules = sys.modules['kaggle_environments.envs.kaggriculture.kaggriculture']
        first_days = modules['selection'].first_yield_days_from_source(infra.read(Path(rules.__file__)))
        for source in contract.sources():
            check()
            row = context.rows[(source['cohort'],source['game'])]
            stage = GAME/'artifacts'/modules['fresh'].STAGES[source['cohort']]
            def turns():
                for value in modules['driver']._trace(stage,stage/'traces'/(source['game']+'.jsonl.gz'),row['trace_sha256']):
                    check()
                    yield value
            selection = modules['selection'].select_target(turns(),side=source['side'],first_yield_days=first_days,guard=check)
            selected.append(dict(source=source,selection=selection))
            infra.write(PRIVATE/(source['role']+'.selection.local.json'),selected[-1],MAX_SMALL)
        require(len(selected) == 4 and check()
            and all(sha(infra.read(REPO/p)) == h for p,h in identities.items()))
        binding = dict(schema='task-harvest-window-binding-v1',base_commit=BASE,salt=secrets.token_hex(32),
            selection_started_sha256=start_sha,completed_sources=context.identities,
            default_sha256=sha(infra.read(GAME/'main.py')),selections=selected)
        binding_sha = infra.write(PRIVATE/'binding.local.json',binding)
        reg['binding_commitment_sha256'] = binding_sha
        validate_registration(reg,contract)
        (REPO/EVIDENCE).mkdir(parents=True,exist_ok=True)
        infra.write(REPO/REGISTRATION,reg)
    finally:
        error = sys.exc_info()[1]
        close_preserving_primary(lambda value:infra.write(PRIVATE/'selection-closed.local.json',value),
            dict(schema='task-harvest-selection-closed-v1',outcome='prepared' if error is None else 'failed',
                error_kind=None if error is None else type(error).__name__,binding_sha256=binding_sha,
                selection_started_sha256=start_sha,source_roles_scanned=len(selected),
                model_forwards=0,official_transitions=0,retry_allowed=False),error)
    return dict(status='prepared_no_windows',roles=4,eligible_targets=sum(x['selection']['target'] is not None for x in selected),
                source_trace_turns=4*719,model_forwards=0,official_transitions=0)


def binding_read(infra,reg):
    raw = infra.read(PRIVATE/'binding.local.json')
    binding = infra.decode(raw)
    closed = infra.decode(infra.read(PRIVATE/'selection-closed.local.json'))
    require(sha(raw) == reg['binding_commitment_sha256']
        and binding.get('schema') == 'task-harvest-window-binding-v1' and binding.get('base_commit') == BASE
        and closed.get('schema') == 'task-harvest-selection-closed-v1'
        and closed.get('outcome') == 'prepared' and closed.get('error_kind') is None
        and closed.get('binding_sha256') == sha(raw) and closed.get('source_roles_scanned') == 4
        and closed.get('model_forwards') == closed.get('official_transitions') == 0
        and closed.get('retry_allowed') is False
        and closed.get('selection_started_sha256') == binding['selection_started_sha256']
            == sha(infra.read(PRIVATE/'selection-started.local.json')))
    for case in binding['selections']:
        require(infra.decode(infra.read(PRIVATE/(case['source']['role']+'.selection.local.json'),MAX_SMALL)) == case)
    return raw,binding


def marker(kind,commit,binding_sha,role=None):
    require(kind in ('attempt','permit','started','consumed'))
    value = dict(schema='task-harvest-window-marker-v1',kind=kind,
        registration_commit=commit,binding_sha256=binding_sha,retry_allowed=False)
    if kind != 'attempt':
        require(role in ('F01','F02','F03','F04'))
        value['role'] = role
    return value


def budget_check(dm,contract,reserve=0):
    require(dm._owned_size(PRIVATE) <= contract.LIMITS['private_result_bytes']
        and (not STAGE.exists() or dm._owned_size(STAGE)+reserve <= contract.LIMITS['total_output_bytes'])
        and shutil.disk_usage(GAME/'artifacts').free >= contract.LIMITS['disk_floor_bytes']+reserve)


def aggregate_meters(documents):
    calls = {}
    costs = {'prefix':{},'suffix':{}}
    unknown = 0
    for document in documents:
        meter = document['meter']
        for key,value in meter['calls'].items():
            require(type(value) is int and value >= 0)
            calls[key] = calls.get(key,0)+value
        for scope in costs:
            for key,value in meter['budget_by_scope'][scope].items():
                require(type(value) is int and value >= 0)
                costs[scope][key] = costs[scope].get(key,0)+value
        require(type(meter['unknown_budget_observations']) is int)
        unknown += meter['unknown_budget_observations']
    return dict(calls=calls,budget_by_scope=costs,unknown_budget_observations=unknown)


def check_complete_counts(documents,meter,contract):
    paired = [d for d in documents if d['row']['status'] == 'paired']
    n = len(paired)
    require(len(documents) == 4 and meter['unknown_budget_observations'] == 0)
    calls = meter['calls']
    for operation,expected in (('arm',2*n),('actor_construct',2*n),('rival_construct',2*n),
                               ('restore',2*n),('official_step',96*n)):
        require(calls.get(operation+'_attempts',0) == calls.get(operation+'_returns',0) == expected)
    turns = sum(2*(d['target']['step']+48) for d in paired)
    require(turns <= contract.LIMITS['max_controller_calls']
        and calls.get('actor_attempts',0) == calls.get('actor_returns',0) == turns
        and calls.get('rival_attempts',0) == calls.get('rival_returns',0) == turns
        and calls.get('trace_close_failures',0) == 0)
    totals = {key:sum(meter['budget_by_scope'][scope].get(key,0) for scope in ('prefix','suffix'))
              for key in ('task_forwards','crop_forwards','task_rows','crop_rows','controller_calls')}
    require(totals['controller_calls'] == turns
        and totals['task_forwards'] <= contract.LIMITS['max_task_forwards']
        and totals['crop_forwards'] <= contract.LIMITS['max_crop_forwards']
        and totals['task_rows']+totals['crop_rows'] <= contract.LIMITS['max_neural_input_rows'])
    return dict(paired_roles=n,official_transitions=96*n,environment_constructions=2*n,
                original_controller_calls=turns,rival_calls=turns,model_budget_totals=totals)


def run(commit,*,preflight=False,role=None):
    infra,reg,contract,identity,bootstrap,base = authenticate(commit)
    raw,binding = binding_read(infra,reg)
    commitment = sha(raw)
    child = role is not None
    require(not (child and preflight))
    started = time.monotonic()
    latest = dict(available_bytes=None,sample_monotonic=None)
    def available():
        value = base.physical_available()
        latest.update(available_bytes=value,sample_monotonic=time.monotonic())
        return value
    import psutil
    owner = (infra.decode(infra.read(PRIVATE/'owner.local.json')) if child else dict(
        schema='task-pair-fresh-multi-owner-v1',registration_commit=commit,binding_sha256=commitment,
        pid=os.getpid(),created_us=int(psutil.Process().create_time()*1_000_000),
        deadline=started+contract.LIMITS['total_wall_seconds']))
    owner_sha = sha(infra.read(PRIVATE/'owner.local.json')) if child else None
    attempt_sha,result_sha = None,None
    owned = False
    accepted,supervisions,documents = [],[],[]
    def check():
        if available() < contract.LIMITS['running_available_bytes']:
            raise MemoryError('Runtime memory floor')
        if time.monotonic() >= owner['deadline']:
            raise TimeoutError('Campaign deadline')
        require(infra.read(PRIVATE/'binding.local.json') == raw
            and not os.path.lexists(PRIVATE/'closed.local.json'))
        if owner_sha is not None:
            require(sha(infra.read(PRIVATE/'owner.local.json')) == owner_sha)
        return True
    ready = infra.snapshot(READY,reg['public_sources'][READY],commit)
    def gate():
        return ready.await_memory(available,timeout=min(300,owner['deadline']-time.monotonic()))
    try:
        if not child:
            require(all(not os.path.lexists(p) for p in (STAGE,PRIVATE/'attempt.local.json',
                    PRIVATE/'owner.local.json',PRIVATE/'closed.local.json',PRIVATE/'certificate.local.json')))
            if not preflight:
                attempt_sha = infra.write(PRIVATE/'attempt.local.json',marker('attempt',commit,commitment))
                owned = True
                owner_sha = infra.write(PRIVATE/'owner.local.json',owner)
            gate()
        else:
            check()
        modules = runtime(infra,bootstrap,reg)
        dm = modules['driver']
        if child:
            modules['fresh'].verify_live_owner(owner,commit=commit,binding_sha=commitment,
                current_pid=os.getpid(),process=psutil.Process,executable=sys.executable,
                samefile=os.path.samefile,now=time.monotonic())
            require(infra.decode(infra.read(PRIVATE/'attempt.local.json')) == marker('attempt',commit,commitment))
        context = source_context(infra,reg,modules,base,contract,binding,check,child=child)
        budget_check(dm,contract,reserve=contract.LIMITS['per_case_reserve_bytes'])
        if preflight:
            return dict(status='ready',source_games_authenticated=32,
                eligible_targets=sum(c['selection']['target'] is not None for c in binding['selections']),
                model_forwards=0,official_transitions=0)
        if child:
            matches = [c for c in binding['selections'] if c['source']['role'] == role]
            require(len(matches) == 1)
            case = matches[0]
            out = STAGE/role
            require(infra.read(STAGE/'binding.local.json') == raw
                and infra.decode(infra.read(out/'permit.local.json')) == marker('permit',commit,commitment,role))
            for kind in ('started','consumed'):
                infra.write(out/(kind+'.local.json'),marker(kind,commit,commitment,role),MAX_SMALL)
            argv = sys.argv
            try:
                sys.argv = [Path(__file__).name]
                with modules['worker']._quiet_output():
                    modules['window_worker'].execute_case(modules,context,case['source'],case['selection'],out,check)
            finally:
                sys.argv = argv
            require(check() and sha(infra.read(GAME/'main.py')) == binding['default_sha256'])
            identity.verify_public(repo=REPO,game_root=GAME,registration_commit=commit)
            return None
        STAGE.mkdir(exist_ok=False)
        infra.write(STAGE/'binding.local.json',binding)
        for case in binding['selections']:
            readiness = gate()
            require(check() and len(accepted) < 4
                and owner['deadline']-time.monotonic() > contract.LIMITS['job_wall_seconds']+contract.LIMITS['cleanup_seconds'])
            budget_check(dm,contract,reserve=contract.LIMITS['per_case_reserve_bytes'])
            current_role = case['source']['role']
            out = STAGE/current_role
            out.mkdir(exist_ok=False)
            readiness_sha = infra.write(out/'readiness.local.json',readiness,MAX_SMALL)
            infra.write(out/'permit.local.json',marker('permit',commit,commitment,current_role),MAX_SMALL)
            limits = contract.LIMITS
            command = [sys.executable,'-I','-B',str(Path(__file__).resolve()),
                       '--registration-commit',commit,'--case',current_role]
            supervised = infra.supervise_recorded(modules['campaign'],command,GAME,dict(
                max_rss_bytes=limits['max_rss_bytes'],max_commit_bytes=limits['max_commit_bytes'],
                min_available_bytes=limits['running_available_bytes'],wall_seconds=limits['job_wall_seconds'],
                cleanup_seconds=limits['cleanup_seconds'],poll_seconds=limits['poll_seconds']))
            supervisions.append(supervised)
            supervision_sha = infra.write(out/'supervision.local.json',dict(role=current_role,
                supervision=supervised,retry_allowed=False),MAX_SMALL)
            require(modules['acceptance'].supervision_ok(supervised) and check())
            checked = modules['window_worker'].verify_case(modules,context,case['source'],case['selection'],out,check)
            document = checked['document']
            markers = {k:infra.read(out/(k+'.local.json'),MAX_SMALL) for k in ('permit','started','consumed')}
            require(all(infra.decode(v) == marker(k,commit,commitment,current_role) for k,v in markers.items()))
            receipt = dict(schema='task-harvest-window-accepted-v1',role=current_role,
                registration_commit=commit,binding_sha256=commitment,passed=True,retry_allowed=False,
                case_receipt=checked['receipt'],supervision_sha256=supervision_sha,readiness_sha256=readiness_sha,
                markers={k:sha(v) for k,v in markers.items()})
            accepted.append(dict(role=current_role,sha256=infra.write(out/'accepted.local.json',receipt,MAX_SMALL)))
            documents.append(document)
            require(dm._owned_size(out) <= contract.LIMITS['per_case_output_bytes'])
            budget_check(dm,contract)
        require(len(documents) == 4 and check())
        meter = aggregate_meters(documents)
        counts = check_complete_counts(documents,meter,contract)
        rows = [d['row'] for d in documents]
        projection = contract.project(rows)
        again = source_context(infra,reg,modules,base,contract,binding,check)
        require(again.identities == context.identities)
        identity.verify_public(repo=REPO,game_root=GAME,registration_commit=commit)
        result_sha = infra.write(PRIVATE/'result.local.json',dict(schema='task-harvest-window-result-v1',
            registration_commit=commit,binding_sha256=commitment,accepted=accepted,rows=rows,
            projection=projection,meter=meter,counts=counts,training=False,gpu=False,new_seeds=False))
        budget_check(dm,contract)
        check()
    finally:
        if owned:
            error = sys.exc_info()[1]
            closed = dict(schema='task-harvest-window-closed-v1',registration_commit=commit,
                binding_sha256=commitment,outcome='data_collection_complete' if error is None else 'failed',
                error_kind=None if error is None else type(error).__name__,accepted_cases=len(accepted),
                owner_sha256=owner_sha,attempt_sha256=attempt_sha,result_sha256=result_sha,
                child_cleanup_receipts=supervisions,elapsed_seconds=time.monotonic()-started,
                retry_allowed=False,**infra.closing_memory(error,latest,available))
            close_preserving_primary(lambda value:infra.write(PRIVATE/'closed.local.json',value),closed,error)
    certificate = dict(schema='task-harvest-window-certificate-v1',registration_commit=commit,
        binding_sha256=commitment,closed_sha256=sha(infra.read(PRIVATE/'closed.local.json')),
        result_sha256=result_sha,accepted=accepted,all_attempted_cases_passed=True,
        official_continuation_tested=counts['paired_roles']>0,counts=counts,
        minimum_label_signal=projection['minimum_label_signal'],training=False,gpu=False,
        new_seeds=False,complete_games=0,terminal_profit_evaluated=False,promotion=False)
    infra.write(PRIVATE/'certificate.local.json',certificate)
    return dict(status='completed',all_attempted_cases_passed=True,
        official_continuation_tested=counts['paired_roles']>0,**counts,
        minimum_label_signal=projection['minimum_label_signal'],training=False,gpu=False)


class Parser(argparse.ArgumentParser):
    def error(self,_message):
        raise ValueError('Invalid command line')


def main(argv=None):
    parser = Parser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    for name in ('prepare','preflight','run'):
        modes.add_argument('--'+name,action='store_true')
    parser.add_argument('--registration-commit')
    parser.add_argument('--case')
    try:
        try:
            args = parser.parse_args(argv)
        except SystemExit as error:
            if error.code in (None,0):
                return 0
            raise ValueError('Invalid command line') from None
        require(not args.prepare or not (args.registration_commit or args.case))
        require(not args.case or not args.preflight)
        value = (prepare() if args.prepare else
            run(args.registration_commit,preflight=args.preflight,role=args.case))
        if value is not None:
            print(json.dumps(value,allow_nan=False))
        return 0
    except BaseException as error:
        category = ('memory_gate' if isinstance(error,MemoryError) else 'time_gate'
            if isinstance(error,TimeoutError) else 'io_error' if isinstance(error,OSError) else 'invalid_or_runtime')
        print(json.dumps(dict(status='failed',category=category,retry_allowed=False)),file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
