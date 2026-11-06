"""One diagnostic Job using frozen authentication, supervision and storage."""
from copy import deepcopy
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import secrets
import subprocess
import sys
import time
from types import ModuleType


REPO = Path(__file__).resolve().parents[3]
GAME = REPO.parent.parent/'kaggriculture'
BASE = '6d162b2652ae09e0ee739f64815820b741ee2ecf'
PRIOR_COMMIT = '02f8ef2f9691e5451dcf247f8714797a04d6dcc5'
PRIOR_REG = 'review/evidence/task-harvest-unexpressed-runtime-v2-20261106/registration.json'
PRIOR_RESULTS = 'review/evidence/task-harvest-unexpressed-runtime-v2-result-20261106'
PRIOR_PRIVATE = REPO.parent/'private/task-harvest-unexpressed-runtime-v2-20261106'
PRIOR_STAGE = GAME/'artifacts/task-harvest-unexpressed-runtime-v2-20261106'
PRIOR_WITNESS = REPO.parent/'private/task-harvest-unexpressed-runtime-v2-20261106-publication-witness.local.json'
FAILURE_COMMITMENT = '22e898d758631903bb581f4ef6af79bbedada3da758899ac88d95a7293afca80'
FROZEN_HELPERS = 'kaggriculture/learning/task_harvest_unexpressed_runtime_v2/runner.py'
PREFIX = 'kaggriculture/learning/task_harvest_first_step_diagnostic_v1/'
EVIDENCE = 'review/evidence/task-harvest-first-step-diagnostic-v1-20261106'
REGISTRATION, NOTE = EVIDENCE+'/registration.json', EVIDENCE+'/REGISTRATION.zh-CN.md'
PRIVATE = REPO.parent/'private/task-harvest-first-step-diagnostic-v1-20261106'
STAGE = GAME/'artifacts/task-harvest-first-step-diagnostic-v1-20261106'
DOC = 'task-harvest-first-step-diagnostic-v1-'
SCHEMA = DOC+'prereg'
PRIOR_DOC = 'task-harvest-unexpressed-runtime-v2-'


def require(value):
    if not value:
        raise ValueError('First-step diagnostic identity differs')


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def ordered(value):
    return json.dumps(value,separators=(',',':'),ensure_ascii=True,allow_nan=False).encode()


def git(*args):
    result = subprocess.run(['git','-C',str(REPO),*args],stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,stderr=subprocess.DEVNULL,timeout=45,
        creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    require(result.returncode == 0 and len(result.stdout) <= 16*1024**2)
    return result.stdout


def required_sources(infra):
    old = infra.decode(git('show',PRIOR_COMMIT+':'+PRIOR_REG))
    require(len(old['public_sources']) == 469)
    additions = {PRIOR_REG,NOTE,PREFIX+'README.zh-CN.md',FROZEN_HELPERS}
    additions.update(PRIOR_RESULTS+'/'+name for name in
        ('summary.json','certificate.json','RESULTS.zh-CN.md','REVIEW.zh-CN.md','NEXT.zh-CN.md'))
    paths = set(old['public_sources']) | additions
    paths.update(p.relative_to(REPO).as_posix() for p in (REPO/PREFIX).glob('*.py'))
    require({PREFIX+n for n in ('__init__.py','contract.py','worker.py','capture.py','runner.py')}.issubset(paths))
    for path,expected in old['public_sources'].items():
        raw = infra.read(REPO/path)
        if path == 'review/PROGRESS.md':
            original,published = git('show',PRIOR_COMMIT+':'+path),git('show',BASE+':'+path)
            require(sha(original) == expected and published.startswith(original) and raw == published)
        else:
            require(sha(raw) == expected)
    for path in additions-{NOTE,PREFIX+'README.zh-CN.md'}:
        require(infra.read(REPO/path) == git('show',BASE+':'+path))
    return paths


def marker(kind,commit,binding_sha,role=None):
    require(kind in ('attempt','permit','started','consumed'))
    value = dict(schema=DOC+'marker',kind=kind,registration_commit=commit,
                 binding_sha256=binding_sha,retry_allowed=False)
    if kind != 'attempt':
        require(role == 'F01')
        value['role'] = role
    return value


def _no_shadow():
    require(all(not os.path.lexists(GAME/('learning/'+name)) for name in
        ('task_harvest_unexpressed_v1','task_harvest_unexpressed_runtime_v2','task_harvest_first_step_diagnostic_v1')))


def _import(name):
    module = importlib.import_module('learning.'+name)
    require(Path(module.__file__).resolve(strict=True)
        == (REPO/'kaggriculture/learning'/(name.replace('.','/')+'.py')).resolve(strict=True))
    return module


def _legacy():
    """Use old functions in a new namespace with explicit source/path seams."""
    raw = (REPO/FROZEN_HELPERS).read_bytes()
    require(raw == git('show',BASE+':'+FROZEN_HELPERS))
    module = ModuleType('_first_step_diagnostic_helpers')
    module.__file__ = str(Path(__file__).resolve())
    exec(compile(raw,str(REPO/FROZEN_HELPERS),'exec',dont_inherit=True),module.__dict__)
    for key in ('REPO','GAME','BASE','PRIOR_COMMIT','PRIOR_REG','PRIOR_RESULTS','PRIOR_PRIVATE',
                'PRIOR_STAGE','PRIOR_WITNESS','FAILURE_COMMITMENT','PREFIX','EVIDENCE',
                'REGISTRATION','NOTE','PRIVATE','STAGE','DOC','SCHEMA'):
        setattr(module,key,globals()[key])
    module.required_sources,module.marker,module.binding_read = required_sources,marker,binding_read
    module._no_shadow,module.failed_source,module.inherited_points = _no_shadow,failed_source,inherited_points
    return module


def _shared(legacy=None):
    legacy = _legacy() if legacy is None else legacy
    shared = legacy._shared()
    def runtime(infra,bootstrap,reg):
        _no_shadow()
        modules = infra._runtime(bootstrap,reg)
        modules['window_worker'] = _import('task_harvest_first_step_diagnostic_v1.worker')
        maturity = _import('task_harvest_window_v1.selection')
        spec = importlib.util.find_spec('kaggle_environments')
        require(spec is not None and len(spec.submodule_search_locations) == 1)
        path = Path(next(iter(spec.submodule_search_locations)))/'envs/kaggriculture/kaggriculture.py'
        modules['first_yield_days'] = maturity.first_yield_days_from_source(infra.read(path))
        return modules
    shared.runtime = runtime
    return shared


def failure_paths():
    private = {name+'.local.json' for name in ('attempt','binding','closed','owner','selection-closed','selection-started')}
    private.update(role+'.selection.local.json' for role in ('F01','F02','F03','F04'))
    stage = {'binding.local.json','F01/events.jsonl.gz'}
    stage.update('F01/'+name+'.local.json' for name in ('case','consumed','permit','readiness','started','supervision'))
    return {**{'private/'+name:PRIOR_PRIVATE/name for name in private},
            **{'stage/'+name:PRIOR_STAGE/name for name in stage}}


def failed_source(infra,check):
    paths = failure_paths()
    witness_raw = infra.read(PRIOR_WITNESS)
    witness = infra.decode(witness_raw)
    certificate = infra.decode(infra.read(REPO/PRIOR_RESULTS/'certificate.json'))
    require(set(witness) == {'schema','registration_commit','files','salted_commitment_sha256'}
        and witness['schema'] == 'task-harvest-unexpressed-runtime-failure-witness-v2'
        and witness['registration_commit'] == PRIOR_COMMIT and set(witness['files']) == set(paths)
        and certificate['registration_commit'] == PRIOR_COMMIT and certificate['experiment_passed'] is False
        and certificate['originals']['files'] == len(paths) == 18
        and certificate['originals']['bytes'] == 187746
        and certificate['originals']['salted_commitment_sha256'] == witness['salted_commitment_sha256'] == FAILURE_COMMITMENT)
    originals = {}
    for key,path in sorted(paths.items()):
        check()
        item = witness['files'][key]
        require(type(item) is dict and set(item) == {'bytes','sha256'}
                and type(item['bytes']) is int and 0 <= item['bytes'] <= 1024**2)
        raw = infra.read(path,1024**2)
        require(len(raw) == item['bytes'] and sha(raw) == item['sha256'])
        originals[key] = raw
    require(sum(map(len,originals.values())) == 187746)
    binding_raw = originals['private/binding.local.json']
    binding = infra.decode(binding_raw)
    salt = binding['salt']
    require(type(salt) is str and len(salt) == 64 and all(c in '0123456789abcdef' for c in salt))
    commitment = sha(bytes.fromhex(salt)+b'\0task-harvest-unexpressed-runtime-failure-v2\0'+
        json.dumps(witness['files'],sort_keys=True,separators=(',',':'),ensure_ascii=True).encode())
    require(commitment == FAILURE_COMMITMENT)
    reg = infra.decode(infra.read(REPO/PRIOR_REG))
    closed = infra.decode(originals['private/closed.local.json'])
    selected = infra.decode(originals['private/selection-closed.local.json'])
    case = infra.decode(originals['stage/F01/case.local.json'])
    require(sha(binding_raw) == reg['binding_commitment_sha256']
        and binding.get('schema') == PRIOR_DOC+'binding' and binding.get('base_commit') == reg['base_commit']
        and closed.get('schema') == PRIOR_DOC+'closed' and closed.get('registration_commit') == PRIOR_COMMIT
        and closed.get('binding_sha256') == sha(binding_raw) and closed.get('outcome') == 'failed'
        and closed.get('accepted_cases') == 0 and closed.get('result_sha256') is None
        and closed.get('retry_allowed') is False
        and closed.get('owner_sha256') == sha(originals['private/owner.local.json'])
        and closed.get('attempt_sha256') == sha(originals['private/attempt.local.json'])
        and selected.get('schema') == PRIOR_DOC+'selection-closed' and selected.get('outcome') == 'prepared'
        and selected.get('error_kind') is None and selected.get('source_roles_copied') == 4
        and selected.get('binding_sha256') == sha(binding_raw) and selected.get('trace_rows_read') == 0
        and selected.get('target_reselection') is False
        and selected.get('model_forwards') == selected.get('official_transitions') == selected.get('replay_decodes') == 0
        and selected.get('selection_started_sha256') == binding['selection_started_sha256']
            == sha(originals['private/selection-started.local.json'])
        and originals['stage/binding.local.json'] == binding_raw
        and case.get('schema') == 'task-harvest-unexpressed-case-v1'
        and case.get('passed') is False and case.get('status') == 'failed' and case.get('error_kind') == 'invalid')
    calls,preparation = case['meter']['calls'],case['runtime_preparation']
    require(all(calls.get(k) == 1 for k in ('arm_attempts','restore_returns','official_step_attempts','official_step_returns'))
        and all(calls.get(k,0) == 0 for k in ('arm_returns','request_patch_attempts','request_patch_returns'))
        and calls.get('actor_returns') == calls.get('rival_returns') == 241
        and preparation.get('status') == 'prepared'
        and preparation.get('run_pair_attempts') == 1 and preparation.get('run_pair_returns') == 0)
    return dict(binding=binding,originals=originals,binding_sha256=sha(binding_raw),
        closed_sha256=sha(originals['private/closed.local.json']),witness_sha256=sha(witness_raw),
        salted_commitment_sha256=commitment,files=18,bytes=187746)


def inherited_points(previous,contract):
    old = previous['binding']['selections']
    require(type(old) is list and len(old) == 4 and [v['source']['role'] for v in old] == ['F01','F02','F03','F04'])
    case = deepcopy(old[0])
    require(ordered(case['source']) == ordered(contract.sources()[0])
        and previous['originals']['private/F01.selection.local.json'] == ordered(case)+b'\n')
    selection,target = case['selection'],case['selection']['target']
    require(selection.get('schema_version') == 'task-harvest-unexpressed-selection-v1'
        and selection['counts']['trace_lines'] == 719 and selection['reason'] is None
        and target['schema_version'] == 'task-harvest-unexpressed-target-v1'
        and type(target['step']) is int and target['step'] == contract.START_STEP
        and type(target['side']) is int and target['side'] == 0 and target['ordinal'] == 0)
    return [case]


def binding_read(infra,reg):
    raw = infra.read(PRIVATE/'binding.local.json')
    binding = infra.decode(raw)
    closed = infra.decode(infra.read(PRIVATE/'selection-closed.local.json'))
    require(sha(raw) == reg['binding_commitment_sha256'] and binding.get('schema') == DOC+'binding'
        and binding.get('base_commit') == BASE and binding.get('prior_registration_commit') == PRIOR_COMMIT
        and binding.get('prior_failure_commitment_sha256') == FAILURE_COMMITMENT
        and closed.get('schema') == DOC+'selection-closed' and closed.get('outcome') == 'prepared'
        and closed.get('error_kind') is None and closed.get('binding_sha256') == sha(raw)
        and closed.get('source_roles_copied') == 1 and closed.get('trace_rows_read') == 0
        and closed.get('model_forwards') == closed.get('official_transitions') == closed.get('replay_decodes') == 0
        and closed.get('target_reselection') is False and closed.get('retry_allowed') is False
        and binding['selection_started_sha256'] == closed['selection_started_sha256']
            == sha(infra.read(PRIVATE/'selection-started.local.json'))
        and len(binding['selections']) == 1 and binding['selections'][0]['source']['role'] == 'F01'
        and sha(ordered(binding['selections'])) == binding['inherited_selections_sha256']
        and infra.read(PRIVATE/'F01.selection.local.json') == ordered(binding['selections'][0])+b'\n')
    return raw,binding


def prepare():
    require(os.name == 'nt' and sys.flags.isolated and sys.flags.dont_write_bytecode and __debug__)
    legacy = _legacy();shared = _shared(legacy);infra = shared.infrastructure()
    require(git('rev-parse','HEAD').decode().strip() == BASE
        and all(not os.path.lexists(p) for p in (PRIVATE,STAGE,REPO/REGISTRATION)))
    source_map = {p:sha(infra.read(REPO/p)) for p in sorted(required_sources(infra))}
    contract = infra.snapshot(PREFIX+'contract.py',source_map[PREFIX+'contract.py'])
    base = infra.snapshot(infra.BASE_ENTRY,source_map[infra.BASE_ENTRY])
    ready = infra.snapshot(shared.READY,source_map[shared.READY])
    deadline = time.monotonic()+900
    def check():
        if base.physical_available() < contract.LIMITS['running_available_bytes']:
            raise MemoryError('Preparation memory floor')
        if time.monotonic() >= deadline:
            raise TimeoutError('Preparation deadline')
        return True
    ready.await_memory(base.physical_available)
    identity = infra.snapshot(infra.IDENTITY,source_map[infra.IDENTITY])
    infra.preparation_runtime(identity,BASE)
    modules = legacy.metadata_modules()
    PRIVATE.mkdir(exist_ok=False)
    start_sha = infra.write(PRIVATE/'selection-started.local.json',dict(schema=DOC+'selection-started',
        base_commit=BASE,public_sources=source_map,specification=contract.specification(),
        prior_registration_commit=PRIOR_COMMIT,target_reselection=False,
        local_registration_not_third_party_timestamp=True))
    binding_sha,selected = None,[]
    try:
        previous,identities,selected = legacy.prepare_payload(infra,modules,base,contract,check)
        require(len(selected) == 1)
        written = infra.write(PRIVATE/'F01.selection.local.json',selected[0],shared.MAX_SMALL)
        require(written == sha(previous['originals']['private/F01.selection.local.json']))
        require(check() and all(sha(infra.read(REPO/p)) == h for p,h in source_map.items()))
        binding = dict(schema=DOC+'binding',base_commit=BASE,salt=secrets.token_hex(32),
            selection_started_sha256=start_sha,completed_sources=identities,
            prior_registration_commit=PRIOR_COMMIT,prior_binding_sha256=previous['binding_sha256'],
            prior_failed_closed_sha256=previous['closed_sha256'],prior_failure_witness_sha256=previous['witness_sha256'],
            prior_failure_commitment_sha256=previous['salted_commitment_sha256'],prior_failed_original_files=18,
            prior_failed_original_bytes=187746,default_sha256=previous['binding']['default_sha256'],selections=selected,
            inherited_selections_sha256=sha(ordered(selected)),target_reselection=False,
            historical_game_files_rehashed_during_preparation=False)
        binding_sha = infra.write(PRIVATE/'binding.local.json',binding)
        reg = dict(schema=SCHEMA,base_commit=BASE,engine_commit=contract.ENGINE_COMMIT,
            engine_sha256=contract.ENGINE_SHA256,public_sources=source_map,versions=infra._versions(),
            limits=contract.LIMITS,panel=contract.specification(),evaluation=dict(training=False,gpu=False,
                new_seeds=False,independent_evaluation=False,statistical_significance=False,
                promotion=False,terminal_profit=False,retries=False),binding_commitment_sha256=binding_sha)
        shared.validate_registration(reg,contract)
        (REPO/EVIDENCE).mkdir(parents=True,exist_ok=True)
        infra.write(REPO/REGISTRATION,reg)
    finally:
        error = sys.exc_info()[1]
        shared.close_preserving_primary(lambda value:infra.write(PRIVATE/'selection-closed.local.json',value),
            dict(schema=DOC+'selection-closed',outcome='prepared' if error is None else 'failed',
                error_kind=None if error is None else type(error).__name__,binding_sha256=binding_sha,
                selection_started_sha256=start_sha,source_roles_copied=len(selected),trace_rows_read=0,
                model_forwards=0,official_transitions=0,replay_decodes=0,target_reselection=False,retry_allowed=False),error)
    return dict(status='prepared_one_point_diagnostic',copied_roles=1,authenticated_failed_originals=18,
                completed_source_receipts=32,historical_game_files_rehashed=False,trace_rows_read=0,
                target_reselection=False,model_forwards=0,official_transitions=0,replay_decodes=0)


def execute_and_verify(worker,modules,context,case,out,check,contract):
    """Exit zero only for verified diagnosis; the original failed case stays so."""
    try:
        worker.execute_case(modules,context,case['source'],case['selection'],out,check)
    except BaseException as error:
        if isinstance(error,(MemoryError,TimeoutError,OSError,KeyboardInterrupt,SystemExit)):
            raise
        try:
            check()
            checked = worker.verify_case(modules,context,case['source'],case['selection'],out,check)
            contract.accept_diagnostic(checked)
        except BaseException as secondary:
            raise error from secondary
        return checked
    raise ValueError('Diagnostic core returned without its required stop')


def run(commit,*,preflight=False,role=None):
    require(role is None or role == 'F01')
    shared = _shared()
    infra,reg,contract,identity,bootstrap,base = shared.authenticate(commit)
    raw,binding = binding_read(infra,reg)
    commitment,child = sha(raw),role is not None
    require(not(child and preflight))
    started = time.monotonic()
    latest = dict(available_bytes=None,sample_monotonic=None)
    def available():
        value = base.physical_available()
        latest.update(available_bytes=value,sample_monotonic=time.monotonic())
        return value
    import psutil
    owner = infra.decode(infra.read(PRIVATE/'owner.local.json')) if child else dict(
        schema='task-pair-fresh-multi-owner-v1',registration_commit=commit,binding_sha256=commitment,
        pid=os.getpid(),created_us=int(psutil.Process().create_time()*1_000_000),
        deadline=started+contract.LIMITS['total_wall_seconds'])
    owner_sha = sha(infra.read(PRIVATE/'owner.local.json')) if child else None
    attempt_sha,result_sha,accepted_sha = None,None,None
    owned,accepted,supervisions = False,[],[]
    def check():
        if available() < contract.LIMITS['running_available_bytes']:
            raise MemoryError('Runtime memory floor')
        if time.monotonic() >= owner['deadline']:
            raise TimeoutError('Diagnostic deadline')
        require(infra.read(PRIVATE/'binding.local.json') == raw and not os.path.lexists(PRIVATE/'closed.local.json'))
        if owner_sha is not None:
            require(sha(infra.read(PRIVATE/'owner.local.json')) == owner_sha)
        return True
    ready = infra.snapshot(shared.READY,reg['public_sources'][shared.READY],commit)
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
        modules = shared.runtime(infra,bootstrap,reg)
        dm = modules['driver']
        if child:
            modules['fresh'].verify_live_owner(owner,commit=commit,binding_sha=commitment,current_pid=os.getpid(),
                process=psutil.Process,executable=sys.executable,samefile=os.path.samefile,now=time.monotonic())
            require(infra.decode(infra.read(PRIVATE/'attempt.local.json')) == marker('attempt',commit,commitment))
        context = shared.source_context(infra,reg,modules,base,contract,binding,check,child=child)
        shared.budget_check(dm,contract,reserve=0 if child else contract.LIMITS['per_case_reserve_bytes'])
        if preflight:
            return dict(status='ready_one_point_diagnostic',source_games_authenticated=32,
                        cases=1,model_forwards=0,official_transitions=0)
        case = binding['selections'][0]
        require(case['source']['role'] == 'F01' and case['selection']['target']['step'] == 240)
        out = STAGE/'F01'
        if child:
            require(infra.read(STAGE/'binding.local.json') == raw
                and infra.decode(infra.read(out/'permit.local.json')) == marker('permit',commit,commitment,'F01'))
            for kind in ('started','consumed'):
                infra.write(out/(kind+'.local.json'),marker(kind,commit,commitment,'F01'),shared.MAX_SMALL)
            argv = sys.argv
            try:
                sys.argv = [Path(__file__).name]
                with modules['worker']._quiet_output():
                    execute_and_verify(modules['window_worker'],modules,context,case,out,check,contract)
            finally:
                sys.argv = argv
            require(check() and sha(infra.read(GAME/'main.py')) == binding['default_sha256'])
            identity.verify_public(repo=REPO,game_root=GAME,registration_commit=commit)
            return None
        STAGE.mkdir(exist_ok=False)
        infra.write(STAGE/'binding.local.json',binding)
        readiness = gate()
        require(check() and owner['deadline']-time.monotonic() > contract.LIMITS['job_wall_seconds']+contract.LIMITS['cleanup_seconds'])
        # The stage binding occupies bytes in the same single-case total cap.
        shared.budget_check(dm,contract)
        out.mkdir(exist_ok=False)
        readiness_sha = infra.write(out/'readiness.local.json',readiness,shared.MAX_SMALL)
        infra.write(out/'permit.local.json',marker('permit',commit,commitment,'F01'),shared.MAX_SMALL)
        limits = contract.LIMITS
        command = [sys.executable,'-I','-B',str(Path(__file__).resolve()),'--registration-commit',commit,'--case','F01']
        supervised = infra.supervise_recorded(modules['campaign'],command,GAME,dict(
            max_rss_bytes=limits['max_rss_bytes'],max_commit_bytes=limits['max_commit_bytes'],
            min_available_bytes=limits['running_available_bytes'],wall_seconds=limits['job_wall_seconds'],
            cleanup_seconds=limits['cleanup_seconds'],poll_seconds=limits['poll_seconds']))
        supervisions.append(supervised)
        supervision_sha = infra.write(out/'supervision.local.json',dict(role='F01',supervision=supervised,retry_allowed=False),shared.MAX_SMALL)
        require(modules['acceptance'].supervision_ok(supervised) and check())
        checked = modules['window_worker'].verify_case(modules,context,case['source'],case['selection'],out,check)
        diagnostic = contract.accept_diagnostic(checked)
        markers = {k:infra.read(out/(k+'.local.json'),shared.MAX_SMALL) for k in ('permit','started','consumed')}
        require(all(infra.decode(v) == marker(k,commit,commitment,'F01') for k,v in markers.items()))
        accepted_sha = infra.write(out/'accepted.local.json',dict(schema=DOC+'accepted',role='F01',
            registration_commit=commit,binding_sha256=commitment,diagnostic_passed=True,original_case_passed=False,
            accepted_pairs=0,retry_allowed=False,case_receipt=checked['receipt'],diagnostic=diagnostic,
            supervision_sha256=supervision_sha,readiness_sha256=readiness_sha,markers={k:sha(v) for k,v in markers.items()}),shared.MAX_SMALL)
        accepted.append(dict(role='F01',sha256=accepted_sha))
        require(dm._owned_size(out) <= contract.LIMITS['per_case_output_bytes'])
        shared.budget_check(dm,contract)
        again = shared.source_context(infra,reg,modules,base,contract,binding,check)
        require(again.identities == context.identities)
        identity.verify_public(repo=REPO,game_root=GAME,registration_commit=commit)
        result_sha = infra.write(PRIVATE/'result.local.json',dict(schema=DOC+'result',registration_commit=commit,
            binding_sha256=commitment,accepted=accepted,diagnostic=diagnostic,meter=checked['document']['meter'],
            original_case_passed=False,accepted_pairs=0,full_horizon_verified=False,economic_value_evaluated=False,
            training=False,gpu=False,new_seeds=False))
        shared.budget_check(dm,contract)
        check()
    finally:
        if owned:
            error = sys.exc_info()[1]
            shared.close_preserving_primary(lambda value:infra.write(PRIVATE/'closed.local.json',value),
                dict(schema=DOC+'closed',registration_commit=commit,binding_sha256=commitment,
                    outcome='diagnostic_complete' if error is None else 'failed',
                    error_kind=None if error is None else type(error).__name__,accepted_diagnostics=len(accepted),accepted_pairs=0,
                    owner_sha256=owner_sha,attempt_sha256=attempt_sha,result_sha256=result_sha,
                    child_cleanup_receipts=supervisions,elapsed_seconds=time.monotonic()-started,
                    retry_allowed=False,**infra.closing_memory(error,latest,available)),error)
    infra.write(PRIVATE/'certificate.local.json',dict(schema=DOC+'certificate',registration_commit=commit,
        binding_sha256=commitment,closed_sha256=sha(infra.read(PRIVATE/'closed.local.json')),
        result_sha256=result_sha,accepted=accepted,diagnostic_passed=True,diagnostic=diagnostic,
        original_case_passed=False,accepted_pairs=0,full_horizon_verified=False,economic_value_evaluated=False,
        training=False,gpu=False,new_seeds=False,complete_games=0,promotion=False))
    return dict(status='diagnostic_complete',diagnostic_outcome=diagnostic['outcome'],
                original_case_passed=False,accepted_pairs=0,full_horizon_verified=False,training=False,gpu=False)


def main(argv=None):
    shared = _shared()
    shared.prepare,shared.run = prepare,run
    return shared.main(argv)


if __name__ == '__main__':
    raise SystemExit(main())
