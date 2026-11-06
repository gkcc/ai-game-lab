"""Reuse the frozen Job coordinator, inheriting points without selecting again."""
from copy import deepcopy
import hashlib
import importlib
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
BASE = '7f2deadcb1e3dc65d3073081d64177eee4354440'
PRIOR_COMMIT = '7bdf591cbdb4070be69f7a82d43e8202dadb661c'
PRIOR_REG = 'review/evidence/task-harvest-unexpressed-v1-20261106/registration.json'
PRIOR_RESULTS = 'review/evidence/task-harvest-unexpressed-v1-result-20261106'
PRIOR_PRIVATE = REPO.parent/'private/task-harvest-unexpressed-v1-20261106'
PRIOR_STAGE = GAME/'artifacts/task-harvest-unexpressed-v1-20261106'
PRIOR_WITNESS = REPO.parent/'private/task-harvest-unexpressed-v1-20261106-publication-witness.local.json'
FAILURE_COMMITMENT = 'e8882e77576eb946eece64444cb00fb2cc3800b47be8fcb67a3ace56e6b6d627'
WRAPPER = 'kaggriculture/learning/task_harvest_unexpressed_v1/runner.py'
SHARED = 'kaggriculture/learning/task_harvest_window_v1/runner.py'
PREFIX = 'kaggriculture/learning/task_harvest_unexpressed_runtime_v2/'
EVIDENCE = 'review/evidence/task-harvest-unexpressed-runtime-v2-20261106'
REGISTRATION, NOTE = EVIDENCE+'/registration.json', EVIDENCE+'/REGISTRATION.zh-CN.md'
PRIVATE = REPO.parent/'private/task-harvest-unexpressed-runtime-v2-20261106'
STAGE = GAME/'artifacts/task-harvest-unexpressed-runtime-v2-20261106'
DOC = 'task-harvest-unexpressed-runtime-v2-'
SCHEMA = DOC+'prereg'


def require(value):
    if not value:
        raise ValueError('Runtime reproduction identity differs')


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def ordered(value):
    return json.dumps(value, separators=(',', ':'), ensure_ascii=True, allow_nan=False).encode()


def git(*args):
    result = subprocess.run(['git','-C',str(REPO),*args],stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,stderr=subprocess.DEVNULL,timeout=45,
        creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    require(result.returncode == 0 and len(result.stdout) <= 16*1024**2)
    return result.stdout


def required_sources(infra):
    old = infra.decode(git('show',PRIOR_COMMIT+':'+PRIOR_REG))
    require(len(old['public_sources']) == 454)
    additions = {PRIOR_REG,NOTE,PREFIX+'README.zh-CN.md',WRAPPER,SHARED}
    additions.update(PRIOR_RESULTS+'/'+name for name in
        ('summary.json','certificate.json','RESULTS.zh-CN.md','REVIEW.zh-CN.md','NEXT.zh-CN.md'))
    paths = set(old['public_sources']) | additions
    paths.update(p.relative_to(REPO).as_posix() for p in (REPO/PREFIX).glob('*.py'))
    require({PREFIX+n for n in ('__init__.py','contract.py','worker.py','runner.py')}.issubset(paths))
    for path,expected in old['public_sources'].items():
        raw = infra.read(REPO/path)
        if path == 'review/PROGRESS.md':
            original, published = git('show',PRIOR_COMMIT+':'+path), git('show',BASE+':'+path)
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
        require(role in ('F01','F02','F03','F04'))
        value['role'] = role
    return value


def _no_shadow():
    require(all(not os.path.lexists(GAME/('learning/'+name)) for name in
        ('task_harvest_unexpressed_v1','task_harvest_unexpressed_runtime_v2')))


def _import(name):
    module = importlib.import_module('learning.'+name)
    require(Path(module.__file__).resolve(strict=True)
            == (REPO/'kaggriculture/learning'/(name.replace('.','/')+'.py')).resolve(strict=True))
    return module


def metadata_modules():
    _no_shadow()
    sys.path.insert(0,str(REPO/'kaggriculture'))
    modules = {key:_import(name) for key,name in (
        ('sources','task_teacher_fresh_panel_v1.sources'),
        ('driver','task_pair_dev_campaign_v1.driver'))}
    no_execution_imports()
    return modules


def no_execution_imports():
    require(not any(n == prefix or n.startswith(prefix+'.')
        for prefix in ('kaggle_environments','torch') for n in sys.modules))


def schema_document(value):
    if isinstance(value,dict) and isinstance(value.get('schema'),str) and not value['schema'].startswith(DOC):
        for prefix in ('task-harvest-window-','task-harvest-unexpressed-'):
            if value['schema'].startswith(prefix):
                tail = value['schema'][len(prefix):]
                return dict(value,schema=DOC+(tail[:-3] if tail.endswith('-v1') else tail))
    return value


def _shared():
    """Isolate the published wrapper and coordinator; never patch shared imports."""
    raw = (REPO/WRAPPER).read_bytes()
    require(raw == git('show',BASE+':'+WRAPPER))
    wrapper = ModuleType('_runtime_v2_frozen_wrapper')
    wrapper.__file__ = str(Path(__file__).resolve())
    exec(compile(raw,str(REPO/WRAPPER),'exec',dont_inherit=True),wrapper.__dict__)
    for key in ('REPO','GAME','BASE','PRIOR_COMMIT','PRIOR_REG','PRIOR_RESULTS','PREFIX',
                'EVIDENCE','REGISTRATION','NOTE','PRIVATE','STAGE','SCHEMA'):
        setattr(wrapper,key,globals()[key])
    wrapper.required_sources, wrapper.marker = required_sources, marker
    wrapper.binding_read, wrapper._no_shadow = binding_read, _no_shadow
    inherited = wrapper._shared()
    original_infrastructure = inherited.infrastructure
    def infrastructure():
        infra = original_infrastructure()
        original_write = infra.write
        def write(path,value,*args,**kwargs):
            return original_write(path,schema_document(value),*args,**kwargs)
        infra.write = write
        return infra
    inherited.infrastructure = infrastructure
    def runtime(infra,bootstrap,reg):
        _no_shadow()
        modules = infra._runtime(bootstrap,reg)
        modules['window_worker'] = _import('task_harvest_unexpressed_runtime_v2.worker')
        modules['first_yield_days'] = wrapper.first_days(infra,_import('task_harvest_window_v1.selection'))
        return modules
    inherited.runtime = runtime
    return inherited


def failure_paths():
    private = {name+'.local.json' for name in ('attempt','binding','closed','owner','selection-closed','selection-started')}
    private.update(role+'.selection.local.json' for role in ('F01','F02','F03','F04'))
    stage = {'binding.local.json','F01/events.jsonl.gz'}
    stage.update('F01/'+name+'.local.json' for name in ('case','consumed','permit','readiness','started','supervision'))
    return {**{'private/'+name:PRIOR_PRIVATE/name for name in private},
            **{'stage/'+name:PRIOR_STAGE/name for name in stage}}


def failed_source(infra,check):
    """Read only the exact 18 failed originals and their external salt witness."""
    paths = failure_paths()
    witness_raw = infra.read(PRIOR_WITNESS)
    witness = infra.decode(witness_raw)
    certificate = infra.decode(infra.read(REPO/PRIOR_RESULTS/'certificate.json'))
    require(set(witness) == {'schema','registration_commit','files','salted_commitment_sha256'}
        and witness['schema'] == 'task-harvest-unexpressed-failure-witness-v1'
        and witness['registration_commit'] == PRIOR_COMMIT and type(witness['files']) is dict
        and set(witness['files']) == set(paths) and len(paths) == 18
        and certificate['registration_commit'] == PRIOR_COMMIT and certificate['experiment_passed'] is False
        and certificate['originals']['files'] == 18
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
    require(sum(map(len,originals.values())) == certificate['originals']['bytes'])
    binding_raw = originals['private/binding.local.json']
    binding = infra.decode(binding_raw)
    salt = binding['salt']
    require(type(salt) is str and len(salt) == 64 and all(c in '0123456789abcdef' for c in salt))
    commitment = sha(bytes.fromhex(salt)+b'\0task-harvest-unexpressed-failure-v1\0'+
        json.dumps(witness['files'],sort_keys=True,separators=(',',':'),ensure_ascii=True).encode())
    require(commitment == FAILURE_COMMITMENT)
    reg = infra.decode(infra.read(REPO/PRIOR_REG))
    closed = infra.decode(originals['private/closed.local.json'])
    selected = infra.decode(originals['private/selection-closed.local.json'])
    require(sha(binding_raw) == reg['binding_commitment_sha256']
        and binding.get('schema') == 'task-harvest-unexpressed-binding-v1'
        and binding.get('base_commit') == reg['base_commit']
        and closed.get('schema') == 'task-harvest-unexpressed-closed-v1'
        and closed.get('registration_commit') == PRIOR_COMMIT
        and closed.get('binding_sha256') == sha(binding_raw)
        and closed.get('outcome') == 'failed' and closed.get('accepted_cases') == 0
        and closed.get('result_sha256') is None and closed.get('retry_allowed') is False
        and closed.get('owner_sha256') == sha(originals['private/owner.local.json'])
        and closed.get('attempt_sha256') == sha(originals['private/attempt.local.json'])
        and selected.get('schema') == 'task-harvest-unexpressed-selection-closed-v1'
        and selected.get('outcome') == 'prepared' and selected.get('error_kind') is None
        and selected.get('binding_sha256') == sha(binding_raw) and selected.get('source_roles_scanned') == 4
        and selected.get('model_forwards') == selected.get('official_transitions') == 0
        and selected.get('selection_started_sha256') == binding['selection_started_sha256']
            == sha(originals['private/selection-started.local.json'])
        and originals['stage/binding.local.json'] == binding_raw)
    case = infra.decode(originals['stage/F01/case.local.json'])
    require(case.get('passed') is False and case.get('status') == 'failed' and case.get('retry_allowed') is False)
    return dict(binding=binding,originals=originals,binding_sha256=sha(binding_raw),
        closed_sha256=sha(originals['private/closed.local.json']),witness_sha256=sha(witness_raw),
        salted_commitment_sha256=commitment,files=18,bytes=sum(map(len,originals.values())))


def inherited_points(previous,contract):
    """No selector dependency: preserve original nested JSON order and values."""
    selected = previous['binding']['selections']
    require(type(selected) is list and len(selected) == 4
        and ordered([v['source'] for v in selected]) == ordered(contract.sources()))
    result = deepcopy(selected)
    for case in result:
        source, selection = case['source'], case['selection']
        target = selection['target']
        require(selection.get('schema_version') == 'task-harvest-unexpressed-selection-v1'
            and selection['counts']['trace_lines'] == 719 and selection['reason'] is None
            and type(target) is dict and type(target.get('side')) is int and target['side'] == source['side']
            and type(target.get('step')) is int and contract.WINDOW[0] <= target['step'] <= contract.WINDOW[1]
            and target.get('schema_version') == 'task-harvest-unexpressed-target-v1'
            and type(target.get('ordinal')) is int and target['ordinal'] == 0)
        raw = previous['originals']['private/'+source['role']+'.selection.local.json']
        require(raw == ordered(case)+b'\n')
    require(ordered(result) == ordered(selected))
    return result


def prepare_payload(infra,modules,base,contract,check):
    """The whole metadata operation, injectable for a no-file regression."""
    previous = failed_source(infra,check)
    context = modules['sources'].load(guard=check,verify_game_files=False)
    require(context.identities == previous['binding']['completed_sources']
        and all(base.validate_assets(context.manifests[c],modules['driver']) for c in ('C01','C02'))
        and sha(infra.read(GAME/'main.py')) == previous['binding']['default_sha256'])
    selected = inherited_points(previous,contract)
    no_execution_imports()
    check()
    return previous,context.identities,selected


def binding_read(infra,reg):
    raw = infra.read(PRIVATE/'binding.local.json')
    binding = infra.decode(raw)
    closed = infra.decode(infra.read(PRIVATE/'selection-closed.local.json'))
    require(sha(raw) == reg['binding_commitment_sha256'] and binding.get('schema') == DOC+'binding'
        and binding.get('base_commit') == BASE and binding.get('prior_registration_commit') == PRIOR_COMMIT
        and binding.get('prior_failure_commitment_sha256') == FAILURE_COMMITMENT
        and closed.get('schema') == DOC+'selection-closed' and closed.get('outcome') == 'prepared'
        and closed.get('error_kind') is None and closed.get('binding_sha256') == sha(raw)
        and closed.get('source_roles_copied') == 4 and closed.get('trace_rows_read') == 0
        and closed.get('model_forwards') == closed.get('official_transitions') == closed.get('replay_decodes') == 0
        and closed.get('target_reselection') is False and closed.get('retry_allowed') is False
        and binding['selection_started_sha256'] == closed['selection_started_sha256']
            == sha(infra.read(PRIVATE/'selection-started.local.json')))
    require(sha(ordered(binding['selections'])) == binding['inherited_selections_sha256'])
    for case in binding['selections']:
        require(infra.read(PRIVATE/(case['source']['role']+'.selection.local.json')) == ordered(case)+b'\n')
    return raw,binding


def prepare(shared=None):
    require(os.name == 'nt' and sys.flags.isolated and sys.flags.dont_write_bytecode and __debug__)
    shared = _shared() if shared is None else shared
    infra = shared.infrastructure()
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
    modules = metadata_modules()
    PRIVATE.mkdir(exist_ok=False)
    start_sha = infra.write(PRIVATE/'selection-started.local.json',dict(schema=DOC+'selection-started',
        base_commit=BASE,public_sources=source_map,specification=contract.specification(),
        prior_registration_commit=PRIOR_COMMIT,target_reselection=False,
        local_registration_not_third_party_timestamp=True))
    binding_sha,selected = None,[]
    try:
        previous,identities,selected = prepare_payload(infra,modules,base,contract,check)
        for case in selected:
            check()
            path = PRIVATE/(case['source']['role']+'.selection.local.json')
            written = infra.write(path,case,shared.MAX_SMALL)
            require(written == sha(previous['originals']['private/'+path.name]))
        require(check() and all(sha(infra.read(REPO/p)) == h for p,h in source_map.items()))
        binding = dict(schema=DOC+'binding',base_commit=BASE,salt=secrets.token_hex(32),
            selection_started_sha256=start_sha,completed_sources=identities,
            prior_registration_commit=PRIOR_COMMIT,prior_binding_sha256=previous['binding_sha256'],
            prior_failed_closed_sha256=previous['closed_sha256'],prior_failure_witness_sha256=previous['witness_sha256'],
            prior_failure_commitment_sha256=previous['salted_commitment_sha256'],
            prior_failed_original_files=previous['files'],prior_failed_original_bytes=previous['bytes'],
            default_sha256=previous['binding']['default_sha256'],selections=selected,
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
    return dict(status='prepared_frozen_point_reproduction',copied_roles=4,eligible_targets=4,
                authenticated_failed_originals=18,completed_source_receipts=32,
                historical_game_files_rehashed=False,trace_rows_read=0,target_reselection=False,
                model_forwards=0,official_transitions=0,replay_decodes=0)


def main(argv=None):
    shared = _shared()
    shared.prepare = lambda:prepare(shared)
    return shared.main(argv)


if __name__ == '__main__':
    raise SystemExit(main())
