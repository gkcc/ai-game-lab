"""Four frozen points using the published owned coordinator and strict worker."""
import hashlib
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
BASE = '42e78af6f839fc6ebf694166707c43f3005928ac'
PRIOR_COMMIT = '02f8ef2f9691e5451dcf247f8714797a04d6dcc5'
PRIOR_REG = 'review/evidence/task-harvest-unexpressed-runtime-v2-20261106/registration.json'
PRIOR_RESULTS = 'review/evidence/task-harvest-unexpressed-runtime-v2-result-20261106'
PRIOR_PRIVATE = REPO.parent/'private/task-harvest-unexpressed-runtime-v2-20261106'
PRIOR_STAGE = GAME/'artifacts/task-harvest-unexpressed-runtime-v2-20261106'
PRIOR_WITNESS = REPO.parent/'private/task-harvest-unexpressed-runtime-v2-20261106-publication-witness.local.json'
FAILURE_COMMITMENT = '22e898d758631903bb581f4ef6af79bbedada3da758899ac88d95a7293afca80'
DIAGNOSTIC_COMMIT = 'c024d76c64903405905ebe3f8510b88ed42406a0'
DIAGNOSTIC_REG = 'review/evidence/task-harvest-first-step-diagnostic-v1-20261106/registration.json'
DIAGNOSTIC_RESULTS = 'review/evidence/task-harvest-first-step-diagnostic-v1-result-20261106'
DIAGNOSTIC_PRIVATE = REPO.parent/'private/task-harvest-first-step-diagnostic-v1-20261106'
DIAGNOSTIC_STAGE = GAME/'artifacts/task-harvest-first-step-diagnostic-v1-20261106'
DIAGNOSTIC_WITNESS = REPO.parent/'private/task-harvest-first-step-diagnostic-v1-20261106-publication-witness.local.json'
DIAGNOSTIC_COMMITMENT = '7709c26989718b2229439710793cf8c1f4c27c46454fa05774a2df87bcc51a70'
DIAGNOSTIC_DOC = 'task-harvest-first-step-diagnostic-v1-'
FROZEN_HELPERS = 'kaggriculture/learning/task_harvest_unexpressed_runtime_v2/runner.py'
DIAGNOSTIC_HELPERS = 'kaggriculture/learning/task_harvest_first_step_diagnostic_v1/runner.py'
PREFIX = 'kaggriculture/learning/task_harvest_struct_compat_v1/'
EVIDENCE = 'review/evidence/task-harvest-struct-compat-v1-20261106'
REGISTRATION, NOTE = EVIDENCE+'/registration.json', EVIDENCE+'/REGISTRATION.zh-CN.md'
PRIVATE = REPO.parent/'private/task-harvest-struct-compat-v1-20261106'
STAGE = GAME/'artifacts/task-harvest-struct-compat-v1-20261106'
DOC, SCHEMA = 'task-harvest-struct-compat-v1-', 'task-harvest-struct-compat-v1-prereg'


def require(value):
    if not value:
        raise ValueError('Official-container reproduction identity differs')


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
    old = infra.decode(git('show',DIAGNOSTIC_COMMIT+':'+DIAGNOSTIC_REG))
    require(len(old['public_sources']) == 484)
    additions = {DIAGNOSTIC_REG,NOTE,PREFIX+'README.zh-CN.md',FROZEN_HELPERS,DIAGNOSTIC_HELPERS}
    additions.update(DIAGNOSTIC_RESULTS+'/'+name for name in
        ('summary.json','certificate.json','RESULTS.zh-CN.md','REVIEW.zh-CN.md','NEXT.zh-CN.md'))
    paths = set(old['public_sources']) | additions
    paths.update(p.relative_to(REPO).as_posix() for p in (REPO/PREFIX).glob('*.py'))
    require({PREFIX+n for n in ('__init__.py','contract.py','compat.py','worker.py','runner.py')}.issubset(paths))
    for path,expected in old['public_sources'].items():
        raw = infra.read(REPO/path)
        if path == 'review/PROGRESS.md':
            original,published = git('show',DIAGNOSTIC_COMMIT+':'+path),git('show',BASE+':'+path)
            require(sha(original) == expected and published.startswith(original) and raw == published)
        else:
            require(sha(raw) == expected)
    for path in additions-{NOTE,PREFIX+'README.zh-CN.md'}:
        require(infra.read(REPO/path) == git('show',BASE+':'+path))
    return paths


def _clone(path,name):
    raw = (REPO/path).read_bytes()
    require(raw == git('show',BASE+':'+path))
    module = ModuleType(name)
    module.__file__ = str(Path(__file__).resolve())
    exec(compile(raw,str(REPO/path),'exec',dont_inherit=True),module.__dict__)
    return module


def _no_shadow():
    require(all(not os.path.lexists(GAME/('learning/'+name)) for name in (
        'task_harvest_unexpressed_v1','task_harvest_unexpressed_runtime_v2',
        'task_harvest_first_step_diagnostic_v1','task_harvest_struct_compat_v1')))


def _legacy():
    """Rebind only the published coordination seams in a private namespace."""
    module = _clone(FROZEN_HELPERS,'_struct_compat_frozen_four_point_helpers')
    for key in ('REPO','GAME','BASE','PRIOR_COMMIT','PRIOR_REG','PRIOR_RESULTS','PRIOR_PRIVATE',
                'PRIOR_STAGE','PRIOR_WITNESS','FAILURE_COMMITMENT','PREFIX','EVIDENCE',
                'REGISTRATION','NOTE','PRIVATE','STAGE','DOC','SCHEMA'):
        setattr(module,key,globals()[key])
    module.original_binding_read = module.binding_read
    module.required_sources,module.binding_read,module._no_shadow = required_sources,binding_read,_no_shadow
    return module


def utils_identity(infra,contract):
    """Locate the installed package without importing its runtime."""
    spec = importlib.util.find_spec('kaggle_environments')
    require(spec is not None and len(spec.submodule_search_locations) == 1)
    package = Path(next(iter(spec.submodule_search_locations)))
    require(sha(infra.read(package/'utils.py')) == contract.STRUCT_UTILS_SHA256)
    return contract.STRUCT_UTILS_SHA256


def _shared(legacy=None):
    legacy = _legacy() if legacy is None else legacy
    shared = legacy._shared()
    def runtime(infra,bootstrap,reg):
        _no_shadow()
        contract = infra.snapshot(PREFIX+'contract.py',reg['public_sources'][PREFIX+'contract.py'])
        utils_identity(infra,contract)
        modules = infra._runtime(bootstrap,reg)
        modules['window_worker'] = legacy._import('task_harvest_struct_compat_v1.worker')
        maturity = legacy._import('task_harvest_window_v1.selection')
        spec = importlib.util.find_spec('kaggle_environments')
        require(spec is not None and len(spec.submodule_search_locations) == 1)
        rule = Path(next(iter(spec.submodule_search_locations)))/'envs/kaggriculture/kaggriculture.py'
        modules['first_yield_days'] = maturity.first_yield_days_from_source(infra.read(rule))
        return modules
    shared.runtime = runtime
    return shared


def failed_source(infra,check):
    # This frozen helper authenticates the runtime-v2 failure, not the older v1 failure.
    helper = _clone(DIAGNOSTIC_HELPERS,'_struct_compat_runtime_failure_reader')
    return helper.failed_source(infra,check)


def diagnostic_paths():
    private = {name+'.local.json' for name in ('attempt','binding','closed','owner',
        'selection-closed','selection-started','result','certificate')}
    private.add('F01.selection.local.json')
    stage = {'binding.local.json','F01/events.jsonl.gz'}
    stage.update('F01/'+name+'.local.json' for name in
        ('case','consumed','permit','readiness','started','supervision','accepted'))
    return {**{'private/'+name:DIAGNOSTIC_PRIVATE/name for name in private},
            **{'stage/'+name:DIAGNOSTIC_STAGE/name for name in stage}}


def diagnostic_source(infra,check,previous):
    """Authenticate the exact sealed diagnostic inventory, without decoding events."""
    paths = diagnostic_paths()
    witness_raw = infra.read(DIAGNOSTIC_WITNESS)
    witness = infra.decode(witness_raw)
    public = infra.decode(infra.read(REPO/DIAGNOSTIC_RESULTS/'certificate.json'))
    require(set(witness) == {'schema','registration_commit','files','salted_commitment_sha256'}
        and witness['schema'] == 'task-harvest-first-step-diagnostic-witness-v1'
        and witness['registration_commit'] == DIAGNOSTIC_COMMIT
        and type(witness['files']) is dict and set(witness['files']) == set(paths)
        and public['registration_commit'] == DIAGNOSTIC_COMMIT
        and public['diagnostic_collection_passed'] is True and public['original_case_passed'] is False
        and public['official_steps_returned'] == 1 and public['accepted_pairs'] == 0
        and public['originals']['files'] == len(paths) == 18 and public['originals']['bytes'] == 246642
        and public['originals']['salted_commitment_sha256'] == witness['salted_commitment_sha256'] == DIAGNOSTIC_COMMITMENT)
    originals = {}
    for key,path in sorted(paths.items()):
        check()
        item = witness['files'][key]
        require(type(item) is dict and set(item) == {'bytes','sha256'}
            and type(item['bytes']) is int and 0 <= item['bytes'] <= 1024**2)
        raw = infra.read(path,1024**2)
        require(len(raw) == item['bytes'] and sha(raw) == item['sha256'])
        originals[key] = raw
    require(sum(map(len,originals.values())) == 246642)
    binding_raw = originals['private/binding.local.json']
    binding = infra.decode(binding_raw)
    salt = binding['salt']
    require(type(salt) is str and len(salt) == 64 and all(c in '0123456789abcdef' for c in salt))
    commitment = sha(bytes.fromhex(salt)+b'\0task-harvest-first-step-diagnostic-v1\0'+
        json.dumps(witness['files'],sort_keys=True,separators=(',',':'),ensure_ascii=True).encode())
    require(commitment == DIAGNOSTIC_COMMITMENT)
    reg = infra.decode(infra.read(REPO/DIAGNOSTIC_REG))
    closed = infra.decode(originals['private/closed.local.json'])
    result = infra.decode(originals['private/result.local.json'])
    certificate = infra.decode(originals['private/certificate.local.json'])
    selected = infra.decode(originals['private/selection-closed.local.json'])
    case = infra.decode(originals['stage/F01/case.local.json'])
    accepted = infra.decode(originals['stage/F01/accepted.local.json'])
    require(sha(binding_raw) == reg['binding_commitment_sha256']
        and binding.get('schema') == DIAGNOSTIC_DOC+'binding' and binding.get('base_commit') == reg['base_commit']
        and binding['prior_binding_sha256'] == previous['binding_sha256']
        and binding['completed_sources'] == previous['binding']['completed_sources']
        and binding['default_sha256'] == previous['binding']['default_sha256']
        and originals['stage/binding.local.json'] == binding_raw
        and originals['private/F01.selection.local.json'] == previous['originals']['private/F01.selection.local.json']
        and ordered(binding['selections']) == ordered(previous['binding']['selections'][:1])
        and selected.get('schema') == DIAGNOSTIC_DOC+'selection-closed' and selected.get('outcome') == 'prepared'
        and selected.get('binding_sha256') == sha(binding_raw) and selected.get('source_roles_copied') == 1
        and selected.get('trace_rows_read') == 0 and selected.get('target_reselection') is False
        and selected.get('model_forwards') == selected.get('official_transitions') == selected.get('replay_decodes') == 0
        and selected['selection_started_sha256'] == binding['selection_started_sha256']
            == sha(originals['private/selection-started.local.json'])
        and closed.get('schema') == DIAGNOSTIC_DOC+'closed' and closed.get('registration_commit') == DIAGNOSTIC_COMMIT
        and closed.get('binding_sha256') == sha(binding_raw) and closed.get('outcome') == 'diagnostic_complete'
        and closed.get('error_kind') is None and closed.get('accepted_diagnostics') == 1
        and closed.get('accepted_pairs') == 0 and closed.get('retry_allowed') is False
        and closed.get('owner_sha256') == sha(originals['private/owner.local.json'])
        and closed.get('attempt_sha256') == sha(originals['private/attempt.local.json'])
        and closed.get('result_sha256') == sha(originals['private/result.local.json'])
        and certificate.get('closed_sha256') == sha(originals['private/closed.local.json'])
        and certificate.get('result_sha256') == closed['result_sha256']
        and certificate.get('diagnostic_passed') is True and certificate.get('original_case_passed') is False
        and result.get('original_case_passed') is False and result.get('accepted_pairs') == 0
        and result.get('full_horizon_verified') is False and result.get('economic_value_evaluated') is False
        and case.get('schema') == 'task-harvest-unexpressed-case-v1'
        and case.get('passed') is False and case.get('status') == 'failed'
        and accepted.get('diagnostic_passed') is True and accepted.get('original_case_passed') is False
        and accepted.get('case_receipt',{}).get('sha256') == sha(originals['stage/F01/case.local.json'])
        and result.get('accepted') == certificate.get('accepted') == [dict(role='F01',sha256=sha(originals['stage/F01/accepted.local.json']))])
    diagnostic,calls = result['diagnostic'],case['meter']['calls']
    require(diagnostic == certificate['diagnostic'] == accepted['diagnostic']
        and diagnostic.get('passed') is True and diagnostic.get('outcome') == 'located_original_error'
        and diagnostic.get('error_id') == 'STATIC_eb5d37bb2b6c6eaf48d5'
        and diagnostic.get('location_kind') == 'direct_raise' and diagnostic.get('rejected_value_kind') == 'official_Struct'
        and diagnostic['comparison']['semantic_frame_equal'] is True
        and diagnostic['comparison']['ordered_frame_equal'] is True
        and diagnostic['comparison']['original_type_request_check'] == dict(outcome='raises',error_kind='ValueError')
        and result['meter'] == case['meter']
        and all(calls.get(k) == 1 for k in ('arm_attempts','restore_returns','official_step_attempts','official_step_returns'))
        and calls.get('actor_returns') == calls.get('rival_returns') == 241
        and all(calls.get(k,0) == 0 for k in ('arm_returns','request_patch_attempts','request_patch_returns')))
    return dict(binding_sha256=sha(binding_raw),closed_sha256=sha(originals['private/closed.local.json']),
        result_sha256=sha(originals['private/result.local.json']),certificate_sha256=sha(originals['private/certificate.local.json']),
        witness_sha256=sha(witness_raw),salted_commitment_sha256=commitment,files=18,bytes=246642)


def prepare_payload(infra,modules,base,contract,check,legacy=None):
    legacy = _legacy() if legacy is None else legacy
    previous = failed_source(infra,check)
    diagnostic = diagnostic_source(infra,check,previous)
    context = modules['sources'].load(guard=check,verify_game_files=False)
    require(context.identities == previous['binding']['completed_sources']
        and all(base.validate_assets(context.manifests[c],modules['driver']) for c in ('C01','C02'))
        and sha(infra.read(GAME/'main.py')) == previous['binding']['default_sha256'])
    selected = legacy.inherited_points(previous,contract)
    utils_identity(infra,contract)
    legacy.no_execution_imports()
    check()
    return previous,diagnostic,context.identities,selected


def binding_read(infra,reg):
    raw,binding = _legacy().original_binding_read(infra,reg)
    require(binding.get('diagnostic_registration_commit') == DIAGNOSTIC_COMMIT
        and binding.get('diagnostic_commitment_sha256') == DIAGNOSTIC_COMMITMENT
        and binding.get('prior_failed_original_files') == 18 and binding.get('prior_failed_original_bytes') == 187746
        and binding.get('diagnostic_original_files') == 18 and binding.get('diagnostic_original_bytes') == 246642
        and binding.get('prior_lineage_official_steps') == 2
        and binding.get('official_utils_sha256') == reg['panel']['official_utils_sha256']
        and len(binding['selections']) == 4)
    return raw,binding


def prepare(shared=None,legacy=None):
    require(os.name == 'nt' and sys.flags.isolated and sys.flags.dont_write_bytecode and __debug__)
    legacy = _legacy() if legacy is None else legacy
    shared = _shared(legacy) if shared is None else shared
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
    modules = legacy.metadata_modules()
    PRIVATE.mkdir(exist_ok=False)
    start_sha = infra.write(PRIVATE/'selection-started.local.json',dict(schema=DOC+'selection-started',
        base_commit=BASE,public_sources=source_map,specification=contract.specification(),
        prior_registration_commit=PRIOR_COMMIT,diagnostic_registration_commit=DIAGNOSTIC_COMMIT,
        target_reselection=False,local_registration_not_third_party_timestamp=True))
    binding_sha,selected = None,[]
    try:
        previous,diagnostic,identities,selected = prepare_payload(infra,modules,base,contract,check,legacy)
        for case in selected:
            check()
            path = PRIVATE/(case['source']['role']+'.selection.local.json')
            require(infra.write(path,case,shared.MAX_SMALL) == sha(previous['originals']['private/'+path.name]))
        require(check() and all(sha(infra.read(REPO/p)) == h for p,h in source_map.items()))
        binding = dict(schema=DOC+'binding',base_commit=BASE,salt=secrets.token_hex(32),
            selection_started_sha256=start_sha,completed_sources=identities,
            prior_registration_commit=PRIOR_COMMIT,prior_binding_sha256=previous['binding_sha256'],
            prior_failed_closed_sha256=previous['closed_sha256'],prior_failure_witness_sha256=previous['witness_sha256'],
            prior_failure_commitment_sha256=previous['salted_commitment_sha256'],
            prior_failed_original_files=previous['files'],prior_failed_original_bytes=previous['bytes'],
            diagnostic_registration_commit=DIAGNOSTIC_COMMIT,diagnostic_binding_sha256=diagnostic['binding_sha256'],
            diagnostic_closed_sha256=diagnostic['closed_sha256'],diagnostic_result_sha256=diagnostic['result_sha256'],
            diagnostic_certificate_sha256=diagnostic['certificate_sha256'],diagnostic_witness_sha256=diagnostic['witness_sha256'],
            diagnostic_commitment_sha256=diagnostic['salted_commitment_sha256'],
            diagnostic_original_files=diagnostic['files'],diagnostic_original_bytes=diagnostic['bytes'],
            prior_lineage_official_steps=2,official_utils_sha256=contract.STRUCT_UTILS_SHA256,
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
    return dict(status='prepared_four_point_struct_reproduction',copied_roles=4,
        authenticated_prior_originals=36,prior_lineage_official_steps=2,completed_source_receipts=32,
        historical_game_files_rehashed=False,trace_rows_read=0,target_reselection=False,
        model_forwards=0,official_transitions=0,replay_decodes=0)


def main(argv=None):
    legacy = _legacy()
    shared = _shared(legacy)
    shared.prepare = lambda:prepare(shared,legacy)
    return shared.main(argv)


if __name__ == '__main__':
    raise SystemExit(main())
