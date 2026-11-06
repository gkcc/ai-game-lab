"""Thin isolated adaptation of the frozen harvest Job coordinator.

Preparation uses metadata imports only. Execution reuses the original owned
Job, memory, permit, failure closure and acceptance orchestration verbatim in
an isolated module namespace. No imported historical module is mutated.
"""
import importlib
import importlib.util
import json
import os
from pathlib import Path
import secrets
import sys
import time
from types import ModuleType

from hashlib import sha256


REPO = Path(__file__).resolve().parents[3]
GAME = REPO.parent.parent/'kaggriculture'
BASE = '967dbe9691b40c1b3f39eb5bf30b3727496f7e10'
PRIOR_COMMIT = '101e3c155cbe24dc4f152ad8a6d72a35f2195f70'
PRIOR_REG = 'review/evidence/task-harvest-coverage-v1-20261106/registration.json'
PRIOR_RESULTS = 'review/evidence/task-harvest-coverage-v1-result-20261106'
PRIOR_PRIVATE = REPO.parent/'private/task-harvest-coverage-v1-20261106'
PREFIX = 'kaggriculture/learning/task_harvest_unexpressed_v1/'
EVIDENCE = 'review/evidence/task-harvest-unexpressed-v1-20261106'
REGISTRATION, NOTE = EVIDENCE+'/registration.json', EVIDENCE+'/REGISTRATION.zh-CN.md'
PRIVATE = REPO.parent/'private/task-harvest-unexpressed-v1-20261106'
STAGE = GAME/'artifacts/task-harvest-unexpressed-v1-20261106'
SHARED = 'kaggriculture/learning/task_harvest_window_v1/runner.py'
SCHEMA = 'task-harvest-unexpressed-prereg-v1'


def require(value):
    if not value:
        raise ValueError('Unexpressed harvest registration differs')


def sha(raw):
    return sha256(raw).hexdigest()


def git(*args):
    import subprocess
    result = subprocess.run(['git','-C',str(REPO),*args],stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,stderr=subprocess.DEVNULL,timeout=45,
        creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    require(result.returncode == 0 and len(result.stdout) <= 16*1024**2)
    return result.stdout


def required_sources(infra):
    old = infra.decode(git('show',PRIOR_COMMIT+':'+PRIOR_REG))
    require(len(old['public_sources']) == 434)
    additions = {PRIOR_REG,NOTE,PREFIX+'README.zh-CN.md',SHARED}
    additions.update(PRIOR_RESULTS+'/'+name for name in
        ('summary.json','certificate.json','RESULTS.zh-CN.md','REVIEW.zh-CN.md','NEXT.zh-CN.md'))
    paths = set(old['public_sources']) | additions
    paths.update(p.relative_to(REPO).as_posix() for p in (REPO/PREFIX).glob('*.py'))
    require({PREFIX+n for n in ('__init__.py','contract.py','selection.py','adapter.py',
        'core.py','worker.py','runner.py')}.issubset(paths))
    for path,expected in old['public_sources'].items():
        raw = infra.read(REPO/path)
        if path == 'review/PROGRESS.md':
            original = git('show',PRIOR_COMMIT+':'+path)
            published = git('show',BASE+':'+path)
            require(sha(original) == expected and published.startswith(original) and raw == published)
        else:
            require(sha(raw) == expected)
    for path in additions-{NOTE,PREFIX+'README.zh-CN.md'}:
        require(infra.read(REPO/path) == git('show',BASE+':'+path))
    return paths


def marker(kind,commit,binding_sha,role=None):
    require(kind in ('attempt','permit','started','consumed'))
    value = dict(schema='task-harvest-unexpressed-marker-v1',kind=kind,
        registration_commit=commit,binding_sha256=binding_sha,retry_allowed=False)
    if kind != 'attempt':
        require(role in ('F01','F02','F03','F04'))
        value['role'] = role
    return value


def binding_read(infra,reg):
    raw = infra.read(PRIVATE/'binding.local.json')
    binding = infra.decode(raw)
    closed = infra.decode(infra.read(PRIVATE/'selection-closed.local.json'))
    require(sha(raw) == reg['binding_commitment_sha256']
        and binding.get('schema') == 'task-harvest-unexpressed-binding-v1'
        and binding.get('base_commit') == BASE
        and closed.get('schema') == 'task-harvest-unexpressed-selection-closed-v1'
        and closed.get('outcome') == 'prepared' and closed.get('error_kind') is None
        and closed.get('binding_sha256') == sha(raw) and closed.get('source_roles_scanned') == 4
        and closed.get('model_forwards') == closed.get('official_transitions') == 0
        and closed.get('retry_allowed') is False
        and binding['selection_started_sha256'] == closed['selection_started_sha256']
            == sha(infra.read(PRIVATE/'selection-started.local.json')))
    for case in binding['selections']:
        require(infra.decode(infra.read(PRIVATE/(case['source']['role']+'.selection.local.json'))) == case)
    return raw,binding


def _no_shadow():
    require(not os.path.lexists(GAME/'learning/task_harvest_unexpressed_v1'))


def _import(name):
    module = importlib.import_module('learning.'+name)
    relative = 'kaggriculture/learning/'+name.replace('.','/')+'.py'
    require(Path(module.__file__).resolve(strict=True) == (REPO/relative).resolve(strict=True))
    return module


def metadata_modules():
    _no_shadow()
    sys.path.insert(0,str(REPO/'kaggriculture'))
    result = {key:_import(name) for key,name in (
        ('selection','task_harvest_unexpressed_v1.selection'),
        ('maturity','task_harvest_window_v1.selection'),
        ('sources','task_teacher_fresh_panel_v1.sources'),
        ('driver','task_pair_dev_campaign_v1.driver'),
        ('fresh','task_pair_fresh_multi_v1.entry'))}
    no_execution_imports()
    return result


def no_execution_imports():
    require(not any(name == prefix or name.startswith(prefix+'.')
        for prefix in ('kaggle_environments','torch') for name in sys.modules))


def first_days(infra,maturity):
    spec = importlib.util.find_spec('kaggle_environments')
    require(spec is not None and len(spec.submodule_search_locations) == 1)
    rule = Path(next(iter(spec.submodule_search_locations)))/'envs/kaggriculture/kaggriculture.py'
    return maturity.first_yield_days_from_source(infra.read(rule))


def _shared():
    """Authenticate old bytes, then rebind only explicit coordination seams."""
    raw = (REPO/SHARED).read_bytes()
    require(raw == git('show',BASE+':'+SHARED))
    inherited = ModuleType('_unexpressed_owned_job_coordinator')
    # The inherited run() constructs child argv from this file, not the old entry.
    inherited.__file__ = str(Path(__file__).resolve())
    exec(compile(raw,str(REPO/SHARED),'exec',dont_inherit=True),inherited.__dict__)
    for key in ('REPO','GAME','BASE','PRIOR_COMMIT','PRIOR_REG','PRIOR_RESULTS','PREFIX',
                'EVIDENCE','REGISTRATION','NOTE','PRIVATE','STAGE','SCHEMA'):
        setattr(inherited,key,globals()[key])
    inherited.required_sources = required_sources
    inherited.marker = marker
    inherited.binding_read = binding_read
    original_infrastructure = inherited.infrastructure

    def infrastructure():
        infra = original_infrastructure()
        original_write = infra.write
        def write(path,value,*args,**kwargs):
            # Output schema names only; ownership and write semantics stay old.
            if isinstance(value,dict) and isinstance(value.get('schema'),str):
                if value['schema'].startswith('task-harvest-window-'):
                    value = dict(value,schema=value['schema'].replace(
                        'task-harvest-window-','task-harvest-unexpressed-',1))
            return original_write(path,value,*args,**kwargs)
        infra.write = write
        return infra
    inherited.infrastructure = infrastructure
    original_authenticate = inherited.authenticate
    def authenticate(commit):
        _no_shadow()
        return original_authenticate(commit)
    inherited.authenticate = authenticate
    def runtime(infra,bootstrap,reg):
        _no_shadow()
        modules = infra._runtime(bootstrap,reg)
        modules['selection'] = _import('task_harvest_unexpressed_v1.selection')
        modules['window_worker'] = _import('task_harvest_unexpressed_v1.worker')
        modules['first_yield_days'] = first_days(infra,_import('task_harvest_window_v1.selection'))
        return modules
    inherited.runtime = runtime
    return inherited


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
    start_sha = infra.write(PRIVATE/'selection-started.local.json',dict(
        schema='task-harvest-unexpressed-selection-started-v1',base_commit=BASE,
        public_sources=source_map,specification=contract.specification(),
        no_counterfactual_outcomes_read=True,local_registration_not_third_party_timestamp=True))
    binding_sha,selected = None,[]
    try:
        prior_reg = infra.decode(infra.read(REPO/PRIOR_REG))
        prior_raw = infra.read(PRIOR_PRIVATE/'binding.local.json')
        prior = infra.decode(prior_raw)
        closed = infra.decode(infra.read(PRIOR_PRIVATE/'closed.local.json'))
        require(sha(prior_raw) == prior_reg['binding_commitment_sha256']
            and closed['outcome'] == 'completed' and closed['error_kind'] is None
            and closed['registration_commit'] == PRIOR_COMMIT and closed['binding_sha256'] == sha(prior_raw)
            and closed['completed_sources'] == 4)
        context = modules['sources'].load(guard=check)
        require(context.identities == prior['completed_sources']
            and all(base.validate_assets(context.manifests[c],modules['driver']) for c in ('C01','C02'))
            and sha(infra.read(GAME/'main.py')) == prior['default_sha256'])
        days = first_days(infra,modules['maturity'])
        for source in contract.sources():
            check()
            row = context.rows[(source['cohort'],source['game'])]
            stage = GAME/'artifacts'/modules['fresh'].STAGES[source['cohort']]
            stream = modules['driver']._trace(stage,stage/'traces'/(source['game']+'.jsonl.gz'),row['trace_sha256'])
            try:
                value = modules['selection'].select_target(stream,side=source['side'],first_yield_days=days,guard=check)
                require(value['counts']['trace_lines'] == 719)
            finally:
                primary = sys.exc_info()[1]
                try:
                    stream.close()
                except BaseException:
                    if primary is None:
                        raise
            selected.append(dict(source=source,selection=value))
            infra.write(PRIVATE/(source['role']+'.selection.local.json'),selected[-1],shared.MAX_SMALL)
        no_execution_imports()
        require(check() and all(sha(infra.read(REPO/p)) == h for p,h in source_map.items()))
        binding = dict(schema='task-harvest-unexpressed-binding-v1',base_commit=BASE,salt=secrets.token_hex(32),
            selection_started_sha256=start_sha,completed_sources=context.identities,
            prior_binding_sha256=sha(prior_raw),default_sha256=prior['default_sha256'],selections=selected)
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
            dict(schema='task-harvest-unexpressed-selection-closed-v1',outcome='prepared' if error is None else 'failed',
                error_kind=None if error is None else type(error).__name__,binding_sha256=binding_sha,
                selection_started_sha256=start_sha,source_roles_scanned=len(selected),model_forwards=0,
                official_transitions=0,replay_decodes=0,retry_allowed=False),error)
    return dict(status='prepared_no_windows',roles=4,eligible_targets=sum(c['selection']['target'] is not None for c in selected),
                source_trace_turns=4*719,model_forwards=0,official_transitions=0,replay_decodes=0)


def main(argv=None):
    shared = _shared()
    shared.prepare = lambda:prepare(shared)
    return shared.main(argv)


if __name__ == '__main__':
    raise SystemExit(main())
