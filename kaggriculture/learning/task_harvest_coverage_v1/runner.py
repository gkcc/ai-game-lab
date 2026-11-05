"""One registered, read-only metadata pass over four consumed traces.

No game/model bootstrap, subprocess Worker, replay decoding or policy calls.
The existing completed-source loader and bounded trace reader are reused.
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
import stat
import subprocess
import sys
import time
from types import ModuleType


REPO = Path(__file__).resolve().parents[3]
GAME = REPO.parent.parent/'kaggriculture'
BASE = '0b7718ffeb841b6bf7ec5d53547af7b409667aa7'
PRIOR_COMMIT = '28c81b30c03aab8b1675c1286fda12cf43b90f98'
PRIOR_REG = 'review/evidence/task-harvest-window-v1-20261106/registration.json'
PRIOR_RESULTS = 'review/evidence/task-harvest-window-v1-result-20261106'
PRIOR_PRIVATE = REPO.parent/'private/task-harvest-window-v1-20261106'
PREFIX = 'kaggriculture/learning/task_harvest_coverage_v1/'
EVIDENCE = 'review/evidence/task-harvest-coverage-v1-20261106'
REGISTRATION, NOTE = EVIDENCE+'/registration.json', EVIDENCE+'/REGISTRATION.zh-CN.md'
PRIVATE = REPO.parent/'private/task-harvest-coverage-v1-20261106'
INFRA = 'kaggriculture/learning/task_teacher_fresh_panel_v1/runner.py'
READY = 'kaggriculture/learning/task_teacher_fresh_panel_v1/contract.py'
GIB, MIB = 1024**3, 1024**2
LIMITS = dict(initial_available_bytes=6*GIB, running_available_bytes=4*GIB,
    readiness_samples=3, readiness_interval_seconds=10, readiness_timeout_seconds=300,
    max_rss_bytes=2*GIB, wall_seconds=900, max_private_bytes=8*MIB,
    max_file_bytes=MIB, cases=4, source_trace_rows=2876, retries=0)
SCHEMA = 'task-harvest-coverage-prereg-v1'


def require(value):
    if not value:
        raise ValueError('Harvest coverage contract differs')


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def git(*args):
    result = subprocess.run(['git','-C',str(REPO),*args],stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,stderr=subprocess.DEVNULL,timeout=45,
        creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    require(result.returncode == 0 and len(result.stdout) <= 16*MIB)
    return result.stdout


def infrastructure():
    raw = (REPO/INFRA).read_bytes()
    require(raw == git('show',BASE+':'+INFRA))
    module = ModuleType('_coverage_frozen_infrastructure')
    module.__file__ = str(REPO/INFRA)
    exec(compile(raw,module.__file__,'exec',dont_inherit=True),module.__dict__)
    return module


def sources(infra):
    old = infra.decode(git('show',PRIOR_COMMIT+':'+PRIOR_REG))
    require(len(old['public_sources']) == 421)
    additions = {PRIOR_REG, NOTE, PREFIX+'README.zh-CN.md'}
    additions.update(PRIOR_RESULTS+'/'+name for name in
                     ('summary.json','certificate.json','RESULTS.zh-CN.md','REVIEW.zh-CN.md','NEXT.zh-CN.md'))
    paths = set(old['public_sources']) | additions
    paths.update(p.relative_to(REPO).as_posix() for p in (REPO/PREFIX).glob('*.py'))
    require({PREFIX+x for x in ('__init__.py','coverage.py','runner.py')}.issubset(paths))
    for path,expected in old['public_sources'].items():
        current = infra.read(REPO/path)
        if path == 'review/PROGRESS.md':
            earlier = git('show',PRIOR_COMMIT+':'+path)
            published = git('show',BASE+':'+path)
            require(sha(earlier) == expected and published.startswith(earlier) and current == published)
        else:
            require(sha(current) == expected)
    for path in additions-{NOTE,PREFIX+'README.zh-CN.md'}:
        require(infra.read(REPO/path) == git('show',BASE+':'+path))
    return {path:sha(infra.read(REPO/path)) for path in sorted(paths)}


def metadata_modules():
    public = (REPO/'kaggriculture').resolve(strict=True)
    sys.path.insert(0,str(public))
    loaded = {}
    for key,name in (('coverage','task_harvest_coverage_v1.coverage'),
        ('selection','task_harvest_window_v1.selection'),('contract','task_harvest_window_v1.contract'),
        ('sources','task_teacher_fresh_panel_v1.sources'),('driver','task_pair_dev_campaign_v1.driver'),
        ('fresh','task_pair_fresh_multi_v1.entry')):
        module = importlib.import_module('learning.'+name)
        require(Path(module.__file__).resolve(strict=True).is_relative_to(public))
        loaded[key] = module
    no_execution_imports()
    return loaded


def no_execution_imports():
    require(not any(name == prefix or name.startswith(prefix+'.')
                    for prefix in ('kaggle_environments','torch') for name in sys.modules))


def protocol(modules):
    return dict(sources=modules['contract'].sources(),selection_window=[240,648],
        analysis=modules['coverage'].specification(),source_scope='completed_32_only',
        old_missing_coverage_retained=True,private_examples=True,complete_games=0,
        model_forwards=0,official_transitions=0,training=False,new_seeds=False,
        independent_evaluation=False,promotion=False)


class Guard:
    def __init__(self,infra,source_map):
        self.infra = infra
        self.base = infra.snapshot(infra.BASE_ENTRY,source_map[infra.BASE_ENTRY])
        self.ready = infra.snapshot(READY,source_map[READY])
        import psutil
        self.process = psutil.Process()
        self.started = time.monotonic()
        self.samples = 0
        self.minimum_available = None
        self.maximum_rss = 0

    def __call__(self):
        available, rss = self.base.physical_available(), self.process.memory_info().rss
        self.samples += 1
        self.minimum_available = available if self.minimum_available is None else min(self.minimum_available,available)
        self.maximum_rss = max(self.maximum_rss,rss)
        if available < LIMITS['running_available_bytes'] or rss > LIMITS['max_rss_bytes']:
            raise MemoryError('Coverage resource limit')
        if time.monotonic()-self.started >= LIMITS['wall_seconds']:
            raise TimeoutError('Coverage deadline')
        if PRIVATE.exists():
            total = 0
            for path in PRIVATE.iterdir():
                info = path.lstat()
                require(stat.S_ISREG(info.st_mode) and not path.is_symlink()
                    and not (getattr(info,'st_file_attributes',0) & 0x400)
                    and info.st_size <= LIMITS['max_file_bytes'])
                total += info.st_size
            require(total <= LIMITS['max_private_bytes'])
        return True

    def start(self):
        result = self.ready.await_memory(self.base.physical_available,
            timeout=min(LIMITS['readiness_timeout_seconds'],LIMITS['wall_seconds']-(time.monotonic()-self.started)))
        self()
        return result

    def report(self):
        return dict(samples=self.samples,minimum_available_bytes=self.minimum_available,
            maximum_rss_bytes=self.maximum_rss,elapsed_seconds=time.monotonic()-self.started,
            scope='discrete_cooperative_checks_not_continuous_or_windows_job_supervision')


def frozen_runtime(infra,source_map,commit):
    identity = infra.snapshot(infra.IDENTITY,source_map[infra.IDENTITY])
    infra.preparation_runtime(identity,commit)
    no_execution_imports()


def prior_binding(infra):
    reg = infra.decode(infra.read(REPO/PRIOR_REG))
    raw = infra.read(PRIOR_PRIVATE/'binding.local.json')
    require(sha(raw) == reg['binding_commitment_sha256'])
    binding = infra.decode(raw)
    closed = infra.decode(infra.read(PRIOR_PRIVATE/'selection-closed.local.json'))
    result = infra.decode(infra.read(PRIOR_PRIVATE/'coverage-result.local.json'))
    require(closed['outcome'] == 'prepared' and closed['error_kind'] is None
        and closed['binding_sha256'] == sha(raw)
        and result['registration_commit'] == PRIOR_COMMIT and result['binding_sha256'] == sha(raw)
        and result['outcome'] == 'no_eligible_targets_stopped_before_execution'
        and all(c['selection']['target'] is None for c in binding['selections']))
    return raw,binding


def context(modules,prior,guard):
    loaded = modules['sources'].load(guard=guard)
    require(loaded.identities == prior['completed_sources']
        and [c['source'] for c in prior['selections']] == modules['contract'].sources())
    require(sha(guard.infra.read(GAME/'main.py')) == prior['default_sha256'])
    for manifest in loaded.manifests.values():
        require(guard.base.validate_assets(manifest,modules['driver']))
        guard()
    no_execution_imports()
    return loaded


def close(infra,name,value,error):
    try:
        infra.write(PRIVATE/name,value,LIMITS['max_file_bytes'])
    except BaseException:
        if error is None:
            raise


def prepare():
    require(os.name == 'nt' and sys.flags.isolated and sys.flags.dont_write_bytecode and __debug__)
    infra = infrastructure()
    require(git('rev-parse','HEAD').decode().strip() == BASE
        and not PRIVATE.exists() and not (REPO/REGISTRATION).exists())
    source_map = sources(infra)
    modules = metadata_modules()
    guard = Guard(infra,source_map)
    PRIVATE.mkdir(exist_ok=False)
    infra.write(PRIVATE/'prepare-started.local.json',dict(base_commit=BASE,public_sources=source_map,
        protocol=protocol(modules),limits=LIMITS,retry_allowed=False),LIMITS['max_file_bytes'])
    binding_sha = None
    try:
        readiness = guard.start()
        frozen_runtime(infra,source_map,BASE)
        old_raw,old = prior_binding(infra)
        loaded = context(modules,old,guard)
        binding = dict(schema='task-harvest-coverage-binding-v1',base_commit=BASE,salt=secrets.token_hex(32),
            prior_binding_sha256=sha(old_raw),completed_sources=loaded.identities,
            default_sha256=old['default_sha256'],sources=modules['contract'].sources(),readiness=readiness)
        require(source_map == sources(infra) and guard())
        binding_sha = infra.write(PRIVATE/'binding.local.json',binding,LIMITS['max_file_bytes'])
        reg = dict(schema=SCHEMA,base_commit=BASE,prior_registration_commit=PRIOR_COMMIT,
            public_sources=source_map,versions=infra._versions(),limits=LIMITS,protocol=protocol(modules),
            engine_commit=modules['contract'].ENGINE_COMMIT,engine_sha256=modules['contract'].ENGINE_SHA256,
            binding_commitment_sha256=binding_sha)
        (REPO/EVIDENCE).mkdir(parents=True,exist_ok=True)
        infra.write(REPO/REGISTRATION,reg,LIMITS['max_file_bytes'])
    finally:
        error = sys.exc_info()[1]
        close(infra,'prepare-closed.local.json',dict(outcome='prepared' if error is None else 'failed',
            error_kind=None if error is None else type(error).__name__,binding_sha256=binding_sha,
            model_forwards=0,official_transitions=0,trace_rows_decoded=0,retry_allowed=False,
            resources=guard.report()),error)
    return dict(status='prepared_no_trace_analysis',source_roles=4,trace_rows_decoded=0,
                model_forwards=0,official_transitions=0)


def aggregate(analyses,coverage):
    require(type(analyses) is list and len(analyses) == 4)
    total = coverage.empty_counts()
    for analysis in analyses:
        counts = analysis['counts']
        require(type(counts) is dict and set(counts) == set(total))
        for group,values in total.items():
            require(type(counts[group]) is dict and set(counts[group]) == set(values))
            for key in values:
                value = counts[group][key]
                require(type(value) is int and value >= 0)
                total[group][key] += value
    return total


def consume_trace(stream,analysis,*,side,first_days,guard,scan):
    """Require EOF and preserve a primary failure if stream cleanup also fails."""
    def counted():
        for value in stream:
            guard()
            scan['rows'] += 1
            yield value
        scan['eof'] = True
    iterator = counted()
    try:
        value = analysis(iterator,side=side,first_yield_days=first_days,guard=guard)
        require(scan['eof'] and scan['rows'] == 719)
    finally:
        primary = sys.exc_info()[1]
        errors = []
        for source in (iterator,stream):
            try:
                source.close()
            except BaseException as error:
                errors.append(error)
        scan['closed'] = not errors
        scan['close_errors'] = len(errors)
        if errors and primary is None:
            raise errors[0]
    return value


def run(commit):
    require(os.name == 'nt' and sys.flags.isolated and sys.flags.dont_write_bytecode and __debug__
        and type(commit) is str and re.fullmatch('[0-9a-f]{40}',commit)
        and git('rev-parse','HEAD').decode().strip() == commit
        and not git('status','--porcelain','--untracked-files=all'))
    git('merge-base','--is-ancestor',BASE,commit)
    infra = infrastructure()
    reg_raw = infra.read(REPO/REGISTRATION,LIMITS['max_file_bytes'])
    require(reg_raw == git('show',commit+':'+REGISTRATION))
    reg = infra.decode(reg_raw)
    source_map = sources(infra)
    require(reg['schema'] == SCHEMA and reg['base_commit'] == BASE and reg['limits'] == LIMITS
        and reg['prior_registration_commit'] == PRIOR_COMMIT and reg['public_sources'] == source_map
        and reg['versions'] == infra._versions()
        and all(infra.read(REPO/path) == git('show',commit+':'+path) for path in source_map))
    modules = metadata_modules()
    require(reg['protocol'] == protocol(modules) and reg['engine_commit'] == modules['contract'].ENGINE_COMMIT
        and reg['engine_sha256'] == modules['contract'].ENGINE_SHA256)
    binding_raw = infra.read(PRIVATE/'binding.local.json',LIMITS['max_file_bytes'])
    require(sha(binding_raw) == reg['binding_commitment_sha256'])
    binding = infra.decode(binding_raw)
    prepared = infra.decode(infra.read(PRIVATE/'prepare-closed.local.json',LIMITS['max_file_bytes']))
    require(prepared['outcome'] == 'prepared' and prepared['error_kind'] is None
        and prepared['binding_sha256'] == sha(binding_raw)
        and all(not (PRIVATE/name).exists() for name in ('attempt.local.json','closed.local.json','result.local.json')))
    guard = Guard(infra,source_map)
    infra.write(PRIVATE/'attempt.local.json',dict(registration_commit=commit,
        binding_sha256=sha(binding_raw),retry_allowed=False),LIMITS['max_file_bytes'])
    cases, scans = [], []
    result_sha = None
    try:
        guard.start()
        frozen_runtime(infra,source_map,commit)
        old_raw,old = prior_binding(infra)
        require(sha(old_raw) == binding['prior_binding_sha256']
            and binding['sources'] == modules['contract'].sources()
            and binding['completed_sources'] == old['completed_sources'])
        loaded = context(modules,old,guard)
        spec = importlib.util.find_spec('kaggle_environments')
        require(spec is not None and len(spec.submodule_search_locations) == 1)
        rule_path = Path(next(iter(spec.submodule_search_locations)))/'envs/kaggriculture/kaggriculture.py'
        first_days = modules['selection'].first_yield_days_from_source(infra.read(rule_path))
        for source,previous in zip(binding['sources'],old['selections']):
            guard()
            row = loaded.rows[(source['cohort'],source['game'])]
            stage = GAME/'artifacts'/modules['fresh'].STAGES[source['cohort']]
            stream = modules['driver']._trace(stage,stage/'traces'/(source['game']+'.jsonl.gz'),row['trace_sha256'])
            scan = dict(role=source['role'],rows=0,eof=False,closed=False,close_errors=0)
            scans.append(scan)
            analysis = consume_trace(stream,modules['coverage'].analyze,side=source['side'],
                first_days=first_days,guard=guard,scan=scan)
            require(analysis['original_selection'] == previous['selection'] and guard())
            value = dict(source=source,source_trace_sha256=row['trace_sha256'],analysis=analysis)
            case_sha = infra.write(PRIVATE/(source['role']+'.local.json'),value,LIMITS['max_file_bytes'])
            cases.append(dict(role=source['role'],sha256=case_sha,analysis=analysis))
        totals = aggregate([case['analysis'] for case in cases],modules['coverage'])
        require(totals['trace']['rows'] == LIMITS['source_trace_rows']
            and sources(infra) == source_map and context(modules,old,guard).identities == loaded.identities
            and infra.read(PRIVATE/'binding.local.json') == binding_raw)
        no_execution_imports()
        result_sha = infra.write(PRIVATE/'result.local.json',dict(schema='task-harvest-coverage-result-v1',
            registration_commit=commit,binding_sha256=sha(binding_raw),cases=cases,totals=totals,
            trace_scans=scans,
            model_forwards=0,official_transitions=0,replay_decodes=0,complete_games=0,
            training=False,gpu=False,default_unchanged=True,retry_allowed=False),LIMITS['max_file_bytes'])
        guard()
    finally:
        error = sys.exc_info()[1]
        close(infra,'closed.local.json',dict(outcome='completed' if error is None else 'failed',
            registration_commit=commit,binding_sha256=sha(binding_raw),result_sha256=result_sha,
            error_kind=None if error is None else type(error).__name__,completed_sources=len(cases),
            trace_scans=scans,
            model_forwards=0,official_transitions=0,retry_allowed=False,resources=guard.report()),error)
    return dict(status='completed',source_roles=len(cases),source_trace_rows=totals['trace']['rows'],
                model_forwards=0,official_transitions=0,training=False)


class Parser(argparse.ArgumentParser):
    def error(self,_message):
        raise ValueError('Invalid command line')


def main(argv=None):
    parser = Parser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--prepare',action='store_true')
    group.add_argument('--run',action='store_true')
    parser.add_argument('--registration-commit')
    try:
        try:
            args = parser.parse_args(argv)
        except SystemExit as error:
            if error.code in (None,0):
                return 0
            raise ValueError('Invalid command line') from None
        require(not args.prepare or args.registration_commit is None)
        value = prepare() if args.prepare else run(args.registration_commit)
        print(json.dumps(value,allow_nan=False))
        return 0
    except BaseException as error:
        kind = 'memory' if isinstance(error,MemoryError) else 'timeout' if isinstance(error,TimeoutError) else 'invalid_or_runtime'
        print(json.dumps(dict(status='failed',kind=kind,retry_allowed=False)),file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
