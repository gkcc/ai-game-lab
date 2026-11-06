"""Authenticate and consume the four sealed event streams once, without execution.

Use the existing game venv with python -I -B. No source-game loader, model loader,
official environment or subprocess Worker is used by this read-only entry.
"""
import argparse
import gzip
import hashlib
import importlib.metadata
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
BASE = '596c2767342e8311b1272a417c2a83782c29423a'
PREFIX = 'kaggriculture/learning/task_harvest_trajectory_v1/'
EVIDENCE = 'review/evidence/task-harvest-trajectory-v1-20261106'
REGISTRATION, NOTE = EVIDENCE+'/registration.json', EVIDENCE+'/REGISTRATION.zh-CN.md'
PRIVATE = REPO.parent/'private/task-harvest-trajectory-v1-20261106'
PRIOR_REG = 'review/evidence/task-harvest-struct-compat-v1-20261106/registration.json'
PRIOR_RESULTS = 'review/evidence/task-harvest-struct-compat-v1-result-20261106'
PRIOR_PRIVATE = REPO.parent/'private/task-harvest-struct-compat-v1-20261106'
PRIOR_STAGE = GAME/'artifacts/task-harvest-struct-compat-v1-20261106'
PRIOR_WITNESS = REPO.parent/'private/task-harvest-struct-compat-v1-20261106-publication-witness.local.json'
INFRA = 'kaggriculture/learning/task_teacher_fresh_panel_v1/runner.py'
SCHEMA = 'task-harvest-trajectory-v1'


def require(value):
    if not value:
        raise ValueError('Harvest trajectory contract differs')


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=True,
                      allow_nan=False).encode()


def git(*args):
    result = subprocess.run(['git', '-C', str(REPO), *args], stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, timeout=45,
        creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
    require(result.returncode == 0 and len(result.stdout) <= 16*1024**2)
    return result.stdout


def infrastructure():
    raw = (REPO/INFRA).read_bytes()
    require(raw == git('show', BASE+':'+INFRA))
    module = ModuleType('_trajectory_frozen_infrastructure')
    module.__file__ = str(REPO/INFRA)
    exec(compile(raw, module.__file__, 'exec', dont_inherit=True), module.__dict__)
    return module


def no_execution_imports():
    require(not any(name == prefix or name.startswith(prefix+'.')
        for prefix in ('kaggle_environments', 'torch') for name in sys.modules))


def modules(infra, source_map):
    public = (REPO/'kaggriculture').resolve(strict=True)
    sys.path.insert(0, str(public))
    loaded = [infra.snapshot(PREFIX+name+'.py', source_map[PREFIX+name+'.py'])
              for name in ('contract', 'analysis')]
    helper = sys.modules.get('learning.task_single_intervention_v1.agent_adapter')
    require(helper is not None and Path(helper.__file__).resolve(strict=True)
        == public/'learning/task_single_intervention_v1/agent_adapter.py')
    no_execution_imports()
    return loaded


def sources(infra, contract):
    old = infra.decode(git('show', contract.PRIOR_COMMIT+':'+PRIOR_REG))
    require(len(old['public_sources']) == 499)
    additions = {PRIOR_REG, NOTE, PREFIX+'README.zh-CN.md', PREFIX+'FEATURES.zh-CN.md'}
    additions.update(PRIOR_RESULTS+'/'+name for name in (
        'summary.json', 'certificate.json', 'RESULTS.zh-CN.md', 'REVIEW.zh-CN.md', 'NEXT.zh-CN.md'))
    paths = set(old['public_sources']) | additions
    paths.update(p.relative_to(REPO).as_posix() for p in (REPO/PREFIX).glob('*.py'))
    require({PREFIX+n for n in ('__init__.py', 'contract.py', 'runner.py', 'analysis.py',
                               'test_runner.py', 'test_analysis.py')}.issubset(paths))
    for path, expected in old['public_sources'].items():
        current = infra.read(REPO/path)
        if path == 'review/PROGRESS.md':
            original = git('show', contract.PRIOR_COMMIT+':'+path)
            published = git('show', BASE+':'+path)
            require(sha(original) == expected and published.startswith(original) and current == published)
        else:
            require(sha(current) == expected)
    for path in additions-{NOTE, PREFIX+'README.zh-CN.md', PREFIX+'FEATURES.zh-CN.md'}:
        require(infra.read(REPO/path) == git('show', BASE+':'+path))
    return {path:sha(infra.read(REPO/path)) for path in sorted(paths)}


class Guard:
    """Cooperative checks, not an independent OS supervisor."""
    def __init__(self, infra, source_map, limits):
        self.infra, self.limits = infra, limits
        self.base = infra.snapshot(infra.BASE_ENTRY, source_map[infra.BASE_ENTRY])
        import psutil
        self.process = psutil.Process()
        self.started, self.samples, self.minimum_available, self.maximum_rss = time.monotonic(), 0, None, 0

    def __call__(self):
        available, rss = self.base.physical_available(), self.process.memory_info().rss
        self.samples += 1
        self.minimum_available = available if self.minimum_available is None else min(available, self.minimum_available)
        self.maximum_rss = max(rss, self.maximum_rss)
        if available < self.limits['running_available_bytes'] or rss > self.limits['max_rss_bytes']:
            raise MemoryError('Trajectory resource limit')
        if time.monotonic()-self.started >= self.limits['wall_seconds']:
            raise TimeoutError('Trajectory deadline')
        no_execution_imports()
        return True

    def report(self):
        return dict(samples=self.samples, minimum_available_bytes=self.minimum_available,
            maximum_rss_bytes=self.maximum_rss, elapsed_seconds=time.monotonic()-self.started,
            scope='discrete_cooperative_checks_not_continuous_or_windows_job_supervision')


def regular(path):
    path = Path(path).absolute()
    require(path.resolve(strict=True) == path)
    for entry in (path, *path.parents):
        info = entry.lstat()
        require(not entry.is_symlink() and not (getattr(info, 'st_file_attributes', 0) & 0x400))
    info = path.stat()
    require(stat.S_ISREG(info.st_mode))
    return info.st_size


def archive_paths(roles):
    private = {n+'.local.json' for n in ('attempt', 'binding', 'closed', 'owner',
        'selection-closed', 'selection-started', 'result', 'certificate')}
    private.update(role+'.selection.local.json' for role in roles)
    stage = {'binding.local.json'}
    stage.update(role+'/'+n for role in roles for n in (
        'readiness.local.json', 'permit.local.json', 'started.local.json', 'consumed.local.json',
        'case.local.json', 'events.jsonl.gz', 'supervision.local.json', 'accepted.local.json'))
    return {**{'private/'+p:PRIOR_PRIVATE/p for p in private},
            **{'stage/'+p:PRIOR_STAGE/p for p in stage}}


def archive_metadata(infra, contract, check):
    """The gzip body is deliberately never opened here."""
    witness_raw = infra.read(PRIOR_WITNESS)
    witness = infra.decode(witness_raw)
    certificate = infra.decode(infra.read(REPO/PRIOR_RESULTS/'certificate.json'))
    require(witness.get('schema') == 'task-harvest-struct-compat-witness-v1'
        and witness.get('registration_commit') == contract.PRIOR_COMMIT
        and certificate['registration_commit'] == contract.PRIOR_COMMIT
        and certificate['technical_reproduction_passed'] is True
        and certificate['public_summary_sha256'] == sha(infra.read(REPO/PRIOR_RESULTS/'summary.json')))
    paths = archive_paths(contract.ROLES)
    require(type(witness.get('files')) is dict and set(paths) == set(witness['files']) and len(paths) == 45)
    originals, total = {}, 0
    for key, path in sorted(paths.items()):
        check()
        item = witness['files'][key]
        require(type(item) is dict and set(item) == {'bytes', 'sha256'}
            and type(item['bytes']) is int and 0 < item['bytes'] <= contract.LIMITS['max_compressed_stream_bytes']
            and type(item['sha256']) is str and re.fullmatch('[0-9a-f]{64}', item['sha256']))
        require(regular(path) == item['bytes'])
        total += item['bytes']
        if key.endswith('/events.jsonl.gz'):
            continue
        raw = infra.read(path, 1024**2)
        require(sha(raw) == item['sha256'] and len(raw) == item['bytes'])
        originals[key] = infra.decode(raw)
    require(len(originals) == 41 and total == certificate['originals']['bytes'] == 11171803)
    old_binding = originals['private/binding.local.json']
    salt = old_binding['salt']
    require(type(salt) is str and re.fullmatch('[0-9a-f]{64}', salt))
    commitment = sha(bytes.fromhex(salt)+b'\0task-harvest-struct-compat-v1\0'+canonical(witness['files']))
    require(commitment == contract.PRIOR_COMMITMENT == witness['salted_commitment_sha256']
        == certificate['originals']['salted_commitment_sha256'])
    prior_reg = infra.decode(infra.read(REPO/PRIOR_REG))
    require(prior_reg['binding_commitment_sha256'] == witness['files']['private/binding.local.json']['sha256']
        and old_binding['default_sha256'] == sha(infra.read(GAME/'main.py')))
    cases = [originals['stage/'+role+'/case.local.json'] for role in contract.ROLES]
    for role, case in zip(contract.ROLES, cases):
        require(case.get('passed') is True and case['source']['role'] == role
            and case['row']['status'] == 'paired' and case['row']['technical_passed'] is True
            and case['events']['closed'] is True
            and case['events']['sha256'] == witness['files']['stage/'+role+'/events.jsonl.gz']['sha256']
            and type(case['events']['bytes']) is int
            and case['events']['bytes'] == witness['files']['stage/'+role+'/events.jsonl.gz']['bytes'])
    return dict(witness_sha256=sha(witness_raw), files=witness['files'], cases=cases,
        default_sha256=old_binding['default_sha256'], prior_commitment=commitment,
        small_originals_authenticated=41, gzip_body_reads=0)


def rules(infra, contract):
    spec = importlib.util.find_spec('kaggle_environments')
    require(spec is not None and len(spec.submodule_search_locations) == 1)
    path = Path(next(iter(spec.submodule_search_locations)))/'envs/kaggriculture/kaggriculture.py'
    require(sha(infra.read(path)) == contract.ENGINE_SHA256)
    no_execution_imports()


class CompressedReader:
    """A single forward read path, with compressed byte count and SHA."""
    def __init__(self, stream, limit, guard):
        self.stream, self.limit, self.guard = stream, limit, guard
        self.count, self.digest, self.eof = 0, hashlib.sha256(), False

    def read(self, size=-1):
        self.guard()
        require(type(size) is int and (size == -1 or size >= 0))
        size = self.limit-self.count+1 if size < 0 else min(size, self.limit-self.count+1)
        data = self.stream.read(size)
        self.count += len(data)
        self.digest.update(data)
        require(self.count <= self.limit)
        if size and not data:
            self.eof = True
        return data


class EventStream:
    def __init__(self, raw, expected, limits, aggregate, decode, guard):
        require(type(expected.get('bytes')) is int and 0 < expected['bytes'] <= limits['max_compressed_stream_bytes'])
        self.raw = CompressedReader(raw, expected['bytes'], guard)
        self.expected, self.limits, self.aggregate = expected, limits, aggregate
        self.decode, self.guard, self.lines, self.decoded_bytes = decode, guard, 0, 0
        self.started, self.complete = False, False

    def __iter__(self):
        require(not self.started)
        self.started = True
        with gzip.GzipFile(fileobj=self.raw, mode='rb') as stream:
            while True:
                self.guard()
                line = stream.readline(self.limits['max_line_bytes']+1)
                if not line:
                    break
                require(len(line) <= self.limits['max_line_bytes'] and line.endswith(b'\n'))
                self.decoded_bytes += len(line)
                self.aggregate['decoded_bytes'] += len(line)
                require(self.decoded_bytes <= self.limits['max_decoded_stream_bytes']
                    and self.aggregate['decoded_bytes'] <= self.limits['max_decoded_total_bytes'])
                self.lines += 1
                yield self.decode(line)
        while self.raw.read(65536):
            pass
        require(self.raw.eof and self.raw.count == self.expected['bytes']
            and self.raw.digest.hexdigest() == self.expected['sha256'])
        self.complete = True

    def receipt(self):
        require(self.complete)
        return dict(compressed_bytes=self.raw.count, compressed_sha256=self.raw.digest.hexdigest(),
            decoded_bytes=self.decoded_bytes, lines=self.lines, crc_eof_verified=True,
            compressed_eof_verified=True, physical_passes=1)


def private_write(infra, limits, name, value, *, closure=False):
    require(Path(name).name == name and name.endswith('.local.json'))
    raw_bytes = len(json.dumps(value, separators=(',', ':'), ensure_ascii=True, allow_nan=False).encode())+1
    total = sum(regular(p) for p in PRIVATE.iterdir())
    cap = limits['max_private_bytes']-(0 if closure else limits['closure_reserve_bytes'])
    require(raw_bytes <= limits['max_output_file_bytes'] and total+raw_bytes <= cap)
    return infra.write(PRIVATE/name, value, limits['max_output_file_bytes'])


def close(infra, contract, name, value, primary):
    try:
        private_write(infra, contract.LIMITS, name, value, closure=True)
    except BaseException:
        if primary is None:
            raise


def prepare():
    infra = infrastructure()
    require(os.name == 'nt' and sys.flags.isolated and sys.flags.dont_write_bytecode and __debug__
        and git('rev-parse', 'HEAD').decode().strip() == BASE
        and not os.path.lexists(PRIVATE) and not os.path.lexists(REPO/REGISTRATION))
    contract_raw = infra.read(REPO/PREFIX/'contract.py')
    contract = infra.snapshot(PREFIX+'contract.py', sha(contract_raw))
    source_map = sources(infra, contract)
    contract, analysis = modules(infra, source_map)
    guard = Guard(infra, source_map, contract.LIMITS)
    guard()
    rules(infra, contract)
    PRIVATE.mkdir(exist_ok=False)
    start_sha = private_write(infra, contract.LIMITS, 'prepare-started.local.json',
        dict(base_commit=BASE, public_sources=source_map,
             specification=contract.specification(analysis.specification())))
    binding_sha, prior = None, None
    try:
        prior = archive_metadata(infra, contract, guard)
        binding = dict(schema=SCHEMA+'-binding', base_commit=BASE, salt=secrets.token_hex(32),
            started_sha256=start_sha, witness_sha256=prior['witness_sha256'],
            prior_commitment=prior['prior_commitment'], default_sha256=prior['default_sha256'])
        binding_sha = private_write(infra, contract.LIMITS, 'binding.local.json', binding)
        require(guard() and all(sha(infra.read(REPO/p)) == h for p,h in source_map.items()))
        reg = dict(schema=SCHEMA+'-prereg', base_commit=BASE, prior_registration_commit=contract.PRIOR_COMMIT,
            engine_commit=contract.ENGINE_COMMIT, engine_sha256=contract.ENGINE_SHA256,
            public_sources=source_map, limits=contract.LIMITS,
            specification=contract.specification(analysis.specification()),
            binding_commitment_sha256=binding_sha,
            versions=dict(python='.'.join(map(str, sys.version_info[:3])),
                          psutil=importlib.metadata.version('psutil')))
        infra.write(REPO/REGISTRATION, reg)
    finally:
        primary = sys.exc_info()[1]
        close(infra, contract, 'prepare-closed.local.json', dict(schema=SCHEMA+'-prepare-closed',
            outcome='prepared' if primary is None else 'failed', error_kind=None if primary is None else type(primary).__name__,
            binding_sha256=binding_sha, started_sha256=start_sha, retry_allowed=False,
            gzip_body_reads=0, small_originals_authenticated=41 if prior is not None else None,
            resources=guard.report()), primary)
    return dict(status='prepared_metadata_only', small_originals_authenticated=41,
                gzip_body_reads=0, official_transitions=0, model_forwards=0)


def run(commit):
    infra = infrastructure()
    require(os.name == 'nt' and sys.flags.isolated and sys.flags.dont_write_bytecode and __debug__
        and type(commit) is str and re.fullmatch('[0-9a-f]{40}', commit)
        and git('rev-parse', 'HEAD').decode().strip() == commit
        and not git('status', '--porcelain', '--untracked-files=all'))
    git('merge-base', '--is-ancestor', BASE, commit)
    raw = infra.read(REPO/REGISTRATION)
    require(raw == git('show', commit+':'+REGISTRATION))
    reg = infra.decode(raw)
    source_map = reg['public_sources']
    require(all(sha(infra.read(REPO/p)) == h for p,h in source_map.items()))
    contract, analysis = modules(infra, source_map)
    require(reg['schema'] == SCHEMA+'-prereg' and reg['base_commit'] == BASE
        and canonical(reg['limits']) == canonical(contract.LIMITS)
        and canonical(reg['specification']) == canonical(contract.specification(analysis.specification()))
        and reg['versions'] == dict(python='.'.join(map(str, sys.version_info[:3])),
                                  psutil=importlib.metadata.version('psutil')))
    binding_raw = infra.read(PRIVATE/'binding.local.json')
    binding = infra.decode(binding_raw)
    prepared = infra.decode(infra.read(PRIVATE/'prepare-closed.local.json'))
    require(sha(binding_raw) == reg['binding_commitment_sha256'] == prepared['binding_sha256']
        and prepared['outcome'] == 'prepared' and prepared['error_kind'] is None
        and prepared['started_sha256'] == binding['started_sha256']
            == sha(infra.read(PRIVATE/'prepare-started.local.json')))
    guard = Guard(infra, source_map, contract.LIMITS)
    guard()
    attempt_sha = private_write(infra, contract.LIMITS, 'attempt.local.json',
        dict(registration_commit=commit, binding_sha256=sha(binding_raw), retry_allowed=False))
    rows, receipts, aggregate, result_sha = [], [], {'decoded_bytes':0}, None
    try:
        rules(infra, contract)
        prior = archive_metadata(infra, contract, guard)
        require(prior['witness_sha256'] == binding['witness_sha256']
            and prior['prior_commitment'] == binding['prior_commitment']
            and prior['default_sha256'] == binding['default_sha256'])
        for role, case in zip(contract.ROLES, prior['cases']):
            path = PRIOR_STAGE/role/'events.jsonl.gz'
            expected = prior['files']['stage/'+role+'/events.jsonl.gz']
            require(regular(path) == expected['bytes'])
            with path.open('rb') as stream:
                events = EventStream(stream, expected, contract.LIMITS, aggregate, infra.decode, guard)
                report = analysis.analyze_case(events, case, guard)
                receipt = events.receipt()
            require(type(case['events']['lines']) is int and receipt['lines'] == case['events']['lines']
                and type(case['events']['decoded_bytes']) is int
                and receipt['decoded_bytes'] == case['events']['decoded_bytes'])
            private_write(infra, contract.LIMITS, role+'.analysis.local.json', report['private'])
            receipt_sha = private_write(infra, contract.LIMITS, role+'.receipt.local.json', receipt)
            receipts.append(dict(role=role, receipt_sha256=receipt_sha))
            rows.append(report['public'])
        require(len(rows) == 4 and guard()
            and all(sha(infra.read(REPO/p)) == h for p,h in source_map.items())
            and sha(infra.read(GAME/'main.py')) == binding['default_sha256'])
        no_execution_imports()
        result = dict(schema=SCHEMA+'-result', registration_commit=commit,
            status='completed', cases=rows, event_streams=4, physical_passes_per_stream=1,
            event_lines=sum(c['events']['lines'] for c in prior['cases']),
            decoded_bytes=aggregate['decoded_bytes'], analyzed_suffix_transitions=384,
            authenticated_small_originals=41, authenticated_gzip_originals=4,
            inherited_original_commitment=contract.PRIOR_COMMITMENT,
            model_forwards=0, official_transitions=0, complete_games=0,
            training=False, gpu=False, new_seeds=False, terminal_profit=False,
            independent_evaluation=False, promotion=False)
        result_sha = private_write(infra, contract.LIMITS, 'result.local.json', result)
    finally:
        primary = sys.exc_info()[1]
        close(infra, contract, 'closed.local.json', dict(schema=SCHEMA+'-closed',
            registration_commit=commit, outcome='completed' if primary is None else 'failed',
            error_kind=None if primary is None else type(primary).__name__,
            binding_sha256=sha(binding_raw), attempt_sha256=attempt_sha, result_sha256=result_sha,
            completed_cases=len(rows), accepted_receipts=receipts, retry_allowed=False,
            resources=guard.report()), primary)
    return dict(status='completed', completed_cases=4, event_lines=result['event_lines'],
                decoded_bytes=aggregate['decoded_bytes'], model_forwards=0, official_transitions=0)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest='command', required=True)
    subs.add_parser('prepare')
    child = subs.add_parser('run')
    child.add_argument('--registration-commit', required=True)
    args = parser.parse_args(argv)
    try:
        value = prepare() if args.command == 'prepare' else run(args.registration_commit)
    except Exception as error:
        print(json.dumps(dict(status='failed', error_kind=type(error).__name__, retry_allowed=False)))
        return 1
    print(json.dumps(value, separators=(',', ':'), ensure_ascii=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
