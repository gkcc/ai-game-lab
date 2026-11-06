"""Freeze then run eight public constructed official transitions, once.

Reuse existing bounded read/write helpers and the official mechanism fixture.
No historical trajectory/seed inventory or model checkpoint is opened.
"""
import argparse
import hashlib
import importlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import re
import secrets
import subprocess
import sys
import time
from types import ModuleType, SimpleNamespace


REPO = Path(__file__).resolve().parents[3]
GAME = REPO.parent.parent/'kaggriculture'
BASE = 'f4d52334015f449480bda67726a69543d124ead5'
PREFIX = 'kaggriculture/learning/task_harvest_choice_v1/'
EVIDENCE = 'review/evidence/task-harvest-choice-v1-20261106'
REGISTRATION, NOTE = EVIDENCE+'/registration.json', EVIDENCE+'/REGISTRATION.zh-CN.md'
PRIVATE = REPO.parent/'private/task-harvest-choice-v1-20261106'
PRIOR_REG = 'review/evidence/task-harvest-trajectory-v1-20261106/registration.json'
PRIOR_RESULTS = 'review/evidence/task-harvest-trajectory-v1-result-20261106'
PRIOR_PRIVATE = REPO.parent/'private/task-harvest-trajectory-v1-20261106'
HELPER = 'kaggriculture/learning/task_harvest_trajectory_v1/runner.py'
SCHEMA = 'task-harvest-choice-v1'


def require(value):
    if not value:
        raise ValueError('Harvest choice micro-validation differs')


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'),
                      ensure_ascii=True, allow_nan=False).encode()


def git(*args):
    result = subprocess.run(['git', '-C', str(REPO), *args], stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, timeout=45,
        creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
    require(result.returncode == 0 and len(result.stdout) <= 16*1024**2)
    return result.stdout


def helpers():
    raw = (REPO/HELPER).read_bytes()
    require(raw == git('show', BASE+':'+HELPER))
    shared = ModuleType('_choice_frozen_io_helpers')
    shared.__file__ = str(REPO/HELPER)
    exec(compile(raw, shared.__file__, 'exec', dont_inherit=True), shared.__dict__)
    # Private namespace only; only file/JSON helpers are called from this module.
    shared.BASE, shared.PRIVATE = BASE, PRIVATE
    return shared, shared.infrastructure()


def versions():
    return dict(python='.'.join(map(str, sys.version_info[:3])),
        numpy=importlib.metadata.version('numpy'), psutil=importlib.metadata.version('psutil'),
        kaggle_environments=importlib.metadata.version('kaggle-environments'))


def source_map(infra, contract):
    old = infra.decode(git('show', contract.PRIOR_COMMIT+':'+PRIOR_REG))
    require(len(old['public_sources']) == 514)
    extras = {PRIOR_REG, NOTE, PREFIX+'README.zh-CN.md',
        'kaggriculture/decision/verify_mechanisms.py', 'kaggriculture/decision/execution.py',
        'kaggriculture/learning/task_joint_decoder_v1/official_probe.py'}
    extras.update(PRIOR_RESULTS+'/'+n for n in (
        'summary.json', 'certificate.json', 'RESULTS.zh-CN.md', 'REVIEW.zh-CN.md', 'NEXT.zh-CN.md'))
    paths = set(old['public_sources']) | extras
    paths.update(p.relative_to(REPO).as_posix() for p in (REPO/PREFIX).glob('*.py'))
    require({PREFIX+n for n in ('__init__.py', 'adapter.py', 'fixtures.py', 'official_probe.py',
        'contract.py', 'runner.py', 'test_adapter.py', 'test_runner.py')}.issubset(paths))
    for path, expected in old['public_sources'].items():
        raw = infra.read(REPO/path)
        if path == 'review/PROGRESS.md':
            original = git('show', contract.PRIOR_COMMIT+':'+path)
            require(sha(original) == expected and raw == git('show', BASE+':'+path)
                    and raw.startswith(original))
        else:
            require(sha(raw) == expected)
    for path in extras-{NOTE, PREFIX+'README.zh-CN.md'}:
        require(infra.read(REPO/path) == git('show', BASE+':'+path))
    return {p:sha(infra.read(REPO/p)) for p in sorted(paths)}


def metadata(infra, sources):
    public = (REPO/'kaggriculture').resolve(strict=True)
    sys.path.insert(0, str(public))
    modules = []
    for name in ('contract', 'fixtures'):
        relative = PREFIX+name+'.py'
        require(sha(infra.read(REPO/relative)) == sources[relative])
        module = importlib.import_module('learning.task_harvest_choice_v1.'+name)
        require(Path(module.__file__).resolve(strict=True) == (REPO/relative).resolve(strict=True))
        modules.append(module)
    require('kaggle_environments' not in sys.modules and 'torch' not in sys.modules)
    return modules


def installed_identity(infra, contract):
    spec = importlib.util.find_spec('kaggle_environments')
    require(spec is not None and len(spec.submodule_search_locations) == 1)
    root = Path(next(iter(spec.submodule_search_locations)))
    paths = {'envs/kaggriculture/kaggriculture.py':contract.ENGINE_SHA,
        'envs/kaggriculture/kaggriculture.json':contract.SPEC_SHA,
        'core.py':contract.CORE_SHA, 'utils.py':contract.UTILS_SHA}
    require(all(sha(infra.read(root/p)) == h for p,h in paths.items()))
    require(sha(infra.read(REPO/'kaggriculture/bots/teacher_fertilizer_terminal_v2.py'))
            == contract.TEACHER_SHA)
    return paths


def default_identity(infra):
    reg = infra.decode(infra.read(REPO/PRIOR_REG))
    raw = infra.read(PRIOR_PRIVATE/'binding.local.json')
    require(sha(raw) == reg['binding_commitment_sha256'])
    binding = infra.decode(raw)
    require(sha(infra.read(GAME/'main.py')) == binding['default_sha256'])
    return binding['default_sha256']


class Guard:
    """Reuse Windows memory sampler; cooperative, not an OS watchdog."""
    def __init__(self, infra, sources, limits):
        import psutil
        self.process, self.limits = psutil.Process(), limits
        self.base = infra.snapshot(infra.BASE_ENTRY, sources[infra.BASE_ENTRY])
        self.started, self.samples = time.monotonic(), 0
        self.minimum_available, self.maximum_rss = None, 0

    def __call__(self):
        available, rss = self.base.physical_available(), self.process.memory_info().rss
        self.samples += 1
        self.minimum_available = available if self.minimum_available is None else min(available, self.minimum_available)
        self.maximum_rss = max(rss, self.maximum_rss)
        if available < self.limits['running_available_bytes'] or rss > self.limits['max_rss_bytes']:
            raise MemoryError('Choice micro-validation memory floor')
        if time.monotonic()-self.started >= self.limits['wall_seconds']:
            raise TimeoutError('Choice micro-validation deadline')
        return True

    def report(self):
        return dict(samples=self.samples, minimum_available_bytes=self.minimum_available,
            maximum_rss_bytes=self.maximum_rss, elapsed_seconds=time.monotonic()-self.started,
            scope='cooperative_samples_not_independent_os_supervision')


def prepare():
    shared, infra = helpers()
    require(os.name == 'nt' and sys.flags.isolated and sys.flags.dont_write_bytecode and __debug__
        and git('rev-parse', 'HEAD').decode().strip() == BASE
        and not os.path.lexists(PRIVATE) and not os.path.lexists(REPO/REGISTRATION))
    relative = PREFIX+'contract.py'
    contract = infra.snapshot(relative, sha(infra.read(REPO/relative)))
    sources = source_map(infra, contract)
    contract, fixtures = metadata(infra, sources)
    guard = Guard(infra, sources, contract.LIMITS)
    guard()
    installed = installed_identity(infra, contract)
    default = default_identity(infra)
    PRIVATE.mkdir(exist_ok=False)
    write = lambda name, value:shared.private_write(infra, contract.LIMITS, name, value)
    started = write('prepare-started.local.json', dict(base_commit=BASE, public_sources=sources,
        specification=contract.specification(fixtures.specification())))
    binding_sha = None
    try:
        binding_sha = write('binding.local.json', dict(schema=SCHEMA+'-binding', base_commit=BASE,
            salt=secrets.token_hex(32), started_sha256=started, default_sha256=default))
        require(guard() and all(sha(infra.read(REPO/p)) == h for p,h in sources.items()))
        reg = dict(schema=SCHEMA+'-prereg', base_commit=BASE, prior_registration_commit=contract.PRIOR_COMMIT,
            engine_commit=contract.ENGINE_COMMIT, installed_source_sha256=installed,
            teacher_sha256=contract.TEACHER_SHA, public_sources=sources, versions=versions(),
            limits=contract.LIMITS, specification=contract.specification(fixtures.specification()),
            binding_commitment_sha256=binding_sha)
        infra.write(REPO/REGISTRATION, reg)
    finally:
        primary = sys.exc_info()[1]
        shared.close(infra, contract, 'prepare-closed.local.json', dict(outcome='prepared' if primary is None else 'failed',
            error_kind=None if primary is None else type(primary).__name__, binding_sha256=binding_sha,
            started_sha256=started, official_initializations=0, official_action_transitions=0,
            model_forwards=0, retry_allowed=False, resources=guard.report()), primary)
    return dict(status='prepared', official_initializations=0, official_action_transitions=0,
                model_forwards=0, old_trajectory_reads=0, frozen_sources=len(sources))


def runtime(infra, sources, guard):
    guard()
    require('numpy' not in sys.modules)
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    os.environ['OMP_NUM_THREADS'] = '1'
    # The existing official fixture suppresses noisy package-import output.
    # No owned-write interception surrounds imports; the earlier import failure
    # was caused by doing these imports inside a restricted output context.
    mechanism = importlib.import_module('decision.verify_mechanisms')
    execution = importlib.import_module('decision.execution')
    adapter = importlib.import_module('learning.task_harvest_choice_v1.adapter')
    for module, relative in (
        (mechanism, 'kaggriculture/decision/verify_mechanisms.py'),
        (execution, 'kaggriculture/decision/execution.py'),
        (adapter, PREFIX+'adapter.py')):
        require(Path(module.__file__).resolve(strict=True) == (REPO/relative).resolve(strict=True)
            and sha(infra.read(REPO/relative)) == sources[relative])
    guard()
    return SimpleNamespace(env=mechanism.env, obs=mechanism.obs, rules=mechanism.rules,
        ExecutionTrace=execution.ExecutionTrace, agent_factory=adapter.HarvestChoiceAgent)


def validate_result(result, meter, emitted, contract, fixtures):
    expected = {key+suffix:limit for key,limit in contract.PROBE_CAPS.items()
                for suffix in ('_attempted', '_returned')}
    expected.update(direct_boundary_requests=2, evidence_write_failures=0)
    require(canonical(meter) == canonical(expected) and result.get('passed') is True
        and canonical(result.get('counts')) == canonical(meter)
        and type(result.get('model_forwards')) is int and result['model_forwards'] == 0
        and type(result.get('complete_games')) is int and result['complete_games'] == 0)
    pairs = [(name, op) for name in fixtures.NAMES for op in fixtures.OPS]
    require(type(result.get('rows')) is list
        and [(r.get('scene'), r.get('op')) for r in result['rows']] == pairs
        and [v['name'] for v in emitted] == [n+'-'+o.lower()+'.local.json' for n,o in pairs])
    return True


def run(commit):
    shared, infra = helpers()
    require(os.name == 'nt' and sys.flags.isolated and sys.flags.dont_write_bytecode and __debug__
        and type(commit) is str and re.fullmatch('[0-9a-f]{40}', commit)
        and git('rev-parse', 'HEAD').decode().strip() == commit
        and not git('status', '--porcelain', '--untracked-files=all'))
    git('merge-base', '--is-ancestor', BASE, commit)
    raw = infra.read(REPO/REGISTRATION)
    require(raw == git('show', commit+':'+REGISTRATION))
    reg = infra.decode(raw)
    sources = reg['public_sources']
    require(all(sha(infra.read(REPO/p)) == h for p,h in sources.items()))
    contract, fixtures = metadata(infra, sources)
    require(reg['schema'] == SCHEMA+'-prereg' and reg['base_commit'] == BASE
        and canonical(reg['limits']) == canonical(contract.LIMITS)
        and canonical(reg['specification']) == canonical(contract.specification(fixtures.specification()))
        and reg['versions'] == versions())
    binding_raw = infra.read(PRIVATE/'binding.local.json')
    binding = infra.decode(binding_raw)
    ready = infra.decode(infra.read(PRIVATE/'prepare-closed.local.json'))
    require(reg['binding_commitment_sha256'] == sha(binding_raw) == ready['binding_sha256']
        and ready['outcome'] == 'prepared' and ready['error_kind'] is None
        and binding['started_sha256'] == ready['started_sha256']
            == sha(infra.read(PRIVATE/'prepare-started.local.json')))
    guard = Guard(infra, sources, contract.LIMITS)
    guard()
    write = lambda name, value:shared.private_write(infra, contract.LIMITS, name, value)
    attempt = write('attempt.local.json', dict(registration_commit=commit,
        binding_sha256=sha(binding_raw), retry_allowed=False, limits=contract.LIMITS))
    meter, emitted, result_sha = {}, [], None
    try:
        require(installed_identity(infra, contract) == reg['installed_source_sha256']
            and default_identity(infra) == binding['default_sha256'])
        rt = runtime(infra, sources, guard)
        probe = importlib.import_module('learning.task_harvest_choice_v1.official_probe')
        require(Path(probe.__file__).resolve(strict=True) == (REPO/PREFIX/'official_probe.py').resolve(strict=True)
            and canonical(probe.CAPS) == canonical(contract.PROBE_CAPS))
        def emit(name, record):
            guard()
            digest = write(name, record)
            emitted.append(dict(name=name, sha256=digest))
        result = probe.run(rt, guard, emit, meter)
        require(validate_result(result, meter, emitted, contract, fixtures)
            and guard() and all(sha(infra.read(REPO/p)) == h for p,h in sources.items())
            and installed_identity(infra, contract) == reg['installed_source_sha256']
            and sha(infra.read(GAME/'main.py')) == binding['default_sha256'])
        result = dict(schema=SCHEMA+'-result', registration_commit=commit, **result)
        result_sha = write('result.local.json', result)
    finally:
        primary = sys.exc_info()[1]
        shared.close(infra, contract, 'closed.local.json', dict(schema=SCHEMA+'-closed',
            registration_commit=commit, outcome='completed' if primary is None else 'failed',
            error_kind=None if primary is None else type(primary).__name__, binding_sha256=sha(binding_raw),
            attempt_sha256=attempt, result_sha256=result_sha, counts=meter, emitted=emitted,
            retry_allowed=False, resources=guard.report()), primary)
    return dict(status='completed', counts=meter, model_forwards=0, complete_games=0)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    commands.add_parser('prepare')
    commands.add_parser('run').add_argument('--registration-commit', required=True)
    args = parser.parse_args(argv)
    try:
        result = prepare() if args.command == 'prepare' else run(args.registration_commit)
    except Exception as error:
        print(json.dumps(dict(status='failed', error_kind=type(error).__name__, retry_allowed=False)))
        return 1
    print(json.dumps(result, ensure_ascii=True, separators=(',', ':')))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
