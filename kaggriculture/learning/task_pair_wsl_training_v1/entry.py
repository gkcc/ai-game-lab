"""Fail-closed candidate entry for one registered paired WSL training run.

No registration for this entry exists yet. In particular, importing this file
does not read game data or a checkpoint. The caller must independently fix
the private contract digest before execution and supervise this process.
"""

import argparse
from contextlib import contextmanager
import hashlib
import importlib
import importlib.machinery
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
import time
import types


GAME_SOURCE = Path(__file__).resolve().parents[2]
REPO = GAME_SOURCE.parent
REG_REL = 'review/evidence/task-pair-wsl-training-v1-20261030/registration.json'
OUTPUT_NAME = 'task-pair-wsl-training-v1-20261030'
SCHEMA = 'task-pair-wsl-training-registration-v1'
PRIVATE_SCHEMA = 'task-pair-wsl-training-private-contract-v1'
MAX_SOURCE = 4 * 1024 * 1024
MAX_PRIVATE = 65536
HEX_SHA = re.compile(r'[0-9a-f]{64}\Z')
HEX_COMMIT = re.compile(r'[0-9a-f]{40}\Z')
HEX_TOKEN = re.compile(r'[0-9a-f]{32}\Z')
SMOKE_REL = 'kaggriculture/learning/task_pair_gpu_smoke_v1/import_guard.py'
SMOKE_SHA = 'c5ec904b6c7af23ffb4a6ddd31dd8aa47a0e760cc892087f07e038b303c65f3d'
EXTRA_REL = 'kaggriculture/learning/task_pair_wsl_import_v1/guard.py'
BOOTSTRAP_REL = 'kaggriculture/learning/panel_eval_v3/bootstrap.py'
CERT_REL = 'review/evidence/task-pair-mask-recheck-v1-result-20261030/certificate.json'
EARLY_REL = 'review/evidence/task-bc-v1-20261023-run1/model/selected-export.json'
ORIGINAL_REL = 'review/evidence/task-choice-v1-20261023/registration.json'
CONDITIONAL_REL = 'review/evidence/task-teacher-merge-bc-v1-20261028/registration.json'
CACHE_REL = 'review/evidence/task-bc-v1-20261023-run1/cache/manifest.json'
RUNTIME = {'platform': 'linux', 'python': '3.12.3', 'numpy': '2.5.4',
           'jax': '0.4.38', 'jaxlib': '0.4.38', 'optax': '0.2.4'}
WORK = {'updates': 64, 'batch_size': 256, 'min_auxiliary_updates': 32,
        'auxiliary_norm_threshold': 1e-8, 'training_source': 'original_teacher_train',
        'new_games': False, 'independent_game_evaluation': False}
EVALUATION = {
    'sources': ['teacher_dev', 'conditional_old', 'conditional_new'],
    'subsets': ['all', 'unseen_train'],
    'arms': ['baseline', 'control', 'auxiliary'],
    'aggregation': 'group_mean_within_cluster_then_equal_clusters_per_source',
    'insufficient_support': 'unknown',
    'primary_metric': 'plant_same_target_unit',
    'primary_subset': 'unseen_train',
    'primary_comparison': 'auxiliary_minus_control',
    'primary_equal_source_mean_minimum': 0.01,
    'primary_each_source_minimum': -0.02,
    'guard_metrics': ['full_top1', 'full_target_operation'],
    'guard_comparisons': ['auxiliary_minus_control', 'auxiliary_minus_baseline'],
    'guard_each_source_subset_minimum': -0.02,
    'reported_comparisons': ['control_minus_baseline', 'auxiliary_minus_control', 'auxiliary_minus_baseline'],
    'reload_score_comparison': 'exact_numpy_scores_and_first_argmax',
    'decision_scope': 'consumed_development_only_no_promotion',
}
LIMIT_KEYS = frozenset(('work_seconds', 'guard_with_cleanup_seconds', 'load_seconds',
                        'fit_seconds', 'evaluation_seconds', 'gate_seconds', 'term_seconds',
                        'cleanup_seconds', 'outer_total_seconds', 'memory_sample_every',
                        'minimum_initial_available_bytes', 'minimum_sampled_available_bytes'))
REQUIRED_PACKAGES = {
    'task_pair_memory_pipeline_v1': ('pipeline',),
    'task_pair_pidfd_guard_v1': ('gate', 'guard'),
    'task_pair_supervision_v1': ('supervisor',),
    'task_pair_wsl_training_v1': ('entry',),
    'task_pair_wsl_launch_v1': ('launch',),
}
# The pidfd gate is executed from its bounded, hash-checked snapshot, rather
# than imported into this interpreter as a normal module.
SNAPSHOT_ONLY_MODULES = frozenset(('learning.task_pair_pidfd_guard_v1.gate',))
CORE_DEPENDENCIES = frozenset(('numpy', 'jax', 'jaxlib', 'optax'))
OPTIONAL_PLUGIN_ROOTS = frozenset(('jax_plugins', 'jax_cuda12_plugin', 'jax_cuda12_pjrt'))
DEPENDENCY_ROOTS = CORE_DEPENDENCIES | OPTIONAL_PLUGIN_ROOTS | frozenset((
    'absl', 'chex', 'cloudpickle', 'etils', 'ml_dtypes', 'nvidia', 'opt_einsum',
    'pygments', 'scipy', 'toolz', 'typing_extensions'))
STAGES = frozenset(('arguments', 'registration', 'sources', 'imports', 'private_contract',
                    'runtime', 'output', 'history', 'data', 'masks', 'checkpoint', 'device', 'training',
                    'export', 'reload', 'evaluation', 'final_check', 'publish'))
CURRENT_STAGE = 'arguments'


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _stage(name):
    global CURRENT_STAGE
    _require(name in STAGES, 'Invalid entry stage')
    CURRENT_STAGE = name


def _json(raw, maximum):
    _require(type(raw) is bytes and 0 < len(raw) <= maximum, 'Invalid metadata size')
    def pairs(items):
        result = {}
        for key, value in items:
            _require(key not in result, 'Duplicate metadata key')
            result[key] = value
        return result
    def reject(value):
        raise ValueError('Invalid metadata constant')
    try:
        value = json.loads(raw.decode('utf-8-sig'), object_pairs_hook=pairs, parse_constant=reject)
    except (UnicodeError, ValueError, TypeError):
        raise ValueError('Invalid metadata') from None
    _require(type(value) is dict, 'Invalid metadata object')
    return value


def _same(value, expected):
    if type(value) is not type(expected):
        return False
    if type(expected) is dict:
        return value.keys() == expected.keys() and all(_same(value[k], expected[k]) for k in expected)
    if type(expected) is list:
        return len(value) == len(expected) and all(_same(a, b) for a, b in zip(value, expected))
    return value == expected


def _limits(value):
    _require(type(value) is dict and set(value) == LIMIT_KEYS, 'Invalid registered limits')
    _require(all(type(v) is int and v > 0 for v in value.values()), 'Invalid registered limits')
    _require(value['work_seconds'] <= 86400
             and value['work_seconds'] + 10 <= value['guard_with_cleanup_seconds'] <= 86400
             and value['gate_seconds'] <= value['work_seconds']
             and value['term_seconds'] < value['cleanup_seconds']
             and value['work_seconds'] + value['cleanup_seconds'] + 2 <= value['guard_with_cleanup_seconds']
             and value['guard_with_cleanup_seconds'] + 10 <= value['outer_total_seconds'] <= 86400
             and sum(value[k] for k in ('load_seconds', 'fit_seconds', 'evaluation_seconds'))
             <= value['work_seconds']
             and value['memory_sample_every'] == 256
             and value['minimum_initial_available_bytes'] == 4 * 1024**3
             and value['minimum_sampled_available_bytes'] == 2 * 1024**3,
             'Registered limits differ')
    return dict(value)


def _sources(value):
    _require(type(value) is dict and 1 <= len(value) <= 256, 'Invalid registered sources')
    for name, digest in value.items():
        _require(type(name) is str and 0 < len(name) <= 512
                 and (name.startswith('kaggriculture/') or name.startswith('review/evidence/'))
                 and '\\' not in name and ':' not in name and '..' not in name.split('/')
                 and type(digest) is str and HEX_SHA.fullmatch(digest), 'Invalid source identity')
    return dict(value)


def parse_registration(raw):
    value = _json(raw, MAX_SOURCE)
    _require(set(value) == {'schema', 'runtime', 'work', 'evaluation', 'limits', 'source_sha256'}
             and value['schema'] == SCHEMA and _same(value['runtime'], RUNTIME)
             and _same(value['work'], WORK) and _same(value['evaluation'], EVALUATION),
             'Training registration differs')
    return {'schema': SCHEMA, 'runtime': dict(RUNTIME), 'work': dict(WORK), 'evaluation': value['evaluation'],
            'limits': _limits(value['limits']), 'source_sha256': _sources(value['source_sha256'])}


def parse_private_contract(raw, expected_sha256):
    """Expected digest comes from a separately fixed private record, not raw."""
    _require(type(expected_sha256) is str and HEX_SHA.fullmatch(expected_sha256)
             and hashlib.sha256(raw).hexdigest() == expected_sha256, 'Private contract identity differs')
    value = _json(raw, MAX_PRIVATE)
    _require(set(value) == {'schema', 'history_receipt_sha256', 'target_device_kind',
                            'batch_rng_seed', 'runtime_identity'}
             and value['schema'] == PRIVATE_SCHEMA
             and type(value['history_receipt_sha256']) is str
             and HEX_SHA.fullmatch(value['history_receipt_sha256'])
             and type(value['target_device_kind']) is str
             and 1 <= len(value['target_device_kind']) <= 128
             and value['target_device_kind'] == value['target_device_kind'].strip()
             and all(32 <= ord(c) != 127 for c in value['target_device_kind'])
             and type(value['batch_rng_seed']) is int and 0 <= value['batch_rng_seed'] < 2**64,
             'Private contract differs')
    _private_runtime_shape(value['runtime_identity'])
    return value


def _bounded_file(path, maximum):
    """Read one bounded regular-file snapshot from the opened descriptor."""
    path = Path(path)
    info = path.lstat()
    _require(stat.S_ISREG(info.st_mode) and not stat.S_ISLNK(info.st_mode)
             and not getattr(info, 'st_file_attributes', 0) & 0x400 and info.st_size <= maximum,
             'Unexpected input file')
    flags = os.O_RDONLY | getattr(os, 'O_BINARY', 0) | getattr(os, 'O_NOFOLLOW', 0)
    fd = os.open(path, flags)
    try:
        current = os.fstat(fd)
        _require(stat.S_ISREG(current.st_mode) and current.st_size <= maximum, 'Unexpected input file')
        with os.fdopen(fd, 'rb', closefd=False) as stream:
            raw = stream.read(maximum + 1)
        _require(len(raw) == current.st_size and len(raw) <= maximum, 'Input length differs')
        return raw
    finally:
        os.close(fd)


def _git(*args):
    return subprocess.check_output(['git', '-C', str(REPO), *args], stderr=subprocess.DEVNULL, timeout=30)


def _head():
    raw = _git('rev-parse', '--verify', 'HEAD')
    _require(len(raw) == 41 and raw[-1:] == b'\n', 'Invalid Git HEAD')
    value = raw[:-1].decode('ascii')
    _require(HEX_COMMIT.fullmatch(value) is not None, 'Invalid Git HEAD')
    return value


def _blob(commit, relative):
    _require(type(commit) is str and HEX_COMMIT.fullmatch(commit)
             and type(relative) is str and relative and not relative.startswith(('/', '-'))
             and '\\' not in relative and ':' not in relative and '..' not in relative.split('/'),
             'Invalid Git blob')
    spec = commit + ':' + relative
    raw = _git('cat-file', '-s', spec)
    _require(len(raw) <= 32 and raw.strip().isdigit() and int(raw.strip()) <= MAX_SOURCE,
             'Git blob exceeds limit')
    size = int(raw.strip())
    data = _git('show', spec)
    _require(len(data) == size, 'Git blob length differs')
    return data


def registration_gate(commit):
    """Require a new committed registration before private/data access."""
    _require(type(commit) is str and HEX_COMMIT.fullmatch(commit), 'Invalid registration commit')
    _require(_head() == commit, 'Training registration is not HEAD')
    recorded = _blob(commit, REG_REL)  # Missing formal registration fails here.
    workspace = _bounded_file(REPO / REG_REL, MAX_SOURCE)
    _require(recorded == workspace, 'Registration Git and workspace differ')
    value = parse_registration(recorded)
    _require(not _git('status', '--porcelain=v1', '--untracked-files=normal'), 'Worktree is not clean')
    return recorded, value


def verify_registered_sources(commit, sources):
    _require(_head() == commit, 'Git HEAD changed')
    for relative, expected in sources.items():
        path = REPO / relative
        resolved = path.resolve(strict=True)
        _require(resolved.is_relative_to(REPO.resolve(strict=True)), 'Source outside repository')
        workspace = _bounded_file(path, MAX_SOURCE)
        blob = _blob(commit, relative)
        _require(blob == workspace and hashlib.sha256(blob).hexdigest() == expected,
                 'Registered source differs')
    _require(_head() == commit, 'Git HEAD changed')


def _module(relative, digest, label):
    raw = _bounded_file(REPO / relative, MAX_SOURCE)
    _require(hashlib.sha256(raw).hexdigest() == digest, 'Pinned source differs')
    module = types.ModuleType(label)
    module.__file__ = str(REPO / relative)
    exec(compile(raw, '<registered-project-source>', 'exec', dont_inherit=True), module.__dict__)
    return module


def prebind(sources):
    """Bind authenticated project namespaces before third-party imports."""
    _require(sources.get(SMOKE_REL) == SMOKE_SHA and EXTRA_REL in sources,
             'Required import guard is not registered')
    smoke = _module(SMOKE_REL, SMOKE_SHA, 'registered_smoke_guard')
    _require(sources.get(BOOTSTRAP_REL) == smoke.BOOTSTRAP_SHA, 'Bootstrap identity differs')
    historical = _module(BOOTSTRAP_REL, smoke.BOOTSTRAP_SHA, 'registered_initializer_manifest')
    for relative, digest in historical.INITIALIZERS.items():
        _require(sources.get('kaggriculture/' + relative) == digest,
                 'Historical initializer is not registered')
    for name, digest in smoke.NEW_INITIALIZERS.items():
        _require(sources.get('kaggriculture/learning/' + name + '/__init__.py') == digest,
                 'Smoke initializer is not registered')
    extra = _module(EXTRA_REL, sources[EXTRA_REL], 'registered_wsl_guard')
    _require(all(name in extra._PACKAGES and set(modules).issubset(extra._PACKAGES[name])
                 for name, modules in REQUIRED_PACKAGES.items()), 'WSL guard source coverage incomplete')
    required_modules = {f'learning.{name}.{module}' for name, modules in REQUIRED_PACKAGES.items()
                        for module in modules} - SNAPSHOT_ONLY_MODULES
    _require(required_modules.issubset(extra.TRAIN_MODULES), 'WSL guard module coverage incomplete')
    preimport = getattr(extra, 'PREIMPORT_SOURCE_PATHS', None)
    _require(type(preimport) is frozenset and bool(preimport)
             and extra.EXTRA_SOURCE_PATHS.issubset(preimport) and preimport.issubset(sources),
             'Preimport source set is not registered')
    extra.verify_preimport_sources(smoke, GAME_SOURCE, {name: sources[name] for name in preimport})
    root = smoke.bind(GAME_SOURCE)
    extra.verify_extra_sources(smoke, root, {name: sources[name] for name in extra.EXTRA_SOURCE_PATHS})
    return smoke, extra, root


def finish_bootstrap(binding, sources):
    """Load registered modules only after the private runtime precheck."""
    _require(type(binding) is tuple and len(binding) == 3, 'Invalid project import binding')
    smoke, extra, root = binding
    with smoke.protected_search_path(root):
        for name in sorted(extra.TRAIN_MODULES):
            importlib.import_module(name)
        pipeline = importlib.import_module('learning.task_pair_memory_pipeline_v1.pipeline')
        preflight = importlib.import_module('learning.task_pair_data_preflight_v1.run')
        history = importlib.import_module('learning.task_pair_history_binding_v1.history')
        recheck = importlib.import_module('learning.task_pair_mask_recheck_v1.run')
        v2 = importlib.import_module('learning.task_worker_ranking_v2.run')
        train = importlib.import_module('learning.task_bc_v1.train')
        model = importlib.import_module('learning.task_bc_v1.model')
        adapter = importlib.import_module('learning.task_pair_data_adapter_v1.adapter')
        scorer = importlib.import_module('learning.task_pair_numpy_scorer_v1.scorer')
        snapshot = importlib.import_module('learning.task_pair_checkpoint_v1.snapshot')
        numpy = importlib.import_module('numpy')
    extra.verify_loaded_origins(smoke, root, required_modules=extra.TRAIN_MODULES)
    required = set(extra.EXTRA_SOURCE_PATHS) | {SMOKE_REL, EXTRA_REL, CERT_REL, EARLY_REL,
               ORIGINAL_REL, CONDITIONAL_REL, CACHE_REL} | set(history._PATHS)
    # The registration cannot hash itself; its Git/worktree bytes were checked
    # separately by registration_gate.
    for name, loaded in tuple(sys.modules.items()):
        if name == 'decision' or name.startswith('decision.') or name.startswith('learning.'):
            origin = getattr(getattr(loaded, '__spec__', None), 'origin', None)
            if type(origin) is str and origin.endswith('.py'):
                required.add(Path(origin).resolve(strict=True).relative_to(REPO).as_posix())
    _require(required.issubset(sources), 'Imported source closure is not registered')
    _require(sources.get(BOOTSTRAP_REL) == smoke.BOOTSTRAP_SHA, 'Bootstrap identity differs')
    historical = _module(BOOTSTRAP_REL, smoke.BOOTSTRAP_SHA, 'registered_initializer_manifest')
    for relative, digest in historical.INITIALIZERS.items():
        _require(sources.get('kaggriculture/' + relative) == digest,
                 'Historical initializer is not registered')
    for name, digest in smoke.NEW_INITIALIZERS.items():
        _require(sources.get('kaggriculture/learning/' + name + '/__init__.py') == digest,
                 'Smoke initializer is not registered')
    return types.SimpleNamespace(smoke=smoke, extra=extra, root=root, pipeline=pipeline,
                                 preflight=preflight, history=history, recheck=recheck, v2=v2,
                                 train=train, model=model, adapter=adapter, scorer=scorer,
                                 snapshot=snapshot, np=numpy)


def bootstrap(sources):
    """Convenience for callers that already performed the runtime precheck."""
    return finish_bootstrap(prebind(sources), sources)


def _private_runtime_shape(value):
    _require(type(value) is dict and set(value) == {
        'venv_executable', 'interpreter', 'venv_prefix', 'base_prefix', 'search_path', 'allowed_origins'},
        'Invalid private runtime identity')
    def absolute(item):
        return (type(item) is str and 0 < len(item) <= 2048 and '\x00' not in item
                and Path(item).is_absolute())
    _require(all(absolute(value[key]) for key in ('venv_executable', 'interpreter', 'venv_prefix', 'base_prefix')),
             'Invalid private runtime path')
    for key in ('search_path', 'allowed_origins'):
        entries = value[key]
        _require(type(entries) is list and 1 <= len(entries) <= 32
                 and all(absolute(item) for item in entries) and len(set(entries)) == len(entries),
                 'Invalid private runtime paths')


def _runtime_identity(value):
    """Check paths fixed independently before this attempt; not a supply-chain proof."""
    _private_runtime_shape(value)
    interpreter = Path(sys.executable).resolve(strict=False)
    prefix = Path(sys.prefix).resolve(strict=False)
    base = Path(sys.base_prefix).resolve(strict=False)
    search = [str(Path(item).resolve(strict=False)) for item in sys.path]
    origins = [Path(item).resolve(strict=False) for item in value['allowed_origins']]
    _require(sys.executable == value['venv_executable']
             and Path(sys.executable).parent.resolve(strict=False).is_relative_to(prefix)
             and str(interpreter) == value['interpreter'] and str(prefix) == value['venv_prefix']
             and str(base) == value['base_prefix'] and prefix != base
             and search == value['search_path'] and len(set(search)) == len(search),
             'Registered interpreter or search path differs')
    _require(all(origin != prefix and origin.is_relative_to(prefix) and origin.is_dir()
                 and str(origin) in search for origin in origins), 'Invalid permitted dependency roots')
    _require(all(Path(item).is_relative_to(base)
                 or any(Path(item).is_relative_to(origin) for origin in origins) for item in search),
             'Unapproved interpreter search path')
    return tuple(origins)


def child_interpreter(private):
    """Return the bound venv argv[0], not the resolved base executable."""
    _runtime_identity(private['runtime_identity'])
    return private['runtime_identity']['venv_executable']


def _allowed_location(value, roots):
    if type(value) is not str or not value or '\x00' in value or not Path(value).is_absolute():
        return False
    path = Path(value).resolve(strict=False)
    return any(path.is_relative_to(root) for root in roots)


def _spec_identity(spec, roots):
    """Check source or namespace paths, including every namespace location."""
    _require(spec is not None, 'Dependency module specification is absent')
    origin = getattr(spec, 'origin', None)
    locations = getattr(spec, 'submodule_search_locations', None)
    if locations is not None:
        locations = tuple(locations)
        _require(1 <= len(locations) <= 32 and len(set(locations)) == len(locations)
                 and all(_allowed_location(item, roots) for item in locations),
                 'Dependency package search location differs')
    if origin is None:
        _require(locations is not None, 'Dependency has no source or namespace')
    else:
        _require(_allowed_location(origin, roots), 'Dependency source differs')
    return origin, locations


def _candidate_specs(roots, *, loaded=None, path_finder=None, finder=None):
    """Precheck every declared dependency root before its later import."""
    loaded = sys.modules if loaded is None else loaded
    path_finder = importlib.machinery.PathFinder.find_spec if path_finder is None else path_finder
    finder = importlib.util.find_spec if finder is None else finder
    _require(not any(name.partition('.')[0] in CORE_DEPENDENCIES | OPTIONAL_PLUGIN_ROOTS
                     for name in loaded),
             'Core dependency was loaded before its runtime check')
    for name in sorted(DEPENDENCY_ROOTS):
        direct, resolved = path_finder(name, sys.path), finder(name)
        if name not in CORE_DEPENDENCIES and direct is None and resolved is None:
            continue
        _require(_spec_identity(direct, roots) == _spec_identity(resolved, roots),
                 'Dependency finder result differs')


def _loaded_dependencies(roots, *, loaded=None, require_core=False):
    """Verify currently loaded dependency modules, including native submodules."""
    loaded = dict(sys.modules) if loaded is None else dict(loaded)
    if require_core:
        _require(all(loaded.get(name) is not None for name in CORE_DEPENDENCIES),
                 'Core dependencies are missing')
    for name, module in loaded.items():
        if name.partition('.')[0] not in DEPENDENCY_ROOTS:
            continue
        _require(module is not None, 'Dependency module is missing')
        spec = getattr(module, '__spec__', None)
        if spec is None:
            # jaxlib native extensions expose a few generated child modules
            # without specs; bind them to their actual extension parent.
            parent_name, _, child = name.rpartition('.')
            parent = loaded.get(parent_name)
            parent_spec = getattr(parent, '__spec__', None)
            parent_origin = getattr(parent_spec, 'origin', None)
            _require(name.startswith('jaxlib.') and parent is not None
                     and getattr(parent, child, None) is module
                     and type(parent_origin) is str
                     and any(parent_origin.endswith(suffix) for suffix in importlib.machinery.EXTENSION_SUFFIXES)
                     and _allowed_location(parent_origin, roots)
                     and getattr(module, '__file__', None) is None
                     and getattr(module, '__path__', None) is None,
                     'Unbound dependency module')
            continue
        origin, locations = _spec_identity(spec, roots)
        _require(getattr(module, '__loader__', None) is getattr(spec, 'loader', None),
                 'Dependency loader differs')
        if origin is None:
            _require(getattr(module, '__file__', None) is None, 'Namespace file differs')
        else:
            _require(getattr(module, '__file__', None) == origin, 'Dependency file differs')
        if locations is not None:
            _require(tuple(getattr(module, '__path__', ())) == locations,
                     'Dependency package path differs')


def _runtime_versions():
    actual = {'platform': sys.platform, 'python': '.'.join(map(str, sys.version_info[:3]))}
    for name in ('numpy', 'jax', 'jaxlib', 'optax'):
        actual[name] = importlib.metadata.version(name)
    _require(_same(actual, RUNTIME), 'Runtime version differs')


def _runtime_pre(private):
    roots = _runtime_identity(private['runtime_identity'])
    _loaded_dependencies(roots)
    _candidate_specs(roots)
    _runtime_versions()


def _runtime_after(private, *, require_core=False):
    roots = _runtime_identity(private['runtime_identity'])
    _loaded_dependencies(roots, require_core=require_core)
    _runtime_versions()


def _load_jax(private):
    _runtime_after(private)
    jax = importlib.import_module('jax')
    importlib.import_module('jaxlib')
    importlib.import_module('optax')
    _runtime_after(private, require_core=True)
    return jax


def _mem_available(raw):
    _require(type(raw) is bytes and 0 < len(raw) <= MAX_PRIVATE, 'Invalid memory sample')
    matches = re.findall(rb'(?m)^MemAvailable:[\t ]+([0-9]{1,15})[\t ]+kB[\t ]*\r?$', raw)
    _require(len(matches) == 1 and int(matches[0]) > 0, 'Memory availability unavailable')
    return int(matches[0]) * 1024


def _read_meminfo():
    with open('/proc/meminfo', 'rb') as stream:
        return stream.read(MAX_PRIVATE + 1)


class Budget:
    """Sampled deadlines and available-memory thresholds, not hard limits."""
    def __init__(self, limits, *, clock=time.monotonic, read_memory=_read_meminfo):
        self.limits = _limits(limits)
        self.clock, self.read_memory = clock, read_memory
        self.deadline = clock() + limits['work_seconds']
        self.phase_deadline = self.deadline
        self.calls = 0
        _require(_mem_available(read_memory()) >= limits['minimum_initial_available_bytes'],
                 'Insufficient initial available memory')

    def phase(self, name):
        _require(name in ('load', 'fit', 'evaluation'), 'Invalid resource phase')
        now = self.clock()
        # A phase transition must not erase a deadline already exceeded by a
        # long preceding call. The checks remain sampled, not hard interrupts.
        if now >= min(self.deadline, self.phase_deadline):
            raise TimeoutError('Sampled work deadline exceeded')
        self.phase_deadline = min(self.deadline, now + self.limits[name + '_seconds'])
        self.check()

    def check(self):
        if self.clock() >= min(self.deadline, self.phase_deadline):
            raise TimeoutError('Sampled work deadline exceeded')
        self.calls += 1
        if self.calls % self.limits['memory_sample_every'] == 0:
            _require(_mem_available(self.read_memory()) >= self.limits['minimum_sampled_available_bytes'],
                     'Insufficient sampled available memory')


def _unique_clusters(view, expected, np):
    _require(type(expected) in (tuple, list) and all(type(x) is int for x in expected)
             and len(set(expected)) == len(expected), 'Invalid source roles')
    _require(set(map(int, np.unique(view.private_clusters))) == set(expected), 'Source roles differ')


def bind_views(views, roles, original, old, new, np):
    """Bind ordered loaded views to independently authenticated source roles."""
    _require(type(views) is dict and set(views) == {'train', 'teacher_dev', 'conditional_dev'}
             and type(roles) is dict and set(roles) == set(views)
             and all(type(views[name]) in (tuple, list) for name in views)
             and len(views['train']) == len(views['teacher_dev']) == 1
             and len(views['conditional_dev']) == 2, 'Source views differ')
    train = tuple(original['train_seeds'])
    teacher = tuple(original['validation_seeds'])
    old, new = tuple(old), tuple(new)
    _require(len(train) == 8 and len(teacher) == len(old) == len(new) == 2
             and len(set(train + teacher + old + new)) == 14
             and tuple(roles['train']) == train and tuple(roles['teacher_dev']) == teacher
             and tuple(roles['conditional_dev']) == old + new, 'Registered role order differs')
    expected = {'train': train, 'teacher_dev': teacher, 'conditional_old': old, 'conditional_new': new}
    ordered = {'train': views['train'][0], 'teacher_dev': views['teacher_dev'][0],
               'conditional_old': views['conditional_dev'][0], 'conditional_new': views['conditional_dev'][1]}
    for name in expected:
        _unique_clusters(ordered[name], expected[name], np)
    return ordered, expected


def _private_root(game_root):
    root = Path(game_root).resolve(strict=True)
    artifacts = (root / 'artifacts').resolve(strict=True)
    _require(artifacts == root / 'artifacts' and artifacts.is_dir()
             and not artifacts.is_relative_to(REPO.resolve(strict=True)), 'Invalid artifacts root')
    return root, artifacts


def _private_file(path, artifacts, maximum):
    path = Path(path)
    _require(not path.is_symlink() and path.resolve(strict=True).is_relative_to(artifacts),
             'Private file outside artifacts')
    return _bounded_file(path, maximum)


def _mask_directory(root, recheck):
    directory = root / 'artifacts' / recheck.NEW_OUTPUT_NAME
    _require(not directory.is_symlink() and directory.resolve(strict=True) == directory
             and directory.is_dir(), 'Invalid rechecked mask directory')
    return directory


def _write_new(path, raw):
    _require(type(raw) is bytes and len(raw) <= 2 * 1024 * 1024, 'Invalid private output')
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, 'O_BINARY', 0) | getattr(os, 'O_NOFOLLOW', 0)
    fd = os.open(path, flags, 0o600)
    try:
        with os.fdopen(fd, 'wb', closefd=False) as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        os.close(fd)


def _write_json(path, value):
    raw = (json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(',', ':'),
                      allow_nan=False) + '\n').encode('utf-8')
    _write_new(path, raw)


def _history(modules, head, receipt_sha256):
    history, recheck = modules.history, modules.recheck
    commits = (recheck.OLD_COMMIT, recheck.PUBLISHED_COMMIT,
               history.MASK_REGISTRATION_COMMIT, history.MASK_RESULT_COMMIT)
    public = {path: recheck.git_blob(commit, path, allowed=frozenset(history._PATHS))
              for commit, path in zip(commits, history._PATHS, strict=True)}
    bound = history.bind_history(expected_head=head, preflight_registration_commit=commits[0],
        preflight_result_commit=commits[1], mask_registration_commit=commits[2],
        mask_result_commit=commits[3], public_bytes=public, expected_receipt_sha256=receipt_sha256)
    certificate = recheck.git_blob(history.MASK_RESULT_COMMIT, CERT_REL, allowed=frozenset((CERT_REL,)))
    _require(certificate == recheck.git_blob(head, CERT_REL, allowed=frozenset((CERT_REL,)))
             == _bounded_file(REPO / CERT_REL, MAX_SOURCE), 'Historical certificate differs')
    cert = _json(certificate, MAX_SOURCE)
    _require(cert.get('schema') == 'task-pair-mask-recheck-certificate-v1'
             and cert.get('registration_commit') == commits[2]
             and type(cert.get('operator_observed_exit_code')) is int
             and cert['operator_observed_exit_code'] == 0 and cert.get('attempts') == 1
             and cert.get('result_sha256') == hashlib.sha256(public[history.MASK_RESULT_REL]).hexdigest(),
             'Historical process evidence differs')
    old = modules.v2.parse_registration(public[recheck.OLD_REG_REL])
    return bound, old


def _baseline_bytes(root, modules, expected_schema_sha256):
    selected = _json(_bounded_file(REPO / EARLY_REL, MAX_SOURCE), MAX_SOURCE)
    _require(type(selected.get('checkpoint_file')) is str
             and type(selected.get('checkpoint_sha256')) is str
             and HEX_SHA.fullmatch(selected['checkpoint_sha256']), 'Early checkpoint identity differs')
    relative = Path(selected['checkpoint_file'])
    _require(not relative.is_absolute() and '..' not in relative.parts
             and '\\' not in selected['checkpoint_file'], 'Invalid early checkpoint name')
    path = root / 'artifacts/task-bc-v1-20261023-run1/model' / relative
    _require(not path.is_symlink() and path.resolve(strict=True).is_relative_to(
        (root / 'artifacts').resolve(strict=True)), 'Early checkpoint outside artifacts')
    raw = _bounded_file(path, modules.snapshot.MAX_ARCHIVE_BYTES)
    baseline = modules.snapshot.load_checkpoint(raw, expected_sha256=selected['checkpoint_sha256'],
                                                 expected_schema_sha256=expected_schema_sha256)
    return raw, baseline


def _verify_reloaded(model, expected, np):
    """Compare an actual TaskNetwork(path, sha) against authenticated bytes."""
    _require(type(model.params) is dict and set(model.params) == set(expected.params),
             'Reloaded model parameters differ')
    _require(np.array_equal(model.mean, expected.mean.astype(np.float64))
             and np.array_equal(model.scale, expected.scale.astype(np.float64))
             and all(np.array_equal(model.params[k], expected.params[k].astype(np.float64))
                     for k in expected.params), 'Reloaded model values differ')
    metadata = _json(expected.schema_json.encode('utf-8'), MAX_PRIVATE)
    _require(all(model.metadata.get(k) == value for k, value in metadata.items()),
             'Reloaded model schema differs')


def _check_direct_scores(models, sources, modules, schema, check):
    """Require exact per-group deployed CPU scores and first-max choices."""
    np = modules.np
    for model in models.values():
        for source in sources.values():
            view = source.view
            blocks = modules.scorer.score_blocks(view, model, expected_schema_sha256=schema, check=check)
            covered = 0
            for block in blocks:
                _require(block.start == covered and block.stop > block.start, 'Score block order differs')
                origin = int(view.offsets[block.start])
                for group in range(block.start, block.stop):
                    check()
                    lo, hi = map(int, view.offsets[group:group + 2])
                    direct = modules.model.TaskNetwork.score(model, view.shared[group], view.candidates[lo:hi])
                    copied = block.logits[lo - origin:hi - origin]
                    _require(np.array_equal(direct, copied)
                             and int(np.argmax(direct)) == int(block.choices[group - block.start]),
                             'Reloaded scorer output differs')
                covered = block.stop
            _require(covered == len(view.shared), 'Incomplete direct score comparison')


def _verify_data_after(root, modules, frozen, conditional, check):
    manifest = frozen.read_json(REPO / CACHE_REL)
    cache = root / 'artifacts/task-bc-v1-20261023-run1/cache'
    _require(_bounded_file(cache / 'manifest.json', MAX_SOURCE) == _bounded_file(REPO / CACHE_REL, MAX_SOURCE),
             'Local cache manifest changed')
    schema = frozen.local_file(root, 'artifacts/task-bc-v1-20261023-run1/cache/schema.json')
    _require(modules.preflight.stream_sha(schema, check) == manifest['schema_sha256'],
             'Original source schema changed')
    for split in ('train', 'validation'):
        for name, info in manifest['arrays'][split].items():
            relative = Path('artifacts/task-bc-v1-20261023-run1/cache') / info['path']
            path = frozen.local_file(root, relative)
            _require(modules.preflight.stream_sha(path, check) == info['sha256'],
                     'Original source changed during training')
    for name in ('old', 'new'):
        schema = frozen.local_file(root, conditional['schema_path'][name])
        _require(modules.preflight.stream_sha(schema, check) == conditional['schema_sha256'],
                 'Conditional source schema changed')
        path = frozen.local_file(root, conditional['archive_path'][name])
        _require(modules.preflight.stream_sha(path, check) == conditional['archive_sha256'][name],
                 'Conditional source changed during training')


def _check_masks_equal(first, second, np):
    for role in ('teacher_dev', 'conditional_dev'):
        _require(len(first[role]) == len(second[role]), 'Mask shards differ')
        for left, right in zip(first[role], second[role], strict=True):
            _require(set(left) == set(right) and all(np.array_equal(left[k], right[k]) for k in left),
                     'Mask snapshot changed during training')


def assess_development(metrics):
    """Apply the fixed consumed-development screen, separate from execution.

    No unknown required value is promoted to a pass or failure. Exact values
    and this result remain private; this is not a game or promotion decision.
    """
    _require(type(metrics) is dict and set(metrics) == set(EVALUATION['arms']), 'Metric arms differ')
    unknown = False

    def value(arm, source, subset, metric):
        nonlocal unknown
        try:
            item = metrics[arm][source][subset][metric]
        except (KeyError, TypeError):
            raise ValueError('Required development metric is missing') from None
        _require(type(item) is dict and set(item) == {'support', 'value'}, 'Invalid development metric')
        if item['support'] == 'unknown':
            _require(item['value'] is None, 'Invalid unknown metric')
            unknown = True
            return None
        number = item['value']
        _require(item['support'] == 'supported' and type(number) in (int, float)
                 and math.isfinite(number) and 0 <= number <= 1, 'Invalid supported metric')
        return float(number)

    primary = []
    control_vs_baseline = []
    auxiliary_vs_baseline = []
    guards = []
    for source in EVALUATION['sources']:
        auxiliary = value('auxiliary', source, 'unseen_train', EVALUATION['primary_metric'])
        control = value('control', source, 'unseen_train', EVALUATION['primary_metric'])
        baseline = value('baseline', source, 'unseen_train', EVALUATION['primary_metric'])
        if auxiliary is not None and control is not None:
            primary.append(auxiliary - control)
        if control is not None and baseline is not None:
            control_vs_baseline.append(control - baseline)
        if auxiliary is not None and baseline is not None:
            auxiliary_vs_baseline.append(auxiliary - baseline)
        for subset in EVALUATION['subsets']:
            for metric in EVALUATION['guard_metrics']:
                auxiliary = value('auxiliary', source, subset, metric)
                for reference in ('control', 'baseline'):
                    other = value(reference, source, subset, metric)
                    if auxiliary is not None and other is not None:
                        guards.append(auxiliary - other)
    if unknown:
        return {'status': 'unknown', 'primary_equal_source_mean_difference': None,
                'primary_minimum_source_difference': None, 'minimum_guard_difference': None,
                'control_minus_baseline_mean': None, 'auxiliary_minus_baseline_mean': None}
    _require(len(primary) == len(EVALUATION['sources'])
             and len(control_vs_baseline) == len(primary) and len(auxiliary_vs_baseline) == len(primary)
             and len(guards) == len(EVALUATION['sources']) * len(EVALUATION['subsets'])
             * len(EVALUATION['guard_metrics']) * 2, 'Incomplete development metrics')
    mean = math.fsum(primary) / len(primary)
    minimum_primary, minimum_guard = min(primary), min(guards)
    passed = (mean >= EVALUATION['primary_equal_source_mean_minimum']
              and minimum_primary >= EVALUATION['primary_each_source_minimum']
              and minimum_guard >= EVALUATION['guard_each_source_subset_minimum'])
    return {'status': 'passed' if passed else 'failed', 'primary_equal_source_mean_difference': mean,
            'primary_minimum_source_difference': minimum_primary, 'minimum_guard_difference': minimum_guard,
            'control_minus_baseline_mean': math.fsum(control_vs_baseline) / len(control_vs_baseline),
            'auxiliary_minus_baseline_mean': math.fsum(auxiliary_vs_baseline) / len(auxiliary_vs_baseline)}


def run_registered(head, registration, private, root, output, modules, budget, token, *, private_check):
    """Run the actual registered phases after entry/guard preconditions.

    This function reads real data only when called by a future authenticated
    entry. Its private inputs and outputs must never be printed or published.
    """
    np, recheck = modules.np, modules.recheck
    _stage('history')
    bound, old_registration = _history(modules, head, private['history_receipt_sha256'])
    sources = registration['source_sha256']
    _require(all(sources.get(name) == digest for name, digest in old_registration['source_sha256'].items()),
             'Historical data source identities are not registered')
    frozen = recheck.load_frozen_v1_bounded(modules.v2)
    original = frozen.read_json(REPO / ORIGINAL_REL)
    conditional = frozen.read_json(REPO / CONDITIONAL_REL)
    old_roles, new_roles, _ = frozen.source_roles(original, conditional)
    default = _bounded_file(root / 'main.py', MAX_SOURCE)
    _require(hashlib.sha256(default).hexdigest() == sources.get('kaggriculture/main.py'),
             'Default Bot differs')

    _stage('data')
    budget.phase('load')
    views, roles = modules.preflight.load_data(root, frozen, budget.check)
    ordered, expected = bind_views(views, roles, original, old_roles, new_roles, np)
    schema_manifest = frozen.read_json(REPO / CACHE_REL)
    schema = schema_manifest['numeric_feature_schema_sha256']
    _require(type(schema) is str and HEX_SHA.fullmatch(schema)
             and all(view.schema_digest == schema for view in ordered.values()), 'Source schema differs')

    _stage('masks')
    mask_dir = _mask_directory(root, recheck)
    receipt = recheck.read_local_mask(mask_dir, 'binding.local.json', MAX_PRIVATE)
    manifest = recheck.read_local_mask(mask_dir, 'masks.local.json', MAX_PRIVATE)
    lengths = {'teacher_dev': [len(ordered['teacher_dev'].labels)],
               'conditional_dev': [len(ordered['conditional_old'].labels), len(ordered['conditional_new'].labels)]}
    read_mask = lambda name, maximum: recheck.read_local_mask(mask_dir, name, maximum)
    masks = bound.load_masks(receipt, manifest, read_mask, role_lengths=lengths)

    _stage('checkpoint')
    raw, baseline = _baseline_bytes(root, modules, schema)
    train_data = modules.adapter.adapt_r8(ordered['train'], role='train',
        expected_clusters=list(expected['train']), expected_schema_sha256=schema)
    _, class_weights = modules.train.class_weights(train_data['selected_ops'])
    budget.check()

    _stage('device')
    os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
    jax = _load_jax(private)
    devices = jax.devices('gpu')
    _runtime_after(private, require_core=True)
    _require(len(devices) == 1 and devices[0].platform == 'gpu'
             and type(devices[0].device_kind) is str
             and devices[0].device_kind == private['target_device_kind'], 'Registered GPU is unavailable')
    device = devices[0]

    _stage('training')
    budget.phase('fit')
    with jax.default_device(device):
        exports = modules.pipeline.train_and_export(
            train_data, baseline, class_weights, expected_schema_sha256=schema, device=device,
            rng_seed=private['batch_rng_seed'], updates=64, batch_size=256,
            training_source='original_teacher_train', registration_commit=head, check=budget.check)
    _require(exports.completed_updates == 64 and exports.auxiliary_nonzero_updates >= 32,
             'Paired update gate differs')
    _runtime_after(private, require_core=True)

    _stage('export')
    encoded = {'baseline': raw, 'control': exports.control_bytes, 'auxiliary': exports.auxiliary_bytes}
    for name, data in encoded.items():
        _write_new(output / (name + '.npz'), data)
    budget.check()

    _stage('reload')
    models = {}
    for name, data in encoded.items():
        digest = hashlib.sha256(data).hexdigest()
        model = modules.model.TaskNetwork(output / (name + '.npz'), digest)
        expected_snapshot = (baseline if name == 'baseline' else
            modules.snapshot.load_checkpoint(data, expected_sha256=digest, expected_schema_sha256=schema))
        _verify_reloaded(model, expected_snapshot, np)
        models[name] = model
    dev = {
        'teacher_dev': modules.pipeline.DevSource(ordered['teacher_dev'], masks['teacher_dev'][0], expected['teacher_dev']),
        'conditional_old': modules.pipeline.DevSource(ordered['conditional_old'], masks['conditional_dev'][0], expected['conditional_old']),
        'conditional_new': modules.pipeline.DevSource(ordered['conditional_new'], masks['conditional_dev'][1], expected['conditional_new']),
    }

    _stage('evaluation')
    budget.phase('evaluation')
    _check_direct_scores(models, dev, modules, schema, budget.check)
    result = modules.pipeline.evaluate_models(models, dev, expected_schema_sha256=schema, check=budget.check)
    assessment = assess_development(result.metrics)
    budget.check()

    _stage('final_check')
    _require(callable(private_check), 'Missing private contract check')
    _verify_data_after(root, modules, frozen, conditional, budget.check)
    later = bound.load_masks(recheck.read_local_mask(mask_dir, 'binding.local.json', MAX_PRIVATE),
        recheck.read_local_mask(mask_dir, 'masks.local.json', MAX_PRIVATE), read_mask, role_lengths=lengths)
    _check_masks_equal(masks, later, np)
    _require(hashlib.sha256(_bounded_file(root / 'main.py', MAX_SOURCE)).hexdigest()
             == sources.get('kaggriculture/main.py'), 'Default Bot changed')
    for name, data in encoded.items():
        _require(_bounded_file(output / (name + '.npz'), modules.snapshot.MAX_ARCHIVE_BYTES) == data,
                 'Private model output changed')
    verify_registered_sources(head, sources)
    _require(registration_gate(head)[1] == registration, 'Training registration changed')
    private_check()
    _runtime_after(private, require_core=True)
    modules.extra.verify_loaded_origins(modules.smoke, modules.root,
                                        required_modules=modules.extra.TRAIN_MODULES)
    budget.check()

    _stage('publish')
    _write_json(output / 'evaluation.local.json', {'schema': 'task-pair-wsl-private-evaluation-v1',
        'metrics': result.metrics, 'development_screen': assessment,
        'new_games': False, 'independent_game_evaluation': False})
    _write_json(output / 'child-result.local.json', {'schema': 'task-pair-wsl-child-result-v1',
        'completed_updates': exports.completed_updates,
        'auxiliary_nonzero_updates': exports.auxiliary_nonzero_updates,
        'development_screen_status': assessment['status'],
        'supervisor_confirmation': False})
    budget.check()
    return {'schema': 'task-pair-training-supervisor-receipt-v1', 'token': token, 'passed': True,
            'completed_updates': exports.completed_updates,
            'auxiliary_nonzero_updates': exports.auxiliary_nonzero_updates,
            'both_arms_changed': True, 'arms_differ': True}


@contextmanager
def _quiet_native_output():
    """Do not expose native diagnostics or dependency exception contents."""
    sys.stdout.flush()
    sys.stderr.flush()
    saved_out = saved_err = sink = None
    try:
        saved_out, saved_err = os.dup(1), os.dup(2)
        sink = os.open(os.devnull, os.O_WRONLY)
        os.dup2(sink, 1)
        os.dup2(sink, 2)
        yield
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        if saved_out is not None:
            os.dup2(saved_out, 1)
        if saved_err is not None:
            os.dup2(saved_err, 2)
        for fd in (saved_out, saved_err, sink):
            if fd is not None:
                os.close(fd)


class QuietParser(argparse.ArgumentParser):
    def error(self, message):
        raise ValueError('Invalid training arguments')


def main(argv=None):
    global CURRENT_STAGE
    CURRENT_STAGE = 'arguments'
    output = None
    owned_output = False
    try:
        parser = QuietParser(add_help=False)
        parser.add_argument('--registration-commit', required=True)
        parser.add_argument('--game-root', required=True)
        parser.add_argument('--private-contract', required=True)
        parser.add_argument('--expected-private-contract-sha256', required=True)
        parser.add_argument('--receipt-token', required=True)
        args = parser.parse_args(argv)
        _require(type(args.receipt_token) is str and HEX_TOKEN.fullmatch(args.receipt_token),
                 'Invalid supervisor token')
        _require(sys.flags.isolated and sys.flags.dont_write_bytecode and __debug__,
                 'Expected isolated non-optimized Python without bytecode writes')
        with _quiet_native_output():
            _stage('registration')
            _, registration = registration_gate(args.registration_commit)
            _stage('sources')
            verify_registered_sources(args.registration_commit, registration['source_sha256'])
            _stage('imports')
            binding = prebind(registration['source_sha256'])
            _stage('private_contract')
            root, artifacts = _private_root(args.game_root)
            contract_raw = _private_file(args.private_contract, artifacts, MAX_PRIVATE)
            private = parse_private_contract(contract_raw, args.expected_private_contract_sha256)
            _stage('runtime')
            os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
            _runtime_pre(private)
            _stage('imports')
            modules = finish_bootstrap(binding, registration['source_sha256'])
            _runtime_after(private)
            _stage('output')
            output = artifacts / OUTPUT_NAME
            _require(output.resolve(strict=False) == output, 'Invalid private output path')
            output.mkdir(mode=0o700, exist_ok=False)
            owned_output = True
            _write_json(output / 'started.local.json', {'schema': 'task-pair-wsl-training-start-v1',
                                                         'attempt': 1})
            budget = Budget(registration['limits'])
            def private_check():
                later = _private_file(args.private_contract, artifacts, MAX_PRIVATE)
                _require(parse_private_contract(later, args.expected_private_contract_sha256) == private,
                         'Private contract changed')
            result = run_registered(args.registration_commit, registration, private, root,
                                    output, modules, budget, args.receipt_token, private_check=private_check)
    except BaseException as exc:
        if owned_output and output is not None and output.is_dir():
            try:
                _write_json(output / 'failure.local.json', {'schema': 'task-pair-wsl-training-failure-v1',
                    'stage': CURRENT_STAGE, 'error_type': type(exc).__name__})
            except BaseException:
                pass
        result = {'passed': False, 'stage': CURRENT_STAGE,
                  'error_type': type(exc).__name__ if type(exc).__name__ in
                  {'ValueError', 'OSError', 'TimeoutError', 'ImportError', 'MemoryError',
                   'KeyboardInterrupt', 'SystemExit'} else 'OtherError'}
    print(json.dumps(result, sort_keys=True, separators=(',', ':')), flush=True)
    return 0 if result.get('passed') is True else 1


if __name__ == '__main__':
    sys.exit(main())
