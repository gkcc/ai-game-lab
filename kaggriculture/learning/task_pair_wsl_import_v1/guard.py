"""Additional project-import checks after the pinned synthetic smoke guard.

Load this file from a caller-authenticated bounded source snapshot. Before any
of the extra packages are imported, call ``verify_extra_sources`` with the
already bound smoke guard and SHA values from the future training registration.
This module does not authenticate that registration, the smoke guard itself,
third-party dependencies, or data. It is not a filesystem or import lock.
"""

import hashlib
import importlib.machinery
import os
from pathlib import Path
import re
import sys


_SHA = re.compile(r'[0-9a-f]{64}\Z')
_BASE = 'kaggriculture/learning/'

# This fixed list covers the real source loader, current paired-training
# components and this additional guard. The two historical worker entries may
# be loaded from pinned byte snapshots rather than imported as packages.
_PACKAGES = {
    'task_worker_ranking_v1': ('run',),
    'task_worker_ranking_v2': ('run',),
    'task_worker_geometry_diagnostic_v1': ('archive_io',),
    'task_pair_mask_recheck_v1': ('run',),
    'task_pair_checkpoint_v1': ('snapshot',),
    'task_pair_data_adapter_v1': ('adapter',),
    'task_pair_eval_adapter_v1': ('adapter',),
    'task_pair_host_export_v1': ('bridge',),
    'task_pair_orchestration_v1': ('core',),
    'task_pair_wsl_import_v1': ('guard',),
}
_NAMESPACE = 'task_teacher_choice_bc_v1'
_NAMESPACE_MODULES = ('numeric',)

EXTRA_SOURCE_PATHS = frozenset(
    [_BASE + package + '/__init__.py' for package in _PACKAGES]
    + [_BASE + package + '/' + module + '.py'
       for package, modules in _PACKAGES.items() for module in modules]
    + [_BASE + _NAMESPACE + '/' + module + '.py' for module in _NAMESPACE_MODULES]
)

LOAD_DATA_MODULES = frozenset((
    'decision.api', 'learning.bc_crop.features', 'learning.task_choice_v1.features',
    'learning.task_bc_v1.model', 'learning.task_bc_v1.train',
    'learning.task_pair_data_preflight_v1.core', 'learning.task_pair_data_preflight_v1.run',
    'learning.task_teacher_choice_bc_v1.numeric',
    'learning.task_worker_geometry_diagnostic_v1.archive_io',
))
TRAIN_MODULES = LOAD_DATA_MODULES | frozenset((
    'learning.task_pair_training_v1.batching', 'learning.task_pair_training_v1.binding',
    'learning.task_pair_training_v1.evaluation', 'learning.task_pair_training_v1.fit_core',
    'learning.task_pair_training_v1.loss', 'learning.task_pair_training_v1.masks',
    'learning.task_pair_training_v1.objective', 'learning.task_pair_training_v1.packing',
    'learning.task_pair_training_v1.targets', 'learning.task_pair_checkpoint_v1.snapshot',
    'learning.task_pair_data_adapter_v1.adapter', 'learning.task_pair_eval_adapter_v1.adapter',
    'learning.task_pair_host_export_v1.bridge', 'learning.task_pair_orchestration_v1.core',
))


def _present(path):
    # Unlike exists(), this also detects an already-present dangling symlink.
    return os.path.lexists(path)


def _folder(root, package):
    folder = root / 'learning' / package
    if not folder.is_dir() or folder.resolve(strict=True) != folder:
        raise ValueError('Unexpected project package')
    for suffix in ('.py', '.pyc', *importlib.machinery.EXTENSION_SUFFIXES):
        if _present(folder.parent / (package + suffix)):
            raise ValueError('Unexpected project package shadow')
    return folder


def _module(root, package, name, expected, smoke_guard):
    folder = root / 'learning' / package
    source = folder / (name + '.py')
    relative = _BASE + package + '/' + name + '.py'
    if hashlib.sha256(smoke_guard.bounded_snapshot(source)).hexdigest() != expected[relative]:
        raise ValueError('Registered extra source differs')
    for suffix in ('.pyc', *importlib.machinery.EXTENSION_SUFFIXES):
        if _present(folder / (name + suffix)):
            raise ValueError('Unexpected project module shadow')
    other = folder / name
    for suffix in ('.py', '.pyc', *importlib.machinery.EXTENSION_SUFFIXES):
        if _present(other / ('__init__' + suffix)):
            raise ValueError('Unexpected project module package shadow')


def verify_extra_sources(smoke_guard, root, expected_sha256):
    """Check additional registered packages before importing any of them.

    The smoke guard must already be authenticated and bound by the caller.
    ``expected_sha256`` must be the exact subset from an independently
    authenticated training registration, not hashes derived from these files.
    """
    root = Path(root).resolve(strict=True)
    if (type(expected_sha256) is not dict or set(expected_sha256) != EXTRA_SOURCE_PATHS
            or any(type(value) is not str or _SHA.fullmatch(value) is None
                   for value in expected_sha256.values())):
        raise ValueError('Invalid registered extra source identities')
    extra_names = set(_PACKAGES) | {_NAMESPACE}
    if any(name == 'learning.' + package or name.startswith('learning.' + package + '.')
           for name in sys.modules for package in extra_names):
        raise ValueError('Extra project packages were already loaded')
    # The existing guard verifies its original package coverage and bound
    # learning/decision origins before the extra namespace is imported.
    smoke_guard.verify_origins(root)
    for package, modules in _PACKAGES.items():
        folder = _folder(root, package)
        for suffix in ('.pyc', *importlib.machinery.EXTENSION_SUFFIXES):
            if _present(folder / ('__init__' + suffix)):
                raise ValueError('Unexpected project initializer shadow')
        relative = _BASE + package + '/__init__.py'
        if hashlib.sha256(smoke_guard.bounded_snapshot(folder / '__init__.py')).hexdigest() != expected_sha256[relative]:
            raise ValueError('Registered project initializer differs')
        for module in modules:
            _module(root, package, module, expected_sha256, smoke_guard)
    folder = _folder(root, _NAMESPACE)
    for suffix in ('.py', '.pyc', *importlib.machinery.EXTENSION_SUFFIXES):
        if _present(folder / ('__init__' + suffix)):
            raise ValueError('Unexpected namespace initializer')
    for module in _NAMESPACE_MODULES:
        _module(root, _NAMESPACE, module, expected_sha256, smoke_guard)


def verify_loaded_origins(smoke_guard, root, *, required_modules):
    """Check every loaded project origin, allowing the one real namespace.

    The historical smoke guard checks only source-file loaders, but the real
    teacher-choice parent is a PEP 420 namespace. This equivalent post-import
    check admits that one package only at its unique bound project location.
    Call after ``verify_extra_sources`` and after each relevant import stage.
    """
    root = Path(root).resolve(strict=True)
    if type(required_modules) is not frozenset or required_modules not in (LOAD_DATA_MODULES, TRAIN_MODULES):
        raise ValueError('Invalid required project module set')
    smoke_guard.verify_search_path(root)
    learning = sys.modules.get('learning')
    if (learning is None or getattr(learning, '__path__', None) != [str(root / 'learning')]
            or getattr(getattr(learning, '__spec__', None), 'submodule_search_locations', None)
            != [str(root / 'learning')]):
        raise ValueError('Project namespace binding changed')
    namespace_name = 'learning.' + _NAMESPACE
    for name, module in tuple(sys.modules.items()):
        if name != 'decision' and not name.startswith('learning.') and not name.startswith('decision.'):
            continue
        spec = getattr(module, '__spec__', None)
        if name == namespace_name:
            expected = root / 'learning' / _NAMESPACE
            paths = list(getattr(module, '__path__', ()) or ())
            locations = list(getattr(spec, 'submodule_search_locations', ()) or ())
            if (getattr(spec, 'origin', None) is not None
                    or not isinstance(getattr(spec, 'loader', None), importlib.machinery.NamespaceLoader)
                    or getattr(module, '__loader__', None) is not spec.loader
                    or paths != [str(expected)] or locations != [str(expected)]
                    or expected.resolve(strict=True) != expected):
                raise ValueError('Unexpected teacher namespace origin')
            continue
        stem = root
        for component in name.split('.'):
            if not component.isidentifier():
                raise ValueError('Invalid project module name')
            stem /= component
        origin = getattr(spec, 'origin', None)
        loader = getattr(spec, 'loader', None)
        if (type(origin) is not str or not isinstance(loader, importlib.machinery.SourceFileLoader)
                or Path(origin) not in (stem / '__init__.py', stem.with_suffix('.py'))
                or Path(origin).resolve(strict=True) != Path(origin)
                or getattr(module, '__file__', None) != origin
                or getattr(module, '__loader__', None) is not loader):
            raise ValueError('Unexpected project module origin')
        if Path(origin) == stem / '__init__.py':
            paths = list(getattr(module, '__path__', ()) or ())
            locations = list(getattr(spec, 'submodule_search_locations', ()) or ())
            if paths != [str(stem)] or locations != [str(stem)]:
                raise ValueError('Unexpected project package search path')
    if not required_modules.issubset(sys.modules):
        raise ValueError('Required project modules are missing')
