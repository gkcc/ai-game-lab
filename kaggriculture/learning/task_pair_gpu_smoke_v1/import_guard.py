"""Bind project imports for the smoke process from a verified bootstrap snapshot.

This is a local-shadow guard, not an OS lock or a defence against arbitrary
custom import hooks or concurrent changes. Formal callers must authenticate
all sources and the Git identity both before and after the owned attempt.
"""

import hashlib
from contextlib import contextmanager
import importlib.machinery
import importlib.util
import os
from pathlib import Path
import stat
import sys
import types
import uuid


LIMIT = 1024 * 1024
BOOTSTRAP_SHA = 'defdc84ba87104cc513d84756131d564f7224655c9f4fa210fe28a30efbaed68'
NEW_INITIALIZERS = {
    'task_pair_gpu_smoke_v1': '0ba32f1933371a6c3f2c8be82ebd72fe0c5886da36c2b545d7c6893ee76141f3',
    'task_pair_data_preflight_v1': '627293d6c4e0da1b5223292c1e5639d99006d6d732e1ccdba437c30b2db0311f',
    'task_pair_training_v1': '15ef3b12ff6b7680d507a5e25f6b05cce9c0af9c5da832f4a2d550ec65b16e31',
}


def bounded_snapshot(path):
    """Read a regular source once with a strict maximum, never read_bytes."""
    path = Path(path)
    before = path.lstat()
    if (not stat.S_ISREG(before.st_mode) or stat.S_ISLNK(before.st_mode)
            or getattr(before, 'st_file_attributes', 0) & 0x400 or before.st_size > LIMIT):
        raise ValueError('Untrusted source file')
    flags = os.O_RDONLY | getattr(os, 'O_BINARY', 0) | getattr(os, 'O_NOFOLLOW', 0)
    descriptor = os.open(path, flags)
    try:
        current = os.fstat(descriptor)
        if not stat.S_ISREG(current.st_mode) or current.st_size > LIMIT:
            raise ValueError('Untrusted source file')
        with os.fdopen(descriptor, 'rb', closefd=False) as stream:
            raw = stream.read(LIMIT + 1)
        if len(raw) != current.st_size or len(raw) > LIMIT:
            raise ValueError('Untrusted source length')
        return raw
    finally:
        os.close(descriptor)


def verify_new_packages(root):
    """Extend the historical initializer and local-shadow checks to new packages."""
    learning = root / 'learning'
    native = tuple(importlib.machinery.EXTENSION_SUFFIXES)
    for name, digest in NEW_INITIALIZERS.items():
        folder = learning / name
        if not folder.is_dir() or folder.resolve(strict=True) != folder:
            raise ValueError('Unexpected project package')
        if any((learning / (name + suffix)).exists() for suffix in ('.py', '.pyc', *native)):
            raise ValueError('Unexpected project package shadow')
        init = folder / '__init__.py'
        if hashlib.sha256(bounded_snapshot(init)).hexdigest() != digest:
            raise ValueError('Unexpected project initializer')
        if any((folder / ('__init__' + suffix)).exists() for suffix in ('.pyc', *native)):
            raise ValueError('Unexpected project initializer shadow')
        with os.scandir(folder) as entries:
            for entry in entries:
                if not entry.name.lower().endswith('.py') or entry.name.lower() == '__init__.py':
                    continue
                path = folder / entry.name
                info = path.lstat()
                if (not stat.S_ISREG(info.st_mode) or stat.S_ISLNK(info.st_mode)
                        or getattr(info, 'st_file_attributes', 0) & 0x400):
                    raise ValueError('Unexpected project source')
                stem = entry.name[:-3]
                if any((folder / (stem + suffix)).exists() for suffix in ('.pyc', *native)):
                    raise ValueError('Unexpected project module shadow')
                other = folder / stem
                if other.is_dir() and any((other / ('__init__' + suffix)).exists()
                                          for suffix in ('.py', '.pyc', *native)):
                    raise ValueError('Unexpected project package shadow')


def verify_search_path(root):
    # Do not make the checkout an interpreter-wide search path: JAX, NumPy,
    # Optax, jaxlib and plugins may be imported lazily well after this guard.
    # The two project packages are bound explicitly below instead.
    for entry in sys.path:
        if not isinstance(entry, str) or not entry:
            raise ValueError('Unexpected interpreter search path')
        location = Path(entry).resolve()
        if location == root or root in location.parents:
            raise ValueError('Project directory in interpreter search path')


@contextmanager
def protected_search_path(root):
    """Suppress the redundant legacy fixture root insertion during its import."""
    verify_search_path(root)
    original = sys.path
    before = original[:]

    class ReadOnlyPath(list):
        def insert(self, index, item):
            if index == 0 and isinstance(item, str) and Path(item).resolve() == root:
                return
            raise ValueError('Interpreter search path mutation')

        def forbidden(self, *args, **kwargs):
            raise ValueError('Interpreter search path mutation')

        append = clear = extend = pop = remove = reverse = sort = forbidden
        __setitem__ = __delitem__ = __iadd__ = __imul__ = forbidden

    protected = ReadOnlyPath(before)
    sys.path = protected
    try:
        yield
    finally:
        changed = sys.path is not protected or original != before
        sys.path = original
        if changed:
            raise ValueError('Interpreter search path changed')
        verify_search_path(root)


def bind(root):
    """Authenticate the bounded bootstrap once, then pin the project namespace."""
    root = Path(root).resolve(strict=True)
    if not sys.flags.isolated or not sys.flags.dont_write_bytecode or not __debug__:
        raise ValueError('Expected isolated non-optimized interpreter without bytecode writes')
    if any(name == 'learning' or name.startswith('learning.') or name == 'decision'
           or name.startswith('decision.') for name in sys.modules):
        raise ValueError('Project modules were already loaded')
    verify_search_path(root)
    path = root / 'learning' / 'panel_eval_v3' / 'bootstrap.py'
    raw = bounded_snapshot(path)
    if hashlib.sha256(raw).hexdigest() != BOOTSTRAP_SHA:
        raise ValueError('Historical bootstrap identity differs')
    bootstrap = types.ModuleType('frozen_project_import_guard')
    bootstrap.__file__ = str(path)
    exec(compile(raw, str(path), 'exec', dont_inherit=True), bootstrap.__dict__)
    bootstrap.verify_initializers(root)
    verify_new_packages(root)
    cache = root / ('.codex-no-bytecode-' + uuid.uuid4().hex)
    if cache.exists():
        raise ValueError('Bytecode prefix is unexpectedly occupied')
    sys.pycache_prefix = str(cache)
    spec = importlib.machinery.PathFinder.find_spec('learning', [str(root)])
    if (spec is None or spec.loader is not None or spec.submodule_search_locations is None
            or [Path(item).resolve() for item in spec.submodule_search_locations] != [root / 'learning']):
        raise ValueError('Unexpected project namespace')
    spec.submodule_search_locations = [str(root / 'learning')]
    package = importlib.util.module_from_spec(spec)
    package.__path__ = [str(root / 'learning')]
    sys.modules['learning'] = package
    decision_spec = importlib.machinery.PathFinder.find_spec('decision', [str(root)])
    if (decision_spec is None or not isinstance(decision_spec.loader, importlib.machinery.SourceFileLoader)
            or decision_spec.origin != str(root / 'decision' / '__init__.py')
            or list(decision_spec.submodule_search_locations or ()) != [str(root / 'decision')]):
        raise ValueError('Unexpected decision package')
    decision = importlib.util.module_from_spec(decision_spec)
    sys.modules['decision'] = decision
    decision_spec.loader.exec_module(decision)
    verify_origins(root)
    return root


def verify_origins(root):
    """Verify loaded project package/module origins after the imports of interest."""
    root = Path(root)
    verify_search_path(root)
    learning = sys.modules.get('learning')
    if (learning is None or getattr(learning, '__path__', None) != [str(root / 'learning')]
            or getattr(getattr(learning, '__spec__', None), 'submodule_search_locations', None)
            != [str(root / 'learning')]):
        raise ValueError('Project namespace binding changed')
    for name, module in tuple(sys.modules.items()):
        if name != 'decision' and not name.startswith('learning.') and not name.startswith('decision.'):
            continue
        base = name.split('.')[0]
        stem = root / base
        for component in name.split('.')[1:]:
            if not component.isidentifier():
                raise ValueError('Invalid project module name')
            stem /= component
        origin = getattr(getattr(module, '__spec__', None), 'origin', None)
        loader = getattr(getattr(module, '__spec__', None), 'loader', None)
        expected = (stem / '__init__.py', stem.with_suffix('.py'))
        if (type(origin) is not str or not isinstance(loader, importlib.machinery.SourceFileLoader)
                or Path(origin) not in expected or Path(origin).resolve(strict=True) != Path(origin)):
            raise ValueError('Unexpected project module origin')
        if getattr(module, '__file__', None) != origin:
            raise ValueError('Unexpected project module file')
