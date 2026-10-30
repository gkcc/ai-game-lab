"""Bind known project packages before loading the experiment's dependencies."""
import hashlib
import importlib.machinery
import importlib.util
from pathlib import Path
import sys
import types
import uuid

BOOTSTRAP_SHA = 'defdc84ba87104cc513d84756131d564f7224655c9f4fa210fe28a30efbaed68'


def bind(root, *, check_public_prefix=False):
    root = Path(root).resolve(strict=True)
    if not sys.flags.isolated or not sys.flags.dont_write_bytecode or not __debug__:
        raise ValueError('Expected isolated non-optimized interpreter without bytecode writes')
    if any(name == 'learning' or name.startswith('learning.') or name == 'decision'
           or name.startswith('decision.') for name in sys.modules):
        raise ValueError('Project modules were already loaded')
    bootstrap_path = root/'learning/panel_eval_v3/bootstrap.py'
    raw = bootstrap_path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != BOOTSTRAP_SHA:
        raise ValueError('Import guard identity differs')
    bootstrap = types.ModuleType('frozen_project_import_guard')
    bootstrap.__file__ = str(bootstrap_path)
    exec(compile(raw, str(bootstrap_path), 'exec', dont_inherit=True), bootstrap.__dict__)
    bootstrap.verify_initializers(root)
    if check_public_prefix:
        package = root/'learning/task_prefix_diagnostic_v1'
        if not package.is_dir() or package.resolve() != package:
            raise ValueError('Prior diagnostic package differs')
        for suffix in ('.py', '.pyc', *importlib.machinery.EXTENSION_SUFFIXES):
            if (package.parent/(package.name+suffix)).exists():
                raise ValueError('Prior diagnostic package shadow')
        for name in ('runner', 'pairing'):
            if any((package/(name+suffix)).exists() for suffix in ('.pyc', *importlib.machinery.EXTENSION_SUFFIXES)):
                raise ValueError('Prior diagnostic module shadow')
            other = package/name
            if other.is_dir() and any((other/('__init__'+suffix)).exists()
                                      for suffix in ('.py', '.pyc', *importlib.machinery.EXTENSION_SUFFIXES)):
                raise ValueError('Prior diagnostic package shadows module')
    # -B forbids writes; a fresh nonexistent cache prefix also avoids reading
    # pre-existing bytecode. No directory or temporary file is created here.
    cache = root/('.codex-no-bytecode-'+uuid.uuid4().hex)
    if cache.exists():
        raise ValueError('Bytecode prefix is unexpectedly occupied')
    sys.pycache_prefix = str(cache)
    spec = importlib.machinery.PathFinder.find_spec('learning', [str(root)])
    if (spec is None or spec.loader is not None or spec.submodule_search_locations is None
        or [Path(p).resolve() for p in spec.submodule_search_locations] != [(root/'learning').resolve()]):
        raise ValueError('Unexpected project namespace')
    spec.submodule_search_locations = [str(root/'learning')]
    package = importlib.util.module_from_spec(spec)
    package.__path__ = [str(root/'learning')]
    sys.modules['learning'] = package
    sys.path.insert(0, str(root))
