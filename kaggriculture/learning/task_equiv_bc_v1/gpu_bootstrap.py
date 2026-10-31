"""Bind project packages without putting the checkout on dependency sys.path."""
import hashlib
import importlib.machinery
from pathlib import Path
import types


IMPORT_GUARD_SHA = 'c5ec904b6c7af23ffb4a6ddd31dd8aa47a0e760cc892087f07e038b303c65f3d'


def bind():
    root = Path(__file__).resolve().parents[2]
    path = root / 'learning' / 'task_pair_gpu_smoke_v1' / 'import_guard.py'
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 1024 * 1024:
        raise ValueError('Invalid import guard source')
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != IMPORT_GUARD_SHA:
        raise ValueError('Import guard identity differs')
    module = types.ModuleType('frozen_gpu_import_guard')
    module.__file__ = str(path)
    exec(compile(raw, str(path), 'exec', dont_inherit=True), module.__dict__)
    module.bind(root)
    folder = root / 'learning' / 'task_equiv_bc_v1'
    suffixes = ('.py', '.pyc', *importlib.machinery.EXTENSION_SUFFIXES)
    if any((folder.parent / (folder.name + suffix)).exists() for suffix in suffixes):
        raise ValueError('Unexpected task package shadow')
    for stem in ('train', 'loss', 'dataset', 'probe', 'host_gpu'):
        if any((folder / (stem + suffix)).exists() for suffix in suffixes[1:]):
            raise ValueError('Unexpected task module shadow')
        candidate = folder / stem
        if candidate.is_dir() and any((candidate / ('__init__' + suffix)).exists() for suffix in suffixes):
            raise ValueError('Unexpected task subpackage shadow')
    module.verify_origins(root)
    return module, root
