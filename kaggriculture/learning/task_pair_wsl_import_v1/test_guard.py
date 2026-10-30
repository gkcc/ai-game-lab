"""File-free synthetic import and float checks; no real data, models or GPU."""

import os
from pathlib import Path
import subprocess
import sys
import unittest


ROOT = Path(__file__).resolve().parents[2]
SMOKE = ROOT / 'learning' / 'task_pair_gpu_smoke_v1' / 'run.py'
EXTRA = ROOT / 'learning' / 'task_pair_wsl_import_v1' / 'guard.py'

_ISOLATED_PROBE = r'''
import hashlib, importlib, importlib.abc, importlib.machinery, io, os, runpy, struct, sys
from pathlib import Path
from unittest.mock import patch

smoke_path, extra_path = map(Path, sys.argv[1:])
root = smoke_path.resolve().parents[2]
extra = runpy.run_path(str(extra_path))
virtual_site = str(root.parent / 'synthetic-virtual-site-not-a-file')

class Loader(importlib.abc.Loader):
    def create_module(self, spec): return None
    def exec_module(self, module): module.synthetic_origin = module.__spec__.origin

class Site(importlib.abc.PathEntryFinder):
    def find_spec(self, fullname, target=None):
        if fullname in ('learning', 'synthetic_lazy_dependency'):
            return importlib.machinery.ModuleSpec(fullname, Loader(), origin='virtual-site',
                                                  is_package=fullname == 'learning')
        return None

class Root(importlib.abc.PathEntryFinder):
    def __init__(self):
        m = importlib.machinery
        self.real = m.FileFinder(str(root), (m.ExtensionFileLoader, m.EXTENSION_SUFFIXES),
                                 (m.SourceFileLoader, m.SOURCE_SUFFIXES),
                                 (m.SourcelessFileLoader, m.BYTECODE_SUFFIXES))
    def find_spec(self, fullname, target=None):
        if fullname == 'synthetic_lazy_dependency':
            return importlib.machinery.ModuleSpec(fullname, Loader(), origin='virtual-project-shadow')
        return self.real.find_spec(fullname, target)

def hook(path):
    if path == virtual_site: return Site()
    if path == str(root): return Root()
    raise ImportError

sys.path_hooks.insert(0, hook)
sys.path_importer_cache.pop(virtual_site, None)
sys.path_importer_cache.pop(str(root), None)
sys.path.append(virtual_site)
assert importlib.machinery.PathFinder.find_spec('learning', sys.path).origin == 'virtual-site'
assert importlib.machinery.PathFinder.find_spec('synthetic_lazy_dependency', [str(root), virtual_site]).origin == 'virtual-project-shadow'

smoke = runpy.run_path(str(smoke_path))
guard, bound = smoke['bind_project_imports']()
assert bound == root and str(root) not in sys.path
repo = root.parent
expected = {name: hashlib.sha256(guard.bounded_snapshot(repo / name)).hexdigest()
            for name in extra['EXTRA_SOURCE_PATHS']}

# The test derives these hashes only for the synthetic check; a real caller
# must obtain them from an independently authenticated registration.
wrong = dict(expected)
wrong[next(iter(wrong))] = '0' * 64
try:
    extra['verify_extra_sources'](guard, root, wrong)
except ValueError:
    pass
else:
    raise AssertionError('Changed source identity accepted')

real_lexists = os.path.lexists
shadow = str(root / 'learning' / 'task_pair_eval_adapter_v1.py')
def injected_lexists(path):
    return True if str(path) == shadow else real_lexists(path)
with patch('os.path.lexists', side_effect=injected_lexists):
    try:
        extra['verify_extra_sources'](guard, root, expected)
    except ValueError:
        pass
    else:
        raise AssertionError('Extra package shadow accepted')

module_shadow = str(root / 'learning' / 'task_pair_eval_adapter_v1' / 'adapter.pyc')
def injected_module_shadow(path):
    return True if str(path) == module_shadow else real_lexists(path)
with patch('os.path.lexists', side_effect=injected_module_shadow):
    try:
        extra['verify_extra_sources'](guard, root, expected)
    except ValueError:
        pass
    else:
        raise AssertionError('Extra module shadow accepted')

# Each newly added initializer and source must match its independently bound
# registration entry. Virtual siblings exercise shadow checks without files.
added = (('task_pair_history_binding_v1', 'history'),
         ('task_pair_numpy_scorer_v1', 'scorer'))
for package_name, module_name in added:
    folder = root / 'learning' / package_name
    for rel in (f'kaggriculture/learning/{package_name}/__init__.py',
                f'kaggriculture/learning/{package_name}/{module_name}.py'):
        wrong = dict(expected)
        wrong[rel] = '0' * 64
        try:
            extra['verify_extra_sources'](guard, root, wrong)
        except ValueError:
            pass
        else:
            raise AssertionError('New initializer or module identity accepted')
    shadows = (folder.parent / (package_name + '.py'),
               folder / ('__init__' + importlib.machinery.EXTENSION_SUFFIXES[0]),
               folder / (module_name + '.pyc'),
               folder / module_name / '__init__.py')
    for candidate in shadows:
        def injected(path, target=str(candidate)):
            return True if str(path) == target else real_lexists(path)
        with patch('os.path.lexists', side_effect=injected):
            try:
                extra['verify_extra_sources'](guard, root, expected)
            except ValueError:
                pass
            else:
                raise AssertionError('New package or module shadow accepted')

extra['verify_extra_sources'](guard, root, expected)
with guard.protected_search_path(root):
    from learning.task_pair_data_preflight_v1.test_core import fixture
    preflight_name = 'learning.task_pair_data_preflight_v1.run'
    # The historical binding imports preflight; check the missing entry before
    # importing the full training closure, so the failure is meaningful.
    for name in sorted(extra['LOAD_DATA_MODULES'] - {preflight_name}):
        importlib.import_module(name)
    assert preflight_name not in sys.modules
    try:
        extra['verify_loaded_origins'](guard, root, required_modules=extra['LOAD_DATA_MODULES'])
    except ValueError:
        pass
    else:
        raise AssertionError('Missing real load_data module accepted')
    from learning.task_pair_data_preflight_v1 import run as preflight
    for name in sorted(extra['TRAIN_MODULES']):
        importlib.import_module(name)
extra['verify_loaded_origins'](guard, root, required_modules=extra['TRAIN_MODULES'])

# The two new modules and the historical binding's two normal imports are
# independently required. Removing an already-loaded module is safe in this
# isolated synthetic process; restore it immediately after each check.
for name in ('learning.task_pair_history_binding_v1.history',
             'learning.task_pair_numpy_scorer_v1.scorer',
             'learning.task_pair_mask_recheck_v1.run',
             'learning.task_worker_ranking_v2.run'):
    module = sys.modules.pop(name)
    try:
        try:
            extra['verify_loaded_origins'](guard, root, required_modules=extra['TRAIN_MODULES'])
        except ValueError:
            pass
        else:
            raise AssertionError('Required full-training module missing')
    finally:
        sys.modules[name] = module

# Exercise the real load_data import and module-origin call sites, but stop at
# the first metadata read. No private path or actual input is opened.
class FirstRead(Exception): pass
class StopBeforeData:
    def __init__(self): self.origins = []
    def check_import_origin(self, module, relative): self.origins.append(relative)
    def read_json(self, path): raise FirstRead
stop = StopBeforeData()
try:
    preflight.load_data(root / 'synthetic-no-data', stop, lambda: None)
except FirstRead:
    pass
else:
    raise AssertionError('Real loader did not stop at metadata gate')
assert len(stop.origins) == 5
extra['verify_loaded_origins'](guard, root, required_modules=extra['LOAD_DATA_MODULES'])

# The old check cannot accept this legitimate namespace. The new check must
# validate it explicitly rather than removing it or disguising its loader.
try:
    guard.verify_origins(root)
except ValueError:
    pass
else:
    raise AssertionError('Historical guard unexpectedly accepted namespace')
namespace = sys.modules['learning.task_teacher_choice_bc_v1']
saved_path = namespace.__path__
namespace.__path__ = [str(root.parent / 'synthetic-wrong-namespace')]
try:
    try:
        extra['verify_loaded_origins'](guard, root, required_modules=extra['TRAIN_MODULES'])
    except ValueError:
        pass
    else:
        raise AssertionError('Wrong namespace path accepted')
finally:
    namespace.__path__ = saved_path

numeric = sys.modules['learning.task_teacher_choice_bc_v1.numeric']
old_origin = numeric.__spec__.origin
numeric.__spec__.origin = 'synthetic-wrong-origin'
try:
    try:
        extra['verify_loaded_origins'](guard, root, required_modules=extra['TRAIN_MODULES'])
    except ValueError:
        pass
    else:
        raise AssertionError('Wrong source origin accepted')
finally:
    numeric.__spec__.origin = old_origin

package = sys.modules['learning.task_pair_data_adapter_v1']
saved_package_path = package.__path__
package.__path__ = [str(root.parent / 'synthetic-wrong-package')]
try:
    try:
        extra['verify_loaded_origins'](guard, root, required_modules=extra['TRAIN_MODULES'])
    except ValueError:
        pass
    else:
        raise AssertionError('Wrong package path accepted')
finally:
    package.__path__ = saved_package_path
extra['verify_loaded_origins'](guard, root, required_modules=extra['TRAIN_MODULES'])

import synthetic_lazy_dependency
assert synthetic_lazy_dependency.synthetic_origin == 'virtual-site'
assert str(root) not in sys.path

# Synthetic source bytes and numeric fingerprints have one little-endian
# encoding, including normalization of both signs of zero.
import numpy as np
_, views, _ = fixture()
core = sys.modules['learning.task_pair_data_preflight_v1.core']
train, dev = views[0], views[1]
assert core._fingerprint(train, 0) == core._fingerprint(dev, 0)
lo, hi = map(int, train.offsets[:2])
canonical = struct.pack('<q', hi - lo)
for value in list(train.shared[0]) + list(train.candidates[lo:hi].reshape(-1)):
    canonical += struct.pack('<f', 0.0 if value == 0 else float(value))
assert core._fingerprint(train, 0) == hashlib.sha256(canonical).digest()
assert core._group(train, 0)[-1] is True
arr = np.asarray([1.0, -0.0, 2.5], dtype='<f4')
assert arr.tobytes() == struct.pack('<fff', 1.0, -0.0, 2.5)
buf = io.BytesIO()
np.save(buf, arr, allow_pickle=False)
buf.seek(0)
decoded = np.load(buf, allow_pickle=False)
assert decoded.dtype == np.dtype('float32') and decoded.tobytes() == arr.tobytes()
print('isolated-synthetic-ok')
'''


class ImportGuardTests(unittest.TestCase):
    def test_isolated_real_imports_and_synthetic_shadow_and_floats(self):
        env = dict(os.environ, JAX_PLATFORMS='cpu', CUDA_VISIBLE_DEVICES='-1')
        process = subprocess.run([sys.executable, '-B', '-I', '-c', _ISOLATED_PROBE,
                                  str(SMOKE), str(EXTRA)], capture_output=True,
                                 text=True, timeout=30, env=env)
        self.assertEqual(process.returncode, 0, 'Isolated synthetic import probe failed')
        self.assertEqual(process.stdout.strip(), 'isolated-synthetic-ok')


if __name__ == '__main__':
    unittest.main()
