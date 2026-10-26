"""Small real-writer regressions; no game, model, GPU or historical data reads."""
import _io
import builtins
from concurrent.futures import ThreadPoolExecutor
import gzip
import importlib.util
import io
import json
import os
from pathlib import Path
import shutil
import sys
import threading
import unittest

import numpy as np

from .storage import OutputBudgetExceeded, UnsafeWriteError, owned_writes


def _scratch():
    # The standalone ownership utility has no workflow hooks/import side effects.
    explicit = os.environ.get('KAGGRICULTURE_SCRATCH_HELPER')
    codex_root = os.environ.get('CODEX_HOME')
    if explicit:
        path = Path(explicit).expanduser()
    elif codex_root:
        path = Path(codex_root).expanduser() / 'skills/agent-workflow/scripts/artifacts.py'
    else:
        raise unittest.SkipTest('Set KAGGRICULTURE_SCRATCH_HELPER or CODEX_HOME to run owned scratch checks')
    if not path.is_file():
        raise RuntimeError('Configured owned scratch helper is unavailable')
    spec = importlib.util.spec_from_file_location('anchor_storage_owned_artifacts', path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.scratch_directory('anchor-storage-check')


class StorageTests(unittest.TestCase):
    def setUp(self):
        self.scratch = _scratch()
        self.work = self.scratch.__enter__()
        self.addCleanup(self._clean_scratch)

    def _clean_scratch(self):
        self.scratch.__exit__(None, None, None)
        self.assertFalse(self.work.exists(), 'Owned scratch was not removed')

    def test_actual_writers_roundtrip_and_count_encoded_bytes(self):
        source = self.work / 'source.bin'
        source.write_bytes(bytes(range(64)))
        out = self.work / 'outputs'
        expected = np.arange(24, dtype=np.float32).reshape(4, 6)
        payload = {'label': '中文🙂', 'lines': ['a', 'b']}
        with owned_writes(out, 64 * 1024) as budget:
            self.assertFalse(out.exists())
            out.mkdir()
            (out / 'unicode.json').write_text(json.dumps(payload, ensure_ascii=False),
                                             encoding='utf-8')
            (out / 'raw.bin').write_bytes(b'bytes')
            with (out / 'utf16.txt').open('w', encoding='utf-16', newline='\r\n') as stream:
                stream.write('中\n')
                stream.write('🙂\n')
            for part in ('first\n', '第二段\n'):
                with gzip.open(out / 'trace.gz', 'at', encoding='utf-8', newline='\n') as stream:
                    stream.write(part)
            np.save(out / 'array', expected, allow_pickle=False)
            np.savez_compressed(out / 'arrays.npz', first=expected, second=expected.T)
            shutil.copyfile(source, out / 'copy.bin')
            with self.assertRaises(shutil.SameFileError):
                shutil.copyfile(out / 'copy.bin', out / 'copy.bin')
        self.assertEqual(json.loads((out / 'unicode.json').read_text(encoding='utf-8')), payload)
        self.assertEqual((out / 'utf16.txt').read_bytes(), '中\r\n🙂\r\n'.encode('utf-16'))
        self.assertEqual(gzip.decompress((out / 'trace.gz').read_bytes()).decode(), 'first\n第二段\n')
        np.testing.assert_array_equal(np.load(out / 'array.npy', allow_pickle=False), expected)
        with np.load(out / 'arrays.npz', allow_pickle=False) as saved:
            np.testing.assert_array_equal(saved['first'], expected)
            np.testing.assert_array_equal(saved['second'], expected.T)
        self.assertEqual((out / 'copy.bin').read_bytes(), source.read_bytes())
        final_bytes = sum(path.stat().st_size for path in budget.created_paths)
        self.assertGreaterEqual(budget.used_bytes, final_bytes)
        self.assertEqual(budget.snapshot()['created_count'], 7)
        json.dumps(budget.snapshot(), allow_nan=False)

    def test_charge_precedes_write_and_is_not_refunded_or_silently_swallowed(self):
        out = self.work / 'outputs'
        out.mkdir()
        with self.assertRaises(OutputBudgetExceeded):
            with owned_writes(out, 7) as budget:
                with (out / 'file').open('w+b') as stream:
                    stream.write(b'abc')
                    stream.seek(0)
                    stream.write(b'xy')
                    stream.truncate(2)
                with (out / 'file').open('ab') as stream:
                    stream.write(b'zz')
                    with self.assertRaises(OutputBudgetExceeded):
                        stream.write(b'!')
                    self.assertEqual((out / 'file').read_bytes(), b'xyzz')
        self.assertEqual(budget.used_bytes, 7)
        self.assertEqual(budget.remaining_bytes, 0)
        self.assertEqual(budget.snapshot()['refused'], 1)

    def test_inputs_descriptors_sparse_growth_and_outside_paths_fail_closed(self):
        out = self.work / 'outputs'
        out.mkdir()
        frozen = out / 'frozen'
        frozen.write_bytes(b'frozen')
        outside = self.work / 'outside'
        outside.write_bytes(b'outside')
        descriptor = os.open(frozen, os.O_RDONLY)
        try:
            with owned_writes(out, 128) as budget:
                for mode in ('w', 'a', 'r+'):
                    with self.subTest(mode=mode), self.assertRaises(UnsafeWriteError):
                        frozen.open(mode)
                with self.assertRaises(UnsafeWriteError):
                    outside.write_bytes(b'changed')
                with self.assertRaises(UnsafeWriteError):
                    io.FileIO(out / 'fileio', 'w')
                with self.assertRaises(UnsafeWriteError):
                    os.open(out / 'descriptor', os.O_CREAT | os.O_WRONLY)
                with self.assertRaises(UnsafeWriteError):
                    open(descriptor, 'wb', closefd=False)
                with open(descriptor, 'rb', closefd=False) as stream:
                    self.assertEqual(stream.read(), b'frozen')
                with io.FileIO(frozen, 'r') as stream:
                    self.assertEqual(stream.read(), b'frozen')
                with (out / 'new').open('wb') as stream:
                    stream.write(b'abc')
                    with self.assertRaises(io.UnsupportedOperation):
                        stream.fileno()
                    with self.assertRaises((io.UnsupportedOperation, RuntimeError)):
                        np.arange(2, dtype=np.int8).tofile(stream)
                    with self.assertRaises(UnsafeWriteError):
                        stream.seek(100)
                    with self.assertRaises(UnsafeWriteError):
                        stream.truncate(100)
            self.assertEqual(budget.used_bytes, 3)
        finally:
            os.close(descriptor)
        self.assertEqual(frozen.read_bytes(), b'frozen')
        self.assertEqual(outside.read_bytes(), b'outside')
        self.assertFalse((out / 'descriptor').exists())

    def test_two_threads_share_one_atomic_budget(self):
        out = self.work / 'outputs'
        out.mkdir()
        barrier = threading.Barrier(2)

        def write(name):
            with (out / name).open('wb') as stream:
                barrier.wait(timeout=5)
                try:
                    stream.write(b'x' * 40)
                    return True
                except OutputBudgetExceeded:
                    return False

        with self.assertRaises(OutputBudgetExceeded):
            with owned_writes(out, 60) as budget:
                with ThreadPoolExecutor(max_workers=2) as pool:
                    accepted = list(pool.map(write, ('one', 'two')))
                self.assertEqual(sum(accepted), 1)
        self.assertEqual(budget.used_bytes, 40)
        self.assertEqual(sum(path.stat().st_size for path in out.iterdir()), 40)

    def test_real_compressed_numpy_and_copy_writers_cannot_overrun_cap(self):
        source = self.work / 'source'
        source.write_bytes(bytes(range(256)))

        def gzip_write(path):
            with gzip.open(path, 'wb') as stream:
                stream.write(bytes(range(256)))

        writers = dict(gzip=gzip_write,
                       npy=lambda path: np.save(path, np.arange(100), allow_pickle=False),
                       npz=lambda path: np.savez_compressed(path, array=np.arange(100)),
                       copy=lambda path: shutil.copyfile(source, path))
        for name, writer in writers.items():
            with self.subTest(writer=name):
                out = self.work / name
                out.mkdir()
                with self.assertRaises(OutputBudgetExceeded):
                    with owned_writes(out, 32) as budget:
                        writer(out / 'data')
                self.assertLessEqual(budget.used_bytes, 32)
                self.assertLessEqual(sum(path.stat().st_size for path in out.iterdir()),
                                     budget.used_bytes)

    def test_allowlists_and_all_patches_restore_after_exception(self):
        targets = ((builtins, 'open'), (io, 'open'), (_io, 'open'),
                   (io, 'FileIO'), (_io, 'FileIO'), (os, 'open'), (shutil, 'copyfile'))
        originals = {(module, name): getattr(module, name) for module, name in targets}
        out = self.work / 'outputs'
        out.mkdir()
        (out / 'segments').mkdir()
        with self.assertRaisesRegex(ValueError, 'synthetic failure'):
            with owned_writes(out, 64, allowed_paths=['one'],
                              allowed_prefixes=['segments']) as budget:
                (out / 'one').write_bytes(b'1')
                (out / 'segments' / 'part').write_bytes(b'2')
                with self.assertRaises(UnsafeWriteError):
                    (out / 'segments-elsewhere').write_bytes(b'3')
                with self.assertRaises(RuntimeError):
                    with owned_writes(out, 64):
                        self.fail('Nested context was accepted')
                raise ValueError('synthetic failure')
        for (module, name), original in originals.items():
            self.assertIs(getattr(module, name), original)
        self.assertEqual(budget.used_bytes, 2)
        # Outside-context behavior is restored; this file was never guarded.
        (self.work / 'restored').write_bytes(b'normal')


if __name__ == '__main__':
    unittest.main()
