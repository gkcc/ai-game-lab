"""In-memory registration/host checks; no WSL, private data or games."""
from contextlib import ExitStack, redirect_stdout
import io
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from . import experiment as e


class RegistrationTests(unittest.TestCase):
    def setUp(self):
        self.head = 'a' * 40
        self.stage = Path.cwd() / 'synthetic' / 'artifacts' / e.STAGE_NAME
        self.repo = Path.cwd() / 'synthetic' / 'repo'
        self.registration = self.repo / 'evidence' / 'registration.json'
        self.binding = dict(schema='water-residual-private-binding-v1', salt='b' * 64,
            baseline_sha256='c' * 64, train_sha256='d' * 64, dev_sha256='e' * 64,
            schema_sha256='f' * 64, data_registration_commit=e.DATA_COMMIT,
            device_kind='synthetic GPU', rng_seed=e.RNG_SEED, wsl={})
        self.binding_raw = e.previous._encoded(self.binding)
        self.sources = {'synthetic_source.py': '1' * 64}
        self.reg_raw = e.previous._encoded(e._registration(self.sources, e._sha(self.binding_raw)))

    def verify(self, *, binding=None, sources=None, dirty=False):
        binding = self.binding_raw if binding is None else binding
        sources = self.sources if sources is None else sources
        def read(path, _limit):
            return binding if Path(path).name == 'binding.local.json' else self.reg_raw
        def git(*args):
            if args == ('rev-parse', 'HEAD'): return self.head
            if args == ('rev-parse', 'HEAD^'): return e.BASE_COMMIT
            if args == ('status', '--porcelain'): return 'dirty' if dirty else ''
            raise AssertionError('Unexpected synthetic Git query')
        with patch.object(e, 'STAGE', self.stage), patch.object(e, 'REPO', self.repo), \
                patch.object(e, 'REGISTRATION', self.registration), patch.object(e, '_read', side_effect=read), \
                patch.object(e, '_git', side_effect=git), patch.object(e, '_source_hashes', return_value=sources), \
                patch.object(e.subprocess, 'check_output', return_value=self.reg_raw), \
                patch.object(Path, 'is_symlink', return_value=False), patch.object(Path, 'is_dir', return_value=True), \
                patch.object(Path, 'resolve', new=lambda path, strict=False: path):
            return e.verify_registration(self.head, self.stage)

    def test_commit_and_source_binding_and_replacement_rejection(self):
        registration, binding = self.verify()
        self.assertEqual(binding, self.binding)
        self.assertEqual(registration['updates'], 64)
        with self.assertRaises(ValueError): self.verify(binding=self.binding_raw + b' ')
        with self.assertRaises(ValueError): self.verify(sources={'synthetic_source.py': '2' * 64})
        with self.assertRaises(ValueError): self.verify(dirty=True)

    def test_initial_memory_failure_never_starts_or_writes(self):
        with patch.object(e, '_windows', return_value=True), \
                patch.object(e.previous, 'physical_available', return_value=e.MEMORY_INITIAL - 1), \
                patch.object(e, '_write', side_effect=AssertionError('must not write')):
            with self.assertRaises(MemoryError): e.run()

    def test_small_residual_archive_is_validated_and_read_only(self):
        weights = np.zeros(373, dtype=np.float32)
        weights[3] = 0.25
        stream = io.BytesIO()
        np.savez_compressed(stream, weights=weights)
        actual = e._weights(stream.getvalue())
        np.testing.assert_array_equal(actual, weights)
        with self.assertRaises(ValueError): actual.setflags(write=True)
        stream = io.BytesIO()
        np.savez_compressed(stream, weights=np.zeros(373, dtype=np.float32))
        with self.assertRaises(ValueError): e._weights(stream.getvalue())
        stream = io.BytesIO()
        np.savez_compressed(stream, weights=np.ones(10**5, dtype=np.float32))
        with self.assertRaises(ValueError): e._weights(stream.getvalue())

    def test_aggregate_comparison_requires_counts_but_allows_tiny_float_error(self):
        self.assertTrue(e._equivalent({'count': 2, 'nll': 0.5}, {'count': 2, 'nll': 0.50000001}))
        self.assertFalse(e._equivalent({'count': 2}, {'count': 3}))
        self.assertFalse(e._equivalent({'count': 2}, {'count': True}))
        self.assertFalse(e._equivalent({'nll': 0.5}, {'nll': 0.6}))


class HostAcceptanceTests(unittest.TestCase):
    def exercise(self, result, *, fail_result_write=False):
        from learning.task_equiv_bc_v1 import host_gpu
        from learning.task_pair_wsl_launch_v1 import host
        head = 'a' * 40
        wsl = dict(distribution='synthetic', wsl_venv_executable='/synthetic/python',
                   wsl_game_root='/synthetic/game', wsl_repo_root='/synthetic/repo')
        binding = dict(train_sha256='b' * 64, dev_sha256='c' * 64, wsl=wsl, device_kind='synthetic')
        contract = {key: binding[key] for key in ('train_sha256', 'dev_sha256')}
        writes = []
        def record(path, raw):
            name = Path(path).name
            if name == 'host-result.local.json' and fail_result_write:
                raise OSError('synthetic write failure')
            writes.append((name, e.previous._document(raw)))
        with ExitStack() as stack:
            for context in (
                patch.object(e, '_windows', return_value=True),
                patch.object(e.previous, 'physical_available', return_value=6 * 1024**3),
                patch.object(e, '_git', return_value=head),
                patch.object(e, 'verify_registration', return_value=({}, binding)),
                patch.object(e.transfer, 'load_trainable', return_value=([], [], contract)),
                patch.object(e, '_asset_check'),
                patch.object(e, '_older_wsl_contract', return_value=(wsl, 'synthetic')),
                patch.object(e.shutil, 'disk_usage', return_value=SimpleNamespace(free=8 * 1024**3)),
                patch.object(e, '_verify_host_gpu'),
                patch.object(e, '_write', side_effect=record),
                patch.object(host, '_system_wsl', return_value=Path('wsl.exe')),
                patch.object(host_gpu, 'supervise', return_value=result),
                patch.object(e, '_validate_worker_output', side_effect=FileNotFoundError('synthetic')),
                patch.object(host_gpu, 'STAGE_NAME', host_gpu.STAGE_NAME),
                redirect_stdout(io.StringIO()),
            ):
                stack.enter_context(context)
            with self.assertRaises((FileNotFoundError, OSError)):
                e.run()
        return writes

    def test_clean_supervisor_exit_without_model_is_not_success(self):
        result = SimpleNamespace(status='passed', reason='completed', cleanup_confirmed=True,
                                 child=None, stdout_stream=None, stderr_stream=None)
        writes = self.exercise(result)
        self.assertEqual(writes[-1][1]['status'], 'failed')
        self.assertEqual(writes[-1][1]['reason'], 'worker_validation')

    def test_receipt_write_failure_still_waits_for_owned_transport(self):
        child = SimpleNamespace(wait=unittest.mock.Mock())
        stdout, stderr = SimpleNamespace(close=unittest.mock.Mock()), SimpleNamespace(close=unittest.mock.Mock())
        result = SimpleNamespace(status='cleanup_unconfirmed', reason='memory_gate', cleanup_confirmed=False,
                                 child=child, stdout_stream=stdout, stderr_stream=stderr)
        self.exercise(result, fail_result_write=True)
        child.wait.assert_called_once_with()
        stdout.close.assert_called_once_with()
        stderr.close.assert_called_once_with()


if __name__ == '__main__':
    unittest.main()
