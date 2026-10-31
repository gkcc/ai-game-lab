"""Synthetic worker-to-host acceptance; no files, real GPU, or games.

The fake backend supplies deterministic weights and cannot establish that a
real device performed an update. The production data decoding, scoring,
result serialization, and host validation are used with synthetic inputs.
"""
from copy import deepcopy
from dataclasses import replace
import io
import json
from contextlib import ExitStack, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from learning.task_bc_v1 import model
from learning.task_equiv_bc_v1 import host_gpu
from learning.task_pair_wsl_launch_v1 import host as wsl_host
from learning.task_teacher_transfer_dataset_v1 import experiment as transfer
from learning.task_teacher_transfer_dataset_v1.test_dataset import make_groups, matrix
from learning.task_water_residual_v1 import experiment, gpu_worker
from learning.task_water_residual_v1 import test_gpu_worker as synthetic_worker


class WorkerHostIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        synthetic_worker.WorkerTests.setUpClass()

    def fixture(self):
        kwargs, network, backend = synthetic_worker.WorkerTests().setup_execution()
        # The sealed training split contains 58 WATER and one PLANT target.
        # Reproduce those counts with entirely synthetic source groups.
        train = make_groups([('A01', 29), ('A02', 30)], one_operation=True)
        train[0] = replace(train[0], op='PLANT', features=matrix(1000, op='PLANT'))
        train_raw = synthetic_worker.archive(transfer._split_arrays(train, lambda: None))
        kwargs['train_raw'] = train_raw
        kwargs['binding']['train_sha256'] = synthetic_worker.digest(train_raw)
        stage = Path('synthetic-stage')
        files = {
            stage / 'baseline.npz': kwargs['baseline_raw'],
            stage / 'train.npz': train_raw,
            stage / 'dev.npz': kwargs['dev_raw'],
        }
        return kwargs, network, backend, stage, files

    @staticmethod
    def save(stage, files, weights, result, baseline):
        def read(path, limit, guard):
            guard()
            raw = files[path]
            if len(raw) > limit:
                raise ValueError('Synthetic snapshot exceeds its bound')
            return raw

        def write(path, raw):
            if path in files:
                raise FileExistsError('Synthetic output already exists')
            files[path] = raw

        return gpu_worker.save_outputs(stage, weights, result, baseline,
                                       guard=lambda: None, disk_guard=lambda _stage: None,
                                       read_file=read, write_once=write)

    @staticmethod
    def validate(stage, files, network, kwargs):
        def read(path, limit):
            raw = files[path]
            if len(raw) > limit:
                raise ValueError('Synthetic snapshot exceeds its bound')
            return raw

        with patch.object(experiment, 'STAGE', stage), \
                patch.object(experiment, '_read', side_effect=read), \
                patch.object(model, 'TaskNetwork', return_value=network):
            return experiment._validate_worker_output(kwargs['registration_commit'], kwargs['binding'],
                                                       guard=lambda: None)

    def test_worker_result_reloads_and_failed_screen_remains_failed(self):
        kwargs, network, backend, stage, files = self.fixture()
        weights, summary = gpu_worker.execute(**kwargs)
        saved = self.save(stage, files, weights, summary, kwargs['baseline_raw'])
        result = self.validate(stage, files, network, kwargs)

        self.assertEqual(backend.called, 1)
        self.assertEqual(result['training']['selected_groups'], 58)
        self.assertEqual(result['parity']['groups'], 62)
        self.assertGreater(result['parity']['water_groups'], 0)
        self.assertEqual(result['development_screen'], 'failed')
        self.assertEqual(saved['development']['screen'], 'failed')
        self.assertFalse(saved['independent_competitive_evaluation'])
        self.assertEqual(result['official_matches'], 0)

        # The host must rederive the comparison population. An apparently
        # successful but empty GPU comparison is not a passing receipt.
        altered = deepcopy(saved)
        altered['parity']['water_groups'] = 0
        altered['parity']['water_candidates'] = 0
        files[stage / 'gpu-worker-result.local.json'] = json.dumps(altered).encode()
        with self.assertRaises(ValueError):
            self.validate(stage, files, network, kwargs)

    def test_supervisor_pass_without_worker_outputs_is_not_success(self):
        kwargs, _network, _backend, stage, files = self.fixture()
        head, binding = kwargs['registration_commit'], kwargs['binding']
        wsl = dict(distribution='synthetic', wsl_venv_executable='/synthetic/python',
                   wsl_game_root='/synthetic/game', wsl_repo_root='/synthetic/repo')
        binding.update(wsl=wsl, device_kind='synthetic')
        contract = {key: binding[key] for key in ('train_sha256', 'dev_sha256')}

        def read(path, limit):
            if path not in files:
                raise FileNotFoundError('Synthetic worker output is absent')
            raw = files[path]
            if len(raw) > limit:
                raise ValueError('Synthetic snapshot exceeds its bound')
            return raw

        def write(path, raw):
            if path in files:
                raise FileExistsError('Synthetic output already exists')
            files[path] = raw

        supervised = host_gpu.HostResult('passed', 'completed', 0, True)
        with ExitStack() as stack:
            for context in (
                patch.object(experiment, 'STAGE', stage),
                patch.object(experiment, '_windows', return_value=True),
                patch.object(experiment.previous, 'physical_available', return_value=8 * 1024**3),
                patch.object(experiment, '_git', return_value=head),
                patch.object(experiment, 'verify_registration', return_value=({}, binding)),
                patch.object(transfer, 'load_trainable', return_value=([], [], contract)),
                patch.object(experiment, '_older_wsl_contract', return_value=(wsl, 'synthetic')),
                patch.object(experiment.shutil, 'disk_usage', return_value=SimpleNamespace(free=8 * 1024**3)),
                patch.object(experiment, '_verify_host_gpu'),
                patch.object(experiment, '_read', side_effect=read),
                patch.object(experiment, '_write', side_effect=write),
                patch.object(wsl_host, '_system_wsl', return_value=Path('wsl.exe')),
                patch.object(host_gpu, 'supervise', return_value=supervised),
                patch.object(host_gpu, 'STAGE_NAME', host_gpu.STAGE_NAME),
                redirect_stdout(io.StringIO()),
            ):
                stack.enter_context(context)
            with self.assertRaises(FileNotFoundError):
                experiment.run()

        receipt = json.loads(files[stage / 'host-result.local.json'])
        self.assertEqual(receipt['status'], 'failed')
        self.assertEqual(receipt['reason'], 'worker_validation')
        self.assertNotIn(stage / 'residual.local.npz', files)


if __name__ == '__main__':
    unittest.main()
