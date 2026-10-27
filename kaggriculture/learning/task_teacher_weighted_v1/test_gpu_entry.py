import importlib.util
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch, mock_open
import numpy as np
from learning.task_choice_v1.features import OPS
from learning.task_teacher_choice_bc_v1 import numeric
from . import training
from .weights import operation_weights

exp = types.ModuleType('learning.task_teacher_weighted_v1.experiment')
exp.OUT = exp.ROOT = Path(__file__).resolve().parent/'synthetic'
for name in ('read','sha','ready','deadline_check','write_new'): setattr(exp,name,lambda *a: None)
spec = importlib.util.spec_from_file_location('learning.task_teacher_weighted_v1._entry_test',Path(__file__).with_name('gpu_entry.py'))
entry = importlib.util.module_from_spec(spec)
with patch.dict(sys.modules, {'learning.task_teacher_weighted_v1.experiment':exp}): spec.loader.exec_module(entry)


class EntryTests(unittest.TestCase):
    def fixture(self):
        arrays = dict(shared=np.zeros((5,275),np.float32), candidates=np.zeros((5,98),np.float32),
            offsets=np.arange(6,dtype=np.int64), teacher_choice_label=np.zeros(5,np.int64),
            ops=np.arange(5,dtype=np.uint8), seed=np.full(5,7,np.uint32))
        groups=[training.Group(arrays['shared'][i],arrays['candidates'][i:i+1],0,[OPS[i]],7) for i in range(5)]
        cfg=dict(wsl_software_versions={}, cache_sha256='cache', schema_path='schema.json',cache_path='cache.npz',
            schema_sha256='schema',gpu_launch_config_sha256='launch',old_train_seeds=[7],
            expected_operations={op:1 for op in OPS},expected_class_weights=operation_weights(groups),
            data_limits=dict(max_groups=5,max_candidates=5,max_archive_bytes=100,max_uncompressed_bytes=1000),
            training=dict(training_seed=1))
        return arrays,cfg

    def run_worker(self, change=None, drift=False, token='secret', session=True):
        arrays,cfg=self.fixture()
        if change: cfg.update(change)
        def read(path):
            return {'fit-started.json':dict(freeze_sha256='freeze',cache_sha256='cache',attempt=1,retry_allowed=False,host_token_sha256=entry.token_sha('secret')),
                    'gpu-guard-started.json':dict(attempt=1,freeze_sha256='freeze',fit_marker_sha256='marker',host_token_sha256=entry.token_sha('secret'),retry_allowed=False),
                    'gpu-launch.local.json':dict(expected_device='private',expected_device_kind='private'),
                    'schema.json':{}}[Path(path).name]
        marker_state=['marker']
        def sha(path): return {'freeze.json':'freeze','registration.json':'registration','pre-execution-commit.json':'commit',
            'fit-started.json':marker_state[0], 'gpu-guard-started.json':'guard', 'gpu-launch.local.json':'launch','schema.json':'schema','model-final.npz':'model'}[Path(path).name]
        parity=dict(groups=16,synchronized=True,requested_matmul_precision='highest',hardware_precision_measured=False)
        result=dict(train_seeds={7},sha256='model',training_summary=dict(synchronized_updates=128,gpu_verified=True,
                    weights_changed=True,nonzero_gradient_updates=1,inference_parity=parity))
        fake_os=types.SimpleNamespace(name='posix',environ={entry.TOKEN_ENV:token} if token else {},getsid=lambda x:1 if session else 2,getpid=lambda:1)
        with patch.object(entry,'os',fake_os),patch.object(entry,'ready',return_value=cfg),patch.object(entry,'read',side_effect=read), \
             patch.object(entry,'sha',side_effect=sha),patch.object(entry,'check_software_versions'), \
             patch.object(entry,'deadline_check'),patch.object(entry,'write_new') as write, \
             patch.object(numeric,'read',return_value=arrays) as decode,patch.object(training,'fit') as fit:
            def run_fit(*args):
                if drift: marker_state[0]='changed'
                return result
            fit.side_effect=run_fit
            if change or drift or not token or not session or token != 'secret':
                with self.assertRaises(ValueError): entry._worker_impl(exp.OUT,lambda x:None)
                if drift: fit.assert_called_once()
                else: fit.assert_not_called()
                write.assert_not_called()
            else:
                entry._worker_impl(exp.OUT,lambda x:None)
                fit.assert_called_once(); write.assert_called_once()
            if not token or not session or token != 'secret': decode.assert_not_called()
            else: decode.assert_called_once()

    def test_success_and_input_rejections(self):
        self.run_worker()
        self.run_worker({'old_train_seeds':[8]})
        self.run_worker({'expected_operations':{op:0 for op in OPS}})
        self.run_worker({'expected_class_weights':{op:2 for op in OPS}})
        self.run_worker(drift=True)
        self.run_worker(token=None)
        self.run_worker(token='wrong')
        self.run_worker(session=False)

    def test_bad_versions_stop_before_decode(self):
        with patch.object(entry,'ready',return_value={'wsl_software_versions':{}}), \
             patch.object(entry,'check_software_versions',side_effect=ValueError),patch.object(numeric,'read') as decode:
            with self.assertRaises(ValueError): entry._worker_impl(exp.OUT,lambda x:None)
            decode.assert_not_called()

    def test_direct_worker_missing_guard_stops_before_decode(self):
        _,cfg=self.fixture()
        def sha(path):
            if Path(path).name == 'gpu-guard-started.json': raise FileNotFoundError()
            return 'identity'
        with patch.object(entry,'ready',return_value=cfg),patch.object(entry,'check_software_versions'), \
             patch.object(entry,'sha',side_effect=sha),patch.object(numeric,'read') as decode:
            with self.assertRaises(FileNotFoundError): entry._worker_impl(exp.OUT,lambda x:None)
            decode.assert_not_called()

    def test_guard_failure_never_reads_fit_result(self):
        import learning.task_dagger_once_v1.gpu_guard as gpu_guard
        cfg={'cache_sha256':'cache','training':{'work_seconds':1,'process_seconds':3}}
        fake=mock_open()
        marker=dict(attempt=1,retry_allowed=False,freeze_sha256='freeze',cache_sha256='cache',host_token_sha256=entry.token_sha('secret'))
        with patch.object(entry,'ready',return_value=cfg),patch.object(entry,'os',types.SimpleNamespace(name='posix',environ={})), \
             patch.object(Path,'exists',return_value=False),patch.object(Path,'open',fake), \
             patch.object(entry,'sha',return_value='freeze'),patch.object(entry,'write_new'), \
             patch.object(entry,'read',return_value=marker) as read,patch.object(gpu_guard,'supervise',side_effect=RuntimeError):
            with self.assertRaises(RuntimeError): entry.guard(exp.OUT,'secret')
            self.assertEqual(read.call_count,1)

    def test_guard_rejects_changed_marker_after_worker(self):
        import learning.task_dagger_once_v1.gpu_guard as gpu_guard
        state=['marker']
        flags=dict(completed=True,owned_process_group_signalled=True,direct_child_reaped=True,
                   no_live_owned_session_members_before_reap=True)
        def sha(path): return state[0] if Path(path).name == 'gpu-guard-started.json' else 'freeze'
        def supervise(*args): state[0]='changed'; return flags
        marker=dict(attempt=1,retry_allowed=False,freeze_sha256='freeze',cache_sha256='cache',host_token_sha256=entry.token_sha('secret'))
        def read(path): return marker if Path(path).name == 'fit-started.json' else {'passed':True}
        with patch.object(entry,'ready',return_value={'cache_sha256':'cache','training':{'work_seconds':1,'process_seconds':3}}), \
             patch.object(entry,'os',types.SimpleNamespace(name='posix',environ={})),patch.object(Path,'exists',return_value=False), \
             patch.object(Path,'open',mock_open()),patch.object(entry,'sha',side_effect=sha), \
             patch.object(entry,'read',side_effect=read),patch.object(entry,'write_new') as write, \
             patch.object(gpu_guard,'supervise',side_effect=supervise):
            with self.assertRaises(RuntimeError): entry.guard(exp.OUT,'secret')
            self.assertFalse(write.call_args.args[1]['passed'])

    def test_guard_receipt_identity_and_missing_marker_hash(self):
        import learning.task_dagger_once_v1.gpu_guard as gpu_guard
        flags=dict(completed=True,owned_process_group_signalled=True,direct_child_reaped=True,
                   no_live_owned_session_members_before_reap=True)
        for fail_marker_hash in (False, True):
            env={entry.TOKEN_ENV:'old-value'}
            marker=dict(attempt=1,retry_allowed=False,freeze_sha256='freeze-sha',cache_sha256='cache',host_token_sha256=entry.token_sha('secret'))
            def read(path): return marker if Path(path).name == 'fit-started.json' else {'passed':True}
            def sha(path):
                if Path(path).name == 'gpu-guard-started.json':
                    if fail_marker_hash: raise OSError('synthetic')
                    return 'marker-sha'
                return 'freeze-sha'
            with self.subTest(fail_marker_hash=fail_marker_hash), \
                 patch.object(entry,'ready',return_value={'cache_sha256':'cache','training':{'work_seconds':1,'process_seconds':3}}), \
                 patch.object(entry,'os',types.SimpleNamespace(name='posix',environ=env)),patch.object(Path,'exists',return_value=False), \
                 patch.object(Path,'open',mock_open()),patch.object(entry,'sha',side_effect=sha), \
                 patch.object(entry,'read',side_effect=read),patch.object(entry,'deadline_check'), \
                 patch.object(entry,'write_new') as write,patch.object(gpu_guard,'supervise',return_value=flags) as supervise:
                if fail_marker_hash:
                    with self.assertRaises(RuntimeError): entry.guard(exp.OUT,'secret')
                    supervise.assert_not_called()
                else:
                    entry.guard(exp.OUT,'secret')
                    supervise.assert_called_once()
                receipt=write.call_args.args[1]
                self.assertEqual(receipt['freeze_sha256'],'freeze-sha')
                self.assertEqual(receipt['guard_marker_sha256'],None if fail_marker_hash else 'marker-sha')
                self.assertEqual(receipt['passed'],not fail_marker_hash)
                self.assertEqual(receipt['marker'],{'created':True,'complete':True})
                self.assertEqual(env[entry.TOKEN_ENV],'old-value')


if __name__ == '__main__': unittest.main()
