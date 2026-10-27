import importlib.util
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch
import numpy as np
from . import numeric, training

exp = types.ModuleType('learning.task_teacher_choice_bc_v1.experiment')
exp.OUT = Path(__file__).resolve().parent/'synthetic'
for name in ('deadline_check','read','ready','sha','write_new'): setattr(exp,name,lambda *args:None)
spec = importlib.util.spec_from_file_location('learning.task_teacher_choice_bc_v1._gpu_entry_test',Path(__file__).with_name('gpu_entry.py'))
entry = importlib.util.module_from_spec(spec)
with patch.dict(sys.modules,{'learning.task_teacher_choice_bc_v1.experiment':exp}): spec.loader.exec_module(entry)


class EntryTest(unittest.TestCase):
    def test_software_version_identity(self):
        versions=dict(numpy='1',jax='2',optax='3')
        expected=dict(python='.'.join(str(v) for v in sys.version_info[:3]),**versions)
        with patch.object(entry.importlib.metadata,'version',side_effect=lambda name:versions[name]):
            entry.check_software_versions(expected)
            with self.assertRaises(ValueError): entry.check_software_versions({**expected,'jax':'other'})

    def test_teacher_labels_and_no_test_access(self):
        cfg = dict(wsl_software_versions={},source_manifest_sha256='sources',sampling_algorithm='fixed',schema_sha256='schema',gpu_launch_config_sha256='launch',train_seeds=[7],
            data_limits=dict(max_groups=1,max_candidates=2,max_archive_bytes=100,max_uncompressed_bytes=1000),
            training=dict(training_seed=9))
        arrays = dict(shared=np.zeros((1,275),np.float32),candidates=np.zeros((2,98),np.float32),
            offsets=np.array([0,2]),teacher_choice_label=np.array([1]),ops=np.array([0,1]),seed=np.array([7]))
        names=[]; saved={}; seen=[]
        def read(path):
            names.append(Path(path).name)
            return {'fit-started.json':dict(train_data_sha256='data',freeze_sha256='freeze',array_sha256='cache'),
                'train-data-result.json':dict(passed=True,array_sha256='cache',freeze_sha256='freeze',collection_attempt=1,
                    source_manifest_sha256='sources',sampling_algorithm='fixed'),
                'gpu-launch.local.json':dict(expected_device='gpu',expected_device_kind='kind'),
                'schema.json':{}}[Path(path).name]
        def sha(path):
            return {'fit-started.json':'unused','train-data-result.json':'data','freeze.json':'freeze',
                'train.npz':'cache','schema.json':'schema','gpu-launch.local.json':'launch','model-final.npz':'model'}[Path(path).name]
        def fit(groups,*args):
            seen.extend(groups)
            return dict(sha256='model',baseline={},train_seeds={7},training_summary=dict(synchronized_updates=128,
                gpu_verified=True,weights_changed=True,nonzero_gradient_updates=128,
                inference_parity=dict(groups=16,synchronized=True,requested_matmul_precision='highest',hardware_precision_measured=False)))
        with patch.object(entry,'ready',return_value=cfg),patch.object(entry,'check_software_versions'),patch.object(entry,'read',side_effect=read),patch.object(entry,'sha',side_effect=sha), \
             patch.object(entry,'deadline_check'),patch.object(entry,'write_new',side_effect=lambda p,v:saved.update({Path(p).name:v})), \
             patch.object(numeric,'read',return_value=arrays),patch.object(training,'fit',side_effect=fit):
            entry._worker_impl(exp.OUT,lambda _:None)
        self.assertEqual(seen[0]['teacher_choice_label'],1)
        self.assertNotIn('status',seen[0])
        self.assertFalse(any('test' in name for name in names))
        self.assertTrue(saved['fit-result.json']['conditional_teacher_labels'])

    def test_bad_input_fails_before_decode(self):
        cfg=dict(wsl_software_versions={},source_manifest_sha256='sources',sampling_algorithm='fixed')
        marker=dict(train_data_sha256='data',freeze_sha256='freeze',array_sha256='cache')
        source=dict(passed=True,array_sha256='cache',freeze_sha256='freeze',collection_attempt=1,
                    source_manifest_sha256='sources',sampling_algorithm='fixed')
        for change in (dict(array_sha256='other'),dict(source_manifest_sha256='other'),
                       dict(collection_attempt=2),dict(freeze_sha256='other')):
            with self.subTest(change=change), patch.object(entry,'ready',return_value=cfg), \
                 patch.object(entry,'check_software_versions'),patch.object(entry,'read',side_effect=[marker,{**source,**change}]), \
                 patch.object(entry,'sha',side_effect=lambda p:'data' if Path(p).name=='train-data-result.json' else 'freeze'), \
                 patch.object(numeric,'read') as decode:
                with self.assertRaises(ValueError): entry._worker_impl(exp.OUT,lambda _:None)
                decode.assert_not_called()


if __name__ == '__main__': unittest.main()
