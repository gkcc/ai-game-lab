"""The worker retains its failure phase without accessing real files or devices."""
from pathlib import Path
import unittest
from unittest.mock import patch
from learning.task_completion_bc_v1 import numeric
from learning.task_bc_v1 import model
from . import run, experiment


class WorkerFailureTests(unittest.TestCase):
    def test_cache_decode_failure_records_phase_and_zero_forwards(self):
        cfg = {'checkpoint_sha256':'m', 'training_cache_sha256':'c'}
        captured = {}
        def read(path):
            if Path(path).name == 'run-started.json':
                return {'freeze_sha256':'f'}
            return {'data_limits':{'max_groups_per_split':10,'max_candidates_per_split':20},'train_seeds':[1]}
        def save(path, value):
            captured[Path(path).name] = value
        with patch.object(run,'ready',return_value=cfg), patch.object(run,'read',side_effect=read), \
                patch.object(run,'sha',return_value='f'), patch.object(run,'write_new',side_effect=save), \
                patch.object(numeric,'read',side_effect=ValueError('synthetic')):
            with self.assertRaises(ValueError):
                run.worker(Path('synthetic-not-created'))
        result = captured['worker-failure.json']
        self.assertEqual(result['phase'],'cache_decode')
        self.assertEqual(result['progress']['completed_cpu_groups'],0)
        self.assertEqual(result['progress']['completed_gpu_groups'],0)
        self.assertFalse(result['retry_allowed'])

    def test_model_change_after_reload_stops_before_gpu(self):
        cfg = {'checkpoint_sha256':'original', 'training_cache_sha256':'cache'}
        captured = {}
        def read(path):
            if Path(path).name == 'run-started.json':
                return {'freeze_sha256':'f'}
            return {'data_limits':{'max_groups_per_split':10,'max_candidates_per_split':20},'train_seeds':[1]}
        def digest(path):
            return {'freeze.json':'f','train.npz':'cache','model-final.npz':'changed'}[Path(path).name]
        with patch.object(run,'ready',return_value=cfg), patch.object(run,'read',side_effect=read), \
                patch.object(run,'sha',side_effect=digest), \
                patch.object(run,'write_new',side_effect=lambda path,value:captured.update({Path(path).name:value})), \
                patch.object(numeric,'read',return_value={}), patch.object(numeric,'unpack',return_value=[{'seed':1}]), \
                patch.object(model,'TaskNetwork',return_value=object()):
            with self.assertRaises(ValueError):
                run.worker(Path('synthetic-not-created'))
        self.assertEqual(captured['worker-failure.json']['phase'],'model_reload')
        self.assertEqual(captured['worker-failure.json']['progress']['completed_gpu_groups'],0)

    def test_cleanup_requires_reader_and_child(self):
        ok = {'owned_instances_remaining':0,'reader_joined':True,'launched_process_reaped':True}
        self.assertTrue(run.cleanup_complete(ok))
        for key in ('reader_joined','launched_process_reaped'):
            self.assertFalse(run.cleanup_complete(dict(ok,**{key:False})))

    def test_previous_registration_must_match_original_freeze(self):
        with patch.object(experiment,'sha',return_value='changed'):
            with self.assertRaises(ValueError):
                experiment.validate_prior_registration(Path('unused'),{'registration_sha256':'original'})


if __name__ == '__main__':
    unittest.main()
