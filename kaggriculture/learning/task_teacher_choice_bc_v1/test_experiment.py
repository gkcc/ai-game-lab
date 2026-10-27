import hashlib
import json
import unittest
from unittest.mock import patch
from . import experiment
from .experiment import seed_values,verified_coverage_runtime


class TeacherExperimentTest(unittest.TestCase):
    def test_seed_inventory_ignores_outcome_values(self):
        metadata = {'train_seeds':[11,12], 'nested':{'seed':13},
                    'by_seed':{'34':{'score':999,'profit':-27}},
                    'score':123, 'excluded_uint32':[55,56],
                    'seeds':{'train':[71],'holdout':[72]}}
        self.assertEqual(seed_values(metadata),{11,12,13,34,55,56,71,72})
        self.assertEqual(seed_values({'train':[81],'test':[82]},True),{81,82})

    def test_parent_runtime_must_match_published_and_acceptance(self):
        old = json.dumps({'runtime_files':{'module.py':'a'*64}}).encode()
        accepted = {'evidence_sha256':{'freeze.json':hashlib.sha256(old).hexdigest()},
                    'resource_supervision_accepted':False}
        self.assertEqual(verified_coverage_runtime(old,old,accepted),{'module.py':'a'*64})
        changed = json.dumps({'runtime_files':{}}).encode()
        with self.assertRaises(ValueError): verified_coverage_runtime(changed,old,accepted)
        forged = {'evidence_sha256':{'freeze.json':hashlib.sha256(changed).hexdigest()},
                  'resource_supervision_accepted':False}
        with self.assertRaises(ValueError): verified_coverage_runtime(changed,changed,forged)
        with self.assertRaises(ValueError):
            verified_coverage_runtime(old,old,{**accepted,'resource_supervision_accepted':True})

    def test_windows_collection_requires_project_venv(self):
        with patch.object(experiment.os.path,'samefile',side_effect=[True,True,False]), \
             patch.object(experiment,'sha',return_value='a'*64):
            self.assertEqual(experiment.windows_venv_sha(),'a'*64)
        for answers in ([False],[True,False],[True,True,True]):
            with patch.object(experiment.os.path,'samefile',side_effect=answers):
                with self.assertRaises(ValueError): experiment.windows_venv_sha()
        with patch.object(experiment.os.path,'samefile',side_effect=OSError()):
            with self.assertRaises(ValueError): experiment.windows_venv_sha()

    def test_frozen_wsl_versions_match_worker_representation(self):
        from . import gpu_entry
        expected = experiment.WSL_SOFTWARE
        installed = {k:expected[k] for k in ('numpy','jax','optax')}
        with patch.object(gpu_entry.sys,'version_info',(3,12,3)), \
             patch.object(gpu_entry.importlib.metadata,'version',side_effect=lambda name:installed[name]):
            gpu_entry.check_software_versions(expected)
            with self.assertRaises(ValueError):
                gpu_entry.check_software_versions({**expected,'jax':'different'})


if __name__ == '__main__': unittest.main()
