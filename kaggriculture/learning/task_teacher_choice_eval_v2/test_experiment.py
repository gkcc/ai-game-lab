import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from . import experiment


class EvalExperimentTest(unittest.TestCase):
    def test_prior_plan_and_seed_role_are_fixed(self):
        plan = dict(experiment.OFFICIAL)
        plan.pop('per_job_seconds')
        plan.update(experiment.GATES)
        plan.update(opponents=['round2'],seats=[0,1])
        parent = dict(evaluation_plan=plan,reserved_test_seeds=list(experiment.CONSUMED))
        self.assertEqual(experiment.prior_plan(parent),plan)
        with self.assertRaises(ValueError):
            experiment.prior_plan({**parent,'reserved_test_seeds':[1,2]})
        with self.assertRaises(ValueError):
            experiment.prior_plan({**parent,'evaluation_plan':{**plan,'seat_own_min':-3000}})

    def test_explicit_seed_rejections(self):
        experiment.validate_seeds([1,2],{3})
        for seeds in ([1,1],[2941010634,1],[3774915423,1],[1,3],[-1,2]):
            with self.subTest(seeds=seeds),self.assertRaises(ValueError): experiment.validate_seeds(seeds,{3})

    def test_seed_inventory_reads_only_allowed_metadata(self):
        public_reg=experiment.PUBLIC/'review'/'registration.json'
        public_game=experiment.PUBLIC/'review'/'job-inputs'/'g.json'
        public_result=experiment.PUBLIC/'review'/'games-result.json'
        local_reg=experiment.ROOT/'artifacts'/'prior'/'seeds.json'
        local_replay=experiment.ROOT/'artifacts'/'prior'/'replay.json'
        current=experiment.OUT/'root-decision.json'
        private_reg=experiment.ROOT/'artifacts'/'prior'/'private'/'registration.json'
        public_private=experiment.PUBLIC/'review'/'private'/'registration.json'
        data={public_reg:b'{"seed":123456}',public_game:b'{"game":"holdout-234567-round2-0-candidate"}',
              local_reg:b'{"seeds":["345678",456789]}',public_result:b'987654',local_replay:b'876543',current:b'765432',
              private_reg:b'654321',public_private:b'543210'}
        accessed=[]
        def read_bytes(path): accessed.append(path); return data[path]
        tracked='review/registration.json\0review/job-inputs/g.json\0review/games-result.json\0review/private/registration.json\0'
        with patch.object(experiment,'git',return_value=tracked.encode()),patch.object(experiment,'deadline_check'), \
             patch.object(Path,'rglob',return_value=[local_reg,local_replay,current,private_reg]), \
             patch.object(Path,'stat',autospec=True,side_effect=lambda p:SimpleNamespace(st_size=len(data[p]))), \
             patch.object(Path,'read_bytes',autospec=True,side_effect=read_bytes):
            result=experiment.seed_inventory()
        self.assertTrue({123456,234567,345678,456789} <= set(result['excluded_uint32']))
        self.assertTrue(experiment.CONSUMED <= set(result['excluded_uint32']))
        self.assertEqual(set(accessed),{public_reg,public_game,local_reg})

    def test_windows_worker_requires_project_venv(self):
        with patch.object(experiment.os.path,'samefile',side_effect=[True,True,False]), \
             patch.object(experiment,'sha',return_value='a'*64):
            self.assertEqual(experiment.windows_venv_sha(),'a'*64)
        for values in ([False],[True,False],[True,True,True]):
            with patch.object(experiment.os.path,'samefile',side_effect=values):
                with self.assertRaises(ValueError): experiment.windows_venv_sha()
        with patch.object(experiment.os.path,'samefile',side_effect=OSError()):
            with self.assertRaises(ValueError): experiment.windows_venv_sha()


if __name__ == '__main__': unittest.main()
