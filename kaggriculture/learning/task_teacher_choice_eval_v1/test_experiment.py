import unittest
from unittest.mock import patch
from . import experiment


class EvalExperimentTest(unittest.TestCase):
    def test_prior_plan_and_seed_role_are_fixed(self):
        plan = dict(experiment.OFFICIAL)
        plan.pop('per_job_seconds')
        plan.update(experiment.GATES)
        plan.update(opponents=['round2'],seats=[0,1])
        parent = dict(evaluation_plan=plan,reserved_test_seeds=list(experiment.SEEDS))
        self.assertEqual(experiment.prior_plan(parent),plan)
        with self.assertRaises(ValueError):
            experiment.prior_plan({**parent,'reserved_test_seeds':list(reversed(experiment.SEEDS))})
        with self.assertRaises(ValueError):
            experiment.prior_plan({**parent,'evaluation_plan':{**plan,'seat_own_min':-3000}})

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
