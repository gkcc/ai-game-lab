import unittest
from unittest.mock import patch
from . import experiment as e


class ExperimentTest(unittest.TestCase):
    def test_seeds(self):
        seeds = e.select_seeds(e.EXCLUDED)
        self.assertEqual(len(set(seeds)),2)
        self.assertFalse(set(seeds) & e.EXCLUDED)
        self.assertFalse(set(e.select_seeds(set(seeds))) & set(seeds))

    def test_identity_and_head(self):
        with patch.object(e,'sha',return_value='changed'), self.assertRaises(ValueError): e.check_identity('unused','expected')
        with patch.object(e,'git',side_effect=lambda *a: b'base\n' if a[0]=='rev-parse' else b' M file'), self.assertRaises(ValueError): e.clean_head('base')

    def test_deadline(self):
        for value in ('2000-01-01T00:00:00+00:00','2999-01-01T00:00:00'):
            with patch.object(e,'read',return_value={'deadline_utc':value}), self.assertRaises(TimeoutError): e.deadline_check()

    def test_budget_and_gate(self):
        cfg = dict(official_limits=dict(new_games=8,live_transitions=8*719,primary_replay_transitions=8*719,
            independent_replay_transitions=719,total_transitions=17*719),opponents=['round2'],seats=[0,1],
            policies=['candidate','teacher_task'],acceptance=dict(pair_own_income_delta_min=-2000,
            pair_cash_margin_delta_min=-2000,seed_mean_own_income_delta_min=-1000,seed_mean_cash_margin_delta_min=-1000))
        e.validate_budget(cfg)
        cfg['acceptance']['pair_cash_margin_delta_min'] = -2001
        with self.assertRaises(ValueError): e.validate_budget(cfg)


if __name__ == '__main__': unittest.main()
