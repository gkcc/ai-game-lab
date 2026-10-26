"""Stable dispatch, identity and budget contracts without official games."""
from pathlib import Path
import unittest
from unittest.mock import patch

from learning.task_request_gate_v1.campaign import Runner,plan_cells,plan
from learning.task_request_gate_v1.experiment import LIMITS,seeds,gate_identity
from learning.task_request_gate_v1.policy import DEFAULT_LIMITS


def cfg():
    return dict(seeds(),opponents=['round2','market'],holdout_arms=['baseline','gated','teacher_task'],
        game_limits=LIMITS,task_forward_budget=1200000,crop_forward_budget=3000000,
        neural_input_row_budget=60000000,gate_evaluation_budget=1000,policy_turn_budget=600000)


class CampaignTests(unittest.TestCase):
    def test_frozen_population_and_pre_dispatch_bound_includes_all_work(self):
        c=cfg();r=Runner(Path('synthetic'),c)
        values=[s for key in ('train_seeds','validation_seeds','test_seeds') for s in c[key]]
        self.assertEqual(len(set(values)),18)
        jobs=[plan(cell,arm) for split in ('train','dev','holdout') for cell in plan_cells(c,split)
              for arm in (c['holdout_arms'] if split=='holdout' else ('baseline','intervention'))]
        self.assertEqual(len(jobs),176)
        self.assertEqual(LIMITS,DEFAULT_LIMITS)
        for job in jobs:r.reserve(job);r.attempts.append(job)
        self.assertEqual(sum(p['policy']=='teacher_task' for p in jobs),32)
        self.assertEqual(sum(p['parity_kind']=='student' for p in jobs),2)
        self.assertEqual(sum(p['parity_kind']=='teacher_task' for p in jobs),1)
        with self.assertRaises(AssertionError):r.reserve(jobs[-1])
        self.assertEqual((176*2+4)*719,255964)

    def test_no_third_seed_is_dispatched_after_information_stop(self):
        c=cfg();written={}
        class FakeRunner(Runner):
            def execute(self,jobs):
                self.attempts.extend(jobs)
                return {j['game']:j for j in jobs}
        def fake_pair(_out,a,b):
            return dict(seed=a['seed'],opponent=a['opponent'],side=a['side'],baseline_net_profit=40000,
                        eligible=True,x8=[0.]*8,delta_own=0.,delta_opponent=0.,delta_margin=0.,delta_score=0.)
        r=FakeRunner(Path('synthetic'),c)
        with patch('learning.task_request_gate_v1.campaign.read',return_value={'opportunities':[{}]}), \
             patch('learning.task_request_gate_v1.campaign.pair',side_effect=fake_pair), \
             patch('learning.task_request_gate_v1.campaign.write_new',side_effect=lambda p,v:written.__setitem__(p.name,v)):
            pairs,passed=r.collect('train')
        self.assertFalse(passed)
        self.assertEqual(len(pairs),8)
        self.assertEqual(len(r.attempts),16)
        self.assertEqual({p['seed'] for p in r.attempts},set(c['train_seeds'][:2]))
        self.assertFalse(written['early-signal.json']['passed'])

    def test_fitted_gate_identity_cannot_follow_replaced_file(self):
        receipt={'gate-model-freeze.json':dict(fit_count=1,training_pairs_sha256='pairs',gate_sha256='fixed'),
                 'fit-result.json':dict(gate_sha256='fixed'),'development-gate.json':dict(gate_sha256='fixed')}
        hashes={'train-pairs.json':'pairs','gate-model.json':'fixed'}
        with patch('learning.task_request_gate_v1.experiment.read',side_effect=lambda p:receipt[p.name]), \
             patch('learning.task_request_gate_v1.experiment.sha',side_effect=lambda p:hashes[p.name]), \
             patch.object(Path,'exists',return_value=True):
            self.assertEqual(gate_identity(Path('synthetic')),'fixed')
            hashes['gate-model.json']='replacement'
            with self.assertRaises(AssertionError):gate_identity(Path('synthetic'))


if __name__=='__main__':unittest.main()
