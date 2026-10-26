"""Small integration contracts without consuming an official experiment seed."""
import unittest
from unittest.mock import patch
from pathlib import Path
import hashlib
import json
from learning.task_reward_gate_v1.campaign import plan_cells,plan
from learning.task_reward_gate_v1.experiment import seeds,gate_identity,line_endings_only
from learning.task_reward_gate_v1.policy import GateSelector,DEFAULT_LIMITS,GateModel


class CampaignTests(unittest.TestCase):
    def test_published_source_allows_only_crlf_lf_conversion(self):
        self.assertTrue(line_endings_only(b'one\r\ntwo\r\n',b'one\ntwo\n'))
        self.assertFalse(line_endings_only(b'one\r\ntwo\r\n',b'one\nchanged\n'))
        self.assertFalse(line_endings_only(b'one\r\ntwo\r\n',b'one\ntwo\n\n'))

    def test_disjoint_population_and_reserved_window(self):
        cfg={**seeds(),'opponents':['round2','market']}
        values=[v for key in ('train_seeds','validation_seeds','test_seeds') for v in cfg[key]]
        self.assertEqual(len(set(values)),18)
        counts={split:len(plan_cells(cfg,split)) for split in ('train','dev','holdout')}
        self.assertEqual(counts,dict(train=32,dev=8,holdout=32))
        self.assertEqual(2*sum(counts.values()),144)
        cells=plan_cells(cfg,'train')
        self.assertEqual([r['window'] for r in cells[:4]],[0,1,1,2])
        self.assertEqual(sum(plan(c,'baseline')['baseline_parity'] for c in cells),2)
        self.assertFalse(any(plan(c,'intervention')['baseline_parity'] for c in cells))
        self.assertTrue(all(c['window'] is None for c in plan_cells(cfg,'holdout')))

    def test_declared_compute_ceiling_covers_all_primary_work(self):
        # 288 policy episodes (live + reconstruction), at most136 one-turn
        # shadows and two full old-student parity episodes. Independent replay
        # uses recorded actions and therefore adds no model forwards.
        self.assertLessEqual(288*4000+136*32+2*4000,1200000)
        self.assertLessEqual(288*140000+136*2000+2*140000,42000000)
        self.assertLessEqual(144*4*719+136+2*719,416000)
        self.assertEqual((144+144+4)*719,209948)

    def test_limit_contract_accepts_registration_keys(self):
        selector=GateSelector(lambda _value: None,limits=DEFAULT_LIMITS)
        self.assertEqual(selector.counts['task_forwards'],0)

    def test_fitted_identity_rejects_replaced_gate(self):
        fixture={'gate-model-freeze.json':dict(fit_count=1,training_pairs_sha256='pairs',gate_sha256='fitted'),
                 'fit-result.json':dict(gate_sha256='fitted'),
                 'development-gate.json':dict(gate_sha256='fitted')}
        identities={'train-pairs.json':'pairs','gate-model.json':'fitted'}
        with patch('learning.task_reward_gate_v1.experiment.read',side_effect=lambda p:fixture[p.name]), \
             patch('learning.task_reward_gate_v1.experiment.sha',side_effect=lambda p:identities[p.name]), \
             patch.object(Path,'exists',return_value=True):
            self.assertEqual(gate_identity(Path('synthetic')), 'fitted')
            identities['gate-model.json']='replacement'
            with self.assertRaises(AssertionError):gate_identity(Path('synthetic'))

    def test_deployment_load_rejects_valid_but_different_weights(self):
        model=GateModel([0,0],[1,1],[0]*12,0)
        raw=json.dumps(model.to_dict()).encode();expected=hashlib.sha256(raw).hexdigest()
        with patch.object(Path,'read_bytes',return_value=raw):
            self.assertEqual(GateModel.load('synthetic',expected).predict([0]*12),0)
        changed=model.to_dict();changed['intercept']=1
        with patch.object(Path,'read_bytes',return_value=json.dumps(changed).encode()):
            with self.assertRaises(ValueError):GateModel.load('synthetic',expected)


if __name__=='__main__':unittest.main()
