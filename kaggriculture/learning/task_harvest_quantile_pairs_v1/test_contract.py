"""Stable source identity, missing slots and publication boundaries."""
from copy import deepcopy
import unittest
from learning.task_harvest_return_pairs_v1 import contract as previous
from . import contract


def ledger():
    return [dict(origin=o,point=q,split=contract.split(o),status='paired',technical_passed=True,
        H=dict(own_net_profit=20,opponent_net_profit=10,margin=10,score=1),
        W=dict(own_net_profit=15,opponent_net_profit=11,margin=4,score=1),
        seed=17,input_sha256='private',private_inputs={'unsafe':True})
        for o in contract.ORIGINS for q in ('Q0',*contract.POINTS)]


class FrozenContract(unittest.TestCase):
    def test_official_and_teacher_identity_is_the_same_frozen_version(self):
        for key in ('ENGINE_COMMIT','ENGINE_SHA','CORE_SHA','SPEC_SHA','UTILS_SHA','AGENT_SHA','TEACHER_SHA'):
            with self.subTest(key=key):self.assertEqual(getattr(contract,key),getattr(previous,key))

    def test_references_are_reused_and_private_fields_not_projected(self):
        public=contract.projection(ledger())
        self.assertEqual(public['reused_pairs'],4);self.assertEqual(public['completed_new_pairs'],12)
        self.assertEqual(sum(row['reused'] for row in public['rows']),4)
        self.assertFalse(public['training_admission'])
        for row in public['rows']:
            self.assertEqual(set(row),{'origin','point','split','cohort','opponent','side','reused','status','reason','returns'})

    def test_missing_new_slot_is_not_zero_return_or_replacement(self):
        rows=ledger();rows[1].update(status='no_eligible_target',reason='duplicate_index')
        result=contract.projection(rows)
        self.assertEqual(result['completed_new_pairs'],11);self.assertFalse(result['all_twelve_new_pairs_completed'])
        self.assertIsNone(result['rows'][1]['returns'])
        rows[0].update(status='no_eligible_target',reason='no_eligible_target')
        with self.assertRaises(ValueError):contract.projection(rows)

    def test_source_group_and_cash_targets_are_not_silently_repaired(self):
        for alter in ('split','point','cash','score','nan'):
            rows=ledger()
            if alter=='split':rows[0]['split']='diagnostic'
            elif alter=='point':rows[1]['point']='Q2'
            elif alter=='cash':rows[1]['H']['margin']+=1
            elif alter=='score':rows[1]['H']['score']=0
            else:rows[1]['W']['own_net_profit']=float('nan')
            with self.subTest(alter=alter),self.assertRaises(ValueError):contract.projection(rows)


if __name__=='__main__':unittest.main()
