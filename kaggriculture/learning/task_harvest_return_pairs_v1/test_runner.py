"""Pure aggregate contracts: missing targets cannot become successful pairs."""
from copy import deepcopy
import unittest
from . import contract,runner


def records():
    cost=dict(task_forwards=2000,task_rows=6000,crop_forwards=300,crop_rows=300,
        gate_evaluations=0,controller_calls=1438,teacher_proposals=0)
    return [dict(row=dict(role=r,status='paired'),model_costs=deepcopy(cost),official_actions=100)
            for r in contract.ROLES]


class CampaignCounts(unittest.TestCase):
    def test_all_costs_include_both_arms_and_prefix_reconstruction(self):
        value=runner.aggregate_costs(records(),contract)
        self.assertEqual(value['model_costs']['controller_calls'],5752)
        self.assertEqual(value['model_costs']['task_forwards'],8000)
        self.assertEqual(value['official_actions'],400)
        self.assertEqual((value['environment_constructions'],value['restores'],value['explicit_resets']),(8,8,0))
        self.assertEqual(value['verification_reconstructed_replay_decodes'],8)

    def test_missing_case_retained_and_has_zero_execution(self):
        rows=records();rows[2]['row']['status']='no_eligible_target'
        rows[2]['model_costs']={k:0 for k in rows[2]['model_costs']};rows[2]['official_actions']=0
        value=runner.aggregate_costs(rows,contract)
        self.assertEqual((value['paired'],value['terminal_suffixes'],value['rival_calls']),(3,6,4314))
        rows[2]['model_costs']['crop_forwards']=1
        with self.assertRaises(ValueError):runner.aggregate_costs(rows,contract)

    def test_extra_trajectory_unknown_role_boolean_or_oversized_cost_fails(self):
        for key,value in [('controller_calls',2157),('controller_calls',1438.),('crop_forwards',True),
                          ('task_forwards',32001),('teacher_proposals',1)]:
            rows=records();rows[0]['model_costs'][key]=value
            with self.subTest(key=key,value=value),self.assertRaises(ValueError):runner.aggregate_costs(rows,contract)
        rows=records();rows.reverse()
        with self.assertRaises(ValueError):runner.aggregate_costs(rows,contract)
        rows=records();rows[0]['official_actions']=99
        with self.assertRaises(ValueError):runner.aggregate_costs(rows,contract)

    def test_marker_binds_role_and_cannot_request_retry(self):
        marker=runner.marker('permit','a'*40,'b'*64,'F02')
        self.assertEqual(marker['role'],'F02');self.assertFalse(marker['retry_allowed'])
        with self.assertRaises(ValueError):runner.marker('permit','a'*40,'b'*64,'F05')


if __name__=='__main__':unittest.main()
