import io
import json
import unittest
from unittest.mock import patch
from . import campaign,independent_method,independent_runner


class IndependentTest(unittest.TestCase):
    def test_sanitized_ledger_binds_complete_receipt(self):
        row=dict(game='g',seed=1,opponent='round2',side=0,policy='weighted',split='holdout',parity_kind=None,
            passed=True,states=720,action_turns=719,status=['DONE','DONE'],agent_errors=0,
            bank=11,opponent_bank=10,initial_bank=3,opponent_initial_bank=3,net_profit=8,
            opponent_net_profit=7,margin=1,score=1.,trace_sha256='private-digest')
        summary=campaign.sanitized(row)
        audit=dict(game='g',receipt_sha256=dict(row='a'*64,audit='b'*64))
        independent_method.bind_selected(summary,row,audit,'a'*64,'b'*64)
        for altered in ({**audit,'game':'other'}, {**audit,'receipt_sha256':{'row':'c'*64,'audit':'b'*64}}):
            with self.subTest(altered=altered),self.assertRaises(ValueError):
                independent_method.bind_selected(summary,row,altered,'a'*64,'b'*64)
        with self.assertRaises(ValueError): independent_method.bind_selected({**summary,'bank':12},row,audit,'a'*64,'b'*64)

    def test_hash_sample_is_fixed_and_rejects_duplicate(self):
        values=[('game',i,0) for i in range(30)]
        self.assertEqual(independent_method.sample_identities(values),independent_method.sample_identities(values[::-1]))
        with self.assertRaises(ValueError): independent_method.sample_identities(values+[values[0]])

    def test_low_preflight_never_launches_replay(self):
        writes=[]
        class Stream(io.StringIO):
            def close(self): pass
        with patch.object(independent_runner,'ready',return_value={}),patch.object(independent_runner,'verify_commit'), \
            patch.object(campaign,'validate_panel',return_value=[]), \
            patch.object(independent_runner.Path,'exists',return_value=False), \
            patch.object(independent_runner.Path,'open',return_value=Stream()), \
            patch.object(independent_runner,'sha',return_value='a'*64), \
            patch.object(independent_runner,'memory_ready',return_value=False), \
            patch.object(independent_runner,'write_new',side_effect=lambda p,v:writes.append((p.name,v))), \
            patch.object(independent_runner,'supervise') as launch:
            with self.assertRaisesRegex(RuntimeError,'preflight memory'): independent_runner.run()
        launch.assert_not_called()
        self.assertFalse(dict(writes)['independent-runner-failure.json']['retry_allowed'])


if __name__=='__main__': unittest.main()
