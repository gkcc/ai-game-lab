import unittest
from pathlib import Path
from unittest.mock import patch
from . import independent_method as method
from learning.task_completion_bc_v1 import collect
from learning.task_action_impact_diagnostic_v1 import analysis


class IndependentTests(unittest.TestCase):
    def test_sample_is_fixed_and_result_independent(self):
        ids = [('g', i, 0) for i in range(30)]
        self.assertEqual(method.sample_identities(ids), method.sample_identities(list(reversed(ids))))
        self.assertEqual(len(method.sample_identities(ids)), 20)
        for invalid in (ids[:19], ids + ids[:1]):
            with self.assertRaises(ValueError): method.sample_identities(invalid)

    def test_sample_counts_show_absent_operations(self):
        ids = [('g', i, 0) for i in range(20)]
        operations = {x: 'WATER' for x in ids}
        counts = method.sample_operation_counts(ids, operations)
        self.assertEqual(sum(counts.values()), 20)
        self.assertEqual(counts['WATER'], 20)
        self.assertEqual(counts['FERTILIZE'], 0)

    def test_marker_failure_keeps_primary_and_receipt_error(self):
        cfg = dict(test_seeds=[7], candidate_checkpoint_sha256='model', independent_limits=dict(seconds=3, compressed_bytes=1, decoded_bytes=1))
        ledger = [dict(game=f'{s}-{a}', seed=7, opponent='market', side=s, policy=a, task_checkpoint_sha256='model') for s in (0,1) for a in ('candidate','teacher_task')]
        audits = [dict(game=x['game'], certificate_sha256='hash', replay_sha256='r', trace_sha256='t', official_transitions=719) for x in ledger]
        values = {'games-result.json':dict(passed=True, all_owned_processes_reaped=True, game_ledger_sha256='hash', audit_ledger_sha256='hash'), 'game-ledger.json':ledger, 'game-audit-ledger.json':audits}
        game = dict(replay=dict(sha256='r'), trace=dict(sha256='t'))
        def marker(out, state):
            state['created'] = True
            raise OSError('synthetic partial marker')
        with patch.object(method,'ready',return_value=cfg), patch.object(method,'verify_commit'), \
             patch.object(method,'read',side_effect=lambda p:values[Path(p).name]), patch.object(method,'sha',return_value='hash'), \
             patch.object(collect,'current_game',return_value=(game,{})), patch.object(Path,'exists',return_value=False), \
             patch.object(method,'create_marker',side_effect=marker), patch.object(method,'write_new',side_effect=PermissionError('synthetic')):
            with self.assertRaises(method.ReceiptFailure) as caught: method.run()
        self.assertEqual((caught.exception.original_error_kind,caught.exception.receipt_error_kind),('OSError','PermissionError'))

if __name__ == '__main__': unittest.main()
