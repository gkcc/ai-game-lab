"""Pure paired-economic and task-trace accounting checks."""
import unittest
from pathlib import Path
from unittest.mock import mock_open, patch
from . import campaign, evaluation
from .evaluation import count_turn, paired_metrics


CFG = dict(test_seeds=[1,2],opponents=['round2'],acceptance=dict(
    pair_own_income_delta_min=-2000,pair_cash_margin_delta_min=-2000,
    seed_mean_own_income_delta_min=-1000,seed_mean_cash_margin_delta_min=-1000,
    relative_denominator_abs_min=1000))


def panel(own_deltas=None, rival_deltas=None, teacher_profit=40000):
    own_deltas = own_deltas or {(s,side):0 for s in (1,2) for side in (0,1)}
    rival_deltas = rival_deltas or {(s,side):0 for s in (1,2) for side in (0,1)}
    rows = []
    for seed in (1,2):
        for side in (0,1):
            for policy in ('candidate','teacher_task'):
                own = teacher_profit + (own_deltas[(seed,side)] if policy=='candidate' else 0)
                rival = 38000 + (rival_deltas[(seed,side)] if policy=='candidate' else 0)
                rows.append(dict(game=f'{seed}-{side}-{policy}',seed=seed,opponent='round2',side=side,
                    policy=policy,net_profit=own,opponent_net_profit=rival,margin=own-rival,score=1.0))
    return rows


class EconomicTest(unittest.TestCase):
    def test_inclusive_seed_and_seat_thresholds(self):
        d = {(1,0):-2000,(1,1):0,(2,0):-1000,(2,1):-1000}
        result = paired_metrics(panel(d),CFG)
        self.assertTrue(result['exploratory_gate_passed'])
        self.assertEqual([x['mean_own_profit_difference'] for x in result['seed_groups']],[-1000,-1000])

    def test_single_seat_decline_is_not_hidden_by_seed_average(self):
        d = {(1,0):-2001,(1,1):1,(2,0):0,(2,1):0}
        result = paired_metrics(panel(d),CFG)
        self.assertFalse(result['exploratory_gate_passed'])
        self.assertTrue(any(f.get('side')==0 and f['metric']=='own_profit' for f in result['gate_failures']))

    def test_opponent_gain_can_break_margin_gate_even_if_own_is_unchanged(self):
        d = {(1,0):2001,(1,1):0,(2,0):0,(2,1):0}
        result = paired_metrics(panel(rival_deltas=d),CFG)
        self.assertFalse(result['exploratory_gate_passed'])
        self.assertTrue(any(f.get('side')==0 and f['metric']=='margin' for f in result['gate_failures']))

    def test_relative_own_change_is_suppressed_for_small_denominator(self):
        result = paired_metrics(panel(teacher_profit=999),CFG)
        self.assertTrue(all(p['relative_own_profit_difference'] is None for p in result['pairs']))
        boundary = paired_metrics(panel(teacher_profit=1000),CFG)
        self.assertTrue(all(p['relative_own_profit_difference'] == 0 for p in boundary['pairs']))

    def test_candidate_identity_and_task_counts(self):
        trace = dict(step=4,task_checkpoint_sha256='task',checkpoint_sha256='crop',
            load_errors=0,inference_errors=0,fallbacks=0,records=[dict(source='callback',op='FERTILIZE',
            unit_request=['FERTILIZE'],inputs=dict(candidates=[dict(op='WATER'),dict(op='FERTILIZE')]))])
        self.assertEqual(count_turn(trace,4,'candidate','task','crop'),dict(callbacks=1,
            fertilizer_opportunities=1,fertilizer_selections=1,fertilizer_work_requests=1))
        trace['fallbacks']=1
        with self.assertRaises(ValueError): count_turn(trace,4,'candidate','task','crop')

    def test_marker_partial_write_records_creation(self):
        out = Path('synthetic').resolve()
        state = dict(created=False,complete=False)
        with patch.object(evaluation,'OUT',out), patch.object(Path,'open',mock_open()), \
             patch.object(evaluation.json,'dump',side_effect=OSError('synthetic')):
            with self.assertRaises(OSError): evaluation.write_marker(out,'start.json',{},state)
        self.assertEqual(state,dict(created=True,complete=False))

    def test_trace_initialization_failure_has_receipt_without_raw_reads(self):
        cfg = {**CFG,'analysis_limits':dict(compressed_bytes=1,decoded_bytes=1,seconds=1)}
        done = dict(passed=True,game_ledger_sha256='hash',audit_ledger_sha256='hash')
        def read(path): return done if Path(path).name=='games-result.json' else panel()
        def marker(out,name,payload,state): state.update(created=True,complete=True)
        receipts = []
        with patch.object(evaluation,'ready',return_value=cfg), patch.object(evaluation,'verify_commit'), \
             patch.object(campaign,'verify_independent_review'), patch.object(evaluation,'read',side_effect=read), \
             patch.object(evaluation,'sha',return_value='hash'), patch.object(Path,'exists',return_value=False), \
             patch.object(evaluation,'write_marker',side_effect=marker), \
             patch.object(evaluation,'Meter',side_effect=ValueError('synthetic')), \
             patch.object(evaluation,'write_new',side_effect=lambda p,v:receipts.append(v)):
            with self.assertRaises(ValueError): evaluation.trace_summary()
        self.assertTrue(receipts[0]['marker']['created'])
        self.assertIsNone(receipts[0]['raw_reads'])
        self.assertFalse(receipts[0]['retry_allowed'])


if __name__=='__main__': unittest.main()
