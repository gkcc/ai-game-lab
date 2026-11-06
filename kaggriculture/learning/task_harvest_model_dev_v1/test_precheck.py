"""Synthetic points and memory-only weights; not registered real-model checks."""
from copy import deepcopy
import unittest
from unittest.mock import patch

from learning.task_bc_error_diagnostics_v1.shadow import ordered_hash
from learning.task_dagger_once_v1.policy import BoundedAgent
from learning.task_harvest_choice_v1.fixtures import observation
from learning.task_harvest_model_dev_v1 import precheck
from learning.task_harvest_model_dev_v1.test_policy import (
    CROP_SHA, TASK_SHA, MemoryCropNetwork, MemoryTaskNetwork)
from learning.task_harvest_unexpressed_v1.selection import TARGET_SCHEMA


class PrecheckTests(unittest.TestCase):
    def setUp(self):
        for name, replacement in (
            ('learning.task_harvest_model_dev_v1.policy.TaskNetwork', MemoryTaskNetwork),
            ('learning.task_dagger_once_v1.policy.TaskNetwork', MemoryTaskNetwork),
            ('learning.bc_crop.agent.Network', MemoryCropNetwork),
        ):
            monkey = patch(name, replacement)
            monkey.start()
            self.addCleanup(monkey.stop)
        forbidden = patch('numpy.load', side_effect=AssertionError('No real weights in precheck regressions'))
        forbidden.start()
        self.addCleanup(forbidden.stop)

    def points(self):
        result = []
        for case in precheck.sources():
            obs, cfg = observation('mature5dry', side=case['side'])
            actor = BoundedAgent('crop-wheat', CROP_SHA, 'task-harvest', TASK_SHA)
            actor.budget.begin_turn()
            actor.budget.controller()
            request = actor.controller(obs, cfg)
            trace = deepcopy(actor.controller.last)
            trace.update(model_counts={key: value * 3 for key, value in actor.budget.counts.items()},
                         turn_model_counts=dict(actor.budget.counts), task_checkpoint_sha256=TASK_SHA)
            encoded = trace['records'][0]['inputs']
            actions = trace['final_reservations']['unit_actions']
            target = dict(schema_version=TARGET_SCHEMA, side=case['side'], step=obs['step'], ordinal=0,
                          unit=0, target=[4, 4], input_sha256=ordered_hash(encoded),
                          original_unit_request=deepcopy(actions[0]), unit_requests_sha256=ordered_hash(actions))
            result.append(dict(role=case['role'], target=target, inputs=deepcopy(encoded),
                original_request=deepcopy(request), original_trace=trace,
                task=['artifacts/synthetic/task-harvest.npz', TASK_SHA],
                crop=['artifacts/synthetic/crop-wheat.npz', CROP_SHA]))
        return result

    def test_fixed_eight_plus_eight_calls_and_private_evidence(self):
        points, events, meter = self.points(), [], {}
        untouched = deepcopy(points)
        report = precheck.run(points, game_root='.', guard=lambda: None, emit=events.append, meter=meter)
        self.assertTrue(report['passed'])
        self.assertEqual(meter['scope'], 'precheck')
        self.assertEqual(meter['status'], 'passed')
        self.assertEqual((meter['point_attempts'], meter['point_returns']), (4, 4))
        self.assertEqual((meter['actor_construction_attempts'], meter['actor_construction_returns']), (8, 8))
        self.assertEqual((meter['controller_attempts'], meter['controller_returns']), (8, 8))
        self.assertEqual((meter['score_attempts'], meter['score_returns']), (8, 8))
        self.assertEqual(meter['model_costs'], dict(task_forwards=16, task_rows=20,
            crop_forwards=0, crop_rows=0, gate_evaluations=0, controller_calls=8, teacher_proposals=0))
        self.assertEqual(len([e for e in events if e['event'] == 'precheck_controller']), 8)
        self.assertEqual(len([e for e in events if e['event'] == 'precheck_score']), 8)
        self.assertEqual(len([e for e in events if e['event'] == 'precheck_point']), 4)
        self.assertTrue(all(row['added_candidates'] == 1 and row['expanded_selected_new']
                            and row['max_original_logit_error'] == 0. for row in report['points']))
        self.assertTrue(all(not row['historical_cumulative_counts_reconstructed'] for row in report['points']))
        self.assertEqual(points, untouched)
        with self.assertRaisesRegex(ValueError, 'no retry'):
            precheck.run(points, game_root='.', guard=lambda: None, emit=events.append, meter=meter)
        self.assertEqual(meter['controller_attempts'], 8)

    def test_ordinal_digest_and_counter_types_fail_before_construction(self):
        for mutation in ('ordinal', 'digest', 'counter'):
            with self.subTest(mutation=mutation):
                points, meter, events = self.points(), {}, []
                if mutation == 'ordinal':
                    points[0]['target']['ordinal'] = 1
                elif mutation == 'digest':
                    points[0]['target']['input_sha256'] = '0' * 64
                else:
                    points[0]['original_trace']['turn_model_counts']['controller_calls'] = True
                with self.assertRaises(ValueError):
                    precheck.run(points, game_root='.', guard=lambda: None, emit=events.append, meter=meter)
                self.assertEqual(meter['actor_construction_attempts'], 0)
                self.assertEqual(meter['controller_attempts'], 0)
                self.assertEqual(meter['point_returns'], 0)
                self.assertEqual(events[-1]['event'], 'precheck_failure')

    def test_original_market_and_every_core_trace_field_are_checked(self):
        for mutation in ('market', 'label'):
            with self.subTest(mutation=mutation):
                points, events, meter = self.points(), [], {}
                if mutation == 'market':
                    points[0]['original_request']['market'] = [['HIRE']]
                else:
                    points[0]['original_trace']['records'][0]['teacher_label'] = 'wrong'
                with self.assertRaisesRegex(ValueError, 'Historical'):
                    precheck.run(points, game_root='.', guard=lambda: None, emit=events.append, meter=meter)
                self.assertEqual(meter['controller_returns'], 1)
                self.assertEqual(meter['score_attempts'], 0)
                self.assertEqual(events[0]['event'], 'precheck_controller')
                self.assertEqual(events[-1]['event'], 'precheck_failure')

    def test_prefix_score_drift_fails_after_both_score_records_are_emitted(self):
        class BatchDependentNetwork(MemoryTaskNetwork):
            def score(self, shared, candidates):
                return super().score(shared, candidates) + len(candidates)
        points, events, meter = self.points(), [], {}
        with patch('learning.task_dagger_once_v1.policy.TaskNetwork', BatchDependentNetwork), \
             patch('learning.task_harvest_model_dev_v1.policy.TaskNetwork', BatchDependentNetwork):
            with self.assertRaisesRegex(ValueError, 'scores or argmax'):
                precheck.run(points, game_root='.', guard=lambda: None, emit=events.append, meter=meter)
        self.assertEqual(meter['controller_returns'], 2)
        self.assertEqual(meter['score_returns'], 2)
        self.assertEqual(meter['point_returns'], 0)
        self.assertEqual(len([event for event in events if event['event'] == 'precheck_score']), 2)
        self.assertEqual(meter['model_costs']['task_forwards'], 4)

    def test_another_units_virtual_harvest_does_not_satisfy_fixed_target(self):
        points, events, meter, observed_virtuals = self.points(), [], {}, []
        point = points[0]
        obs, cfg = deepcopy(point['inputs']['observation']), deepcopy(point['inputs']['configuration'])
        obs['farms'][0]['hands'] = [[3, 3]]
        obs['private']['inventories'].append({})
        actor = BoundedAgent('crop-wheat', CROP_SHA, 'task-harvest', TASK_SHA)
        actor.budget.begin_turn()
        actor.budget.controller()
        point['original_request'] = actor.controller(obs, cfg)
        trace = deepcopy(actor.controller.last)
        trace.update(model_counts={key: value * 3 for key, value in actor.budget.counts.items()},
                     turn_model_counts=dict(actor.budget.counts), task_checkpoint_sha256=TASK_SHA)
        point['original_trace'], point['inputs'] = trace, deepcopy(trace['records'][0]['inputs'])
        actions = trace['final_reservations']['unit_actions']
        point['target'].update(unit=1, target=[3, 3], original_unit_request=deepcopy(actions[1]),
                               input_sha256=ordered_hash(point['inputs']), unit_requests_sha256=ordered_hash(actions))
        original_expand = precheck._expanded
        def observe_expand(context):
            result = original_expand(context)
            observed_virtuals.append(list(result[2]))
            return result
        with patch.object(precheck, '_expanded', observe_expand):
            with self.assertRaisesRegex(ValueError, 'Frozen target has no corresponding'):
                precheck.run(points, game_root='.', guard=lambda: None, emit=events.append, meter=meter)
        self.assertEqual(observed_virtuals, [['u0:x4:y4:HARVEST']])
        self.assertEqual(meter['controller_returns'], 2)
        self.assertEqual(meter['score_attempts'], 0)
        self.assertEqual(meter['point_returns'], 0)

    def test_capture_seam_restores_on_controller_error_and_budget_is_retained(self):
        points, events, meter, created = self.points(), [], {}, []
        real = precheck.BoundedHarvestAgent
        failure = ValueError('synthetic disabled scorer failure')
        def build(*args, **kwargs):
            actor = real(*args, **kwargs)
            original = actor.controller.controller.__globals__['_task_select']
            actor.network.score = lambda *_args: (_ for _ in ()).throw(failure)
            created.append((actor, original))
            return actor
        with patch.object(precheck, 'BoundedHarvestAgent', build):
            with self.assertRaises(ValueError) as caught:
                precheck.run(points, game_root='.', guard=lambda: None, emit=events.append, meter=meter)
        self.assertIs(caught.exception, failure)
        actor, hook = created[0]
        self.assertIs(actor.controller.controller.__globals__['_task_select'], hook)
        self.assertIsNone(actor.previous)
        self.assertEqual((meter['controller_attempts'], meter['controller_returns']), (2, 1))
        self.assertEqual(meter['model_costs']['controller_calls'], 2)
        self.assertEqual(meter['model_costs']['task_forwards'], 2)
        self.assertEqual(meter['score_attempts'], 0)

    def test_failure_emission_does_not_replace_primary_exception(self):
        points, meter = self.points(), {}
        primary = ValueError('synthetic model construction failure')
        def broken_emit(_event):
            raise OSError('synthetic sink unavailable')
        with patch.object(precheck, 'BoundedAgent', side_effect=primary):
            with self.assertRaises(ValueError) as caught:
                precheck.run(points, game_root='.', guard=lambda: None, emit=broken_emit, meter=meter)
        self.assertIs(caught.exception, primary)
        self.assertEqual(meter['emission_failures'], 1)
        self.assertEqual((meter['actor_construction_attempts'], meter['actor_construction_returns']), (1, 0))
        self.assertEqual(meter['status'], 'failed')

    def test_each_failed_write_is_counted_once_and_primary_is_preserved(self):
        for tail_fails in (False, True):
            with self.subTest(tail_fails=tail_fails):
                points, meter, attempted, saved = self.points(), {}, [], []
                primary = OSError('synthetic first evidence write failed')
                def emit(event):
                    attempted.append(event)
                    if len(attempted) == 1:
                        raise primary
                    if tail_fails:
                        raise OSError('synthetic failure evidence write failed')
                    saved.append(event)
                with self.assertRaises(OSError) as caught:
                    precheck.run(points, game_root='.', guard=lambda: None, emit=emit, meter=meter)
                self.assertIs(caught.exception, primary)
                self.assertEqual([event['event'] for event in attempted],
                                 ['precheck_controller', 'precheck_failure'])
                self.assertEqual(meter['emission_failures'], 2 if tail_fails else 1)
                self.assertEqual(meter['controller_returns'], 1)
                self.assertEqual(meter['score_attempts'], 0)
                if not tail_fails:
                    self.assertEqual(saved[0]['meter']['emission_failures'], 1)


if __name__ == '__main__':
    unittest.main()
