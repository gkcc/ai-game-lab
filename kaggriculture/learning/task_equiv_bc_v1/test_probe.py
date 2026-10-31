"""Synthetic controller tests; no real trace, model, game, or GPU."""
from copy import deepcopy
import unittest
from unittest.mock import patch

import numpy as np

from decision.api import observe
from learning.task_choice_v1.features import (
    CANDIDATE_SCOPE, OPS, SCHEMA_VERSION, candidate_id, schema_hash,
)
from .probe import evaluate_choice, probe_group, task_agent_factory
from . import probe


FEATURE_NAMES = (['unit.index', 'unit.x', 'unit.y', 'target.x', 'target.y', 'task.distance']
                 + ['task.op.' + op for op in OPS])
FEATURE_NAMES += ['local.' + str(i) for i in range(98 - len(FEATURE_NAMES))]
SHARED_NAMES = ['shared.' + str(i) for i in range(275)]
SCHEMA = dict(schema_version=SCHEMA_VERSION,
              schema_sha256=schema_hash(FEATURE_NAMES, SHARED_NAMES),
              feature_names=FEATURE_NAMES, shared_feature_names=SHARED_NAMES, ops=list(OPS))


def candidate(unit, target, op):
    x = np.zeros(98, dtype=np.float32)
    fields = {'unit.index': unit, 'unit.x': unit, 'unit.y': 0,
              'target.x': target[0], 'target.y': target[1],
              'task.distance': abs(unit-target[0]) + target[1]}
    fields.update({'task.op.' + name: int(name == op) for name in OPS})
    for name, value in fields.items():
        x[FEATURE_NAMES.index(name)] = value
    return dict(id=candidate_id(unit, target, op), unit=unit, target=target, op=op, x=x.tolist())


def example():
    obs = dict(player=0, step=3, day=0, hour=3,
               farms=[dict(farmer=[0, 0], hands=[[1, 0], [2, 0], [3, 0]],
                           tiles=[[None] * 4 for _ in range(4)]),
                      dict(farmer=[0, 0], hands=[], tiles=[[None] * 4 for _ in range(4)])],
               private=dict(shed={}, seeds={}, inventories=[{}, {}, {}, {}]), market={}, town={})
    cfg = {'marketParams': {}}
    obs, cfg = observe(obs, cfg).observation, observe(obs, cfg).configuration
    candidates = [candidate(0, [0, 1], 'WATER'), candidate(0, [0, 2], 'WATER'),
                  candidate(0, [0, 3], 'DIG'), candidate(1, [0, 1], 'WATER'),
                  candidate(2, [0, 1], 'WATER')]
    planned = dict(seeds={}, needed_seeds={}, supply={}, projected_shed={}, unit_actions=[],
                   available_units=[0, 1, 2],
                   interpretation='before assignment: reservations and requests, not official executed state')
    inputs = dict(schema_version=SCHEMA_VERSION, schema_sha256=SCHEMA['schema_sha256'],
                  feature_names=FEATURE_NAMES, shared_feature_names=SHARED_NAMES,
                  shared_x=[0.] * 275, candidate_scope=CANDIDATE_SCOPE, observation=obs,
                  configuration=cfg, planned=planned, candidates=candidates,
                  inventory_item_order=[[], [], [], []], shed_item_order=[])
    action = dict(farmer=['PASS'], hands=[['PASS'], ['PASS'], ['PASS']], market=[])
    return obs, cfg, inputs, action


def crop_record(seconds=0.1):
    # Field order matches the real frozen CropAgent's complete record.
    return dict(step=3, decision=0, unit=3, target=[0, 0], x=[], schema_sha256='schema',
                teacher='WHEAT', chosen='WHEAT', outcome='move_with_seed', gate_reason=None,
                inference_seconds=seconds, logits=None, prior={'inference_seconds': 7.},
                selected_seed_before=1, source='teacher', unit_request=['PASS'], buy_seed_orders=[])


class FakeAgent:
    def __init__(self, ordinal, force, options, calls):
        self.ordinal, self.force, self.options, self.calls = ordinal, force, options, calls
        self.probe_target_hits = self.probe_forced_hits = 0
        self.probe_prior_crop_count = None
        self.last = None

    def __call__(self, obs, cfg):
        self.calls.append(self.force)
        _, _, inputs, action = example()
        ids = [c['id'] for c in inputs['candidates']]
        if 'raise_for' in self.options and self.force == self.options['raise_for']:
            raise RuntimeError('private exception text')
        if self.force == ids[1]:
            action['farmer'] = ['EAST']
        if 'late_prefix_request_for' in self.options and self.force == self.options['late_prefix_request_for']:
            action['hands'][2] = ['WEST']
        seeds = {'WHEAT': 1, 'CARROT': 2}
        if self.force == ids[2]:
            seeds = {'CARROT': 2, 'WHEAT': 1}  # same values, different iteration order
        reservations = dict(seeds=seeds, needed_seeds={}, supply={}, projected_shed={},
                            unit_actions=[action['farmer'], *action['hands']])
        pre = deepcopy(inputs)
        pre_target = [3, 2] if self.options.get('different_historical_prefix') else [3, 3]
        pre_candidate = candidate(3, pre_target, 'WATER')
        pre['candidates'] = [pre_candidate]
        pre['planned']['available_units'] = [3]
        if 'prefix_drift_for' in self.options and self.force == self.options['prefix_drift_for']:
            pre['planned']['supply'] = {'WHEAT': 1}
        prefix = dict(inputs=pre, teacher_label=pre_candidate['id'], chosen=pre_candidate['id'],
                      unit=3, target=pre_target, op='WATER', source='teacher',
                      unit_request=deepcopy(action['hands'][2]))
        chosen = self.force or ids[0]
        spec = next(c for c in inputs['candidates'] if c['id'] == chosen)
        target = dict(inputs=inputs, teacher_label=ids[0], chosen=chosen, unit=spec['unit'],
                      target=spec['target'], op=spec['op'], source='callback' if self.force else 'teacher',
                      unit_request=[action['farmer'], *action['hands']][spec['unit']])
        crop = crop_record(seconds=0.2 if self.force else 0.1)
        if 'crop_drift_for' in self.options and self.force == self.options['crop_drift_for']:
            crop['prior']['inference_seconds'] = 8.  # not the timing field being normalized
        crop_records = [crop]
        if 'crop_after_for' in self.options and self.force == self.options['crop_after_for']:
            extra = crop_record()
            extra['chosen'] = 'CARROT'
            crop_records.append(extra)
        if self.force is None and self.options.get('mismatch_second_reference') and self.calls.count(None) == 2:
            reservations['seeds']['WHEAT'] = 99
        self.last = dict(step=obs['step'], records=[prefix, target], crop_records=crop_records,
                         final_reservations=reservations, checkpoint_sha256=None,
                         load_errors=0, inference_errors=0, fallbacks=0)
        self.probe_target_hits = 1
        self.probe_forced_hits = int(self.force is not None)
        self.probe_prior_crop_count = 1
        return action


def setup_probe(options=None):
    obs, cfg, inputs, action = example()
    calls, options = [], options or {}
    def factory(ordinal, forced):
        return FakeAgent(ordinal, forced, options, calls)
    args = dict(expected_inputs=inputs, expected_label=inputs['candidates'][0]['id'],
                expected_teacher_prefix=(candidate_id(3, [3, 3], 'WATER'),),
                expected_request=action, expected_schema=SCHEMA, agent_factory=factory)
    return obs, cfg, inputs, calls, args


class ProbeTest(unittest.TestCase):
    def test_same_different_and_order_sensitive_unknown_are_separate(self):
        obs, cfg, inputs, calls, args = setup_probe()
        result = probe_group(obs, cfg, 1, **args)
        ids = [c['id'] for c in inputs['candidates']]
        self.assertEqual(result['status'], 'ok')
        self.assertEqual(calls, [None, None, ids[3], ids[1], ids[2]])
        self.assertEqual(result['positive_ids'], [ids[0], ids[3]])
        self.assertEqual(result['different_request_ids'], [ids[1]])
        self.assertEqual(result['unknown_ids'], [ids[2]])
        self.assertEqual(result['features'].shape, (5, 373))
        self.assertEqual(result['features'].dtype, np.float32)
        np.testing.assert_array_equal(result['positive_mask'], [True, False, False, True, False])
        np.testing.assert_array_equal(result['verified_mask'], [True, True, False, True, False])
        third = result['alternatives'][2]
        self.assertEqual((third['request'], third['reservations'], third['outcome']),
                         ('same', 'different', 'unknown'))
        self.assertNotIn('observation', result)
        self.assertNotIn('seed', result)

    def test_target_prefix_and_prior_crop_must_match(self):
        _, _, inputs, _ = example()
        alt = inputs['candidates'][3]['id']
        for option in ('prefix_drift_for', 'crop_drift_for'):
            with self.subTest(option=option):
                obs, cfg, _, _, args = setup_probe({option: alt})
                result = probe_group(obs, cfg, 1, **args)
                self.assertIn(alt, result['unknown_ids'])
                self.assertNotIn(alt, result['positive_ids'])

    def test_same_request_different_later_crop_is_unknown(self):
        _, _, inputs, _ = example()
        alt = inputs['candidates'][3]['id']
        obs, cfg, _, _, args = setup_probe({'crop_after_for': alt})
        result = probe_group(obs, cfg, 1, **args)
        alternative = next(row for row in result['alternatives'] if row['candidate_id'] == alt)
        self.assertEqual((alternative['request'], alternative['reservations'], alternative['crop_records'],
                          alternative['outcome']), ('same', 'same', 'different', 'unknown'))
        self.assertIn(alt, result['unknown_ids'])

    def test_finished_prefix_request_does_not_claim_pre_target_drift(self):
        _, _, inputs, _ = example()
        alt = inputs['candidates'][1]['id']
        obs, cfg, _, _, args = setup_probe({'late_prefix_request_for': alt})
        result = probe_group(obs, cfg, 1, **args)
        self.assertIn(alt, result['different_request_ids'])
        self.assertNotIn(alt, result['unknown_ids'])

    def test_exceptions_are_unknown_and_guard_exception_stops(self):
        _, _, inputs, _ = example()
        alt = inputs['candidates'][1]['id']
        obs, cfg, _, _, args = setup_probe({'raise_for': alt})
        result = probe_group(obs, cfg, 1, **args)
        self.assertIn(alt, result['unknown_ids'])
        self.assertNotIn('private exception text', str(result))
        calls = []
        def guard():
            calls.append(1)
            if len(calls) == 4:
                raise TimeoutError('resource stopped')
        obs, cfg, _, _, args = setup_probe()
        args['guard'] = guard
        with self.assertRaises(Exception) as raised:
            probe_group(obs, cfg, 1, **args)
        self.assertIsInstance(raised.exception.__cause__, TimeoutError)

    def test_resource_and_teacher_setup_errors_stop_all_three_paths(self):
        for error in (MemoryError, TimeoutError, FileNotFoundError, PermissionError, probe.TeacherSetupError):
            for phase in ('reference', 'alternative', 'evaluation'):
                with self.subTest(error=error.__name__, phase=phase):
                    obs, cfg, inputs, _, args = setup_probe()
                    ids = [candidate['id'] for candidate in inputs['candidates']]
                    original = args['agent_factory']
                    selected = None if phase == 'reference' else ids[2]

                    def factory(ordinal, forced):
                        if forced == selected:
                            if error is probe.TeacherSetupError:
                                with patch.object(probe, '_build_task_agent', side_effect=ValueError('source identity')):
                                    return task_agent_factory(ordinal, forced)
                            def fail(_observation, _configuration):
                                raise error('synthetic resource error')
                            return fail
                        return original(ordinal, forced)

                    if phase == 'evaluation':
                        reference = probe_group(obs, cfg, 1, retain_reference=True, **args)['reference']
                        with self.assertRaises(error):
                            evaluate_choice(obs, cfg, 1, expected_inputs=inputs,
                                expected_label=args['expected_label'],
                                expected_teacher_prefix=args['expected_teacher_prefix'],
                                reference=reference, chosen_id=selected, agent_factory=factory)
                    else:
                        args['agent_factory'] = factory
                        with self.assertRaises(error):
                            probe_group(obs, cfg, 1, **args)

    def test_reference_mismatch_has_no_training_group(self):
        obs, cfg, _, _, args = setup_probe({'mismatch_second_reference': True})
        result = probe_group(obs, cfg, 1, **args)
        self.assertEqual(result, dict(status='unknown', reason='reference_mismatch'))

    def test_equal_target_and_action_do_not_hide_different_recorded_teacher_prefix(self):
        obs, cfg, _, calls, args = setup_probe({'different_historical_prefix': True})
        result = probe_group(obs, cfg, 1, **args)
        self.assertEqual(result, dict(status='unknown', reason='historical_prefix_mismatch'))
        self.assertEqual(calls, [None, None])
        # The expected prefix is mandatory and its length is the target ordinal.
        obs, cfg, _, _, args = setup_probe()
        args['expected_teacher_prefix'] = ()
        with self.assertRaises(ValueError):
            probe_group(obs, cfg, 1, **args)

    def test_replay_metadata_and_opponent_private_input_are_rejected(self):
        obs, cfg, _, calls, args = setup_probe()
        dirty = deepcopy(obs)
        dirty['farms'][1]['private'] = {'secret': 42}
        with self.assertRaises(ValueError):
            probe_group(dirty, cfg, 1, **args)
        dirty = deepcopy(obs)
        dirty['info'] = {'seed': 123}
        with self.assertRaises(ValueError):
            probe_group(dirty, cfg, 1, **args)
        self.assertEqual(calls, [])

    def test_three_categories_do_not_backfill_or_repeat(self):
        obs, cfg, inputs, _, args = setup_probe()
        # Factory's actual candidate set must also be changed, so check the pure
        # policy on a synthetic ordered group without exposing any private data.
        from .probe import _pick_alternatives
        candidates = inputs['candidates']
        selected = _pick_alternatives(candidates, candidates[0])
        self.assertEqual(selected, [candidates[3]['id'], candidates[1]['id'], candidates[2]['id']])
        missing_category = [candidates[0], candidates[3], candidates[4], candidates[2]]
        selected = _pick_alternatives(missing_category, missing_category[0])
        self.assertEqual(selected, [candidates[3]['id'], candidates[2]['id']])
        self.assertEqual(len(selected), len(set(selected)))

    def test_default_factory_runs_pinned_teacher_on_synthetic_observation(self):
        # No official environment, model or saved trajectory is opened.
        crops = ('WHEAT', 'CARROT', 'TOMATO', 'STRAWBERRY', 'MELON')
        farm = dict(money=1000, farmer=[1, 1], hands=[[2, 1]],
                    unlocked_quadrants=[0], hires_today=0,
                    tiles=[[None] * 4 for _ in range(4)])
        obs = dict(player=0, step=24, day=1, hour=0,
                   farms=[deepcopy(farm), deepcopy(farm)],
                   private=dict(shed={}, seeds={name: 2 for name in crops}, inventories=[{}, {}]),
                   market=dict(inventory={name: 10000 for name in crops},
                               prices={name: 50 for name in crops}, params={}),
                   town=dict(unlocked_shops=[]))
        cfg = dict(episodeSteps=720, boardSize=4, turnsPerDay=24, shedCapacity=100,
                   maxMarketOrdersPerTurn=8, townShopSellInterval=24, farmHandCostMult=1,
                   marketParams={})
        obs, cfg = observe(obs, cfg).observation, observe(obs, cfg).configuration
        teacher = task_agent_factory(0, None)
        action = teacher(deepcopy(obs), deepcopy(cfg))
        target = teacher.last['records'][0]
        inp = target['inputs']
        schema = dict(schema_version=inp['schema_version'], schema_sha256=inp['schema_sha256'],
                      feature_names=inp['feature_names'], shared_feature_names=inp['shared_feature_names'],
                      ops=list(OPS))
        result = probe_group(obs, cfg, 0, expected_inputs=inp, expected_label=target['teacher_label'],
                             expected_teacher_prefix=(), retain_reference=True,
                             expected_request=action, expected_schema=schema)
        self.assertEqual(result['status'], 'ok')
        self.assertEqual(result['features'].shape[1], 373)
        self.assertTrue(result['positive_mask'][result['teacher_index']])
        self.assertGreater(len(result['alternatives']), 0)
        evaluate = dict(expected_inputs=inp, expected_label=target['teacher_label'],
                        expected_teacher_prefix=(), reference=result['reference'])
        self.assertEqual(evaluate_choice(obs, cfg, 0, chosen_id=target['teacher_label'], **evaluate),
                         dict(outcome='match', controller_called=False))
        alternative = result['alternatives'][0]['candidate_id']
        evaluated = evaluate_choice(obs, cfg, 0, chosen_id=alternative, **evaluate)
        self.assertTrue(evaluated['controller_called'])
        self.assertIn(evaluated['outcome'], ('match', 'different'))

    def _reference(self):
        obs, cfg, inputs, calls, args = setup_probe()
        normal = probe_group(obs, cfg, 1, **args)
        self.assertNotIn('reference', normal)
        saved = probe_group(obs, cfg, 1, retain_reference=True, **args)
        self.assertEqual(repr(saved['reference']), '<PrivateTeacherReference>')
        evaluation = dict(expected_inputs=inputs, expected_label=args['expected_label'],
                          expected_teacher_prefix=args['expected_teacher_prefix'],
                          reference=saved['reference'], agent_factory=args['agent_factory'])
        calls.clear()
        return obs, cfg, inputs, calls, evaluation

    def test_evaluate_teacher_direct_and_alternatives_use_at_most_one_call(self):
        obs, cfg, inputs, calls, args = self._reference()
        ids = [candidate['id'] for candidate in inputs['candidates']]
        self.assertEqual(evaluate_choice(obs, cfg, 1, chosen_id=ids[0], **args),
                         dict(outcome='match', controller_called=False))
        self.assertEqual(calls, [])
        for candidate_id, outcome in ((ids[3], 'match'), (ids[1], 'different'), (ids[2], 'different')):
            calls.clear()
            self.assertEqual(evaluate_choice(obs, cfg, 1, chosen_id=candidate_id, **args),
                             dict(outcome=outcome, controller_called=True))
            self.assertEqual(calls, [candidate_id])

    def test_evaluate_inconsistent_reference_or_candidate_is_unknown_without_call(self):
        obs, cfg, inputs, calls, args = self._reference()
        chosen = inputs['candidates'][0]['id']
        wrong_prefix = dict(args, expected_teacher_prefix=('wrong-prefix',))
        self.assertEqual(evaluate_choice(obs, cfg, 1, chosen_id=chosen, **wrong_prefix),
                         dict(outcome='unknown', controller_called=False))
        altered = deepcopy(inputs)
        altered['shared_x'][0] = 1.
        self.assertEqual(evaluate_choice(obs, cfg, 1, chosen_id=chosen,
                         **dict(args, expected_inputs=altered)),
                         dict(outcome='unknown', controller_called=False))
        self.assertEqual(evaluate_choice(obs, cfg, 1, chosen_id='not-an-offered-id', **args),
                         dict(outcome='unknown', controller_called=False))
        self.assertEqual(calls, [])
        # The opaque reference protects against accidental post-capture changes.
        args['reference'].action['farmer'] = ['WEST']
        self.assertEqual(evaluate_choice(obs, cfg, 1, chosen_id=chosen, **args),
                         dict(outcome='unknown', controller_called=False))

    def test_evaluate_attempt_failure_and_prefix_drift_are_unknown(self):
        obs, cfg, inputs, _, args = self._reference()
        chosen = inputs['candidates'][3]['id']
        for option in ('raise_for', 'prefix_drift_for', 'crop_drift_for'):
            with self.subTest(option=option):
                _, _, _, calls, bad_args = setup_probe({option: chosen})
                current = dict(args, agent_factory=bad_args['agent_factory'])
                self.assertEqual(evaluate_choice(obs, cfg, 1, chosen_id=chosen, **current),
                                 dict(outcome='unknown', controller_called=True))
                self.assertEqual(calls, [chosen])
        def fail_factory(_ordinal, _chosen):
            raise RuntimeError('private factory error')
        self.assertEqual(evaluate_choice(obs, cfg, 1, chosen_id=chosen,
                         **dict(args, agent_factory=fail_factory)),
                         dict(outcome='unknown', controller_called=False))

    def test_call_marker_does_not_count_failed_input_copy(self):
        from .probe import _execute
        events = []
        class Uncopyable:
            def __deepcopy__(self, _memo):
                raise ValueError('copy failed')
        def controller(*_args):
            events.append('controller')
        with self.assertRaises(ValueError):
            _execute(lambda _ordinal, _forced: controller, {'value': Uncopyable()}, {}, 0,
                     'candidate', lambda: None, on_called=lambda: events.append('called'))
        self.assertEqual(events, [])

    def test_evaluate_only_uses_crop_prefix_for_development_metric(self):
        obs, cfg, inputs, _, args = self._reference()
        chosen = inputs['candidates'][3]['id']
        _, _, _, calls, changed = setup_probe({'crop_after_for': chosen})
        result = evaluate_choice(obs, cfg, 1, chosen_id=chosen,
                                 **dict(args, agent_factory=changed['agent_factory']))
        self.assertEqual(result, dict(outcome='match', controller_called=True))
        self.assertEqual(calls, [chosen])


if __name__ == '__main__':
    unittest.main()
