"""Synthetic model plumbing only: no checkpoint reads, game or training.

The task scorer uses the frozen TaskNetwork arithmetic with in-memory synthetic
parameters. The crop scorer is a controlled stub. These tests are not evidence
about the real checkpoint's predictions or the registered real-forward budget.
"""
from copy import deepcopy
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

from learning.task_bc_v1.model import DIMENSIONS, TaskNetwork
from learning.task_choice_v1.features import OPS, SCHEMA_VERSION
from learning.task_dagger_once_v1.policy import BoundedAgent
from learning.task_harvest_choice_v1.fixtures import observation
from learning.task_harvest_model_dev_v1.policy import BoundedHarvestAgent
from learning.task_pair_dev_games_v1.panel import GAME_LIMITS


CROP_SHA, TASK_SHA = 'c' * 64, 'd' * 64


class MemoryTaskNetwork(TaskNetwork):
    def __init__(self, path, expected_sha):
        path = Path(path).stem
        if path not in ('task-harvest', 'task-plant', 'task-tie') or expected_sha != TASK_SHA:
            raise ValueError('Synthetic task identity mismatch')
        self.path, self.sha, self.last = path, expected_sha, None
        self.metadata, self.score_calls = None, 0

    def __call__(self, inputs):
        if any(k in inputs for k in ('teacher_label', 'chosen', 'teacher_priority')):
            raise AssertionError('Label was passed to synthetic model')
        if self.metadata is None:
            self.metadata = dict(schema_version=SCHEMA_VERSION, ops=list(OPS),
                dimensions=list(DIMENSIONS), schema_sha256=inputs['schema_sha256'],
                feature_names=inputs['feature_names'], shared_feature_names=inputs['shared_feature_names'])
            self.mean, self.scale = np.zeros(373), np.ones(373)
            self.params = {}
            for layer in range(3):
                self.params['w' + str(layer)] = np.zeros((DIMENSIONS[layer], DIMENSIONS[layer + 1]))
                self.params['b' + str(layer)] = np.zeros(DIMENSIONS[layer + 1])
            if self.path != 'task-tie':
                op = 'HARVEST' if self.path == 'task-harvest' else 'PLANT'
                index = 275 + inputs['feature_names'].index('task.op.' + op)
                self.params['w0'][index, 0] = 1.
                self.params['w1'][0, 0] = self.params['w2'][0, 0] = 1.
        return super().__call__(inputs)

    def score(self, shared, candidates):
        self.score_calls += 1
        return super().score(shared, candidates)


class MemoryCropNetwork:
    def __init__(self, path, expected_sha):
        path = Path(path).stem
        if path != 'crop-wheat' or expected_sha != CROP_SHA:
            raise ValueError('Synthetic crop identity mismatch')
        self.path, self.sha, self.calls = path, expected_sha, 0

    def logits(self, _x, _names):
        self.calls += 1
        return np.asarray([1., 0., 0., 0., 0.])


def inputs(*, side=0, market=False):
    obs, cfg = observation('mature5dry', side=side)
    # A constructed controller input, not a claim of a natural step-zero farm.
    obs['step'] = 0
    if market:
        own = obs['farms'][side]
        own['money'], own['hands'] = 3000, [[0, 0], [1, 0]]
        obs['private']['inventories'] = [{}, {}, {}]
        for x, y in ((0, 0), (1, 0), (2, 0), (3, 0), (4, 0),
                     (0, 1), (1, 1), (2, 1), (3, 1), (4, 1), (0, 2)):
            own['tiles'][y][x] = None
    return obs, cfg


def crop_records(trace):
    return [{key: value for key, value in record.items() if key != 'inference_seconds'}
            for record in trace['crop_records']]


class PolicyTests(unittest.TestCase):
    def setUp(self):
        for name, replacement in (
            ('learning.task_harvest_model_dev_v1.policy.TaskNetwork', MemoryTaskNetwork),
            ('learning.task_dagger_once_v1.policy.TaskNetwork', MemoryTaskNetwork),
            ('learning.bc_crop.agent.Network', MemoryCropNetwork),
        ):
            monkey = patch(name, replacement)
            monkey.start()
            self.addCleanup(monkey.stop)
        forbidden = patch('numpy.load', side_effect=AssertionError('No checkpoint read in synthetic tests'))
        forbidden.start()
        self.addCleanup(forbidden.stop)

    def agent(self, task='task-harvest', **kwargs):
        return BoundedHarvestAgent('crop-wheat', CROP_SHA, task, TASK_SHA, **kwargs)

    def test_disabled_full_request_crop_model_and_market_match_old_actor(self):
        obs, cfg = inputs(market=True)
        base = BoundedAgent('crop-wheat', CROP_SHA, 'task-plant', TASK_SHA)
        agent = self.agent('task-plant', enabled=False)
        expected = base(obs, cfg)
        self.assertEqual(agent(obs, cfg), expected)
        self.assertIn('BUY_SEED', [order[0] for order in expected['market']])
        self.assertIn('HIRE', [order[0] for order in expected['market']])
        self.assertEqual(agent.last['final_reservations'], base.last['final_reservations'])
        self.assertEqual(crop_records(agent.last), crop_records(base.last))
        self.assertGreater(len(agent.last['crop_records']), 0)
        self.assertTrue(all(record['source'] == 'model' and record['chosen'] == 'WHEAT'
                            for record in agent.last['crop_records']))
        self.assertTrue(any(record['teacher'] != record['chosen'] for record in agent.last['crop_records']))
        for new, old in zip(agent.last['records'], base.last['records']):
            self.assertEqual({k: v for k, v in new.items() if k != 'task_model'}, old)
        self.assertEqual(agent.last['model_counts'], base.last['model_counts'])
        self.assertEqual(agent.controller.crop.mode, 'model')
        self.assertEqual(agent.last['checkpoint_sha256'], CROP_SHA)
        self.assertEqual(agent.last['task_checkpoint_sha256'], TASK_SHA)

    def test_expanded_choice_records_existing_forward_without_rescoring(self):
        obs, cfg = inputs()
        base = BoundedAgent('crop-wheat', CROP_SHA, 'task-harvest', TASK_SHA)
        agent = self.agent()
        self.assertEqual(base(obs, cfg)['farmer'], ['WATER'])
        self.assertEqual(agent(obs, cfg)['farmer'], ['HARVEST'])
        record = agent.last['records'][0]
        self.assertEqual(record['teacher_label'], 'u0:x4:y4:WATER')
        self.assertEqual(record['chosen'], 'u0:x4:y4:HARVEST')
        self.assertEqual(record['unit_request'], ['HARVEST'])
        self.assertTrue(record['harvest_choice_extension']['selected_virtual_harvest'])
        evidence = record['task_model']
        self.assertEqual(evidence, dict(checkpoint_sha256=TASK_SHA, logits=[0., 1.],
                                       argmax_index=1, chosen=record['chosen'], fallbacks=0))
        self.assertEqual(evidence['logits'][:1], base.network.last['logits'])
        self.assertEqual(agent.network.score_calls, 1)
        self.assertEqual(agent.budget.counts['task_forwards'], 1)
        self.assertEqual(agent.budget.counts['task_rows'], 2)
        self.assertEqual(agent.budget.counts['crop_forwards'], 0)
        self.assertEqual(agent.controller.crop.network.calls, 0)
        agent.network.last['logits'][0] = 99.
        self.assertEqual(evidence['logits'], [0., 1.])

    def test_exact_tie_keeps_original_prefix_and_never_forces_harvest(self):
        obs, cfg = inputs()
        agent = self.agent('task-tie')
        self.assertEqual(agent(obs, cfg)['farmer'], ['WATER'])
        record = agent.last['records'][0]
        self.assertEqual(record['harvest_choice_extension']['added_candidate_ids'], ['u0:x4:y4:HARVEST'])
        self.assertFalse(record['harvest_choice_extension']['selected_virtual_harvest'])
        self.assertEqual(record['task_model']['logits'], [0., 0.])
        self.assertEqual(record['task_model']['argmax_index'], 0)
        self.assertEqual(agent.network.score_calls, 1)

    def test_all_callbacks_and_both_heads_charge_existing_shared_budget(self):
        obs, cfg = inputs(market=True)
        agent = self.agent('task-plant')
        agent(obs, cfg)
        trace, counts = agent.last, agent.last['model_counts']
        self.assertGreater(len(trace['records']), 1)
        self.assertGreater(len(trace['crop_records']), 0)
        self.assertEqual(counts['task_forwards'], len(trace['records']))
        self.assertEqual(counts['task_forwards'], agent.network.score_calls)
        self.assertEqual(counts['task_rows'], sum(len(r['inputs']['candidates']) for r in trace['records']))
        self.assertEqual(counts['crop_forwards'], len(trace['crop_records']))
        self.assertEqual(counts['crop_forwards'], agent.controller.crop.network.calls)
        self.assertEqual(counts['crop_rows'], counts['crop_forwards'])
        self.assertEqual(counts['controller_calls'], 1)
        self.assertEqual(counts['gate_evaluations'], 0)
        self.assertEqual(counts['teacher_proposals'], 0)
        self.assertEqual(trace['turn_model_counts'], counts)

    def test_hard_budget_rejects_before_extra_model_call(self):
        obs, cfg = inputs()
        limits = dict(GAME_LIMITS, max_neural_input_rows_per_turn=1)
        agent = self.agent(limits=limits)
        with self.assertRaisesRegex(RuntimeError, 'budget exhausted'):
            agent(obs, cfg)
        self.assertEqual(agent.network.score_calls, 0)
        self.assertEqual(agent.budget.counts['task_forwards'], 0)
        self.assertEqual(agent.budget.counts['controller_calls'], 1)
        self.assertIsNone(agent.last)
        obs, cfg = inputs(market=True)
        agent = self.agent('task-plant', limits=dict(GAME_LIMITS, max_task_forwards_per_turn=1))
        with self.assertRaisesRegex(RuntimeError, 'budget exhausted'):
            agent(obs, cfg)
        self.assertEqual(agent.network.score_calls, 1)
        self.assertEqual(agent.budget.counts['task_forwards'], 1)
        self.assertIsNone(agent.last)

    def test_turn_order_seat_and_new_episode_use_old_guards(self):
        obs, cfg = inputs()
        agent = self.agent()
        later = deepcopy(obs)
        later['step'] = 1
        with self.assertRaisesRegex(ValueError, 'begin at zero'):
            agent(later, cfg)
        self.assertEqual(agent.network.score_calls, 0)
        first = agent(obs, cfg)
        agent(later, cfg)
        old_count = agent.network.score_calls
        with self.assertRaisesRegex(ValueError, 'out-of-order'):
            agent(later, cfg)
        other, _ = inputs(side=1)
        other['step'] = 2
        with self.assertRaisesRegex(ValueError, 'Cross-episode'):
            agent(other, cfg)
        self.assertEqual(agent.network.score_calls, old_count)
        other['step'] = 0
        self.assertEqual(agent(other, cfg), first)
        self.assertEqual(agent.last['model_counts']['controller_calls'], 1)
        self.assertEqual(agent.last['model_counts']['task_forwards'], 1)
        self.assertEqual(agent.previous, (1, 0))
        self.assertEqual(len(agent.last['records']), 1)

    def test_missing_models_and_model_errors_never_fall_back(self):
        for path, identity in ((None, TASK_SHA), ('task-harvest', None), (None, None)):
            with self.assertRaisesRegex(ValueError, 'no teacher fallback'):
                BoundedHarvestAgent('crop-wheat', CROP_SHA, path, identity)
        with self.assertRaisesRegex(ValueError, 'Synthetic crop identity'):
            BoundedHarvestAgent('crop-wheat', None, 'task-harvest', TASK_SHA)
        with self.assertRaisesRegex(ValueError, 'Synthetic task identity'):
            BoundedHarvestAgent('crop-wheat', CROP_SHA, 'task-harvest', 'wrong')
        with self.assertRaises(TypeError):
            self.agent(enabled=1)
        obs, cfg = inputs()
        agent = self.agent()
        failure = ValueError('synthetic scorer failed')
        with patch.object(agent.network, 'score', side_effect=failure):
            with self.assertRaises(ValueError) as caught:
                agent(obs, cfg)
        self.assertIs(caught.exception, failure)
        self.assertEqual(agent.budget.counts['task_forwards'], 1)
        self.assertIsNone(agent.last)
        self.assertEqual(agent.controller.records, [])

    def test_unknown_model_choice_rejected_without_hidden_teacher_action(self):
        class InvalidNetwork(MemoryTaskNetwork):
            def __call__(self, _inputs):
                return 'u0:x99:y99:HARVEST'
        obs, cfg = inputs()
        with patch('learning.task_harvest_model_dev_v1.policy.TaskNetwork', InvalidNetwork):
            agent = self.agent()
        with self.assertRaisesRegex(ValueError, 'no fallback'):
            agent(obs, cfg)
        self.assertEqual(agent.budget.counts['task_forwards'], 1)
        self.assertIsNone(agent.last)
        self.assertEqual(agent.controller.records, [])


if __name__ == '__main__':
    unittest.main()
