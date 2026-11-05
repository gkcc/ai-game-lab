"""Small stable matching and real budget-wrapper contracts, no model files."""
from contextlib import nullcontext
from copy import deepcopy
from fractions import Fraction
import io
import itertools
import json
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from learning.task_dagger_once_v1 import policy as bounded
from . import decoder, policy


def candidate(unit, x, y=0, op='WATER'):
    return dict(id=f'{unit}:{x}:{y}:{op}', unit=unit, target=[x, y], op=op)


def brute(candidates, scores):
    best = None
    for size in range(1, len(candidates)+1):
        for indices in itertools.combinations(range(len(candidates)), size):
            edges = [candidates[i] for i in indices]
            if len({c['unit'] for c in edges}) != size or len({tuple(c['target']) for c in edges}) != size:
                continue
            keys = tuple(sorted(decoder.candidate_key(c) for c in edges))
            total = sum((Fraction.from_float(float(scores[i])) for i in indices), Fraction())
            if best is None or size > best[0] or (size == best[0] and
                    (total > best[1] or (total == best[1] and keys < best[2]))):
                best = (size, total, keys, indices)
    chosen = min(best[3], key=lambda i: (-scores[i], decoder.candidate_key(candidates[i])))
    return candidates[chosen]['id'], tuple(key[-1] for key in best[2]), best[0]


class MatchingTests(unittest.TestCase):
    def test_assignment_avoids_greedy_shared_target_conflict(self):
        graph = [candidate(0, 0), candidate(1, 0), candidate(1, 1)]
        result = decoder.choose(graph, [8., 10., 9.])
        self.assertEqual(result.choice_id, graph[2]['id'])
        self.assertEqual(result.matching_ids, (graph[0]['id'], graph[2]['id']))
        self.assertEqual((result.cardinality, result.total_logit), (2, 17.))

    def test_cardinality_precedes_negative_sum_and_common_translation(self):
        graph = [candidate(0, 0), candidate(1, 0), candidate(1, 1)]
        expected = decoder.choose(graph, [8., 10., 9.])
        negative = decoder.choose(graph, [-12., -10., -11.])
        self.assertEqual((negative.choice_id, negative.matching_ids),
                         (expected.choice_id, expected.matching_ids))
        self.assertEqual(negative.cardinality, 2)
        self.assertEqual(negative.total_logit, -23.)

    def test_single_unit_uses_encoder_order_not_raw_id_or_external_order(self):
        # Raw string IDs place x=1 before x=2, but canonical y=1 comes first.
        graph = [candidate(0, 1, 2), candidate(0, 2, 1)]
        result = decoder.choose(graph, [4., 4.])
        canonical = sorted(graph, key=decoder.candidate_key)
        self.assertEqual(result.choice_id, canonical[0]['id'])
        self.assertEqual(result.choice_id, decoder.choose(list(reversed(graph)), [4., 4.]).choice_id)
        self.assertEqual(result.choice_id, decoder.choose(canonical, [4., 4.]).choice_id)

    def test_parallel_operations_keep_best_then_semantic_key(self):
        graph = [candidate(0, 0, op='WATER'), candidate(0, 0, op='HARVEST')]
        self.assertEqual(decoder.choose(graph, [1., 2.]).choice_id, graph[1]['id'])
        self.assertEqual(decoder.choose(graph, [2., 2.]).choice_id, graph[1]['id'])
        self.assertEqual(decoder.choose(graph, [3., 2.]).cardinality, 1)

    def test_whole_matching_equal_sum_tie_uses_sorted_semantic_keys(self):
        graph = [candidate(0, 2, 1), candidate(0, 1, 2),
                 candidate(1, 2, 1), candidate(1, 1, 2)]
        result = decoder.choose(graph, [2., 2., 2., 2.])
        self.assertEqual(result.matching_ids, (graph[0]['id'], graph[3]['id']))
        for order in itertools.permutations(range(4)):
            permuted = decoder.choose([graph[i] for i in order], [2.]*4)
            self.assertEqual(permuted, result)

    def test_third_unit_augmentation_can_reassign_an_earlier_match(self):
        graph = [candidate(0, 0), candidate(0, 1), candidate(1, 0),
                 candidate(1, 2), candidate(2, 2)]
        scores = [10., 9., 9., 1., 9.]
        result = decoder.choose(graph, scores)
        self.assertEqual(result.matching_ids, (graph[1]['id'], graph[2]['id'], graph[4]['id']))
        self.assertEqual((result.choice_id, result.matching_ids, result.cardinality), brute(graph, scores))

    def test_exact_sum_preserves_small_increment_beside_large_logit(self):
        graph = [candidate(0, 0), candidate(0, 1), candidate(1, 0), candidate(1, 1)]
        # Both ordinary float totals round to 2**54; cross matching is +1 larger.
        result = decoder.choose(graph, [float(2**54), float(2**54), 1., 0.])
        self.assertEqual(result.matching_ids, (graph[1]['id'], graph[2]['id']))

    def test_tiny_and_extreme_finite_scores_do_not_become_infinite_preferences(self):
        graph = [candidate(0, 0), candidate(1, 1)]
        result = decoder.choose(graph, [1.7e308, 1.7e308])
        self.assertEqual(result.cardinality, 2)
        self.assertIsNone(result.total_logit)
        self.assertEqual(decoder.choose(graph, [5e-324, -5e-324]).total_logit, 0.)

    def test_small_graphs_match_exhaustive_reference(self):
        all_edges = [candidate(unit, x) for unit in range(2) for x in range(3)]
        patterns = ([0.]*6, [3., -2., 1., 4., -1., 2.], [-3., -2., -1., -4., -5., -6.])
        for mask in range(1, 1 << len(all_edges)):
            indices = [i for i in range(len(all_edges)) if mask & (1 << i)]
            graph = [all_edges[i] for i in indices]
            for scores in patterns:
                values = [scores[i] for i in indices]
                actual = decoder.choose(graph, values)
                self.assertEqual((actual.choice_id, actual.matching_ids, actual.cardinality), brute(graph, values))

    def test_invalid_graph_and_scores_fail_closed(self):
        c = candidate(0, 0)
        malformed = [([], []), ([c], []), ([c, c], [1., 1.]),
                     ([c], [True]), ([c], [float('nan')]), ([c], [float('inf')]),
                     ([dict(c, unit=True)], [1.]), ([dict(c, target=[False, 0])], [1.]),
                     ([dict(c, op='')], [1.]), ([c], ['1'])]
        for graph, scores in malformed:
            with self.subTest(graph=graph, scores=scores), self.assertRaises(decoder.DecoderError):
                decoder.choose(graph, scores)


class FakeNetwork:
    calls = 0
    mode = 'normal'

    def __init__(self, _path, sha):
        self.sha, self.last = sha, None

    def score(self, *_):
        raise AssertionError('The decoder must not request a second score pass')

    def __call__(self, inputs):
        type(self).calls += 1
        scores = list(inputs['fixture_scores'])
        chosen = inputs['candidates'][max(range(len(scores)), key=lambda i: scores[i])]['id']
        if self.mode == 'stale':
            return chosen
        self.last = dict(checkpoint_sha256=self.sha, chosen=chosen, logits=scores, fallbacks=0)
        if self.mode == 'bad_choice': self.last['chosen'] = 'not-current'
        if self.mode == 'short': self.last['logits'] = scores[:-1]
        if self.mode == 'nonfinite': self.last['logits'][0] = float('nan')
        if self.mode == 'wrong_argmax':
            chosen = inputs['candidates'][0]['id'];self.last['chosen'] = chosen
        return chosen


class FakeController:
    def __init__(self, selector, *_):
        self.task_selector = selector
        self.crop = SimpleNamespace(network=SimpleNamespace(logits=lambda *_: [1.]))

    def __call__(self, observation, configuration):
        self.records = []
        graph = [candidate(0, 0), candidate(1, 0), candidate(1, 1)]
        if not observation.get('no_tasks'):
            inputs = dict(candidates=graph, fixture_scores=[8., 10., 9.])
            if observation.get('inject_label'): inputs['teacher_label'] = graph[0]['id']
            chosen = self.task_selector(deepcopy(inputs))
            self.records.append(dict(inputs=inputs, chosen=chosen, teacher_label=graph[0]['id']))
            if observation.get('two_callbacks'):
                # Simulate current controller resource/reservation updates. The
                # previous matching's u0A disappears; the only live task is C.
                inputs = dict(candidates=[candidate(0, 2)], fixture_scores=[-5.])
                chosen = self.task_selector(deepcopy(inputs))
                self.records.append(dict(inputs=inputs, chosen=chosen, teacher_label=chosen))
        self.crop.network.logits([], [])
        self.last = dict(step=observation['step'], records=self.records, load_errors=0,
                         inference_errors=0, fallbacks=0)
        return dict(farmer=['PASS'], hands=[], market=[])


class PolicyTests(unittest.TestCase):
    def setUp(self):
        FakeNetwork.calls, FakeNetwork.mode = 0, 'normal'
        observe = lambda o,c: SimpleNamespace(observation=o, configuration=c)
        for module, name, value in ((bounded, 'TaskNetwork', FakeNetwork),
                                    (bounded, 'TaskAgent', FakeController),
                                    (bounded, 'observe', observe), (policy, 'observe', observe)):
            patcher = patch.object(module, name, value)
            patcher.start();self.addCleanup(patcher.stop)

    def agent(self, **kw):
        return policy.JointAgent('crop', 'crop-identity', 'task', 'task-identity', **kw)

    def test_one_counted_forward_per_callback_and_recompute_live_graph(self):
        agent = self.agent()
        agent(dict(step=0, player=0, two_callbacks=True), {})
        records = agent.last['decoder_records']
        self.assertEqual(FakeNetwork.calls, 2)
        self.assertEqual(agent.last['model_counts']['task_forwards'], 2)
        self.assertEqual(agent.last['model_counts']['task_rows'], 4)
        self.assertEqual(agent.last['model_counts']['crop_forwards'], 1)
        self.assertEqual(agent.last['model_counts']['controller_calls'], 1)
        self.assertEqual(records[0]['original_network_choice'], candidate(1, 0)['id'])
        self.assertEqual(records[0]['decoder_choice'], candidate(1, 1)['id'])
        self.assertEqual(records[1]['decoder_choice'], candidate(0, 2)['id'])
        self.assertEqual(records[1]['matching_ids'], [candidate(0, 2)['id']])
        self.assertEqual([r['chosen'] for r in agent.last['records']], [r['decoder_choice'] for r in records])

    def test_turn_buffers_and_episode_budget_reset(self):
        agent = self.agent()
        agent(dict(step=0, player=0, two_callbacks=True), {})
        agent(dict(step=1, player=0), {})
        self.assertEqual(len(agent.last['decoder_records']), 1)
        self.assertEqual(agent.last['decoder_records'][0]['ordinal'], 0)
        self.assertEqual(agent.last['model_counts']['task_forwards'], 3)
        agent(dict(step=0, player=1, no_tasks=True), {})
        self.assertEqual(agent.last['decoder_records'], [])
        self.assertEqual(agent.last['model_counts']['task_forwards'], 0)
        self.assertIsNone(agent.network.last)
        with self.assertRaises(ValueError): agent(dict(step=1, player=0), {})
        self.assertIsNone(agent.last)

    def test_existing_budget_and_training_label_reject_before_extra_forward(self):
        agent = self.agent(limits=dict(bounded.LIMITS, task_forwards=1))
        with self.assertRaises(RuntimeError): agent(dict(step=0, player=0, two_callbacks=True), {})
        self.assertEqual(FakeNetwork.calls, 1)
        self.assertIsNone(agent.last)
        other = self.agent()
        with self.assertRaises(ValueError): other(dict(step=0, player=0, inject_label=True), {})
        self.assertEqual(FakeNetwork.calls, 1)

    def test_stale_missing_short_nonfinite_and_mismatched_current_forward_rejected(self):
        for mode in ('stale', 'bad_choice', 'short', 'nonfinite', 'wrong_argmax'):
            with self.subTest(mode=mode):
                FakeNetwork.mode = 'normal';agent = self.agent()
                agent(dict(step=0, player=0), {})
                FakeNetwork.mode = mode
                with self.assertRaises(decoder.DecoderError): agent(dict(step=1, player=0), {})
                self.assertIsNone(agent.last)

    def test_only_outer_trace_written_once_and_existing_trace_cannot_reset_episode(self):
        agent = self.agent()
        destination = Mock();destination.exists.return_value = False
        agent.trace = destination
        output = io.StringIO()
        with patch.object(policy.gzip, 'open', return_value=nullcontext(output)):
            agent(dict(step=0, player=0, two_callbacks=True), {})
        lines = output.getvalue().splitlines()
        self.assertEqual(len(lines), 1)
        self.assertEqual(json.loads(lines[0]), agent.last)
        self.assertIsNone(agent.base.trace)
        before = FakeNetwork.calls;destination.exists.return_value = True
        with self.assertRaises(FileExistsError): agent(dict(step=0, player=1), {})
        self.assertEqual(FakeNetwork.calls, before)
        self.assertIsNone(agent.last)
        self.assertEqual(agent.decoder_records, [])

    def test_trace_write_failure_is_not_a_successful_turn(self):
        agent = self.agent();agent.trace = Mock();agent.trace.exists.return_value = False
        with patch.object(policy.gzip, 'open', side_effect=OSError('synthetic write failure')):
            with self.assertRaises(OSError): agent(dict(step=0, player=0), {})
        self.assertIsNone(agent.last)


if __name__ == '__main__':
    unittest.main()
