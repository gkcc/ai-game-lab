"""Pure full-controller regressions; no official engine or network execution."""
from copy import deepcopy
import unittest
from unittest.mock import patch

from learning.task_choice_v1.agent import TaskAgent
from learning.task_choice_v1.features import encode_candidates
from learning.task_harvest_choice_v1.adapter import HarvestChoiceAgent, SCOPE
from learning.task_harvest_choice_v1.fixtures import NAMES, observation


def choose_harvest(inputs, unit=0):
    matches = [c['id'] for c in inputs['candidates'] if c['unit'] == unit and c['op'] == 'HARVEST']
    return matches[0] if matches else inputs['candidates'][0]['id']


def market_fixture():
    obs, cfg = observation('mature5dry')
    own = obs['farms'][0]
    own['money'], own['hands'] = 3000, [[0, 0], [1, 0]]
    obs['private']['inventories'] = [{}, {}, {}]
    for x, y in ((0, 0), (1, 0), (2, 0), (3, 0), (4, 0),
                 (0, 1), (1, 1), (2, 1), (3, 1), (4, 1), (0, 2)):
        own['tiles'][y][x] = None
    return obs, cfg


class AuditedAgent(HarvestChoiceAgent):
    """Observe the public encoding before the actual controller callback."""
    def __init__(self, *args, **kwargs):
        self.before_after = []
        super().__init__(*args, **kwargs)

    def select(self, context):
        original, _ = encode_candidates(context)
        before = deepcopy(context['tasks'])
        result = super().select(context)
        self.before_after.append((original, before, deepcopy(context['tasks'])))
        return result


class AdapterTests(unittest.TestCase):
    def assert_preserved(self, agent):
        self.assertEqual(len(agent.before_after), len(agent.last['records']))
        for (original, before, after), record in zip(agent.before_after, agent.last['records']):
            actual = record['inputs']
            for key in original:
                if key not in ('candidate_scope', 'candidates'):
                    self.assertEqual(actual[key], original[key], key)
            self.assertEqual(actual['candidates'][:len(original['candidates'])], original['candidates'])
            self.assertEqual(len(before), len(after))
            changed = [i for i, (old, new) in enumerate(zip(before, after)) if old != new]
            extension = record.get('harvest_choice_extension', {})
            if extension.get('selected_virtual_harvest'):
                self.assertEqual(len(changed), 1)
                old, new = before[changed[0]], after[changed[0]]
                self.assertEqual((old[0], old[2]), (new[0], new[2]))
                self.assertEqual(old[1], ['WATER'])
                self.assertEqual(new[1], ['HARVEST'])
            else:
                self.assertEqual(changed, [])
            candidates = actual['candidates']
            self.assertEqual(len({c['id'] for c in candidates}), len(candidates))
            for c in candidates:
                self.assertEqual(set(c), {'id', 'unit', 'target', 'op', 'x'})

    def test_four_scenes_both_sides_preserve_original_and_disabled(self):
        for name in NAMES:
            for side in (0, 1):
                with self.subTest(scene=name, side=side):
                    obs, cfg = observation(name, side=side)
                    original = deepcopy((obs, cfg))
                    base, enabled, disabled = TaskAgent(), AuditedAgent(), HarvestChoiceAgent(enabled=False)
                    expected = base(obs, cfg)
                    self.assertEqual(enabled(obs, cfg), expected)
                    self.assertEqual(disabled(obs, cfg), expected)
                    self.assertEqual(disabled.last['records'], base.last['records'])
                    self.assertEqual(enabled.last['final_reservations'], base.last['final_reservations'])
                    self.assertEqual([r['teacher_label'] for r in enabled.last['records']],
                                     [r['teacher_label'] for r in base.last['records']])
                    additions = enabled.last['records'][0]['harvest_choice_extension']['added_candidate_ids']
                    self.assertEqual(additions, ['u0:x4:y4:HARVEST']
                                     if name in ('mature5dry', 'mature5dry_full_low') else [])
                    self.assertEqual((obs, cfg), original)
                    self.assert_preserved(enabled)

    def test_original_ids_preserve_nonempty_buy_and_hire_order(self):
        obs, cfg = market_fixture()
        base = TaskAgent()
        expected = base(obs, cfg)
        operations = [order[0] for order in expected['market']]
        self.assertIn('BUY_SEED', operations)
        self.assertIn('HIRE', operations)
        self.assertGreater(len(base.last['records']), 1)
        for enabled in (True, False):
            choices = iter(r['chosen'] for r in base.last['records'])
            agent = AuditedAgent(lambda _inputs: next(choices), enabled=enabled)
            self.assertEqual(agent(obs, cfg), expected)
            self.assertEqual(agent.last['final_reservations'], base.last['final_reservations'])
            self.assertEqual([r['teacher_label'] for r in agent.last['records']],
                             [r['teacher_label'] for r in base.last['records']])
            self.assert_preserved(agent)

    def test_virtual_harvest_is_real_controller_choice_and_input_is_detached(self):
        obs, cfg = observation('mature5dry_full_low')
        original = deepcopy((obs, cfg))
        def select(inputs):
            self.assertEqual(inputs['candidate_scope'], SCOPE)
            self.assertNotIn('teacher_label', inputs)
            identifier = choose_harvest(inputs)
            # A caller may mutate its copy but cannot change recorded inputs,
            # the encoded observation or the actual teacher task action.
            inputs['observation']['private']['shed']['FERTILIZER'] = -99
            inputs['planned']['unit_actions'][0] = ['DIG']
            inputs['candidates'][0]['op'] = 'DIG'
            return identifier
        agent = AuditedAgent(select)
        request = agent(obs, cfg)
        self.assertEqual(request['farmer'], ['HARVEST'])
        record = agent.last['records'][0]
        self.assertEqual((record['teacher_label'], record['chosen']),
                         ('u0:x4:y4:WATER', 'u0:x4:y4:HARVEST'))
        self.assertEqual(record['unit_request'], ['HARVEST'])
        self.assertEqual(record['inputs']['observation']['private']['shed']['FERTILIZER'], 100)
        self.assertEqual(record['inputs']['candidates'][0]['op'], 'WATER')
        candidate = next(c for c in record['inputs']['candidates'] if c['id'] == record['chosen'])
        features = dict(zip(record['inputs']['feature_names'], candidate['x']))
        self.assertEqual(features['task.op.HARVEST'], 1.)
        self.assertEqual(features['task.op.WATER'], 0.)
        self.assertEqual(features['task.distance'], 0.)
        self.assertEqual((obs, cfg), original)
        self.assert_preserved(agent)

    def test_multiple_units_claim_site_once_and_rebuild_after_each_assignment(self):
        obs, cfg = observation('mature5dry')
        own = obs['farms'][0]
        own['hands'] = [[4, 4], [3, 4]]
        obs['private']['inventories'] = [{}, {}, {}]
        own['tiles'][3][4], own['tiles'][4][3] = None, None
        agent = AuditedAgent(lambda inputs: choose_harvest(inputs, 1))
        request = agent(obs, cfg)
        self.assertEqual(request['hands'][0], ['HARVEST'])
        records = agent.last['records']
        self.assertEqual(len(records), 3)
        self.assertEqual(records[0]['harvest_choice_extension']['added_candidate_ids'],
                         ['u0:x4:y4:HARVEST', 'u1:x4:y4:HARVEST'])
        self.assertEqual(sum(r['target'] == [4, 4] for r in records), 1)
        for record in records[1:]:
            self.assertNotIn(1, record['inputs']['planned']['available_units'])
            self.assertFalse(any(c['unit'] == 1 or c['target'] == [4, 4]
                                 for c in record['inputs']['candidates']))
        self.assertEqual(len({r['unit'] for r in records}), len(records))
        self.assert_preserved(agent)

    def test_seed_reservations_are_current_on_later_callbacks(self):
        obs, cfg = observation('mature5dry')
        own = obs['farms'][0]
        own['hands'] = [[4, 3], [3, 4]]
        own['tiles'][3][4], own['tiles'][4][3] = None, None
        obs['private']['inventories'] = [{}, {}, {}]
        obs['private']['seeds']['WHEAT'] = 1
        # Public prices make WHEAT the fixed crop teacher's profitable choice.
        for crop in ('CARROT', 'TOMATO', 'STRAWBERRY', 'MELON'):
            obs['market']['params'][crop] = dict(base=1, below_target=0, above_target=0)
        agent = AuditedAgent(choose_harvest)
        request = agent(obs, cfg)
        self.assertEqual(request['farmer'], ['HARVEST'])
        self.assertEqual(request['hands'], [['PLANT', 'WHEAT'], ['PASS']])
        records = agent.last['records']
        self.assertEqual([r['inputs']['planned']['seeds']['WHEAT'] for r in records], [1, 1, 0])
        self.assertEqual(agent.last['final_reservations']['needed_seeds'], {'WHEAT': 1})
        self.assertEqual(records[2]['inputs']['planned']['unit_actions'][1], ['PLANT', 'WHEAT'])
        self.assertGreater(records[2]['inputs']['planned']['supply']['WHEAT'],
                           records[1]['inputs']['planned']['supply']['WHEAT'])
        self.assert_preserved(agent)

    def test_preallocated_fertilizer_pickup_is_not_available_to_extension(self):
        obs, cfg = observation('mature5dry')
        own = obs['farms'][0]
        own['money'], own['hands'] = 1000, [[4, 3], [2, 2]]
        obs['private']['inventories'] = [{}, {'FERTILIZER': 1}, {}]
        obs['private']['shed']['FERTILIZER'] = 1
        own['tiles'][3][4] = dict(kind='PLANT', crop='STRAWBERRY', planted_day=1,
            watered_today=False, consecutive_unwatered=1, yield_units=0,
            max_lifespan_step=456, fertilized_until_day=-1)
        own['tiles'][2][2] = deepcopy(own['tiles'][4][4])
        def select(inputs):
            harvest = [c['id'] for c in inputs['candidates'] if c['unit'] == 2 and c['op'] == 'HARVEST']
            return harvest[0] if harvest else next(c['id'] for c in inputs['candidates'] if c['op'] == 'FERTILIZE')
        agent = AuditedAgent(select)
        request = agent(obs, cfg)
        self.assertEqual(request['farmer'], ['PICKUP', 'FERTILIZER', 1])
        self.assertEqual(request['hands'], [['FERTILIZE'], ['HARVEST']])
        for record in agent.last['records']:
            planned = record['inputs']['planned']
            self.assertNotIn(0, planned['available_units'])
            self.assertEqual(planned['projected_shed']['FERTILIZER'], 0)
            self.assertEqual(planned['unit_actions'][0], ['PICKUP', 'FERTILIZER', 1])
            self.assertFalse(any(c['unit'] == 0 for c in record['inputs']['candidates']))
        self.assert_preserved(agent)

    def test_private_input_filter_and_cross_game_buffers(self):
        first, cfg = observation('mature5dry')
        agent = HarvestChoiceAgent(choose_harvest)
        agent(first, cfg)
        old_trace = deepcopy(agent.last)
        second, cfg = observation('immature', side=1)
        second['step'], second['day'], second['hour'] = 0, 0, 0
        for farm in second['farms']:
            for row in farm['tiles']:
                for tile in row:
                    if isinstance(tile, dict):
                        tile['planted_day'] = 0
        second['farms'][0]['private'] = {'opponent_secret': 123}
        second['farms'][0]['hidden_future'] = 456
        second['private']['unknown_secret'] = 789
        second['farms'][1]['tiles'][4][4]['future_price'] = 1000
        cfg['seed'] = 999
        clean_second, original_cfg = deepcopy(second), deepcopy(cfg)
        result = agent(second, cfg)
        fresh = HarvestChoiceAgent(choose_harvest)
        self.assertEqual(result, fresh(second, cfg))
        self.assertEqual(agent.last['records'], fresh.last['records'])
        self.assertEqual(agent.last['final_reservations'], fresh.last['final_reservations'])
        self.assertEqual(agent.last['step'], 0)
        self.assertEqual(old_trace['step'], 241)
        self.assertEqual(old_trace['records'][0]['chosen'], 'u0:x4:y4:HARVEST')
        for record in agent.last['records']:
            inputs = record['inputs']
            self.assertNotIn('seed', inputs['configuration'])
            self.assertNotIn('private', inputs['observation']['farms'][0])
            self.assertNotIn('hidden_future', inputs['observation']['farms'][0])
            self.assertNotIn('unknown_secret', inputs['observation']['private'])
            self.assertNotIn('future_price', inputs['observation']['farms'][1]['tiles'][4][4])
        self.assertEqual((second, cfg), (clean_second, original_cfg))

    def test_invalid_selection_fails_without_teacher_fallback(self):
        obs, cfg = observation('mature5dry')
        for invalid in ('u0:x99:y99:HARVEST', None, 0):
            with self.subTest(selection=invalid):
                agent = HarvestChoiceAgent(lambda _inputs: invalid)
                with self.assertRaisesRegex(ValueError, 'no fallback'):
                    agent(obs, cfg)
                self.assertIsNone(agent.last)
                self.assertEqual(agent.records, [])
        with self.assertRaises(TypeError):
            HarvestChoiceAgent(task_selector='not callable')
        with self.assertRaises(TypeError):
            HarvestChoiceAgent(enabled=1)

    def test_teacher_mode_cannot_open_model_weights(self):
        obs, cfg = observation('mature5dry')
        with patch('learning.bc_crop.agent.Network', side_effect=AssertionError('network forbidden')), \
             patch('numpy.load', side_effect=AssertionError('weights forbidden')):
            agent = HarvestChoiceAgent(choose_harvest)
            self.assertEqual(agent(obs, cfg)['farmer'], ['HARVEST'])
            self.assertEqual(agent.crop.mode, 'teacher')
            self.assertIsNone(agent.crop.network)
            self.assertIsNone(agent.last['checkpoint_sha256'])
        with self.assertRaises(TypeError):
            HarvestChoiceAgent(checkpoint='unused')


if __name__ == '__main__':
    unittest.main()
