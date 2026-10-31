"""Synthetic rule-controller probes, with no game, source file or model."""
from copy import deepcopy
import hashlib
import unittest
from unittest.mock import patch

from decision.api import CROPS, observe
from . import teacher


SOURCE = b'''\
def agent(observation, configuration):
    farm = observation['farms'][observation['player']]
    private = observation['private']
    units = [farm['farmer'], *farm['hands']]
    actions = [['PASS'] for _ in units]
    available = set(range(len(units)))
    seeds = dict(private['seeds'])
    needed_seeds = {}
    supply = {c: 0 for c in ('WHEAT','CARROT','TOMATO','STRAWBERRY','MELON')}
    projected_shed = dict(private['shed'])
    remaining, today, last_plant_age, final_day = 100, 12, 4, False
    tasks = [([0, 0], ['WATER'], 1.0), ([1, 0], ['HARVEST'], 2.0)]
    choices = [(1.0, 0, 0), (2.0, 1, 1)]
    *_, i, j = max(choices)
    actions[i] = list(tasks[j][1])
    return {'farmer': actions[0], 'hands': actions[1:], 'market': []}
'''


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def observation(step=0, player=0):
    farm = dict(money=100, tiles=[[None, None], [None, None]], farmer=[0, 0], hands=[[1, 1]],
                unlocked_quadrants=[], hires_today=0)
    return dict(player=player, step=step, day=0, hour=step, farms=[deepcopy(farm), deepcopy(farm)],
                private=dict(shed={}, seeds={crop:1 for crop in CROPS}, inventories=[{}, {}]),
                market=dict(inventory={crop:0 for crop in CROPS},
                            prices={crop:1 for crop in CROPS}, params={}),
                town=dict(unlocked_shops=[]))


def configuration():
    return dict(episodeSteps=720, boardSize=2, turnsPerDay=24, shedCapacity=100,
                maxMarketOrdersPerTurn=5, townShopSellInterval=3, farmHandCostMult=1,
                marketParams={})


def plain(source, obs, cfg):
    namespace = {}
    exec(compile(source, '<synthetic-teacher>', 'exec'), namespace)
    view = observe(obs, cfg)
    return namespace['agent'](view.observation, view.configuration)


class TeacherTests(unittest.TestCase):
    def probe(self, source=SOURCE, alias='market'):
        return teacher.TeacherProbe(source, sha(source), alias)

    def test_original_joint_request_and_teacher_label_are_preserved(self):
        for alias in teacher.ALIASES:
            with self.subTest(alias=alias):
                probe = self.probe(alias=alias)
                obs, cfg = observation(), configuration()
                original_obs, original_cfg = deepcopy(obs), deepcopy(cfg)
                self.assertEqual(probe(obs, cfg), plain(SOURCE, obs, cfg))
                self.assertEqual(obs, original_obs); self.assertEqual(cfg, original_cfg)
                self.assertEqual(probe.last['step'], 0)
                self.assertEqual(len(probe.last['records']), 1)
                record = probe.last['records'][0]
                self.assertEqual(record['source'], 'teacher')
                self.assertEqual(record['chosen'], record['teacher_label'])
                self.assertEqual((record['unit'], record['target'], record['op']), (1, [1, 0], 'HARVEST'))
                self.assertEqual(record['unit_request'], ['HARVEST'])
                self.assertIn(record['teacher_label'], [c['id'] for c in record['inputs']['candidates']])
                self.assertEqual(len(record['inputs']['shared_x']), 275)
                self.assertTrue(all(len(c['x']) == 98 for c in record['inputs']['candidates']))
                self.assertNotIn('teacher_label', record['inputs'])
                self.assertEqual(repr(probe), '<PrivateTeacherProbe>')

    def test_observation_and_configuration_are_projected_for_one_player(self):
        obs, cfg = observation(player=1), configuration()
        obs['farms'][0]['private'] = {'seeds': {'MELON': 999}}
        obs['farms'][0]['inventories'] = [{'MELON':999}]
        cfg['seed'] = 123456
        probe = self.probe(); probe(obs, cfg)
        inputs = probe.last['records'][0]['inputs']
        self.assertEqual(inputs['observation']['player'], 1)
        self.assertNotIn('private', inputs['observation']['farms'][0])
        self.assertNotIn('inventories', inputs['observation']['farms'][0])
        self.assertNotIn('seed', inputs['configuration'])

    def test_multiple_assignments_preserve_order_and_unit_requests(self):
        source = SOURCE.replace(
            b'    *_, i, j = max(choices)\n    actions[i] = list(tasks[j][1])',
            b'    for _ in range(2):\n        *_, i, j = max(choices)\n'
            b'        actions[i] = list(tasks[j][1])\n'
            b'        choices = [item for item in choices if item[-2] != i]')
        probe = self.probe(source)
        self.assertEqual(probe(observation(), configuration()), plain(source, observation(), configuration()))
        records = probe.last['records']
        self.assertEqual([r['op'] for r in records], ['HARVEST', 'WATER'])
        self.assertEqual([r['unit_request'] for r in records], [['HARVEST'], ['WATER']])

    def test_teacher_priority_does_not_enter_encoded_inputs(self):
        alternate = SOURCE.replace(b'choices = [(1.0, 0, 0), (2.0, 1, 1)]',
                                   b'choices = [(3.0, 0, 0), (2.0, 1, 1)]')
        first, second = self.probe(), self.probe(alternate)
        first(observation(), configuration()); second(observation(), configuration())
        left, right = first.last['records'][0], second.last['records'][0]
        self.assertNotEqual(left['teacher_label'], right['teacher_label'])
        self.assertEqual(left['inputs'], right['inputs'])

    def test_source_identity_and_injection_point_fail_closed(self):
        with self.assertRaises(teacher.ProbeError):
            teacher.TeacherProbe(SOURCE, '0'*64, 'market')
        with self.assertRaises(teacher.ProbeError):
            teacher.TeacherProbe(SOURCE, sha(SOURCE), 'other')
        absent = SOURCE.replace(b'max(choices)', b'min(choices)')
        multiple = SOURCE + b'\ndef helper(choices):\n    return max(choices)\n'
        for source in (absent, multiple):
            with self.assertRaises(teacher.ProbeError): self.probe(source)

    def test_reserved_callback_bindings_and_references_are_rejected(self):
        for prefix in (b'_probe_task_select = lambda *_: None\n',
                       b'def helper():\n    return _probe_task_select\n',
                       b"globals()['_probe_task_select'] = None\n",
                       b'match {}:\n    case {**_probe_task_select}:\n        pass\n'):
            with self.assertRaises(teacher.ProbeError): self.probe(prefix + SOURCE)

    def test_callback_identity_is_rechecked_after_source_initialization(self):
        def corrupt(code, namespace):
            exec(code, namespace)
            namespace[teacher.RESERVED] = object()
        with patch.object(teacher, 'exec', side_effect=corrupt, create=True):
            with self.assertRaises(teacher.ProbeError): self.probe()

    def test_candidate_limit_stops_before_encoding_without_truncation(self):
        source = SOURCE.replace(b'choices = [(1.0, 0, 0), (2.0, 1, 1)]',
                                b'choices = [(float(i), 0, 0) for i in range(257)]')
        probe = self.probe(source)
        with patch.object(teacher, 'encode_candidates', side_effect=AssertionError('must not encode')):
            with self.assertRaises(teacher.ProbeError): probe(observation(), configuration())
        self.assertIsNone(probe.last)

    def test_callback_limit_stops_before_thirty_third_encoding(self):
        source = SOURCE.replace(
            b'    *_, i, j = max(choices)\n    actions[i] = list(tasks[j][1])',
            b'    for _ in range(33):\n        *_, i, j = max(choices)\n'
            b'        actions[i] = list(tasks[j][1])')
        probe = self.probe(source)
        with patch.object(teacher, 'encode_candidates', wraps=teacher.encode_candidates) as encode:
            with self.assertRaises(teacher.ProbeError): probe(observation(), configuration())
        self.assertEqual(encode.call_count, 32)
        self.assertIsNone(probe.last)

    def test_invalid_teacher_candidate_fails_closed(self):
        source = SOURCE.replace(b"([1, 0], ['HARVEST'], 2.0)", b"([1, 0], ['INVALID'], 2.0)")
        probe = self.probe(source)
        with self.assertRaises(teacher.ProbeError): probe(observation(), configuration())
        self.assertIsNone(probe.last)

    def test_encoder_mutation_of_resource_reservations_is_rejected(self):
        probe = self.probe()
        real = teacher.encode_candidates
        def corrupt(context):
            result = real(context)
            context['seeds']['WHEAT'] += 1
            return result
        with patch.object(teacher, 'encode_candidates', side_effect=corrupt):
            with self.assertRaises(teacher.ProbeError): probe(observation(), configuration())
        self.assertIsNone(probe.last)
        self.assertEqual(probe(observation(), configuration()), plain(SOURCE, observation(), configuration()))

    def test_encoder_cannot_change_unwatched_demand_or_later_market_request(self):
        source = SOURCE.replace(b'    needed_seeds = {}\n',
                                b"    needed_seeds = {}\n    demand = {'WHEAT': 7}\n")
        source = source.replace(b"'market': []}", b"'market': [['SELL', 'WHEAT', demand['WHEAT']]]}")
        real = teacher.encode_candidates
        def corrupt(context):
            context['demand']['WHEAT'] += 100
            return real(context)
        probe = self.probe(source)
        with patch.object(teacher, 'encode_candidates', side_effect=corrupt):
            actual = probe(observation(), configuration())
        self.assertEqual(actual, plain(source, observation(), configuration()))
        self.assertEqual(actual['market'], [['SELL', 'WHEAT', 7]])

    def test_records_do_not_cross_turn_or_survive_failure(self):
        probe = self.probe()
        probe(observation(0), configuration())
        old_record = deepcopy(probe.last['records'][0])
        probe(observation(1), configuration())
        self.assertEqual((probe.last['step'], len(probe.last['records'])), (1, 1))
        self.assertEqual(old_record['inputs']['observation']['step'], 0)
        with self.assertRaises(teacher.ProbeError): probe(observation(719), configuration())
        self.assertIsNone(probe.last)
        probe(observation(0, player=1), configuration())
        self.assertEqual(probe.last['records'][0]['inputs']['observation']['player'], 1)

    def test_clear_releases_turn_records_and_next_turn_starts_empty(self):
        probe = self.probe()
        probe(observation(0), configuration())
        self.assertEqual(len(probe.last['records']), 1)
        probe.clear()
        self.assertIsNone(probe.last)
        self.assertEqual(probe._records, [])
        probe(observation(1), configuration())
        self.assertEqual((probe.last['step'], len(probe.last['records'])), (1, 1))
        self.assertEqual(probe.last['records'][0]['inputs']['observation']['step'], 1)

    def test_clear_during_turn_is_rejected(self):
        holder = {}
        def guard():
            holder['probe'].clear()
        probe = teacher.TeacherProbe(SOURCE, sha(SOURCE), 'market', guard=guard)
        holder['probe'] = probe
        with self.assertRaises(teacher.ProbeError): probe(observation(), configuration())
        self.assertIsNone(probe.last)
        probe.clear()


if __name__ == '__main__':
    unittest.main()
