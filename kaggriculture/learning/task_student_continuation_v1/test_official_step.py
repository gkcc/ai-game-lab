"""Small official one-step fixtures; no saved games or personal data read."""
from copy import deepcopy
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from decision.verify_mechanisms import make
from kaggle_environments.core import Environment
from . import official_step


PASS = {'farmer': ['PASS'], 'hands': [], 'market': []}
EAST = {'farmer': ['EAST'], 'hands': [], 'market': []}


def plain(value):
    return json.loads(json.dumps(value, allow_nan=False))


def fixture(step, joint=None):
    """Construct two real official frames, not a simulated full history."""
    env = make('kaggriculture', configuration={'seed': 123456789, 'weedSpawnChance': 0.02}, debug=False)
    env.info['test_marker'] = 'PRIVATE_FIXTURE_MARKER'
    env.state[0].observation.step = step
    for state in env.state:
        state.observation.day, state.observation.hour = divmod(step, 24)
        state.observation.private['inventories'][0] = {'WHEAT': 1, 'CARROT': 1}
    before = plain(env.state)
    env.steps = [None] * step + [deepcopy(env.state)]
    env.step(deepcopy(joint) if joint is not None else [deepcopy(PASS), deepcopy(PASS)])
    after = plain(env.state)
    frames = [None] * 720
    frames[step], frames[step + 1] = before, after
    return dict(name='kaggriculture', configuration=plain(env.configuration),
                info=plain(env.info), steps=frames)


class OfficialStepTests(unittest.TestCase):
    def call(self, replay, step, side=0, baseline=PASS, alternative=EAST, guard=None):
        return official_step.compare_step(replay, step=step, side=side,
            baseline_request=baseline, intervention_request=alternative, guard=guard)

    def test_official_boundary_frames_match_and_input_stays_unchanged(self):
        for step in (0, 23, 71, 718):
            with self.subTest(step=step):
                replay = fixture(step)
                saved = json.dumps(replay, ensure_ascii=False, allow_nan=False)
                result = self.call(replay, step)
                self.assertTrue(result['factual_reproduced'])
                self.assertTrue(result['request_changed'])
                self.assertEqual(result['scope'], 'one_official_transition')
                self.assertFalse(result['complete_game'])
                self.assertFalse(result['future_return_evaluated'])
                self.assertEqual(json.dumps(replay, ensure_ascii=False, allow_nan=False), saved)

    def test_same_request_has_no_physical_state_change_on_a_day_boundary(self):
        replay = fixture(23)
        result = self.call(replay, 23, alternative=PASS)
        self.assertFalse(result['request_changed'])
        self.assertFalse(result['state_changed'])
        self.assertEqual(result, self.call(replay, 23, alternative=PASS))

    def test_both_seats_keep_opponent_request_and_redact_results(self):
        for side in (0, 1):
            original = [deepcopy(PASS), deepcopy(PASS)]
            original[1 - side] = deepcopy(EAST)
            replay = fixture(0, original)
            result = self.call(replay, 0, side=side)
            self.assertTrue(result['state_changed'])
            encoded = json.dumps(result, allow_nan=False)
            self.assertNotIn('PRIVATE_FIXTURE_MARKER', encoded)
            self.assertNotIn('123456789', encoded)
            self.assertEqual(set(result), {'factual_reproduced', 'request_changed', 'state_changed',
                                           'scope', 'complete_game', 'future_return_evaluated'})

    def test_different_but_ineffective_request_does_not_count_as_state_change(self):
        replay = fixture(0)
        result = self.call(replay, 0, alternative={'farmer': ['DIG'], 'hands': [], 'market': []})
        self.assertTrue(result['request_changed'])
        self.assertFalse(result['state_changed'])

    def test_market_order_sequence_is_part_of_the_complete_request(self):
        ordered = {'farmer': ['PASS'], 'hands': [],
                   'market': [['BUY_SEED', 'WHEAT', 1], ['BUY_SEED', 'CARROT', 1]]}
        reordered = deepcopy(ordered)
        reordered['market'].reverse()
        rival = {'farmer': ['EAST'], 'hands': [], 'market': [['BUY_SEED', 'WHEAT', 1]]}
        replay = fixture(0, [ordered, rival])
        with self.assertRaises(official_step.OfficialStepError):
            self.call(replay, 0, baseline=reordered, alternative=ordered)
        result = self.call(replay, 0, baseline=ordered, alternative=reordered)
        self.assertTrue(result['request_changed'])
        self.assertFalse(result['state_changed'])

    def test_wrong_baseline_rejected_without_executing_an_arm(self):
        replay = fixture(0)
        with patch.object(official_step, 'make', wraps=official_step.make) as make_spy:
            with self.assertRaises(official_step.OfficialStepError):
                self.call(replay, 0, baseline=EAST)
        self.assertEqual(make_spy.call_count, 0)

    def test_imported_official_core_identity_is_required_before_execution(self):
        replay = fixture(0)
        self.assertIs(official_step.make, official_step.official_core.make)
        self.assertIs(Environment, official_step.official_core.Environment)
        with patch.object(official_step.official_core, '__file__', official_step.rules.__file__), \
             patch.object(official_step, 'make', wraps=official_step.make) as make_spy:
            with self.assertRaisesRegex(official_step.OfficialStepError, '^Official core identity differs$'):
                self.call(replay, 0)
        self.assertEqual(make_spy.call_count, 0)

    def test_official_core_read_error_propagates_unchanged(self):
        replay = fixture(0)
        core_path, original_read = Path(official_step.official_core.__file__), Path.read_bytes
        signal = OSError('core-read-sentinel')
        def read_bytes(path):
            if path == core_path:
                raise signal
            return original_read(path)
        with patch.object(Path, 'read_bytes', new=read_bytes):
            with self.assertRaises(OSError) as caught:
                self.call(replay, 0)
        self.assertIs(caught.exception, signal)

    def test_factual_frame_mismatch_stops_before_intervention(self):
        replay = fixture(0)
        replay['steps'][1][0]['observation']['farms'][0]['money'] += 1
        with patch.object(official_step, 'make', wraps=official_step.make) as make_spy:
            with self.assertRaises(official_step.OfficialStepError):
                self.call(replay, 0)
        self.assertEqual(make_spy.call_count, 1)

    def test_inventory_order_is_part_of_factual_v2(self):
        replay = fixture(0)
        inventory = replay['steps'][1][0]['observation']['private']['inventories'][0]
        replay['steps'][1][0]['observation']['private']['inventories'][0] = dict(reversed(list(inventory.items())))
        with self.assertRaises(official_step.OfficialStepError):
            self.call(replay, 0)

    def test_invalid_metadata_coordinates_or_incomplete_request_rejected(self):
        replay = fixture(0)
        for step, side in ((719, 0), (-1, 0), (True, 0), (0, True), (0, 2)):
            with self.subTest(step=step, side=side), self.assertRaises(official_step.OfficialStepError):
                self.call(replay, step, side)
        for field, value in (('name', 'other'), ('steps', []), ('info', {})):
            bad = deepcopy(replay); bad[field] = value
            with self.subTest(field=field), self.assertRaises(official_step.OfficialStepError):
                self.call(bad, 0)
        bad = deepcopy(replay); bad['configuration']['seed'] = 123
        with self.assertRaises(official_step.OfficialStepError):
            self.call(bad, 0)
        with self.assertRaises(official_step.OfficialStepError):
            self.call(replay, 0, alternative={'farmer': ['PASS']})

    def test_guard_stops_after_factual_arm_before_intervention(self):
        replay = fixture(0)
        calls = [0]
        def guard():
            calls[0] += 1
            if calls[0] == 5:
                raise TimeoutError('guard')
        with patch.object(official_step, 'make', wraps=official_step.make) as make_spy:
            with self.assertRaises(TimeoutError):
                self.call(replay, 0, guard=guard)
        self.assertEqual(make_spy.call_count, 1)

    def test_guard_exceptions_propagate_unchanged_at_every_checkpoint(self):
        replay = fixture(0)
        count = [0]
        def observe_guard():
            count[0] += 1
        self.call(replay, 0, guard=observe_guard)
        self.assertGreaterEqual(count[0], 6)
        for checkpoint in range(1, count[0] + 1):
            with self.subTest(checkpoint=checkpoint):
                calls, signal = [0], RuntimeError('guard-sentinel')
                def guard():
                    calls[0] += 1
                    if calls[0] == checkpoint:
                        raise signal
                with self.assertRaises(RuntimeError) as caught:
                    self.call(replay, 0, guard=guard)
                self.assertIs(caught.exception, signal)
        for cls in (ValueError, OSError, MemoryError, TimeoutError, KeyboardInterrupt):
            with self.subTest(kind=cls.__name__):
                calls, signal = [0], cls('guard-sentinel')
                def guard():
                    calls[0] += 1
                    if calls[0] == 3:
                        raise signal
                with self.assertRaises(cls) as caught:
                    self.call(replay, 0, guard=guard)
                self.assertIs(caught.exception, signal)

    def test_engine_resource_failures_propagate_on_both_arms(self):
        replay = fixture(0)
        original_make, original_step = official_step.make, Environment.step
        for site in ('make', 'step'):
            for failure_call in (1, 2):
                for cls in (OSError, MemoryError, TimeoutError):
                    with self.subTest(site=site, arm=failure_call, kind=cls.__name__):
                        calls, signal = [0], cls('resource-sentinel')
                        def controlled_make(*args, **kwargs):
                            calls[0] += 1
                            if calls[0] == failure_call:
                                raise signal
                            return original_make(*args, **kwargs)
                        def controlled_step(environment, *args, **kwargs):
                            calls[0] += 1
                            if calls[0] == failure_call:
                                raise signal
                            return original_step(environment, *args, **kwargs)
                        target = (patch.object(official_step, 'make', side_effect=controlled_make)
                                  if site == 'make' else patch.object(Environment, 'step', new=controlled_step))
                        with target, self.assertRaises(cls) as caught:
                            self.call(replay, 0)
                        self.assertIs(caught.exception, signal)
                        self.assertEqual(calls[0], failure_call)

    def test_nonresource_engine_failure_is_explicit_and_sanitized(self):
        replay = fixture(0)
        for site in ('make', 'step'):
            for cls in (ValueError, RuntimeError):
                with self.subTest(site=site, kind=cls.__name__):
                    signal = cls('PRIVATE_SYNTHETIC_VALUE')
                    target = (patch.object(official_step, 'make', side_effect=signal)
                              if site == 'make' else patch.object(Environment, 'step', side_effect=signal))
                    with target, self.assertRaisesRegex(official_step.OfficialStepError,
                                                         '^Official transition failed$') as caught:
                        self.call(replay, 0)
                    self.assertNotIn('PRIVATE_SYNTHETIC_VALUE', str(caught.exception))
                    self.assertIsNone(caught.exception.__cause__)


if __name__ == '__main__':
    unittest.main()
