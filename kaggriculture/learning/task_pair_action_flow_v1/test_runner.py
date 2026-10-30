"""Synthetic regression cases; never opens real matches or model files."""
import gzip
import hashlib
import io
import json
from pathlib import Path
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

from learning.task_pair_action_flow_v1 import runner


def event(unit, status, *, step=7, player=0, selected='HARVEST', requested='NORTH'):
    return dict(step=step, player=player, unit=unit, source='callback', selected_op=selected,
                request_op=requested, label_support='unknown' if status is None else 'observed',
                status=status)


def compact(units):
    return {'units': [[] for _ in range(7)] + [units]}


def metrics(events):
    return {'result': {'events': events}}


def selection(*units):
    return {unit: {'source': 'callback', 'selected_op': 'HARVEST', 'request_op': 'NORTH'}
            for unit in units}


class StatusTests(unittest.TestCase):
    def test_changed_units_join_labels_and_keep_absent_unknown(self):
        out = runner.execution_summary(compact(['same', 'old', 'removed']),
            compact(['same', 'new', 'replacement', 'added']),
            metrics([event(1, 'blocked'), event(2, None)]), metrics([event(1, 'executed')]),
            selection(1, 2), selection(1), 7, 0, False)
        self.assertEqual(out['changed_unit_requests'], 3)
        self.assertEqual(out['unit_status']['baseline']['blocked'], 1)
        self.assertEqual(out['unit_status']['baseline']['unknown'], 2)
        self.assertEqual(out['unit_status']['control']['executed'], 1)
        self.assertEqual(out['unit_status']['control']['unknown'], 2)
        # Same SELL request can execute differently after a different DROP.
        self.assertFalse(out['market_request_changed'])
        self.assertEqual(out['market_order_execution_status'], 'unknown_not_saved_per_order')

    def test_same_unit_request_is_not_counted_as_a_changed_request(self):
        out = runner.execution_summary(compact(['x']), compact(['x']),
            metrics([event(0, 'executed')]), metrics([event(0, 'no_effect')]),
            selection(0), selection(0), 7, 0, True)
        self.assertEqual(out['changed_unit_requests'], 0)
        self.assertEqual(sum(out['unit_status']['control'].values()), 0)
        self.assertTrue(out['market_request_changed'])
        self.assertEqual(out['market_order_execution_status'], 'unknown_not_saved_per_order')

    def test_mismatched_or_duplicate_selected_event_fails_closed(self):
        for events in ([event(0, 'executed', requested='SOUTH')],
                       [event(0, 'executed'), event(0, 'blocked')]):
            with self.subTest(events=events), self.assertRaises(ValueError):
                runner.execution_summary(compact(['old']), compact(['new']),
                    metrics(events), metrics([]), selection(0), {}, 7, 0, False)


class InputTests(unittest.TestCase):
    def test_trace_tap_marks_eof_only_after_exhaustion(self):
        rows = iter([{'step': 7, 'records': [{'unit': 0, 'source': 'callback',
                    'op': 'HARVEST', 'unit_request': ['NORTH']}]}])
        chosen, done = {}, {}
        with patch.object(runner, '_turn', side_effect=lambda line, *_: line['records']):
            stream = runner.capture_branch(rows, 7, 0, chosen, done)
            next(stream)
            self.assertFalse(done.get('eof', False))
            with self.assertRaises(StopIteration):
                next(stream)
        self.assertTrue(done['eof'])
        self.assertEqual(chosen[0]['request_op'], 'NORTH')

    def test_guard_initial_runtime_and_elapsed_limits(self):
        state = {'memory': runner.INITIAL_MEMORY, 'time': 0}
        guard = runner.Guard(available=lambda: state['memory'], clock=lambda: state['time'])
        self.assertTrue(guard.check(initial=True))
        state['memory'] = runner.RUN_MEMORY - 1
        with self.assertRaises(MemoryError):
            guard.check()
        state.update(memory=runner.INITIAL_MEMORY, time=runner.MAX_SECONDS)
        with self.assertRaises(TimeoutError):
            guard.check()
        entered_earlier = runner.Guard(available=lambda: runner.INITIAL_MEMORY,
                                       clock=lambda: runner.MAX_SECONDS, started=0)
        with self.assertRaises(TimeoutError):
            entered_earlier.check(initial=True)

    def test_summary_is_removed_when_post_write_sample_fails(self):
        class FakePath:
            present = False
            def __truediv__(self, _name):
                return self
            def exists(self):
                return self.present
            def unlink(self, missing_ok=False):
                self.present = False

        output = FakePath()
        def fake_write(_root, path, _result):
            path.present = True
        def timed_out():
            self.assertTrue(output.present)
            raise TimeoutError('synthetic')
        with patch.object(runner.driver, '_write_new', side_effect=fake_write):
            with self.assertRaises(TimeoutError):
                runner.write_summary(output, {'status': 'passed'}, timed_out)
        self.assertFalse(output.present)

    def test_replay_same_open_handle_binds_digest_and_seed(self):
        payload = json.dumps({'info': {'seed': 123}, 'steps': []}).encode()
        compressed = gzip.compress(payload)
        expected = hashlib.sha256(compressed).hexdigest()
        row = {'game': 'fixture', 'seed': 123, 'replay_gzip_sha256': expected}
        fake = io.BytesIO(compressed)
        with patch.object(runner.driver, '_no_links'), patch.object(Path, 'is_file', return_value=True), \
             patch.object(Path, 'open', return_value=fake) as opened, \
             patch.object(runner, 'compact_replay', return_value={'ok': True}) as parse:
            result = runner.read_compact(Path('test-root'), row, 0, lambda: True)
        self.assertEqual(result, {'ok': True})
        self.assertEqual(opened.call_count, 1)
        self.assertEqual(parse.call_args.args[1], {'side': 0, 'expected': row})
        bad = dict(row, seed=124)
        with patch.object(runner.driver, '_no_links'), patch.object(Path, 'is_file', return_value=True), \
             patch.object(Path, 'open', return_value=io.BytesIO(compressed)), \
             patch.object(runner, 'compact_replay'):
            with self.assertRaises(ValueError):
                runner.read_compact(Path('test-root'), bad, 0, lambda: True)

    def test_paired_entry_requires_accepted_branch_and_trace_eof(self):
        accepted, public = {}, {}
        for alias in runner.ALIASES:
            for side in (0, 1):
                for arm in runner.ARMS:
                    accepted[alias, side, arm] = {
                        'row': {'game': arm, 'trace_sha256': 'synthetic'}, 'metrics': metrics([])}
                public[alias, side] = {'comparison': {'complete_request_prefix': {
                    'scope': 'same_complete_history', 'status': 'observed',
                    'reason': 'actor_request_difference', 'step': 7}}}

        def trace(*_):
            for step in range(719):
                yield {'step': step, 'records': []}

        def flow(_ref, _new, left, right, **_):
            list(left)
            list(right)
            return {'scope': 'v2_semantic_common_history', 'trace_turns_per_game': 719,
                    'first_branch_kind': 'actor', 'first_branch_step': 7,
                    'actor_request_changed_at_first_branch': True,
                    'actor_only_branch_attribution': 'observed',
                    'actor_request_components_differ': {'farmer': True, 'hands': False, 'market': False},
                    'counts': {}, 'first_same_input_choice_difference': None,
                    'post_branch_windows': {}, 'interpretation': 'synthetic'}

        def replay(_stage, row, _side, _check):
            return compact(['old' if row['game'] == 'baseline' else 'new'])

        with patch.object(runner, 'read_compact', side_effect=replay), \
             patch.object(runner.driver, '_trace', side_effect=trace), \
             patch.object(runner, '_turn', return_value=[]), \
             patch.object(runner.analysis, 'compare_pair', side_effect=flow), \
             redirect_stdout(io.StringIO()):
            result = runner.compare_existing(Path('synthetic'), accepted, public, lambda: True)
        self.assertEqual(len(result), 4)
        for row in result:
            self.assertEqual(row['execution_labels']['changed_unit_requests'], 1)
            self.assertEqual(row['execution_labels']['unit_status']['baseline']['unknown'], 1)

        def not_consumed(*_args, **_kwargs):
            return flow(None, None, iter(()), iter(()))

        with patch.object(runner, 'read_compact', side_effect=replay), \
             patch.object(runner.driver, '_trace', side_effect=trace), \
             patch.object(runner.analysis, 'compare_pair', side_effect=not_consumed):
            with self.assertRaises(ValueError):
                runner.compare_existing(Path('synthetic'), accepted, public, lambda: True)


if __name__ == '__main__':
    unittest.main()
