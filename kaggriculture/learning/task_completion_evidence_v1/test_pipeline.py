import unittest
from unittest.mock import patch

from . import pipeline


def games():
    result = []
    for seed in (1, 2):
        for arm in ('baseline', 'candidate', 'teacher_task'):
            key = f'{seed}-round2-0-{arm}'
            result.append(dict(key=key, seed=seed, opponent='round2', side=0, arm=arm))
        key = f'{seed}-market-1-candidate'
        result.append(dict(key=key, seed=seed, opponent='market', side=1, arm='candidate'))
    return result


class PipelineTests(unittest.TestCase):
    def test_eight_games_sixteen_distinct_paths_and_metadata_binding(self):
        sample = games()
        indexed = {i: g for i, g in enumerate(sample)}
        def path(_root, game_part, kind):
            return (kind, game_part)
        enriched = [dict(g, replay=g['key'], trace=g['key']) for g in sample]
        with patch.object(pipeline, 'validate_manifest', return_value=({i: g for i, g in enumerate(enriched)}, set())), patch.object(pipeline, 'raw_path', side_effect=path):
            self.assertEqual(len(pipeline.select_games({}, enriched)), 8)
            changed = [dict(g) for g in enriched]
            changed[0]['seed'] = 99
            with self.assertRaises(ValueError):
                pipeline.select_games({}, changed)
            with self.assertRaises(ValueError):
                pipeline.select_games({}, enriched[:-1] + [enriched[0]])

    def test_raw_path_refuses_escape_and_link(self):
        for value in ('../outside', 'games/../old', chr(67)+':/synthetic', 'traces/not-replay'):
            with self.assertRaises(ValueError):
                pipeline.raw_path(pipeline.SOURCE, dict(path=value, size=1, sha256='0'*64), 'replay')

    def test_trace_mismatch_propagates_and_no_extra_turn(self):
        with patch.object(pipeline, 'validate_trace', side_effect=ValueError('mismatch')):
            with self.assertRaisesRegex(ValueError, 'mismatch'):
                pipeline.selections_from_trace(iter([{}]), {}, lambda: None)
        with patch.object(pipeline, 'validate_trace', return_value=[]):
            with self.assertRaisesRegex(ValueError, 'extra'):
                pipeline.selections_from_trace(iter([{}] * 720), {}, lambda: None)

    def test_event_join_filters_and_flushes_empty_turns(self):
        class Tracker:
            def __init__(self): self.calls = []
            def step(self, step, selections, events, day): self.calls.append((step, list(events)))
            def finish(self): return self.calls
        tracker = Tracker()
        join = pipeline.EventJoin(tracker, [[] for _ in range(719)], [0]*719, 0, lambda: None)
        join.sink(dict(step=0, phase='unit', player=1, private='not passed'))
        join.sink(dict(step=0, phase='unit', player=0, unit=0))
        join.sink(dict(step=2, phase='market', player=0))
        rows = join.finish()
        self.assertEqual(len(rows), 719)
        self.assertEqual(len(rows[0][1]), 1)
        self.assertEqual(rows[1][1], [])
        with self.assertRaises(ValueError):
            join.sink(dict(step=1, phase='unit', player=0))

    def test_rows_are_strict_allowlist(self):
        row = dict(game='g', side=0, first_step=0, last_step=0, first_ordinal=0, unit=0,
                   target=[0, 0], op='PLANT', chosen='x', status='unknown', reason='unconfirmed',
                   observed_steps=1, had_blocked=False, reason_counts={})
        self.assertEqual(pipeline.validate_rows([row], dict(key='g', side=0)), [row])
        self.assertEqual(pipeline.validate_rows([row], dict(key='g', side=0), 1), [row])
        with self.assertRaisesRegex(ValueError, 'count mismatch'):
            pipeline.validate_rows([row], dict(key='g', side=0), 2)
        with self.assertRaisesRegex(ValueError, 'Duplicate episode origin'):
            pipeline.validate_rows([row, dict(row)], dict(key='g', side=0), 2)
        for forbidden in ('teacher_label', 'observation', 'opponent_private'):
            with self.assertRaises(ValueError):
                pipeline.validate_rows([dict(row, **{forbidden: 'secret'})], dict(key='g', side=0))

    def test_deadline_failure_stops_join(self):
        def fail(): raise TimeoutError('budget')
        join = pipeline.EventJoin(None, [], [], 0, fail)
        with self.assertRaises(TimeoutError):
            join.sink(dict(step=0))


if __name__ == '__main__':
    unittest.main()
