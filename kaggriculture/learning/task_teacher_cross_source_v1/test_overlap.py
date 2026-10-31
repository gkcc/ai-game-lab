"""Pure synthetic checks; no archives, models, games, or seed values."""
from dataclasses import asdict
import json
import unittest
from unittest.mock import patch

import numpy as np

from . import overlap as o


def old_group(*, value=0.0, target=0, reverse=False):
    shared = np.zeros(o.SHARED_DIM, dtype=np.float32)
    shared[0] = value
    candidates = np.zeros((2, o.LOCAL_DIM), dtype=np.float32)
    candidates[:, 0] = (0, 1)
    if reverse:
        candidates = candidates[::-1].copy()
    return o.OldGroup(shared, candidates, target)


def new_from(old, *, split='train', target=None):
    raw = np.concatenate((np.broadcast_to(old.shared, (len(old.candidates), o.SHARED_DIM)),
                          old.candidates), axis=1)
    return o.NewGroup(raw, old.target_index if target is None else target, split)


class OverlapTests(unittest.TestCase):
    def setUp(self):
        self.mean = np.zeros(o.DIM, dtype=np.float64)
        self.scale = np.ones(o.DIM, dtype=np.float64)

    def compare(self, old, new, guard=lambda: None):
        return o.compare(old, new, mean=self.mean, scale=self.scale, guard=guard)

    def test_order_labels_and_split_summary(self):
        first = old_group(target=0)
        conflict = old_group(target=1)
        unique = old_group(value=10, target=1)
        reversed_group = old_group(reverse=True)
        new = [new_from(first, split='train'), new_from(unique, split='dev'),
               new_from(reversed_group, split='dev')]
        result = self.compare(iter((first, conflict, unique)), new)
        self.assertEqual(result.old_groups, 3)
        self.assertEqual([(r.raw, r.normalized) for r in result.rows],
                         [('conflict', 'conflict'), ('same', 'same'), ('none', 'none')])
        self.assertEqual(result.summary['raw']['train'], {'same': 0, 'conflict': 1, 'none': 0})
        self.assertEqual(result.summary['raw']['dev'], {'same': 1, 'conflict': 0, 'none': 1})
        self.assertEqual(result.summary['normalized'], result.summary['raw'])

    def test_signed_zero_has_one_canonical_representation(self):
        old = old_group()
        old.shared[0] = np.float32(-0.0)
        old.candidates[0, 1] = np.float32(-0.0)
        new = new_from(old)
        new.features[new.features == 0] = np.float32(0.0)
        self.assertEqual(self.compare([old], [new]).rows[0], o.Match('train', 'same', 'same'))

    def test_normalized_rounding_collision_is_separate_from_raw(self):
        old = old_group(value=np.float32(1.0), target=0)
        other = old_group(value=np.nextafter(np.float32(1), np.float32(2)), target=0)
        self.mean[0] = -1000.0
        same = new_from(other, split='train', target=0)
        conflict = new_from(other, split='dev', target=1)
        result = self.compare([old], [same, conflict])
        self.assertEqual([(row.raw, row.normalized) for row in result.rows],
                         [('none', 'same'), ('none', 'conflict')])

    def test_hash_collision_requires_exact_bytes(self):
        old = old_group(value=1, target=0)
        new = [new_from(old, split='train'), new_from(old_group(value=2), split='dev')]
        with patch.object(o, '_digest', return_value=b'x' * 32):
            result = self.compare([old], new)
        self.assertEqual([row.raw for row in result.rows], ['same', 'none'])
        self.assertEqual([row.normalized for row in result.rows], ['same', 'none'])

    def test_old_iterator_is_single_pass_and_guard_stops_work(self):
        old = old_group()

        class Once:
            def __init__(self):
                self.calls = 0

            def __iter__(self):
                self.calls += 1
                if self.calls != 1:
                    raise AssertionError('old input was iterated twice')
                yield old

        stream = Once()
        self.assertEqual(self.compare(stream, [new_from(old)]).old_groups, 1)
        self.assertEqual(stream.calls, 1)
        calls = []

        def stop():
            calls.append(True)
            if len(calls) == 4:
                raise TimeoutError('synthetic guard')

        with self.assertRaises(TimeoutError):
            self.compare([old], [new_from(old)], guard=stop)

    def test_incremental_matcher_finalizes_once_and_rejects_later_input(self):
        first, other = old_group(), old_group(value=10, target=1)
        new = [new_from(first), new_from(other, split='dev')]
        matcher = o.Matcher(new, mean=self.mean, scale=self.scale, guard=lambda: None)
        matcher.consume(first)
        matcher.consume(other)
        result = matcher.finish()
        self.assertEqual(result.old_groups, 2)
        self.assertEqual([row.raw for row in result.rows], ['same', 'same'])
        with self.assertRaises(ValueError):
            matcher.finish()
        with self.assertRaises(ValueError):
            matcher.consume(first)

    def test_interrupted_consume_cannot_finalize_partial_match(self):
        old = old_group()
        active = [False]
        calls = []

        def stop():
            if active[0]:
                calls.append(True)
                if len(calls) == 3:
                    raise TimeoutError('synthetic interruption')

        matcher = o.Matcher([new_from(old)], mean=self.mean, scale=self.scale, guard=stop)
        active[0] = True
        with self.assertRaises(TimeoutError):
            matcher.consume(old)
        active[0] = False
        with self.assertRaises(ValueError):
            matcher.finish()
        with self.assertRaises(ValueError):
            matcher.consume(old)

    def test_calling_two_scans_requires_both_to_return_before_finish(self):
        old = old_group()
        events = []

        def first(consume):
            consume(old)
            events.append('first_returned')

        def second(consume):
            consume(old)
            events.append('second_returned')

        def failed(consume):
            consume(old)
            raise ValueError('synthetic final identity check failed')

        def scan_all(scans):
            matcher = o.Matcher([new_from(old)], mean=self.mean, scale=self.scale, guard=lambda: None)
            for scan in scans:
                scan(matcher.consume)
            return matcher.finish()

        self.assertEqual(scan_all((first, second)).old_groups, 2)
        self.assertEqual(events, ['first_returned', 'second_returned'])
        with patch.object(o.Matcher, 'finish', side_effect=AssertionError('must not finalize')):
            with self.assertRaisesRegex(ValueError, 'final identity'):
                scan_all((first, failed))

    def test_invalid_groups_and_normalization_fail_closed(self):
        old = old_group()
        base = new_from(old)
        variants = [o.NewGroup(base.features.astype(np.float64), 0, 'train'),
                    o.NewGroup(base.features, True, 'train'),
                    o.NewGroup(base.features, 2, 'train'),
                    o.NewGroup(base.features, 0, 'other'),
                    o.NewGroup(np.zeros((257, o.DIM), dtype=np.float32), 0, 'train')]
        bad = base.features.copy()
        bad[0, 0] = np.nan
        variants.append(o.NewGroup(bad, 0, 'train'))
        bad = base.features.copy()
        bad[1, 0] = 10
        variants.append(o.NewGroup(bad, 0, 'train'))
        for group in variants:
            with self.subTest(kind='synthetic'), self.assertRaises(ValueError):
                self.compare([old], [group])
        with self.assertRaises(ValueError):
            self.compare([old], [base] * (o.MAX_NEW_GROUPS + 1))
        self.scale[0] = 0
        with self.assertRaises(ValueError):
            self.compare([old], [base])

    def test_result_and_repr_do_not_expose_features_or_digests(self):
        old = old_group(value=42)
        new = new_from(old)
        result = self.compare([old], [new])
        encoded = json.dumps(asdict(result))
        for word in ('features', 'candidates', 'shared', 'digest', 'seed'):
            self.assertNotIn(word, encoded)
        self.assertNotIn('42', repr(old) + repr(new))


if __name__ == '__main__':
    unittest.main()
