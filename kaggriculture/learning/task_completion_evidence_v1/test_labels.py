import unittest
from copy import deepcopy
from learning.task_completion_evidence_v1.labels import EpisodeTracker, observed_effect


def plant(**kw):
    return dict(kind='PLANT', crop='WHEAT', yield_units=3, watered_today=False,
                fertilized_until_day=9, **kw)


def event(op, before, after, inv_before=(), inv_after=(), seeds_before=None, seeds_after=None):
    return dict(step=1, player=0, unit=0, phase='unit', requested=[op], effective=[op],
                position=[2, 3], status='executed', tile_before=before, tile_after=after,
                before=dict(inventories=[list(inv_before)], seeds=seeds_before or {}),
                after=dict(inventories=[list(inv_after)], seeds=seeds_after or {}))


def selection(step=1, op='DIG', request=None, unit=0, target=None, ordinal=0):
    return dict(step=step, unit=unit, target=target or [2, 3], op=op,
                chosen=f'u{unit}:x2:y3:{op}', unit_request=request or [op], ordinal=ordinal)


class EvidenceTests(unittest.TestCase):
    def test_five_effects(self):
        water = plant(); wet = deepcopy(water); wet['watered_today'] = True
        fertilized = deepcopy(water)
        cases = [
            ('DIG', event('DIG', {'kind': 'WEED'}, None)),
            ('WATER', event('WATER', water, wet)),
            ('PLANT', event('PLANT', None, plant(), seeds_before={'WHEAT': 2}, seeds_after={'WHEAT': 1})),
            ('HARVEST', event('HARVEST', plant(), None, [('WHEAT', 2)], [('WHEAT', 5)])),
            ('FERTILIZE', event('FERTILIZE', water, fertilized, [('FERTILIZER', 2)], [('FERTILIZER', 1)])),
        ]
        cases[2][1]['effective'] = ['PLANT', 'WHEAT']
        for op, e in cases:
            with self.subTest(op=op):
                self.assertTrue(observed_effect(op, e, 5))

    def test_three_steps_blocked_then_success(self):
        t = EpisodeTracker('g', 0)
        move = event('EAST', None, None); move['position'] = [1, 3]
        t.step(1, [selection(request=['EAST'])], [move], 0)
        blocked = event('DIG', {'kind': 'WEED'}, None); blocked.update(step=2, effective=['PASS'], status='blocked')
        t.step(2, [selection(2)], [blocked], 0)
        success = event('DIG', {'kind': 'WEED'}, None); success['step'] = 3
        t.step(3, [selection(3)], [success], 0)
        row = t.finish()[0]
        self.assertEqual((row['status'], row['first_step'], row['last_step'], row['observed_steps'], row['had_blocked']),
                         ('completed_observed', 1, 3, 3, True))

    def test_no_effect_and_missing_fields(self):
        for e in [event('DIG', None, None), {'step': 1, 'player': 0, 'unit': 0, 'phase': 'unit'}]:
            t = EpisodeTracker('g', 0); t.step(1, [selection()], [e], 0)
            self.assertEqual(t.finish()[0]['status'], 'unknown')

    def test_other_unit_and_conflicting_duplicate_events(self):
        e = event('DIG', {'kind': 'WEED'}, None); e['unit'] = 1
        for events in [[e], [dict(e, unit=0), dict(e, unit=0)]]:
            t = EpisodeTracker('g', 0); t.step(1, [selection()], events, 0)
            self.assertEqual(t.finish()[0]['status'], 'unknown')

    def test_interruption_and_gap_do_not_join(self):
        t = EpisodeTracker('g', 0); t.step(1, [selection()], [], 0)
        t.step(3, [selection(3)], [], 0); rows = t.finish()
        self.assertEqual([(r['first_step'], r['status']) for r in rows], [(1, 'unknown'), (3, 'unknown')])

    def test_out_of_order_and_duplicate_unit(self):
        t = EpisodeTracker('g', 0); t.step(1, [selection()], [], 0)
        with self.assertRaises(ValueError): t.step(1, [], [], 0)
        t.step(2, [selection(2), selection(2, ordinal=1)], [], 0)
        self.assertEqual(t.finish()[0]['status'], 'unknown')

    def test_blocked_and_no_effect_reasons(self):
        for status, expected in [('blocked', 'official_rewrite_blocked'), ('no_effect', 'official_no_effect')]:
            e = event('DIG', {'kind': 'WEED'}, {'kind': 'WEED'})
            e['status'] = status
            if status == 'blocked': e['effective'] = ['PASS']
            t = EpisodeTracker('g', 0); t.step(1, [selection()], [e], 0)
            row = t.finish()[0]
            self.assertEqual((row['status'], row['reason'], row['reason_counts'][expected]), ('unknown', expected, 1))
            self.assertEqual(row['had_blocked'], status == 'blocked')

    def test_blocked_then_success_retains_count(self):
        t = EpisodeTracker('g', 0)
        blocked = event('DIG', {'kind': 'WEED'}, {'kind': 'WEED'})
        blocked.update(status='blocked', effective=['PASS'])
        t.step(1, [selection()], [blocked], 0)
        success = event('DIG', {'kind': 'WEED'}, None); success['step'] = 2
        t.step(2, [selection(2)], [success], 0)
        row = t.finish()[0]
        self.assertEqual((row['status'], row['reason'], row['had_blocked']), ('completed_observed', 'atomic_effect', True))
        self.assertEqual(row['reason_counts'], {'official_rewrite_blocked': 1})

    def test_wrong_request_or_position(self):
        for e in [dict(event('DIG', {'kind': 'WEED'}, None), position=[1, 3]),
                  dict(event('DIG', {'kind': 'WEED'}, None), requested=['WATER'])]:
            t = EpisodeTracker('g', 0); t.step(1, [selection()], [e], 0)
            self.assertEqual(t.finish()[0]['status'], 'unknown')


if __name__ == '__main__': unittest.main()
