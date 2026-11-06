"""Memory-only quantile regressions; no real source, weights or game engine."""
from copy import deepcopy
import unittest

from learning.task_harvest_model_dev_v1.policy import BoundedHarvestAgent
from learning.task_harvest_return_pairs_v1 import selection as frozen
from learning.task_harvest_return_pairs_v1.test_selection import (
    CROP_SHA, TASK_SHA, patch_synthetic, scenario)
from learning.task_harvest_quantile_pairs_v1.selection import select_quantiles


def source_rows(count, *, side=0):
    """Repeated constructed controller scenes, followed by empty metadata.

    Each active turn has one real synthetic new-H callback at ordinal one.
    These scenes test the source contract; they are not a natural trajectory.
    """
    actor = BoundedHarvestAgent('crop-wheat', CROP_SHA, 'task-harvest', TASK_SHA)
    rows = []
    for step in range(max(3, count + 2)):
        obs, cfg = scenario(step, side=side, active=1 <= step <= count)
        actor(obs, cfg)
        rows.append(deepcopy(actor.last))
    for step in range(len(rows), 719):
        row = deepcopy(rows[-1])
        row['step'], row['model_counts']['controller_calls'] = step, step + 1
        rows.append(row)
    return rows


class SelectionTests(unittest.TestCase):
    def setUp(self):
        patch_synthetic(self)

    def prior(self, rows, *, side=0):
        return frozen.select_target(rows, side=side, task_sha=TASK_SHA,
                                    crop_sha=CROP_SHA)['target']

    def select(self, rows, prior, *, side=0, guard=None):
        return select_quantiles(rows, side=side, task_sha=TASK_SHA,
                                crop_sha=CROP_SHA, old_target=prior, guard=guard)

    def test_small_and_registered_population_quantiles_keep_empty_slots(self):
        cases = ((0, [0, None, None, None]), (1, [0, 0, 0, 0]),
                 (2, [0, 0, 0, 1]), (3, [0, 0, 1, 2]), (4, [0, 1, 2, 3]),
                 (23, [0, 7, 14, 22]), (24, [0, 7, 15, 23]))
        for count, expected in cases:
            with self.subTest(count=count):
                rows = source_rows(count, side=1)
                prior = self.prior(rows, side=1)
                result = self.select(rows, prior, side=1)
                self.assertEqual(result['indices'], expected)
                self.assertEqual(result['eligible_count'], count)
                self.assertEqual(result['counts']['qualifying_callbacks'], count)
                self.assertEqual(result['counts']['trace_lines'], 719)
                self.assertEqual(result['reference'], prior)
                self.assertEqual([point['label'] for point in result['points']], ['Q1', 'Q2', 'Q3'])
                used = {0}
                for point, index in zip(result['points'], expected[1:]):
                    self.assertEqual(point['index'], index)
                    if index is None:
                        self.assertIsNone(point['target'])
                        self.assertEqual(point['reason'], 'no_eligible_target')
                    elif index in used:
                        self.assertIsNone(point['target'])
                        self.assertEqual(point['reason'], 'duplicate_index')
                    else:
                        used.add(index)
                        target = point['target']
                        self.assertIsNone(point['reason'])
                        self.assertEqual((target['step'], target['ordinal'], target['side']),
                                         (index + 1, 1, 1))
                        self.assertEqual(target['input_sha256'], frozen.ordered_hash(
                            rows[index + 1]['records'][1]['inputs']))
                        self.assertEqual(frozen.validate_target(target), target)

    def test_single_pass_consumes_tail_and_eof_without_mutating_old_module(self):
        rows = source_rows(4)
        prior = self.prior(rows)
        original = frozen.select_target
        globals_before = dict(vars(frozen))
        progress = dict(iterations=0, rows=0, eof=0, guards=0)

        class Once:
            def __iter__(self):
                progress['iterations'] += 1
                if progress['iterations'] != 1:
                    raise AssertionError('Source was scanned again')
                return self

            def __next__(self):
                index = progress['rows']
                if index == len(rows):
                    progress['eof'] += 1
                    raise StopIteration
                progress['rows'] += 1
                return rows[index]

        def guard():
            progress['guards'] += 1
            self.assertIs(frozen.select_target, original)
            self.assertIs(frozen.validate_target, globals_before['validate_target'])

        result = self.select(Once(), prior, guard=guard)
        self.assertEqual(result['eligible_count'], 4)
        self.assertEqual({key: progress[key] for key in ('iterations', 'rows', 'eof')},
                         dict(iterations=1, rows=719, eof=1))
        self.assertGreaterEqual(progress['guards'], 720)
        self.assertEqual(set(vars(frozen)), set(globals_before))
        self.assertTrue(all(vars(frozen)[key] is value for key, value in globals_before.items()))
        self.assertEqual(self.prior(rows), prior)

    def test_outcomes_and_teacher_labels_do_not_select_quantiles(self):
        rows = source_rows(4)
        prior = self.prior(rows)
        expected = self.select(rows, prior)
        for row in rows:
            row['terminal_margin'] = 999999 if row['step'] % 2 else -999999
            row['future_return'] = {'H': row['step'], 'W': -row['step']}
            for record in row['records']:
                record['teacher_label'] = 'not-a-selection-input'
        self.assertEqual(self.select(rows, prior), expected)

    def test_corrupt_tail_missing_extra_row_identity_and_cost_rejected(self):
        pristine = source_rows(4)
        prior = self.prior(pristine)
        validator = frozen.validate_target
        for defect in ('tail', 'missing', 'extra', 'identity', 'cost'):
            with self.subTest(defect=defect):
                rows = deepcopy(pristine)
                if defect == 'tail':
                    rows[-1]['step'] -= 1
                elif defect == 'missing':
                    rows.pop()
                elif defect == 'extra':
                    rows.append(deepcopy(rows[-1]))
                elif defect == 'identity':
                    rows[-1]['task_checkpoint_sha256'] = '0' * 64
                else:
                    rows[-1]['model_counts']['task_rows'] += 1
                with self.assertRaises(ValueError):
                    self.select(rows, prior)
                self.assertIs(frozen.validate_target, validator)

    def test_error_at_eof_is_not_hidden_by_earlier_selected_points(self):
        rows = source_rows(4)
        prior = self.prior(rows)

        def broken_stream():
            yield from rows
            raise OSError('Synthetic stream integrity failure at EOF')

        with self.assertRaises(OSError):
            self.select(broken_stream(), prior)

    def test_prior_identity_mismatch_or_missing_prior_is_rejected(self):
        rows = source_rows(4)
        prior = self.prior(rows)
        for altered in (None, dict(prior, input_sha256='0' * 64),
                        dict(prior, ordinal=prior['ordinal'] + 1)):
            with self.subTest(prior=altered is None):
                with self.assertRaises(ValueError):
                    self.select(rows, altered)
        with self.assertRaises(ValueError):
            self.select(source_rows(0), prior)
        # Existing core compares target fields individually, allowing JSON
        # serializers to change only the outer object's key order.
        reordered = dict(reversed(list(prior.items())))
        self.assertEqual(self.select(rows, reordered), self.select(rows, prior))

    def test_full_candidate_and_final_reservation_validation_is_preserved(self):
        pristine = source_rows(4)
        prior = self.prior(pristine)
        for defect in ('water', 'private', 'request'):
            with self.subTest(defect=defect):
                rows = deepcopy(pristine)
                record = rows[4]['records'][1]
                if defect == 'water':
                    index = next(i for i, candidate in enumerate(record['inputs']['candidates'])
                                 if candidate['id'] == 'u1:x3:y4:WATER')
                    record['inputs']['candidates'].pop(index)
                    record['task_model']['logits'].pop(index)
                    record['task_model']['argmax_index'] -= 1
                elif defect == 'private':
                    record['inputs']['observation']['farms'][1]['private'] = {'future_cash': 1}
                else:
                    rows[4]['final_reservations']['unit_actions'][1] = ['WATER']
                with self.assertRaises(ValueError):
                    self.select(rows, prior)

    def test_guard_failure_propagates_without_patching_frozen_selector(self):
        rows = source_rows(4)
        prior = self.prior(rows)
        validator = frozen.validate_target
        calls = 0

        def guard():
            nonlocal calls
            calls += 1
            if calls == 30:
                raise RuntimeError('Synthetic resource stop')

        with self.assertRaises(RuntimeError):
            self.select(rows, prior, guard=guard)
        self.assertIs(frozen.validate_target, validator)
        self.assertEqual(self.prior(rows), prior)


if __name__ == '__main__':
    unittest.main()
