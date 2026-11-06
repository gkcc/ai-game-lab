"""Memory-only source rows and synthetic weights; no natural-game claim."""
from copy import deepcopy
import unittest
from unittest.mock import patch

import numpy as np

from learning.task_harvest_choice_v1.fixtures import observation
from learning.task_harvest_model_dev_v1.policy import BoundedHarvestAgent
from learning.task_harvest_model_dev_v1.test_policy import (
    CROP_SHA, TASK_SHA, MemoryCropNetwork, MemoryTaskNetwork)
from learning.task_harvest_return_pairs_v1 import selection


class PlanningCropNetwork(MemoryCropNetwork):
    def logits(self, x, names):
        self.calls += 1
        # A controlled synthetic head makes downstream reservation propagation
        # observable: a prior WATER chooses CARROT, otherwise WHEAT.
        selected = 1 if x[list(names).index('planning.prior_actions.WATER')] else 0
        result = np.zeros(5)
        result[selected] = 1.
        return result


def patch_synthetic(case):
    for name, replacement in (
        ('learning.task_harvest_model_dev_v1.policy.TaskNetwork', MemoryTaskNetwork),
        ('learning.bc_crop.agent.Network', PlanningCropNetwork),
    ):
        monkey = patch(name, replacement)
        monkey.start()
        case.addCleanup(monkey.stop)
    forbidden = patch('numpy.load', side_effect=AssertionError('Synthetic tests cannot read weights'))
    forbidden.start()
    case.addCleanup(forbidden.stop)


def scenario(step, *, side=0, active=False, new_harvest=True, downstream=True):
    obs, cfg = observation('mature5dry', side=side)
    obs['step'], obs['day'], obs['hour'] = step, step // 24, step % 24
    for farm in obs['farms']:
        for row in farm['tiles']:
            for tile in row:
                if isinstance(tile, dict):
                    tile.update(crop='WHEAT', planted_day=0, watered_today=True, yield_units=1)
    if active:
        own = obs['farms'][side]
        own['money'], own['hands'] = 3000, [[3, 4]] + ([[2, 4]] if downstream else [])
        obs['private']['inventories'] = [{} for _ in range(1 + len(own['hands']))]
        for x, watered in ((4, True), (3, not new_harvest)):
            own['tiles'][4][x].update(crop='MELON', planted_day=-10,
                watered_today=watered, yield_units=5, consecutive_unwatered=0 if watered else 1)
        if downstream:
            own['tiles'][4][2] = None
    return obs, cfg


def source_rows(*, side=0, new_harvest=True, downstream=True):
    """Three actual synthetic controller turns, then zero-callback metadata.

    The extension appears at ordinal 1, after an ordinary old HARVEST. The row
    suffix tests EOF/counter parsing; it is not an official state trajectory.
    """
    actor = BoundedHarvestAgent('crop-wheat', CROP_SHA, 'task-harvest', TASK_SHA)
    rows, requests = [], []
    for step in range(3):
        obs, cfg = scenario(step, side=side, active=step == 1, new_harvest=new_harvest, downstream=downstream)
        requests.append(actor(obs, cfg))
        rows.append(deepcopy(actor.last))
    for step in range(3, 719):
        row = deepcopy(rows[2])
        row['step'], row['model_counts']['controller_calls'] = step, step + 1
        rows.append(row)
    return rows, requests


class SelectionTests(unittest.TestCase):
    def setUp(self):
        patch_synthetic(self)

    def select(self, rows, *, side=0):
        return selection.select_target(rows, side=side, task_sha=TASK_SHA, crop_sha=CROP_SHA)

    def test_nonzero_ordinal_real_new_choice_and_complete_eof(self):
        rows, _ = source_rows()
        progress = dict(rows=0, eof=False)
        def stream():
            for row in rows:
                progress['rows'] += 1
                yield row
            progress['eof'] = True
        result = self.select(stream())
        target = result['target']
        self.assertEqual((target['step'], target['ordinal'], target['unit'], target['target']), (1, 1, 1, [3, 4]))
        self.assertEqual(target['harvest_choice'], 'u1:x3:y4:HARVEST')
        self.assertEqual(target['water_choice'], 'u1:x3:y4:WATER')
        self.assertEqual(target['unit_request'], ['HARVEST'])
        self.assertEqual(target['input_sha256'], selection.ordered_hash(rows[1]['records'][1]['inputs']))
        self.assertEqual(progress, dict(rows=719, eof=True))
        self.assertEqual(result['counts']['trace_lines'], 719)
        self.assertEqual(result['counts']['qualifying_callbacks'], 1)
        self.assertIsNone(result['reason'])
        self.assertEqual(selection.validate_target(target), target)

    def test_no_actual_new_harvest_returns_none_without_replacement(self):
        rows, _ = source_rows(new_harvest=False)
        result = self.select(rows)
        self.assertIsNone(result['target'])
        self.assertEqual(result['counts']['trace_lines'], 719)
        self.assertEqual(result['counts']['qualifying_callbacks'], 0)
        self.assertIsNotNone(result['reason'])

    def test_corrupt_tail_missing_or_extra_row_rejected_after_first_hit(self):
        for defect in ('tail', 'missing', 'extra', 'identity'):
            with self.subTest(defect=defect):
                rows, _ = source_rows()
                if defect == 'tail':
                    rows[-1]['step'] = 716
                elif defect == 'missing':
                    rows.pop()
                elif defect == 'extra':
                    rows.append(deepcopy(rows[-1]))
                else:
                    rows[-1]['task_checkpoint_sha256'] = '0' * 64
                with self.assertRaises(ValueError):
                    self.select(rows)

    def test_private_fields_and_missing_paired_water_are_rejected(self):
        for defect in ('private', 'water'):
            with self.subTest(defect=defect):
                rows, _ = source_rows()
                if defect == 'private':
                    rows[1]['records'][0]['inputs']['observation']['farms'][1]['private'] = {'future_cash': 100}
                else:
                    record = rows[1]['records'][1]
                    index = next(i for i, c in enumerate(record['inputs']['candidates']) if c['id'] == 'u1:x3:y4:WATER')
                    record['inputs']['candidates'].pop(index)
                    record['task_model']['logits'].pop(index)
                    record['task_model']['argmax_index'] -= 1
                with self.assertRaises(ValueError):
                    self.select(rows)

    def test_future_offline_outcomes_and_teacher_labels_do_not_choose_point(self):
        rows, _ = source_rows()
        expected = self.select(rows)
        for row in rows:
            row['future_margin'] = -999999 if row['step'] % 2 else 999999
            for record in row['records']:
                record['teacher_label'] = 'unused-for-selection'
        self.assertEqual(self.select(rows), expected)

    def test_target_bounds_sha_seat_and_final_request_are_checked(self):
        rows, _ = source_rows(side=1)
        target = self.select(rows, side=1)['target']
        self.assertEqual(target['side'], 1)
        for key, value in (('ordinal', 32), ('unit', target['unit_count']), ('target', [10, 4]),
                           ('side', True), ('task_checkpoint_sha256', 'bad'), ('unit_request', ['WATER'])):
            with self.subTest(field=key):
                altered = dict(target, **{key: value})
                with self.assertRaises(ValueError):
                    selection.validate_target(altered)
        broken = deepcopy(rows)
        broken[1]['final_reservations']['unit_actions'][1] = ['WATER']
        with self.assertRaises(ValueError):
            self.select(broken, side=1)


if __name__ == '__main__':
    unittest.main()
