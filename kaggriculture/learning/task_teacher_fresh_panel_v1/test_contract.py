"""Small regressions for timing, teacher identity and paired reporting."""
import argparse
from copy import deepcopy
import json
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from . import contract as c


class Clock:
    def __init__(self):
        self.value = 0.

    def __call__(self):
        return self.value

    def sleep(self, duration):
        self.value += duration


def panel_rows():
    teachers, old = [], []
    for cohort, aliases in c.COHORTS.items():
        for alias, role in aliases.items():
            for opponent in c.OPPONENTS:
                for side in c.SIDES:
                    teachers.append(dict(cohort=cohort, row=dict(seed_alias=alias,
                        opponent=opponent, side=side, net_profit=120,
                        opponent_net_profit=90, margin=30, score=1.)))
                    old.append(dict(seed_alias=role, opponent=opponent, side=side,
                        baseline=dict(own_net_profit=100, opponent_net_profit=90,
                                      margin=10, score=1.),
                        control=dict(own_net_profit=130, opponent_net_profit=90,
                                     margin=40, score=1.)))
    return teachers, dict(pairs=old)


class ContractTests(unittest.TestCase):
    def test_two_opponents_both_seats_and_no_task_checkpoint(self):
        plans = c.plans(SimpleNamespace(seeds=(11, 29)))
        self.assertEqual(len(plans), 8)
        self.assertEqual({(p['seed'], p['opponent'], p['side']) for p in plans},
                         {(s, o, p) for s in (11, 29) for o in c.OPPONENTS for p in (0, 1)})
        for plan in plans:
            job = dict(plan, bot_sha256='a'*64, rival_sha256='b'*64,
                       task_checkpoint_sha256=None)
            self.assertEqual(c.validate_plan(job, plan, job=True), job)
            job['task_checkpoint_sha256'] = 'c'*64
            with self.assertRaises(ValueError):
                c.validate_plan(job, plan, job=True)

    def test_memory_dip_resets_consecutive_samples(self):
        clock = Clock()
        readings = iter((10, 5, 10, 10, 10))
        result = c.await_memory(lambda: next(readings), clock=clock, sleep=clock.sleep,
                                floor=6, samples=3, interval=10, timeout=60)
        self.assertEqual(result['consecutive_samples'], 3)
        self.assertEqual(result['elapsed_seconds'], 40.)

    def test_memory_reader_cannot_succeed_after_deadline(self):
        clock, calls = Clock(), []
        def read():
            calls.append(True)
            if len(calls) == 3:
                clock.value += 21
            return 10
        with self.assertRaises(TimeoutError) as raised:
            c.await_memory(read, clock=clock, sleep=clock.sleep,
                           floor=6, samples=3, interval=10, timeout=30)
        self.assertEqual(clock.value, 41.)
        self.assertEqual(raised.exception.last_available_bytes, 10)

    def test_memory_reader_time_is_included_in_receipt(self):
        clock = Clock()
        def read():
            clock.value += 2
            return 10
        result = c.await_memory(read, clock=clock, sleep=clock.sleep,
                                floor=6, samples=3, interval=10, timeout=30)
        self.assertEqual(result['elapsed_seconds'], 26.)

    def test_resource_arguments_must_be_finite(self):
        for kwargs in ({'timeout': float('inf')}, {'interval': float('nan')}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                c.await_memory(lambda: 10, **kwargs)

    def test_argv_is_hidden_after_parse_and_restored_on_error(self):
        module = SimpleNamespace(argparse=argparse, __file__='entry.py')
        original = ['entry.py', '--seed', '17']
        with patch.object(sys, 'argv', original):
            with self.assertRaisesRegex(RuntimeError, 'probe'):
                with c.hide_parsed_argv(module):
                    parser = module.argparse.ArgumentParser()
                    parser.add_argument('--seed', type=int)
                    self.assertEqual(parser.parse_args().seed, 17)
                    self.assertEqual(sys.argv, ['entry.py'])
                    raise RuntimeError('probe')
            self.assertIs(sys.argv, original)
            self.assertIs(module.argparse, argparse)

    def test_teacher_adapter_restores_on_error(self):
        original = lambda *a: None
        module = SimpleNamespace(validate_plan=original)
        with self.assertRaises(RuntimeError):
            with c.teacher_adapter(module):
                self.assertIs(module.validate_plan, c.validate_plan)
                raise RuntimeError('probe')
        self.assertIs(module.validate_plan, original)

    def test_paired_means_include_absolute_results_and_shared_teacher(self):
        rows, old = panel_rows()
        rows[0]['row'].update(net_profit=80, margin=-10, score=0.)
        private, public = c.projection(rows, old, 'a'*40)
        self.assertEqual(public['seed_clusters'], 4)
        self.assertEqual(public['games'], 16)
        self.assertEqual(public['old_student_games'], 32)
        self.assertEqual(public['pairs_per_comparison'], 16)
        b, t = public['overall']['baseline'], public['overall']['control']
        self.assertEqual(b['mean_teacher']['own_net_profit'], 117.5)
        self.assertEqual(b['mean_student']['own_net_profit'], 100.)
        self.assertEqual(b['mean_teacher_minus_student']['own_net_profit'], 17.5)
        self.assertEqual(t['mean_teacher_minus_student']['own_net_profit'], -12.5)
        self.assertEqual(b['teacher_wins'], 15)
        self.assertEqual(b['teacher_losses'], 1)
        self.assertEqual(t['mean_teacher'], b['mean_teacher'])
        self.assertEqual(public['by_opponent']['market']['baseline']['mean_teacher']['own_net_profit'], 115.)
        self.assertEqual(private['by_seed']['F01']['baseline']['mean_teacher']['own_net_profit'], 110.)

    def test_projection_rejects_incomplete_duplicate_or_wrong_seat(self):
        rows, old = panel_rows()
        duplicate = deepcopy(rows)
        duplicate[-1] = deepcopy(duplicate[0])
        boolean_side = deepcopy(rows)
        boolean_side[0]['row']['side'] = False
        for bad in (rows[:-1], duplicate, boolean_side):
            with self.subTest(rows=len(bad)), self.assertRaises(ValueError):
                c.projection(bad, old, 'a'*40)

    def test_projection_rejects_score_inconsistent_with_cash_margin(self):
        rows, old = panel_rows()
        old['pairs'][0]['baseline']['score'] = 0.
        with self.assertRaises(ValueError):
            c.projection(rows, old, 'a'*40)

    def test_public_projection_does_not_copy_private_input_fields(self):
        rows, old = panel_rows()
        rows[0]['row'].update(seed=987654321, model_digest='private-model-marker',
                              source_path='private-path-marker')
        _private, public = c.projection(rows, old, 'a'*40)
        rendered = json.dumps(public)
        for marker in ('987654321', 'private-model-marker', 'private-path-marker', 'F01'):
            self.assertNotIn(marker, rendered)
        self.assertNotIn('pairs', public)


if __name__ == '__main__':
    unittest.main()
