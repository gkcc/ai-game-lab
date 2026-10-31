"""Synthetic aggregation and gate tests; no private data, model or GPU."""
from types import SimpleNamespace
from pathlib import Path
import io
import unittest
from unittest.mock import patch

import numpy as np

from learning.task_equiv_bc_v1 import experiment, gpu_worker, probe


def group(eligible=True):
    return dict(status='ok', features=np.zeros((3, 373), np.float32), teacher_index=0,
        positive_mask=np.array([True, eligible, False], bool),
        verified_mask=np.array([True, eligible, eligible], bool), weight=1.0,
        positive_ids=['teacher', 'same'] if eligible else ['teacher'],
        different_request_ids=['different'] if eligible else [], unknown_ids=[],
        alternatives=[{}, {}] if eligible else [], reference=object())


class Network:
    def __init__(self, choice):
        self.choice, self.last = choice, None

    def __call__(self, _inputs):
        self.last = {'logits': [2.0, 1.0, 0.0]}
        return self.choice


class Probe:
    def evaluate_choice(self, _obs, _cfg, _ordinal, *, chosen_id, **_kwargs):
        return {'outcome': 'unknown' if chosen_id == 'unknown' else
                           'different' if chosen_id == 'different' else 'match',
                'controller_called': chosen_id != 'teacher'}


class ExperimentTests(unittest.TestCase):
    def test_training_payload_round_trip_and_role_gates(self):
        groups = [group(i in (0, 1, 24, 25)) for i in range(48)]
        roles = [0] * 24 + [1] * 24
        counts = experiment._summarize(groups, roles)
        self.assertEqual((counts['eligible'], counts['eligible_per_role']), (4, [2, 2, 0]))
        decoded, eligible = gpu_worker.unpack_groups(experiment._pack_training(groups, roles))
        self.assertEqual((len(decoded), eligible), (48, [2, 2]))
        self.assertFalse(decoded[0]['positive_mask'][2])

    def test_development_gate_and_unknown_are_distinct(self):
        sample = {'inputs': {'observation': {}, 'configuration': {}},
                  'label': {'teacher_label': 'teacher'}, 'teacher_prefix': (), 'decision': 0}
        cases = [(sample, group()) for _ in range(24)]
        networks = {'baseline': Network('teacher'), 'control': Network('different'),
                    'experiment': Network('teacher')}
        stats, screen = experiment._evaluate(cases, networks, {}, Probe(), lambda: True)
        self.assertEqual(screen, 'passed')
        self.assertEqual(stats['control']['different'], 24)
        self.assertEqual(stats['experiment']['match'], 24)
        networks['baseline'] = Network('unknown')
        stats, screen = experiment._evaluate(cases, networks, {}, Probe(), lambda: True)
        self.assertEqual(screen, 'unknown')
        self.assertEqual(stats['baseline']['unknown'], 24)

    def test_budget_and_relative_path_reject_invalid_inputs(self):
        clock = {'time': 0, 'memory': 5 * 1024**3}
        budget = experiment.Budget(lambda: clock['memory'], lambda: 5 * 1024**3,
                                   clock=lambda: clock['time'])
        budget.start = budget.phase_start = 0
        self.assertTrue(budget.initial())
        clock['memory'] = 5 * 1024**3 - 1
        with self.assertRaises(MemoryError): budget.initial()
        clock['memory'] = 4 * 1024**3 - 1
        with self.assertRaises(MemoryError): budget.check()
        clock.update(memory=5 * 1024**3, time=2400)
        with self.assertRaises(TimeoutError): budget.check()
        clock['time'] = 4199
        budget.phase(600)
        self.assertTrue(budget.check())
        clock['time'] = 4200
        with self.assertRaises(TimeoutError): budget.check()
        disk_budget = experiment.Budget(lambda: 6 * 1024**3, lambda: 4 * 1024**3 - 1,
                                        clock=lambda: 0)
        disk_budget.start = disk_budget.phase_start = 0
        with self.assertRaises(experiment.DiskFloorError) as error: disk_budget.initial()
        self.assertEqual(experiment._category(error.exception), 'disk_gate')
        for path in ('../x', '/absolute', 'a\\b', 'a/../b'):
            with self.subTest(path=path), self.assertRaises(ValueError): experiment._safe_relative(path)

    def test_wrapped_probe_guard_preserves_stop_category(self):
        for error, expected in ((MemoryError(), 'memory_gate'), (TimeoutError(), 'time_gate')):
            def stop():
                raise error
            with self.subTest(expected=expected):
                try:
                    probe._guard(stop)
                except Exception as exc:
                    self.assertEqual(experiment._category(exc), expected)
                else:
                    self.fail('Guard did not stop')

    def test_gpu_failure_projects_only_allowlisted_fields(self):
        raw = (b'{"phase":"device","status":"failed","category":"memory",'
               b'"private":"do-not-publish"}\n')
        with patch.object(Path, 'is_file', return_value=True), \
             patch.object(Path, 'stat', return_value=SimpleNamespace(st_size=len(raw))), \
             patch.object(Path, 'read_bytes', return_value=raw):
            result = experiment._gpu_failure(Path('synthetic'))
        self.assertEqual(result, {'phase': 'device', 'category': 'memory'})

    def test_interrupted_unconfirmed_cleanup_remains_unconfirmed(self):
        class Child:
            def __init__(self): self.waits = 0
            def wait(self, timeout):
                self.waits += 1
                if self.waits == 1: raise KeyboardInterrupt()
                return 0
            def poll(self): return 0 if self.waits >= 2 else None
        out, err = io.BytesIO(), io.BytesIO()
        child = Child()
        host = SimpleNamespace(child=child, stdout_stream=out, stderr_stream=err)
        with self.assertRaises(experiment.CleanupUnconfirmed) as failure:
            experiment._wait_unconfirmed(host)
        self.assertEqual(experiment._category(failure.exception), 'gpu_cleanup_unconfirmed')
        self.assertEqual(child.waits, 2)
        self.assertTrue(out.closed and err.closed)
        class Exited:
            def wait(self, timeout): return 0
            def poll(self): return 0
        out, err = io.BytesIO(), io.BytesIO()
        with self.assertRaises(experiment.CleanupUnconfirmed):
            experiment._wait_unconfirmed(SimpleNamespace(child=Exited(), stdout_stream=out, stderr_stream=err))
        self.assertTrue(out.closed and err.closed)


if __name__ == '__main__':
    unittest.main()
