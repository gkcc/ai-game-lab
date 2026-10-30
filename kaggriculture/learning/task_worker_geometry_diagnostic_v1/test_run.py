"""Focused, synthetic coverage for geometry orchestration and privacy projection."""
from copy import deepcopy
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import unittest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('worker_geometry_run', HERE / 'run.py')
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def counts(groups=100, plant=40, competing=20, farther=0, equal=20):
    return {'groups': groups, 'teacher_plant_groups': plant,
            'same_target_unit_competition_groups': competing,
            'teacher_has_farther_alternative_groups': farther,
            'equal_distance_competition_groups': equal}


def sources(row):
    return {name: {'all_steps': deepcopy(row), 'encoded_step_2': deepcopy(row),
                   'competition_categories_may_overlap': True} for name in ('old', 'new')}


class RunnerTest(unittest.TestCase):
    def test_small_counts_and_zero_share_the_same_threshold(self):
        value = sources(counts(competing=20, farther=1, equal=20))
        result = runner.public_training_summary(value)
        self.assertEqual(result['old']['teacher_has_farther_alternative_groups'], 'below_20')
        self.assertEqual(result['old']['equal_distance_competition_groups'], 'at_least_20')
        self.assertNotIn('groups', result['old'])
        value = sources(counts(competing=20, farther=0, equal=20))
        self.assertEqual(runner.public_training_summary(value)['old']
                         ['teacher_has_farther_alternative_groups'], 'below_20')

    def test_thresholds_allow_overlapping_categories_without_publishing_counts(self):
        value = sources(counts(groups=100, plant=60, competing=40, farther=20, equal=20))
        result = runner.public_training_summary(value)
        self.assertEqual(result['old']['equal_distance_competition_groups'], 'at_least_20')
        self.assertNotIn('same_target_unit_competition_groups', result['old'])
        value = sources(counts(groups=100, plant=60, competing=40, farther=40, equal=40))
        self.assertEqual(runner.public_training_summary(value)['old']
                         ['teacher_has_farther_alternative_groups'], 'at_least_20')

    def test_time_and_overlap_differences_do_not_publish_numeric_counts(self):
        value = sources(counts(groups=200, plant=160, competing=120, farther=60, equal=60))
        for source in value.values():
            source['encoded_step_2'] = counts(groups=199, plant=159, competing=119, farther=59, equal=60)
        result = runner.public_training_summary(value)
        self.assertEqual(set(result), {'old', 'new'})
        self.assertTrue(all(isinstance(item, str) for row in result.values() for item in row.values()))

    def test_union_threshold_does_not_reveal_small_categories(self):
        value = sources(counts(groups=100, plant=40, competing=20, farther=1, equal=19))
        row = runner.public_training_summary(value)['old']
        self.assertEqual(set(row.values()), {'below_20'})
        self.assertNotIn('same_target_unit_competition_groups', row)

    def test_invalid_hierarchy_is_rejected(self):
        value = sources(counts(groups=100, plant=20, competing=40))
        with self.assertRaises(ValueError):
            runner.public_training_summary(value)

    def test_only_three_trace_turns_and_target_index_are_decoded(self):
        turns = [dict(step=i, load_errors=0, inference_errors=0, fallbacks=0,
                      records=[{'ordinal': j} for j in range(6)]) for i in range(3)]
        raw = gzip.compress((''.join(json.dumps(item) + '\n' for item in turns)).encode())
        trace, record = runner.decode_target(raw, len(raw), hashlib.sha256(raw).hexdigest())
        self.assertEqual((trace['step'], record['ordinal']), (2, 5))
        with self.assertRaises(ValueError):
            runner.decode_target(raw, len(raw), '0' * 64)
        turns[-1]['records'] = []
        raw = gzip.compress((''.join(json.dumps(item) + '\n' for item in turns)).encode())
        with self.assertRaises(ValueError):
            runner.decode_target(raw, len(raw), hashlib.sha256(raw).hexdigest())

    def test_invalid_cli_does_not_echo_private_argument(self):
        sentinel = 'PRIVATE_SENTINEL_ONLY_FOR_TEST'
        result = subprocess.run([sys.executable, '-B', '-I', str(HERE / 'run.py'), '--unknown', sentinel],
                                capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertNotIn(sentinel, result.stdout + result.stderr)

    def test_package_and_module_shadows_are_rejected(self):
        package = Path('synthetic') / 'learning' / 'example'
        runner.reject_shadows(package, ('module',), exists=lambda _path: False)
        for shadow in (package.parent / 'example.py', package / 'module.pyc',
                       package / 'module' / '__init__.py',
                       package / ('__init__' + runner.importlib.machinery.EXTENSION_SUFFIXES[0])):
            with self.subTest(shadow=shadow.name), self.assertRaises(ValueError):
                runner.reject_shadows(package, ('module',), exists=lambda path: path == shadow)

    def test_isolated_runtime_binding_works_without_reading_private_data(self):
        source = ("import importlib.util,sys; p=sys.argv[1]; "
                  "s=importlib.util.spec_from_file_location('geometry_entry_check',p); "
                  "m=importlib.util.module_from_spec(s); s.loader.exec_module(m); "
                  "m.bind_runtime(); print('ok')")
        result = subprocess.run([sys.executable, '-B', '-I', '-c', source, str(HERE / 'run.py')],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0)
        self.assertEqual(result.stdout.strip(), 'ok')


if __name__ == '__main__':
    unittest.main()
