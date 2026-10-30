"""Small synthetic regressions for the registered comparison entrypoint."""
import importlib.util
from pathlib import Path
import subprocess
import sys
import types
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import psutil

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from learning.task_bc_v1 import model
from learning.task_teacher_choice_bc_v1 import numeric
from learning.task_worker_geometry_diagnostic_v1 import archive_io
from learning.task_worker_ranking_v1 import test_core as synthetic


spec = importlib.util.spec_from_file_location('task_worker_ranking_entry', Path(__file__).with_name('run.py'))
entry = importlib.util.module_from_spec(spec)
spec.loader.exec_module(entry)


class RunnerTests(unittest.TestCase):
    def _run_synthetic_archives(self, sizes, split):
        """Exercise the real merge and ranking core using only synthetic groups."""
        data = synthetic.fixture(sizes)

        def subset(first, last):
            start, end = int(data['offsets'][first]), int(data['offsets'][last])
            return dict(shared=data['shared'][first:last].copy(),
                        candidates=data['candidates'][start:end].copy(),
                        offsets=data['offsets'][first:last + 1].copy() - start,
                        teacher_choice_label=data['teacher_choice_label'][first:last].copy(),
                        ops=data['ops'][start:end].copy(),
                        seed=data['cluster_ids'][first:last].copy() + np.uint32(10))

        groups, rows = len(data['shared']), len(data['candidates'])
        old, new = subset(0, split), subset(split, groups)
        config = dict(old_train_seeds=[10, 11], new_train_seeds=[12, 13],
                      expected_groups=dict(old=len(old['shared']), new=len(new['shared'])),
                      expected_candidates=dict(old=len(old['candidates']), new=len(new['candidates'])),
                      data_limits=dict(max_groups=groups, max_candidates=rows,
                                       max_archive_bytes=1, max_uncompressed_bytes=1),
                      archive_sha256=dict(old='c' * 64, new='d' * 64))
        registration = dict(groups_expected=groups, candidates_expected=rows,
                            max_model_calls=2 * groups, max_scored_rows=2 * rows,
                            sampled_deadline_seconds=300)
        schema = dict(shared_feature_names=data['shared_feature_names'],
                      feature_names=data['feature_names'])
        early_roles = dict(train_seeds=list(range(8)), validation_seeds=[8, 9])
        early_export = dict(checkpoint_file='synthetic.npz', checkpoint_sha256='a' * 64)
        panel = dict(merged_checkpoint='synthetic.npz', merged_checkpoint_sha256='b' * 64)
        seal = dict(checkpoint_sha256='b' * 64)

        def scorer_for(_path, identity):
            # A picks the teacher in both scopes; B picks a different task.
            scores = [2, 1, 0] if identity == 'a' * 64 else [0, 1, 2]
            return synthetic.FakeScorer(lambda _shared: scores)

        with patch.object(psutil, 'virtual_memory', return_value=types.SimpleNamespace(available=8 * 1024**3)), \
             patch.object(archive_io, 'registered_inputs', return_value=(config, schema, dict(old='old', new='new'))), \
             patch.object(numeric, 'read', side_effect=[old, new]), \
             patch.object(entry, 'read_json', side_effect=[early_roles, early_roles, early_export, panel, seal]), \
             patch.object(entry, 'local_file', side_effect=[Path('synthetic-a'), Path('synthetic-b')]), \
             patch.object(entry, 'sha', side_effect=lambda path: 'a' * 64 if path.name == 'synthetic-a' else 'b' * 64), \
             patch.object(entry, 'check_import_origin'), \
             patch.object(model, 'TaskNetwork', side_effect=scorer_for) as factory:
            result = entry.run(Path.cwd(), registration)
            return result, factory.call_args_list

    def test_in_memory_insufficient_coverage_does_not_load_either_head(self):
        result, calls = self._run_synthetic_archives((4, 5, 5, 5), 9)
        self.assertEqual(result['coverage'], 'insufficient')
        self.assertIs(result['models_loaded'], False)
        self.assertIs(result['model_inference_executed'], False)
        self.assertEqual(calls, [])

    def test_in_memory_complete_workflow_preserves_head_order_and_direction(self):
        result, calls = self._run_synthetic_archives((5, 5, 5, 5), 10)
        self.assertEqual(result['coverage'], 'sufficient')
        self.assertEqual([call.args[1] for call in calls], ['a' * 64, 'b' * 64])
        self.assertEqual((result['model_a'], result['model_b']), ('early_bc', 'merged_bc'))
        self.assertEqual(result['conditional_teacher_top1']['model_a_minus_model_b_band'], '>+20pp')
        self.assertEqual(result['global_guard'], 'pass')

    def test_source_roles_reject_overlap_and_invalid_types(self):
        early = {'train_seeds': list(range(8)), 'validation_seeds': [8, 9]}
        good = {'old_train_seeds': [10, 11], 'new_train_seeds': [12, 13]}
        self.assertEqual(tuple(map(len, entry.source_roles(early, good))), (2, 2, 10))
        for bad in ({'old_train_seeds': [10, 11], 'new_train_seeds': [11, 13]},
                    {'old_train_seeds': [10, 11], 'new_train_seeds': [9, 13]},
                    {'old_train_seeds': [True, 11], 'new_train_seeds': [12, 13]}):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                entry.source_roles(early, bad)

    def test_early_model_must_bind_same_demonstration_roles(self):
        roles = {'train_seeds': [1, 2], 'validation_seeds': [3]}
        entry.require_early_training_match(roles, dict(roles))
        with self.assertRaises(ValueError):
            entry.require_early_training_match(roles, {'train_seeds': [1, 3], 'validation_seeds': [2]})

    def test_artifacts_directory_must_be_directly_inside_game_root(self):
        root = Path('synthetic-game-root')
        with patch.object(Path, 'is_dir', return_value=True):
            entry.require_direct_artifacts(root, root / 'artifacts')
            with self.assertRaises(ValueError):
                entry.require_direct_artifacts(root, root.parent / 'synthetic-external')

    def test_registration_requires_all_bound_sources_and_fixed_budgets(self):
        doc = {'schema': 'task-worker-ranking-registration-v1',
               'source_sha256': {name: 'a' * 64 for name in entry.SOURCE_PATHS},
               'groups_expected': 5104, 'candidates_expected': 322151,
               'max_model_calls': 10208, 'max_scored_rows': 644302,
               'sampled_deadline_seconds': 300}
        with patch.object(entry, 'local_file', return_value=object()), patch.object(entry, 'sha', return_value='a' * 64):
            entry.verify_sources(doc)
            missing = dict(doc, source_sha256=dict(doc['source_sha256']))
            missing['source_sha256'].pop(next(iter(entry.SOURCE_PATHS)))
            with self.assertRaises(ValueError):
                entry.verify_sources(missing)
            with self.assertRaises(ValueError):
                entry.verify_sources(dict(doc, max_model_calls=10209))

    def test_registration_must_match_head_and_committed_bytes(self):
        class RegistrationPath:
            def relative_to(self, _):
                return Path('registration.json')

            def read_bytes(self):
                return b'{}\r\n'

        commit = 'a' * 40
        with patch.object(entry, 'REGISTRATION', RegistrationPath()):
            with patch.object(entry.subprocess, 'check_output', side_effect=[commit.encode(), b'{}\n']):
                entry.verify_registration_commit(commit)
            with patch.object(entry.subprocess, 'check_output', side_effect=[('b' * 40).encode()]):
                with self.assertRaises(ValueError):
                    entry.verify_registration_commit(commit)
            with patch.object(entry.subprocess, 'check_output', side_effect=[commit.encode(), b'changed']):
                with self.assertRaises(ValueError):
                    entry.verify_registration_commit(commit)

    def test_shared_score_budget_limits_both_models(self):
        class SyntheticScorer:
            metadata = {}

            def score(self, shared, candidates):
                return [0.0] * len(candidates)

        config = {'max_model_calls': 2, 'max_scored_rows': 5, 'sampled_deadline_seconds': 300}
        budget = entry.ScoreBudget(config, entry.time.monotonic())
        first, second = budget.wrapped(SyntheticScorer()), budget.wrapped(SyntheticScorer())
        self.assertEqual(first.score([], [1, 2]), [0.0, 0.0])
        self.assertEqual(second.score([], [1, 2, 3]), [0.0, 0.0, 0.0])
        with self.assertRaises(ValueError):
            first.score([], [1])
        self.assertEqual(budget.failure, 'scoring_budget')

    def test_sampled_deadline_reports_reason(self):
        config = {'max_model_calls': 1, 'max_scored_rows': 1, 'sampled_deadline_seconds': 1}
        budget = entry.ScoreBudget(config, 0)
        with patch.object(entry.time, 'monotonic', return_value=1):
            with self.assertRaises(TimeoutError):
                budget.check()
        self.assertEqual(budget.failure, 'deadline')

    def test_same_named_package_and_extension_shadows_are_rejected(self):
        package = Path(__file__).resolve().parent
        entry.reject_shadows(package, ('core',), exists=lambda path: False)
        shadows = [package / 'core' / '__init__.py',
                   package / ('core' + entry.importlib.machinery.EXTENSION_SUFFIXES[0]),
                   package.parent / (package.name + '.py')]
        for shadow in shadows:
            with self.subTest(shadow=shadow.name), self.assertRaises(ValueError):
                entry.reject_shadows(package, ('core',), exists=lambda path: path == shadow)

    def test_runtime_requires_isolation_before_imports(self):
        flags = types.SimpleNamespace(isolated=0, dont_write_bytecode=1)
        with patch.object(entry.sys, 'flags', flags), self.assertRaises(ValueError):
            entry.bind_runtime({})

    def test_invalid_cli_never_echoes_the_synthetic_argument(self):
        sentinel = 'synthetic-sensitive-argument-marker'
        command = [sys.executable, '-B', '-I', str(Path(__file__).with_name('run.py')),
                   '--game-root', 'unused', '--output-dir', 'unused',
                   '--registration-commit', 'a' * 40, '--unexpected', sentinel]
        result = subprocess.run(command, capture_output=True, text=True, timeout=30)
        self.assertNotEqual(result.returncode, 0)
        self.assertNotIn(sentinel, result.stdout + result.stderr)
        self.assertNotIn('--unexpected', result.stdout + result.stderr)

    def test_failed_git_lookup_never_echoes_synthetic_directory(self):
        sentinel = 'synthetic-sensitive-directory-marker'
        program = '''import importlib.util
from pathlib import Path
spec = importlib.util.spec_from_file_location('entry', %r)
entry = importlib.util.module_from_spec(spec)
spec.loader.exec_module(entry)
entry.REPO = Path(%r)
try:
    entry.verify_registration_commit('a' * 40)
except Exception:
    print('failed safely')
''' % (str(Path(__file__).with_name('run.py')), sentinel)
        result = subprocess.run([sys.executable, '-B', '-I', '-c', program],
                                capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0)
        self.assertIn('failed safely', result.stdout)
        self.assertNotIn(sentinel, result.stdout + result.stderr)

    def test_corrupt_pending_result_never_becomes_final(self):
        pending, final, output, writer = MagicMock(), MagicMock(), MagicMock(), MagicMock()
        output.__truediv__.side_effect = lambda name: pending if name == 'result.pending' else final
        pending.open.return_value.__enter__.return_value = writer
        pending.read_bytes.return_value = b'corrupted'
        final.exists.return_value = False
        with patch.object(entry.os, 'fsync'), self.assertRaises(ValueError):
            entry.publish_result(output, {'coverage': 'insufficient'})
        pending.rename.assert_not_called()

        pending.reset_mock()
        pending.open.return_value.__enter__.return_value = writer
        pending.read_bytes.side_effect = lambda: writer.write.call_args.args[0]
        with patch.object(entry.os, 'fsync'):
            entry.publish_result(output, {'coverage': 'insufficient'})
        pending.rename.assert_called_once_with(final)


if __name__ == '__main__':
    unittest.main()
