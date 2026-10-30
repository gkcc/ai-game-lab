"""Synthetic-only tests for the independent registration entry."""
import contextlib
from copy import deepcopy
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import types
import unittest
from unittest.mock import MagicMock, patch


spec = importlib.util.spec_from_file_location('task_worker_ranking_v2_test_entry',
                                               Path(__file__).with_name('run.py'))
entry = importlib.util.module_from_spec(spec)
spec.loader.exec_module(entry)


class RegistrationPath:
    def __init__(self, content):
        self.content = content

    def relative_to(self, root):
        return Path('registration.json')

    def open(self, mode):
        if mode != 'rb':
            raise AssertionError('Unexpected open mode')
        return io.BytesIO(self.content)

    def read_bytes(self):
        return self.content


class RegistrationTests(unittest.TestCase):
    def _verify(self, worktree, committed, *, head=None):
        commit = 'a' * 40
        answer = [(head or commit).encode(), committed]
        with patch.object(entry, 'REGISTRATION', RegistrationPath(worktree)), \
             patch.object(entry.subprocess, 'check_output', side_effect=answer):
            return entry.verify_registration_commit(commit)

    def test_exact_lf_and_exact_crlf_both_pass(self):
        for raw in (b'{"x":1}\n', b'{"x":1}\r\n'):
            with self.subTest(style=len(raw)):
                self.assertEqual(self._verify(raw, raw), raw)

    def test_any_different_bytes_or_head_fail(self):
        for worktree, committed in ((b'{}\r\n', b'{}\n'), (b'{}\n', b'{}\r\n'),
                                    (b'{"x":1}\n', b'{"x":2}\n')):
            with self.subTest(worktree=worktree, committed=committed), self.assertRaises(ValueError):
                self._verify(worktree, committed)
        with self.assertRaises(ValueError):
            self._verify(b'{}\n', b'{}\n', head='b' * 40)

    def test_frozen_v1_rejects_equal_crlf_bytes_but_v2_accepts(self):
        frozen = entry.load_frozen_v1()
        raw, commit = b'{}\r\n', 'a' * 40
        with patch.object(frozen, 'REGISTRATION', RegistrationPath(raw)), \
             patch.object(frozen.subprocess, 'check_output', side_effect=[commit.encode(), raw]):
            with self.assertRaises(ValueError):
                frozen.verify_registration_commit(commit)
        self.assertEqual(self._verify(raw, raw), raw)

    def test_registration_too_large_fails_and_commit_format_is_strict(self):
        with self.assertRaises(ValueError):
            self._verify(b'x' * (entry.REGISTRATION_LIMIT + 1), b'x')
        with patch.object(entry.subprocess, 'check_output') as git:
            with self.assertRaises(ValueError):
                entry.verify_registration_commit('synthetic-invalid')
            git.assert_not_called()

    def test_registration_rejects_duplicate_keys_and_nonstandard_numbers(self):
        self.assertEqual(entry.parse_registration(b'{"gate":{"count":20}}'), {'gate': {'count': 20}})
        bad = (b'{"gate":1,"gate":2}', b'{"gate":{"count":19,"count":20}}',
               b'{"gate":{"count":NaN}}', b'{"value":Infinity}')
        for raw in bad:
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                entry.parse_registration(raw)

    def test_v1_fixed_source_identity_is_required(self):
        with patch.object(entry, 'V1_SHA256', '0' * 64), self.assertRaises(ValueError):
            entry.load_frozen_v1()

    def test_v2_source_set_is_complete_before_v1_projection(self):
        frozen = entry.load_frozen_v1()
        all_sources = frozen.SOURCE_PATHS | entry.V2_SOURCE_PATHS
        hashes = {name: 'a' * 64 for name in all_sources}
        hashes[entry.V1_RELATIVE] = entry.V1_SHA256
        registration = deepcopy(entry.REGISTRATION_CONTRACT)
        registration['source_sha256'] = hashes
        with patch.object(frozen, 'local_file', side_effect=lambda root, name: Path(name)), \
             patch.object(frozen, 'sha', side_effect=lambda path: hashes[path.as_posix()]):
            adapted = entry.verify_sources(registration, frozen)
            self.assertEqual(adapted['schema'], 'task-worker-ranking-registration-v1')
            self.assertEqual(set(adapted['source_sha256']), frozen.SOURCE_PATHS)
            self.assertEqual(registration['schema'], 'task-worker-ranking-registration-v2')
            missing = dict(registration, source_sha256=dict(hashes))
            missing['source_sha256'].pop(next(iter(entry.V2_SOURCE_PATHS)))
            with self.assertRaises(ValueError):
                entry.verify_sources(missing, frozen)
            changed = dict(registration, source_sha256=dict(hashes))
            changed['source_sha256'][next(iter(entry.V2_SOURCE_PATHS))] = 'b' * 64
            with self.assertRaises(ValueError):
                entry.verify_sources(changed, frozen)
            wrong_v1 = dict(registration, source_sha256=dict(hashes))
            wrong_v1['source_sha256'][entry.V1_RELATIVE] = 'b' * 64
            with self.assertRaises(ValueError):
                entry.verify_sources(wrong_v1, frozen)
            extra = dict(registration, source_sha256=dict(hashes, extra='a' * 64))
            with self.assertRaises(ValueError):
                entry.verify_sources(extra, frozen)
            with self.assertRaises(ValueError):
                entry.verify_sources(dict(registration, schema='wrong'), frozen)
            with self.assertRaises(ValueError):
                entry.verify_sources(dict(registration, max_model_calls=10209), frozen)

    def test_every_execution_contract_field_rejects_changes_and_type_drift(self):
        frozen = entry.load_frozen_v1()
        registration = deepcopy(entry.REGISTRATION_CONTRACT)
        registration['source_sha256'] = {name: 'a' * 64 for name in
                                          frozen.SOURCE_PATHS | entry.V2_SOURCE_PATHS}
        registration['source_sha256'][entry.V1_RELATIVE] = entry.V1_SHA256
        mutations = [
            (('schema',), 'other'),
            (('scope', 'source'), 'other'),
            (('scope', 'encoded_steps'), 'step_2'),
            (('scope', 'purpose'), 'independent_test'),
            (('scope', 'new_training'), 0),
            (('scope', 'new_games'), True),
            (('selection', 'teacher_operation'), 'WATER'),
            (('selection', 'candidate_operation'), 'WATER'),
            (('selection', 'target'), 'different'),
            (('selection', 'min_distinct_units'), 1),
            (('selection', 'min_distinct_units'), True),
            (('selection', 'score_input'), 'same_target_only'),
            (('selection', 'max_candidates_per_group'), 256.0),
            (('selection', 'tie_break'), 'last_candidate'),
            (('gate', 'min_eligible_groups'), 19),
            (('gate', 'min_eligible_groups'), 20.0),
            (('gate', 'source_clusters_exact'), 3),
            (('gate', 'min_eligible_groups_per_cluster'), 4),
            (('gate', 'no_models_if_insufficient'), 1),
            (('model_a',), 'merged_bc'),
            (('model_b',), 'early_bc'),
            (('metrics', 'cluster_aggregation'), 'group_weighted'),
            (('metrics', 'primary'), 'overall_teacher_top1'),
            (('metrics', 'direction'), 'model_b_minus_model_a'),
            (('metrics', 'global_guard', 'metrics'), ['overall_teacher_top1']),
            (('metrics', 'global_guard', 'minimum_difference_pp'), -4),
            (('metrics', 'global_guard', 'minimum_difference_pp'), -5.0),
            (('metrics', 'global_guard', 'comparison'), 'greater_than'),
            (('metrics', 'global_guard', 'role'), 'promotion'),
            (('metrics', 'primary_difference_bands'), ['<-20pp', 'other']),
            (('metrics', 'public_statistics'), ['global_guard']),
            (('minimum_initial_available_bytes',), 4294967295),
            (('minimum_initial_available_bytes',), 4294967296.0),
            (('groups_expected',), 5104.0),
            (('candidates_expected',), 322150),
            (('max_model_calls',), 10207),
            (('max_scored_rows',), 644301),
            (('sampled_deadline_seconds',), 301),
        ]
        with patch.object(frozen, 'verify_sources') as legacy_validator:
            for path, replacement in mutations:
                with self.subTest(path=path, replacement=replacement):
                    changed = deepcopy(registration)
                    node = changed
                    for key in path[:-1]:
                        node = node[key]
                    node[path[-1]] = replacement
                    with self.assertRaises(ValueError):
                        entry.verify_sources(changed, frozen)
            for path in ((), ('gate',)):
                changed = deepcopy(registration)
                node = changed if not path else changed[path[0]]
                node['unregistered_claim'] = 'synthetic'
                with self.assertRaises(ValueError):
                    entry.verify_sources(changed, frozen)
            legacy_validator.assert_not_called()

    def test_error_output_never_echoes_synthetic_arguments_or_git_command(self):
        sentinel = 'synthetic-private-argument-marker'
        stderr = io.StringIO()
        with patch.object(sys, 'argv', ['run', '--unexpected', sentinel]), \
             contextlib.redirect_stderr(stderr):
            self.assertEqual(entry.cli(), 1)
        self.assertEqual(json.loads(stderr.getvalue()),
                         {'status': 'failed', 'stage': 'arguments', 'error_type': 'ValueError'})
        self.assertNotIn(sentinel, stderr.getvalue())
        self.assertNotIn('--unexpected', stderr.getvalue())

        stderr = io.StringIO()
        failure = subprocess.CalledProcessError(1, [sentinel])
        with patch.object(entry, 'main', side_effect=failure), contextlib.redirect_stderr(stderr):
            self.assertEqual(entry.cli(), 1)
        self.assertNotIn(sentinel, stderr.getvalue())
        self.assertEqual(json.loads(stderr.getvalue())['error_type'], 'CalledProcessError')

    def test_registration_failure_stops_before_source_or_data_access(self):
        marker = 'synthetic-private-registration-marker'
        args = types.SimpleNamespace(registration_commit='a' * 40)
        stderr = io.StringIO()
        with patch.object(entry.QuietParser, 'parse_args', return_value=args), \
             patch.object(entry, 'verify_registration_commit', side_effect=ValueError(marker)), \
             patch.object(entry, 'load_frozen_v1') as load, contextlib.redirect_stderr(stderr):
            self.assertEqual(entry.cli(), 1)
        load.assert_not_called()
        self.assertNotIn(marker, stderr.getvalue())
        self.assertEqual(json.loads(stderr.getvalue()),
                         {'status': 'failed', 'stage': 'registration', 'error_type': 'ValueError'})

    def test_synthetic_main_delegates_after_checks_and_publishes_v2_identity(self):
        root, artifacts, output = MagicMock(), MagicMock(), MagicMock()
        root.__truediv__.return_value.resolve.return_value = artifacts
        game_root, output_arg = MagicMock(), MagicMock()
        game_root.resolve.return_value, output_arg.resolve.return_value = root, output
        output.parent, output.name = artifacts, 'task-worker-ranking-v2-20261030'
        args = types.SimpleNamespace(game_root=game_root, output_dir=output_arg,
                                     registration_commit='a' * 40)
        registration = {'source_sha256': {'kaggriculture/main.py': 'b' * 64}}
        raw = json.dumps(registration).encode()
        frozen = MagicMock()
        frozen.sha.return_value = 'b' * 64
        frozen.run.return_value = {'schema': 'task-worker-ranking-result-v1', 'coverage': 'sufficient'}
        adapted = {'synthetic': True}
        with patch.object(entry.QuietParser, 'parse_args', return_value=args), \
             patch.object(entry, 'verify_registration_commit', return_value=raw) as verify_commit, \
             patch.object(entry, 'load_frozen_v1', return_value=frozen), \
             patch.object(entry, 'verify_sources', return_value=adapted) as verify_sources, \
             contextlib.redirect_stdout(io.StringIO()):
            entry.main()
        self.assertEqual(verify_commit.call_count, 2)
        self.assertEqual(verify_sources.call_count, 2)
        frozen.run.assert_called_once_with(root, adapted)
        output.mkdir.assert_called_once_with(exist_ok=False)
        published = frozen.publish_result.call_args.args[1]
        self.assertEqual(published['schema'], 'task-worker-ranking-result-v2')
        self.assertEqual(published['registration_commit'], args.registration_commit)

    def test_output_created_failure_certificate_has_only_safe_stage_and_type(self):
        root, artifacts, output = MagicMock(), MagicMock(), MagicMock()
        root.__truediv__.return_value.resolve.return_value = artifacts
        game_root, output_arg = MagicMock(), MagicMock()
        game_root.resolve.return_value, output_arg.resolve.return_value = root, output
        output.parent, output.name = artifacts, 'task-worker-ranking-v2-20261030'
        started_path, failure_path = MagicMock(), MagicMock()
        output.__truediv__.side_effect = lambda name: started_path if name == 'started.json' else failure_path
        started_stream, failure_stream = io.StringIO(), io.StringIO()
        started_path.open.return_value.__enter__.return_value = started_stream
        failure_path.open.return_value.__enter__.return_value = failure_stream
        args = types.SimpleNamespace(game_root=game_root, output_dir=output_arg,
                                     registration_commit='a' * 40)
        raw = json.dumps({'source_sha256': {'kaggriculture/main.py': 'b' * 64}}).encode()
        frozen = MagicMock()
        frozen.sha.return_value = 'b' * 64
        marker = 'synthetic-private-evaluation-marker'
        frozen.run.side_effect = ValueError(marker)
        stderr = io.StringIO()
        with patch.object(entry.QuietParser, 'parse_args', return_value=args), \
             patch.object(entry, 'verify_registration_commit', return_value=raw), \
             patch.object(entry, 'load_frozen_v1', return_value=frozen), \
             patch.object(entry, 'verify_sources', return_value={'synthetic': True}), \
             contextlib.redirect_stderr(stderr):
            self.assertEqual(entry.cli(), 1)
        self.assertEqual(json.loads(failure_stream.getvalue()),
                         {'schema': 'task-worker-ranking-failure-v2',
                          'stage': 'evaluation', 'error_type': 'ValueError'})
        self.assertEqual(json.loads(stderr.getvalue()),
                         {'status': 'failed', 'stage': 'evaluation', 'error_type': 'ValueError'})
        self.assertNotIn(marker, stderr.getvalue() + failure_stream.getvalue())
        frozen.publish_result.assert_not_called()


if __name__ == '__main__':
    unittest.main()
