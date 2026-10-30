"""Synthetic checks for the data-only entrypoint and resource contract."""
from copy import deepcopy
import contextlib
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import types
import unittest
from unittest.mock import Mock, patch

import numpy as np


spec = importlib.util.spec_from_file_location('task_pair_preflight_entry', Path(__file__).with_name('run.py'))
entry = importlib.util.module_from_spec(spec)
spec.loader.exec_module(entry)


class RunnerTests(unittest.TestCase):
    def test_virtual_data_sources_flow_and_tampered_digest_rejection(self):
        """Exercise the loader and analysis without opening a real data file.

        The memmap subclasses are RAM-backed substitutes, not OS memory maps.
        """
        game_source = str(Path(__file__).resolve().parents[2])
        with patch.object(sys, 'path', [game_source, *sys.path]):
            from learning.task_pair_data_preflight_v1 import core, projection, test_core
            from learning.task_bc_v1 import train
            from learning.task_teacher_choice_bc_v1 import numeric
            from learning.task_worker_geometry_diagnostic_v1 import archive_io

            arrays, _, roles = test_core.fixture()
            cache_arrays = []
            for source in arrays[:2]:
                converted = {}
                for key, value in source.items():
                    converted[key] = np.asarray(value).view(np.memmap)
                    converted[key].flags.writeable = False
                cache_arrays.append(converted)
            digest = 'a' * 64
            cache_infos = {}
            for split, source in zip(('train', 'validation'), cache_arrays, strict=True):
                cache_infos[split] = {
                    key: {'path': split + '/' + key + '.npy', 'sha256': digest,
                          'shape': list(value.shape), 'dtype': str(value.dtype)}
                    for key, value in source.items()
                }
            manifest = {
                'source_dataset_manifest_sha256': digest, 'schema_sha256': digest,
                'summary_sha256': digest,
                'numeric_feature_schema_sha256': test_core.SCHEMA['schema_sha256'],
                'passed': True, 'max_group_size': 3, 'arrays': cache_infos,
            }
            original = {'train_seeds': roles['train'], 'validation_seeds': roles['teacher_dev']}
            conditional = {
                'data_limits': {'max_groups': 99, 'max_candidates': 999,
                                'max_archive_bytes': 1000, 'max_uncompressed_bytes': 10000},
                'archive_sha256': {'old': digest, 'new': digest},
                'expected_groups': {'old': len(arrays[2]['shared']), 'new': len(arrays[3]['shared'])},
                'expected_candidates': {'old': len(arrays[2]['candidates']),
                                        'new': len(arrays[3]['candidates'])},
            }
            contract = {
                'groups': {'train': len(arrays[0]['shared']), 'teacher_dev': len(arrays[1]['shared']),
                           'conditional_dev': len(arrays[2]['shared']) + len(arrays[3]['shared'])},
                'candidates': {'train': len(arrays[0]['candidates']),
                               'teacher_dev': len(arrays[1]['candidates']),
                               'conditional_dev': len(arrays[2]['candidates']) + len(arrays[3]['candidates'])},
                'cache_total_bytes': 16,
            }
            root = Path.cwd() / 'synthetic-nonexistent-root'

            def exercise(tampered):
                local_manifest = deepcopy(manifest)
                if tampered:
                    local_manifest['arrays']['train']['shared']['sha256'] = 'b' * 64
                frozen = types.SimpleNamespace(
                    check_import_origin=Mock(),
                    read_json=Mock(side_effect=[original, {}, {'cache_manifest_sha256': digest},
                                                local_manifest, {'registration_sha256': digest}, test_core.SCHEMA]),
                    require_early_training_match=Mock(),
                    source_roles=Mock(return_value=(roles['conditional_dev'][:2],
                                                   roles['conditional_dev'][2:], None)),
                    sha=Mock(return_value=digest), local_file=Mock(side_effect=lambda base, rel: base / rel),
                )
                with patch.dict(entry.CONTRACT, contract), \
                     patch.object(Path, 'resolve', lambda self, strict=False: self), \
                     patch.object(Path, 'is_dir', return_value=True), \
                     patch.object(Path, 'stat', return_value=types.SimpleNamespace(st_size=1)), \
                     patch.object(Path, 'read_bytes', return_value=b'synthetic-manifest'), \
                     patch.object(entry, 'stream_sha', return_value=digest) as hashes, \
                     patch.object(train, 'load_split', side_effect=cache_arrays) as cache_read, \
                     patch.object(archive_io, 'registered_inputs', return_value=(
                         conditional, test_core.SCHEMA, {'old': 'synthetic-old', 'new': 'synthetic-new'})), \
                     patch.object(numeric, 'read', side_effect=[arrays[2], arrays[3]]) as archive_read:
                    if tampered:
                        with self.assertRaises(ValueError):
                            entry.load_data(root, frozen, lambda: None)
                        self.assertEqual(cache_read.call_count, 0)
                        self.assertEqual(archive_read.call_count, 0)
                        return
                    views, loaded_roles = entry.load_data(root, frozen, lambda: None)
                    self.assertEqual(hashes.call_count, 16)
                    self.assertEqual(cache_read.call_count, 2)
                    self.assertEqual(archive_read.call_count, 2)
                    private = core.analyze(views['train'], views['teacher_dev'], views['conditional_dev'],
                                           loaded_roles, check=lambda: None)
                    result = projection.project(private, views, loaded_roles)
                    self.assertEqual(result['label_conflict'], 'detected')
                    self.assertFalse(result['unmatched_primary_ready'])

            exercise(False)
            exercise(True)

    def test_complete_source_set_and_contract_are_required(self):
        v2 = entry.load_v2()
        frozen = v2.load_frozen_v1()
        all_sources = frozen.SOURCE_PATHS | v2.V2_SOURCE_PATHS | entry.DATA_SOURCE_PATHS
        hashes = {name: 'a' * 64 for name in all_sources}
        hashes[entry.V2_RELATIVE] = entry.V2_SHA256
        registration = deepcopy(entry.CONTRACT)
        registration['source_sha256'] = hashes
        with patch.object(frozen, 'local_file', side_effect=lambda root, name: Path(name)), \
             patch.object(frozen, 'sha', side_effect=lambda path: hashes[path.as_posix()]):
            entry.verify_sources(registration, v2, frozen)
            missing = deepcopy(registration)
            missing['source_sha256'].pop(next(iter(entry.DATA_SOURCE_PATHS)))
            with self.assertRaises(ValueError):
                entry.verify_sources(missing, v2, frozen)
            for key, value in (('training', True), ('max_group_width', 255),
                               ('operation_groups_min', 19), ('new_games', 0),
                               ('conditional_source_primary_policy', 'aggregate_only')):
                changed = deepcopy(registration)
                changed[key] = value
                with self.subTest(key=key), self.assertRaises(ValueError):
                    entry.verify_sources(changed, v2, frozen)

    def test_memory_and_sampled_deadline_reject_without_changing_resources(self):
        fake = types.SimpleNamespace(virtual_memory=lambda: types.SimpleNamespace(available=0))
        with self.assertRaises(ValueError):
            entry.Budget(fake)
        fake.virtual_memory = lambda: types.SimpleNamespace(available=8 * 1024**3)
        with patch.object(entry.time, 'monotonic', side_effect=[0, 900]):
            budget = entry.Budget(fake)
            with self.assertRaises(TimeoutError):
                budget.check()

    def test_historical_numeric_overlap_reference_must_reproduce(self):
        historic = {'train_validation_numeric_overlap': {
            'validation_examples_with_train_numeric_match': 3978,
            'shared_numeric_groups_with_multiple_labels': 0}}
        private = types.SimpleNamespace(overlap={'teacher_dev': [
            {'all': {'same_label_only': 3978, 'label_conflict': 0}}]})
        entry.verify_historical_overlap(private, historic)
        private.overlap['teacher_dev'][0]['all']['same_label_only'] -= 1
        with self.assertRaises(ValueError):
            entry.verify_historical_overlap(private, historic)
        private.overlap['teacher_dev'][0]['all']['label_conflict'] = 1
        with self.assertRaises(ValueError):
            entry.verify_historical_overlap(private, historic)

    def test_unknown_cli_argument_does_not_echo_sensitive_marker(self):
        marker = 'synthetic-sensitive-marker'
        command = [sys.executable, '-B', '-I', str(Path(__file__).with_name('run.py')),
                   '--game-root', 'unused', '--output-dir', 'unused',
                   '--registration-commit', 'a' * 40, '--unexpected', marker]
        completed = subprocess.run(command, capture_output=True, text=True, timeout=30)
        self.assertNotEqual(completed.returncode, 0)
        self.assertNotIn(marker, completed.stdout + completed.stderr)
        self.assertNotIn('--unexpected', completed.stdout + completed.stderr)

    def test_interrupt_output_suppresses_exception_text(self):
        marker = 'synthetic-sensitive-interrupt-marker'
        captured = io.StringIO()
        with patch.object(entry, 'main', side_effect=KeyboardInterrupt(marker)), \
             contextlib.redirect_stderr(captured):
            self.assertEqual(entry.cli(), 1)
        self.assertIn('KeyboardInterrupt', captured.getvalue())
        self.assertNotIn(marker, captured.getvalue())

    def test_local_masks_round_trip_and_corruption_prevents_manifest(self):
        store = {}

        class SinkBytes(io.BytesIO):
            def fileno(self):
                return 0

            def __exit__(self, *args):
                store[self.name] = self.getvalue()
                return super().__exit__(*args)

        class SinkText(io.StringIO):
            def fileno(self):
                return 0

            def __exit__(self, *args):
                store[self.name] = self.getvalue()
                return super().__exit__(*args)

        class Node:
            def __init__(self, name):
                self.name = name

            def open(self, mode, encoding=None):
                if mode == 'xb':
                    stream = SinkBytes()
                elif mode == 'x':
                    stream = SinkText()
                elif mode == 'r':
                    return io.StringIO(store[self.name])
                else:
                    raise AssertionError('unexpected mode')
                stream.name = self.name
                return stream

        class Output:
            def __truediv__(self, name):
                return Node(name)

        def masks():
            return {'seen_in_train': np.array([True, False]),
                    'label_conflict': np.array([False, False]),
                    'qualifying_plant': np.array([True, True])}

        private = types.SimpleNamespace(masks={'teacher_dev': [masks()],
                                               'conditional_dev': [masks(), masks()]})
        original_load = np.load

        def load_memory(path, **kwargs):
            return original_load(io.BytesIO(store[path.name]), **kwargs)

        def digest_memory(path, check):
            return hashlib.sha256(store[path.name]).hexdigest()

        with patch.object(entry.os, 'fsync'), patch.object(np, 'load', side_effect=load_memory), \
             patch.object(entry, 'stream_sha', side_effect=digest_memory):
            entry.save_masks(Output(), private, 'a' * 40)
        manifest = json.loads(store['masks.local.json'])
        self.assertEqual(manifest['registration_commit'], 'a' * 40)
        self.assertEqual(manifest['conditional_source_order'], ['old', 'new'])
        self.assertEqual(len(manifest['masks']), 9)

        store.clear()
        with patch.object(entry.os, 'fsync'), \
             patch.object(np, 'load', return_value=np.array([False, False])), \
             patch.object(entry, 'stream_sha', side_effect=digest_memory), self.assertRaises(ValueError):
            entry.save_masks(Output(), private, 'a' * 40)
        self.assertNotIn('masks.local.json', store)


if __name__ == '__main__':
    unittest.main()
