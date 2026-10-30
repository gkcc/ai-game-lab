"""Synthetic-only tests for the unregistered mask recheck entry."""
from copy import deepcopy
import contextlib
import hashlib
import io
import json
import numbers
from pathlib import Path
import subprocess
import sys
import types
import unittest
from unittest.mock import Mock, patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from learning.task_pair_data_preflight_v1 import core, projection, run as preflight, test_core
from learning.task_pair_mask_recheck_v1 import run as entry
from learning.task_pair_training_v1 import masks, recompute
from learning.task_worker_ranking_v2 import run as v2


def synthetic_input():
    arrays, source_views, roles = test_core.fixture()
    views = {'train': [source_views[0]], 'teacher_dev': [source_views[1]],
             'conditional_dev': source_views[2:]}
    # The existing fixture deliberately contains a label conflict. This
    # independent synthetic scenario models the previously registered zero gate.
    arrays[0]['labels'][1] = 0
    arrays[2]['teacher_choice_label'][1] = 0
    private = core.analyze(views['train'], views['teacher_dev'], views['conditional_dev'],
                           roles, check=lambda: None)
    files, entries = {}, []
    for role in masks.ROLES:
        for source, shard in enumerate(private.masks[role]):
            for kind in masks.KINDS:
                name = f'{role}-{source}-{kind}.npy'
                buffer = io.BytesIO()
                np.save(buffer, shard[kind], allow_pickle=False)
                raw = buffer.getvalue()
                files[name] = raw
                entries.append({'name': name, 'sha256': hashlib.sha256(raw).hexdigest(),
                                'length': len(shard[kind])})
    manifest = {'schema': 'task-pair-data-local-masks-v1',
                'registration_commit': entry.OLD_COMMIT,
                'conditional_source_order': ['old', 'new'], 'masks': entries}
    published = projection.project(private, views, roles)
    published.update({'schema': 'task-pair-data-preflight-result-v1',
                      'registration_commit': entry.OLD_COMMIT})
    return views, roles, private, files, manifest, published


class RecheckTests(unittest.TestCase):
    def virtual_main(self, *, fail_analysis=False, fail_runtime=False):
        """Exercise main's ordering and receipts with a memory-only filesystem."""
        events = []
        writes = {}

        class VirtualPath:
            def __init__(self, text):
                self.text = str(text)

            def __truediv__(self, name):
                return VirtualPath(self.text + '/' + str(name))

            def __eq__(self, other):
                return isinstance(other, VirtualPath) and self.text == other.text

            def __str__(self):
                return self.text

            def resolve(self, strict=False):
                return self

            def is_dir(self):
                return True

            def is_symlink(self):
                return False

            def mkdir(self, exist_ok=False):
                if exist_ok:
                    raise AssertionError('Output must be exclusive')
                events.append('mkdir')

        def write(path, value):
            name = str(path).rsplit('/', 1)[-1]
            if name in writes:
                raise FileExistsError('duplicate synthetic receipt')
            events.append(name)
            writes[name] = deepcopy(value)

        old = {'source_sha256': {'kaggriculture/main.py': entry.DEFAULT_BOT_SHA}}
        registration = {'source_sha256': {'kaggriculture/main.py': entry.DEFAULT_BOT_SHA}}
        fake_frozen = types.SimpleNamespace(
            require_direct_artifacts=Mock(), reject_shadows=Mock(),
            local_file=Mock(return_value='synthetic-default-bot'),
            sha=Mock(return_value=entry.DEFAULT_BOT_SHA),
            bind_runtime=Mock(side_effect=lambda value: events.append('bind')),
            check_import_origin=Mock(), read_json=Mock(return_value={'synthetic': True}),
            publish_result=Mock(side_effect=lambda output, result: events.append('publish')),
        )
        fake_v2 = types.SimpleNamespace(load_frozen_v1=lambda: fake_frozen,
                                        strict_contract_equal=v2.strict_contract_equal)
        budget = types.SimpleNamespace(check=lambda: None)
        fake_preflight = types.SimpleNamespace(
            load_v2=lambda: fake_v2, CONTRACT=entry.CONTRACT,
            verify_sources=Mock(side_effect=lambda *args: events.append('old-sources')),
            Budget=lambda _: budget,
            load_data=Mock(side_effect=lambda *args: (events.append('load') or ({}, {}))),
        )

        def verify_new(*args):
            events.append('new-registration')
            return b'synthetic', registration

        def verify_old(*args):
            events.append('old-history')
            return old, {'synthetic': True}

        def analysis(*args, **kwargs):
            events.append('analysis')
            if fail_analysis:
                raise ValueError('synthetic-private-detail')
            return object(), {'teacher_dev': [2], 'conditional_dev': [2, 2]}

        def fresh(*args):
            events.append('fresh')
            return 'f' * 64

        def runtime():
            events.append('runtime')
            result = dict(entry.RUNTIME)
            if fail_runtime:
                result['numpy'] = 'different'
            return result

        args = ['run.py', '--game-root', 'virtual-game-root', '--registration-commit', 'b' * 40]
        patches = (
            patch.object(sys, 'argv', args), patch.object(entry, 'Path', VirtualPath),
            patch.object(entry, 'verify_game_root'), patch.object(entry, 'write_json_exclusive', side_effect=write),
            patch.object(entry, 'load_preflight', return_value=fake_preflight),
            patch.object(entry, 'load_frozen_v1_bounded', return_value=fake_frozen),
            patch.object(entry, 'verify_new_registration', side_effect=verify_new),
            patch.object(entry, 'verify_old_history', side_effect=verify_old),
            patch.object(entry, 'verify_new_sources', side_effect=lambda *args: events.append('new-sources')),
            patch.object(entry, 'runtime_identity', side_effect=runtime),
            patch.object(entry, 'read_local_mask', return_value=b'{}'),
            patch.object(entry, 'compare_current_masks', side_effect=analysis),
            patch.object(entry, 'create_fresh_masks', side_effect=fresh),
            contextlib.redirect_stdout(io.StringIO()),
        )
        with contextlib.ExitStack() as stack:
            for item in patches:
                stack.enter_context(item)
            if fail_analysis or fail_runtime:
                with self.assertRaises(ValueError):
                    entry.main()
            else:
                entry.main()
        return events, writes

    def compare(self, *, changed=None, published_change=False, history_failure=False):
        views, roles, expected, files, manifest, published = synthetic_input()
        if changed is not None:
            name = 'teacher_dev-0-qualifying_plant.npy'
            altered = expected.masks['teacher_dev'][0]['qualifying_plant'].copy()
            altered[0] = not altered[0]
            buffer = io.BytesIO()
            np.save(buffer, altered, allow_pickle=False)
            files[name] = buffer.getvalue()
            next(item for item in manifest['masks'] if item['name'] == name)['sha256'] = hashlib.sha256(files[name]).hexdigest()
        if published_change:
            published['all_groups_primary_ready'] = not published['all_groups_primary_ready']
        validate_history = Mock(side_effect=ValueError('synthetic') if history_failure else None)
        preflight = types.SimpleNamespace(verify_historical_overlap=validate_history)
        read = Mock(side_effect=lambda name, maximum: files[name][:maximum + 1])
        result = entry.compare_current_masks(preflight, core, projection, masks, recompute, v2,
                                             views, roles, {'synthetic': True}, published,
                                             json.dumps(manifest).encode(), read, lambda: None)
        return result, expected, validate_history, read

    def test_source_reanalysis_and_saved_masks_agree(self):
        (private, lengths), expected, validate_history, read = self.compare()
        self.assertTrue(recompute.verify_masks(private.masks, expected.masks))
        self.assertEqual(lengths, {'teacher_dev': [2], 'conditional_dev': [2, 2]})
        self.assertEqual(validate_history.call_count, 1)
        self.assertEqual(read.call_count, 9)

    def test_main_verifies_before_load_and_publishes_only_after_fresh_binding(self):
        events, writes = self.virtual_main()
        self.assertLess(events.index('new-registration'), events.index('load'))
        self.assertLess(events.index('old-history'), events.index('load'))
        self.assertLess(events.index('old-sources'), events.index('load'))
        self.assertLess(events.index('new-sources'), events.index('load'))
        self.assertLess(events.index('runtime'), events.index('load'))
        self.assertLess(events.index('analysis'), events.index('fresh'))
        self.assertLess(events.index('fresh'), events.index('binding.local.json'))
        self.assertLess(events.index('binding.local.json'), events.index('publish'))
        for check in ('new-registration', 'old-history', 'old-sources', 'new-sources', 'runtime'):
            positions = [i for i, value in enumerate(events) if value == check]
            self.assertEqual(len(positions), 2, check)
            self.assertGreater(positions[1], events.index('fresh'), check)
            self.assertLess(positions[1], events.index('binding.local.json'), check)
        new_positions = [i for i, name in enumerate(events) if name == 'new-sources']
        old_positions = [i for i, name in enumerate(events) if name == 'old-sources']
        self.assertTrue(all(new < old for new, old in zip(new_positions, old_positions, strict=True)))
        self.assertEqual(writes['binding.local.json']['fresh_manifest_sha256'], 'f' * 64)
        self.assertFalse(writes['binding.local.json']['old_creation_time_proven'])

    def test_analysis_failure_keeps_safe_stage_without_fresh_or_result(self):
        events, writes = self.virtual_main(fail_analysis=True)
        self.assertEqual(writes['failure.json']['stage'], 'analysis')
        self.assertEqual(writes['failure.json']['error_type'], 'ValueError')
        self.assertNotIn('synthetic-private-detail', json.dumps(writes['failure.json']))
        self.assertNotIn('fresh', events)
        self.assertNotIn('binding.local.json', events)
        self.assertNotIn('publish', events)

    def test_runtime_drift_stops_before_any_dataset_load(self):
        events, writes = self.virtual_main(fail_runtime=True)
        self.assertEqual(writes['failure.json']['stage'], 'guard')
        self.assertEqual(writes['failure.json']['error_type'], 'ValueError')
        self.assertNotIn('load', events)
        self.assertNotIn('fresh', events)
        self.assertNotIn('publish', events)

    def check_fresh_masks_in_memory(self, *, corrupt_header=False):
        _, _, private, _, _, _ = synthetic_input()
        lengths = {'teacher_dev': [2], 'conditional_dev': [2, 2]}
        store = {}

        class SinkBytes(io.BytesIO):
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
                    if self.name in store:
                        raise FileExistsError('synthetic output already exists')
                    stream = SinkBytes()
                    stream.name = self.name
                    return stream
                raise AssertionError('unexpected output mode')

        class Output:
            def __truediv__(self, name):
                return Node(name)

        header = io.BytesIO()
        np.lib.format.write_array_header_1_0(header, {
            'descr': '|b1', 'fortran_order': False, 'shape': (1_000_000_000,),
        })
        huge_declared_shape = header.getvalue()
        self.assertLess(len(huge_declared_shape), 4096)

        def write_manifest(node, value):
            if node.name in store:
                raise FileExistsError('synthetic manifest already exists')
            store[node.name] = (json.dumps(value, ensure_ascii=False, indent=2) + '\n').encode('utf-8')

        def bounded_memory(root, name, maximum):
            raw = store[name]
            if corrupt_header and name == 'teacher_dev-0-seen_in_train.npy':
                raw = huge_declared_shape
            if len(raw) > maximum:
                raise ValueError('synthetic output exceeds bound')
            return raw

        fake_preflight = types.SimpleNamespace(save_masks=Mock(side_effect=AssertionError('legacy writer called')))
        with contextlib.ExitStack() as stack:
            stack.enter_context(patch.object(entry.os, 'fsync'))
            unsafe_load = stack.enter_context(patch.object(np, 'load', side_effect=AssertionError('unsafe load')))
            stack.enter_context(patch.object(entry, 'read_local_mask', side_effect=bounded_memory))
            stack.enter_context(patch.object(entry, 'write_json_exclusive', side_effect=write_manifest))
            if corrupt_header:
                allocation = stack.enter_context(patch.object(np, 'frombuffer',
                                                               side_effect=AssertionError('payload reached')))
                large_attempts = []
                for name in ('empty', 'zeros', 'ones', 'full'):
                    original = getattr(np, name)

                    def guarded(shape, *args, _original=original, **kwargs):
                        try:
                            dimensions = (shape,) if isinstance(shape, numbers.Integral) else tuple(shape)
                            count = 1
                            for dimension in dimensions:
                                count *= int(dimension)
                        except (TypeError, ValueError):
                            return _original(shape, *args, **kwargs)
                        if count > 1_000_000:
                            large_attempts.append(True)
                            raise AssertionError('large synthetic allocation attempted')
                        return _original(shape, *args, **kwargs)

                    stack.enter_context(patch.object(np, name, side_effect=guarded))
                with self.assertRaises(ValueError):
                    entry.create_fresh_masks(fake_preflight, masks, recompute, Output(), private, lengths, 'b' * 40)
                allocation.assert_not_called()
                self.assertEqual(large_attempts, [])
                digest = None
            else:
                digest = entry.create_fresh_masks(fake_preflight, masks, recompute, Output(), private, lengths,
                                                  'b' * 40)
            unsafe_load.assert_not_called()
        fake_preflight.save_masks.assert_not_called()
        self.assertEqual(len(store), 10)  # Nine masks and one manifest.
        if not corrupt_header:
            self.assertEqual(digest, hashlib.sha256(store['masks.local.json']).hexdigest())
        self.assertEqual(json.loads(store['masks.local.json'])['registration_commit'], 'b' * 40)

    def test_fresh_masks_are_written_then_bounded_read_back_and_compared(self):
        self.check_fresh_masks_in_memory()

    def test_replaced_fresh_mask_huge_header_is_rejected_without_load_or_allocation(self):
        self.check_fresh_masks_in_memory(corrupt_header=True)

    def test_modified_equal_length_mask_is_rejected_even_with_updated_sha(self):
        with self.assertRaises(ValueError):
            self.compare(changed=True)

    def test_published_projection_or_historical_check_mismatch_stops(self):
        with self.assertRaises(ValueError):
            self.compare(published_change=True)
        with self.assertRaises(ValueError):
            self.compare(history_failure=True)

    def test_old_commits_must_be_ancestors_and_blobs_match_exactly(self):
        old = b'{"old":true}\r\n'
        published = b'{"result":true}\n'
        blobs = {entry.OLD_REG_REL: old, entry.OLD_RESULT_REL: published}
        fake = types.SimpleNamespace(parse_registration=lambda raw: json.loads(raw))
        with patch.object(entry, 'is_ancestor', return_value=True) as ancestor, \
             patch.object(entry, 'git_blob', side_effect=lambda commit, path: blobs[path]), \
             patch.object(entry, 'bounded_file', side_effect=lambda root, path, limit: blobs[path]):
            self.assertEqual(entry.verify_old_history(fake, 'b' * 40), ({'old': True}, {'result': True}))
            self.assertEqual(ancestor.call_count, 2)
        with patch.object(entry, 'is_ancestor', return_value=False), self.assertRaises(ValueError):
            entry.verify_old_history(fake, 'b' * 40)
        mismatched = dict(blobs, **{entry.OLD_REG_REL: old.replace(b'\r\n', b'\n')})
        with patch.object(entry, 'is_ancestor', return_value=True), \
             patch.object(entry, 'git_blob', side_effect=lambda commit, path: blobs[path]), \
             patch.object(entry, 'bounded_file', side_effect=lambda root, path, limit: mismatched[path]), \
             self.assertRaises(ValueError):
            entry.verify_old_history(fake, 'b' * 40)
        # Even a restored worktree is insufficient if the new HEAD changed one
        # of the two historical blobs.
        for changed in (entry.OLD_REG_REL, entry.OLD_RESULT_REL):
            def altered_head(commit, path):
                return b'new-head-mismatch' if commit == 'b' * 40 and path == changed else blobs[path]
            with self.subTest(changed=changed), \
                 patch.object(entry, 'is_ancestor', return_value=True), \
                 patch.object(entry, 'git_blob', side_effect=altered_head), \
                 patch.object(entry, 'bounded_file', side_effect=lambda root, path, limit: blobs[path]), \
                 self.assertRaises(ValueError):
                entry.verify_old_history(fake, 'b' * 40)

    def test_only_new_head_is_given_to_existing_head_verifier(self):
        raw = b'{"new":true}'
        fake = types.SimpleNamespace(REGISTRATION=None,
                                     verify_registration_commit=Mock(return_value=raw),
                                     parse_registration=lambda value: json.loads(value))
        with patch.object(entry, 'git_blob', return_value=raw), \
             patch.object(entry, 'bounded_file', return_value=raw):
            self.assertEqual(entry.verify_new_registration(fake, 'b' * 40), (raw, {'new': True}))
        fake.verify_registration_commit.assert_called_once_with('b' * 40)
        self.assertNotEqual('b' * 40, entry.OLD_COMMIT)

    def test_registered_source_set_and_historical_identity_are_exact(self):
        contents = {'old.py': b'old source', 'new.py': b'new source'}
        digests = {name: hashlib.sha256(raw).hexdigest() for name, raw in contents.items()}
        old = {'source_sha256': {'old.py': digests['old.py']}}
        registration = deepcopy(entry.CONTRACT)
        registration['source_sha256'] = dict(digests)
        fake = types.SimpleNamespace(local_file=lambda root, name: name)
        with patch.object(entry, 'bounded_file', side_effect=lambda root, name, maximum: contents[name]), \
             patch.object(entry, 'git_blob', side_effect=lambda commit, name, **kwargs: contents[name]):
            entry.verify_new_sources(registration, old, v2, fake, 'b' * 40, extras={'new.py'})
            registration['source_sha256']['old.py'] = 'c' * 64
            with self.assertRaises(ValueError):
                entry.verify_new_sources(registration, old, v2, fake, 'b' * 40, extras={'new.py'})
            registration['source_sha256']['old.py'] = digests['old.py']
            registration['source_sha256']['unregistered.py'] = 'd' * 64
            with self.assertRaises(ValueError):
                entry.verify_new_sources(registration, old, v2, fake, 'b' * 40, extras={'new.py'})

    def test_restored_worktree_cannot_hide_different_head_source(self):
        contents = {'old.py': b'old source', 'new.py': b'new source'}
        digests = {name: hashlib.sha256(raw).hexdigest() for name, raw in contents.items()}
        old = {'source_sha256': {'old.py': digests['old.py']}}
        registration = deepcopy(entry.CONTRACT)
        registration['source_sha256'] = dict(digests)
        fake = types.SimpleNamespace(local_file=lambda root, name: name)
        for changed in contents:
            def head_blob(commit, name, **kwargs):
                return b'different HEAD source' if name == changed else contents[name]
            with self.subTest(changed=changed), \
                 patch.object(entry, 'bounded_file', side_effect=lambda root, name, maximum: contents[name]), \
                 patch.object(entry, 'git_blob', side_effect=head_blob), self.assertRaises(ValueError):
                entry.verify_new_sources(registration, old, v2, fake, 'b' * 40, extras={'new.py'})

    def test_oversize_git_blob_is_rejected_before_show(self):
        with patch.object(entry.subprocess, 'check_output', return_value=b'4194305\n') as git:
            with self.assertRaises(ValueError):
                entry.git_blob('a' * 40, entry.OLD_REG_REL)
            self.assertEqual(git.call_count, 1)

    def test_oversize_workspace_source_stops_before_legacy_sha(self):
        old = {'source_sha256': {'old.py': 'a' * 64}}
        registration = deepcopy(entry.CONTRACT)
        registration['source_sha256'] = dict(old['source_sha256'])
        registration['source_sha256'].update({name: 'b' * 64 for name in entry.EXTRA_SOURCE_PATHS})
        fake_frozen = types.SimpleNamespace(local_file=Mock(return_value='synthetic-source'))
        fake_preflight = types.SimpleNamespace(verify_sources=Mock())
        oversized = b'x' * (entry.METADATA_LIMIT + 1)

        class VirtualPath:
            def __init__(self, name):
                self.name = str(name)

            def __truediv__(self, other):
                return VirtualPath(self.name + '/' + str(other))

            @property
            def parts(self):
                return tuple(self.name.replace('\\', '/').split('/'))

            def resolve(self, strict=False):
                return self

            def is_absolute(self):
                return self.name.startswith('/')

            def is_symlink(self):
                return False

            def is_relative_to(self, root):
                return True

            def is_file(self):
                return True

            def open(self, mode):
                self_mode = mode
                if self_mode != 'rb':
                    raise AssertionError('unexpected synthetic file mode')
                return io.BytesIO(oversized)

        with patch.object(entry, 'Path', VirtualPath), patch.object(entry, 'git_blob') as git, \
             self.assertRaises(ValueError):
            entry.verify_sources_in_order(registration, old, v2, fake_frozen, 'b' * 40, fake_preflight)
        git.assert_not_called()
        fake_preflight.verify_sources.assert_not_called()

    def test_frozen_v1_executes_verified_snapshot_without_a_second_read(self):
        raw = b'NAME_AT_EXEC = __name__\nFILE_AT_EXEC = __file__\nVALUE = 7\n'
        fake = types.SimpleNamespace(V1_RELATIVE='synthetic.py',
                                     V1_SHA256=hashlib.sha256(raw).hexdigest(),
                                     load_frozen_v1=Mock(side_effect=AssertionError('unsafe reread')))
        with patch.object(entry, 'bounded_file', side_effect=[raw, ValueError('source changed')]) as reader:
            loaded = entry.load_frozen_v1_bounded(fake)
        self.assertEqual(loaded.VALUE, 7)
        self.assertEqual(loaded.NAME_AT_EXEC, 'task_worker_ranking_v1_frozen_entry')
        self.assertEqual(loaded.FILE_AT_EXEC, str(entry.REPO / fake.V1_RELATIVE))
        reader.assert_called_once_with(entry.REPO, fake.V1_RELATIVE, entry.METADATA_LIMIT)
        fake.load_frozen_v1.assert_not_called()
        for value in (b'different', ValueError('too large')):
            with self.subTest(failure=type(value).__name__), \
                 patch.object(entry, 'bounded_file', side_effect=value if isinstance(value, Exception) else None,
                              return_value=None if isinstance(value, Exception) else value), \
                 patch('builtins.compile') as compiler, self.assertRaises(ValueError):
                entry.load_frozen_v1_bounded(fake)
            compiler.assert_not_called()
        fake.load_frozen_v1.assert_not_called()

    def test_mask_path_traversal_rejected_without_opening_a_directory(self):
        for name in ('../secret', 'a/b', 'a\\b'):
            with self.subTest(name=name), self.assertRaises(ValueError):
                entry.read_local_mask('nonexistent', name, 16)

    def test_game_root_identity_is_checked_before_output_creation(self):
        raw = b'synthetic default bot'
        with patch.object(entry, 'bounded_file', return_value=raw), \
             patch.object(entry, 'DEFAULT_BOT_SHA', hashlib.sha256(raw).hexdigest()):
            entry.verify_game_root('synthetic-root')
        with patch.object(entry, 'bounded_file', return_value=raw), \
             patch.object(entry, 'DEFAULT_BOT_SHA', '0' * 64), self.assertRaises(ValueError):
            entry.verify_game_root('synthetic-root')

    def test_runtime_requires_exact_platform_and_versions(self):
        expected = dict(entry.RUNTIME)
        with patch.object(entry, 'runtime_identity', return_value=expected):
            entry.verify_runtime()
        for key in expected:
            wrong = dict(expected)
            wrong[key] = 'different'
            with self.subTest(field=key), patch.object(entry, 'runtime_identity', return_value=wrong), \
                 self.assertRaises(ValueError):
                entry.verify_runtime()

    def test_cli_rejects_arguments_and_interrupt_without_sensitive_text(self):
        marker = 'synthetic-sensitive-marker'
        command = [sys.executable, '-B', '-I', str(Path(__file__).with_name('run.py')),
                   '--game-root', 'unused', '--registration-commit', 'a' * 40,
                   '--unexpected', marker]
        result = subprocess.run(command, capture_output=True, text=True, timeout=30)
        self.assertNotEqual(result.returncode, 0)
        self.assertNotIn(marker, result.stdout + result.stderr)
        captured = io.StringIO()
        with patch.object(entry, 'main', side_effect=KeyboardInterrupt(marker)), \
             contextlib.redirect_stderr(captured):
            self.assertEqual(entry.cli(), 1)
        self.assertIn('KeyboardInterrupt', captured.getvalue())
        self.assertNotIn(marker, captured.getvalue())

    def test_public_projection_has_no_private_manifest_hash_or_data(self):
        result = entry.public_result('b' * 40)
        self.assertFalse(result['old_creation_time_proven'])
        self.assertFalse(result['checkpoint_weights_loaded'])
        self.assertFalse(result['training'])
        for key in result:
            self.assertNotIn('sha256', key)
            self.assertNotIn('seed', key)
            self.assertNotIn('path', key)
        with self.assertRaises(ValueError):
            entry.public_result('not-a-commit')


if __name__ == '__main__':
    unittest.main()
