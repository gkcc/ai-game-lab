"""In-memory Git/bytes and private-mask binding checks; no real data or GPU."""

from contextlib import ExitStack
from copy import deepcopy
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from learning.task_pair_data_preflight_v1 import run as preflight
from learning.task_pair_history_binding_v1 import history
from learning.task_pair_mask_recheck_v1 import run as recheck
from learning.task_pair_training_v1 import test_binding as binding_fixture
from learning.task_pair_training_v1.test_binding import encode, sha


class HistoryTests(unittest.TestCase):
    def setUp(self):
        self.mask_fixture = binding_fixture.BindingTests('test_valid_fixed_binding_loads_nine_masks')
        self.mask_fixture.setUp()
        self.head = 'e' * 40
        self.commits = (recheck.OLD_COMMIT, recheck.PUBLISHED_COMMIT,
                        history.MASK_REGISTRATION_COMMIT, history.MASK_RESULT_COMMIT, self.head)
        old_sources = {'synthetic/source.py': '1' * 64}
        old_registration = dict(deepcopy(preflight.CONTRACT), source_sha256=old_sources)
        old_result = {'schema': 'task-pair-data-preflight-result-v1',
                      'registration_commit': recheck.OLD_COMMIT, 'coverage': {},
                      'unmatched_coverage': {}, 'conditional_sources': {}, 'overlap': {},
                      'label_conflict': 'none', 'all_groups_primary_ready': True,
                      'unmatched_primary_ready': True}
        new_sources = dict(old_sources)
        new_sources.update({name: '2' * 64 for name in recheck.EXTRA_SOURCE_PATHS})
        new_registration = dict(deepcopy(recheck.CONTRACT), source_sha256=new_sources)
        new_result = recheck.public_result(history.MASK_REGISTRATION_COMMIT)
        self.public = {path: encode(value) for path, value in zip(
            history._PATHS, (old_registration, old_result, new_registration, new_result))}
        self.blobs = {(commit, path): self.public[path]
                      for commit, path in zip(self.commits[:4], history._PATHS)}
        self.blobs.update({(self.head, path): data for path, data in self.public.items()})
        self.workspace = dict(self.public)
        manifest = dict(self.mask_fixture.manifest, registration_commit=history.MASK_REGISTRATION_COMMIT)
        self.manifest = encode(manifest)
        receipt = dict(self.mask_fixture.receipt, registration_commit=history.MASK_REGISTRATION_COMMIT,
                       previous_registration_commit=recheck.OLD_COMMIT,
                       previous_result_commit=recheck.PUBLISHED_COMMIT,
                       fresh_manifest_sha256=sha(self.manifest))
        self.receipt = encode(receipt)
        # This synthetic expected value is fixed before the tested modifications.
        self.fixed_receipt_sha = sha(self.receipt)
        self.reader = Mock(side_effect=lambda name, maximum: self.mask_fixture.files[name][:maximum + 1])

    def patch_git(self, stack, *, head=None, ancestry=None):
        head = self.head if head is None else head
        stack.enter_context(patch.object(history, 'current_head', return_value=head))
        def is_ancestor(parent, child):
            if ancestry is not None:
                return ancestry(parent, child)
            return self.commits.index(parent) < self.commits.index(child)
        stack.enter_context(patch.object(recheck, 'is_ancestor', side_effect=is_ancestor))
        def blob(commit, path, *, allowed=None):
            if path not in allowed:
                raise AssertionError('Unregistered synthetic blob')
            return self.blobs[(commit, path)]
        stack.enter_context(patch.object(recheck, 'git_blob', side_effect=blob))
        stack.enter_context(patch.object(recheck, 'bounded_file',
            side_effect=lambda root, path, maximum: self.workspace[path][:maximum + 1]))

    def invoke(self, **changes):
        args = dict(expected_head=self.head, preflight_registration_commit=self.commits[0],
                    preflight_result_commit=self.commits[1], mask_registration_commit=self.commits[2],
                    mask_result_commit=self.commits[3], public_bytes=self.public,
                    expected_receipt_sha256=self.fixed_receipt_sha)
        args.update(changes)
        return history.bind_history(**args)

    def replace_public_everywhere(self, path, raw):
        self.public[path] = raw
        self.workspace[path] = raw
        self.blobs[(self.commits[history._PATHS.index(path)], path)] = raw
        self.blobs[(self.head, path)] = raw

    def test_ordered_history_then_existing_private_binding(self):
        with ExitStack() as stack:
            self.patch_git(stack)
            bound = self.invoke()
        self.reader.assert_not_called()  # The public phase never reads masks.
        self.assertEqual(repr(bound), '<PrivateHistoricalBinding>')
        masks = bound.load_masks(self.receipt, self.manifest, self.reader,
                                 role_lengths=self.mask_fixture.lengths)
        self.assertEqual(self.reader.call_count, 9)
        self.assertEqual([len(x['seen_in_train']) for x in masks['conditional_dev']], [2, 4])
        self.assertFalse(masks['teacher_dev'][0]['qualifying_plant'].flags.writeable)

    def test_every_historical_head_or_worktree_byte_change_is_rejected(self):
        for path in history._PATHS:
            for source in ('supplied', 'historical', 'head', 'worktree'):
                with self.subTest(path=path, source=source):
                    self.setUp()
                    if source == 'supplied':
                        self.public[path] += b' '
                    elif source == 'historical':
                        commit = self.commits[history._PATHS.index(path)]
                        self.blobs[(commit, path)] += b' '
                    elif source == 'head':
                        self.blobs[(self.head, path)] += b' '
                    else:
                        self.workspace[path] += b' '
                    with ExitStack() as stack:
                        self.patch_git(stack)
                        with self.assertRaises(ValueError):
                            self.invoke()
                    self.reader.assert_not_called()

    def test_wrong_head_ancestor_or_fixed_commit_stop_before_blob_reads(self):
        for mode in ('head', 'ancestor', 'identity'):
            with self.subTest(mode=mode), ExitStack() as stack:
                self.patch_git(stack, head=('f' * 40 if mode == 'head' else None),
                               ancestry=(lambda parent, child: False) if mode == 'ancestor' else None)
                get_blob = stack.enter_context(patch.object(recheck, 'git_blob',
                                                           side_effect=AssertionError('Blob read reached')))
                with self.assertRaises(ValueError):
                    self.invoke(**({'mask_result_commit': 'c' * 40} if mode == 'identity' else {}))
                get_blob.assert_not_called()
                self.reader.assert_not_called()

    def test_head_change_during_check_is_rejected(self):
        with ExitStack() as stack:
            self.patch_git(stack)
            stack.enter_context(patch.object(history, 'current_head', side_effect=[self.head, 'f' * 40]))
            with self.assertRaises(ValueError):
                self.invoke()

    def test_caller_mapping_changes_cannot_replace_checked_public_bytes(self):
        def workspace_read(root, path, maximum):
            if path == history._PATHS[0]:
                self.public[history.MASK_RESULT_REL] += b' '
            return self.workspace[path][:maximum + 1]

        with ExitStack() as stack:
            self.patch_git(stack)
            stack.enter_context(patch.object(recheck, 'bounded_file', side_effect=workspace_read))
            bound = self.invoke()
        masks = bound.load_masks(self.receipt, self.manifest, self.reader,
                                 role_lengths=self.mask_fixture.lengths)
        self.assertEqual(self.reader.call_count, 9)
        self.assertFalse(masks['teacher_dev'][0]['qualifying_plant'].flags.writeable)

    def test_fixed_receipt_digest_and_manifest_are_checked_before_mask_reads(self):
        with ExitStack() as stack:
            self.patch_git(stack)
            bound = self.invoke()
        for receipt, manifest in ((self.receipt + b' ', self.manifest),
                                  (self.receipt, self.manifest + b' ')):
            with self.assertRaises(ValueError):
                bound.load_masks(receipt, manifest, self.reader, role_lengths=self.mask_fixture.lengths)
        self.reader.assert_not_called()
        with ExitStack() as stack:
            self.patch_git(stack)
            with self.assertRaises(ValueError):
                self.invoke(expected_receipt_sha256='not-an-independent-digest')

    def test_wrong_role_lengths_and_tampered_private_mask_are_rejected(self):
        with ExitStack() as stack:
            self.patch_git(stack)
            bound = self.invoke()
        wrong = {'teacher_dev': [4], 'conditional_dev': [2, 4]}
        with self.assertRaises(ValueError):
            bound.load_masks(self.receipt, self.manifest, self.reader, role_lengths=wrong)
        name = 'teacher_dev-0-seen_in_train.npy'
        self.mask_fixture.files[name] += b'x'
        with self.assertRaises(ValueError):
            bound.load_masks(self.receipt, self.manifest, self.reader,
                             role_lengths=self.mask_fixture.lengths)

    def test_invalid_json_duplicate_keys_and_strict_boolean_flags_rejected(self):
        path = history.MASK_RESULT_REL
        raw = self.public[path]
        duplicate = b'{"schema":"duplicate",' + raw[1:]
        changed = raw.replace(b'"old_creation_time_proven":false', b'"old_creation_time_proven":0')
        self.assertNotEqual(changed, raw)
        for bad in (duplicate, changed):
            with self.subTest(kind=len(bad)):
                self.setUp()
                self.replace_public_everywhere(path, bad)
                with ExitStack() as stack:
                    self.patch_git(stack)
                    with self.assertRaises(ValueError):
                        self.invoke()

    def test_registration_source_set_and_unchanged_old_hashes_are_required(self):
        path = recheck.NEW_REG_REL
        for mode in ('changed-old', 'missing-extra', 'unexpected', 'bad-hash'):
            with self.subTest(mode=mode):
                self.setUp()
                registration = deepcopy(recheck.CONTRACT)
                registration['source_sha256'] = {'synthetic/source.py': '1' * 64}
                registration['source_sha256'].update({name: '2' * 64 for name in recheck.EXTRA_SOURCE_PATHS})
                if mode == 'changed-old':
                    registration['source_sha256']['synthetic/source.py'] = '3' * 64
                elif mode == 'missing-extra':
                    del registration['source_sha256'][next(iter(recheck.EXTRA_SOURCE_PATHS))]
                elif mode == 'unexpected':
                    registration['source_sha256']['unexpected.py'] = '4' * 64
                else:
                    registration['source_sha256']['synthetic/source.py'] = 'invalid'
                self.replace_public_everywhere(path, encode(registration))
                with ExitStack() as stack:
                    self.patch_git(stack)
                    with self.assertRaises(ValueError):
                        self.invoke()


if __name__ == '__main__':
    unittest.main()
