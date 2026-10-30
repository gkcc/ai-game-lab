"""Synthetic, in-memory tests of the mask-recheck binding contract."""

import copy
import hashlib
import io
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock, patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from learning.task_pair_training_v1.binding import load_bound_masks
from learning.task_pair_training_v1.masks import KINDS


def encode(value):
    return json.dumps(value, separators=(',', ':'), ensure_ascii=False).encode('utf-8')


def sha(value):
    return hashlib.sha256(value).hexdigest()


def npy(values):
    stream = io.BytesIO()
    np.save(stream, np.asarray(values, dtype=bool), allow_pickle=False)
    return stream.getvalue()


class BindingTests(unittest.TestCase):
    def setUp(self):
        self.new, self.old, self.old_result = 'a' * 40, 'b' * 40, 'c' * 40
        self.lengths = {'teacher_dev': [3], 'conditional_dev': [2, 4]}
        self.files = {}
        entries = []
        for role, shards in self.lengths.items():
            for index, length in enumerate(shards):
                for kind in KINDS:
                    name = f'{role}-{index}-{kind}.npy'
                    data = [False] * length
                    if kind in ('seen_in_train', 'qualifying_plant'):
                        data[0] = True
                    self.files[name] = npy(data)
                    entries.append({'name': name, 'sha256': sha(self.files[name]), 'length': length})
        self.manifest = {'schema': 'task-pair-data-local-masks-v1',
                         'registration_commit': self.new,
                         'conditional_source_order': ['old', 'new'], 'masks': entries}
        self.receipt = {'schema': 'task-pair-mask-recheck-binding-v1',
                        'registration_commit': self.new,
                        'previous_registration_commit': self.old,
                        'previous_result_commit': self.old_result,
                        'fresh_manifest_sha256': sha(encode(self.manifest)),
                        'conditional_source_order': ['old', 'new'],
                        'old_snapshot_matches_recomputation': True,
                        'fresh_masks_match_recomputation': True,
                        'old_creation_time_proven': False}
        self.result = {'schema': 'task-pair-mask-recheck-result-v1',
                       'registration_commit': self.new,
                       'previous_registration_commit': self.old,
                       'old_snapshot_matches_recomputation': True,
                       'fresh_masks_match_recomputation': True,
                       'historical_overlap_matches': True,
                       'previous_public_projection_matches': True,
                       'old_creation_time_proven': False,
                       'checkpoint_weights_loaded': False, 'model_inference': False,
                       'training': False, 'new_games': False}
        self.receipt_bytes = encode(self.receipt)
        self.manifest_bytes = encode(self.manifest)
        self.result_bytes = encode(self.result)
        # These expected digests are fixed before any tested modification.
        self.expected_receipt = sha(self.receipt_bytes)
        self.expected_result = sha(self.result_bytes)
        self.reader = Mock(side_effect=lambda name, limit: self.files[name][:limit + 1])

    def invoke(self, **changes):
        arguments = {'receipt_bytes': self.receipt_bytes,
                     'manifest_bytes': self.manifest_bytes,
                     'public_result_bytes': self.result_bytes,
                     'read_file': self.reader,
                     'expected_receipt_sha256': self.expected_receipt,
                     'expected_public_result_sha256': self.expected_result,
                     'registration_commit': self.new,
                     'previous_registration_commit': self.old,
                     'previous_result_commit': self.old_result,
                     'role_lengths': self.lengths}
        arguments.update(changes)
        return load_bound_masks(**arguments)

    def with_receipt(self, receipt):
        """Use a different trusted synthetic fixture to exercise its contract."""
        raw = encode(receipt)
        return {'receipt_bytes': raw, 'expected_receipt_sha256': sha(raw)}

    def with_result(self, result):
        raw = encode(result)
        return {'public_result_bytes': raw, 'expected_public_result_sha256': sha(raw)}

    def with_manifest(self, manifest):
        raw = encode(manifest)
        receipt = dict(self.receipt, fresh_manifest_sha256=sha(raw))
        return {'manifest_bytes': raw, **self.with_receipt(receipt)}

    def test_valid_fixed_binding_loads_nine_masks(self):
        result = self.invoke()
        self.assertEqual(self.reader.call_count, 9)
        self.assertEqual([len(x['seen_in_train']) for x in result['conditional_dev']], [2, 4])
        self.assertTrue(result['teacher_dev'][0]['qualifying_plant'][0])
        self.assertFalse(result['teacher_dev'][0]['label_conflict'].any())
        mask = result['teacher_dev'][0]['qualifying_plant']
        self.assertFalse(mask.flags.writeable)
        with self.assertRaises(ValueError):
            mask[0] = False

    def test_fixed_receipt_and_public_result_byte_changes_fail_before_reads(self):
        for field, modified in (('receipt_bytes', self.receipt_bytes + b' '),
                                ('public_result_bytes', self.result_bytes + b' ')):
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.invoke(**{field: modified})
        self.reader.assert_not_called()

    def test_manifest_change_fails_before_any_mask_read(self):
        with self.assertRaises(ValueError):
            self.invoke(manifest_bytes=self.manifest_bytes + b' ')
        self.reader.assert_not_called()

    def test_replacing_receipt_and_manifest_cannot_replace_previously_fixed_sha(self):
        name = 'teacher_dev-0-qualifying_plant.npy'
        self.files[name] = npy([False, False, False])
        altered = copy.deepcopy(self.manifest)
        next(item for item in altered['masks'] if item['name'] == name)['sha256'] = sha(self.files[name])
        forged_manifest = encode(altered)
        forged_receipt = encode(dict(self.receipt, fresh_manifest_sha256=sha(forged_manifest)))
        with self.assertRaises(ValueError):
            self.invoke(receipt_bytes=forged_receipt, manifest_bytes=forged_manifest)
        self.reader.assert_not_called()

    def test_private_binding_schema_commits_flags_and_source_order_are_strict(self):
        variants = []
        for key, wrong in (('schema', 'wrong'), ('registration_commit', self.old),
                           ('previous_registration_commit', self.new),
                           ('previous_result_commit', self.new),
                           ('fresh_manifest_sha256', 'X' * 64),
                           ('conditional_source_order', ['new', 'old']),
                           ('old_snapshot_matches_recomputation', 1),
                           ('fresh_masks_match_recomputation', False),
                           ('old_creation_time_proven', 0)):
            altered = copy.deepcopy(self.receipt)
            altered[key] = wrong
            variants.append(altered)
        extra = dict(self.receipt, unexpected=True)
        missing = dict(self.receipt)
        del missing['previous_result_commit']
        variants.extend((extra, missing))
        for index, item in enumerate(variants):
            with self.subTest(index=index), self.assertRaises(ValueError):
                self.invoke(**self.with_receipt(item))
        self.reader.assert_not_called()

    def test_public_result_schema_commits_and_all_success_flags_are_strict(self):
        variants = []
        for key in ('old_snapshot_matches_recomputation', 'fresh_masks_match_recomputation',
                    'historical_overlap_matches', 'previous_public_projection_matches'):
            altered = dict(self.result)
            altered[key] = 1  # int is not the JSON boolean true.
            variants.append(altered)
        for key in ('old_creation_time_proven', 'checkpoint_weights_loaded',
                    'model_inference', 'training', 'new_games'):
            altered = dict(self.result)
            altered[key] = 0  # int is not the JSON boolean false.
            variants.append(altered)
        variants += [dict(self.result, schema='wrong'),
                     dict(self.result, registration_commit=self.old),
                     dict(self.result, previous_registration_commit=self.new),
                     dict(self.result, unexpected=True)]
        for index, item in enumerate(variants):
            with self.subTest(index=index), self.assertRaises(ValueError):
                self.invoke(**self.with_result(item))
        self.reader.assert_not_called()

    def test_duplicate_json_keys_and_nonstandard_constants_are_rejected(self):
        def duplicate(raw):
            self.assertEqual(raw[:1], b'{')
            return b'{"schema":"duplicate",' + raw[1:]

        bad_receipt = duplicate(self.receipt_bytes)
        with self.assertRaises(ValueError):
            self.invoke(receipt_bytes=bad_receipt, expected_receipt_sha256=sha(bad_receipt))
        bad_result = duplicate(self.result_bytes)
        with self.assertRaises(ValueError):
            self.invoke(public_result_bytes=bad_result, expected_public_result_sha256=sha(bad_result))
        bad_manifest = duplicate(self.manifest_bytes)
        forged = dict(self.receipt, fresh_manifest_sha256=sha(bad_manifest))
        with self.assertRaises(ValueError):
            self.invoke(manifest_bytes=bad_manifest, **self.with_receipt(forged))
        nan_receipt = self.receipt_bytes.replace(b'"old_creation_time_proven":false',
                                                 b'"old_creation_time_proven":NaN')
        self.assertNotEqual(nan_receipt, self.receipt_bytes)
        with self.assertRaises(ValueError):
            self.invoke(receipt_bytes=nan_receipt, expected_receipt_sha256=sha(nan_receipt))
        self.reader.assert_not_called()

    def test_manifest_order_role_lengths_and_changed_file_fail(self):
        altered = copy.deepcopy(self.manifest)
        altered['conditional_source_order'].reverse()
        with self.assertRaises(ValueError):
            self.invoke(**self.with_manifest(altered))
        with self.assertRaises(ValueError):
            self.invoke(role_lengths={'teacher_dev': [4], 'conditional_dev': [2, 4]})
        self.reader.assert_not_called()
        self.files['teacher_dev-0-seen_in_train.npy'] += b'x'
        with self.assertRaises(ValueError):
            self.invoke()

    def test_zero_conflict_is_required_even_with_consistent_manifest_and_receipt(self):
        name = 'teacher_dev-0-label_conflict.npy'
        self.files[name] = npy([True, False, False])
        altered = copy.deepcopy(self.manifest)
        next(item for item in altered['masks'] if item['name'] == name)['sha256'] = sha(self.files[name])
        with self.assertRaises(ValueError):
            self.invoke(**self.with_manifest(altered))

    def test_huge_declared_npy_shape_is_rejected_before_any_payload_allocation(self):
        name = 'teacher_dev-0-seen_in_train.npy'
        stream = io.BytesIO()
        np.lib.format.write_array_header_1_0(stream, {
            'descr': '|b1', 'fortran_order': False, 'shape': (1_000_000_000,),
        })
        self.files[name] = stream.getvalue()  # A short header only; no large array is created.
        altered = copy.deepcopy(self.manifest)
        next(item for item in altered['masks'] if item['name'] == name)['sha256'] = sha(self.files[name])
        with patch('numpy.load', side_effect=AssertionError('unsafe npy load')), \
             patch('numpy.fromfile', side_effect=AssertionError('unsafe npy allocation')), \
             patch('numpy.frombuffer', side_effect=AssertionError('payload reached')), \
             self.assertRaises(ValueError):
            self.invoke(**self.with_manifest(altered))

    def test_invalid_expected_identities_and_input_types_fail_before_reads(self):
        bad = [dict(expected_receipt_sha256=''),
               dict(expected_public_result_sha256='A' * 64),
               dict(previous_result_commit='not-a-commit'),
               dict(registration_commit=self.new.upper()),
               dict(receipt_bytes=bytearray(self.receipt_bytes)),
               dict(manifest_bytes=b'x' * 65537),
               dict(read_file=None)]
        for index, values in enumerate(bad):
            with self.subTest(index=index), self.assertRaises(ValueError):
                self.invoke(**values)
        self.reader.assert_not_called()


if __name__ == '__main__':
    unittest.main()
