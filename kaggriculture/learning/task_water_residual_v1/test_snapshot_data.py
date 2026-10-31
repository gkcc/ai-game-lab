"""Synthetic bounded snapshots, without reading private models or games."""
import hashlib
import io
import unittest
from unittest.mock import patch
import zipfile

import numpy as np

from learning.task_teacher_transfer_dataset_v1 import experiment as transfer
from learning.task_teacher_transfer_dataset_v1.test_dataset import group
from . import snapshot_data


def archive(arrays):
    output = io.BytesIO()
    np.savez_compressed(output, **arrays)
    return output.getvalue()


def decode(raw, split='train', digest=None):
    return snapshot_data.read_split(raw, digest or hashlib.sha256(raw).hexdigest(), split, guard=lambda: None)


class SnapshotTests(unittest.TestCase):
    def test_round_trip_preserves_source_and_immutable_features(self):
        original = [group('A01', value=1), group('A02', value=2)]
        raw = archive(transfer._split_arrays(original, lambda: None))
        decoded = decode(raw)
        self.assertEqual([g.role for g in decoded], ['A01', 'A02'])
        for a, b in zip(original, decoded):
            np.testing.assert_array_equal(a.features, b.features)
            self.assertEqual(a.target_index, b.target_index)
            with self.assertRaises(ValueError): b.features.setflags(write=True)
        with self.assertRaises(ValueError): decode(raw, split='dev')
        with self.assertRaises(ValueError): decode(raw, digest='0' * 64)

    def test_bad_offsets_operation_and_duplicates_are_rejected(self):
        original = group('A01', value=1)
        good = transfer._split_arrays([original], lambda: None)
        for key, value in (('offsets', np.array([0, 100], dtype=np.int32)),
                           ('operation', np.array([0], dtype=np.int8))):
            changed = dict(good, **{key: value})
            with self.subTest(key=key), self.assertRaises(ValueError): decode(archive(changed))
        duplicate = transfer._split_arrays([original, original], lambda: None)
        with self.assertRaises(ValueError): decode(archive(duplicate))

    def test_oversized_npy_header_is_rejected_before_allocation(self):
        raw = archive(transfer._split_arrays([], lambda: None))
        with zipfile.ZipFile(io.BytesIO(raw)) as source:
            entries = {entry.filename: source.read(entry) for entry in source.infolist()}
        header = io.BytesIO()
        np.lib.format.write_array_header_1_0(header, dict(descr='<f4', fortran_order=False,
                                                         shape=(10**12, 373)))
        output = io.BytesIO()
        with zipfile.ZipFile(output, 'w', compression=zipfile.ZIP_DEFLATED) as target:
            for name, data in entries.items():
                target.writestr(name, header.getvalue() if name == 'features.npy' else data)
        with patch.object(np, 'load', side_effect=AssertionError('allocation must not start')):
            with self.assertRaises(ValueError): decode(output.getvalue())


if __name__ == '__main__':
    unittest.main()
