"""Focused regressions for the read-only sampler diagnostic."""
import importlib.util
import hashlib
import io
from pathlib import Path
import sys
import subprocess
import time
import unittest
from unittest import mock
import zipfile

import numpy as np

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('sampling_diagnostic', HERE / 'analyze.py')
diagnostic = importlib.util.module_from_spec(spec)
spec.loader.exec_module(diagnostic)
sys.path.insert(0, str(HERE.parents[1]))
from learning.task_bc_v1.train import microbatches


class SamplingDiagnosticTest(unittest.TestCase):
    def test_wide_group_early_or_late_changes_admission(self):
        widths = np.array([512] + [1] * 256)
        early, _, early_padding = diagnostic.original_batch(widths, range(257))
        late_order = list(range(1, 129)) + [0] + list(range(129, 257))
        late, skipped, late_padding = diagnostic.original_batch(widths, late_order)
        self.assertEqual((len(early), early_padding), (128, 65536))
        self.assertIn(0, early)
        self.assertEqual((len(late), skipped, late_padding), (256, 1, 256))
        self.assertNotIn(0, late)

    def test_microbatch_geometry_matches_existing_public_helper(self):
        widths = np.array([1, 2, 3, 3, 5, 6])
        offsets = np.concatenate(([0], np.cumsum(widths)))
        ids = [5, 2, 0, 4, 1, 3]
        shapes = list(microbatches({'offsets': offsets}, ids, budget=8))
        expected = (len(shapes), sum(capacity * width for _, capacity, width in shapes))
        self.assertEqual(diagnostic.microbatch_geometry(widths, ids, budget=8), expected)
        self.assertEqual(sorted(i for group, _, _ in shapes for i in group), sorted(ids))

    def test_empty_or_oversized_groups_are_rejected(self):
        for width in (0, 65537):
            with self.subTest(width=width), self.assertRaises(ValueError):
                diagnostic.padded_width(width)
        with self.assertRaises(ValueError):
            diagnostic.original_batch(np.array([1]), [])

    def test_matched_group_counts_and_update_weighting(self):
        old_updates = diagnostic.UPDATES
        try:
            diagnostic.UPDATES = 3
            widths = np.ones(40, dtype=np.int64)
            labels = np.array([0] * 20 + [1] * 20)
            result = diagnostic.reconstruct(widths, labels, 20, 23, (2, 3, 1),
                                            ('DIG', 'PLANT'), time.monotonic())
        finally:
            diagnostic.UPDATES = old_updates
        self.assertEqual(result['total_exposures_each_method'], 120)
        self.assertEqual(result['batch_groups_min'], result['batch_groups_max'])
        for row in result['by_source'].values():
            self.assertEqual(row['reconstructed_exposures'], 60)
            self.assertEqual(row['matched_uniform_exposures'], 60)
            self.assertEqual(row['reconstructed_mean_update_share'], 0.5)

    def test_small_category_withholds_whole_dimension(self):
        population = np.ones(41, dtype=np.int64)
        counts = np.ones(41, dtype=np.int32)
        weights = np.ones(41, dtype=np.float64)
        ids = np.arange(41)
        masks = {'first': ids < 20, 'second': (ids >= 20) & (ids < 40), 'small': ids == 40}
        result = diagnostic.aggregate(population, counts, weights, counts, weights, masks, 1)
        self.assertEqual(result, {'withheld': True, 'reason': 'insufficient_group_size'})

    def test_update_weight_differs_from_raw_exposure_count(self):
        ids = np.arange(40)
        counts = np.zeros(40, dtype=np.int32)
        weights = np.zeros(40, dtype=np.float64)
        counts[0], weights[0] = 1, 1.0
        counts[20:30], weights[20:30] = 1, 0.1
        masks = {'a': ids < 20, 'b': ids >= 20}
        rows = diagnostic.aggregate(ids, counts, weights, counts, weights, masks, 2)
        self.assertAlmostEqual(rows['a']['reconstructed_exposure_share'], 1 / 11, places=6)
        self.assertEqual(rows['a']['reconstructed_mean_update_share'], 0.5)
        self.assertEqual(rows['b']['reconstructed_mean_update_share'], 0.5)

    def test_invalid_cli_does_not_echo_private_argument(self):
        private_sentinel = 'PRIVATE_SENTINEL_ONLY_FOR_TEST'
        result = subprocess.run([sys.executable, '-B', '-I', str(HERE / 'analyze.py'),
                                 '--unknown', private_sentinel], capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertNotIn(private_sentinel, result.stdout + result.stderr)

    def test_metadata_decodes_the_same_authenticated_snapshot(self):
        stream = io.BytesIO()
        np.savez(stream, offsets=np.array([0, 2, 3]), teacher_choice_label=np.array([1, 0]),
                 ops=np.array([0, 1, 0], dtype=np.uint8))
        raw = stream.getvalue()
        digest = hashlib.sha256(raw).hexdigest()
        limits = {'max_uncompressed_bytes': 4 * 1024 * 1024}
        widths, chosen = diagnostic.extract_training_metadata(raw, digest, 2, 3,
            {'DIG': 1, 'PLANT': 1}, ('DIG', 'PLANT'), limits, time.monotonic())
        np.testing.assert_array_equal(widths, [2, 1])
        np.testing.assert_array_equal(chosen, [1, 0])
        with self.assertRaises(ValueError):
            diagnostic.extract_training_metadata(raw + b'x', digest, 2, 3,
                {'DIG': 1, 'PLANT': 1}, ('DIG', 'PLANT'), limits, time.monotonic())

    def test_metadata_rejects_wrong_dtype_and_nonmonotone_offsets(self):
        cases = (np.array([0, 2, 3], dtype=np.int32),
                 np.array([0, 3, 2, 3], dtype=np.uint8),
                 np.array([0, 3, 2, 3], dtype=np.int64))
        for offsets in cases:
            with self.subTest(dtype=str(offsets.dtype), offsets=len(offsets)):
                stream = io.BytesIO()
                labels = np.zeros(len(offsets) - 1, dtype=np.int64)
                np.savez(stream, offsets=offsets, teacher_choice_label=labels,
                         ops=np.zeros(3, dtype=np.uint8))
                raw = stream.getvalue()
                with self.assertRaises(ValueError):
                    diagnostic.extract_training_metadata(raw, hashlib.sha256(raw).hexdigest(),
                        len(labels), 3, {'DIG': len(labels)}, ('DIG',),
                        {'max_uncompressed_bytes': 4 * 1024 * 1024}, time.monotonic())

    def test_metadata_header_is_checked_before_numpy_allocation(self):
        def array_bytes(array):
            stream = io.BytesIO()
            np.lib.format.write_array(stream, array, allow_pickle=False)
            return stream.getvalue()
        bad = io.BytesIO()
        np.lib.format.write_array_header_1_0(bad, {'descr': '<i8', 'fortran_order': False,
                                                   'shape': (10**12,)})
        zipped = io.BytesIO()
        with zipfile.ZipFile(zipped, 'w', compression=zipfile.ZIP_STORED) as archive:
            archive.writestr('offsets.npy', bad.getvalue())
            archive.writestr('teacher_choice_label.npy', array_bytes(np.array([0, 0], dtype=np.int64)))
            archive.writestr('ops.npy', array_bytes(np.array([0, 0, 0], dtype=np.uint8)))
        raw = zipped.getvalue()
        with mock.patch.object(diagnostic.np, 'load', side_effect=AssertionError('allocation attempted')):
            with self.assertRaises(ValueError):
                diagnostic.extract_training_metadata(raw, hashlib.sha256(raw).hexdigest(), 2, 3,
                    {'DIG': 2}, ('DIG',), {'max_uncompressed_bytes': 4 * 1024 * 1024}, time.monotonic())


if __name__ == '__main__':
    unittest.main()
