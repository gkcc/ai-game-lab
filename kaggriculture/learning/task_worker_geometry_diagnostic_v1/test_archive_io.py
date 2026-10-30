"""In-memory tests for selective authenticated training archive decoding."""
import hashlib
import importlib.util
import io
from pathlib import Path
import unittest
from unittest import mock
import zipfile

import numpy as np

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('worker_archive_io', HERE / 'archive_io.py')
archive_io = importlib.util.module_from_spec(spec)
spec.loader.exec_module(archive_io)


def archive_bytes(wrong_dtype=False):
    stream = io.BytesIO()
    np.savez(stream, shared=np.zeros((1, 275), dtype=np.float32),
             candidates=np.zeros((2, 98), dtype=np.float32), offsets=np.array([0, 2], dtype=np.int64),
             teacher_choice_label=np.array([0], dtype=np.int64),
             ops=np.array([0, 1], dtype=np.int64 if wrong_dtype else np.uint8),
             seed=np.array([object()], dtype=object))
    return stream.getvalue()


class ArchiveTest(unittest.TestCase):
    def test_only_allowed_arrays_are_decoded(self):
        raw = archive_bytes()
        result = archive_io.decode_arrays(raw, hashlib.sha256(raw).hexdigest(), 1, 2, 4 * 1024**2)
        self.assertEqual(set(result), {'shared', 'candidates', 'offsets', 'teacher_choice_label', 'ops'})

    def test_wrong_hash_and_dtype_fail_closed(self):
        raw = archive_bytes()
        with self.assertRaises(ValueError):
            archive_io.decode_arrays(raw + b'x', hashlib.sha256(raw).hexdigest(), 1, 2, 4 * 1024**2)
        raw = archive_bytes(wrong_dtype=True)
        with mock.patch.object(archive_io.np, 'load', side_effect=AssertionError('decoded before validation')):
            with self.assertRaises(ValueError):
                archive_io.decode_arrays(raw, hashlib.sha256(raw).hexdigest(), 1, 2, 4 * 1024**2)

    def test_oversized_array_header_rejected_before_allocation(self):
        raw = archive_bytes()
        old = io.BytesIO(raw)
        new = io.BytesIO()
        forged = io.BytesIO()
        np.lib.format.write_array_header_1_0(forged, {'descr': '<f4', 'fortran_order': False,
                                                      'shape': (10**12, 275)})
        with zipfile.ZipFile(old) as source, zipfile.ZipFile(new, 'w', zipfile.ZIP_STORED) as target:
            for item in source.infolist():
                target.writestr(item.filename, forged.getvalue() if item.filename == 'shared.npy'
                                else source.read(item.filename))
        raw = new.getvalue()
        with mock.patch.object(archive_io.np, 'load', side_effect=AssertionError('allocation attempted')):
            with self.assertRaises(ValueError):
                archive_io.decode_arrays(raw, hashlib.sha256(raw).hexdigest(), 1, 2, 4 * 1024**2)


if __name__ == '__main__':
    unittest.main()
