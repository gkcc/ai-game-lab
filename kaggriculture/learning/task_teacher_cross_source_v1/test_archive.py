"""In-memory synthetic archive tests; no private data, model, GPU or games."""
from contextlib import contextmanager
import hashlib
import io
import struct
import warnings
import zipfile
import unittest
from unittest.mock import patch

import numpy as np

from learning.task_teacher_cross_source_v1 import archive as a


def payload(*, groups=3, widths=(2, 1, 3)):
    candidates = sum(widths)
    arrays = dict(shared=np.arange(groups * 275, dtype=np.float32).reshape(groups, 275),
        candidates=np.arange(candidates * 98, dtype=np.float32).reshape(candidates, 98),
        offsets=np.array([0, *np.cumsum(widths)], dtype=np.int64),
        teacher_choice_label=np.zeros(groups, dtype=np.int64),
        ops=np.zeros(candidates, dtype=np.uint8), seed=np.zeros(groups, dtype=np.uint32))
    return arrays


def encode(arrays):
    out = io.BytesIO()
    np.savez(out, **arrays)
    return out.getvalue()


def members(raw):
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        return [(entry.filename, z.read(entry)) for entry in z.infolist()]


def encode_members(items, compression=zipfile.ZIP_STORED):
    out = io.BytesIO()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        with zipfile.ZipFile(out, 'w', compression=compression) as z:
            for name, raw in items:
                z.writestr(name, raw)
    return out.getvalue()


class ArchiveTests(unittest.TestCase):
    def scan(self, raw, *, groups=3, candidates=6, consume=lambda *_: None, guard=lambda: None,
             digest=None, stream=None):
        stream = io.BytesIO(raw) if stream is None else stream
        @contextmanager
        def opener(_path):
            yield stream, len(raw)
        expected = digest or hashlib.sha256(raw).hexdigest()
        with patch.object(a, '_open_regular', opener):
            return a.scan_archive('synthetic', expected_sha256=expected, expected_groups=groups,
                                  expected_candidates=candidates, guard=guard, consume=consume)

    def test_only_four_payloads_are_read_and_each_complete_group_is_immutable(self):
        arrays = payload()
        rows, decoded = [], []
        original = a._read_payload
        def tracked(z, info, shape, dtype, guard):
            decoded.append(info.filename)
            return original(z, info, shape, dtype, guard)
        def consume(index, shared, candidates, target):
            self.assertFalse(shared.flags.writeable)
            self.assertFalse(candidates.flags.writeable)
            with self.assertRaises(ValueError): shared.setflags(write=True)
            with self.assertRaises(ValueError): candidates.setflags(write=True)
            start, end = arrays['offsets'][index:index + 2]
            np.testing.assert_array_equal(shared, arrays['shared'][index])
            np.testing.assert_array_equal(candidates, arrays['candidates'][start:end])
            rows.append((index, len(candidates), target))
        with patch.object(a, '_read_payload', tracked):
            result = self.scan(encode(arrays), consume=consume)
        self.assertEqual(decoded, ['shared.npy', 'candidates.npy', 'offsets.npy', 'teacher_choice_label.npy'])
        self.assertEqual(rows, [(0, 2, 0), (1, 1, 0), (2, 3, 0)])
        self.assertEqual(result, dict(groups=3, candidates=6, verified_after_scan=True))

    def test_bound_sha_group_count_and_structural_failure_precede_callback(self):
        raw = encode(payload())
        for kwargs in (dict(digest='0' * 64), dict(groups=4), dict(candidates=7)):
            with self.subTest(condition=kwargs), self.assertRaises(a.ArchiveError):
                self.scan(raw, consume=lambda *_: self.fail('callback reached'), **kwargs)

    def test_all_headers_checked_before_any_array_is_allocated(self):
        raw = encode(payload())
        entries = members(raw)
        header = io.BytesIO()
        np.lib.format.write_array_header_1_0(header,
            dict(descr='<u4', fortran_order=False, shape=(10**12,)))
        bad = encode_members([(name, header.getvalue() if name == 'seed.npy' else content)
                              for name, content in entries])
        with patch.object(a, '_read_payload', side_effect=AssertionError('allocation reached')):
            with self.assertRaises(a.ArchiveError): self.scan(bad)

    def test_malformed_header_never_leaks_its_content_in_exception(self):
        # NumPy includes unexpected header keys in ValueError messages. The
        # archive boundary must replace those messages, without reaching the
        # consumer or leaking the private synthetic marker.
        marker = 'SYNTHETIC_PRIVATE_HEADER_MARKER'
        header = ("{'" + marker + "': 1}\n").encode('ascii')
        bad_header = b'\x93NUMPY\x01\x00' + struct.pack('<H', len(header)) + header
        entries = members(encode(payload()))
        for target in ('seed.npy', 'shared.npy'):
            with self.subTest(target=target):
                bad = encode_members([(name, bad_header if name == target else content)
                                      for name, content in entries])
                with self.assertRaises(a.ArchiveError) as caught:
                    self.scan(bad, consume=lambda *_: self.fail('callback reached'))
                self.assertNotIn(marker, str(caught.exception))
                self.assertTrue(caught.exception.__suppress_context__)

    def test_declared_header_limit_is_checked_before_header_read(self):
        # NumPy's max_header_size is enforced only AFTER reading the declared
        # header. Bound that read before giving the header to NumPy.
        header = b' ' * (a.HEADER_LIMIT + 1)
        bad_header = b'\x93NUMPY\x02\x00' + struct.pack('<I', len(header)) + header
        entries = members(encode(payload()))
        bad = encode_members([(name, bad_header if name == 'seed.npy' else content)
                              for name, content in entries])
        original = zipfile.ZipExtFile.read
        requested = []
        def tracked(member, size=-1):
            if getattr(member, 'name', None) == 'seed.npy':
                requested.append(size)
            return original(member, size)
        with patch.object(zipfile.ZipExtFile, 'read', tracked), self.assertRaises(a.ArchiveError):
            self.scan(bad, consume=lambda *_: self.fail('callback reached'))
        self.assertTrue(all(0 <= size <= a.HEADER_LIMIT for size in requested))

    def test_rejects_duplicate_missing_extra_and_compressed_members(self):
        entries = members(encode(payload()))
        invalid = [encode_members(entries + [entries[0]]), encode_members(entries[:-1]),
                   encode_members(entries + [('extra.npy', b'x')]),
                   encode_members(entries, compression=zipfile.ZIP_DEFLATED)]
        for raw in invalid:
            with self.subTest(case=len(raw)), self.assertRaises(a.ArchiveError): self.scan(raw)

    def test_invalid_shapes_dtypes_fortran_offsets_labels_and_nonfinite_rejected(self):
        changes = []
        for key, transform in (
            ('shared', lambda x: x.astype(np.float64)),
            ('shared', lambda x: np.asfortranarray(x)),
            ('candidates', lambda x: x[:, :-1].copy()),
            ('offsets', lambda x: np.array([0, 2, 1, 6], dtype=np.int64)),
            ('offsets', lambda x: np.array([0, 2, 3, 5], dtype=np.int64)),
            ('teacher_choice_label', lambda x: np.array([2, 0, 0], dtype=np.int64)),
            ('ops', lambda x: x.astype(np.int16)),
            ('seed', lambda x: x.astype(np.int64)),
        ):
            arrays = payload(); arrays[key] = transform(arrays[key]); changes.append(arrays)
        arrays = payload(); arrays['candidates'][0, 0] = np.nan; changes.append(arrays)
        for arrays in changes:
            with self.subTest(case=len(changes)), self.assertRaises(a.ArchiveError):
                self.scan(encode(arrays), consume=lambda *_: self.fail('callback reached'))

    def test_width_above_limit_is_rejected_and_never_truncated(self):
        # Total width remains legal; only the first individual group violates
        # the per-group bound, so the scanner must inspect the offsets.
        arrays = payload(groups=2, widths=(257, 1))
        with self.assertRaises(a.ArchiveError):
            self.scan(encode(arrays), groups=2, candidates=258, consume=lambda *_: self.fail('callback reached'))

    def test_rehash_detects_change_to_unread_member_after_consumption(self):
        raw = encode(payload())
        stream = io.BytesIO(raw)
        visited = []
        def consume(index, *_):
            visited.append(index)
            if index == 0:
                stream.seek(0); stream.write(b'X')
        with self.assertRaises(a.ArchiveError): self.scan(raw, stream=stream, consume=consume)
        self.assertEqual(visited, [0, 1, 2])

    def test_guard_and_callback_exceptions_propagate_and_release_gate(self):
        raw = encode(payload())
        class Stop(Exception): pass
        for phase in ('guard', 'callback'):
            def guard():
                if phase == 'guard': raise Stop()
            def callback(*_):
                if phase == 'callback': raise Stop()
            with self.subTest(phase=phase), self.assertRaises(Stop):
                self.scan(raw, guard=guard, consume=callback)
            self.assertTrue(self.scan(raw)['verified_after_scan'])

    def test_standard_guard_and_callback_errors_are_not_reclassified(self):
        raw = encode(payload())
        for phase in ('guard', 'callback'):
            for error in (EOFError, UnicodeError, OverflowError, zipfile.BadZipFile):
                signal = error('synthetic')
                def guard():
                    if phase == 'guard': raise signal
                def callback(*_):
                    if phase == 'callback': raise signal
                with self.subTest(phase=phase, error=error.__name__):
                    with self.assertRaises(error) as caught:
                        self.scan(raw, guard=guard, consume=callback)
                    self.assertIs(caught.exception, signal)
                    self.assertTrue(self.scan(raw)['verified_after_scan'])

    def test_reentrant_scan_rejected_without_interrupting_outer_scan(self):
        raw = encode(payload())
        seen = []
        def consume(index, *_):
            with self.assertRaises(a.ArchiveError): self.scan(raw)
            seen.append(index)
        self.assertTrue(self.scan(raw, consume=consume)['verified_after_scan'])
        self.assertEqual(len(seen), 3)


if __name__ == '__main__':
    unittest.main()
