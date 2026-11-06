"""In-memory regressions for the exact single-pass event-reader boundary."""
import gzip
import hashlib
import io
import json
import unittest

from . import runner
from .contract import LIMITS


class CountingInput(io.BytesIO):
    def __init__(self, value):
        super().__init__(value)
        self.bytes_read = 0

    def read(self, size=-1):
        result = super().read(size)
        self.bytes_read += len(result)
        return result

    def seek(self, *args):
        raise AssertionError('Reader must not rewind')


class StreamTests(unittest.TestCase):
    def make(self, payload=b'{"step":1}\n{"step":2}\n', *, raw=None, expected=None, limits=None, guard=lambda:None):
        raw = gzip.compress(payload, mtime=0) if raw is None else raw
        source = CountingInput(raw)
        expected = expected or dict(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
        stream = runner.EventStream(source, expected, limits or LIMITS,
            {'decoded_bytes':0}, json.loads, guard)
        return source, stream

    def test_one_forward_pass_is_complete_and_cannot_repeat(self):
        source, stream = self.make()
        self.assertEqual(list(stream), [{'step':1}, {'step':2}])
        receipt = stream.receipt()
        self.assertEqual(receipt['lines'], 2)
        self.assertEqual(source.bytes_read, receipt['compressed_bytes'])
        self.assertTrue(receipt['crc_eof_verified'])
        with self.assertRaises(ValueError):
            list(stream)

    def test_partial_consumer_cannot_obtain_success_receipt(self):
        _, stream = self.make()
        values = iter(stream)
        self.assertEqual(next(values), {'step':1})
        with self.assertRaises(ValueError):
            stream.receipt()
        values.close()

    def test_corrupt_crc_and_truncation_never_complete(self):
        raw = gzip.compress(b'{"step":1}\n', mtime=0)
        for bad in (raw[:-5], raw[:-8]+bytes([raw[-8]^1])+raw[-7:]):
            with self.subTest(length=len(bad)):
                _, stream = self.make(raw=bad)
                with self.assertRaises((EOFError, gzip.BadGzipFile)):
                    list(stream)
                with self.assertRaises(ValueError):
                    stream.receipt()

    def test_hash_mismatch_rejects_even_valid_gzip(self):
        raw = gzip.compress(b'{"step":1}\n', mtime=0)
        _, stream = self.make(raw=raw, expected=dict(bytes=len(raw), sha256='0'*64))
        with self.assertRaises(ValueError):
            list(stream)
        self.assertFalse(stream.complete)

    def test_output_bounds_and_missing_newline_reject(self):
        for limits, payload in (
            (dict(LIMITS, max_line_bytes=3), b'{"step":1}\n'),
            (dict(LIMITS, max_decoded_stream_bytes=3), b'{"step":1}\n'),
            (dict(LIMITS, max_decoded_total_bytes=3), b'{"step":1}\n'),
            (LIMITS, b'{"step":1}')):
            with self.subTest(limits=limits, newline=payload.endswith(b'\n')):
                _, stream = self.make(payload, limits=limits)
                with self.assertRaises(ValueError):
                    list(stream)
                self.assertFalse(stream.complete)

    def test_guard_exception_is_primary_and_no_receipt(self):
        def failed():
            raise MemoryError('synthetic floor')
        _, stream = self.make(guard=failed)
        with self.assertRaises(MemoryError):
            list(stream)
        self.assertFalse(stream.complete)

    def test_extra_compressed_bytes_not_ignored(self):
        original = gzip.compress(b'{"step":1}\n', mtime=0)
        _, stream = self.make(raw=original+b'\0', expected=dict(bytes=len(original),
                    sha256=hashlib.sha256(original).hexdigest()))
        with self.assertRaises(ValueError):
            list(stream)
        self.assertFalse(stream.complete)

    def test_noninteger_byte_limit_rejected(self):
        for value in (True, 1.0):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.make(expected=dict(bytes=value, sha256='0'*64))


if __name__ == '__main__':
    unittest.main()
