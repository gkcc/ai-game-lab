"""Synthetic checks for bounded, redacted counter aggregation."""
import copy
import gzip
import hashlib
import io
import json
import unittest
from contextlib import contextmanager
from unittest.mock import patch

from learning.task_pair_capacity_v1 import analysis


def _events():
    total = {key: 0 for key in analysis.KEYS}
    task_calls = 0
    events = []
    for step in range(analysis.TURNS):
        turn = {key: 0 for key in analysis.KEYS}
        turn['controller_calls'] = 1
        turn['task_forwards'] = 6 if step < 341 else 5
        for _ in range(turn['task_forwards']):
            turn['task_rows'] += 50 if task_calls < 391 else 49
            task_calls += 1
        turn['crop_forwards'] = turn['crop_rows'] = 2 if step < 9 else 1
        total = {key: total[key] + turn[key] for key in analysis.KEYS}
        events.append(dict(step=step, model_counts=dict(total), turn_model_counts=turn,
                           load_errors=0, inference_errors=0, fallbacks=0,
                           private_field='must-not-be-emitted'))
    return events


def _raw(events):
    payload = b''.join(json.dumps(event).encode() + b'\n' for event in events)
    compressed = gzip.compress(payload, mtime=0)
    return io.BytesIO(compressed), hashlib.sha256(compressed).hexdigest(), len(compressed)


class AnalysisTests(unittest.TestCase):
    def test_aggregates_fixed_bins_without_emitting_event_fields(self):
        result = analysis.analyze(*_raw(_events()))
        self.assertEqual(result['recorded_callbacks'], 580)
        self.assertEqual([item['callbacks'] for item in result['bins']], [100] * 5 + [80])
        self.assertEqual(result['bins'][-1]['cumulative'], analysis.EXPECTED_FINAL)
        self.assertFalse(result['callbacks_are_official_transitions'])
        self.assertNotIn('must-not-be-emitted', json.dumps(result))
        self.assertNotIn('private_field', json.dumps(result))

    def test_rejects_sequence_count_and_error_corruption(self):
        events = _events()
        changes = []
        bad = copy.deepcopy(events); bad[12]['step'] = 13; changes.append(bad)
        bad = copy.deepcopy(events); bad[12]['turn_model_counts']['task_rows'] += 1; changes.append(bad)
        bad = copy.deepcopy(events); bad[12]['fallbacks'] = 1; changes.append(bad)
        bad = copy.deepcopy(events); bad[12]['model_counts']['task_rows'] = True; changes.append(bad)
        bad = copy.deepcopy(events); bad.append(copy.deepcopy(bad[-1])); changes.append(bad)
        bad = copy.deepcopy(events); bad.pop(); changes.append(bad)
        for replacement in (0, 1):
            bad = copy.deepcopy(events)
            moved = bad[0]['turn_model_counts']['task_forwards'] - replacement
            bad[0]['turn_model_counts']['task_forwards'] = replacement
            bad[1]['turn_model_counts']['task_forwards'] += moved
            running = 0
            for event in bad:
                running += event['turn_model_counts']['task_forwards']
                event['model_counts']['task_forwards'] = running
            changes.append(bad)
        for bad in changes:
            with self.subTest(), self.assertRaises(ValueError):
                analysis.analyze(*_raw(bad))

    def test_rejects_wrong_identity(self):
        stream, _digest, size = _raw(_events())
        with self.assertRaises(ValueError):
            analysis.analyze(stream, '0' * 64, size)
        stream, digest, size = _raw(_events())
        with self.assertRaises(ValueError):
            analysis.analyze(stream, digest, size + 1)

    def test_rejects_duplicate_keys_nonfinite_and_missing_newline(self):
        events = _events()
        lines = [json.dumps(event).encode() for event in events]
        cases = [
            b'\n'.join([lines[0].replace(b'"step": 0', b'"step": 0, "step": 0'), *lines[1:]]) + b'\n',
            b'\n'.join([lines[0].replace(b'"must-not-be-emitted"', b'NaN'), *lines[1:]]) + b'\n',
            b'\n'.join(lines),
        ]
        for payload in cases:
            compressed = gzip.compress(payload, mtime=0)
            with self.subTest(), self.assertRaises(ValueError):
                analysis.analyze(io.BytesIO(compressed), hashlib.sha256(compressed).hexdigest(), len(compressed))

    def test_rejects_line_and_decoded_size_excess(self):
        for name, bound in (('MAX_LINE_BYTES', 20), ('MAX_DECODED_BYTES', 20)):
            with self.subTest(name=name), patch.object(analysis, name, bound), self.assertRaises(ValueError):
                analysis.analyze(*_raw(_events()))

    def test_stream_switch_during_decode_cannot_change_verified_snapshot(self):
        events = _events()
        original, digest, size = _raw(events)
        original_bytes = original.getvalue()
        expected = analysis.analyze(io.BytesIO(original_bytes), digest, size)
        altered = copy.deepcopy(events)
        altered[0]['turn_model_counts']['task_rows'] -= 40
        altered[100]['turn_model_counts']['task_rows'] += 40
        running = 0
        for event in altered:
            running += event['turn_model_counts']['task_rows']
            event['model_counts']['task_rows'] = running
        other, other_digest, other_size = _raw(altered)
        self.assertNotEqual(expected, analysis.analyze(other, other_digest, other_size))
        other_bytes = other.getvalue()
        real_gzip = gzip.GzipFile

        @contextmanager
        def switch_source(*args, **kwargs):
            original.seek(0); original.truncate(); original.write(other_bytes); original.seek(0)
            with real_gzip(*args, **kwargs) as stream:
                try:
                    yield stream
                finally:
                    original.seek(0); original.truncate(); original.write(original_bytes); original.seek(0)

        with patch.object(analysis.gzip, 'GzipFile', side_effect=switch_source):
            self.assertEqual(analysis.analyze(original, digest, size), expected)

    def test_statistics_use_nearest_rank(self):
        self.assertEqual(analysis._statistics([1, 2, 3, 4]),
                         dict(total=10, mean=2.5, median=2.5, p95_nearest_rank=4, max=4))


if __name__ == '__main__':
    unittest.main()
