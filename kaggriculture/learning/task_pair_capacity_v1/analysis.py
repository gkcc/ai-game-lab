"""Descriptive, bounded analysis of one already recorded synthetic trace.

This is not a game runner and cannot infer a future callback or a failure cause.
Only counters are returned; no input event, identity, seed, or path is emitted.
"""
import gzip
import hashlib
import io
import json
import math

from learning.task_pair_dev_games_v1.capture import (
    MAX_COMPRESSED_BYTES, MAX_DECODED_BYTES, MAX_LINE_BYTES,
    _hash_stream, _nonfinite, _unique_pairs,
)
from learning.task_pair_dev_games_v1.panel import GAME_LIMITS


TURNS = 580
SLICES = ((0, 99), (100, 199), (200, 299), (300, 399),
          (400, 499), (500, 579))
KEYS = ('task_forwards', 'task_rows', 'crop_forwards', 'crop_rows',
        'gate_evaluations', 'controller_calls', 'teacher_proposals')
METRICS = ('task_forwards', 'task_rows', 'crop_forwards', 'crop_rows', 'total_rows')
EXPECTED_FINAL = dict(task_forwards=3241, task_rows=159200, crop_forwards=589,
                      crop_rows=589, gate_evaluations=0, controller_calls=580,
                      teacher_proposals=0)


def _counts(value):
    if (type(value) is not dict or set(value) != set(KEYS)
            or any(type(value[key]) is not int or value[key] < 0 for key in KEYS)):
        raise ValueError('Invalid counter structure')
    return {key: value[key] for key in KEYS}


def _statistics(values):
    values = sorted(values)
    n = len(values)
    if not n:
        raise ValueError('Empty predeclared time bin')
    middle = n // 2
    median = values[middle] if n % 2 else (values[middle - 1] + values[middle]) / 2
    return dict(total=sum(values), mean=sum(values) / n, median=median,
                p95_nearest_rank=values[math.ceil(.95 * n) - 1], max=values[-1])


def analyze(raw, expected_sha, expected_bytes):
    """Consume one seekable binary stream and return only fixed-bin summaries.

    The caller binds expected_sha and expected_bytes to its separately stored
    local failure evidence. Verify the stream before and after decoding.
    """
    if (type(expected_sha) is not str or len(expected_sha) != 64
            or any(char not in '0123456789abcdef' for char in expected_sha)
            or type(expected_bytes) is not int or not 0 < expected_bytes <= MAX_COMPRESSED_BYTES):
        raise ValueError('Invalid input identity')
    raw.seek(0)
    snapshot = raw.read(expected_bytes + 1)
    if (type(snapshot) is not bytes or len(snapshot) != expected_bytes
            or hashlib.sha256(snapshot).hexdigest() != expected_sha):
        raise ValueError('Input identity mismatch')
    prior = {key: 0 for key in KEYS}
    bins = []
    total_bytes = 0
    with gzip.GzipFile(fileobj=io.BytesIO(snapshot), mode='rb') as stream:
        for first, last in SLICES:
            observations = {name: [] for name in METRICS}
            for step in range(first, last + 1):
                line = stream.readline(MAX_LINE_BYTES + 1)
                total_bytes += len(line)
                if (not line or not line.endswith(b'\n') or len(line) > MAX_LINE_BYTES
                        or total_bytes > MAX_DECODED_BYTES):
                    raise ValueError('Trace length bound or newline mismatch')
                event = json.loads(line, object_pairs_hook=_unique_pairs, parse_constant=_nonfinite)
                if (type(event) is not dict or type(event.get('step')) is not int
                        or event['step'] != step
                        or any(type(event.get(key)) is not int or event[key] != 0
                               for key in ('load_errors', 'inference_errors', 'fallbacks'))):
                    raise ValueError('Trace sequence or error flag mismatch')
                cumulative = _counts(event.get('model_counts'))
                turn = _counts(event.get('turn_model_counts'))
                if (any(cumulative[key] - prior[key] != turn[key] for key in KEYS)
                        or turn['controller_calls'] != 1
                        or turn['crop_rows'] != turn['crop_forwards']
                        or turn['task_rows'] < turn['task_forwards']
                        or (turn['task_forwards'] == 0 and turn['task_rows'] != 0)
                        or turn['crop_forwards'] > turn['task_forwards']
                        or any(cumulative[key] != 0 for key in ('gate_evaluations', 'teacher_proposals'))):
                    raise ValueError('Inconsistent per-callback counters')
                if (any(cumulative[key] > GAME_LIMITS[key]
                        for key in ('task_forwards', 'crop_forwards', 'controller_calls'))
                        or cumulative['task_rows'] + cumulative['crop_rows'] > GAME_LIMITS['neural_input_rows']
                        or any(turn[key] > GAME_LIMITS['max_' + key + '_per_turn']
                               for key in ('task_forwards', 'crop_forwards', 'controller_calls'))
                        or turn['task_rows'] + turn['crop_rows'] > GAME_LIMITS['max_neural_input_rows_per_turn']):
                    raise ValueError('Recorded counters exceed the fixed limits')
                for key in METRICS[:-1]:
                    observations[key].append(turn[key])
                observations['total_rows'].append(turn['task_rows'] + turn['crop_rows'])
                prior = cumulative
            bins.append(dict(first_callback_step=first, last_callback_step=last,
                             callbacks=last - first + 1,
                             metrics={key: _statistics(observations[key]) for key in METRICS},
                             cumulative={key: prior[key] for key in KEYS},
                             remaining_input_rows=GAME_LIMITS['neural_input_rows']
                             - prior['task_rows'] - prior['crop_rows']))
        if stream.read(1):
            raise ValueError('Trace contains unexpected additional callbacks')
    if prior != EXPECTED_FINAL or _hash_stream(raw) != expected_sha:
        raise ValueError('Final counters or input identity mismatch')
    return dict(schema='task-pair-capacity-v1', recorded_callbacks=TURNS,
                callbacks_are_official_transitions=False,
                failed_callback_observed=False, future_callbacks_observed=False,
                fixed_limits=dict(GAME_LIMITS), bins=bins)
