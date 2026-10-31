"""Assemble a private, conflict-aware dataset from authenticated comparisons.

The caller authenticates the source, comparison summaries and normalization
checkpoint. This module reads no replay or file and does not train a model.
All feature arrays, coordinates and fingerprints in returned examples are
private; only the aggregate summary is suitable for publication after the
project's usual disclosure review. A one-step difference is not a value label.
"""
from collections import Counter
from dataclasses import dataclass, replace
import hashlib
import struct

import numpy as np

from learning.task_choice_v1.features import SCHEMA_VERSION, schema_hash
from learning.task_student_continuation_v1.analysis import _result
from learning.task_student_continuation_v1.samples import OPPONENTS, ROLES
from .sampling import WINDOWS


SHARED_DIM = 275
CANDIDATE_DIM = 98
DIM = SHARED_DIM + CANDIDATE_DIM
TRAIN_ROLES = ROLES[:2]
DEV_ROLES = ROLES[2:]
_EXAMPLE_TOKEN = object()  # In-process contract guard, not provenance authentication.


@dataclass(frozen=True, repr=False)
class PrivateExample:
    role: str
    opponent: str
    side: int
    window: int
    step: int
    ordinal: int
    features: np.ndarray
    teacher_index: int
    student_index: int
    third_index: int
    _float64_bytes: bytes
    _schema: tuple
    _input32_fingerprint: bytes = b''
    _deployment_fingerprint: bytes = b''
    _training_fingerprint: bytes = b''
    _validated: object = None

    def __repr__(self):
        return '<PrivateTaskExample>'


def _numeric_vector(value, length):
    try:
        # NumPy would silently coerce a mixed bool/float list to floats.
        if isinstance(value, (list, tuple)) and any(
                isinstance(item, (bool, np.bool_)) for item in value):
            raise ValueError('Invalid numeric vector')
        data = np.asarray(value)
        if data.shape != (length,) or data.dtype.kind not in 'iuf':
            raise ValueError('Invalid numeric vector')
        converted = np.asarray(data, dtype=np.float64)
    except (TypeError, ValueError, OverflowError):
        raise ValueError('Invalid numeric vector') from None
    if not np.isfinite(converted).all():
        raise ValueError('Nonfinite numeric vector')
    return converted


def _names(value, length):
    if (type(value) is not list or len(value) != length
            or any(type(name) is not str or not name for name in value)
            or len(set(value)) != length):
        raise ValueError('Invalid feature schema')
    return tuple(value)


def _canonical(value, dtype):
    """Copy to fixed byte order and normalize negative zero, preserving order."""
    with np.errstate(over='ignore', invalid='ignore'):
        array = np.array(value, dtype=dtype, order='C', copy=True)
    if not np.isfinite(array).all():
        raise ValueError('Nonfinite model input')
    array[array == 0] = 0
    return array


def _key(array):
    # Candidate count, column count, order and dtype all affect the key.
    return (array.dtype.str, array.shape, array.tobytes(order='C'))


def _digest(domain, key):
    dtype, shape, data = key
    return hashlib.sha256(domain + dtype.encode('ascii') + b'\x00'
                          + struct.pack('>II', *shape) + data).digest()


def make_example(*, role, opponent, side, window, step, ordinal, inputs,
                 teacher_index, student_index, third_index, teacher_summary,
                 third_summary):
    """Return a private eligible example, or None for an unverified contrast.

    Summary validation is shared with the preceding official-transition
    diagnostic. The teacher must reproduce the factual state and the third
    candidate must change the actual next physical state. Validation failures
    raise rather than becoming invisible exclusions. Inputs may contain private
    observation fields; this function retains only numeric model features.
    """
    if (role not in ROLES or opponent not in OPPONENTS
            or type(side) is not int or side not in (0, 1)
            or type(window) is not int or not 0 <= window < len(WINDOWS)
            or type(step) is not int or not WINDOWS[window][0] <= step <= WINDOWS[window][1]
            or type(ordinal) is not int or ordinal < 0):
        raise ValueError('Invalid private example coordinate')
    if type(inputs) is not dict or inputs.get('schema_version') != SCHEMA_VERSION:
        raise ValueError('Invalid task input')
    shared_names = _names(inputs.get('shared_feature_names'), SHARED_DIM)
    candidate_names = _names(inputs.get('feature_names'), CANDIDATE_DIM)
    if inputs.get('schema_sha256') != schema_hash(candidate_names, shared_names):
        raise ValueError('Task schema identity differs')
    shared = _numeric_vector(inputs.get('shared_x'), SHARED_DIM)
    candidates = inputs.get('candidates')
    if type(candidates) is not list or not 2 <= len(candidates) <= 256:
        raise ValueError('Invalid candidate count')
    vectors, ids = [], []
    for candidate in candidates:
        if type(candidate) is not dict or type(candidate.get('id')) is not str or not candidate['id']:
            raise ValueError('Invalid candidate identity')
        ids.append(candidate['id'])
        vectors.append(_numeric_vector(candidate.get('x'), CANDIDATE_DIM))
    if len(set(ids)) != len(ids):
        raise ValueError('Duplicate candidate identity')
    n = len(candidates)
    if (type(teacher_index) is not int or not 0 <= teacher_index < n
            or type(student_index) is not int or not 0 <= student_index < n
            or teacher_index == student_index):
        raise ValueError('Invalid teacher or student index')
    if third_index is not None and (type(third_index) is not int
            or not 0 <= third_index < n or third_index in (teacher_index, student_index)):
        raise ValueError('Invalid third index')
    if (third_index is None) != (third_summary is None):
        raise ValueError('Third candidate and comparison must agree')

    teacher = _result(teacher_summary)
    third = _result(third_summary) if third_summary is not None else None
    if (teacher['status'] != 'same' or third is None or third['status'] != 'different'
            or third['official']['state_changed'] is not True):
        return None

    joined = np.concatenate((np.broadcast_to(shared, (n, SHARED_DIM)), np.stack(vectors)), axis=1)
    raw64 = _canonical(joined, '<f8')
    raw32 = _canonical(joined, '<f4')
    # Immutable backing prevents later mutation through an ndarray view.
    raw64_bytes, raw32_bytes = raw64.tobytes(), raw32.tobytes()
    features = np.frombuffer(raw32_bytes, dtype='<f4').reshape(n, DIM)
    return PrivateExample(role, opponent, side, window, step, ordinal, features,
                          teacher_index, student_index, third_index, raw64_bytes,
                          (shared_names, candidate_names),
                          _digest(b'raw-float32\x00', _key(raw32)),
                          _validated=_EXAMPLE_TOKEN)


def _coordinate(example):
    # Fixed order independent of caller iteration, private result or payoff.
    return (ROLES.index(example.role), OPPONENTS.index(example.opponent),
            example.side, example.window, example.step, example.ordinal)


def _validate_example(example):
    """Reject accidental fabrication or alteration between assembly stages.

    Source authenticity remains the calling collector's responsibility; this
    in-process token and consistency check are not a security boundary.
    """
    if not isinstance(example, PrivateExample) or example._validated is not _EXAMPLE_TOKEN:
        raise ValueError('Unverified dataset example')
    if (example.role not in ROLES or example.opponent not in OPPONENTS
            or type(example.side) is not int or example.side not in (0, 1)
            or type(example.window) is not int or not 0 <= example.window < len(WINDOWS)
            or type(example.step) is not int
            or not WINDOWS[example.window][0] <= example.step <= WINDOWS[example.window][1]
            or type(example.ordinal) is not int or example.ordinal < 0):
        raise ValueError('Invalid private example coordinate')
    x = example.features
    if (not isinstance(x, np.ndarray) or x.dtype != np.dtype('float32')
            or x.ndim != 2 or x.shape[1] != DIM or not 2 <= x.shape[0] <= 256
            or x.flags.writeable or not np.isfinite(x).all()
            or type(example._float64_bytes) is not bytes
            or len(example._float64_bytes) != x.size * 8):
        raise ValueError('Invalid private example features')
    labels = (example.teacher_index, example.student_index, example.third_index)
    if (any(type(index) is not int or not 0 <= index < len(x) for index in labels)
            or len(set(labels)) != 3):
        raise ValueError('Invalid private example labels')
    raw64 = np.frombuffer(example._float64_bytes, dtype='<f8').reshape(x.shape)
    if (not np.isfinite(raw64).all()
            or _key(_canonical(raw64, '<f4')) != _key(_canonical(x, '<f4'))
            or example._input32_fingerprint != _digest(b'raw-float32\x00',
                                                        _key(_canonical(x, '<f4')))):
        raise ValueError('Private feature copy differs')


def build_dataset(examples, *, mean, scale):
    """Deduplicate by exact actual float32 inputs and apply frozen coverage gates.

    The passed normalization belongs to the authenticated frozen checkpoint.
    Deployment's float64 and training's float32 normalized fingerprints are
    kept on the private examples only. Equality of raw OR normalized float32
    inputs is closed transitively; conflicting label triples remove the entire
    component. Training wins a cross-split duplicate; coordinates break all
    other ties. No outcome or model performance participates in selection.
    """
    if type(examples) not in (tuple, list) or len(examples) > 96:
        raise ValueError('Invalid bounded example collection')
    mu, sigma = _numeric_vector(mean, DIM), _numeric_vector(scale, DIM)
    if not (sigma >= 1).all():
        raise ValueError('Invalid checkpoint normalization')
    accepted = []
    for example in examples:
        if example is None:
            continue
        _validate_example(example)
        accepted.append(example)
    accepted.sort(key=_coordinate)
    if len({_coordinate(item) for item in accepted}) != len(accepted):
        raise ValueError('Duplicate sample coordinate')
    if accepted and any(item._schema != accepted[0]._schema for item in accepted):
        raise ValueError('Inconsistent task feature schema')

    n = len(accepted)
    parents = list(range(n))
    keys64, keys_raw32, keys_network32 = [], [], []
    by_raw32, by_network32 = {}, {}

    def root(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i

    def unite(a, b):
        a, b = root(a), root(b)
        if a != b:
            parents[b] = a

    enriched = []
    for i, item in enumerate(accepted):
        shape = item.features.shape
        raw64 = np.frombuffer(item._float64_bytes, dtype='<f8').reshape(shape)
        with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
            deploy64 = _canonical((raw64 - mu) / sigma, '<f8')
            # Match train._batch exactly: float32 raw -> float64 normalize -> float32.
            network32 = _canonical((item.features.astype(np.float64) - mu) / sigma, '<f4')
        raw32 = _canonical(item.features, '<f4')
        key64, key_raw, key_network = _key(deploy64), _key(raw32), _key(network32)
        keys64.append(key64)
        keys_raw32.append(key_raw)
        keys_network32.append(key_network)
        enriched.append(replace(item,
            _deployment_fingerprint=_digest(b'deployment-float64\x00', key64),
            _training_fingerprint=_digest(b'training-float32\x00', key_network)))
        for lookup, key in ((by_raw32, key_raw), (by_network32, key_network)):
            if key in lookup:
                unite(i, lookup[key])
            else:
                lookup[key] = i

    components = {}
    for i in range(n):
        components.setdefault(root(i), []).append(i)
    train, dev = [], []
    counts = Counter()
    for members in components.values():
        labels = {(accepted[i].teacher_index, accepted[i].student_index,
                   accepted[i].third_index) for i in members}
        if len({keys64[i] for i in members}) > 1:
            counts['deployment_precision_collision_groups'] += 1
        if len({keys_raw32[i] for i in members}) > 1:
            counts['normalization_collision_groups'] += 1
        if len(labels) != 1:
            counts['conflict_groups'] += 1
            counts['conflict_examples_removed'] += len(members)
            continue
        training = [i for i in members if accepted[i].role in TRAIN_ROLES]
        development = [i for i in members if accepted[i].role in DEV_ROLES]
        if training:
            train.append(enriched[training[0]])
            counts['same_split_train_removed'] += len(training) - 1
            if development:
                counts['cross_split_groups'] += 1
                counts['cross_split_dev_removed'] += len(development)
        elif development:
            dev.append(enriched[development[0]])
            counts['same_split_dev_removed'] += len(development) - 1

    train.sort(key=_coordinate)
    dev.sort(key=_coordinate)
    roles = Counter(item.role for item in train + dev)
    train_windows = len({item.window for item in train})
    dev_windows = len({item.window for item in dev})
    train_opponents = len({item.opponent for item in train})
    dev_opponents = len({item.opponent for item in dev})
    passed = (len(train) >= 16 and all(roles[r] >= 4 for r in TRAIN_ROLES)
              and len(dev) >= 8 and all(roles[r] >= 2 for r in DEV_ROLES)
              and min(train_windows, dev_windows, train_opponents, dev_opponents) >= 2)
    summary = dict(status='passed' if passed else 'failed', source_count=len(examples),
                   eligible_count=n, ineligible_count=len(examples)-n,
                   train_groups=len(train), dev_groups=len(dev),
                   train_by_role={role: roles[role] for role in TRAIN_ROLES},
                   dev_by_role={role: roles[role] for role in DEV_ROLES},
                   train_windows=train_windows, dev_windows=dev_windows,
                   train_opponents=train_opponents, dev_opponents=dev_opponents,
                   training=False, independent_evaluation=False)
    for name in ('conflict_groups', 'conflict_examples_removed', 'same_split_train_removed',
                 'same_split_dev_removed', 'cross_split_groups', 'cross_split_dev_removed',
                 'deployment_precision_collision_groups', 'normalization_collision_groups'):
        summary[name] = counts[name]
    return tuple(train), tuple(dev), summary
