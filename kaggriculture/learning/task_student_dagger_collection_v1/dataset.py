"""Private, seed-role-separated conditional demonstrations from consumed games.

Inputs and comparison summaries must already be authenticated by the caller.
This module reads no files, runs no model or game, and does no training. A third
candidate causing a different official next state is a behavioral contrast, not
a finding that it has lower value. Only the aggregate summary may be considered
for publication after the project's normal disclosure checks.
"""
from collections import Counter
from dataclasses import dataclass, replace

import numpy as np

from learning.task_choice_v1.features import SCHEMA_VERSION, schema_hash
from learning.task_student_continuation_v1.analysis import _result
from learning.task_student_continuation_v1.samples import OPPONENTS
from learning.task_student_collection_v1.dataset import (
    CANDIDATE_DIM, DIM, SHARED_DIM, _canonical, _digest, _key, _names, _numeric_vector,
)
from learning.task_student_collection_v1.sampling import WINDOWS


TRAIN_ROLES = ('T01', 'T02', 'T03', 'T04')
DEV_ROLES = ('V01', 'V02')
ROLES = TRAIN_ROLES + DEV_ROLES
MAX_WINDOWS = len(ROLES) * len(OPPONENTS) * 2 * len(WINDOWS)
_TOKEN = object()  # In-process assembly guard; not provenance authentication.


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
    _input32_fingerprint: bytes
    _deployment_fingerprint: bytes = b''
    _training_fingerprint: bytes = b''
    _validated: object = None

    def __repr__(self):
        return '<PrivateDaggerTaskExample>'


def _coordinate(role, opponent, side, window, step, ordinal):
    if (role not in ROLES or opponent not in OPPONENTS
            or type(side) is not int or side not in (0, 1)
            or type(window) is not int or not 0 <= window < len(WINDOWS)
            or type(step) is not int or not WINDOWS[window][0] <= step <= WINDOWS[window][1]
            or type(ordinal) is not int or ordinal < 0):
        raise ValueError('Invalid private example coordinate')
    return (ROLES.index(role), OPPONENTS.index(opponent), side, window, step, ordinal)


def make_example(*, role, opponent, side, window, step, ordinal, inputs,
                 teacher_index, student_index, third_index, teacher_summary, third_summary):
    """Return an eligible private example or None for an unverified contrast.

    An invalid or internally contradictory input raises. A well-formed unknown,
    missing contrast, or comparison that does not establish the registered
    physical difference returns None. Only numeric model features are retained;
    private raw observations, IDs and comparison documents are not retained.
    """
    _coordinate(role, opponent, side, window, step, ordinal)
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
    if third_index is not None and (type(third_index) is not int or not 0 <= third_index < n
                                    or third_index in (teacher_index, student_index)):
        raise ValueError('Invalid third index')
    if (third_index is None) != (third_summary is None):
        raise ValueError('Third candidate and comparison must agree')
    teacher = _result(teacher_summary)
    third = _result(third_summary) if third_summary is not None else None
    if (teacher['status'] != 'same' or third is None or third['status'] != 'different'
            or third['official']['state_changed'] is not True):
        return None
    joined = np.concatenate((np.broadcast_to(shared, (n, SHARED_DIM)), np.stack(vectors)), axis=1)
    raw64, raw32 = _canonical(joined, '<f8'), _canonical(joined, '<f4')
    features = np.frombuffer(raw32.tobytes(), dtype='<f4').reshape(n, DIM)
    return PrivateExample(role, opponent, side, window, step, ordinal, features,
                          teacher_index, student_index, third_index, raw64.tobytes(),
                          (shared_names, candidate_names),
                          _digest(b'raw-float32\x00', _key(raw32)),
                          _validated=(_TOKEN, role, opponent, side, window, step, ordinal,
                                      teacher_index, student_index, third_index))


def _order(item):
    return _coordinate(item.role, item.opponent, item.side, item.window, item.step, item.ordinal)


def _validate_example(item):
    if not isinstance(item, PrivateExample):
        raise ValueError('Unverified dataset example')
    expected = (_TOKEN, item.role, item.opponent, item.side, item.window, item.step,
                item.ordinal, item.teacher_index, item.student_index, item.third_index)
    if item._validated != expected:
        raise ValueError('Unverified dataset example')
    _order(item)
    x = item.features
    if (not isinstance(x, np.ndarray) or x.dtype != np.dtype('float32') or x.ndim != 2
            or x.shape[1] != DIM or not 2 <= len(x) <= 256 or x.flags.writeable
            or not np.isfinite(x).all() or type(item._float64_bytes) is not bytes
            or len(item._float64_bytes) != x.size * 8):
        raise ValueError('Invalid private example features')
    indices = (item.teacher_index, item.student_index, item.third_index)
    if (any(type(index) is not int or not 0 <= index < len(x) for index in indices)
            or len(set(indices)) != 3):
        raise ValueError('Invalid private example indices')
    x32 = _canonical(x, '<f4')
    original = np.frombuffer(item._float64_bytes, dtype='<f8').reshape(x.shape)
    if (not np.isfinite(original).all() or _key(_canonical(original, '<f4')) != _key(x32)
            or item._input32_fingerprint != _digest(b'raw-float32\x00', _key(x32))):
        raise ValueError('Private feature copy differs')


def build_dataset(examples, *, mean, scale):
    """Build train/dev groups with global, precision-aware conflict closure.

    Each real role/opponent/side/window occurs at most once. Compare both raw
    float32 and the exact normalized float32 input used by the training batch.
    If connected samples disagree on any index, remove the entire component;
    otherwise keep the first fixed coordinate, preferring training to dev.
    The caller authenticates checkpoint normalization and the input schema.
    """
    if type(examples) not in (list, tuple) or len(examples) > MAX_WINDOWS:
        raise ValueError('Invalid bounded example collection')
    mu, sigma = _numeric_vector(mean, DIM), _numeric_vector(scale, DIM)
    if not (sigma >= 1).all():
        raise ValueError('Invalid checkpoint normalization')
    accepted = []
    for item in examples:
        if item is None:
            continue
        _validate_example(item)
        accepted.append(item)
    accepted.sort(key=_order)
    if len({(x.role, x.opponent, x.side, x.window) for x in accepted}) != len(accepted):
        raise ValueError('Multiple selected samples in a fixed source window')
    if accepted and any(x._schema != accepted[0]._schema for x in accepted):
        raise ValueError('Inconsistent task feature schema')

    parents = list(range(len(accepted)))
    by_raw, by_normalized = {}, {}
    deploy_keys, raw_keys, enriched = [], [], []

    def root(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i

    def unite(a, b):
        a, b = root(a), root(b)
        if a != b:
            parents[b] = a

    for i, item in enumerate(accepted):
        raw64 = np.frombuffer(item._float64_bytes, dtype='<f8').reshape(item.features.shape)
        with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
            deployed = _canonical((raw64 - mu) / sigma, '<f8')
            # Same order as train._batch: raw float32 -> float64 -> normalize -> float32.
            network = _canonical((item.features.astype(np.float64) - mu) / sigma, '<f4')
        raw = _canonical(item.features, '<f4')
        k64, kraw, knet = _key(deployed), _key(raw), _key(network)
        deploy_keys.append(k64)
        raw_keys.append(kraw)
        enriched.append(replace(item,
            _deployment_fingerprint=_digest(b'deployment-float64\x00', k64),
            _training_fingerprint=_digest(b'training-float32\x00', knet)))
        for lookup, key in ((by_raw, kraw), (by_normalized, knet)):
            if key in lookup:
                unite(i, lookup[key])
            else:
                lookup[key] = i
    components = {}
    for i in range(len(accepted)):
        components.setdefault(root(i), []).append(i)

    train, dev, counts = [], [], Counter()
    for members in components.values():
        labels = {(accepted[i].teacher_index, accepted[i].student_index, accepted[i].third_index)
                  for i in members}
        counts['deployment_precision_collision_groups'] += len({deploy_keys[i] for i in members}) > 1
        counts['normalization_collision_groups'] += len({raw_keys[i] for i in members}) > 1
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
        else:
            dev.append(enriched[development[0]])
            counts['same_split_dev_removed'] += len(development) - 1
    train.sort(key=_order)
    dev.sort(key=_order)

    def coverage(items, roles):
        counts_by_role = Counter(item.role for item in items)
        counts_by_opp = Counter(item.opponent for item in items)
        counts_by_window = Counter(item.window for item in items)
        counts_by_side = Counter(item.side for item in items)
        return dict(by_role={role: counts_by_role[role] for role in roles},
                    by_opponent={opponent: counts_by_opp[opponent] for opponent in OPPONENTS},
                    by_window={str(window): counts_by_window[window] for window in range(len(WINDOWS))},
                    by_side={str(side): counts_by_side[side] for side in (0, 1)},
                    opponents=len(counts_by_opp), windows=len(counts_by_window))

    tr, dv = coverage(train, TRAIN_ROLES), coverage(dev, DEV_ROLES)
    passed = (len(train) >= 16 and all(tr['by_role'][r] >= 4 for r in TRAIN_ROLES)
              and len(dev) >= 8 and all(dv['by_role'][r] >= 2 for r in DEV_ROLES)
              and min(tr['opponents'], tr['windows'], dv['opponents'], dv['windows']) >= 2)
    summary = dict(status='passed' if passed else 'failed', source_count=len(examples),
                   eligible_count=len(accepted), ineligible_count=len(examples)-len(accepted),
                   train_groups=len(train), dev_groups=len(dev),
                   train_by_role=tr['by_role'], dev_by_role=dv['by_role'],
                   train_by_opponent=tr['by_opponent'], dev_by_opponent=dv['by_opponent'],
                   train_by_window=tr['by_window'], dev_by_window=dv['by_window'],
                   train_by_side=tr['by_side'], dev_by_side=dv['by_side'],
                   train_opponents=tr['opponents'], dev_opponents=dv['opponents'],
                   train_windows=tr['windows'], dev_windows=dv['windows'],
                   training=False, independent_evaluation=False)
    for name in ('conflict_groups', 'conflict_examples_removed', 'same_split_train_removed',
                 'same_split_dev_removed', 'cross_split_groups', 'cross_split_dev_removed',
                 'deployment_precision_collision_groups', 'normalization_collision_groups'):
        summary[name] = counts[name]
    return tuple(train), tuple(dev), summary
