"""Merge authenticated conditional examples without reading files or training.

The caller is responsible for authenticating source identity, game completion,
sample eligibility, the frozen checkpoint and the role-to-seed binding. This
module enforces only the input, split, exact-input deduplication and engineering
coverage contracts. Its output is private except for the aggregate summary.
"""
from collections import Counter
from dataclasses import dataclass, replace

import numpy as np

from learning.task_student_collection_v1.dataset import DIM, SHARED_DIM, _canonical, _key, _numeric_vector
from learning.task_student_dagger_collection_v1.sampling import OPPONENTS, WINDOWS


TRAIN_ROLES = ('F01', 'F02', 'D01', 'D02', 'D03', 'D04',
               'A01', 'A02', 'A03', 'A04', 'A05', 'A06')
DEV_ROLES = ('F03', 'F04', 'D05', 'D06', 'A07', 'A08', 'A09', 'A10')
ROLES = TRAIN_ROLES + DEV_ROLES
MAX_GROUPS = 9 + 19 + 40 * len(WINDOWS)


@dataclass(frozen=True, repr=False)
class Group:
    """One private, already authenticated conditional example."""

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

    def __repr__(self):
        return '<PrivateTaskGroup>'


def _coordinate(group):
    return (ROLES.index(group.role), OPPONENTS.index(group.opponent), group.side,
            group.window, group.step, group.ordinal)


def _check_and_copy(group):
    if type(group) is not Group:
        raise ValueError('Invalid task group')
    if (type(group.role) is not str or group.role not in ROLES
            or type(group.opponent) is not str or group.opponent not in OPPONENTS
            or type(group.side) is not int or group.side not in (0, 1)
            or type(group.window) is not int or not 0 <= group.window < len(WINDOWS)
            or type(group.step) is not int
            or not WINDOWS[group.window][0] <= group.step <= WINDOWS[group.window][1]
            or type(group.ordinal) is not int or group.ordinal < 0):
        raise ValueError('Invalid task group coordinate')
    features = group.features
    if (type(features) is not np.ndarray or features.dtype != np.dtype('float32')
            or features.ndim != 2 or features.shape[1] != DIM
            or not 3 <= features.shape[0] <= 256 or not np.isfinite(features).all()):
        raise ValueError('Invalid task group feature array')
    shared = features[:, :SHARED_DIM]
    if not np.array_equal(shared, np.broadcast_to(shared[:1], shared.shape)):
        raise ValueError('Inconsistent shared features in task group')
    indices = (group.teacher_index, group.student_index, group.third_index)
    if (any(type(value) is not int or not 0 <= value < features.shape[0] for value in indices)
            or len(set(indices)) != 3):
        raise ValueError('Invalid task group indices')
    # The caller may reuse or mutate its arrays after returning. Keep a private
    # immutable canonical copy, with signed zero handled like the old dataset.
    raw = _canonical(features, '<f4')
    immutable = np.frombuffer(raw.tobytes(order='C'), dtype='<f4').reshape(raw.shape)
    return replace(group, features=immutable)


def _distribution(groups, roles):
    by_role = Counter(item.role for item in groups)
    by_opponent = Counter(item.opponent for item in groups)
    by_side = Counter(item.side for item in groups)
    by_window = Counter(item.window for item in groups)
    return dict(
        roles={role: by_role[role] for role in roles},
        opponents={opponent: by_opponent[opponent] for opponent in OPPONENTS},
        sides={str(side): by_side[side] for side in (0, 1)},
        windows={str(window): by_window[window] for window in range(len(WINDOWS))},
    )


def _side_checks(groups, distribution, *, minimum, minimum_roles):
    n = len(groups)
    role_counts = tuple(distribution['roles'].values())
    return dict(
        minimum_groups=n >= minimum,
        minimum_roles=sum(count >= 2 for count in role_counts) >= minimum_roles,
        max_role_share=n > 0 and all(3 * count <= n for count in role_counts),
        both_opponents=all(count >= 2 for count in distribution['opponents'].values()),
        both_sides=all(count >= 2 for count in distribution['sides'].values()),
        three_windows=sum(count > 0 for count in distribution['windows'].values()) >= 3,
    )


def merge(groups, *, mean, scale):
    """Return immutable (train, dev, aggregate) after exact-input deduplication.

    The caller provides only *eligible* groups; this function cannot itself
    verify official transitions or earlier evidence. A raw-f32 or normalized-
    f32 equality joins whole ordered candidate matrices, with transitive closure.
    A conflicting label triple removes its entire component. Otherwise training
    wins cross-split duplicates, followed by fixed coordinate order. The gate is
    a new feasibility criterion, not proof of an old gate or model performance.
    """
    if type(groups) not in (tuple, list) or len(groups) > MAX_GROUPS:
        raise ValueError('Invalid bounded group collection')
    mu, sigma = _numeric_vector(mean, DIM), _numeric_vector(scale, DIM)
    if not (sigma >= 1).all():
        raise ValueError('Invalid checkpoint normalization')
    accepted = [_check_and_copy(item) for item in groups]
    source_counts = Counter(item.role[0] for item in accepted)
    if source_counts['F'] > 9 or source_counts['D'] > 19 or source_counts['A'] > 40 * len(WINDOWS):
        raise ValueError('Source group budget exceeded')
    coordinates = [(item.role, item.opponent, item.side, item.window) for item in accepted]
    if len(set(coordinates)) != len(coordinates):
        raise ValueError('More than one group for a source window')
    accepted.sort(key=_coordinate)

    parents = list(range(len(accepted)))

    def root(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i

    def union(a, b):
        a, b = root(a), root(b)
        if a != b:
            parents[b] = a

    raw_keys, normalized_keys = [], []
    by_raw, by_normalized = {}, {}
    for i, group in enumerate(accepted):
        raw_key = _key(group.features)
        with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
            normalized = _canonical((group.features.astype(np.float64) - mu) / sigma, '<f4')
        normalized_key = _key(normalized)
        raw_keys.append(raw_key)
        normalized_keys.append(normalized_key)
        for lookup, key in ((by_raw, raw_key), (by_normalized, normalized_key)):
            if key in lookup:
                union(i, lookup[key])
            else:
                lookup[key] = i

    components = {}
    for i in range(len(accepted)):
        components.setdefault(root(i), []).append(i)

    train_indices, dev_indices, counts = [], [], Counter()
    for members in components.values():
        labels = {(accepted[i].teacher_index, accepted[i].student_index, accepted[i].third_index)
                  for i in members}
        if len(labels) != 1:
            counts['conflict_components'] += 1
            counts['conflict_groups_removed'] += len(members)
            continue
        training = [i for i in members if accepted[i].role in TRAIN_ROLES]
        development = [i for i in members if accepted[i].role in DEV_ROLES]
        if training:
            train_indices.append(training[0])
            counts['same_split_train_removed'] += len(training) - 1
            if development:
                counts['cross_split_components'] += 1
                counts['cross_split_dev_removed'] += len(development)
        elif development:
            dev_indices.append(development[0])
            counts['same_split_dev_removed'] += len(development) - 1
        else:
            raise RuntimeError('Unassigned task group')

    train_indices.sort()
    dev_indices.sort()
    train = tuple(accepted[i] for i in train_indices)
    dev = tuple(accepted[i] for i in dev_indices)
    # Verify the actual retained representatives, independently of the counts.
    for keys in (raw_keys, normalized_keys):
        tr = {keys[i] for i in train_indices}
        dv = {keys[i] for i in dev_indices}
        if len(tr) != len(train) or len(dv) != len(dev) or tr & dv:
            raise RuntimeError('Retained task inputs are not unique')
    removal_fields = ('conflict_groups_removed', 'same_split_train_removed',
                      'same_split_dev_removed', 'cross_split_dev_removed')
    if len(train) + len(dev) + sum(counts[name] for name in removal_fields) != len(accepted):
        raise RuntimeError('Task group counts do not balance')

    train_dist = _distribution(train, TRAIN_ROLES)
    dev_dist = _distribution(dev, DEV_ROLES)
    checks = dict(train=_side_checks(train, train_dist, minimum=32, minimum_roles=6),
                  dev=_side_checks(dev, dev_dist, minimum=16, minimum_roles=4))
    passed = all(all(side.values()) for side in checks.values())
    summary = dict(
        status='passed' if passed else 'failed',
        source_groups=len(accepted), train_groups=len(train), dev_groups=len(dev),
        components=len(components),
        conflict_components=counts['conflict_components'],
        conflict_groups_removed=counts['conflict_groups_removed'],
        same_split_train_removed=counts['same_split_train_removed'],
        same_split_dev_removed=counts['same_split_dev_removed'],
        cross_split_components=counts['cross_split_components'],
        cross_split_dev_removed=counts['cross_split_dev_removed'],
        train_distribution=train_dist, dev_distribution=dev_dist,
        gate_checks=checks, training=False, independent_evaluation=False,
    )
    return train, dev, summary
