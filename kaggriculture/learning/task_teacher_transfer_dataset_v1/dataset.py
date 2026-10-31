"""Private, seed-split transfer groups and a bounded BC feasibility gate.

Only authenticated, preselected *supported* groups belong here. The caller
must check each teacher ordinal against ``rank_ordinals`` using that turn's
authenticated callback count, verify the common controller prefix and model
schema, and bind anonymous roles to already-completed seed clusters. This
module does not read files, load models, run a game, or authenticate sources.
Its returned arrays are private; only the fixed aggregate summary is publishable
after the caller's separate privacy review.
"""
from collections import Counter
from dataclasses import dataclass, replace
import hashlib
import json

import numpy as np

from decision.api import PRODUCTS
from learning.task_choice_v1.features import OPS
from learning.task_student_collection_v1.dataset import DIM, SHARED_DIM, _canonical, _key, _numeric_vector


TRAIN_ROLES = ('A01', 'A02')
DEV_ROLES = ('A07', 'A08')
ROLES = TRAIN_ROLES + DEV_ROLES
OPPONENTS = ('market', 'round2')
STEPS = (48, 172, 296, 420, 544, 668)
HOURS = tuple(step % 24 for step in STEPS)
MAX_CALLBACKS, MAX_SELECTED, MAX_CANDIDATES = 32, 4, 256
MAX_GROUPS = len(ROLES) * len(OPPONENTS) * 2 * len(STEPS) * MAX_SELECTED
_SELECT_DOMAIN = b'kaggriculture.teacher-transfer-v1.sample\x00'
_REPRESENTATIVE_DOMAIN = b'kaggriculture.teacher-transfer-v1.representative\x00'
_OP_START = SHARED_DIM + 5 + len(PRODUCTS)


@dataclass(frozen=True, repr=False)
class Group:
    """One private terminal-encoded candidate set with a teacher target."""

    role: str
    opponent: str
    side: int
    step: int
    teacher_ordinal: int
    terminal_ordinal: int
    op: str
    features: np.ndarray
    target_index: int

    def __repr__(self):
        return '<PrivateTeacherTransferGroup>'


def _coordinate(role, opponent, side, step):
    if (type(role) is not str or role not in ROLES
            or type(opponent) is not str or opponent not in OPPONENTS
            or type(side) is not int or side not in (0, 1)
            or type(step) is not int or step not in STEPS):
        raise ValueError('Invalid public transfer coordinate')
    return role, opponent, side, step


def _digest(domain, coordinate):
    payload = json.dumps(list(coordinate), ensure_ascii=True, separators=(',', ':'),
                         allow_nan=False).encode('ascii')
    return hashlib.sha256(domain + payload).digest()


def rank_ordinals(role, opponent, side, step, total):
    """Rank all callbacks using public coordinates, before support is known.

    Return at most four original ordinals in hash order. Unknown or unsupported
    choices must not be replaced; the caller must freeze the returned roster.
    """
    coordinate = _coordinate(role, opponent, side, step)
    if type(total) is not int or not 0 <= total <= MAX_CALLBACKS:
        raise ValueError('Invalid teacher callback count')
    return tuple(sorted(range(total), key=lambda ordinal:
                        (_digest(_SELECT_DOMAIN, (*coordinate, ordinal)), ordinal))[:MAX_SELECTED])


def _checked_copy(group):
    if type(group) is not Group:
        raise ValueError('Invalid private transfer group')
    _coordinate(group.role, group.opponent, group.side, group.step)
    if (type(group.teacher_ordinal) is not int or not 0 <= group.teacher_ordinal < MAX_CALLBACKS
            or type(group.terminal_ordinal) is not int or not 0 <= group.terminal_ordinal < MAX_CALLBACKS
            # A supported shared prefix has one entry per earlier callback in
            # each stream. Equal ordinals are necessary, not proof of a match.
            or group.teacher_ordinal != group.terminal_ordinal
            or type(group.op) is not str or group.op not in OPS):
        raise ValueError('Invalid callback or operation')
    x = group.features
    if (type(x) is not np.ndarray or x.dtype != np.dtype('float32')
            or x.ndim != 2 or x.shape[1] != DIM or not 1 <= x.shape[0] <= MAX_CANDIDATES
            or not np.isfinite(x).all() or type(group.target_index) is not int
            or not 0 <= group.target_index < len(x)):
        raise ValueError('Invalid transfer features or target')
    shared = x[:, :SHARED_DIM]
    if not np.array_equal(shared, np.broadcast_to(shared[:1], shared.shape)):
        raise ValueError('Inconsistent shared transfer features')
    # The authenticated task schema places the five task-operation one-hot
    # columns after unit/target and carried-item features. Check that the
    # aggregated operation is actually the selected candidate's operation.
    encoded_ops = x[:, _OP_START:_OP_START + len(OPS)]
    if (encoded_ops.shape != (len(x), len(OPS))
            or not np.isin(encoded_ops, (0.0, 1.0)).all()
            or not np.all(encoded_ops.sum(axis=1) == 1.0)
            or encoded_ops[group.target_index, OPS.index(group.op)] != 1.0):
        raise ValueError('Transfer operation does not match candidate features')
    raw = _canonical(x, '<f4')
    copied = np.frombuffer(raw.tobytes(order='C'), dtype='<f4').reshape(raw.shape)
    return replace(group, features=copied)


def _source_key(group):
    return (group.role, group.opponent, group.side, group.step, group.teacher_ordinal)


def _representative_key(group):
    coordinate = _source_key(group)
    return (_digest(_REPRESENTATIVE_DOMAIN, coordinate), coordinate)


def _distribution(groups, roles):
    by_role = Counter(group.role for group in groups)
    by_opponent = Counter(group.opponent for group in groups)
    by_side = Counter(group.side for group in groups)
    by_hour = Counter(group.step % 24 for group in groups)
    by_op = Counter(group.op for group in groups)
    by_opponent_hour = Counter((group.opponent, group.step % 24) for group in groups)
    return dict(
        roles={role: by_role[role] for role in roles},
        opponents={opponent: by_opponent[opponent] for opponent in OPPONENTS},
        sides={str(side): by_side[side] for side in (0, 1)},
        hours={str(hour): by_hour[hour] for hour in HOURS},
        operations={op: by_op[op] for op in OPS},
        opponent_hours={opponent: {str(hour): by_opponent_hour[(opponent, hour)] for hour in HOURS}
                        for opponent in OPPONENTS},
    )


def _gate(groups, distribution, *, minimum, role_min, opponent_min, side_min):
    return dict(
        minimum_groups=len(groups) >= minimum,
        per_role=all(value >= role_min for value in distribution['roles'].values()),
        per_opponent=all(value >= opponent_min for value in distribution['opponents'].values()),
        per_side=all(value >= side_min for value in distribution['sides'].values()),
        four_hours=sum(value > 0 for value in distribution['hours'].values()) >= 4,
        per_opponent_two_hours=all(sum(value > 0 for value in hours.values()) >= 2
                                   for hours in distribution['opponent_hours'].values()),
        two_ops=sum(value > 0 for value in distribution['operations'].values()) >= 2,
    )


def merge(groups, *, mean, scale, guard):
    """Return ``(train, dev, aggregate)`` after split-safe group deduplication.

    Single-candidate groups have no categorical-choice signal and are excluded
    from both the corpora and every data gate. Equal raw or normalized ordered
    groups are joined transitively. Conflicting target indices discard their
    entire component. Nonconflicting components with no distinction among their
    normalized candidate rows are excluded too; a fixed coordinate hash picks
    the remaining representatives with train priority over dev. Row-overlap and
    partial target-tie counts are diagnostics, not proof of independence.
    """
    if not callable(guard) or type(groups) not in (tuple, list) or len(groups) > MAX_GROUPS:
        raise ValueError('Invalid bounded transfer group collection')
    guard()
    # Snapshot normalization so later caller mutation cannot change how groups
    # are deduplicated midway through this single merge.
    mu, sigma = _numeric_vector(mean, DIM).copy(), _numeric_vector(scale, DIM).copy()
    if not (sigma >= 1).all():
        raise ValueError('Invalid checkpoint normalization')
    accepted = []
    cells = {}
    for group in groups:
        guard()
        item = _checked_copy(group)
        cell = (item.role, item.opponent, item.side, item.step)
        cells.setdefault(cell, []).append(item)
        accepted.append(item)
    for members in cells.values():
        guard()
        if (len(members) > MAX_SELECTED
                or len({group.teacher_ordinal for group in members}) != len(members)
                or len({group.terminal_ordinal for group in members}) != len(members)):
            raise ValueError('Repeated or excessive callbacks in a sampled turn')
        ordered = sorted(members, key=lambda group: group.teacher_ordinal)
        if any(left.terminal_ordinal >= right.terminal_ordinal
               for left, right in zip(ordered, ordered[1:])):
            raise ValueError('Teacher and terminal callback order differs')

    singleton_removed = sum(len(item.features) == 1 for item in accepted)
    accepted = [item for item in accepted if len(item.features) >= 2]
    accepted.sort(key=lambda group: (0 if group.role in TRAIN_ROLES else 1,
                                     _representative_key(group)))
    parents = list(range(len(accepted)))

    def root(index):
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(left, right):
        left, right = root(left), root(right)
        if left != right:
            parents[right] = left

    raw_keys, normalized_keys = [], []
    by_raw, by_normalized = {}, {}
    for index, group in enumerate(accepted):
        guard()
        raw_key = _key(group.features)
        with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
            normalized = _canonical((group.features.astype(np.float64) - mu) / sigma, '<f4')
        normalized_key = _key(normalized)
        raw_keys.append(raw_key)
        normalized_keys.append(normalized_key)
        for lookup, key in ((by_raw, raw_key), (by_normalized, normalized_key)):
            if key in lookup:
                union(index, lookup[key])
            else:
                lookup[key] = index

    components = {}
    for index in range(len(accepted)):
        guard()
        components.setdefault(root(index), []).append(index)

    row_bytes = DIM * np.dtype('<f4').itemsize

    def target_ties(index):
        """Count other normalized candidate rows equal to this target row."""
        raw = normalized_keys[index][2]
        target = accepted[index].target_index
        selected = raw[target * row_bytes:(target + 1) * row_bytes]
        return sum(position != target and raw[position * row_bytes:(position + 1) * row_bytes] == selected
                   for position in range(len(accepted[index].features)))

    train_indices, dev_indices, counts = [], [], Counter()
    for members in components.values():
        guard()
        if len({accepted[index].target_index for index in members}) != 1:
            counts['conflict_components'] += 1
            counts['conflict_groups_removed'] += len(members)
            continue
        if any(target_ties(index) == len(accepted[index].features) - 1 for index in members):
            counts['indistinguishable_groups_removed'] += len(members)
            continue
        training = [index for index in members if accepted[index].role in TRAIN_ROLES]
        development = [index for index in members if accepted[index].role in DEV_ROLES]
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
            raise RuntimeError('Unassigned transfer group')
    train_indices.sort()
    dev_indices.sort()
    train = tuple(accepted[index] for index in train_indices)
    dev = tuple(accepted[index] for index in dev_indices)

    for keys in (raw_keys, normalized_keys):
        left, right = {keys[i] for i in train_indices}, {keys[i] for i in dev_indices}
        if len(left) != len(train) or len(right) != len(dev) or left & right:
            raise RuntimeError('Retained transfer groups are not unique')
    removed = ('conflict_groups_removed', 'indistinguishable_groups_removed', 'same_split_train_removed',
               'same_split_dev_removed', 'cross_split_dev_removed')
    if len(groups) != singleton_removed + len(train) + len(dev) + sum(counts[key] for key in removed):
        raise RuntimeError('Transfer group counts do not balance')

    # A group can differ overall while some per-candidate inputs repeat. Report
    # both representations; this does not establish independence of clean rows.
    train_raw_rows, train_normalized_rows = set(), set()
    for index in train_indices:
        guard()
        raw, normalized = raw_keys[index][2], normalized_keys[index][2]
        for start in range(0, len(raw), row_bytes):
            train_raw_rows.add(raw[start:start + row_bytes])
            train_normalized_rows.add(normalized[start:start + row_bytes])
    overlap_raw = overlap_normalized = overlap_any = 0
    for index in dev_indices:
        guard()
        raw, normalized = raw_keys[index][2], normalized_keys[index][2]
        has_raw = any(raw[start:start + row_bytes] in train_raw_rows
                      for start in range(0, len(raw), row_bytes))
        has_normalized = any(normalized[start:start + row_bytes] in train_normalized_rows
                             for start in range(0, len(normalized), row_bytes))
        overlap_raw += int(has_raw)
        overlap_normalized += int(has_normalized)
        overlap_any += int(has_raw or has_normalized)

    train_dist, dev_dist = _distribution(train, TRAIN_ROLES), _distribution(dev, DEV_ROLES)
    checks = dict(train=_gate(train, train_dist, minimum=32, role_min=8, opponent_min=6, side_min=4),
                  dev=_gate(dev, dev_dist, minimum=16, role_min=4, opponent_min=3, side_min=2))
    passed = all(all(side.values()) for side in checks.values())
    summary = dict(
        status='passed' if passed else 'failed', source_groups=len(groups),
        singleton_groups_removed=singleton_removed, components=len(components),
        conflict_components=counts['conflict_components'],
        conflict_groups_removed=counts['conflict_groups_removed'],
        indistinguishable_groups_removed=counts['indistinguishable_groups_removed'],
        same_split_train_removed=counts['same_split_train_removed'],
        same_split_dev_removed=counts['same_split_dev_removed'],
        cross_split_components=counts['cross_split_components'],
        cross_split_dev_removed=counts['cross_split_dev_removed'],
        train_groups=len(train), dev_groups=len(dev),
        train_distribution=train_dist, dev_distribution=dev_dist, gate_checks=checks,
        dev_groups_with_raw_row_overlap=overlap_raw,
        dev_groups_with_normalized_row_overlap=overlap_normalized,
        dev_groups_with_any_row_overlap=overlap_any,
        dev_groups_without_row_overlap=len(dev) - overlap_any,
        train_groups_with_target_normalized_tie=sum(target_ties(i) > 0 for i in train_indices),
        dev_groups_with_target_normalized_tie=sum(target_ties(i) > 0 for i in dev_indices),
        training=False, independent_evaluation=False,
    )
    guard()
    return train, dev, summary
