"""Split-aware deduplication without diagnostic influence on training.

The caller authenticates each input, supplies fixed train/development roles,
and must keep arrays unchanged during this call. This module reads no files,
models or games. It retains original private Items and emits only aggregate
counts; callers remain responsible for immutable snapshots and safe output.

Training-only feature components determine training retention using training
labels alone. A separate feature-only graph over *all* original inputs masks
every diagnostic item connected to any original training item, including a
training conflict or duplicate. Such diagnostic targets are never inspected.
Only components containing diagnostic inputs alone use diagnostic labels.
"""
import numpy as np

from learning.task_multisource_rank_v1 import dedup


TRAIN = frozenset(('new_train', 'old_train'))
DIAGNOSTIC = frozenset(('new_dev', 'old_diag'))
_PRIORITY = {'new_train': 0, 'old_train': 1, 'new_dev': 0, 'old_diag': 1}


class SplitDedupError(ValueError):
    """The supplied private split inputs are inconsistent."""


def _require(value):
    if not value:
        raise SplitDedupError('Invalid private split deduplication input')


def _features_only(item):
    """Do not access a target until its component's split is known."""
    _require(type(item) is dedup.Item and type(item.pool) is str and item.pool in dedup.POOLS
             and type(item.role) is str and dedup._ROLE.fullmatch(item.role) is not None
             and type(item.shared) is np.ndarray and item.shared.dtype == np.dtype('float32')
             and item.shared.shape == (dedup.SHARED_DIM,) and np.isfinite(item.shared).all()
             and type(item.candidates) is np.ndarray and item.candidates.dtype == np.dtype('float32')
             and item.candidates.ndim == 2 and item.candidates.shape[1] == dedup.LOCAL_DIM
             and 1 <= len(item.candidates) <= dedup.MAX_CANDIDATES and np.isfinite(item.candidates).all())


def _target(item):
    value = item.target
    _require(type(value) is int and 0 <= value < len(item.candidates))
    return value


def deduplicate(items, *, mean, scale, guard):
    """Return ``(retained, summary)`` with no diagnostic influence on train.

    Both input representations are complete ordered candidate groups, not
    unordered individual rows. SHA buckets are followed by exact byte checks;
    equal raw or equal normalized float32 representations connect components.
    Within each split, conflicting teacher indices discard that component;
    otherwise new_train precedes old_train and new_dev precedes old_diag,
    with original input order breaking ties. Returned items retain input order.

    Labels in diagnostic components connected to any training input are not
    read or compared. The caller must not mutate input objects while running;
    guard failures propagate, and no partially calculated result is returned.
    """
    _require(type(items) in (list, tuple) and len(items) <= dedup.MAX_GROUPS and callable(guard)
             and type(mean) is np.ndarray and mean.dtype == np.dtype('float64')
             and mean.shape == (dedup.DIM,) and type(scale) is np.ndarray
             and scale.dtype == np.dtype('float64') and scale.shape == (dedup.DIM,))
    guard()
    mean, scale = mean.copy(), scale.copy()
    _require(np.isfinite(mean).all() and np.isfinite(scale).all() and np.all(scale >= 1))
    train_parent = list(range(len(items)))
    all_parent = list(range(len(items)))
    all_indexes = ({}, {})
    train_indexes = ({}, {})

    def root(parents, index):
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(parents, left, right):
        left, right = root(parents, left), root(parents, right)
        if left != right:
            parents[right] = left

    def insert(indexes, parents, kind, key, digest, index):
        bucket = indexes[kind].setdefault(digest, [])
        for previous in bucket:
            guard()
            if dedup._keys(items[previous], mean, scale)[kind] == key:
                union(parents, index, previous)
                return
        bucket.append(index)

    incoming = dict.fromkeys(dedup.POOLS, 0)
    kept = dict.fromkeys(dedup.POOLS, 0)
    duplicates = dict.fromkeys(dedup.POOLS, 0)
    conflicts = dict.fromkeys(dedup.POOLS, 0)
    diagnostic_excluded = dict.fromkeys(dedup.POOLS, 0)
    train_indices = []
    for item in items:
        # Split classification alone is needed before choosing the first
        # graph; no diagnostic feature or target is read at this point.
        _require(type(item) is dedup.Item and type(item.pool) is str and item.pool in dedup.POOLS)
        incoming[item.pool] += 1
    train_indices = [index for index, item in enumerate(items) if item.pool in TRAIN]
    train_fingerprints = {}
    for index in train_indices:
        guard()
        item = items[index]
        _features_only(item)
        keys = dedup._keys(item, mean, scale)
        digests = tuple(dedup._digest(key) for key in keys)
        _require(all(type(value) is bytes and len(value) == 32 for value in digests))
        train_fingerprints[index] = digests
        for kind, (key, digest) in enumerate(zip(keys, digests)):
            insert(train_indexes, train_parent, kind, key, digest, index)
    for index in train_indices:
        guard()
        _features_only(items[index])
        _require(tuple(dedup._digest(key) for key in dedup._keys(items[index], mean, scale))
                 == train_fingerprints[index])
    train_components = {}
    for index in train_indices:
        train_components.setdefault(root(train_parent, index), []).append(index)
    retained_indexes = []

    def decide(members):
        guard()
        if len({_target(items[index]) for index in members}) != 1:
            for index in members:
                conflicts[items[index].pool] += 1
            return True
        selected = min(members, key=lambda index: (_PRIORITY[items[index].pool], index))
        retained_indexes.append(selected)
        kept[items[selected].pool] += 1
        for index in members:
            if index != selected:
                duplicates[items[index].pool] += 1
        return False

    train_targets = {index: _target(items[index]) for index in train_indices}
    train_conflicts = sum(int(decide(members)) for members in train_components.values())

    # Only after every training decision has been fixed do diagnostic features
    # join a separate graph. The original training inputs, not just retained
    # ones, participate in this second graph's exclusion rule.
    fingerprints = []
    for index, item in enumerate(items):
        guard()
        _features_only(item)
        keys = dedup._keys(item, mean, scale)
        digests = tuple(dedup._digest(key) for key in keys)
        _require(all(type(value) is bytes and len(value) == 32 for value in digests))
        if item.pool in TRAIN:
            _require(digests == train_fingerprints[index] and _target(item) == train_targets[index])
        fingerprints.append(digests)
        for kind, (key, digest) in enumerate(zip(keys, digests)):
            insert(all_indexes, all_parent, kind, key, digest, index)
    # Detect ordinary mutation before diagnostic targets are used. This is not
    # a defence against hostile concurrent modification or hash substitution.
    for index, item in enumerate(items):
        guard()
        _features_only(item)
        _require(tuple(dedup._digest(key) for key in dedup._keys(item, mean, scale)) == fingerprints[index])
        if item.pool in TRAIN:
            _require(_target(item) == train_targets[index])
    all_components = {}
    for index in range(len(items)):
        all_components.setdefault(root(all_parent, index), []).append(index)
    diagnostic_only_count = diagnostic_conflicts = train_connected_count = 0
    for members in all_components.values():
        guard()
        diagnostic = [index for index in members if items[index].pool in DIAGNOSTIC]
        if not diagnostic:
            continue
        if any(items[index].pool in TRAIN for index in members):
            train_connected_count += 1
            for index in diagnostic:
                diagnostic_excluded[items[index].pool] += 1
        else:
            diagnostic_only_count += 1
            diagnostic_conflicts += int(decide(diagnostic))
    retained_indexes.sort()
    _require(all(incoming[pool] == kept[pool] + duplicates[pool] + conflicts[pool]
                 + diagnostic_excluded[pool] for pool in dedup.POOLS))
    guard()
    summary = dict(input_count=len(items), retained_count=len(retained_indexes),
        input_by_pool=incoming, retained_by_pool=kept,
        duplicate_removed_by_pool=duplicates, conflict_removed_by_pool=conflicts,
        diagnostic_excluded_train_connected_by_pool=diagnostic_excluded,
        train_component_count=len(train_components), all_feature_component_count=len(all_components),
        train_conflict_components=train_conflicts, diagnostic_only_components=diagnostic_only_count,
        diagnostic_conflict_components=diagnostic_conflicts,
        diagnostic_train_connected_components=train_connected_count)
    return [items[index] for index in retained_indexes], summary
