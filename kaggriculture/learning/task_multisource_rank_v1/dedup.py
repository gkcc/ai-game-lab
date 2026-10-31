"""Private, conflict-aware deduplication of complete candidate groups.

This module does not read files, models or games. Callers authenticate the
input sources and normalization and must keep the supplied arrays unchanged.
Only per-group temporary matrices and hash indexes are constructed; returned
items are the original objects. No sample is replaced after a removal.
"""
from dataclasses import dataclass
import hashlib
import re

import numpy as np


SHARED_DIM = 275
LOCAL_DIM = 98
DIM = SHARED_DIM + LOCAL_DIM
MAX_GROUPS = 1024
MAX_CANDIDATES = 256
POOLS = ('old_train', 'new_train', 'old_diag', 'new_dev')
_PRIORITY = {'old_train': 0, 'new_train': 0, 'new_dev': 1, 'old_diag': 2}
_ROLE = re.compile(r'[A-Za-z0-9_]{1,16}\Z')


@dataclass(frozen=True, slots=True, repr=False)
class Item:
    shared: np.ndarray
    candidates: np.ndarray
    target: int
    pool: str
    role: str

    def __repr__(self):
        return '<PrivateCandidateGroup>'


def _require(value):
    if not value:
        raise ValueError('Invalid private deduplication input')


def _digest(raw):
    return hashlib.sha256(raw).digest()


def _validate(item):
    _require(type(item) is Item and type(item.pool) is str and item.pool in POOLS
             and type(item.role) is str and _ROLE.fullmatch(item.role) is not None
             and type(item.shared) is np.ndarray and item.shared.dtype == np.dtype('float32')
             and item.shared.shape == (SHARED_DIM,) and np.isfinite(item.shared).all()
             and type(item.candidates) is np.ndarray and item.candidates.dtype == np.dtype('float32')
             and item.candidates.ndim == 2 and item.candidates.shape[1] == LOCAL_DIM
             and 1 <= len(item.candidates) <= MAX_CANDIDATES and np.isfinite(item.candidates).all()
             and type(item.target) is int and 0 <= item.target < len(item.candidates))


def _keys(item, mean, scale):
    """Return ephemeral ordered raw/normalized keys for just one group."""
    count = len(item.candidates)
    raw = np.empty((count, DIM), dtype='<f4')
    raw[:, :SHARED_DIM] = item.shared
    raw[:, SHARED_DIM:] = item.candidates
    _require(np.isfinite(raw).all())
    # Normalize signed zeros in *both* representations, preserving row order.
    raw[raw == 0] = np.float32(0)
    with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
        normalized = ((raw.astype(np.float64) - mean) / scale).astype('<f4')
    _require(np.isfinite(normalized).all())
    normalized[normalized == 0] = np.float32(0)
    return raw.tobytes(order='C'), normalized.tobytes(order='C')


def deduplicate(items, *, mean, scale, guard):
    """Merge exact raw or normalized groups, then remove conflicting targets.

    Within a nonconflicting component the earliest training item wins; if no
    training item exists, new_dev precedes old_diag. Indexes are stored only
    in memory and only anonymized counts are returned. Any failure, including
    a guard exception, prevents a result. This is not an OS lock against
    concurrent mutation by another process.
    """
    _require(type(items) in (list, tuple) and len(items) <= MAX_GROUPS and callable(guard)
             and type(mean) is np.ndarray and mean.dtype == np.dtype('float64') and mean.shape == (DIM,)
             and type(scale) is np.ndarray and scale.dtype == np.dtype('float64') and scale.shape == (DIM,))
    guard()
    mean, scale = mean.copy(), scale.copy()
    _require(np.isfinite(mean).all() and np.isfinite(scale).all() and np.all(scale >= 1))
    parents = list(range(len(items)))
    # Each bucket contains at most one representative for each exact value.
    # Hash collisions always reach a byte comparison; no hash is proof of equality.
    indexes = ({}, {})
    observed = []

    def root(index):
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(left, right):
        left, right = root(left), root(right)
        if left != right:
            parents[right] = left

    for index, item in enumerate(items):
        guard()
        _validate(item)
        keys = _keys(item, mean, scale)
        fingerprints = tuple(_digest(key) for key in keys)
        _require(all(type(value) is bytes and len(value) == 32 for value in fingerprints))
        observed.append(fingerprints)
        for kind, (key, fingerprint) in enumerate(zip(keys, fingerprints)):
            bucket = indexes[kind].setdefault(fingerprint, [])
            matched = False
            for previous in bucket:
                guard()
                previous_key = _keys(items[previous], mean, scale)[kind]
                if previous_key == key:
                    union(index, previous)
                    matched = True
                    break
            if not matched:
                bucket.append(index)

    # Recheck stable inputs after callbacks, before making a result available.
    # This detects ordinary mutation, not malicious change-and-restore races.
    for index, item in enumerate(items):
        guard()
        _validate(item)
        _require(tuple(_digest(key) for key in _keys(item, mean, scale)) == observed[index])

    components = {}
    for index in range(len(items)):
        components.setdefault(root(index), []).append(index)
    retained_indexes = []
    incoming = {pool: 0 for pool in POOLS}
    kept = {pool: 0 for pool in POOLS}
    duplicates = {pool: 0 for pool in POOLS}
    conflicts = {pool: 0 for pool in POOLS}
    for item in items:
        incoming[item.pool] += 1
    conflict_components = 0
    for members in components.values():
        guard()
        if len({items[index].target for index in members}) != 1:
            conflict_components += 1
            for index in members:
                conflicts[items[index].pool] += 1
            continue
        chosen = min(members, key=lambda index: (_PRIORITY[items[index].pool], index))
        retained_indexes.append(chosen)
        kept[items[chosen].pool] += 1
        for index in members:
            if index != chosen:
                duplicates[items[index].pool] += 1
    retained_indexes.sort()
    _require(all(incoming[pool] == kept[pool] + duplicates[pool] + conflicts[pool] for pool in POOLS))
    guard()
    summary = dict(input_count=len(items), retained_count=len(retained_indexes),
                   input_by_pool=incoming, retained_by_pool=kept,
                   duplicate_removed_by_pool=duplicates, conflict_removed_by_pool=conflicts,
                   component_count=len(components), conflict_components=conflict_components)
    return [items[index] for index in retained_indexes], summary
