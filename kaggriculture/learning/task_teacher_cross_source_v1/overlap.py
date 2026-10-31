"""Compare old and new teacher groups without reading data or training.

The caller authenticates input provenance, split roles and the baseline
normalization. A match means equal *ordered numeric representations* and
equal or different target indices; it is not proof of matching entities,
independent game states, official execution, or economic value. No group is
removed or changed. Per-group results should remain private until separately
reviewed for publication.

Old groups are consumed once and one at a time. Two indexes hold raw32 and
normalized32 byte keys only for the at-most-384 new groups. A SHA-256 hit is
always followed by an exact byte comparison, including on hash collisions.
When feeding external archives, callers must not call ``finish`` until every
archive scan has returned successfully, including its final identity check.
This in-memory object cannot itself observe an external scanner's completion.
"""
from dataclasses import dataclass
import hashlib

import numpy as np


SHARED_DIM = 275
LOCAL_DIM = 98
DIM = SHARED_DIM + LOCAL_DIM
MAX_CANDIDATES = 256
MAX_NEW_GROUPS = 384
SPLITS = ('train', 'dev')
STATUSES = ('same', 'conflict', 'none')


@dataclass(frozen=True, slots=True, repr=False)
class OldGroup:
    """One old teacher choice in its original candidate order."""

    shared: np.ndarray
    candidates: np.ndarray
    target_index: int

    def __repr__(self):
        return '<PrivateOldGroup>'


@dataclass(frozen=True, slots=True, repr=False)
class NewGroup:
    """One new group, with an anonymous split and a target index."""

    features: np.ndarray
    target_index: int
    split: str

    def __repr__(self):
        return '<PrivateNewGroup>'


@dataclass(frozen=True, slots=True)
class Match:
    """Results for one new group, in the caller's original input order."""

    split: str
    raw: str
    normalized: str


@dataclass(frozen=True, slots=True)
class Comparison:
    rows: tuple
    summary: dict
    old_groups: int


def _require(condition):
    if not condition:
        raise ValueError('Invalid cross-source comparison input')


def _target(value, count):
    _require(isinstance(value, (int, np.integer))
             and not isinstance(value, (bool, np.bool_)) and 0 <= int(value) < count)
    return int(value)


def _normalization(mean, scale):
    _require(type(mean) is np.ndarray and mean.dtype == np.dtype('float64') and mean.shape == (DIM,)
             and type(scale) is np.ndarray and scale.dtype == np.dtype('float64') and scale.shape == (DIM,))
    mean, scale = mean.copy(), scale.copy()
    _require(np.isfinite(mean).all() and np.isfinite(scale).all() and np.all(scale > 0))
    return mean, scale


def _canonical(values):
    """Use explicit little-endian float32 and identify positive/negative zero."""
    result = np.array(values, dtype='<f4', copy=True, order='C')
    _require(np.isfinite(result).all())
    result[result == 0] = 0.0
    return result


def _keys(raw, mean, scale):
    raw = _canonical(raw)
    with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
        normalized = ((raw.astype(np.float64) - mean) / scale).astype('<f4')
    normalized = _canonical(normalized)
    return raw.tobytes(order='C'), normalized.tobytes(order='C')


def _new(group, mean, scale):
    _require(type(group) is NewGroup and group.split in SPLITS and type(group.split) is str)
    raw = group.features
    _require(type(raw) is np.ndarray and raw.dtype == np.dtype('float32') and raw.ndim == 2
             and raw.shape[1] == DIM and 1 <= len(raw) <= MAX_CANDIDATES
             and np.isfinite(raw).all()
             and np.array_equal(raw[:, :SHARED_DIM], np.broadcast_to(raw[:1, :SHARED_DIM],
                                                                      (len(raw), SHARED_DIM))))
    label = _target(group.target_index, len(raw))
    return _keys(raw, mean, scale), label


def _old(group, mean, scale):
    _require(type(group) is OldGroup)
    shared, candidates = group.shared, group.candidates
    _require(type(shared) is np.ndarray and shared.dtype == np.dtype('float32')
             and shared.shape == (SHARED_DIM,) and np.isfinite(shared).all()
             and type(candidates) is np.ndarray and candidates.dtype == np.dtype('float32')
             and candidates.ndim == 2 and candidates.shape[1] == LOCAL_DIM
             and 1 <= len(candidates) <= MAX_CANDIDATES and np.isfinite(candidates).all())
    label = _target(group.target_index, len(candidates))
    raw = np.concatenate((np.broadcast_to(shared, (len(candidates), SHARED_DIM)), candidates), axis=1)
    return _keys(raw, mean, scale), label


def _digest(raw):
    return hashlib.sha256(raw).digest()


def _hash(raw):
    value = _digest(raw)
    _require(type(value) is bytes and len(value) == 32)
    return value


def _summary(rows):
    result = {kind: {split: {status: 0 for status in STATUSES} for split in SPLITS}
              for kind in ('raw', 'normalized')}
    for row in rows:
        result['raw'][row.split][row.raw] += 1
        result['normalized'][row.split][row.normalized] += 1
    return result


class Matcher:
    """Incrementally consume old choices and finalize exactly once.

    ``guard`` is called before groups and hash-candidate checks and must raise
    to stop work. A failed consume or finish permanently closes the matcher;
    callers cannot turn partial work into a result. ``finish`` can only
    establish that this object finished; the caller is responsible for first
    confirming that *all* external source scans and their final rehashes passed.
    """

    def __init__(self, new_groups, mean, scale, guard):
        _require(callable(guard) and type(new_groups) in (list, tuple)
                 and len(new_groups) <= MAX_NEW_GROUPS)
        guard()
        self._mean, self._scale = _normalization(mean, scale)
        self._guard = guard
        self._indexes = ({}, {})
        self._outcomes = [['none', 'none'] for _ in new_groups]
        self._splits = []
        self._old_count = 0
        self._state = 'initializing'
        for index, group in enumerate(new_groups):
            guard()
            keys, label = _new(group, self._mean, self._scale)
            self._splits.append(group.split)
            for kind, raw in enumerate(keys):
                self._indexes[kind].setdefault(_hash(raw), []).append((raw, label, index))
        guard()
        self._state = 'open'

    def _begin(self):
        _require(self._state == 'open')
        self._state = 'busy'

    def _failed(self):
        self._state = 'failed'
        self._indexes = ({}, {})

    def consume(self, group):
        """Examine exactly one bounded old group, without retaining it."""
        self._begin()
        try:
            self._guard()
            keys, old_label = _old(group, self._mean, self._scale)
            for kind, raw in enumerate(keys):
                for new_raw, new_label, index in self._indexes[kind].get(_hash(raw), ()):
                    self._guard()
                    if new_raw != raw:
                        continue
                    if old_label != new_label:
                        self._outcomes[index][kind] = 'conflict'
                    elif self._outcomes[index][kind] == 'none':
                        self._outcomes[index][kind] = 'same'
            self._guard()
            self._old_count += 1
        except BaseException:
            self._failed()
            raise
        self._state = 'open'

    def finish(self):
        """Return one result after callers have confirmed every source scan."""
        self._begin()
        try:
            self._guard()
            rows = tuple(Match(split, outcome[0], outcome[1])
                         for split, outcome in zip(self._splits, self._outcomes))
            result = Comparison(rows, _summary(rows), self._old_count)
        except BaseException:
            self._failed()
            raise
        self._state = 'finished'
        self._indexes = ({}, {})
        return result


def compare(old_groups, new_groups, *, mean, scale, guard):
    """Compare one iterable; a failed iteration never returns partial results.

    ``same`` and ``conflict`` refer to target *indices*. Conflict wins when
    both appear in old data. Raw and normalized matches remain independent.
    Callers feeding separately authenticated callback scans should use
    ``Matcher`` and call ``finish`` only after every scan returns successfully.
    """
    matcher = Matcher(new_groups, mean=mean, scale=scale, guard=guard)
    iterator = iter(old_groups)
    while True:
        guard()
        try:
            group = next(iterator)
        except StopIteration:
            break
        matcher.consume(group)
    return matcher.finish()
