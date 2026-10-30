"""Build label-conditional training masks from already authenticated groups.

These masks use the teacher label and must never be passed to a deployed model.
The caller must verify the source arrays and their feature/identity schema first.
"""
import numpy as np

from learning.task_choice_v1.features import OPS

MAX_WIDTH = 256
MAX_CAPACITY = 4096


def plant_targets(data, ids, capacity, width, expected_eligible=None):
    """Return per-row eligibility and same-target PLANT candidate masks.

    The two candidate dimensions are padded to ``capacity`` and ``width``.
    An optional whole-data eligibility vector checks that a precomputed anchor
    plan still refers to the authenticated groups. Distinct unit IDs are
    required; candidate truncation and repeated units within a target fail.
    """
    if (type(capacity) is not int or type(width) is not int
            or not 1 <= capacity <= MAX_CAPACITY or not 1 <= width <= MAX_WIDTH):
        raise ValueError('Invalid padded shape')
    ids = np.asarray(ids)
    if ids.ndim != 1 or ids.dtype.kind not in 'iu' or len(ids) > capacity:
        raise ValueError('Invalid group indices')
    offsets = np.asarray(data['offsets'])
    labels = np.asarray(data['labels'])
    operations = np.asarray(data['candidate_ops'])
    identity = np.asarray(data['candidate_identity'])
    n = len(labels)
    if (offsets.ndim != 1 or len(offsets) != n + 1 or offsets.dtype.kind not in 'iu'
            or labels.ndim != 1 or labels.dtype.kind not in 'iu'
            or operations.ndim != 1 or operations.dtype.kind not in 'iu'
            or identity.ndim != 2 or identity.shape != (len(operations), 3)
            or identity.dtype.kind not in 'iu' or int(offsets[0]) != 0
            or int(offsets[-1]) != len(operations)):
        raise ValueError('Invalid candidate arrays')
    if np.any(ids >= n) or (ids.dtype.kind == 'i' and np.any(ids < 0)) or len(np.unique(ids)) != len(ids):
        raise ValueError('Invalid or repeated group indices')
    if expected_eligible is not None:
        expected_eligible = np.asarray(expected_eligible)
        if expected_eligible.shape != (n,) or expected_eligible.dtype != np.dtype(bool):
            raise ValueError('Invalid expected eligibility')
    mask = np.zeros((capacity, width), dtype=bool)
    eligible = np.zeros(capacity, dtype=bool)
    plant = OPS.index('PLANT')
    for row, identifier in enumerate(ids):
        group = int(identifier)
        lo, hi, label = int(offsets[group]), int(offsets[group + 1]), int(labels[group])
        if not (0 <= lo < hi <= len(operations) and hi - lo <= width and 0 <= label < hi - lo):
            raise ValueError('Invalid group bounds or label')
        actual = lo + label
        same = ((operations[lo:hi] == plant)
                & (identity[lo:hi, 1] == identity[actual, 1])
                & (identity[lo:hi, 2] == identity[actual, 2]))
        units = identity[lo:hi, 0][same]
        if len(np.unique(units)) != len(units):
            raise ValueError('Duplicate unit at same PLANT target')
        eligible[row] = operations[actual] == plant and len(units) >= 2
        if eligible[row]:
            mask[row, :hi - lo] = same
        if expected_eligible is not None and eligible[row] != expected_eligible[group]:
            raise ValueError('Training eligibility differs from the plan')
    return eligible, mask


def qualifying_groups(data, *, width=256, chunk_size=4096):
    """Derive anchor eligibility from the supplied authenticated split only.

    Whole-candidate masks are temporary and bounded by ``chunk_size * width``;
    the result is private supervision metadata, never a deployment feature.
    """
    if (type(width) is not int or type(chunk_size) is not int
            or not 1 <= width <= MAX_WIDTH or not 1 <= chunk_size <= MAX_CAPACITY):
        raise ValueError('Invalid eligibility chunk size')
    labels = np.asarray(data['labels'])
    if labels.ndim != 1 or not len(labels):
        raise ValueError('Invalid group labels')
    eligible = np.empty(len(labels), dtype=bool)
    for start in range(0, len(labels), chunk_size):
        stop = min(len(labels), start + chunk_size)
        current, _ = plant_targets(data, np.arange(start, stop, dtype=np.int64), stop - start, width)
        eligible[start:stop] = current
    return eligible
