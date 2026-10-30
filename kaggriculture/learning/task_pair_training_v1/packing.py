"""Pack authenticated training groups and independently verify every row.

This is the intended source-to-batch adapter, not an authentication mechanism.
Its caller must bind the exact training arrays, schema, normalization, labels,
class weights and eligibility to the registered sources before invoking it.
"""
import numpy as np

from learning.task_bc_v1 import train
from learning.task_choice_v1.features import OPS
from learning.task_pair_training_v1.targets import plant_targets


def pack_training_batch(data, ids, mean, scale, weights, eligible):
    """Return shared microbatches for both arms, checked against source IDs.

    Identifiers and supervision remain private metadata and do not become input
    features. Source verification occurs before this function and is not a lock
    against a concurrent writer. No incomplete group or candidate is permitted.
    """
    ids = np.asarray(ids)
    mean, scale, weights, eligible = (np.asarray(x) for x in (mean, scale, weights, eligible))
    n = len(data['labels'])
    if (ids.ndim != 1 or ids.dtype.kind not in 'iu' or not 1 <= len(ids) <= 256
            or np.any(ids >= n) or (ids.dtype.kind == 'i' and np.any(ids < 0))
            or len(np.unique(ids)) != len(ids)):
        raise ValueError('Invalid training batch IDs')
    if (mean.shape != (373,) or scale.shape != (373,) or mean.dtype.kind != 'f'
            or scale.dtype.kind != 'f' or not np.isfinite(mean).all()
            or not np.isfinite(scale).all() or np.any(scale < 1)
            or weights.shape != (5,) or weights.dtype.kind != 'f'
            or not np.isfinite(weights).all() or np.any(weights <= 0)
            or eligible.shape != (n,) or eligible.dtype != np.dtype(bool)):
        raise ValueError('Invalid training normalization or supervision')
    parts = []
    for group_ids, capacity, width in train.microbatches(data, ids):
        features, candidate_mask, labels, group_weights = train.padded(
            data, group_ids, capacity, width, mean, scale, weights)
        active, target_mask = plant_targets(data, group_ids, capacity, width, eligible)
        # Do not trust membership IDs alone: independently reconstruct the
        # normalized candidate rows, label and class weight from the source.
        for row, identifier in enumerate(group_ids):
            group = int(identifier)
            lo, hi = map(int, data['offsets'][group:group + 2])
            length = hi - lo
            raw = np.concatenate((np.broadcast_to(data['shared'][group], (length, 275)),
                                  data['candidates'][lo:hi]), axis=1)
            expected = ((raw - mean) / scale).astype(np.float32)
            label = int(data['labels'][group])
            identities = data['candidate_identity'][lo:hi]
            ops = data['candidate_ops'][lo:hi]
            selected_op = int(data['selected_ops'][group])
            if not 0 <= selected_op < len(OPS) or int(ops[label]) != selected_op:
                raise ValueError('Selected operation differs from its source candidate')
            same = ((ops == OPS.index('PLANT'))
                    & (identities[:, 1] == identities[label, 1])
                    & (identities[:, 2] == identities[label, 2]))
            qualified = bool(ops[label] == OPS.index('PLANT')
                             and len(np.unique(identities[same, 0])) >= 2)
            expected_targets = np.zeros(width, dtype=bool)
            if qualified:
                expected_targets[:length] = same
            if (not np.array_equal(features[row, :length], expected)
                    or int(labels[row]) != label
                    or group_weights[row] != np.float32(weights[selected_op])
                    or not np.all(candidate_mask[row, :length])
                    or np.any(candidate_mask[row, length:]) or bool(active[row]) != qualified
                    or not np.array_equal(target_mask[row], expected_targets)):
                raise ValueError('Packed training row differs from its source')
        parts.append({'ids': np.asarray(group_ids, dtype=np.int64), 'features': features,
                      'candidate_mask': candidate_mask, 'labels': labels,
                      'group_weights': group_weights, 'target_mask': target_mask,
                      'eligible': active})
    if not parts:
        raise ValueError('No packed training groups')
    packed_ids = np.concatenate([part['ids'] for part in parts])
    if len(packed_ids) != len(ids) or not np.array_equal(np.sort(packed_ids), np.sort(ids)):
        raise ValueError('Packed groups differ from the requested batch')
    return parts
