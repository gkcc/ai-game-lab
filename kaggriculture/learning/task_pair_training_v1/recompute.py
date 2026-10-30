"""Compare a fresh source-derived preflight with the old local mask snapshot.

This comparison binds mask values only after the caller has authenticated the
source registration, arrays, roles and source order. It does not establish when
an old manifest was created or supply a filesystem lock.
"""
import numpy as np

from learning.task_pair_training_v1.masks import KINDS, ROLES


def verify_masks(recomputed, saved):
    """Reject changed, missing, reordered or conflicting private masks.

    Both mappings must already be in authenticated source order: one teacher
    shard and old/new conditional shards. If two mask values happen to be
    identical, comparing them cannot independently authenticate source order.
    """
    if (type(recomputed) is not dict or type(saved) is not dict
            or set(recomputed) != set(ROLES) or set(saved) != set(ROLES)):
        raise ValueError('Invalid private mask roles')
    for role, count in (('teacher_dev', 1), ('conditional_dev', 2)):
        fresh, old = recomputed[role], saved[role]
        if type(fresh) not in (tuple, list) or type(old) not in (tuple, list) or len(fresh) != count or len(old) != count:
            raise ValueError('Invalid private mask shards')
        for new_shard, old_shard in zip(fresh, old, strict=True):
            if (type(new_shard) is not dict or type(old_shard) is not dict
                    or set(new_shard) != set(KINDS) or set(old_shard) != set(KINDS)):
                raise ValueError('Invalid private mask fields')
            length = None
            for key in KINDS:
                actual, prior = new_shard[key], old_shard[key]
                if (not isinstance(actual, np.ndarray) or not isinstance(prior, np.ndarray)
                        or actual.dtype != np.dtype(bool) or prior.dtype != np.dtype(bool)
                        or actual.ndim != 1 or prior.shape != actual.shape
                        or not 1 <= len(actual) <= 1_000_000 or not np.array_equal(actual, prior)):
                    raise ValueError('Recomputed mask differs from the local snapshot')
                if length is None:
                    length = len(actual)
                elif len(actual) != length:
                    raise ValueError('Private mask lengths differ')
            if np.any(new_shard['label_conflict']):
                raise ValueError('Registered zero-conflict gate is not satisfied')
    return True
