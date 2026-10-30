"""Build one private, reproducible batch plan shared by two training arms.

The caller must derive ``eligible`` and ``clusters`` only from the authenticated
training split. Each batch deliberately includes a qualifying anchor, so this
sampling distribution is not the natural training distribution. Do not log or
publish the returned indices, cluster identities or random seed.
"""

import numpy as np


MIN_ELIGIBLE = 20
MAX_CLUSTER = 2**32 - 1
MAX_SEED = 2**64 - 1
MAX_GROUPS = 2**31 - 1
MAX_UPDATES = 1024
MAX_BATCH_SIZE = 4096
MAX_PLAN_ENTRIES = 4_194_304


def make_batches(eligible, clusters, *, rng_seed, updates=64, batch_size=256):
    """Return ``(batches, anchors)`` of private training-group indices.

    The first two anchors come from different clusters. In the first cycle,
    every eligible index appears exactly once, and subsequent cycles repeat the
    same order. The other members of each batch are a uniform sample without
    replacement from all groups except that batch's anchor. Batches may overlap
    each other. Both arms must consume this same returned plan.

    Reproduction additionally requires identical inputs and NumPy version;
    the seed alone is insufficient. This function does not inspect examples,
    calculate eligibility, read validation data, or persist or print a plan.
    """
    if (not isinstance(eligible, np.ndarray) or eligible.ndim != 1
            or eligible.dtype != np.dtype(bool)
            or not isinstance(clusters, np.ndarray) or clusters.ndim != 1
            or clusters.dtype.kind not in ('i', 'u')
            or eligible.shape != clusters.shape):
        raise ValueError('Invalid training group metadata')
    count = len(eligible)
    if count == 0 or count > MAX_GROUPS:
        raise ValueError('Invalid training group count')
    if np.any(clusters < 0) or np.any(clusters > MAX_CLUSTER):
        raise ValueError('Invalid training cluster identity')
    if type(rng_seed) is not int or not 0 <= rng_seed <= MAX_SEED:
        raise ValueError('Invalid batch random seed')
    if type(updates) is not int or not 1 <= updates <= MAX_UPDATES:
        raise ValueError('Invalid update count')
    if (type(batch_size) is not int or not 2 <= batch_size <= MAX_BATCH_SIZE
            or batch_size > count or updates * batch_size > MAX_PLAN_ENTRIES):
        raise ValueError('Invalid batch size')

    eligible_ids = np.flatnonzero(eligible)
    if len(eligible_ids) < MIN_ELIGIBLE:
        raise ValueError('Insufficient qualifying training groups')
    if len(np.unique(clusters[eligible_ids])) < 2:
        raise ValueError('Insufficient qualifying training clusters')

    rng = np.random.default_rng(rng_seed)
    shuffled = rng.permutation(eligible_ids)
    different = np.flatnonzero(clusters[shuffled] != clusters[shuffled[0]])
    second_pos = int(different[0])
    order = np.concatenate((shuffled[:1], shuffled[second_pos:second_pos + 1],
                            shuffled[1:second_pos], shuffled[second_pos + 1:]))
    anchors = np.resize(order, updates).astype(np.int64, copy=False)
    batches = np.empty((updates, batch_size), dtype=np.int64)
    for i, anchor in enumerate(anchors):
        batches[i, 0] = anchor
        # Draw from [0,count-2] then skip over the anchor, preserving a uniform
        # one-to-one mapping to [0,count) excluding that anchor.
        sampled = rng.choice(count - 1, size=batch_size - 1, replace=False)
        batches[i, 1:] = sampled + (sampled >= anchor)
    return batches, anchors
