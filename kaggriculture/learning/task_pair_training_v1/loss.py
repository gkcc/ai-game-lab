"""Teacher-conditioned worker ranking loss for one padded microbatch.

This only scores the workers offered for the teacher's chosen PLANT target.
The mask and teacher label are supervision for training, not model inputs or a
deployment-time choice of target. The caller must verify distinct unit IDs,
matching target coordinates, real candidate rows, and the official provenance.
"""

import numpy as np


def validate_inputs(logits, target_mask, labels, eligible, eligible_count):
    """Validate NumPy host inputs before placing a microbatch on a backend.

    ``eligible_count`` is the number of eligible rows in the *whole effective
    batch*, not this microbatch. Hence it may exceed the local eligible count.
    Call this on all prepared microbatches and independently verify that the
    denominator equals the sum of their eligible counts before GPU execution.
    """
    logits = np.asarray(logits)
    target_mask = np.asarray(target_mask)
    labels = np.asarray(labels)
    eligible = np.asarray(eligible)
    if logits.ndim != 2 or not np.issubdtype(logits.dtype, np.floating):
        raise ValueError('Logits must be a two-dimensional floating array')
    rows, width = logits.shape
    if rows < 1 or width < 1 or not np.isfinite(logits).all():
        raise ValueError('Logits must be finite and nonempty')
    if target_mask.shape != (rows, width) or target_mask.dtype != np.dtype(bool):
        raise ValueError('Target mask shape or type differs')
    if labels.shape != (rows,) or not np.issubdtype(labels.dtype, np.integer):
        raise ValueError('Labels shape or type differs')
    if eligible.shape != (rows,) or eligible.dtype != np.dtype(bool):
        raise ValueError('Eligibility shape or type differs')
    if np.any(labels < 0) or np.any(labels >= width):
        raise ValueError('Labels must index an existing padded candidate')
    if isinstance(eligible_count, (bool, np.bool_)) or not isinstance(eligible_count, (int, np.integer)):
        raise ValueError('The effective batch denominator must be an integer')
    if eligible_count < int(np.count_nonzero(eligible)):
        raise ValueError('The effective batch denominator is too small')
    rows_with_teacher = np.flatnonzero(eligible)
    if rows_with_teacher.size:
        if np.any(np.count_nonzero(target_mask[eligible], axis=1) < 2):
            raise ValueError('An eligible row needs at least two target candidates')
        if not np.all(target_mask[rows_with_teacher, labels[rows_with_teacher]]):
            raise ValueError('The teacher label must belong to the target mask')


def conditional_worker_loss(logits, target_mask, labels, eligible, eligible_count, xp=np):
    """Return the microbatch contribution to the effective batch mean.

    Inputs must first pass ``validate_inputs`` on the host. This pure function
    uses only operations shared by NumPy and JAX and can be differentiated or
    JIT-compiled. Each ineligible row has exactly one finite dummy candidate and
    a dummy label, so even a fully ineligible microbatch yields finite zero loss
    and gradients. A zero denominator is valid only for an empty effective batch
    of eligible rows; division uses one in that case. Finite float32 logits can
    still overflow when their extremes are subtracted, so a training caller must
    synchronize and reject nonfinite losses and gradients before any update.
    """
    dummy = xp.arange(logits.shape[1])[None, :] == 0
    safe_mask = xp.where(eligible[:, None], target_mask, dummy)
    safe_labels = xp.where(eligible, labels, 0)
    scores = xp.where(safe_mask, logits, -xp.inf)
    shifted = scores - xp.max(scores, axis=1, keepdims=True)
    selected = shifted[xp.arange(logits.shape[0]), safe_labels]
    per_row = xp.log(xp.sum(xp.exp(shifted), axis=1)) - selected
    summed = xp.sum(xp.where(eligible, per_row, 0))
    return summed / xp.maximum(eligible_count, 1)
