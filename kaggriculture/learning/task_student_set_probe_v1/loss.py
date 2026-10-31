"""Conditional-set objective with a shared frozen-baseline KL regularizer.

For each authenticated group, P consists exactly of the teacher and original
baseline choices (one index if they coincide). V is P plus exactly one verified
different-state candidate. The latter is a *behavioral contrast*, not evidence
of lower reward. Candidates outside V are unknown; they participate in the
full-candidate baseline KL but are not assigned a supervised class.

Call ``validate_batch`` on host NumPy arrays before backend execution. For JAX,
also check computed losses, gradients and optimizer state for finiteness on the
host before applying any update: those value-dependent checks cannot be silently
performed inside a regular JIT trace. This module does not train a model.
"""
import numpy as np


KL_WEIGHT = 0.02


def validate_batch(logits, frozen_logits, candidate_mask, positive_mask, verified_mask,
                   teacher_index, student_index):
    """Reject malformed or nonfinite host data; return no transformed labels.

    ``candidate_mask`` denotes all real controller candidates, *not* the full
    set of official legal actions. ``positive_mask`` and ``verified_mask`` are
    supervision-only masks produced by an independently authenticated probe.
    This function validates their structure, not the upstream evidence.
    """
    if (not isinstance(logits, np.ndarray) or logits.ndim != 2 or not logits.size
            or logits.dtype.kind != 'f' or not np.isfinite(logits).all()
            or not isinstance(frozen_logits, np.ndarray) or frozen_logits.shape != logits.shape
            or frozen_logits.dtype != logits.dtype or not np.isfinite(frozen_logits).all()):
        raise ValueError('Invalid finite floating-point logits')
    rows, width = logits.shape
    for mask in (candidate_mask, positive_mask, verified_mask):
        if not isinstance(mask, np.ndarray) or mask.shape != logits.shape or mask.dtype != np.dtype(bool):
            raise ValueError('Invalid boolean candidate mask')
    for indices in (teacher_index, student_index):
        if (not isinstance(indices, np.ndarray) or indices.shape != (rows,) or indices.dtype.kind not in 'iu'
                or np.any(indices < 0) or np.any(indices >= width)):
            raise ValueError('Invalid teacher or student index')
    if (np.any(verified_mask & ~candidate_mask) or np.any(positive_mask & ~verified_mask)
            or not np.all(np.any(candidate_mask, axis=1))):
        raise ValueError('Candidate and verified masks are inconsistent')
    expected_positive = np.zeros_like(candidate_mask)
    row = np.arange(rows)
    expected_positive[row, teacher_index] = True
    expected_positive[row, student_index] = True
    if (not np.array_equal(positive_mask, expected_positive)
            or not np.all(np.sum(verified_mask & ~positive_mask, axis=1) == 1)):
        raise ValueError('Each group needs the exact positive set and one verified contrast')


def _max_logsum(logits, mask, xp):
    """Stable masked logsumexp as separate maximum and centered log-sum."""
    maximum = xp.max(xp.where(mask, logits, -xp.inf), axis=1)
    # Remove padding *before* subtraction so arbitrary padding cannot affect
    # the calculation or overflow while evaluating an unselected branch.
    safe = xp.where(mask, logits, maximum[:, None])
    shifted = xp.where(mask, safe - maximum[:, None], -xp.inf)
    logsum = xp.log(xp.sum(xp.exp(shifted), axis=1))
    return maximum, logsum


def _log_probabilities(logits, candidate_mask, xp):
    maximum, logsum = _max_logsum(logits, candidate_mask, xp)
    safe = xp.where(candidate_mask, logits, maximum[:, None])
    return xp.where(candidate_mask, safe - maximum[:, None] - logsum[:, None], 0.0)


def _frozen(value, xp):
    if xp is np:
        return value
    if getattr(xp, '__name__', None) != 'jax.numpy':
        raise ValueError('Only NumPy and JAX NumPy backends are supported')
    from jax import lax
    return lax.stop_gradient(value)


def per_group(logits, frozen_logits, candidate_mask, positive_mask, verified_mask,
              teacher_index, student_index, *, xp=np):
    """Return (teacher-only, positive-set, KL) arrays, one entry per group.

    In NumPy mode validation is automatic. JAX callers must first validate
    corresponding host arrays, and separately check numerical outputs/gradients
    after evaluation. Frozen logits have an explicit stop-gradient in JAX.
    """
    if xp is np:
        validate_batch(logits, frozen_logits, candidate_mask, positive_mask,
                       verified_mask, teacher_index, student_index)
    frozen_logits = _frozen(frozen_logits, xp)
    vmax, vlog = _max_logsum(logits, verified_mask, xp)
    pmax, plog = _max_logsum(logits, positive_mask, xp)
    teacher = logits[xp.arange(logits.shape[0]), teacher_index]
    teacher_loss = (vmax - teacher) + vlog
    set_loss = (vmax - pmax) + (vlog - plog)

    old_logp = _log_probabilities(frozen_logits, candidate_mask, xp)
    new_logp = _log_probabilities(logits, candidate_mask, xp)
    old_probability = xp.where(candidate_mask, xp.exp(old_logp), 0.0)
    # Padded log-probabilities were set to zero above, avoiding 0 * infinity.
    kl = xp.sum(old_probability * (old_logp - new_logp), axis=1)
    if xp is np:
        validate_finite_outputs(teacher_loss, set_loss, kl)
    return teacher_loss, set_loss, kl


def objective(logits, frozen_logits, candidate_mask, positive_mask, verified_mask,
              teacher_index, student_index, *, arm, xp=np):
    """Mean primary loss plus 0.02 * mean KL(old || current) over candidates."""
    if arm not in ('control', 'set'):
        raise ValueError('Unknown comparison arm')
    teacher, grouped, kl = per_group(logits, frozen_logits, candidate_mask, positive_mask,
                                     verified_mask, teacher_index, student_index, xp=xp)
    primary = teacher if arm == 'control' else grouped
    mean_primary, mean_kl = xp.mean(primary), xp.mean(kl)
    value = mean_primary + KL_WEIGHT * mean_kl
    if xp is np:
        validate_finite_outputs(value, mean_primary, mean_kl)
    return value, (mean_primary, mean_kl)


def validate_finite_outputs(*values):
    """Host-side stop check for every loss, gradient and optimizer-state leaf."""
    if not values:
        raise ValueError('At least one numerical output is required')
    for value in values:
        array = np.asarray(value)
        if array.dtype.kind not in 'fiu' or not np.isfinite(array).all():
            raise ValueError('Nonfinite or nonnumeric optimization output')
