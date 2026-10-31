"""Full-candidate teacher CE plus a verified-request comparison auxiliary.

The masks are supervision only, never policy features. Unknown candidates
remain in the full teacher CE and are excluded only from the auxiliary. A
verified different request is not a claim of worse utility or negative reward.
"""
import numpy as np


AUXILIARY_WEIGHT = 0.25


def validate_batch(logits, candidate_mask, positive_mask, verified_mask, labels, weights):
    """Validate host arrays before running the backend-generic objective."""
    if (not isinstance(logits, np.ndarray) or logits.ndim != 2
            or logits.dtype.kind != 'f' or not logits.size or not np.isfinite(logits).all()):
        raise ValueError('Invalid logits')
    rows, width = logits.shape
    for mask in (candidate_mask, positive_mask, verified_mask):
        if not isinstance(mask, np.ndarray) or mask.shape != (rows, width) or mask.dtype != np.dtype(bool):
            raise ValueError('Invalid candidate mask')
    if (np.any(positive_mask & ~verified_mask) or np.any(verified_mask & ~candidate_mask)
            or not np.all(np.any(candidate_mask, axis=1))):
        raise ValueError('Inconsistent verification masks')
    if (not isinstance(labels, np.ndarray) or labels.shape != (rows,) or labels.dtype.kind not in 'iu'
            or np.any(labels < 0) or np.any(labels >= width)
            or not np.all(positive_mask[np.arange(rows), labels])):
        raise ValueError('Teacher must be a verified positive candidate')
    if (not isinstance(weights, np.ndarray) or weights.shape != (rows,) or weights.dtype.kind != 'f'
            or not np.isfinite(weights).all() or np.any(weights <= 0)
            or not np.isfinite(np.sum(weights, dtype=np.float64))):
        raise ValueError('Invalid group weights')


def _max_and_logsum(scores, mask, xp):
    selected = xp.where(mask, scores, -xp.inf)
    maximum = xp.max(selected, axis=1)
    shifted = xp.where(mask, scores - maximum[:, None], -xp.inf)
    return maximum, xp.log(xp.sum(xp.exp(shifted), axis=1))


def components(logits, candidate_mask, positive_mask, verified_mask, labels, weights, xp=np):
    """Return weighted means and eligibility, with a common all-group denominator.

    Inputs must first pass ``validate_batch``. All masks are nonempty because
    they include the teacher. Auxiliary terms are exactly zero unless P has at
    least two verified positives and U contains a verified different request.
    A finite-input extreme may overflow float32; callers must reject nonfinite
    computed loss/gradient values before any optimization step.
    """
    full_max, full_logsum = _max_and_logsum(logits, candidate_mask, xp)
    verified_max, verified_logsum = _max_and_logsum(logits, verified_mask, xp)
    positive_max, positive_logsum = _max_and_logsum(logits, positive_mask, xp)
    teacher = logits[xp.arange(logits.shape[0]), labels]
    ce = (full_max - teacher) + full_logsum
    control_aux = (verified_max - teacher) + verified_logsum
    experiment_aux = ((verified_max - positive_max) + verified_logsum - positive_logsum)
    eligible = (xp.sum(positive_mask, axis=1) >= 2) & xp.any(verified_mask & ~positive_mask, axis=1)
    denominator = xp.sum(weights)
    return (xp.sum(weights * ce) / denominator,
            xp.sum(weights * xp.where(eligible, control_aux, 0)) / denominator,
            xp.sum(weights * xp.where(eligible, experiment_aux, 0)) / denominator,
            eligible)


def objective(logits, candidate_mask, positive_mask, verified_mask, labels, weights, *, arm, xp=np):
    if arm not in ('control', 'experiment'):
        raise ValueError('Unknown arm')
    ce, control, experiment, eligible = components(logits, candidate_mask, positive_mask,
                                                    verified_mask, labels, weights, xp=xp)
    aux = control if arm == 'control' else experiment
    return ce + AUXILIARY_WEIGHT * aux, (ce, aux, xp.sum(eligible))
