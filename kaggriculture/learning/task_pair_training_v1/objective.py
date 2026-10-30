"""Pure two-arm task-head objectives for an already prepared microbatch.

The original task network has three affine layers and ReLU after the first two.
This module neither reads data nor performs optimization, device selection or
inference in a game. Target masks, eligibility and labels are supervision only;
they are never network features. Callers must bind them to authenticated input
and validate the complete effective batch before differentiating on any device.
"""

import numbers

import numpy as np

from learning.task_bc_v1.train import weighted_group_loss
from learning.task_pair_training_v1.loss import conditional_worker_loss
from learning.task_pair_training_v1.loss import validate_inputs as validate_worker_inputs


DEFAULT_WORKER_COEFFICIENT = 0.25


def validate_parameters(params, input_width):
    """Validate host layer consistency; the experiment binds exact dimensions.

    Accepting consistent small dimensions makes independent synthetic tests
    possible. Production callers must separately require the original fixed
    task model's dimensions and authenticated initialization.
    """
    if (type(params) is not dict or set(params) != {'w0', 'b0', 'w1', 'b1', 'w2', 'b2'}
            or isinstance(input_width, (bool, np.bool_)) or not isinstance(input_width, numbers.Integral)
            or input_width < 1):
        raise ValueError('Invalid task model parameters')
    current = int(input_width)
    for layer in range(3):
        weights, bias = params['w' + str(layer)], params['b' + str(layer)]
        if (not isinstance(weights, np.ndarray) or not isinstance(bias, np.ndarray)
                or weights.ndim != 2 or bias.ndim != 1 or weights.shape[0] != current
                or weights.shape[1] < 1 or bias.shape != (weights.shape[1],)
                or weights.dtype.kind != 'f' or bias.dtype.kind != 'f'
                or not np.isfinite(weights).all() or not np.isfinite(bias).all()):
            raise ValueError('Invalid task layer')
        current = weights.shape[1]
    if current != 1:
        raise ValueError('The task network must output one candidate score')


def validate_inputs(features, candidate_mask, labels, group_weights, ce_denominator,
                    target_mask, eligible, eligible_count,
                    worker_coefficient=DEFAULT_WORKER_COEFFICIENT):
    """Check host inputs for a microbatch before the backend objective is used.

    The two denominators describe the entire effective batch. This check can
    reject a denominator too small for this microbatch, but the caller must also
    verify exact equality with the total weights and eligible rows accumulated
    over *all* its microbatches. Padding uses one real dummy candidate, label 0,
    zero group weight and no target/eligible contribution.
    """
    if (not isinstance(features, np.ndarray) or features.ndim != 3
            or features.dtype.kind != 'f' or min(features.shape) < 1
            or not np.isfinite(features).all()):
        raise ValueError('Invalid feature array')
    rows, width, _ = features.shape
    if (not isinstance(candidate_mask, np.ndarray) or candidate_mask.shape != (rows, width)
            or candidate_mask.dtype != np.dtype(bool)
            or not np.all(np.any(candidate_mask, axis=1))):
        raise ValueError('Invalid candidate mask')
    if (not isinstance(labels, np.ndarray) or labels.shape != (rows,)
            or labels.dtype.kind not in 'iu' or np.any(labels >= width)
            or (labels.dtype.kind == 'i' and np.any(labels < 0))):
        raise ValueError('Invalid label indices')
    if not np.all(candidate_mask[np.arange(rows), labels]):
        raise ValueError('Teacher label must be a real candidate')
    if (not isinstance(group_weights, np.ndarray) or group_weights.shape != (rows,)
            or group_weights.dtype.kind != 'f' or not np.isfinite(group_weights).all()
            or np.any(group_weights < 0)):
        raise ValueError('Invalid group weights')
    if (isinstance(ce_denominator, (bool, np.bool_)) or not isinstance(ce_denominator, numbers.Real)
            or not np.isfinite(ce_denominator) or ce_denominator <= 0):
        raise ValueError('Invalid effective-batch class-weight denominator')
    local_weight = float(np.sum(group_weights, dtype=np.float64))
    if local_weight > float(ce_denominator) + 1e-6 * max(1.0, abs(float(ce_denominator))):
        raise ValueError('The effective-batch class-weight denominator is too small')
    if (isinstance(worker_coefficient, (bool, np.bool_))
            or not isinstance(worker_coefficient, numbers.Real)
            or not np.isfinite(worker_coefficient) or worker_coefficient < 0):
        raise ValueError('Invalid worker coefficient')
    if not isinstance(target_mask, np.ndarray) or not isinstance(eligible, np.ndarray):
        raise ValueError('Target mask and eligibility must be host arrays')
    validate_worker_inputs(np.zeros((rows, width), dtype=np.float32), target_mask,
                           labels, eligible, eligible_count)
    if np.any(target_mask & ~candidate_mask) or np.any(eligible & (group_weights <= 0)):
        raise ValueError('Target mask or eligible weight is invalid')
    padded = group_weights == 0
    if np.any(padded):
        dummy = np.zeros((int(np.count_nonzero(padded)), width), dtype=bool)
        dummy[:, 0] = True
        if (not np.array_equal(candidate_mask[padded], dummy)
                or np.any(labels[padded] != 0) or np.any(target_mask[padded])
                or np.any(eligible[padded])):
            raise ValueError('Padding must use one harmless dummy candidate')


def forward(params, features, xp=np):
    """Three-layer task network with the original zero-slope ReLU at zero."""
    x = features
    for layer in range(3):
        x = x @ params['w' + str(layer)] + params['b' + str(layer)]
        if layer < 2:
            # jax.nn.relu has derivative 0 at 0; maximum has derivative 0.5.
            x = xp.where(x > 0, x, xp.zeros_like(x))
    return x[..., 0]


def ce_only_objective(params, features, candidate_mask, labels, group_weights,
                      ce_denominator, xp=np):
    """Control objective: forward and original full-candidate weighted CE only."""
    return weighted_group_loss(forward(params, features, xp), candidate_mask, labels,
                               group_weights, ce_denominator, xp)


def objective_components(params, features, candidate_mask, labels, group_weights,
                         ce_denominator, target_mask, eligible, eligible_count, xp=np):
    """Return this microbatch's separately normalized CE and auxiliary terms.

    Both denominators are for the full effective batch. Call the host validator
    first; this backend-generic function is suitable for JAX differentiation/JIT.
    """
    logits = forward(params, features, xp)
    ce = weighted_group_loss(logits, candidate_mask, labels, group_weights,
                             ce_denominator, xp)
    worker = conditional_worker_loss(logits, target_mask, labels, eligible,
                                     eligible_count, xp)
    return ce, worker


def combined_objective(params, features, candidate_mask, labels, group_weights,
                       ce_denominator, target_mask, eligible, eligible_count,
                       worker_coefficient=DEFAULT_WORKER_COEFFICIENT, xp=np):
    """Add the worker loss; even a zero coefficient still evaluates that path.

    Use ``ce_only_objective`` for a true control arm that must not evaluate the
    worker path, particularly when an invalid auxiliary input would fail.
    """
    ce, worker = objective_components(params, features, candidate_mask, labels,
                                      group_weights, ce_denominator, target_mask,
                                      eligible, eligible_count, xp)
    return ce + worker_coefficient * worker
