"""Compose existing paired-update primitives without sources, files or logging.

The caller must authenticate the training mapping, schema, model, normalization
and class weights, choose and authenticate its explicit device, fix a budget,
and supervise the process. This function is not a production experiment entry.
"""

from dataclasses import dataclass
import numbers
from types import MappingProxyType

import numpy as np

from learning.task_pair_training_v1 import batching, fit_core, packing, targets


_ARRAYS = {
    'shared': 'float32', 'candidates': 'float32', 'offsets': 'int64', 'labels': 'int64',
    'selected_ops': 'int8', 'candidate_ops': 'int8', 'candidate_identity': 'int16', 'seeds': 'int64',
}
_RESULT_FIELDS = frozenset(('control_params', 'experiment_params', 'steps', 'completed_updates',
                            'aux_nonzero_updates', 'control_changed_from_initial',
                            'experiment_changed_from_initial', 'arms_differ', 'device_platform'))
_STEP_FIELDS = frozenset(('aux_gradient_nonzero', 'control_changed', 'experiment_changed', 'arms_differ'))


@dataclass(frozen=True, repr=False)
class PairResult:
    """In-memory weights and coarse update evidence; never contains group IDs."""
    control_params: object
    experiment_params: object
    completed_updates: int
    auxiliary_nonzero_updates: int
    both_arms_changed: bool
    arms_differ: bool

    def __repr__(self):
        return '<PrivatePairedUpdateResult>'


def _mapping_snapshot(data):
    """Snapshot only the keys and references, never the underlying arrays."""
    if type(data) is not MappingProxyType or set(data) != set(_ARRAYS):
        raise ValueError('Expected read-only training mapping')
    values = {key: data[key] for key in _ARRAYS}
    for key, dtype in _ARRAYS.items():
        array = values[key]
        if (not isinstance(array, np.ndarray) or array.dtype != np.dtype(dtype)
                or array.flags.writeable):
            raise ValueError('Training arrays differ')
    if values['labels'].ndim != 1 or values['candidate_ops'].ndim != 1:
        raise ValueError('Training mapping shape differs')
    n, m = len(values['labels']), len(values['candidate_ops'])
    shapes = {'shared': (n, 275), 'candidates': (m, 98), 'offsets': (n + 1,),
              'labels': (n,), 'selected_ops': (n,), 'candidate_ops': (m,),
              'candidate_identity': (m, 3), 'seeds': (n,)}
    if n < 1 or m < n or any(values[key].shape != shape for key, shape in shapes.items()):
        raise ValueError('Training mapping shape differs')
    return MappingProxyType(values)


def _vector_snapshot(value, length, kind):
    """Bound a normalization/weight vector before copying it into private memory."""
    if isinstance(value, np.ndarray):
        if value.shape != (length,) or value.dtype.kind != 'f' or value.dtype.itemsize > 32:
            raise ValueError('Training vector shape or dtype differs')
        array = np.asarray(value)
    elif type(value) in (list, tuple):
        if len(value) != length or not all(isinstance(item, numbers.Real) for item in value):
            raise ValueError('Training vector shape or dtype differs')
        array = np.asarray(value)
        if array.shape != (length,) or array.dtype.kind != 'f' or array.dtype.itemsize > 32:
            raise ValueError('Training vector shape or dtype differs')
    else:
        raise ValueError('Training vector shape or dtype differs')
    result = np.array(array, copy=True)
    if (not np.isfinite(result).all()
            or (kind == 'scale' and np.any(result < 1))
            or (kind == 'weight' and np.any(result <= 0))):
        raise ValueError('Training vector values differ')
    result.setflags(write=False)
    return result


def _state_snapshot(params, mean, scale, class_weights):
    """Copy bounded initial weights and normalization before external callbacks."""
    if len(params) != 6 or set(params) != {'w0', 'b0', 'w1', 'b1', 'w2', 'b2'}:
        raise ValueError('Initial model fields differ')
    initial = dict(params)
    if fit_core._validate_initial_before_copy(initial) != 373:
        raise ValueError('Training input width differs')
    initial = {key: np.array(value, copy=True) for key, value in initial.items()}
    fit_core._validate_initial_before_copy(initial)
    for value in initial.values():
        value.setflags(write=False)
    return (initial, _vector_snapshot(mean, 373, 'mean'),
            _vector_snapshot(scale, 373, 'scale'), _vector_snapshot(class_weights, 5, 'weight'))


def train_pair(data, params, mean, scale, class_weights, *, device, rng_seed,
               updates, batch_size, check):
    """Execute one fixed caller-specified paired budget and return private weights.

    ``data`` is the already authenticated R8-style read-only mapping; eligibility
    is derived exclusively from that mapping, and both arms use one batch plan.
    The sampled plan, cluster identities and RNG seed are not returned. ``check``
    is a caller-supplied sampled cancellation hook, not a hard deadline.

    This pure composition does not authenticate the source, model or device,
    persist results, enforce an OS process budget, or evaluate a game. Its tests
    use synthetic data, exactly two 8-group batches, and an explicitly selected
    CPU. A future production entry must separately fix 64x256 and RTX identity.
    """
    if (type(updates) is not int or not 1 <= updates <= batching.MAX_UPDATES
            or type(batch_size) is not int or not 2 <= batch_size <= 256
            or type(rng_seed) is not int or not 0 <= rng_seed <= batching.MAX_SEED
            or not callable(check) or type(params) not in (dict, MappingProxyType)):
        raise ValueError('Invalid paired budget or input')
    initial_params, frozen_mean, frozen_scale, frozen_weights = _state_snapshot(
        params, mean, scale, class_weights)
    check()
    source = _mapping_snapshot(data)
    eligibility = targets.qualifying_groups(source, width=256, chunk_size=4096)
    check()
    batches, _ = batching.make_batches(eligibility, source['seeds'], rng_seed=rng_seed,
                                       updates=updates, batch_size=batch_size)
    check()
    packed = 0

    def pack(ids):
        nonlocal packed
        parts = packing.pack_training_batch(source, ids, frozen_mean, frozen_scale,
                                            frozen_weights, eligibility)
        packed += 1
        return parts

    result = fit_core.fit_pair(initial_params, batches, pack, device=device, check=check)
    if (type(result) is not dict or set(result) != _RESULT_FIELDS
            or type(result['completed_updates']) is not int
            or result['completed_updates'] != updates or packed != updates
            or type(result['steps']) is not list or len(result['steps']) != updates
            or type(result['aux_nonzero_updates']) is not int
            or type(result['control_changed_from_initial']) is not bool
            or type(result['experiment_changed_from_initial']) is not bool
            or type(result['arms_differ']) is not bool
            or result['control_changed_from_initial'] is not True
            or result['experiment_changed_from_initial'] is not True
            or result['arms_differ'] is not True
            or not all(type(step) is dict and set(step) == _STEP_FIELDS
                       and all(type(step[key]) is bool for key in _STEP_FIELDS)
                       and step['control_changed'] and step['experiment_changed']
                       for step in result['steps'])
            or result['aux_nonzero_updates'] != sum(step['aux_gradient_nonzero'] for step in result['steps'])
            or result['aux_nonzero_updates'] < 1
            or result['device_platform'] != getattr(device, 'platform', None)
            or type(result['control_params']) is not dict
            or type(result['experiment_params']) is not dict):
        raise ValueError('Paired update evidence is incomplete')
    return PairResult(MappingProxyType(result['control_params'].copy()),
                      MappingProxyType(result['experiment_params'].copy()),
                      result['completed_updates'], result['aux_nonzero_updates'], True, True)
