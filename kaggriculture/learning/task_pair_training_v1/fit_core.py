"""In-memory paired updates for a caller-authenticated task-head experiment.

The caller must separately authenticate the split, feature schema, initial
parameters, batch plan and device identity; enforce the registered production
budget; and handle process deadlines, private output and checkpoint storage.
This module never selects a device, reads data or writes files. Supervision is
accepted separately from features and is used only by the loss functions.
"""

import numpy as np

from learning.task_pair_training_v1 import objective


MICRO_KEYS = frozenset(('ids', 'features', 'candidate_mask', 'labels',
                        'group_weights', 'target_mask', 'eligible'))
MAX_UPDATES = 1024
MAX_BATCH = 4096
MAX_MICROBATCH_CANDIDATES = 65536
MAX_BATCH_PADDED_CANDIDATES = 1 << 18
MAX_INPUT_WIDTH = 1024
MAX_LAYER_WIDTH = 1024
MAX_MODEL_BYTES = 8 * 1024 * 1024
WORKER_COEFFICIENT = 0.25
AUX_NONZERO_THRESHOLD = 1e-8


def _validate_initial_before_copy(params):
    """Bound logical parameter shapes and bytes before scanning or copying."""
    names = {'w0', 'b0', 'w1', 'b1', 'w2', 'b2'}
    if (type(params) is not dict or set(params) != names
            or not all(isinstance(value, np.ndarray) and value.dtype == np.dtype('float32')
                       for value in params.values())):
        raise ValueError('Initial model must contain exactly the float32 parameter arrays')
    input_width = None
    previous = None
    logical_bytes = 0
    for layer in range(3):
        weights, bias = params['w' + str(layer)], params['b' + str(layer)]
        if (weights.ndim != 2 or bias.ndim != 1
                or not 1 <= weights.shape[0] <= MAX_LAYER_WIDTH
                or not 1 <= weights.shape[1] <= MAX_LAYER_WIDTH
                or bias.shape != (weights.shape[1],)
                or (previous is not None and weights.shape[0] != previous)):
            raise ValueError('Invalid or oversized initial layer dimensions')
        if layer == 0:
            input_width = weights.shape[0]
            if input_width > MAX_INPUT_WIDTH:
                raise ValueError('Invalid or oversized input width')
        previous = weights.shape[1]
        logical_bytes += int(weights.nbytes) + int(bias.nbytes)
        if logical_bytes > MAX_MODEL_BYTES:
            raise ValueError('Initial model exceeds its logical byte budget')
    if previous != 1:
        raise ValueError('The task network must output one candidate score')
    # Once dimensions and aggregate bytes are bounded, scan for nonfinite
    # values using the same structural rule as the original objective.
    objective.validate_parameters(params, input_width)
    return input_width


def _snapshot_batch(plan, raw_parts, input_width):
    """Copy and validate one shared effective batch; never return identifiers."""
    if type(raw_parts) not in (tuple, list) or not 1 <= len(raw_parts) <= len(plan):
        raise ValueError('Invalid microbatch collection')
    padded_candidates = 0
    # Bound every source array before making snapshots; callback data is not a
    # reason to allocate arbitrary arrays inside the core.
    for part in raw_parts:
        if (type(part) is not dict or set(part) != MICRO_KEYS
                or any(not isinstance(part[name], np.ndarray) for name in MICRO_KEYS)):
            raise ValueError('Invalid microbatch fields')
        ids, features = part['ids'], part['features']
        if (ids.ndim != 1 or ids.dtype.kind not in 'iu' or not len(ids)
                or features.ndim != 3 or features.dtype != np.dtype('float32')
                or features.shape[0] < len(ids) or features.shape[0] > MAX_BATCH
                or not 1 <= features.shape[1] <= 256
                or features.shape[0] * features.shape[1] > MAX_MICROBATCH_CANDIDATES
                or features.shape[2] != input_width):
            raise ValueError('Invalid microbatch dimensions or identities')
        capacity, width = features.shape[:2]
        if (part['candidate_mask'].shape != (capacity, width)
                or part['candidate_mask'].dtype != np.dtype(bool)
                or part['target_mask'].shape != (capacity, width)
                or part['target_mask'].dtype != np.dtype(bool)
                or part['labels'].shape != (capacity,) or part['labels'].dtype != np.dtype('int32')
                or part['group_weights'].shape != (capacity,)
                or part['group_weights'].dtype != np.dtype('float32')
                or part['eligible'].shape != (capacity,) or part['eligible'].dtype != np.dtype(bool)):
            raise ValueError('Invalid microbatch arrays')
        padded_candidates += capacity * width
        if padded_candidates > MAX_BATCH_PADDED_CANDIDATES:
            raise ValueError('Effective batch exceeds padded candidate budget')
    parts, identifiers = [], []
    for part in raw_parts:
        copied = {name: np.array(part[name], copy=True) for name in MICRO_KEYS}
        ids = copied['ids']
        features = copied['features']
        if (ids.ndim != 1 or ids.dtype.kind not in 'iu' or not len(ids)
                or np.any(ids < 0) or len(np.unique(ids)) != len(ids)
                or features.ndim != 3 or features.shape[0] < len(ids)):
            raise ValueError('Invalid microbatch dimensions or identities')
        weights = copied['group_weights']
        if (weights.shape != (features.shape[0],) or weights.dtype.kind != 'f'
                or np.any(weights[:len(ids)] <= 0) or np.any(weights[len(ids):] != 0)):
            raise ValueError('Real groups and padding differ')
        identifiers.extend(int(value) for value in ids)
        parts.append(copied)
    if len(identifiers) != len(plan) or len(set(identifiers)) != len(identifiers):
        raise ValueError('Microbatches must cover each planned group once')
    # Bucketed microbatches may reorder the plan, but never change membership.
    if sorted(identifiers) != sorted(int(value) for value in plan):
        raise ValueError('Microbatch identities differ from the plan')
    denominator = sum(float(np.sum(part['group_weights'], dtype=np.float64)) for part in parts)
    eligible_count = sum(int(np.count_nonzero(part['eligible'])) for part in parts)
    if not np.isfinite(denominator) or denominator <= 0 or eligible_count < 1:
        raise ValueError('The effective batch has no eligible worker target')
    ce_denom = np.float32(denominator)
    if not np.isfinite(ce_denom) or ce_denom <= 0:
        raise ValueError('Invalid effective batch weight')
    for part in parts:
        objective.validate_inputs(part['features'], part['candidate_mask'], part['labels'],
                                  part['group_weights'], float(ce_denom), part['target_mask'],
                                  part['eligible'], eligible_count, WORKER_COEFFICIENT)
    return parts, ce_denom, eligible_count


def fit_pair(initial_params, batches, pack_batch, *, device, check):
    """Run paired pure JAX updates on one explicit caller-selected device.

    ``pack_batch`` is called once per plan row and returns a list/tuple of host
    dictionaries with the exact ``MICRO_KEYS`` fields. IDs correspond to real
    rows at the start of each padded microbatch. A complete batch is copied,
    checked and reused for both arms; IDs must cover the plan exactly once.

    Returns private device-resident parameters and per-update boolean evidence;
    neither IDs nor random seeds are returned. ``check()`` may raise to cancel;
    it is sampled between operations and is not a hard kernel timeout. The caller
    must validate actual GPU identity, checkpoints and official game behavior.
    """
    import jax
    import jax.numpy as jnp
    import optax

    if (not isinstance(device, jax.Device) or device not in jax.devices(device.platform)
            or not callable(check) or not callable(pack_batch)):
        raise ValueError('Explicit available device and callbacks are required')
    if (not isinstance(batches, np.ndarray) or batches.ndim != 2
            or batches.dtype.kind not in 'iu' or not 1 <= batches.shape[0] <= MAX_UPDATES
            or not 1 <= batches.shape[1] <= MAX_BATCH or batches.size > 4_194_304
            or np.any(batches < 0)):
        raise ValueError('Invalid batch plan')
    plan = np.array(batches, copy=True)
    if any(len(np.unique(row)) != len(row) for row in plan):
        raise ValueError('Repeated group within a batch')
    input_width = _validate_initial_before_copy(initial_params)
    initial = {name: np.array(value, copy=True) for name, value in initial_params.items()}
    check()

    def put(tree):
        return jax.device_put(tree, device)

    def finite_on_device(tree):
        for leaf in jax.tree_util.tree_leaves(tree):
            if not isinstance(leaf, jax.Array) or leaf.devices() != {device}:
                raise ValueError('Execution moved off the selected device')
            leaf.block_until_ready()
            finite = jnp.all(jnp.isfinite(leaf))
            if finite.devices() != {device}:
                raise ValueError('Execution moved off the selected device')
            if not bool(np.asarray(finite.block_until_ready())):
                raise FloatingPointError('Nonfinite loss, gradient, parameter or optimizer state')

    def different(left, right):
        flags = jax.tree_util.tree_map(lambda a, b: jnp.any(a != b), left, right)
        finite_on_device(flags)
        return any(bool(np.asarray(flag)) for flag in jax.tree_util.tree_leaves(flags))

    def ce_loss(parameters, features, mask, labels, weights, denominator):
        return objective.ce_only_objective(parameters, features, mask, labels, weights,
                                           denominator, xp=jnp)

    def worker_loss(parameters, features, mask, labels, weights, denominator,
                    targets, eligible, eligible_count):
        return objective.objective_components(parameters, features, mask, labels, weights,
                                              denominator, targets, eligible, eligible_count,
                                              xp=jnp)[1]

    ce_grad = jax.jit(jax.value_and_grad(ce_loss))
    worker_grad = jax.jit(jax.value_and_grad(worker_loss))
    control = put({key: value.copy() for key, value in initial.items()})
    experiment = put({key: value.copy() for key, value in initial.items()})
    reference = put({key: value.copy() for key, value in initial.items()})
    control_opt = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(3e-4))
    experiment_opt = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(3e-4))
    # Optax creates scalar counters, which otherwise use JAX's default device
    # even when the supplied model is already on another explicitly chosen one.
    with jax.default_device(device):
        control_state = control_opt.init(control)
        experiment_state = experiment_opt.init(experiment)
        finite_on_device((control, experiment, reference, control_state, experiment_state))

    @jax.jit
    def update_control(params, state, gradient):
        delta, next_state = control_opt.update(gradient, state, params)
        return optax.apply_updates(params, delta), next_state

    @jax.jit
    def update_experiment(params, state, gradient):
        delta, next_state = experiment_opt.update(gradient, state, params)
        return optax.apply_updates(params, delta), next_state

    steps = []
    with jax.default_device(device), jax.default_matmul_precision('highest'):
        for row in plan:
            check()
            supplied = row.copy()
            supplied.setflags(write=False)
            parts, denominator, eligible_count = _snapshot_batch(row, pack_batch(supplied), input_width)
            check()
            denominator_dev = put(np.asarray(denominator, dtype=np.float32))
            eligible_dev = put(np.asarray(eligible_count, dtype=np.int32))
            control_total = jax.tree_util.tree_map(jnp.zeros_like, control)
            experiment_ce_total = jax.tree_util.tree_map(jnp.zeros_like, experiment)
            worker_total = jax.tree_util.tree_map(jnp.zeros_like, experiment)
            loss_a = put(np.float32(0))
            loss_b = put(np.float32(0))
            loss_worker = put(np.float32(0))
            for part in parts:
                check()
                x, mask, labels, weights, targets, eligible = put(tuple(
                    part[name] for name in ('features', 'candidate_mask', 'labels', 'group_weights',
                                            'target_mask', 'eligible')))
                finite_on_device((x, mask, labels, weights, targets, eligible, denominator_dev, eligible_dev))
                va, ga = ce_grad(control, x, mask, labels, weights, denominator_dev)
                vb, gb = ce_grad(experiment, x, mask, labels, weights, denominator_dev)
                vw, gw = worker_grad(experiment, x, mask, labels, weights, denominator_dev,
                                     targets, eligible, eligible_dev)
                finite_on_device((va, ga, vb, gb, vw, gw))
                control_total = jax.tree_util.tree_map(lambda a, b: a + b, control_total, ga)
                experiment_ce_total = jax.tree_util.tree_map(lambda a, b: a + b, experiment_ce_total, gb)
                worker_total = jax.tree_util.tree_map(lambda a, b: a + b, worker_total, gw)
                loss_a, loss_b, loss_worker = loss_a + va, loss_b + vb, loss_worker + vw
            experiment_total = jax.tree_util.tree_map(
                lambda ce, worker: ce + WORKER_COEFFICIENT * worker, experiment_ce_total, worker_total)
            worker_norm = optax.global_norm(worker_total)
            control_norm = optax.global_norm(control_total)
            experiment_norm = optax.global_norm(experiment_total)
            experiment_loss = loss_b + WORKER_COEFFICIENT * loss_worker
            # Both arms are checked and synchronized before either optimizer update.
            finite_on_device((loss_a, loss_b, loss_worker, experiment_loss, control_total, experiment_ce_total,
                              worker_total, experiment_total, worker_norm, control_norm,
                              experiment_norm, control, experiment, control_state, experiment_state))
            check()
            next_control, next_control_state = update_control(control, control_state, control_total)
            next_experiment, next_experiment_state = update_experiment(experiment, experiment_state,
                                                                        experiment_total)
            finite_on_device((next_control, next_control_state, next_experiment, next_experiment_state))
            steps.append({'aux_gradient_nonzero': float(np.asarray(worker_norm)) > AUX_NONZERO_THRESHOLD,
                          'control_changed': different(control, next_control),
                          'experiment_changed': different(experiment, next_experiment),
                          'arms_differ': different(next_control, next_experiment)})
            control, control_state = next_control, next_control_state
            experiment, experiment_state = next_experiment, next_experiment_state
            check()
    with jax.default_device(device):
        return {'control_params': control, 'experiment_params': experiment, 'steps': steps,
                'completed_updates': len(steps),
                'aux_nonzero_updates': sum(row['aux_gradient_nonzero'] for row in steps),
                'control_changed_from_initial': different(reference, control),
                'experiment_changed_from_initial': different(reference, experiment),
                'arms_differ': different(control, experiment),
                'device_platform': device.platform}
