"""Bounded WATER-only residual ranking on authenticated in-memory examples.

The caller authenticates the source, task schema, baseline logits, role split,
seed and device. In particular, it must include only turns where the frozen
controller's *global* choice is WATER, and map the teacher target into that
turn's ordered WATER subset. This module does not open files, games or models.

Training receives float32 copies of the baseline logits. Within each group it
subtracts the first real candidate's normalized features before the dot
product; the resulting common-score shift cancels mathematically in CE and KL.
This removes roundoff-only gradients on features identical within a group.
Deployment retains the original float64 logits and computes an uncentered
float32 residual, so operation order and near ties can differ. The caller must
check actual CPU/GPU scores and deployed choices separately.
"""
from dataclasses import dataclass

import numpy as np

from decision.api import PRODUCTS
from learning.task_choice_v1.features import OPS


DIM = 373
SHARED_DIM = 275
OP_START = SHARED_DIM + 5 + len(PRODUCTS)
MAX_GROUPS = 384
MAX_CANDIDATES = 256
GROUPS_PER_ROLE = 8
UPDATES = 64
LEARNING_RATE = 1e-4
KL_WEIGHT = 0.1
CLIP_NORM = 1.0


class WaterFitError(ValueError):
    """An in-memory sample, batch or optimizer contract is invalid."""


def _require(value):
    if not value:
        raise WaterFitError('Invalid private WATER fit input')


def _readonly(array):
    """Copy into an immutable byte buffer; caller references cannot mutate it."""
    data = np.ascontiguousarray(array)
    return np.frombuffer(data.tobytes(order='C'), dtype=data.dtype).reshape(data.shape)


@dataclass(frozen=True, slots=True, repr=False)
class _Group:
    role: int
    features: np.ndarray
    baseline_logits: np.ndarray
    teacher_index: int

    def __repr__(self):
        return '<PrivateWaterFitGroup>'


@dataclass(frozen=True, slots=True, repr=False)
class _Batch:
    features: np.ndarray
    baseline_logits: np.ndarray
    mask: np.ndarray
    teacher_index: np.ndarray

    def __repr__(self):
        return '<PrivateWaterFitBatch>'


@dataclass(frozen=True, slots=True, repr=False)
class FitResult:
    """Private learned vector and minimal synchronized-update evidence."""
    weights: np.ndarray
    updates_completed: int
    nonzero_gradient_seen: bool
    weights_changed: bool

    def __repr__(self):
        return '<PrivateWaterFitResult>'


def prepare_groups(groups, *, mean, scale, guard):
    """Snapshot and normalize two bounded role pools without JAX or I/O.

    Each input dictionary has exactly role, features, baseline_logits and
    teacher_index. Features are deployment-visible raw float32 values; labels
    and role never enter their columns. Source provenance and the global WATER
    routing condition must already have been verified by the caller.
    """
    _require(callable(guard) and type(groups) in (tuple, list)
             and 2 * GROUPS_PER_ROLE <= len(groups) <= MAX_GROUPS)
    guard()
    _require(type(mean) is np.ndarray and mean.dtype == np.dtype('float64') and mean.shape == (DIM,)
             and type(scale) is np.ndarray and scale.dtype == np.dtype('float64') and scale.shape == (DIM,))
    mu, sigma = mean.copy(), scale.copy()
    _require(np.isfinite(mu).all() and np.isfinite(sigma).all() and np.all(sigma >= 1))
    prepared, counts = [], [0, 0]
    for group in groups:
        guard()
        _require(type(group) is dict and set(group) == {'role', 'features', 'baseline_logits', 'teacher_index'})
        role, x, base, label = (group[k] for k in ('role', 'features', 'baseline_logits', 'teacher_index'))
        _require(type(role) is int and role in (0, 1) and type(x) is np.ndarray
                 and x.dtype == np.dtype('float32') and x.ndim == 2 and x.shape[1] == DIM
                 and 2 <= len(x) <= MAX_CANDIDATES and type(base) is np.ndarray
                 and base.dtype == np.dtype('float32') and base.shape == (len(x),)
                 and type(label) is int and 0 <= label < len(x))
        raw, frozen = np.array(x, copy=True, order='C'), np.array(base, copy=True, order='C')
        _require(np.isfinite(raw).all() and np.isfinite(frozen).all()
                 and np.array_equal(raw[:, :SHARED_DIM],
                                    np.broadcast_to(raw[:1, :SHARED_DIM], (len(raw), SHARED_DIM))))
        ops = raw[:, OP_START:OP_START + len(OPS)]
        water = OPS.index('WATER')
        _require(ops.shape == (len(raw), len(OPS)) and np.isin(ops, (0.0, 1.0)).all()
                 and np.all(ops.sum(axis=1) == 1.0) and np.all(ops[:, water] == 1.0))
        with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
            normalized = ((raw.astype(np.float64) - mu) / sigma).astype(np.float32)
        _require(normalized.shape == raw.shape and np.isfinite(normalized).all())
        prepared.append(_Group(role, _readonly(normalized), _readonly(frozen), label))
        counts[role] += 1
    _require(all(count >= GROUPS_PER_ROLE for count in counts))
    guard()
    return tuple(prepared)


def sample_indices(prepared, rng):
    """Draw eight distinct groups per role from a caller-owned fixed RNG."""
    _require(type(prepared) is tuple and isinstance(rng, np.random.Generator)
             and all(type(group) is _Group for group in prepared))
    roles = [[i for i, group in enumerate(prepared) if group.role == role] for role in (0, 1)]
    _require(all(len(items) >= GROUPS_PER_ROLE for items in roles))
    return np.concatenate([rng.choice(items, size=GROUPS_PER_ROLE, replace=False) for items in roles])


def make_batch(prepared, indices, *, guard):
    """Pad candidate groups only; metadata and labels are not input features."""
    _require(callable(guard) and type(prepared) is tuple and prepared
             and all(type(group) is _Group for group in prepared)
             and type(indices) is np.ndarray and indices.shape == (2 * GROUPS_PER_ROLE,)
             and indices.dtype.kind in 'iu' and np.all(indices >= 0) and np.all(indices < len(prepared))
             and len(set(map(int, indices))) == len(indices))
    guard()
    selected = [prepared[int(index)] for index in indices]
    _require(sum(group.role == 0 for group in selected) == GROUPS_PER_ROLE
             and sum(group.role == 1 for group in selected) == GROUPS_PER_ROLE)
    # One fixed width across the whole fit avoids compiling new shapes per step.
    width = max(len(group.features) for group in prepared)
    _require(2 <= width <= MAX_CANDIDATES)
    features = np.zeros((len(selected), width, DIM), dtype=np.float32)
    base = np.zeros((len(selected), width), dtype=np.float32)
    mask = np.zeros((len(selected), width), dtype=bool)
    target = np.zeros(len(selected), dtype=np.int32)
    for row, group in enumerate(selected):
        guard()
        count = len(group.features)
        features[row, :count], base[row, :count] = group.features, group.baseline_logits
        mask[row, :count], target[row] = True, group.teacher_index
    _require(np.isfinite(features).all() and np.isfinite(base).all()
             and np.all(mask[:, 0]) and np.all(mask[np.arange(len(selected)), target]))
    return _Batch(features, base, mask, target)


def _log_probabilities(scores, mask, xp):
    maximum = xp.max(xp.where(mask, scores, -xp.inf), axis=1, keepdims=True)
    # Mask before subtracting, so padding cannot overflow an unselected branch.
    safe = xp.where(mask, scores, maximum)
    shifted = safe - maximum
    denominator = xp.sum(xp.where(mask, xp.exp(shifted), xp.zeros_like(shifted)), axis=1, keepdims=True)
    return xp.where(mask, shifted - xp.log(denominator), xp.zeros_like(shifted))


def _components(weights, features, baseline_logits, mask, teacher_index, xp):
    reference = features[:, :1, :]
    # Each first candidate is real. Mask before subtracting to ensure padding
    # cannot create an overflow or a gradient, even with extreme padded values.
    safe_features = xp.where(mask[:, :, None], features, reference)
    relative = safe_features - reference
    residual = xp.sum(relative * weights[None, None, :], axis=2)
    logits = baseline_logits + residual
    old = _log_probabilities(baseline_logits, mask, xp)
    new = _log_probabilities(logits, mask, xp)
    labels = new[xp.arange(new.shape[0]), teacher_index]
    probability = xp.where(mask, xp.exp(old), xp.zeros_like(old))
    kl = xp.sum(xp.where(mask, probability * (old - new), xp.zeros_like(old)), axis=1)
    ce_mean, kl_mean = xp.mean(-labels), xp.mean(kl)
    return ce_mean + KL_WEIGHT * kl_mean, (ce_mean, kl_mean)


def _numpy_components(weights, batch):
    """Pure numeric diagnostic for tests; not a model or public report."""
    _require(type(weights) is np.ndarray and weights.dtype == np.dtype('float32')
             and weights.shape == (DIM,) and np.isfinite(weights).all() and type(batch) is _Batch)
    with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
        value, (ce, kl) = _components(weights, batch.features, batch.baseline_logits,
                                      batch.mask, batch.teacher_index, np)
    _require(np.isfinite(value) and np.isfinite(ce) and np.isfinite(kl))
    return float(value), float(ce), float(kl)


def _ready(tree, device, jax, jnp):
    """Synchronize and require every array to remain finite on the fixed device."""
    for leaf in jax.tree_util.tree_leaves(tree):
        if not isinstance(leaf, jax.Array) or leaf.devices() != {device}:
            raise RuntimeError('WATER fit array left the registered device')
        leaf.block_until_ready()
        finite = jnp.all(jnp.isfinite(leaf))
        finite.block_until_ready()
        if not bool(np.asarray(finite)):
            raise FloatingPointError('WATER fit produced a nonfinite value')


def fit(groups, *, mean, scale, device, guard, rng_seed):
    """Run exactly 64 equal-role, synchronized Adam updates on one device.

    The caller verifies GPU identity and owns process and physical-memory
    limits. Supplying a CPU device is allowed for small synthetic tests; no
    automatic device discovery or CPU fallback is performed here. Failures and
    guard exceptions propagate and do not return a partial weight vector.
    """
    _require(callable(guard) and type(rng_seed) is int and 0 <= rng_seed < 2**32)
    guard()
    prepared = prepare_groups(groups, mean=mean, scale=scale, guard=guard)
    # Delay all JAX imports until after the caller's resource guard and sample checks.
    import jax
    import jax.numpy as jnp
    import optax

    _require(isinstance(device, jax.Device) and device in jax.devices(device.platform))
    rng = np.random.default_rng(rng_seed)
    optimizer = optax.chain(optax.clip_by_global_norm(CLIP_NORM), optax.adam(LEARNING_RATE))

    def objective(weights, features, baseline, mask, labels):
        return _components(weights, features, jax.lax.stop_gradient(baseline), mask, labels, jnp)

    value_grad = jax.jit(jax.value_and_grad(objective, has_aux=True))
    nonzero, completed = False, 0
    with jax.default_device(device), jax.default_matmul_precision('highest'):
        weights = jax.device_put(np.zeros(DIM, dtype=np.float32), device)
        state = optimizer.init(weights)
        _ready((weights, state), device, jax, jnp)
        for _ in range(UPDATES):
            guard()
            batch = make_batch(prepared, sample_indices(prepared, rng), guard=guard)
            placed = tuple(jax.device_put(value, device) for value in
                           (batch.features, batch.baseline_logits, batch.mask, batch.teacher_index))
            _ready(placed, device, jax, jnp)
            (value, components), gradient = value_grad(weights, *placed)
            _ready((value, components, gradient, weights, state), device, jax, jnp)
            nonzero = nonzero or bool(np.any(np.asarray(jax.device_get(gradient)) != 0))
            guard()
            update, new_state = optimizer.update(gradient, state, weights)
            new_weights = optax.apply_updates(weights, update)
            _ready((update, new_weights, new_state), device, jax, jnp)
            guard()
            weights, state = new_weights, new_state
            completed += 1
        guard()
        _ready((weights, state), device, jax, jnp)
        final = np.asarray(jax.device_get(weights))
    _require(final.dtype == np.dtype('float32') and final.shape == (DIM,) and np.isfinite(final).all()
             and completed == UPDATES)
    return FitResult(_readonly(final), completed, bool(nonzero), bool(np.any(final != 0)))
