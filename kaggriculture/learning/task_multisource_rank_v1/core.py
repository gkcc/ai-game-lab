"""Private, bounded multisource task-ranking computation.

The caller authenticates data, schema, model identity and split membership;
the source tag is used only for batch balance, never as a feature. This
module neither reads files nor saves weights, runs games, or evaluates a
candidate. The selector accepts the ordered opportunity set produced by the
existing controller; that set is not the complete official action space.
"""
from dataclasses import dataclass

import numpy as np

from decision.api import PRODUCTS
from learning.task_choice_v1.features import OPS


SHARED_DIM = 275
LOCAL_DIM = 98
DIM = SHARED_DIM + LOCAL_DIM
OP_OFFSET = 5 + len(PRODUCTS)
MAX_CANDIDATES = 256
MAX_GROUPS = 8192
PER_SOURCE = 8
UPDATES = 64
LEARNING_RATE = 1e-4
KL_WEIGHT = 0.1
CLIP_NORM = 1.0
SOURCES = ('old', 'new')


class RankError(ValueError):
    """A bounded ranking input or fixed numerical contract is invalid."""


def _require(condition):
    if not condition:
        raise RankError('Invalid private multisource ranking input')


def _immutable(array):
    contiguous = np.ascontiguousarray(array)
    return np.frombuffer(contiguous.tobytes(), dtype=contiguous.dtype).reshape(contiguous.shape)


@dataclass(frozen=True, slots=True, repr=False)
class Group:
    """One private teacher choice, without a replicated shared feature matrix."""

    shared: np.ndarray
    candidates: np.ndarray
    baseline: np.ndarray
    target: int
    source: str

    def __repr__(self):
        return '<PrivateRankGroup>'


@dataclass(frozen=True, slots=True, repr=False)
class _PreparedGroup:
    shared: np.ndarray
    candidates: np.ndarray
    baseline: np.ndarray
    target: int
    source: str

    def __repr__(self):
        return '<PrivatePreparedRankGroup>'


@dataclass(frozen=True, slots=True, repr=False)
class _Batch:
    features: np.ndarray
    baseline: np.ndarray
    mask: np.ndarray
    target: np.ndarray

    def __repr__(self):
        return '<PrivateRankBatch>'


@dataclass(frozen=True, slots=True, repr=False)
class FitResult:
    weights: np.ndarray
    updates_completed: int
    nonzero_gradient_seen: bool
    weights_changed: bool

    def __repr__(self):
        return '<PrivateRankFitResult>'


def _inputs(shared, candidates, baseline):
    _require(type(shared) is np.ndarray and shared.dtype == np.dtype('float32')
             and shared.shape == (SHARED_DIM,) and np.isfinite(shared).all()
             and type(candidates) is np.ndarray and candidates.dtype == np.dtype('float32')
             and candidates.ndim == 2 and candidates.shape[1] == LOCAL_DIM
             and 1 <= len(candidates) <= MAX_CANDIDATES and np.isfinite(candidates).all()
             and type(baseline) is np.ndarray and baseline.dtype == np.dtype('float64')
             and baseline.shape == (len(candidates),) and np.isfinite(baseline).all())
    operations = candidates[:, OP_OFFSET:OP_OFFSET + len(OPS)]
    _require(operations.shape == (len(candidates), len(OPS))
             and np.isin(operations, (0.0, 1.0)).all()
             and np.all(operations.sum(axis=1) == 1.0))


def _group(group):
    _require(type(group) is Group and type(group.source) is str and group.source in SOURCES)
    _inputs(group.shared, group.candidates, group.baseline)
    _require(type(group.target) is int and 0 <= group.target < len(group.candidates))


def _normalization(mean, scale):
    _require(type(mean) is np.ndarray and mean.dtype == np.dtype('float64')
             and mean.shape == (DIM,) and np.isfinite(mean).all()
             and type(scale) is np.ndarray and scale.dtype == np.dtype('float64')
             and scale.shape == (DIM,) and np.isfinite(scale).all() and np.all(scale > 0))
    return mean.copy(), scale.copy()


def _normalized(raw, mean, scale):
    with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
        value = ((raw.astype(np.float64) - mean) / scale).astype(np.float32)
    _require(np.isfinite(value).all())
    return value


def prepare_groups(groups, *, mean, scale, guard):
    """Snapshot and normalize split 275/98-dimensional data, without JAX.

    Every baseline is checked for a change in argmax on conversion to the
    float32 training representation. Input arrays are copied into immutable
    buffers. A guard failure is propagated and no partial pool is returned.
    """
    _require(callable(guard) and type(groups) in (tuple, list)
             and 2 * PER_SOURCE <= len(groups) <= MAX_GROUPS)
    guard()
    mean, scale = _normalization(mean, scale)
    prepared, counts = [], {source: 0 for source in SOURCES}
    for group in groups:
        guard()
        _group(group)
        with np.errstate(over='ignore', invalid='ignore'):
            baseline32 = group.baseline.astype(np.float32)
        _require(np.isfinite(baseline32).all()
                 and int(np.argmax(group.baseline)) == int(np.argmax(baseline32)))
        shared = _normalized(group.shared, mean[:SHARED_DIM], scale[:SHARED_DIM])
        candidates = _normalized(group.candidates, mean[SHARED_DIM:], scale[SHARED_DIM:])
        prepared.append(_PreparedGroup(_immutable(shared), _immutable(candidates),
                                       _immutable(baseline32), group.target, group.source))
        counts[group.source] += 1
    _require(all(counts[source] >= PER_SOURCE for source in SOURCES))
    guard()
    return tuple(prepared)


def sample_indices(prepared, rng):
    """Draw eight distinct indices per source with the caller's fixed RNG."""
    _require(type(prepared) is tuple and isinstance(rng, np.random.Generator)
             and all(type(group) is _PreparedGroup for group in prepared))
    pools = [[i for i, group in enumerate(prepared) if group.source == source] for source in SOURCES]
    _require(all(len(pool) >= PER_SOURCE for pool in pools))
    return np.concatenate([rng.choice(pool, size=PER_SOURCE, replace=False) for pool in pools])


def make_batch(prepared, indices, *, guard):
    """Expand only one balanced batch and mask its padded candidates."""
    _require(callable(guard) and type(prepared) is tuple and prepared
             and all(type(group) is _PreparedGroup for group in prepared)
             and type(indices) is np.ndarray and indices.shape == (2 * PER_SOURCE,)
             and indices.dtype.kind in 'iu' and np.all(indices >= 0) and np.all(indices < len(prepared))
             and len(set(map(int, indices))) == len(indices))
    guard()
    chosen = [prepared[int(index)] for index in indices]
    _require(all(sum(group.source == source for group in chosen) == PER_SOURCE for source in SOURCES))
    # A fixed width for all batches avoids recompilation when sampled widths differ.
    width = max(len(group.candidates) for group in prepared)
    _require(1 <= width <= MAX_CANDIDATES)
    features = np.zeros((len(chosen), width, DIM), dtype=np.float32)
    baseline = np.zeros((len(chosen), width), dtype=np.float32)
    mask = np.zeros((len(chosen), width), dtype=bool)
    target = np.zeros(len(chosen), dtype=np.int32)
    for row, group in enumerate(chosen):
        guard()
        n = len(group.candidates)
        features[row, :n, :SHARED_DIM] = group.shared
        features[row, :n, SHARED_DIM:] = group.candidates
        baseline[row, :n] = group.baseline
        mask[row, :n] = True
        target[row] = group.target
    _require(np.isfinite(features).all() and np.isfinite(baseline).all()
             and np.all(mask[:, 0]) and np.all(mask[np.arange(len(chosen)), target]))
    return _Batch(features, baseline, mask, target)


def _logp(scores, mask, xp):
    maximum = xp.max(xp.where(mask, scores, -xp.inf), axis=1, keepdims=True)
    safe = xp.where(mask, scores, maximum)
    shifted = safe - maximum
    denominator = xp.sum(xp.where(mask, xp.exp(shifted), xp.zeros_like(shifted)), axis=1, keepdims=True)
    return xp.where(mask, shifted - xp.log(denominator), xp.zeros_like(shifted))


def _components(weights, features, baseline, mask, target, xp):
    first = features[:, :1, :]
    relative = xp.where(mask[:, :, None], features, first) - first
    residual = xp.sum(relative * weights[None, None, :], axis=2)
    old = _logp(baseline, mask, xp)
    new = _logp(baseline + residual, mask, xp)
    ce = xp.mean(-new[xp.arange(new.shape[0]), target])
    probability = xp.where(mask, xp.exp(old), xp.zeros_like(old))
    kl = xp.mean(xp.sum(xp.where(mask, probability * (old - new), xp.zeros_like(old)), axis=1))
    return ce + KL_WEIGHT * kl, (ce, kl)


def _ready(tree, device, jax, jnp):
    for leaf in jax.tree_util.tree_leaves(tree):
        if not isinstance(leaf, jax.Array) or leaf.devices() != {device}:
            raise RuntimeError('Ranking fit array left the registered CPU device')
        leaf.block_until_ready()
        finite = jnp.all(jnp.isfinite(leaf))
        finite.block_until_ready()
        if not bool(np.asarray(finite)):
            raise FloatingPointError('Ranking fit produced a nonfinite value')


def fit(groups, *, mean, scale, device, guard, rng_seed):
    """Run exactly 64 balanced synchronized Adam updates on an explicit CPU.

    The caller owns the wall, memory and attempt limits. All groups are checked
    before JAX is imported here; no checkpoint is written or partial vector
    returned after an error. The float32 baseline is a training approximation;
    deployment and evaluation must use the original float64 baseline.
    """
    _require(callable(guard) and type(rng_seed) is int and 0 <= rng_seed < 2**32)
    guard()
    prepared = prepare_groups(groups, mean=mean, scale=scale, guard=guard)
    import jax
    import jax.numpy as jnp
    import optax

    _require(isinstance(device, jax.Device) and device.platform == 'cpu' and device in jax.devices('cpu'))
    rng = np.random.default_rng(rng_seed)
    optimizer = optax.chain(optax.clip_by_global_norm(CLIP_NORM), optax.adam(LEARNING_RATE))

    def objective(weights, features, baseline, mask, target):
        return _components(weights, features, jax.lax.stop_gradient(baseline), mask, target, jnp)

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
                           (batch.features, batch.baseline, batch.mask, batch.target))
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
    _require(final.dtype == np.dtype('float32') and final.shape == (DIM,)
             and np.isfinite(final).all() and completed == UPDATES)
    return FitResult(_immutable(final), completed, bool(nonzero), bool(np.any(final != 0)))


def select_values(shared, candidates, baseline, mean, scale, weights):
    """Choose from inference-visible arrays without requiring labels/source.

    Float64 baseline scores are not cast. For nonzero weights only one
    candidate group's normalized float32 features are expanded. Ties retain
    the first candidate. No task-operation category is permanently excluded.
    """
    _inputs(shared, candidates, baseline)
    mean, scale = _normalization(mean, scale)
    _require(type(weights) is np.ndarray and weights.dtype == np.dtype('float32')
             and weights.shape == (DIM,) and np.isfinite(weights).all())
    if not np.any(weights):
        return int(np.argmax(baseline))
    raw = np.concatenate((np.broadcast_to(shared, (len(candidates), SHARED_DIM)), candidates), axis=1)
    normalized = _normalized(raw, mean, scale)
    with np.errstate(over='ignore', invalid='ignore'):
        residual = normalized @ weights
        adjusted = baseline + residual.astype(np.float64)
    _require(residual.dtype == np.dtype('float32') and np.isfinite(residual).all()
             and np.isfinite(adjusted).all())
    return int(np.argmax(adjusted))


def select(group, mean, scale, weights):
    """Convenience adapter for an authenticated offline Group."""
    _group(group)
    return select_values(group.shared, group.candidates, group.baseline, mean, scale, weights)
