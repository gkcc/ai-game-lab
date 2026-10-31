"""Small state-conditioned residual on a frozen task selector.

The caller authenticates the source, model schema and normalization, enforces
resource/attempt limits, and keeps returned parameters private. This module
does not read or write files, run games, or choose candidates outside the
controller's existing opportunity set. The JAX fit uses maximum-centered
float32 baseline scores, while deployment centers the original float64 baseline. Floating-point
ordering and centered versus uncentered arithmetic may differ near ties;
actual-data parity must be checked separately before using a fitted model.
"""
from dataclasses import dataclass

import numpy as np

from learning.task_multisource_rank_v1 import core as rank


SHARED_DIM, LOCAL_DIM, DIM = rank.SHARED_DIM, rank.LOCAL_DIM, rank.DIM
HIDDEN = 16
UPDATES = 64
LEARNING_RATE = 1e-3
KL_WEIGHT = 0.1
CLIP_NORM = 1.0
INIT_BOUND = 0.05


class StateResidualError(ValueError):
    """Private state-residual inputs or a numerical contract are invalid."""


def _require(condition):
    if not condition:
        raise StateResidualError('Invalid private state-residual input')


@dataclass(frozen=True, slots=True, repr=False)
class Params:
    ws: np.ndarray
    wc: np.ndarray
    b: np.ndarray
    v: np.ndarray

    def __repr__(self):
        return '<PrivateStateResidualParams>'


@dataclass(frozen=True, slots=True, repr=False)
class FitResult:
    params: Params
    updates_completed: int
    nonzero_gradient_seen: bool
    output_layer_changed: bool

    def __repr__(self):
        return '<PrivateStateResidualFitResult>'


def _checked_params(params):
    _require(type(params) is Params)
    shapes = {'ws': (SHARED_DIM, HIDDEN), 'wc': (LOCAL_DIM, HIDDEN),
              'b': (HIDDEN,), 'v': (HIDDEN,)}
    arrays = {}
    for name, shape in shapes.items():
        value = getattr(params, name)
        _require(type(value) is np.ndarray and value.dtype == np.dtype('float32')
                 and value.shape == shape and np.isfinite(value).all())
        arrays[name] = rank._immutable(value)
    return Params(**arrays)


def initial_params(rng_seed):
    """Bounded deterministic hidden initialization with exactly zero output."""
    _require(type(rng_seed) is int and 0 <= rng_seed < 2**32)
    rng = np.random.default_rng(np.random.SeedSequence([rng_seed, 0x53544154]))
    return _checked_params(Params(
        ws=rng.uniform(-INIT_BOUND, INIT_BOUND, (SHARED_DIM, HIDDEN)).astype(np.float32),
        wc=rng.uniform(-INIT_BOUND, INIT_BOUND, (LOCAL_DIM, HIDDEN)).astype(np.float32),
        b=np.zeros(HIDDEN, dtype=np.float32), v=np.zeros(HIDDEN, dtype=np.float32)))


def _training_baseline(baseline):
    """Center in float64 before the training cast; reject lost first choice."""
    _require(type(baseline) is np.ndarray and baseline.dtype == np.dtype('float64')
             and baseline.ndim == 1 and 1 <= len(baseline) <= rank.MAX_CANDIDATES
             and np.isfinite(baseline).all())
    with np.errstate(over='ignore', invalid='ignore'):
        centered64 = baseline - np.max(baseline)
        centered32 = centered64.astype(np.float32)
    _require(np.isfinite(centered64).all() and np.isfinite(centered32).all()
             and int(np.argmax(baseline)) == int(np.argmax(centered32)))
    return centered64, centered32


def prepare_groups(groups, *, mean, scale, guard):
    """Center each baseline then reuse the existing authenticated group prep."""
    _require(callable(guard) and type(groups) in (list, tuple)
             and 2 * rank.PER_SOURCE <= len(groups) <= rank.MAX_GROUPS)
    centered = []
    for group in groups:
        guard()
        rank._group(group)
        baseline64, _baseline32 = _training_baseline(group.baseline)
        centered.append(rank.Group(group.shared, group.candidates, baseline64,
                                   group.target, group.source))
    return rank.prepare_groups(centered, mean=mean, scale=scale, guard=guard)


def _components(params, features, baseline, mask, target, xp):
    """Masked CE plus old-to-new KL, using first-candidate-relative output."""
    first = features[:, :1, :]
    safe = xp.where(mask[:, :, None], features, first)
    shared = safe[:, :1, :SHARED_DIM]
    local = safe[:, :, SHARED_DIM:]
    shared_hidden = xp.einsum('bnd,dh->bnh', shared, params['ws'])
    local_hidden = xp.einsum('bnd,dh->bnh', local, params['wc'])
    preactivation = shared_hidden + local_hidden + params['b'][None, None, :]
    hidden = xp.tanh(preactivation)
    relative = hidden - hidden[:, :1, :]
    residual = xp.sum(relative * params['v'][None, None, :], axis=2)
    old = rank._logp(baseline, mask, xp)
    new = rank._logp(baseline + residual, mask, xp)
    ce = xp.mean(-new[xp.arange(new.shape[0]), target])
    probability = xp.where(mask, xp.exp(old), xp.zeros_like(old))
    kl = xp.mean(xp.sum(xp.where(mask, probability * (old - new), xp.zeros_like(old)), axis=1))
    finite = xp.all(xp.isfinite(preactivation)) & xp.all(xp.isfinite(hidden)) & xp.all(xp.isfinite(residual))
    loss = xp.where(finite, ce + KL_WEIGHT * kl, xp.asarray(np.nan, dtype=baseline.dtype))
    return loss, (ce, kl)


def fit(groups, *, mean, scale, device, guard, rng_seed):
    """Run exactly 64 balanced, synchronized Adam updates on explicit CPU.

    Each float64 baseline is shifted by its maximum before the existing group
    preparation checks the float32 argmax. Hidden initialization and batch sampling use separate
    deterministic streams. Any error returns no partial parameter result;
    nonzero hidden initialization alone is not evidence of training.
    """
    _require(callable(guard) and type(rng_seed) is int and 0 <= rng_seed < 2**32)
    guard()
    prepared = prepare_groups(groups, mean=mean, scale=scale, guard=guard)
    initialized = initial_params(rng_seed)
    import jax
    import jax.numpy as jnp
    import optax

    _require(isinstance(device, jax.Device) and device.platform == 'cpu' and device in jax.devices('cpu'))
    rng = np.random.default_rng(np.random.SeedSequence([rng_seed, 0x42415443]))
    optimizer = optax.chain(optax.clip_by_global_norm(CLIP_NORM), optax.adam(LEARNING_RATE))

    def objective(params, features, baseline, mask, target):
        return _components(params, features, jax.lax.stop_gradient(baseline), mask, target, jnp)

    value_grad = jax.jit(jax.value_and_grad(objective, has_aux=True))
    seen, completed = False, 0
    with jax.default_device(device), jax.default_matmul_precision('highest'):
        params = {name: jax.device_put(getattr(initialized, name), device) for name in ('ws', 'wc', 'b', 'v')}
        state = optimizer.init(params)
        rank._ready((params, state), device, jax, jnp)
        for _ in range(UPDATES):
            guard()
            batch = rank.make_batch(prepared, rank.sample_indices(prepared, rng), guard=guard)
            placed = tuple(jax.device_put(value, device) for value in
                           (batch.features, batch.baseline, batch.mask, batch.target))
            rank._ready(placed, device, jax, jnp)
            (value, components), gradient = value_grad(params, *placed)
            rank._ready((value, components, gradient, params, state), device, jax, jnp)
            seen = seen or any(bool(np.any(np.asarray(jax.device_get(leaf)) != 0))
                               for leaf in jax.tree_util.tree_leaves(gradient))
            guard()
            update, new_state = optimizer.update(gradient, state, params)
            new_params = optax.apply_updates(params, update)
            rank._ready((update, new_params, new_state), device, jax, jnp)
            guard()
            params, state = new_params, new_state
            completed += 1
        guard()
        rank._ready((params, state), device, jax, jnp)
        final = Params(**{name: np.asarray(jax.device_get(params[name]))
                          for name in ('ws', 'wc', 'b', 'v')})
    final = _checked_params(final)
    changed = bool(np.any(final.v != 0))
    _require(completed == UPDATES and seen and changed)
    return FitResult(final, completed, bool(seen), changed)


def _deployment_scores(baseline, residual):
    """Remove common offsets before float64 addition, rejecting overflow."""
    _require(type(baseline) is np.ndarray and baseline.dtype == np.dtype('float64')
             and baseline.ndim == 1 and len(baseline) > 0 and np.isfinite(baseline).all()
             and type(residual) is np.ndarray and residual.dtype == np.dtype('float32')
             and residual.shape == baseline.shape and np.isfinite(residual).all())
    with np.errstate(over='ignore', invalid='ignore'):
        centered = baseline - np.max(baseline)
        relative = residual.astype(np.float64) - np.float64(residual[0])
        scores = centered + relative
    _require(np.isfinite(centered).all() and np.isfinite(relative).all() and np.isfinite(scores).all())
    return scores


def select_values(shared, candidates, baseline, mean, scale, params):
    """Return the first maximum using only observable features and logits.

    The baseline remains float64. A zero output vector returns its original
    argmax exactly, including ties and extreme but finite baseline values.
    Otherwise, subtract the baseline maximum and the first float32 residual
    before adding in float64; this prevents irrelevant common offsets from
    swallowing meaningful differences. Real-model comparison with
    centered JAX arithmetic is a separate responsibility of the caller.
    """
    rank._inputs(shared, candidates, baseline)
    mean, scale = rank._normalization(mean, scale)
    params = _checked_params(params)
    if not np.any(params.v):
        return int(np.argmax(baseline))
    shared32 = rank._normalized(shared, mean[:SHARED_DIM], scale[:SHARED_DIM])
    local32 = rank._normalized(candidates, mean[SHARED_DIM:], scale[SHARED_DIM:])
    with np.errstate(over='ignore', invalid='ignore'):
        preactivation = shared32 @ params.ws + local32 @ params.wc + params.b
        _require(np.isfinite(preactivation).all())
        hidden = np.tanh(preactivation)
        residual = hidden @ params.v
    _require(hidden.dtype == np.dtype('float32') and residual.dtype == np.dtype('float32')
             and np.isfinite(hidden).all() and np.isfinite(residual).all())
    adjusted = _deployment_scores(baseline, residual)
    return int(np.argmax(adjusted))
