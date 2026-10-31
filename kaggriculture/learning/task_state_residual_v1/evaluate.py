"""Private offline assessment and CPU parity for state-conditioned ranking.

Only aggregates leave this module. The caller authenticates source groups,
roles, schema and parameters. Historical labels are not an independent game
test. Parity checks both the requested train width and width 256 in fixed
16-row CPU batches. Matching shape and formula do not imply bitwise identity
with the separately compiled value-and-gradient training program.
"""
import numpy as np

from learning.task_choice_v1.features import OPS
from learning.task_multisource_rank_v1 import core as rank
from learning.task_multisource_rank_v1 import evaluate as prior
from learning.task_water_residual_v1 import policy
from . import core


MAX_GROUPS = rank.MAX_GROUPS
PARITY_ATOL, PARITY_RTOL = 1e-4, 1e-5
PARITY_BATCH, PARITY_WIDTH = 16, 256


class EvaluationError(ValueError):
    """An offline scoring or numerical parity contract failed."""


def _require(value):
    if not value:
        raise EvaluationError('Invalid private state-residual evaluation')


def _normalization(mean, scale):
    mean, scale = rank._normalization(mean, scale)
    return prior._snapshot(mean, 'float64', (rank.DIM,)), prior._snapshot(scale, 'float64', (rank.DIM,))


def _snapshot_group(group):
    rank._group(group)
    n = len(group.candidates)
    return rank.Group(prior._snapshot(group.shared, 'float32', (rank.SHARED_DIM,)),
                      prior._snapshot(group.candidates, 'float32', (n, rank.LOCAL_DIM)),
                      prior._snapshot(group.baseline, 'float64', (n,)),
                      group.target, group.source)


def _numpy_residual(group, mean, scale, params):
    """Private uncentered float32 scores matching the deployment selector."""
    if not np.any(params.v):
        return np.zeros(len(group.candidates), dtype=np.float32)
    shared = rank._normalized(group.shared, mean[:rank.SHARED_DIM], scale[:rank.SHARED_DIM])
    local = rank._normalized(group.candidates, mean[rank.SHARED_DIM:], scale[rank.SHARED_DIM:])
    with np.errstate(over='ignore', invalid='ignore'):
        preactivation = shared @ params.ws + local @ params.wc + params.b
        _require(np.isfinite(preactivation).all())
        hidden = np.tanh(preactivation)
        residual = hidden @ params.v
    _require(hidden.dtype == np.dtype('float32') and residual.dtype == np.dtype('float32')
             and residual.shape == (len(group.candidates),) and np.isfinite(residual).all())
    return residual


def assess(groups, mean, scale, params, metadata, *, roles=None, role_labels=(), guard=lambda: None):
    """Aggregate teacher-target matches and NLL on authenticated private groups.

    ``roles`` and ``role_labels`` are anonymous non-feature labels; absent
    roles and target operations have unknown coverage. No per-group inputs,
    scores, chosen indexes, labels or model parameters are returned.
    """
    _require(type(groups) in (list, tuple) and len(groups) <= MAX_GROUPS and callable(guard))
    guard()
    schema = policy._schema(metadata)
    policy._normalization(mean, scale, schema)
    expected_ops = tuple(rank.SHARED_DIM + rank.OP_OFFSET + i for i in range(len(OPS)))
    _require(schema[3] == expected_ops)
    mean, scale = _normalization(mean, scale)
    params = core._checked_params(params)
    selected_roles = prior._role_selection(groups, roles, role_labels)
    rows = []
    for i, group in enumerate(groups):
        guard()
        group = _snapshot_group(group)
        raw = np.concatenate((np.broadcast_to(group.shared, (len(group.candidates), rank.SHARED_DIM)),
                              group.candidates), axis=1)
        operations = policy._operations(raw, schema)
        baseline_choice = int(np.argmax(group.baseline))
        residual = _numpy_residual(group, mean, scale, params)
        adjusted = core._deployment_scores(group.baseline, residual) if np.any(params.v) else group.baseline
        _require(np.isfinite(adjusted).all())
        selected = core.select_values(group.shared, group.candidates, group.baseline, mean, scale, params)
        _require(type(selected) is int and selected == int(np.argmax(adjusted)))
        before, after = baseline_choice == group.target, selected == group.target
        rows.append(dict(role=None if selected_roles is None else selected_roles[i],
            target_op=OPS[int(np.argmax(operations[group.target]))],
            baseline_correct=before, residual_correct=after, changed=baseline_choice != selected,
            op_flips=not np.array_equal(operations[baseline_choice], operations[selected]),
            improved=not before and after, regressed=before and not after,
            baseline_nll=prior._nll(group.baseline, group.target), residual_nll=prior._nll(adjusted, group.target)))
    guard()
    by_operation = {op: prior._summary([row for row in rows if row['target_op'] == op]) for op in OPS}
    by_role = {}
    if selected_roles is not None:
        for role in role_labels:
            local = [row for row in rows if row['role'] == role]
            by_role[role] = dict(overall=prior._summary(local),
                by_target_operation={op: prior._summary([row for row in local if row['target_op'] == op])
                                     for op in OPS})
    guard()
    return dict(overall=prior._summary(rows), by_target_operation=by_operation,
                by_role=by_role, independent_evaluation=False)


def _jax_batch(params, features, baseline, mask, jnp):
    """Replicate the fit's padded JAX expression, including float32 addition."""
    first = features[:, :1, :]
    safe = jnp.where(mask[:, :, None], features, first)
    shared = safe[:, :1, :rank.SHARED_DIM]
    local = safe[:, :, rank.SHARED_DIM:]
    preactivation = (jnp.einsum('bnd,dh->bnh', shared, params['ws'])
                     + jnp.einsum('bnd,dh->bnh', local, params['wc'])
                     + params['b'][None, None, :])
    hidden = jnp.tanh(preactivation)
    relative = hidden - hidden[:, :1, :]
    residual = jnp.sum(relative * params['v'][None, None, :], axis=2)
    return residual, baseline + residual, jnp.all(jnp.isfinite(preactivation))


def parity(groups, mean, scale, params, *, device, guard, train_width):
    """Check centered float32 baseline, JAX relative scores, and argmax parity.

    The caller sets the CPU backend and enforces memory and time budgets. Each
    JAX batch is synchronized, checked for finite values and kept on the
    explicit CPU. ``train_width`` covers each group it can accommodate;
    width 256 covers all groups. The caller must verify that every training
    group fits ``train_width``. Each width is run once, including when the two
    widths coincide. No input, logits or per-group indices are returned.
    """
    _require(type(groups) in (list, tuple) and 0 < len(groups) <= MAX_GROUPS and callable(guard)
             and type(train_width) is int and 1 <= train_width <= PARITY_WIDTH)
    guard()
    mean, scale = _normalization(mean, scale)
    params = core._checked_params(params)
    # Validate the complete fixed pool and cast choice before invoking JAX.
    snapshots = []
    for group in groups:
        guard()
        group = _snapshot_group(group)
        try:
            _centered64, baseline32 = core._training_baseline(group.baseline)
        except core.StateResidualError:
            raise EvaluationError('Centered baseline is nonfinite or changes first choice') from None
        snapshots.append((group, baseline32))
    import jax
    import jax.numpy as jnp
    _require(isinstance(device, jax.Device) and device.platform == 'cpu' and device in jax.devices('cpu'))
    compute = jax.jit(lambda p, x, b, m: _jax_batch(p, x, b, m, jnp))
    max_error = 0.0
    groups_by_width = {}
    widths = (PARITY_WIDTH,) if train_width == PARITY_WIDTH else (train_width, PARITY_WIDTH)
    with jax.default_device(device), jax.default_matmul_precision('highest'):
        placed_params = {name: jax.device_put(getattr(params, name), device) for name in ('ws', 'wc', 'b', 'v')}
        rank._ready(placed_params, device, jax, jnp)
        for width in widths:
            eligible = [item for item in snapshots if len(item[0].candidates) <= width]
            groups_by_width[str(width)] = 0
            for start in range(0, len(eligible), PARITY_BATCH):
                guard()
                chunk = eligible[start:start + PARITY_BATCH]
                features = np.zeros((PARITY_BATCH, width, rank.DIM), dtype=np.float32)
                baseline = np.zeros((PARITY_BATCH, width), dtype=np.float32)
                mask = np.zeros((PARITY_BATCH, width), dtype=bool)
                # Padded batch rows are harmless valid singleton rows, never counted.
                mask[:, 0] = True
                expected = []
                for row, (group, baseline32) in enumerate(chunk):
                    guard()
                    n = len(group.candidates)
                    shared = rank._normalized(group.shared, mean[:rank.SHARED_DIM], scale[:rank.SHARED_DIM])
                    local = rank._normalized(group.candidates, mean[rank.SHARED_DIM:], scale[rank.SHARED_DIM:])
                    features[row, :n, :rank.SHARED_DIM] = shared
                    features[row, :n, rank.SHARED_DIM:] = local
                    baseline[row, :n] = baseline32
                    mask[row, :n] = True
                    residual = _numpy_residual(group, mean, scale, params)
                    chosen = core.select_values(group.shared, group.candidates, group.baseline, mean, scale, params)
                    expected.append((group, residual, chosen))
                placed = tuple(jax.device_put(value, device) for value in (features, baseline, mask))
                rank._ready(placed, device, jax, jnp)
                relative, training_scores, finite = compute(placed_params, *placed)
                rank._ready((relative, training_scores, finite), device, jax, jnp)
                if not bool(np.asarray(jax.device_get(finite))):
                    raise EvaluationError('Nonfinite JAX state-residual activation')
                relative = np.asarray(jax.device_get(relative))
                training_scores = np.asarray(jax.device_get(training_scores))
                for row, (group, residual, selected) in enumerate(expected):
                    guard()
                    n = len(group.candidates)
                    actual = relative[row, :n].astype(np.float64)
                    wanted = residual.astype(np.float64) - np.float64(residual[0])
                    _require(np.isfinite(actual).all() and np.isfinite(wanted).all())
                    max_error = max(max_error, float(np.max(np.abs(actual - wanted))))
                    if not np.allclose(actual, wanted, atol=PARITY_ATOL, rtol=PARITY_RTOL):
                        raise EvaluationError('JAX and deployment relative scores differ')
                    if int(np.argmax(training_scores[row, :n])) != selected:
                        raise EvaluationError('Float32 training and float64 deployment choices differ')
                    groups_by_width[str(width)] += 1
    guard()
    _require(groups_by_width[str(PARITY_WIDTH)] == len(groups))
    return dict(groups=len(groups), allclose=True, baseline_argmax_mismatches=0,
                training_choice_mismatches=0, maximum_absolute_error=max_error,
                fixed_batch=PARITY_BATCH, fixed_width=PARITY_WIDTH, train_width=train_width,
                train_width_groups=groups_by_width[str(train_width)],
                full_width_groups=groups_by_width[str(PARITY_WIDTH)],
                groups_by_width=groups_by_width, device='cpu')
