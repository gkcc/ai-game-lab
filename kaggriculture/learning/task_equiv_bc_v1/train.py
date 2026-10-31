"""Small in-memory paired task-head trainer; no source, device or split selection.

The caller authenticates a checkpoint, sample provenance, batch plan and
explicit device, and supplies a sampled guard. Nothing here plays a game,
selects fresh seeds, reads a checkpoint file or writes trained weights.
"""
from dataclasses import dataclass
import hashlib
import io
import json
import numbers
import re
from types import MappingProxyType
import unicodedata

import numpy as np

from learning.task_bc_v1.model import DIMENSIONS
from learning.task_pair_checkpoint_v1 import snapshot
from learning.task_pair_training_v1.objective import forward
from . import loss


MAX_UPDATES = 32
MAX_BATCH = 256
MAX_PADDED_CANDIDATES = 65536
MAX_SOURCE_CANDIDATES = 65536
PARAMS = tuple(snapshot.PARAMS)


@dataclass(frozen=True, repr=False)
class PairedResult:
    control_params: object
    experiment_params: object
    device: object
    records: tuple
    completed_updates: int
    initial_equal: bool
    control_changed: bool
    experiment_changed: bool
    arms_differ: bool
    checkpoint_identity: str

    def __repr__(self):
        return '<PrivateEquivalencePairedResult>'


def _checkpoint(checkpoint):
    if not isinstance(checkpoint, snapshot.Checkpoint):
        raise ValueError('Expected a previously authenticated task checkpoint')
    schema = snapshot._parse_metadata(np.asarray(checkpoint.schema_json))
    schema = snapshot._validate_schema(schema, checkpoint.bound_schema_sha256)
    if not hasattr(checkpoint.params, 'keys') or set(checkpoint.params) != set(PARAMS):
        raise ValueError('Invalid initial parameters')
    arrays = {name: checkpoint.params[name] for name in PARAMS}
    arrays.update(mean=checkpoint.mean, scale=checkpoint.scale)
    snapshot._validate_arrays(arrays, schema)
    params = {name: np.array(arrays[name], dtype=np.float32, copy=True) for name in PARAMS}
    mean, scale = np.array(checkpoint.mean, copy=True), np.array(checkpoint.scale, copy=True)
    return params, mean, scale, schema


def _checkpoint_identity(params, mean, scale, schema):
    """Bind exported normalization/schema to the exact initial task weights."""
    digest = hashlib.sha256(json.dumps(schema, sort_keys=True, separators=(',', ':')).encode())
    for name, value in [(name, params[name]) for name in PARAMS] + [('mean', mean), ('scale', scale)]:
        # dtype, shape and byte length are explicit so distinct arrays cannot
        # alias by concatenation. The fingerprint stays local to the result.
        data = np.ascontiguousarray(value)
        header = json.dumps([name, data.dtype.str, data.shape, data.nbytes], separators=(',', ':')).encode()
        digest.update(len(header).to_bytes(8, 'big'))
        digest.update(header)
        digest.update(data.tobytes())
    return digest.hexdigest()


def _copy_groups(groups):
    if type(groups) not in (tuple, list) or not groups or len(groups) > 65536:
        raise ValueError('Expected a bounded nonempty group list')
    result, total = [], 0
    for source in groups:
        if type(source) is not dict or not {'features', 'teacher_index', 'positive_mask',
                                           'verified_mask', 'weight'}.issubset(source):
            raise ValueError('Missing group fields')
        if 'status' in source and source['status'] != 'ok':
            raise ValueError('Unresolved group cannot enter training')
        x, p, u, teacher, weight = (source[key] for key in
                                     ('features', 'positive_mask', 'verified_mask', 'teacher_index', 'weight'))
        if (not isinstance(x, np.ndarray) or x.ndim != 2 or x.shape[1] != DIMENSIONS[0]
                or not len(x) or x.dtype != np.dtype('float32') or not np.isfinite(x).all()
                or not isinstance(p, np.ndarray) or p.dtype != np.dtype(bool) or p.shape != (len(x),)
                or not isinstance(u, np.ndarray) or u.dtype != np.dtype(bool) or u.shape != (len(x),)
                or type(teacher) is not int or not 0 <= teacher < len(x)
                or isinstance(weight, (bool, np.bool_)) or not isinstance(weight, numbers.Real)
                or not np.isfinite(weight) or weight != 1 or np.any(p & ~u) or not p[teacher]):
            raise ValueError('Invalid verified task group')
        if not np.array_equal(x[:, :275], np.broadcast_to(x[0, :275], (len(x), 275))):
            raise ValueError('Shared observation must agree across candidate rows')
        weight = np.float32(weight)
        if not np.isfinite(weight) or weight <= 0:
            raise ValueError('Invalid group weight')
        total += len(x)
        if total > MAX_SOURCE_CANDIDATES:
            raise ValueError('Candidate source budget exceeded')
        result.append(dict(features=np.array(x, copy=True), positive_mask=np.array(p, copy=True),
                           verified_mask=np.array(u, copy=True), teacher_index=teacher, weight=weight))
    return result


def _batch(groups, ids, mean, scale):
    width = max(len(groups[i]['features']) for i in ids)
    if len(ids) * width > MAX_PADDED_CANDIDATES:
        raise ValueError('Padded candidate budget exceeded')
    x = np.zeros((len(ids), width, DIMENSIONS[0]), dtype=np.float32)
    valid = np.zeros((len(ids), width), dtype=bool)
    positive, verified = np.zeros_like(valid), np.zeros_like(valid)
    labels = np.empty(len(ids), np.int32)
    weights = np.empty(len(ids), np.float32)
    for row, i in enumerate(ids):
        group = groups[i]
        n = len(group['features'])
        x[row, :n] = (group['features'].astype(np.float64) - mean) / scale
        valid[row, :n] = True
        positive[row, :n], verified[row, :n] = group['positive_mask'], group['verified_mask']
        labels[row], weights[row] = group['teacher_index'], group['weight']
    if not np.isfinite(x).all():
        raise ValueError('Nonfinite normalized features')
    loss.validate_batch(np.zeros(valid.shape, np.float32), valid, positive, verified, labels, weights)
    return x, valid, positive, verified, labels, weights


def fit_paired(groups, *, checkpoint, batches, device, guard):
    """Run the same caller-frozen batches through both arms on one device.

    Base loss is full-candidate weighted teacher CE in both arms. The control
    auxiliary is logsumexp(U)-teacher; the experiment auxiliary is
    logsumexp(U)-logsumexp(P). Both use coefficient 0.25 and all-group weight
    as denominator. Ineligible groups have zero auxiliary contribution.
    Guard must raise on failure; its return value is ignored. Checks are
    sampled, not asynchronous interruption or a GPU memory cap.
    """
    import jax
    import jax.numpy as jnp
    import optax

    if (not callable(guard) or not isinstance(device, jax.Device)
            or device not in jax.devices(device.platform)):
        raise ValueError('An explicit available device and guard are required')
    guard()
    initial, mean, scale, schema = _checkpoint(checkpoint)
    initial_identity = _checkpoint_identity(initial, mean, scale, schema)
    groups = _copy_groups(groups)
    if (not isinstance(batches, np.ndarray) or batches.ndim != 2 or batches.dtype.kind not in 'iu'
            or not 1 <= batches.shape[0] <= MAX_UPDATES or not 1 <= batches.shape[1] <= MAX_BATCH
            or np.any(batches < 0) or np.any(batches >= len(groups))):
        raise ValueError('Invalid frozen batch plan')
    plan = np.array(batches, dtype=np.int64, copy=True)
    if any(len(set(row.tolist())) != len(row) for row in plan):
        raise ValueError('Duplicate group in a batch')
    guard()

    def ready(tree):
        for value in jax.tree_util.tree_leaves(tree):
            if not isinstance(value, jax.Array) or value.devices() != {device}:
                raise RuntimeError('Array used the wrong device')
            value.block_until_ready()
            if not bool(np.asarray(jnp.all(jnp.isfinite(value)).block_until_ready())):
                raise FloatingPointError('Nonfinite training value')

    def compute(params, batch, arm):
        x, valid, positive, verified, labels, weights = batch
        logits = forward(params, x, xp=jnp)
        return loss.objective(logits, valid, positive, verified, labels, weights, arm=arm, xp=jnp)

    control_grad = jax.jit(jax.value_and_grad(lambda p, b: compute(p, b, 'control'), has_aux=True))
    experiment_grad = jax.jit(jax.value_and_grad(lambda p, b: compute(p, b, 'experiment'), has_aux=True))
    control_aux_grad = jax.jit(jax.grad(lambda p, b: compute(p, b, 'control')[1][1]))
    experiment_aux_grad = jax.jit(jax.grad(lambda p, b: compute(p, b, 'experiment')[1][1]))
    optimizer = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(3e-4))
    with jax.default_device(device), jax.default_matmul_precision('highest'):
        baseline = jax.device_put(initial, device)
        control = jax.device_put({k: v.copy() for k, v in initial.items()}, device)
        experiment = jax.device_put({k: v.copy() for k, v in initial.items()}, device)
        control_state, experiment_state = optimizer.init(control), optimizer.init(experiment)
        ready((baseline, control, experiment, control_state, experiment_state))
        records = []
        for ids in plan:
            guard()
            host = _batch(groups, ids, mean, scale)
            batch = jax.device_put(host, device)
            ready(batch)
            (control_value, control_components), control_gradient = control_grad(control, batch)
            (experiment_value, experiment_components), experiment_gradient = experiment_grad(experiment, batch)
            caux = control_aux_grad(control, batch)
            eaux = experiment_aux_grad(experiment, batch)
            cnorm = optax.global_norm(control_gradient)
            enorm = optax.global_norm(experiment_gradient)
            canorm = optax.global_norm(caux)
            eanorm = optax.global_norm(eaux)
            gradient_gap = optax.global_norm(jax.tree_util.tree_map(
                lambda a, b: a - b, control_gradient, experiment_gradient))
            ready((control_value, control_components, control_gradient, cnorm,
                   experiment_value, experiment_components, experiment_gradient, enorm,
                   caux, eaux, canorm, eanorm, gradient_gap))
            guard()
            control_update, new_control_state = optimizer.update(control_gradient, control_state, control)
            experiment_update, new_experiment_state = optimizer.update(experiment_gradient, experiment_state, experiment)
            new_control = optax.apply_updates(control, control_update)
            new_experiment = optax.apply_updates(experiment, experiment_update)
            ready((new_control, new_experiment, new_control_state, new_experiment_state))
            record = {'control_loss': float(np.asarray(control_value)),
                      'experiment_loss': float(np.asarray(experiment_value)),
                      'control_base': float(np.asarray(control_components[0])),
                      'experiment_base': float(np.asarray(experiment_components[0])),
                      'control_aux': float(np.asarray(control_components[1])),
                      'experiment_aux': float(np.asarray(experiment_components[1])),
                      'eligible_groups': int(np.asarray(control_components[2])),
                      'control_gradient_norm': float(np.asarray(cnorm)),
                      'experiment_gradient_norm': float(np.asarray(enorm)),
                      'control_unscaled_aux_gradient_norm': float(np.asarray(canorm)),
                      'experiment_unscaled_aux_gradient_norm': float(np.asarray(eanorm)),
                      'gradient_gap_at_current_arm_parameters': float(np.asarray(gradient_gap)),
                      'device_platform': device.platform, 'device_kind': device.device_kind}
            if record['eligible_groups'] != int(np.asarray(experiment_components[2])):
                raise ValueError('Arm eligibility differs')
            records.append(record)
            control, experiment = new_control, new_experiment
            control_state, experiment_state = new_control_state, new_experiment_state
            guard()
        def differs(a, b):
            return any(not np.array_equal(np.asarray(a[name]), np.asarray(b[name])) for name in PARAMS)
        control_changed = differs(baseline, control)
        experiment_changed = differs(baseline, experiment)
        arms_differ = differs(control, experiment)
    return PairedResult(MappingProxyType(control.copy()), MappingProxyType(experiment.copy()),
                        device, tuple(records), len(records), True,
                        control_changed, experiment_changed, arms_differ, initial_identity)


def export_pair(result, *, checkpoint, registration_commit, source_kind, expected_device_kind=None):
    """Return private NPZ bytes in the existing TaskNetwork schema; do not write.

    Unlike the old 64-update exporter, provenance describes this 0.25-versus-
    0.25 experiment without pretending that the control has no auxiliary.
    The caller privately binds the exact expected device kind for real data and
    is responsible for authorized exclusive writes and reload.
    """
    import jax

    if (not isinstance(result, PairedResult) or not result.initial_equal
            or type(result.completed_updates) is not int or not 1 <= result.completed_updates <= MAX_UPDATES
            or len(result.records) != result.completed_updates
            or type(registration_commit) is not str or re.fullmatch(r'[0-9a-f]{40}', registration_commit) is None
            or source_kind not in ('synthetic_only', 'validated_counterfactual_requests')):
        raise ValueError('Invalid paired export context')
    if source_kind == 'validated_counterfactual_requests':
        # The outer runner must also authenticate the data and supervise the
        # execution. These checks independently prevent a partial/CPU run from
        # being exported under the real-data provenance label.
        device_kind = getattr(result.device, 'device_kind', None)
        _verify_expected_device(device_kind, expected_device_kind)
        if (not isinstance(result.device, jax.Device)
                or result.device.platform != 'gpu' or result.device not in jax.devices('gpu')):
            raise ValueError('Real export requires the registered GPU')
        _real_export_gate(result, device_kind)
    initial, mean, scale, schema = _checkpoint(checkpoint)
    if _checkpoint_identity(initial, mean, scale, schema) != result.checkpoint_identity:
        raise ValueError('Export checkpoint differs from training initialization')
    arms = []
    for name, values in (('control', result.control_params), ('experiment', result.experiment_params)):
        if set(values) != set(PARAMS):
            raise ValueError('Invalid result parameters')
        host = {}
        for key in PARAMS:
            value = values[key]
            if not isinstance(value, jax.Array) or value.devices() != {result.device}:
                raise ValueError('Parameter device differs')
            value.block_until_ready()
            host[key] = np.asarray(jax.device_get(value), dtype=np.float32).copy()
        snapshot._validate_arrays(dict(host, mean=mean, scale=scale), schema)
        metadata = dict(schema, equivalence_training=dict(schema='task-equiv-bc-v1', arm=name,
            registration_commit=registration_commit, training_source=source_kind,
            updates=result.completed_updates, auxiliary_weight=loss.AUXILIARY_WEIGHT,
            auxiliary_denominator='all_group_weight'))
        stream = io.BytesIO()
        np.savez_compressed(stream, **host, mean=mean, scale=scale,
                            metadata=np.asarray(json.dumps(metadata, sort_keys=True)))
        raw = stream.getvalue()
        snapshot.load_checkpoint(raw, expected_sha256=hashlib.sha256(raw).hexdigest(),
                                 expected_schema_sha256=checkpoint.bound_schema_sha256)
        arms.append(raw)
    return tuple(arms)


def _verify_expected_device(device_kind, expected_device_kind):
    """Check the exact private, caller-bound device kind without disclosing it."""
    if (type(expected_device_kind) is not str or not 1 <= len(expected_device_kind) <= 128
            or not expected_device_kind.strip()
            or any(unicodedata.category(char).startswith('C') for char in expected_device_kind)
            or type(device_kind) is not str or expected_device_kind != device_kind):
        raise ValueError('Device does not match private registration')


def _real_export_gate(result, device_kind):
    """Require the registered update and first-shared-parameter evidence."""
    if (not isinstance(result, PairedResult) or result.initial_equal is not True
            or result.completed_updates != 32 or len(result.records) != 32
            or result.control_changed is not True or result.experiment_changed is not True
            or result.arms_differ is not True):
        raise ValueError('Real paired update evidence is incomplete')
    if type(device_kind) is not str or any(
            type(record) is not dict or record.get('device_platform') != 'gpu'
            or record.get('device_kind') != device_kind for record in result.records):
        raise ValueError('Per-update device evidence differs')
    first_gap = result.records[0].get('gradient_gap_at_current_arm_parameters')
    if (isinstance(first_gap, (bool, np.bool_)) or not isinstance(first_gap, numbers.Real)
            or not np.isfinite(first_gap) or first_gap <= 1e-8):
        raise ValueError('First shared-parameter gradient difference is not observed')
