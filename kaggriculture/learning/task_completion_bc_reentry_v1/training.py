"""Bounded task-choice imitation probe. Inputs are frozen teacher opportunities."""
from dataclasses import dataclass
from pathlib import Path
import json
import time
import numpy as np
from learning.task_bc_v1.model import DIMENSIONS, TaskNetwork, continuous_mask, sha
from learning.task_choice_v1.features import OPS, SCHEMA_VERSION, schema_hash

@dataclass
class Group:
    shared: object
    candidates: object
    chosen: int
    candidate_ops: object
    seed: object
    status: str


def as_groups(groups):
    return [Group(**g) if isinstance(g, dict) else g for g in groups]


def validate(groups):
    for g in groups:
        s, c = np.asarray(g.shared), np.asarray(g.candidates)
        ops = np.asarray(g.candidate_ops)
        if s.shape != (275,) or c.ndim != 2 or c.shape[1] != 98 or not len(c):
            raise ValueError('Invalid feature shape')
        if not np.isfinite(s).all() or not np.isfinite(c).all():
            raise ValueError('Nonfinite features')
        if ops.shape != (len(c),) or any(o not in OPS for o in ops):
            raise ValueError('Invalid operations')
        if not isinstance(g.chosen, (int, np.integer)) or isinstance(g.chosen, (bool, np.bool_)) or not 0 <= g.chosen < len(c):
            raise ValueError('Invalid chosen index')
        if g.status not in ('completed_observed', 'unknown'):
            raise ValueError('Invalid completion status')
        if not isinstance(g.seed, (int, np.integer)) or isinstance(g.seed, (bool, np.bool_)) or not 0 <= g.seed <= 0xffffffff:
            raise ValueError('Invalid seed')


def inputs(g):
    """Only feature arrays enter the scorer; metadata and labels are excluded."""
    return np.concatenate((np.broadcast_to(g.shared, (len(g.candidates), 275)), g.candidates), axis=1)


def probabilities(logits, mask):
    logits, mask = np.asarray(logits, float), np.asarray(mask, bool)
    if logits.ndim != 2 or logits.shape != mask.shape or not mask.any(axis=1).all() or not np.isfinite(logits[mask]).all():
        raise ValueError('Invalid logits or mask')
    z = np.where(mask, logits, -np.inf)
    z = np.exp(z - z.max(axis=1, keepdims=True))
    return z / z.sum(axis=1, keepdims=True)


def compare_logits(gpu_logits, cpu_logits):
    gpu = np.asarray(gpu_logits, dtype=np.float64)
    cpu = np.asarray(cpu_logits, dtype=np.float64)
    if gpu.ndim != 1 or not len(gpu) or gpu.shape != cpu.shape or not np.isfinite(gpu).all() or not np.isfinite(cpu).all():
        raise ValueError('Invalid inference comparison')
    error = float(np.max(np.abs(gpu-cpu)))
    if not np.allclose(gpu, cpu, atol=1e-4, rtol=1e-5) or np.argmax(gpu) != np.argmax(cpu):
        raise ValueError('GPU/CPU inference mismatch')
    return error


def baseline_fit(groups):
    groups = as_groups(groups)
    validate(groups)
    counts = {op: 1.0 for op in OPS}  # Laplace smoothing
    for g in groups:
        if g.status == 'completed_observed':
            counts[g.candidate_ops[g.chosen]] += 1
    return counts


def baseline_probabilities(ops, counts):
    ops = list(ops)
    if not ops or any(o not in OPS for o in ops) or any(not np.isfinite(counts[o]) or counts[o] <= 0 for o in OPS):
        raise ValueError('Invalid baseline input')
    present = set(ops)
    total = sum(counts[o] for o in present)
    return np.array([counts[o] / total / ops.count(o) for o in ops])


def _parity_forward(jax, x, params, device):
    with jax.default_matmul_precision('highest'):
        x = jax.device_put(x, device)
        for i in range(3):
            x = x @ params[f'w{i}'] + params[f'b{i}']
            if i < 2: x = jax.nn.relu(x)
        x.block_until_ready()
    return x


def fit(train_groups, schema, checkpoint_path, training_seed, expected_device, expected_device_kind, progress=None):
    """Train exactly 128 updates on completed groups, GPU required; never accepts test data."""
    started = time.monotonic()
    if progress is not None and not callable(progress): raise ValueError('Invalid progress callback')
    progress_state = dict(phase='initializing', updates_started=0, updates_completed=0,
                          parity_started=0, parity_completed=0, checkpoint_written=False,
                          device_validated=False)
    def report(phase, **changes):
        progress_state.update(phase=phase, **changes)
        if progress is not None: progress(progress_state.copy())
    report('initializing')
    train_groups = as_groups(train_groups)
    validate(train_groups)
    groups = [g for g in train_groups if g.status == 'completed_observed']
    if not groups:
        raise ValueError('No completed training groups')
    shared_names, names = schema['shared_feature_names'], schema['feature_names']
    if len(shared_names) != 275 or len(names) != 98 or schema['schema_sha256'] != schema_hash(names, shared_names):
        raise ValueError('Invalid schema')
    shared = np.asarray([g.shared for g in groups], dtype=np.float64)
    candidates = np.concatenate([np.asarray(g.candidates, dtype=np.float64) for g in groups])
    mean = np.concatenate((shared.mean(axis=0), candidates.mean(axis=0)))
    scale = np.maximum(1., np.concatenate((shared.std(axis=0), candidates.std(axis=0))) )
    mask = continuous_mask(shared_names + names)
    mean[~mask], scale[~mask] = 0., 1.
    import jax
    import jax.numpy as jnp
    import optax
    try:
        device = jax.devices('gpu')[0]
    except (RuntimeError, IndexError) as exc:
        raise RuntimeError('JAX GPU is required') from exc
    if device.platform != 'gpu' or not expected_device or str(device) != expected_device or not expected_device_kind or device.device_kind != expected_device_kind:
        raise RuntimeError('JAX GPU is required')
    report('device_validated', device_validated=True)
    if not isinstance(training_seed, (int, np.integer)) or isinstance(training_seed, (bool, np.bool_)) or not 0 <= training_seed <= 0xffffffff:
        raise ValueError('Invalid training seed')
    if max(len(g.candidates) for g in groups) > 65536:
        raise ValueError('Whole group exceeds padded candidate budget')
    rng = np.random.default_rng(training_seed)
    params = {k: jax.device_put(v, device) for i, (a,b) in enumerate(zip(DIMENSIONS[:-1], DIMENSIONS[1:])) for k,v in ((f'w{i}', (rng.standard_normal((a,b))*np.sqrt(2/a)).astype(np.float32)), (f'b{i}', np.zeros(b, np.float32)))}
    initial = {k: np.asarray(v).copy() for k,v in params.items()}
    optimizer = optax.chain(optax.clip_by_global_norm(1.), optax.adam(3e-4))
    state = optimizer.init(params)
    def loss(p, x, valid, labels, weights):
        for i in range(3):
            x = x @ p[f'w{i}'] + p[f'b{i}']
            if i < 2: x = jax.nn.relu(x)
        scores = jnp.where(valid, x[...,0], -jnp.inf)
        return jnp.sum((jax.scipy.special.logsumexp(scores, axis=1) - scores[jnp.arange(len(labels)), labels]) * weights) / jnp.sum(weights)
    @jax.jit
    def step(p, state, x, valid, labels, weights):
        value, grad = jax.value_and_grad(loss)(p, x, valid, labels, weights)
        updates, state = optimizer.update(grad, state, p)
        return optax.apply_updates(p, updates), state, value, optax.global_norm(grad)
    shapes, nonzero = set(), 0
    for update in range(128):
        if time.monotonic() - started > 600: raise TimeoutError('Synchronized training budget exceeded')
        ids = rng.permutation(len(groups))
        batch, width = [], 0
        for i in ids:
            candidate = groups[i]
            new_width = max(width, 1 << (len(candidate.candidates)-1).bit_length())
            if (1 << len(batch).bit_length())*new_width > 65536:
                continue
            batch.append(candidate)
            width = new_width
            if len(batch) == 256: break
        capacity = 1 << (len(batch)-1).bit_length()
        shapes.add((capacity, width))
        x = np.zeros((capacity, width, 373), np.float32)
        valid = np.zeros((capacity, width), bool)
        valid[:,0] = True
        labels = np.zeros(capacity, np.int32)
        weights = np.zeros(capacity, np.float32)
        for i,g in enumerate(batch):
            n = len(g.candidates)
            x[i,:n] = (inputs(g)-mean)/scale
            valid[i,:n] = True
            labels[i], weights[i] = g.chosen, 1
        report('update_started', updates_started=update+1)
        params, state, value, grad_norm = step(params, state, jax.device_put(x,device), jax.device_put(valid,device), jax.device_put(labels,device), jax.device_put(weights,device))
        grad_norm.block_until_ready()
        report('update_completed', updates_completed=update+1)
        if time.monotonic() - started > 600: raise TimeoutError('Synchronized training budget exceeded')
        if value.device != device or grad_norm.device != device or any(v.device != device for v in params.values()): raise RuntimeError('Training executed on unexpected device')
        if not np.isfinite(np.asarray(value)) or not np.isfinite(np.asarray(grad_norm)): raise ValueError('Nonfinite training output')
        nonzero += int(float(grad_norm) > 0)
    gpu_params = params
    params = {k: np.asarray(v) for k,v in params.items()}
    if not all(np.isfinite(v).all() for v in params.values()): raise ValueError('Nonfinite parameters')
    changed = any(not np.array_equal(initial[k], v) for k,v in params.items())
    if not nonzero or not changed: raise ValueError('No effective training updates')
    if time.monotonic() - started > 600: raise TimeoutError('Synchronized training budget exceeded')
    path = Path(checkpoint_path)
    if path.suffix != '.npz': raise ValueError('Checkpoint must end in .npz')
    if path.exists(): raise FileExistsError(path)
    metadata = dict(schema_version=SCHEMA_VERSION, schema_sha256=schema['schema_sha256'], shared_feature_names=shared_names, feature_names=names, ops=list(OPS), dimensions=list(DIMENSIONS), updates=128, model_seed=int(training_seed))
    # This file is private and unsealed until the caller records successful acceptance.
    # On failure retain it as evidence; never overwrite or retry.
    with path.open('xb') as f:
        np.savez_compressed(f, **params, mean=mean, scale=scale, metadata=np.asarray(json.dumps(metadata)))
    report('checkpoint_written', checkpoint_written=True)
    network = TaskNetwork(path, sha(path))
    compared_rows, max_error = 0, 0.
    for index, g in enumerate(groups[:16]):
        if time.monotonic() - started > 600: raise TimeoutError('Synchronized training budget exceeded')
        report('parity_started', parity_started=index+1)
        x = _parity_forward(jax, ((inputs(g)-mean)/scale).astype(np.float32), gpu_params, device)
        report('parity_completed', parity_completed=index+1)
        if x.device != device: raise RuntimeError('Inference executed on unexpected device')
        max_error = max(max_error, compare_logits(np.asarray(x[...,0]), network.score(g.shared, g.candidates)))
        compared_rows += len(g.candidates)
        if time.monotonic() - started > 600: raise TimeoutError('Synchronized training budget exceeded')
    return dict(checkpoint=str(path), sha256=sha(path), baseline=baseline_fit(groups), train_seeds=set(g.seed for g in train_groups), training_summary=dict(synchronized_updates=128, nonzero_gradient_updates=nonzero, weights_changed=changed, gpu_verified=True, shape_count=len(shapes), shapes=sorted(shapes), elapsed_seconds=time.monotonic()-started, inference_parity=dict(groups=min(16,len(groups)), candidate_rows=compared_rows, max_abs_error=max_error, synchronized=True, requested_matmul_precision='highest', hardware_precision_measured=False)))


def evaluate(groups, model, baseline, train_seeds, holdout=True):
    groups = as_groups(groups)
    validate(groups)
    if holdout and set(g.seed for g in groups) & set(train_seeds): raise ValueError('Train/evaluation seed overlap')
    rows = []
    for g in groups:
        logits = np.asarray(model.score(g.shared, g.candidates), float)
        if logits.shape != (len(g.candidates),): raise ValueError('Invalid model logits shape')
        p = probabilities(logits[None,:], np.ones((1,len(logits)), bool))[0]
        q = baseline_probabilities(g.candidate_ops, baseline)
        maximum = logits.max()
        model_ce = float(np.log(np.exp(logits-maximum).sum()) + (maximum - logits[g.chosen]))
        if not np.isfinite(model_ce): raise ValueError('Nonfinite model CE')
        rows.append((g, p, q, model_ce))
    def summarize(subset):
        if not subset: return dict(count=0, model_ce=None, model_top1=None, baseline_ce=None, baseline_top1=None, model_operation_accuracy=None, baseline_operation_accuracy=None)
        def metrics(j):
            return (float(np.mean([row[3] if j == 1 else -np.log(row[j][row[0].chosen]) for row in subset])), float(np.mean([np.argmax(row[j]) == row[0].chosen for row in subset])))
        mc, mt = metrics(1); bc, bt = metrics(2)
        return dict(count=len(subset), model_ce=mc, model_top1=mt, baseline_ce=bc, baseline_top1=bt, model_operation_accuracy=float(np.mean([r[0].candidate_ops[np.argmax(r[1])] == r[0].candidate_ops[r[0].chosen] for r in subset])), baseline_operation_accuracy=float(np.mean([r[0].candidate_ops[np.argmax(r[2])] == r[0].candidate_ops[r[0].chosen] for r in subset])))
    return dict(overall=summarize(rows), by_status={s:summarize([r for r in rows if r[0].status == s]) for s in ('completed_observed','unknown')}, by_seed={str(s):summarize([r for r in rows if r[0].seed == s]) for s in sorted(set(g.seed for g in groups), key=str)}, by_operation={o:summarize([r for r in rows if r[0].candidate_ops[r[0].chosen] == o]) for o in OPS}, multi_candidate=summarize([r for r in rows if len(r[0].candidates)>1]), cross_operation=summarize([r for r in rows if len(set(r[0].candidate_ops))>1]))
