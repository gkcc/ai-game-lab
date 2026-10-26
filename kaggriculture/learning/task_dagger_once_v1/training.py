"""One SHA-bound GPU warm-start update; no checkpoint search or CPU fallback.

The official environment and NumPy deployment remain CPU work.  Teacher labels
are targets only; only declared shared/candidate features enter the network.
"""
import argparse
from copy import deepcopy
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import shutil
import time

os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
import numpy as np

from learning.task_bc_v1.model import DIMENSIONS, TaskNetwork, numpy_logits, sha
from learning.task_bc_v1.train import (class_weights, load_split, microbatches,
                                      padded, verify_cache, weighted_group_loss)

ROOT = Path(__file__).resolve().parents[2]
PARAMS = ('w0', 'b0', 'w1', 'b1', 'w2', 'b2')
TRAINING = dict(updates=512, effective_batch_groups=256,
    old_groups_per_update=128, new_groups_per_update=128,
    learning_rate=1e-4, gradient_norm_clip=1.0,
    microbatch_padded_candidates=65536, training_real_row_limit=20000000,
    training_padded_row_limit=40000000, offline_row_limit=12000000,
    independent_offline_row_reserve=100000, synchronized_seconds=600,
    process_seconds=1200, new_dev_ce_ratio_max=.95,
    old_validation_ce_ratio_max=1.02, cpu_gpu_atol=1e-4,
    cpu_gpu_rtol=1e-5, offline_block_rows=8192)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')


def validate_config(cfg):
    actual = cfg.get('training')
    if (not isinstance(actual, dict) or set(actual) != set(TRAINING)
            or any(type(actual[k]) is not type(v) or actual[k] != v
                   for k, v in TRAINING.items())):
        raise ValueError('Registered fixed update configuration differs')
    if type(cfg.get('training_seed')) is not int or not 0 <= cfg['training_seed'] < 2**32:
        raise ValueError('A fixed uint32 training RNG seed is required')


def validate_device(platform, device_kind, expected):
    if (type(expected) is not str or not expected.strip()
            or platform != 'gpu' or type(device_kind) is not str
            or expected not in device_kind):
        raise ValueError('Registered GPU device is unavailable or mismatched')


def same_array_bits(left, right):
    left, right = np.asarray(left), np.asarray(right)
    return (left.dtype == right.dtype and left.shape == right.shape
            and left.tobytes(order='C') == right.tobytes(order='C'))


def warm_start_arrays(arrays):
    """No random reinitialization; float32 learning must preserve initial values."""
    values = {}
    for i in range(3):
        for name, shape in (('w'+str(i), (DIMENSIONS[i], DIMENSIONS[i+1])),
                            ('b'+str(i), (DIMENSIONS[i+1],))):
            raw = np.asarray(arrays[name])
            value = raw.astype(np.float32)
            if raw.shape != shape or not np.isfinite(raw).all() or not np.array_equal(raw, value):
                raise ValueError('Task tensors cannot be exactly warm-started in float32')
            values[name] = value.copy()
    normalizer = {k: np.asarray(arrays[k]).copy() for k in ('mean', 'scale')}
    if (any(v.shape != (373,) or not np.isfinite(v).all() for v in normalizer.values())
            or not np.all(normalizer['scale'] >= 1)):
        raise ValueError('Invalid frozen task normalizer')
    return values, normalizer


def array_digest(value):
    value = np.asarray(value)
    h = hashlib.sha256()
    h.update(str(value.dtype).encode('ascii'))
    h.update(json.dumps(list(value.shape)).encode('ascii'))
    h.update(value.tobytes(order='C'))
    return h.hexdigest()


class Consumption:
    """Debit before each real numerical evaluation; failed work is not refunded."""
    def __init__(self, limits=TRAINING):
        self.limits = limits
        self.counts = dict(training_real_candidate_rows=0,
            training_padded_candidate_rows=0, gradient_evaluations=0,
            cpu_offline_candidate_rows=0, gpu_offline_candidate_rows=0,
            optimizer_updates_issued=0, optimizer_updates_completed=0,
            old_train_group_presentations=0, new_train_group_presentations=0)

    def training(self, real, padded_rows):
        if type(real) is not int or type(padded_rows) is not int or not 0 < real <= padded_rows:
            raise ValueError('Invalid complete-group consumption')
        if (self.counts['training_real_candidate_rows']+real > self.limits['training_real_row_limit']
                or self.counts['training_padded_candidate_rows']+padded_rows > self.limits['training_padded_row_limit']):
            raise RuntimeError('Registered gradient row budget exhausted before call')
        self.counts['training_real_candidate_rows'] += real
        self.counts['training_padded_candidate_rows'] += padded_rows
        self.counts['gradient_evaluations'] += 1

    def offline(self, platform, rows):
        if platform not in ('cpu', 'gpu') or type(rows) is not int or rows <= 0:
            raise ValueError('Invalid offline scoring consumption')
        total = self.counts['cpu_offline_candidate_rows'] + self.counts['gpu_offline_candidate_rows']
        available = self.limits['offline_row_limit']-self.limits['independent_offline_row_reserve']
        if total+rows > available:
            raise RuntimeError('Registered offline row budget exhausted before call')
        self.counts[platform+'_offline_candidate_rows'] += rows

    def before_update(self):
        if self.counts['optimizer_updates_issued'] >= self.limits['updates']:
            raise RuntimeError('Registered optimizer update budget exhausted')
        self.counts['optimizer_updates_issued'] += 1


def mixture_indices(rng, old_count, new_count):
    if old_count <= 0 or new_count <= 0:
        raise ValueError('Both training pools must be nonempty')
    return (rng.integers(0, old_count, size=TRAINING['old_groups_per_update']),
            rng.integers(0, new_count, size=TRAINING['new_groups_per_update']))


def validate_split_seeds(old_train, old_validation, new_train, new_dev, cfg):
    train, dev, holdout = (set(cfg[k]) for k in ('train_seeds', 'validation_seeds', 'test_seeds'))
    if (set(map(int, new_train['seeds'])) != train or set(map(int, new_dev['seeds'])) != dev
            or train & dev or (train | dev) & holdout):
        raise ValueError('Training/development/holdout seed isolation failed')
    old_train_seeds = set(map(int, old_train['seeds']))
    old_validation_seeds = set(map(int, old_validation['seeds']))
    if old_train_seeds & old_validation_seeds or (old_train_seeds | old_validation_seeds) & (train | dev | holdout):
        raise ValueError('Old/new split isolation failed')


def compact_microbatches(data, ids):
    """Remove only empty padding rows from existing deterministic whole buckets."""
    for group_ids, maximum, width in microbatches(data, ids,
            budget=TRAINING['microbatch_padded_candidates']):
        capacity = min(maximum, 1 << (len(group_ids)-1).bit_length())
        yield group_ids, capacity, width


def numeric_block(data, start, stop, mean, scale):
    owners = np.searchsorted(data['offsets'][1:], np.arange(start, stop), side='right')
    raw = np.concatenate((data['shared'][owners], data['candidates'][start:stop]), axis=1).astype(np.float64)
    result = (raw-mean)/scale
    if not np.isfinite(result).all():
        raise ValueError('Nonfinite normalized feature inputs')
    return result


def score_cpu(params, data, mean, scale, consumed, deadline):
    result = np.empty(len(data['candidates']), np.float64)
    for start in range(0, len(result), TRAINING['offline_block_rows']):
        deadline()
        stop = min(start+TRAINING['offline_block_rows'], len(result))
        x = numeric_block(data, start, stop, mean, scale)
        consumed.offline('cpu', stop-start)
        result[start:stop] = numpy_logits(params, x)
    deadline()
    return result


def conditional_metrics(logits, data, weights):
    """Whole-group CE and first-maximum accuracy, including agreement examples."""
    logits = np.asarray(logits, dtype=np.float64)
    if logits.shape != (len(data['candidates']),) or not np.isfinite(logits).all():
        raise ValueError('Invalid complete candidate scores')
    total, denominator, correct = 0., 0., 0
    predictions = np.empty(len(data['labels']), dtype=np.int64)
    for i, label in enumerate(data['labels']):
        lo, hi = map(int, data['offsets'][i:i+2])
        if not 0 <= int(label) < hi-lo:
            raise ValueError('Label is not a current candidate')
        scores = logits[lo:hi]
        shifted = scores-scores.max()
        loss = float(np.log(np.exp(shifted).sum())-shifted[int(label)])
        weight = float(weights[int(data['selected_ops'][i])])
        if not np.isfinite(loss) or loss < 0 or not np.isfinite(weight) or weight <= 0:
            raise ValueError('Invalid weighted conditional loss')
        predictions[i] = int(np.argmax(scores))
        correct += int(predictions[i] == label)
        total += loss*weight
        denominator += weight
    if denominator <= 0:
        raise ValueError('Empty offline evaluation')
    return dict(groups=len(predictions), candidates=len(logits),
        weighted_ce=total/denominator, exact_top1=correct/len(predictions),
        prediction_sha256=array_digest(predictions))


def offline_gate(initial_dev, final_dev, initial_old, final_old, parity):
    values = [initial_dev['weighted_ce'], final_dev['weighted_ce'],
              initial_old['weighted_ce'], final_old['weighted_ce']]
    finite = all(np.isfinite(v) and v >= 0 for v in values)
    valid = finite and values[0] > 0 and values[2] > 0
    checks = dict(valid_positive_reference_ce=bool(valid),
        new_dev_ce_improves_five_percent=bool(valid and values[1] <= TRAINING['new_dev_ce_ratio_max']*values[0]),
        old_validation_ce_retained=bool(valid and values[3] <= TRAINING['old_validation_ce_ratio_max']*values[2]),
        export_cpu_gpu_parity=bool(parity.get('passed', False)))
    return dict(passed=all(checks.values()), checks=checks,
        new_dev_ce_ratio=values[1]/values[0] if valid else None,
        old_validation_ce_ratio=values[3]/values[2] if valid else None,
        engineering_screen_not_economic_evidence=True, checkpoint_selection=False)


def _fit(out, cfg, consumed, started):
    import jax
    import jax.numpy as jnp
    import optax

    out = Path(out)
    def deadline():
        if time.monotonic()-started > TRAINING['process_seconds']:
            raise TimeoutError('Registered fit process deadline')
    deadline()
    local_launch = (out/'gpu-launch.local.json').read_bytes()
    if hashlib.sha256(local_launch).hexdigest() != cfg['gpu_launch_config_sha256']:
        raise ValueError('Private GPU launch configuration identity changed')
    expected_device = json.loads(local_launch)['expected_device_substring']
    device = jax.devices('gpu')[0]
    validate_device(device.platform, str(device.device_kind), expected_device)
    old_cache, new_cache = ROOT/cfg['old_cache'], out/'cache'
    if sha(old_cache/'manifest.json') != cfg['old_cache_manifest_sha256']:
        raise ValueError('Old train cache manifest changed')
    # Exactly one actual array verification per cache here.  training_ready only
    # binds the manifest and source/launch identities, not these large arrays.
    old_manifest, new_manifest = verify_cache(old_cache), verify_cache(new_cache)
    schema, old_schema = read(new_cache/'schema.json'), read(old_cache/'schema.json')
    if schema != old_schema:
        raise ValueError('New student data changed the frozen feature schema')
    old_train, old_validation = load_split(old_cache, 'train'), load_split(old_cache, 'validation')
    new_train, new_dev = load_split(new_cache, 'train'), load_split(new_cache, 'validation')
    validate_split_seeds(old_train, old_validation, new_train, new_dev, cfg)
    if not np.isfinite(old_train['shared']).all() or not np.isfinite(new_train['shared']).all():
        raise ValueError('Invalid training shared features')
    counts, weights = class_weights(old_train['selected_ops'])
    model = out/'model'
    model.mkdir(exist_ok=False)
    original = ROOT/cfg['task_checkpoint']
    if sha(original) != cfg['task_checkpoint_sha256']:
        raise ValueError('Frozen initial task model changed')
    shutil.copyfile(original, model/'initial.npz')
    network = TaskNetwork(model/'initial.npz', cfg['task_checkpoint_sha256'])
    if any(network.metadata[k] != schema[k] for k in
            ('schema_version', 'schema_sha256', 'shared_feature_names', 'feature_names', 'ops')):
        raise ValueError('Initial model and task data schemas differ')
    with np.load(model/'initial.npz', allow_pickle=False) as source:
        host_params, raw_normalizer = warm_start_arrays(source)
    if any(not np.array_equal(network.params[k], host_params[k]) for k in PARAMS):
        raise ValueError('Original deployment and warm-start task tensors differ')
    mean, scale = network.mean.copy(), network.scale.copy()
    initial_params = {k: v.copy() for k, v in host_params.items()}
    initial_digest = {k: array_digest(v) for k, v in initial_params.items()}
    normalization_digest = {k: array_digest(v) for k, v in raw_normalizer.items()}
    write_new(model/'device-local.json', dict(device_kind=device.device_kind,
        device=str(device), platform=device.platform, jax=jax.__version__, optax=optax.__version__))
    initial_dev_logits = score_cpu(host_params, new_dev, mean, scale, consumed, deadline)
    initial_dev = conditional_metrics(initial_dev_logits, new_dev, weights)
    initial_old = conditional_metrics(score_cpu(host_params, old_validation, mean, scale, consumed, deadline), old_validation, weights)
    del initial_dev_logits
    params = jax.device_put(host_params, device)
    optimizer = optax.chain(optax.clip_by_global_norm(TRAINING['gradient_norm_clip']),
                           optax.adam(TRAINING['learning_rate']))
    state = optimizer.init(params)

    def forward(p, x):
        for layer in range(3):
            x = x @ p['w'+str(layer)] + p['b'+str(layer)]
            if layer < 2:
                x = jax.nn.relu(x)
        return x[..., 0]

    @jax.jit
    def value_grad(p, x, mask, labels, group_weights, divisor):
        def loss(current):
            return weighted_group_loss(forward(current, x), mask, labels, group_weights, divisor, jnp)
        return jax.value_and_grad(loss)(p)

    @jax.jit
    def apply(p, current_state, gradient):
        updates, next_state = optimizer.update(gradient, current_state, p)
        return optax.apply_updates(p, updates), next_state, optax.global_norm(gradient)

    compiled, compiled_apply = {}, None
    rng = np.random.default_rng(cfg['training_seed'])
    losses, norms, gradient_gpu = [], [], set()
    compile_seconds, packing_seconds = 0., 0.
    training_started = time.monotonic()
    with jax.default_matmul_precision('highest'):
        for update in range(TRAINING['updates']):
            deadline()
            if time.monotonic()-training_started > TRAINING['synchronized_seconds']:
                raise TimeoutError('Registered training-loop deadline')
            old_ids, new_ids = mixture_indices(rng, len(old_train['labels']), len(new_train['labels']))
            denominator = np.float32(weights[old_train['selected_ops'][old_ids]].sum(dtype=np.float64)
                + weights[new_train['selected_ops'][new_ids]].sum(dtype=np.float64))
            if not np.isfinite(denominator) or denominator <= 0:
                raise ValueError('Invalid effective-batch weight denominator')
            gradient = jax.tree_util.tree_map(jnp.zeros_like, params)
            batch_loss = jnp.asarray(0., dtype=jnp.float32)
            for source, ids, population in ((old_train, old_ids, 'old'), (new_train, new_ids, 'new')):
                for group_ids, capacity, width in compact_microbatches(source, ids):
                    deadline()
                    tick = time.monotonic()
                    arrays = padded(source, group_ids, capacity, width, mean, scale, weights)
                    if not np.isfinite(arrays[0]).all():
                        raise ValueError('Nonfinite training feature values')
                    packing_seconds += time.monotonic()-tick
                    placed = [jax.device_put(v, device) for v in arrays]
                    divisor = jax.device_put(denominator, device)
                    key = (capacity, width)
                    if key not in compiled:
                        tick = time.monotonic()
                        compiled[key] = value_grad.lower(params, *placed, divisor).compile()
                        compile_seconds += time.monotonic()-tick
                    deadline()
                    if time.monotonic()-training_started > TRAINING['synchronized_seconds']:
                        raise TimeoutError('Registered training-loop deadline')
                    real = sum(int(source['offsets'][i+1]-source['offsets'][i]) for i in group_ids)
                    consumed.training(real, capacity*width)
                    consumed.counts[population+'_train_group_presentations'] += len(group_ids)
                    value, part = compiled[key](params, *placed, divisor)
                    gradient = jax.tree_util.tree_map(lambda a, b: a+b, gradient, part)
                    batch_loss = batch_loss+value
            if compiled_apply is None:
                tick = time.monotonic()
                compiled_apply = apply.lower(params, state, gradient).compile()
                compile_seconds += time.monotonic()-tick
            deadline()
            if time.monotonic()-training_started > TRAINING['synchronized_seconds']:
                raise TimeoutError('Registered training-loop deadline')
            consumed.before_update()
            params, state, norm = compiled_apply(params, state, gradient)
            norm.block_until_ready()
            loss_value, norm_value = float(batch_loss), float(norm)
            consumed.counts['optimizer_updates_completed'] += 1
            if not np.isfinite(loss_value) or not np.isfinite(norm_value):
                raise ValueError('Nonfinite synchronized GPU loss or gradient')
            gradient_gpu.update(d.platform for d in norm.devices())
            losses.append(loss_value)
            norms.append(norm_value)
            if (update+1) % 64 == 0:
                print(json.dumps(dict(updates=update+1, loss=loss_value, gradient_norm=norm_value)), flush=True)
            deadline()
            if time.monotonic()-training_started > TRAINING['synchronized_seconds']:
                raise TimeoutError('Registered synchronized training-loop deadline')
    train_seconds = time.monotonic()-training_started
    if any({d.platform for d in value.devices()} != {'gpu'} for value in params.values()):
        raise ValueError('Updated task tensors left the registered GPU')
    host_params = {k: np.asarray(v).copy() for k, v in params.items()}
    if (any(not np.isfinite(v).all() for v in host_params.values())
            or gradient_gpu != {'gpu'} or not any(n > 0 for n in norms)
            or not any(not np.array_equal(initial_params[k], host_params[k]) for k in PARAMS)):
        raise ValueError('No valid GPU task-weight update was established')
    if consumed.counts['old_train_group_presentations'] != 65536 or consumed.counts['new_train_group_presentations'] != 65536:
        raise ValueError('Balanced fixed presentation counts changed')
    metadata = deepcopy(network.metadata)
    metadata.update(experiment='task_dagger_once_v1', updates=512, parent_checkpoint_sha256=cfg['task_checkpoint_sha256'],
        parent_epoch=network.metadata.get('epoch'), epoch=None, model_seed=cfg['training_seed'],
        training=TRAINING, training_seed=cfg['training_seed'],
        cache_manifest_sha256=sha(new_cache/'manifest.json'), old_cache_manifest_sha256=cfg['old_cache_manifest_sha256'],
        checkpoint_selection='single final update512; no model or epoch search', normalization='exact original mean/scale',
        optimizer='fresh Adam moments; one fixed warm-start update run', teacher_label_role='target only')
    with (model/'trained.npz').open('xb') as stream:
        np.savez_compressed(stream, **host_params, **raw_normalizer,
                            metadata=np.asarray(json.dumps(metadata, sort_keys=True)))
    final_sha = sha(model/'trained.npz')
    write_new(out/'model-freeze.json', dict(checkpoint_sha256=final_sha, fit_count=1,
        updates=512, registration_sha256=sha(out/'registration.json'),
        cache_manifest_sha256=sha(new_cache/'manifest.json'), initial_task_sha256=cfg['task_checkpoint_sha256']))
    exported = TaskNetwork(model/'trained.npz', final_sha)
    with np.load(model/'trained.npz', allow_pickle=False) as final:
        if any(not same_array_bits(raw_normalizer[k], final[k]) for k in raw_normalizer):
            raise ValueError('Export changed frozen normalizer bytes')
    if any(not np.array_equal(exported.params[k], host_params[k]) for k in PARAMS):
        raise ValueError('Export changed final task tensor values')
    final_dev_logits = score_cpu(exported.params, new_dev, exported.mean, exported.scale, consumed, deadline)
    final_dev = conditional_metrics(final_dev_logits, new_dev, weights)
    final_old = conditional_metrics(score_cpu(exported.params, old_validation, exported.mean, exported.scale,
        consumed, deadline), old_validation, weights)
    # In-memory final values must reproduce the reloaded deployed values. This
    # is a separate counted CPU evaluation, not inferred from export metadata.
    in_memory = score_cpu(host_params, new_dev, mean, scale, consumed, deadline)
    if not np.array_equal(in_memory, final_dev_logits):
        raise ValueError('Reloaded CPU deployment changed task logits')
    del in_memory
    infer = jax.jit(forward)
    gpu_logits = np.empty_like(final_dev_logits)
    gpu_params = jax.device_put({k: v.astype(np.float32) for k, v in exported.params.items()}, device)
    with jax.default_matmul_precision('highest'):
        for start in range(0, len(gpu_logits), TRAINING['offline_block_rows']):
            deadline()
            stop = min(start+TRAINING['offline_block_rows'], len(gpu_logits))
            x = numeric_block(new_dev, start, stop, exported.mean, exported.scale).astype(np.float32)
            consumed.offline('gpu', stop-start)
            scored = infer(gpu_params, jax.device_put(x, device))
            scored.block_until_ready()
            if {d.platform for d in scored.devices()} != {'gpu'}:
                raise ValueError('GPU parity computation ran on another device')
            gpu_logits[start:stop] = np.asarray(scored)
    disagreement = sum(int(np.argmax(final_dev_logits[lo:hi])) != int(np.argmax(gpu_logits[lo:hi]))
        for lo, hi in zip(new_dev['offsets'][:-1], new_dev['offsets'][1:]))
    close = bool(np.isfinite(gpu_logits).all() and np.allclose(final_dev_logits, gpu_logits,
        atol=TRAINING['cpu_gpu_atol'], rtol=TRAINING['cpu_gpu_rtol']))
    parity = dict(passed=close and disagreement == 0, groups=len(new_dev['labels']),
        candidates=len(gpu_logits), cpu_dtype='float64', gpu_dtype='float32',
        atol=TRAINING['cpu_gpu_atol'], rtol=TRAINING['cpu_gpu_rtol'],
        max_absolute_logit_error=float(np.max(np.abs(final_dev_logits-gpu_logits))),
        first_argmax_disagreements=disagreement, reloaded_cpu_logits_exact=True)
    if not np.isfinite(parity['max_absolute_logit_error']):
        raise ValueError('Nonfinite export parity result')
    for name, values in (('final-dev-cpu-logits.npy', final_dev_logits), ('final-dev-gpu-logits.npy', gpu_logits)):
        with (model/name).open('xb') as stream:
            np.save(stream, values, allow_pickle=False)
    gate = offline_gate(initial_dev, final_dev, initial_old, final_old, parity)
    gate.update(checkpoint_sha256=final_sha, initial_task_sha256=cfg['task_checkpoint_sha256'],
        metrics=dict(initial_new_dev=initial_dev, final_new_dev=final_dev,
                     initial_old_validation=initial_old, final_old_validation=final_old), export_parity=parity)
    if sha(original) != cfg['task_checkpoint_sha256'] or sha(model/'initial.npz') != cfg['task_checkpoint_sha256']:
        raise ValueError('Original model identity changed during training')
    deadline()
    offline_used = consumed.counts['cpu_offline_candidate_rows']+consumed.counts['gpu_offline_candidate_rows']
    report = dict(schema='task-dagger-once-fit-v1', completed=True, fit_count=1, optimizer_updates=512,
        checkpoint_sha256=final_sha, initial_task_sha256=cfg['task_checkpoint_sha256'],
        crop_checkpoint_sha256=cfg['crop_checkpoint_sha256'], registration_sha256=sha(out/'registration.json'),
        new_cache_manifest_sha256=sha(new_cache/'manifest.json'),
        old_cache_manifest_sha256=cfg['old_cache_manifest_sha256'],
        gpu_training=True, actual_gpu=True, gpu_platform='gpu', expected_device_verified=True,
        gpu_speedup_measured=False, official_simulation_on_gpu=False,
        nonzero_gradient_updates=sum(n > 0 for n in norms), gradient_norm_min=min(norms),
        gradient_norm_max=max(norms), loss_min=min(losses), loss_max=max(losses),
        changed_parameter_tensors=sum(not np.array_equal(initial_params[k], host_params[k]) for k in PARAMS),
        initial_parameter_sha256=initial_digest, final_parameter_sha256={k:array_digest(v) for k,v in host_params.items()},
        frozen_normalizer_array_sha256=normalization_digest, frozen_normalizer_bits_preserved=True,
        old_training_class_counts=counts.tolist(), old_training_class_weights=weights.tolist(),
        old_train_groups=len(old_train['labels']), new_train_groups=len(new_train['labels']),
        new_dev_groups=len(new_dev['labels']), old_validation_groups=len(old_validation['labels']),
        normalization_recomputed=False, class_weights_use_old_training_only=True,
        offline_input_representation='Float32 ragged cache values scored with float64 CPU deployment arithmetic; raw live observation actions are verified separately in official matches.',
        consumption=dict(consumed.counts), independent_offline_row_reserve=TRAINING['independent_offline_row_reserve'],
        offline_rows_total_used=offline_used, offline_total_limit=TRAINING['offline_row_limit'],
        offline_remaining_total=TRAINING['offline_row_limit']-offline_used,
        budget_interpretation='Training rows count one full value-and-gradient evaluation presentation, including padding; not FLOPs. CPU and GPU offline calls are separate row evaluations. Reserved independent rows are not spent by this fit.',
        cache_array_verifications={'old':1,'new':1}, old_cache_arrays=len(old_manifest['arrays']['train'])+len(old_manifest['arrays']['validation']),
        new_cache_arrays=len(new_manifest['arrays']['train'])+len(new_manifest['arrays']['validation']),
        timings=dict(synchronized_training_loop_seconds=train_seconds, explicit_compile_seconds=compile_seconds,
                     training_packing_seconds=packing_seconds, process_seconds=time.monotonic()-started,
                     training_loop_includes='packing, compilation, transfers, all updates synchronized, and reporting'),
        offline_gate_passed=gate['passed'], competitive_improvement_established=False,
        model_selection=False, retry_allowed=False,
        software=dict(numpy=np.__version__, jax=jax.__version__, optax=optax.__version__))
    write_new(out/'fit-result.json', report)
    write_new(out/'offline-gate.json', gate)
    deadline()


def _worker(out, cfg, started):
    consumed = Consumption()
    try:
        _fit(Path(out), cfg, consumed, started)
    except BaseException as exc:
        path = Path(out)/'training-failed.json'
        if not path.exists():
            write_new(path, dict(kind=type(exc).__name__, fit_complete=False,
                retry_allowed=False, games_allowed=False, consumption=dict(consumed.counts),
                consumption_is_debited_work_not_guaranteed_completed=True,
                elapsed_seconds=time.monotonic()-started))
        raise


def fit(out):
    # Delayed import keeps pure CPU contract tests independent of preregistration.
    from .experiment import training_ready
    out = Path(out).resolve()
    cfg = training_ready(out)
    validate_config(cfg)
    write_new(out/'training-started.json', dict(fit_count=1, updates=512,
        registration_sha256=sha(out/'registration.json'), process_seconds=1200,
        retry_allowed=False, checkpoint_selection=False))
    started = time.monotonic()
    process = multiprocessing.get_context('spawn').Process(target=_worker, args=(str(out), cfg, started))
    try:
        process.start()
        process.join(max(0., TRAINING['process_seconds']-(time.monotonic()-started)))
        if process.is_alive():
            raise TimeoutError('Hard registered fit process deadline')
        if process.exitcode != 0:
            raise RuntimeError('Single GPU fit failed; preserve partial evidence')
        report, gate = read(out/'fit-result.json'), read(out/'offline-gate.json')
        if not report['completed'] or sha(out/'model/trained.npz') != report['checkpoint_sha256'] or report['checkpoint_sha256'] != gate['checkpoint_sha256']:
            raise ValueError('Missing or inconsistent fit completion')
        return report
    except BaseException as exc:
        if process.is_alive():
            process.terminate()
            process.join(5)
            if process.is_alive():
                process.kill()
                process.join(5)
        path = out/'training-failed.json'
        if not path.exists():
            write_new(path, dict(kind=type(exc).__name__, retry_allowed=False,
                fit_complete=False, games_allowed=False, partial_accounting_unavailable=True,
                registered_caps_remain_consumed_bounds=True,
                elapsed_seconds=time.monotonic()-started))
        raise
    finally:
        if not process.is_alive():
            process.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True, type=Path)
    fit(parser.parse_args().out)
