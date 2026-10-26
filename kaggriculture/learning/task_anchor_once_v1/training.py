"""One SHA-bound GPU update with a frozen original-S old-distribution anchor.

The official environment and NumPy deployment remain CPU work.  Teacher labels
are targets only; only declared shared/candidate features enter the network.
The frozen anchor is a training-only target, absent from deployed inference.
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
    anchor_coefficient=1.0, anchor_temperature=1.0,
    microbatch_padded_candidates=65536, training_real_row_limit=20000000,
    training_padded_row_limit=40000000, anchor_real_row_limit=10000000,
    anchor_padded_row_limit=20000000, offline_row_limit=12000000,
    independent_offline_row_reserve=100000, synthetic_row_reserve=1024,
    synchronized_seconds=600,
    process_seconds=1200, known_dev_ce_ratio_max=.95,
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
    if type(cfg.get('training_seed')) is not int or cfg['training_seed'] != 2026102707:
        raise ValueError('The original PCG64 training RNG seed is required')


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
            anchor_real_candidate_rows=0, anchor_padded_candidate_rows=0,
            anchor_forward_evaluations=0, anchor_train_group_presentations=0,
            cpu_offline_candidate_rows=0, gpu_offline_candidate_rows=0,
            optimizer_updates_issued=0, optimizer_updates_completed=0,
            old_train_group_presentations=0, dagger_train_group_presentations=0)

    def training(self, real, padded_rows):
        if type(real) is not int or type(padded_rows) is not int or not 0 < real <= padded_rows:
            raise ValueError('Invalid complete-group consumption')
        if (self.counts['training_real_candidate_rows']+real > self.limits['training_real_row_limit']
                or self.counts['training_padded_candidate_rows']+padded_rows > self.limits['training_padded_row_limit']):
            raise RuntimeError('Registered gradient row budget exhausted before call')
        self.counts['training_real_candidate_rows'] += real
        self.counts['training_padded_candidate_rows'] += padded_rows
        self.counts['gradient_evaluations'] += 1

    def anchor(self, real, padded_rows):
        if type(real) is not int or type(padded_rows) is not int or not 0 < real <= padded_rows:
            raise ValueError('Invalid complete-group anchor consumption')
        if (self.counts['anchor_real_candidate_rows']+real > self.limits['anchor_real_row_limit']
                or self.counts['anchor_padded_candidate_rows']+padded_rows > self.limits['anchor_padded_row_limit']):
            raise RuntimeError('Registered frozen-S row budget exhausted before call')
        self.counts['anchor_real_candidate_rows'] += real
        self.counts['anchor_padded_candidate_rows'] += padded_rows
        self.counts['anchor_forward_evaluations'] += 1

    def offline(self, platform, rows):
        if platform not in ('cpu', 'gpu') or type(rows) is not int or rows <= 0:
            raise ValueError('Invalid offline scoring consumption')
        total = self.counts['cpu_offline_candidate_rows'] + self.counts['gpu_offline_candidate_rows']
        available = (self.limits['offline_row_limit']-self.limits['independent_offline_row_reserve']
                     - self.limits['synthetic_row_reserve'])
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


def validate_split_seeds(old_train, old_validation, dagger_train, known_dev, cfg):
    """Cached seed roles remain historical; only the official panel is fresh."""
    configured = {}
    for name, count in (('training_source_seeds', 4), ('known_screen_seeds', 2),
                        ('validation_seeds', 2), ('test_seeds', 8)):
        values = cfg.get(name)
        if (type(values) is not list or len(values) != count
                or any(type(v) is not int or not 0 <= v < 2**32 for v in values)
                or len(set(values)) != count):
            raise ValueError('Invalid explicit seed role: '+name)
        configured[name] = set(values)
    actual = []
    for source in (old_train, old_validation, dagger_train, known_dev):
        values = np.asarray(source['seeds'])
        if (values.ndim != 1 or not len(values) or values.dtype.kind not in 'iu'
                or np.any(values < 0) or np.any(values >= 2**32)):
            raise ValueError('Invalid cached seed array')
        actual.append(set(map(int, values)))
    if (actual[2] != configured['training_source_seeds']
            or actual[3] != configured['known_screen_seeds']):
        raise ValueError('Cached training/known-screen identity differs from registration')
    # The original cache arrays provide the old train and old-validation seed
    # identities; these are never relabelled with the fresh official seeds.
    roles = [*actual, configured['validation_seeds'], configured['test_seeds']]
    if any(left & right for i, left in enumerate(roles) for right in roles[i+1:]):
        raise ValueError('Historical train/screens and fresh official seeds overlap')
    return dict(old_training_source_seeds=sorted(actual[0]),
        old_validation_source_seeds=sorted(actual[1]),
        training_source_seeds=sorted(actual[2]), known_screen_seeds=sorted(actual[3]),
        fresh_validation_seeds=sorted(configured['validation_seeds']),
        fresh_test_seeds=sorted(configured['test_seeds']),
        known_screens_are_independent=False)


def masked_log_softmax(logits, mask, xp=np):
    """Finite masked log-probabilities, including unused all-padding groups."""
    has_candidates = xp.any(mask, axis=1, keepdims=True)
    masked = xp.where(mask, logits, -xp.inf)
    maximum = xp.where(has_candidates, xp.max(masked, axis=1, keepdims=True), 0.)
    shifted = xp.where(mask, xp.where(mask, logits, 0.)-maximum, 0.)
    total = xp.sum(xp.where(mask, xp.exp(shifted), 0.), axis=1, keepdims=True)
    log_total = xp.log(xp.where(has_candidates, total, 1.))
    return xp.where(mask, shifted-log_total, 0.)


def weighted_anchor_kl(logits, frozen_logits, mask, group_weights, denominator,
                       xp=np, stop_gradient=None):
    """Weighted KL(p_S || p_C), with its own old-group weight denominator."""
    prior = masked_log_softmax(frozen_logits, mask, xp)
    if stop_gradient is not None:
        prior = stop_gradient(prior)
    current = masked_log_softmax(logits, mask, xp)
    probabilities = xp.where(mask, xp.exp(prior), 0.)
    # Masked entries use finite zero logs, so padding cannot create 0*(-inf).
    losses = xp.sum(probabilities*(prior-current), axis=1)
    return xp.sum(losses*group_weights)/denominator


def anchored_group_loss(logits, frozen_logits, mask, labels, group_weights,
                        ce_denominator, anchor_denominator, xp=np,
                        stop_gradient=None):
    return (weighted_group_loss(logits, mask, labels, group_weights, ce_denominator, xp)
        + TRAINING['anchor_coefficient']*weighted_anchor_kl(logits, frozen_logits,
            mask, group_weights, anchor_denominator, xp, stop_gradient))


def numerical_check_groups(splits):
    """Freeze midpoint groups from offsets before any score or error is read."""
    chosen = {}
    for name, source in splits.items():
        n = len(source['labels'])
        if n < 16:
            raise ValueError('Independent numerical check requires sixteen complete groups')
        indices = [(2*i+1)*n//32 for i in range(16)]
        rows = sum(int(source['offsets'][i+1]-source['offsets'][i]) for i in indices)
        if rows > 6250:
            raise ValueError('Fixed independent numerical groups exceed the split row cap')
        chosen[name] = dict(groups=16, group_indices=indices, candidate_rows=rows,
            offset_array_sha256=array_digest(source['offsets']))
    # Four split populations x initial/final x CPU/GPU; this is a reservation,
    # never permission for this fit to spend the independent review allowance.
    presentations = sum(part['candidate_rows'] for part in chosen.values())*4
    if presentations > TRAINING['independent_offline_row_reserve']:
        raise ValueError('Fixed independent numerical groups exceed the total row reserve')
    return dict(schema='task-anchor-independent-numerical-groups-v1', splits=chosen,
        initial_and_final=True, cpu_and_gpu=True, candidate_row_presentations=presentations,
        reserved_rows=TRAINING['independent_offline_row_reserve'],
        selection='floor((i+0.5)*N/16), i=0..15; complete groups; no replacement',
        scored_by_fit=False)


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
        known_dev_ce_improves_five_percent=bool(valid and values[1] <= TRAINING['known_dev_ce_ratio_max']*values[0]),
        old_validation_ce_retained=bool(valid and values[3] <= TRAINING['old_validation_ce_ratio_max']*values[2]),
        export_cpu_gpu_parity=bool(parity.get('passed', False)))
    return dict(passed=all(checks.values()), checks=checks,
        known_dev_ce_ratio=values[1]/values[0] if valid else None,
        old_validation_ce_ratio=values[3]/values[2] if valid else None,
        engineering_screen_not_economic_evidence=True, checkpoint_selection=False)


def _fit(out, cfg, consumed, started, output_budget):
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
    old_cache = (ROOT/cfg['old_cache']).resolve()
    dagger_cache = (ROOT/cfg['dagger_cache']).resolve()
    if dagger_cache in (old_cache, (out/'cache').resolve()):
        raise ValueError('DAgger inputs must reuse the separate frozen historical cache')
    if sha(old_cache/'manifest.json') != cfg['old_cache_manifest_sha256']:
        raise ValueError('Old train cache manifest changed')
    if sha(dagger_cache/'manifest.json') != cfg['dagger_cache_manifest_sha256']:
        raise ValueError('Historical DAgger cache manifest changed')
    # Exactly one actual array verification per cache here.  training_ready only
    # binds the manifest and source/launch identities, not these large arrays.
    old_manifest, dagger_manifest = verify_cache(old_cache), verify_cache(dagger_cache)
    schema, old_schema = read(dagger_cache/'schema.json'), read(old_cache/'schema.json')
    if schema != old_schema:
        raise ValueError('Reused student data changed the frozen feature schema')
    old_train, old_validation = load_split(old_cache, 'train'), load_split(old_cache, 'validation')
    dagger_train, known_dev = load_split(dagger_cache, 'train'), load_split(dagger_cache, 'validation')
    seed_roles = validate_split_seeds(old_train, old_validation, dagger_train, known_dev, cfg)
    if not np.isfinite(old_train['shared']).all() or not np.isfinite(dagger_train['shared']).all():
        raise ValueError('Invalid training shared features')
    counts, weights = class_weights(old_train['selected_ops'])
    model = out/'model'
    model.mkdir(exist_ok=False)
    write_new(model/'numerical-check-groups.json', numerical_check_groups(dict(
        old_train=old_train, old_validation=old_validation,
        dagger_train=dagger_train, known_dev=known_dev)))
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
    initial_dev_logits = score_cpu(host_params, known_dev, mean, scale, consumed, deadline)
    initial_dev = conditional_metrics(initial_dev_logits, known_dev, weights)
    initial_old_logits = score_cpu(host_params, old_validation, mean, scale, consumed, deadline)
    initial_old = conditional_metrics(initial_old_logits, old_validation, weights)
    for name, values in (('initial-known-dev-cpu-logits.npy', initial_dev_logits),
                         ('initial-old-validation-cpu-logits.npy', initial_old_logits)):
        with (model/name).open('xb') as stream:
            np.save(stream, values, allow_pickle=False)
    del initial_dev_logits, initial_old_logits
    params = jax.device_put(host_params, device)
    # A separate immutable tree is never passed to the optimizer. Its forward
    # output is stopped again inside the KL objective for an explicit boundary.
    frozen_params = jax.tree_util.tree_map(jax.lax.stop_gradient,
        jax.device_put(initial_params, device))
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
    def anchor_forward(p, x):
        return jax.lax.stop_gradient(forward(p, x))

    @jax.jit
    def anchored_value_grad(p, x, mask, labels, group_weights, divisor,
                            anchor_divisor, frozen_logits):
        def loss(current):
            return anchored_group_loss(forward(current, x), frozen_logits, mask,
                labels, group_weights, divisor, anchor_divisor, jnp,
                jax.lax.stop_gradient)
        return jax.value_and_grad(loss)(p)

    @jax.jit
    def apply(p, current_state, gradient):
        updates, next_state = optimizer.update(gradient, current_state, p)
        return optax.apply_updates(p, updates), next_state, optax.global_norm(gradient)

    compiled, compiled_anchor, compiled_apply = {}, {}, None
    rng = np.random.Generator(np.random.PCG64(cfg['training_seed']))
    losses, norms, gradient_gpu, anchor_gpu = [], [], set(), set()
    compile_seconds, packing_seconds = 0., 0.
    training_started = time.monotonic()
    with jax.default_matmul_precision('highest'):
        for update in range(TRAINING['updates']):
            deadline()
            if time.monotonic()-training_started > TRAINING['synchronized_seconds']:
                raise TimeoutError('Registered training-loop deadline')
            old_ids, new_ids = mixture_indices(rng, len(old_train['labels']), len(dagger_train['labels']))
            old_weight_sum = weights[old_train['selected_ops'][old_ids]].sum(dtype=np.float64)
            denominator = np.float32(old_weight_sum
                + weights[dagger_train['selected_ops'][new_ids]].sum(dtype=np.float64))
            anchor_denominator = np.float32(old_weight_sum)
            if (not np.isfinite(denominator) or denominator <= 0
                    or not np.isfinite(anchor_denominator) or anchor_denominator <= 0):
                raise ValueError('Invalid effective-batch weight denominator')
            gradient = jax.tree_util.tree_map(jnp.zeros_like, params)
            batch_loss = jnp.asarray(0., dtype=jnp.float32)
            for source, ids, population in ((old_train, old_ids, 'old'), (dagger_train, new_ids, 'dagger')):
                for group_ids, capacity, width in compact_microbatches(source, ids):
                    deadline()
                    tick = time.monotonic()
                    arrays = padded(source, group_ids, capacity, width, mean, scale, weights)
                    if not np.isfinite(arrays[0]).all():
                        raise ValueError('Nonfinite training feature values')
                    packing_seconds += time.monotonic()-tick
                    placed = [jax.device_put(v, device) for v in arrays]
                    divisor = jax.device_put(denominator, device)
                    key = (population, capacity, width)
                    anchor_key = (capacity, width)
                    real = sum(int(source['offsets'][i+1]-source['offsets'][i]) for i in group_ids)
                    if population == 'old':
                        if anchor_key not in compiled_anchor:
                            tick = time.monotonic()
                            compiled_anchor[anchor_key] = anchor_forward.lower(frozen_params, placed[0]).compile()
                            compile_seconds += time.monotonic()-tick
                        deadline()
                        if time.monotonic()-training_started > TRAINING['synchronized_seconds']:
                            raise TimeoutError('Registered training-loop deadline')
                        consumed.anchor(real, capacity*width)
                        consumed.counts['anchor_train_group_presentations'] += len(group_ids)
                        frozen_logits = compiled_anchor[anchor_key](frozen_params, placed[0])
                        anchor_gpu.update(d.platform for d in frozen_logits.devices())
                        arguments = (*placed, divisor,
                            jax.device_put(anchor_denominator, device), frozen_logits)
                        evaluator = anchored_value_grad
                    else:
                        arguments = (*placed, divisor)
                        evaluator = value_grad
                    if key not in compiled:
                        tick = time.monotonic()
                        compiled[key] = evaluator.lower(params, *arguments).compile()
                        compile_seconds += time.monotonic()-tick
                    deadline()
                    if time.monotonic()-training_started > TRAINING['synchronized_seconds']:
                        raise TimeoutError('Registered training-loop deadline')
                    consumed.training(real, capacity*width)
                    consumed.counts[population+'_train_group_presentations'] += len(group_ids)
                    value, part = compiled[key](params, *arguments)
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
            or gradient_gpu != {'gpu'} or anchor_gpu != {'gpu'} or not any(n > 0 for n in norms)
            or not any(not np.array_equal(initial_params[k], host_params[k]) for k in PARAMS)):
        raise ValueError('No valid GPU task-weight update was established')
    if any(consumed.counts[k] != 65536 for k in
            ('old_train_group_presentations', 'dagger_train_group_presentations',
             'anchor_train_group_presentations')):
        raise ValueError('Balanced fixed presentation counts changed')
    if any(not same_array_bits(initial_params[k], np.asarray(frozen_params[k])) for k in PARAMS):
        raise ValueError('Frozen original-S anchor tensors changed')
    metadata = deepcopy(network.metadata)
    metadata.update(experiment='task_anchor_once_v1', updates=512, parent_checkpoint_sha256=cfg['task_checkpoint_sha256'],
        parent_epoch=network.metadata.get('epoch'), epoch=None, model_seed=cfg['training_seed'],
        training=TRAINING, training_seed=cfg['training_seed'],
        cache_manifest_sha256=sha(dagger_cache/'manifest.json'), old_cache_manifest_sha256=cfg['old_cache_manifest_sha256'],
        checkpoint_selection='single final update512; no model or epoch search', normalization='exact original mean/scale',
        optimizer='fresh Adam moments; one fixed warm-start update run', teacher_label_role='target only',
        anchor='KL(p_original_S||p_candidate), old training groups only; coefficient1 temperature1',
        anchor_in_deployment=False)
    with (model/'trained.npz').open('xb') as stream:
        np.savez_compressed(stream, **host_params, **raw_normalizer,
                            metadata=np.asarray(json.dumps(metadata, sort_keys=True)))
    final_sha = sha(model/'trained.npz')
    write_new(out/'model-freeze.json', dict(checkpoint_sha256=final_sha, fit_count=1,
        updates=512, registration_sha256=sha(out/'registration.json'),
        cache_manifest_sha256=sha(dagger_cache/'manifest.json'), initial_task_sha256=cfg['task_checkpoint_sha256']))
    exported = TaskNetwork(model/'trained.npz', final_sha)
    with np.load(model/'trained.npz', allow_pickle=False) as final:
        if any(not same_array_bits(raw_normalizer[k], final[k]) for k in raw_normalizer):
            raise ValueError('Export changed frozen normalizer bytes')
    if any(not np.array_equal(exported.params[k], host_params[k]) for k in PARAMS):
        raise ValueError('Export changed final task tensor values')
    final_dev_logits = score_cpu(exported.params, known_dev, exported.mean, exported.scale, consumed, deadline)
    final_dev = conditional_metrics(final_dev_logits, known_dev, weights)
    final_old_logits = score_cpu(exported.params, old_validation, exported.mean, exported.scale, consumed, deadline)
    final_old = conditional_metrics(final_old_logits, old_validation, weights)
    # In-memory final values must reproduce the reloaded deployed values. This
    # is a separate counted CPU evaluation, not inferred from export metadata.
    in_memory = score_cpu(host_params, known_dev, mean, scale, consumed, deadline)
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
            x = numeric_block(known_dev, start, stop, exported.mean, exported.scale).astype(np.float32)
            consumed.offline('gpu', stop-start)
            scored = infer(gpu_params, jax.device_put(x, device))
            scored.block_until_ready()
            if {d.platform for d in scored.devices()} != {'gpu'}:
                raise ValueError('GPU parity computation ran on another device')
            gpu_logits[start:stop] = np.asarray(scored)
    disagreement = sum(int(np.argmax(final_dev_logits[lo:hi])) != int(np.argmax(gpu_logits[lo:hi]))
        for lo, hi in zip(known_dev['offsets'][:-1], known_dev['offsets'][1:]))
    close = bool(np.isfinite(gpu_logits).all() and np.allclose(final_dev_logits, gpu_logits,
        atol=TRAINING['cpu_gpu_atol'], rtol=TRAINING['cpu_gpu_rtol']))
    parity = dict(passed=close and disagreement == 0, groups=len(known_dev['labels']),
        candidates=len(gpu_logits), cpu_dtype='float64', gpu_dtype='float32',
        atol=TRAINING['cpu_gpu_atol'], rtol=TRAINING['cpu_gpu_rtol'],
        max_absolute_logit_error=float(np.max(np.abs(final_dev_logits-gpu_logits))),
        first_argmax_disagreements=disagreement, reloaded_cpu_logits_exact=True)
    if not np.isfinite(parity['max_absolute_logit_error']):
        raise ValueError('Nonfinite export parity result')
    for name, values in (('final-known-dev-cpu-logits.npy', final_dev_logits),
                         ('final-old-validation-cpu-logits.npy', final_old_logits),
                         ('final-known-dev-gpu-logits.npy', gpu_logits)):
        with (model/name).open('xb') as stream:
            np.save(stream, values, allow_pickle=False)
    gate = offline_gate(initial_dev, final_dev, initial_old, final_old, parity)
    gate.update(checkpoint_sha256=final_sha, initial_task_sha256=cfg['task_checkpoint_sha256'],
        metrics=dict(initial_known_dev=initial_dev, final_known_dev=final_dev,
                     initial_old_validation=initial_old, final_old_validation=final_old), export_parity=parity)
    if sha(original) != cfg['task_checkpoint_sha256'] or sha(model/'initial.npz') != cfg['task_checkpoint_sha256']:
        raise ValueError('Original model identity changed during training')
    deadline()
    offline_used = consumed.counts['cpu_offline_candidate_rows']+consumed.counts['gpu_offline_candidate_rows']
    report = dict(schema='task-anchor-once-fit-v1', completed=True, fit_count=1, optimizer_updates=512,
        checkpoint_sha256=final_sha, initial_task_sha256=cfg['task_checkpoint_sha256'],
        crop_checkpoint_sha256=cfg['crop_checkpoint_sha256'], registration_sha256=sha(out/'registration.json'),
        dagger_cache_manifest_sha256=sha(dagger_cache/'manifest.json'),
        old_cache_manifest_sha256=cfg['old_cache_manifest_sha256'],
        gpu_training=True, actual_gpu=True, gpu_platform='gpu', expected_device_verified=True,
        gpu_speedup_measured=False, official_simulation_on_gpu=False,
        nonzero_gradient_updates=sum(n > 0 for n in norms), gradient_norm_min=min(norms),
        gradient_norm_max=max(norms), loss_min=min(losses), loss_max=max(losses),
        changed_parameter_tensors=sum(not np.array_equal(initial_params[k], host_params[k]) for k in PARAMS),
        initial_parameter_sha256=initial_digest, final_parameter_sha256={k:array_digest(v) for k,v in host_params.items()},
        frozen_normalizer_array_sha256=normalization_digest, frozen_normalizer_bits_preserved=True,
        old_training_class_counts=counts.tolist(), old_training_class_weights=weights.tolist(),
        old_train_groups=len(old_train['labels']), dagger_train_groups=len(dagger_train['labels']),
        known_dev_groups=len(known_dev['labels']), old_validation_groups=len(old_validation['labels']),
        normalization_recomputed=False, class_weights_use_old_training_only=True,
        seed_roles=seed_roles, known_screens_are_independent=False,
        anchor_checkpoint_sha256=cfg['task_checkpoint_sha256'], frozen_anchor_bits_preserved=True,
        anchor_in_deployment=False, frozen_anchor_forward_evaluations=consumed.counts['anchor_forward_evaluations'],
        anchor_extra_passes=dict(real_candidate_rows=consumed.counts['anchor_real_candidate_rows'],
            padded_candidate_rows=consumed.counts['anchor_padded_candidate_rows'],
            group_presentations=consumed.counts['anchor_train_group_presentations']),
        combined_training_real_candidate_rows=consumed.counts['training_real_candidate_rows']+consumed.counts['anchor_real_candidate_rows'],
        combined_training_padded_candidate_rows=consumed.counts['training_padded_candidate_rows']+consumed.counts['anchor_padded_candidate_rows'],
        cache_roles=dict(old='historical train and old validation',
            dagger='unchanged prior student-state train and known development; no fresh observations'),
        offline_input_representation='Float32 ragged cache values scored with float64 CPU deployment arithmetic; raw live observation actions are verified separately in official matches.',
        consumption=dict(consumed.counts), independent_offline_row_reserve=TRAINING['independent_offline_row_reserve'],
        synthetic_row_reserve=TRAINING['synthetic_row_reserve'],
        reserves_are_not_actual_consumption=True,
        offline_rows_total_used=offline_used, offline_total_limit=TRAINING['offline_row_limit'],
        offline_remaining_total=TRAINING['offline_row_limit']-offline_used,
        offline_primary_row_limit=(TRAINING['offline_row_limit']
            - TRAINING['independent_offline_row_reserve']-TRAINING['synthetic_row_reserve']),
        budget_interpretation='Candidate value-and-gradient and extra frozen-S forwards are debited separately before calls, including padding; rows are presentations, not FLOPs. CPU and GPU offline calls are separate row evaluations. Independent and synthetic allowances are reserved, not this fit\'s actual work; external synthetic work is accounted separately.',
        cache_array_verifications={'old':1,'dagger':1}, old_cache_arrays=len(old_manifest['arrays']['train'])+len(old_manifest['arrays']['validation']),
        dagger_cache_arrays=len(dagger_manifest['arrays']['train'])+len(dagger_manifest['arrays']['validation']),
        timings=dict(synchronized_training_loop_seconds=train_seconds, explicit_compile_seconds=compile_seconds,
                     training_packing_seconds=packing_seconds, process_seconds=time.monotonic()-started,
                     training_loop_includes='packing, compilation, transfers, all updates synchronized, and reporting'),
        offline_gate_passed=gate['passed'], competitive_improvement_established=False,
        model_selection=False, retry_allowed=False,
        output_storage_before_final_reports=output_budget.snapshot(),
        software=dict(numpy=np.__version__, jax=jax.__version__, optax=optax.__version__))
    write_new(out/'fit-result.json', report)
    write_new(out/'offline-gate.json', gate)
    deadline()


def _worker_receipt(out, budget, consumed, started, error=None):
    from .storage import owned_writes
    # A refused write locks the large allowance. This small, separately bounded
    # receipt remains within the parent's pre-reserved bookkeeping allowance.
    with owned_writes(out, 16384,
            allowed_paths=['training-storage.json', 'training-failed.json']):
        write_new(out/'training-storage.json', dict(schema='task-anchor-fit-storage-v1',
            worker_output=budget.snapshot(), fit_completed=error is None,
            worker_output_allowance_includes_receipt=False,
            receipt_allowance_bytes=16384, receipt_accounted_by='parent bookkeeping reserve'))
        if error is not None and not (out/'training-failed.json').exists():
            write_new(out/'training-failed.json', dict(kind=type(error).__name__, fit_complete=False,
                retry_allowed=False, games_allowed=False, consumption=dict(consumed.counts),
                consumption_is_debited_work_not_guaranteed_completed=True,
                output_storage=budget.snapshot(), elapsed_seconds=time.monotonic()-started))


def _worker(out, cfg, started):
    from .storage import owned_writes
    out = Path(out)
    consumed = Consumption()
    budget = owned_writes(out, 512*1024**2, allowed_prefixes=['model'],
        allowed_paths=['model-freeze.json', 'fit-result.json', 'offline-gate.json'])
    try:
        # Optax's transitive absl import probes the OS temporary directory with
        # os.open. Initialize dependencies before the artifact-only IO adapter;
        # no experiment model, cache, tensor forward or fit is read/run here.
        # This startup is inside the same process deadline, not a filesystem
        # sandbox claim. All _fit artifact writes remain inside the allowance.
        import jax
        import optax
        with budget:
            _fit(out, cfg, consumed, started, budget)
    except BaseException as exc:
        _worker_receipt(out, budget, consumed, started, exc)
        raise
    _worker_receipt(out, budget, consumed, started)


def _launch_fit(out, cfg):
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


def fit(out):
    # Delayed imports keep pure CPU tests independent of preregistration and IO.
    from .experiment import training_ready
    from .storage import owned_writes
    out = Path(out).resolve()
    cfg = training_ready(out)
    validate_config(cfg)
    with owned_writes(out, 65536,
            allowed_paths=['training-started.json', 'training-failed.json']):
        return _launch_fit(out, cfg)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True, type=Path)
    fit(parser.parse_args().out)
