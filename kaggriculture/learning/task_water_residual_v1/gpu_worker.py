"""One private, registered WATER residual GPU worker.

Input bytes are immutable snapshots with independently registered identities.
The worker does not play a game, access an unused holdout, or promote a model.
The development screen is a consumed-data diagnostic, not competitive proof.
The Linux/Windows owners independently supervise time, memory and cleanup.
"""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import re
import runpy
import shutil
import stat
import sys
import time
import zipfile


MAX_MODEL_BYTES = 2 * 1024**2
MAX_METADATA_BYTES = 1024**2
MAX_WEIGHT_BYTES = 64 * 1024
MIN_FREE_BYTES = 4 * 1024**3
WORK_SECONDS = 570
_HEX40 = re.compile(r'[0-9a-f]{40}\Z')
_HEX64 = re.compile(r'[0-9a-f]{64}\Z')
_NONCE = re.compile(r'[0-9a-f]{32}\Z')


def _require(value):
    if not value:
        raise ValueError('Invalid private WATER worker input')


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _json(raw):
    _require(type(raw) is bytes and 0 < len(raw) <= MAX_METADATA_BYTES)
    def pairs(items):
        result = {}
        for key, value in items:
            _require(key not in result)
            result[key] = value
        return result
    def reject(_value):
        raise ValueError('Invalid private metadata')
    value = json.loads(raw.decode('utf-8-sig'), object_pairs_hook=pairs, parse_constant=reject)
    _require(type(value) is dict)
    return value


def _bounded_read(path, limit, guard):
    """Read one regular-file snapshot through one bounded descriptor."""
    path = Path(path)
    guard()
    before = path.lstat()
    _require(stat.S_ISREG(before.st_mode) and not stat.S_ISLNK(before.st_mode)
             and not getattr(before, 'st_file_attributes', 0) & 0x400 and 0 < before.st_size <= limit)
    fd = os.open(path, os.O_RDONLY | getattr(os, 'O_BINARY', 0) | getattr(os, 'O_NOFOLLOW', 0))
    try:
        info = os.fstat(fd)
        _require(stat.S_ISREG(info.st_mode) and 0 < info.st_size <= limit)
        chunks, size = [], 0
        with os.fdopen(fd, 'rb', closefd=False) as source:
            while True:
                guard()
                part = source.read(min(1024**2, limit - size + 1))
                if not part:
                    break
                size += len(part)
                _require(size <= limit)
                chunks.append(part)
        _require(size == info.st_size)
        guard()
        return b''.join(chunks)
    finally:
        os.close(fd)


def _write_exclusive(path, raw):
    _require(type(raw) is bytes and raw)
    with Path(path).open('xb') as target:
        if target.write(raw) != len(raw):
            raise OSError('Incomplete private worker output')


def _disk_guard(stage):
    free = shutil.disk_usage(stage).free
    if type(free) is not int or free < MIN_FREE_BYTES:
        raise OSError('Private worker disk floor not met')


def _claim(stage, commit, nonce, *, guard, read_file=None, write_once=None,
           exists=None, pid=None, parent=None, session=None, parent_stat=None):
    """Accept only one session owned by the recorded live Linux supervisor."""
    _require(type(commit) is str and _HEX40.fullmatch(commit) is not None
             and type(nonce) is str and _NONCE.fullmatch(nonce) is not None and callable(guard))
    read_file = _bounded_read if read_file is None else read_file
    write_once = _write_exclusive if write_once is None else write_once
    exists = os.path.lexists if exists is None else exists
    pid = os.getpid if pid is None else pid
    parent = os.getppid if parent is None else parent
    session = os.getsid if session is None else session
    if parent_stat is None:
        from learning.task_pair_pidfd_guard_v1.guard import _stat
        parent_stat = _stat
    stage = Path(stage)
    guard()
    host = _json(read_file(stage / 'host-attempt.local.json', MAX_METADATA_BYTES, guard))
    supervisor_path = stage / 'gpu-supervisor-attempt.local.json'
    supervisor_raw = read_file(supervisor_path, MAX_METADATA_BYTES, guard)
    supervisor = _json(supervisor_raw)
    _require(host == dict(schema='water-residual-host-attempt-v1', commit=commit, retry_allowed=False)
             and set(supervisor) == {'schema', 'registration_commit', 'nonce', 'supervisor_pid',
                                    'supervisor_starttime', 'retry_allowed'}
             and supervisor['schema'] == 'water-residual-supervisor-attempt-v1'
             and supervisor['registration_commit'] == commit and supervisor['nonce'] == nonce
             and supervisor['retry_allowed'] is False
             and type(supervisor['supervisor_pid']) is int and supervisor['supervisor_pid'] > 1
             and type(supervisor['supervisor_starttime']) is int and supervisor['supervisor_starttime'] >= 0)

    def check_parent():
        current, owner = pid(), parent()
        _require(type(current) is int and current > 1 and type(owner) is int
                 and owner == supervisor['supervisor_pid'] and session(0) == current)
        info = parent_stat(owner)
        _require(info is not None and info.pid == owner and info.starttime == supervisor['supervisor_starttime']
                 and info.state not in (b'Z', b'X', b'x'))

    check_parent()
    _require(not any(exists(stage / name) for name in ('gpu-worker-attempt.local.json',
                 'residual.local.npz', 'gpu-worker-result.local.json')))
    marker = dict(schema='water-residual-worker-attempt-v1', registration_commit=commit,
                  nonce=nonce, retry_allowed=False)
    marker_raw = (json.dumps(marker, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n').encode()
    marker_path = stage / 'gpu-worker-attempt.local.json'
    write_once(marker_path, marker_raw)
    _require(read_file(marker_path, MAX_METADATA_BYTES, guard) == marker_raw
             and read_file(supervisor_path, MAX_METADATA_BYTES, guard) == supervisor_raw)
    check_parent()
    guard()
    return check_parent


def _registration_contract(registration, binding, commit):
    from learning.task_water_residual_v1 import fit
    _require(type(registration) is dict and type(binding) is dict and type(commit) is str
             and _HEX40.fullmatch(commit) is not None
             and binding.get('schema') == 'water-residual-private-binding-v1'
             and all(type(binding.get(key)) is str and _HEX64.fullmatch(binding[key]) is not None
                     for key in ('train_sha256', 'dev_sha256', 'baseline_sha256', 'schema_sha256'))
             and type(binding.get('device_kind')) is str and binding['device_kind'].strip() == binding['device_kind']
             and bool(binding['device_kind']) and type(binding.get('rng_seed')) is int
             and 0 <= binding['rng_seed'] < 2**32
             and registration.get('schema') == 'water-residual-gpu-prereg-v1'
             and registration.get('train_source_groups') == 59 and registration.get('dev_source_groups') == 62
             and registration.get('training_operations') == ['WATER']
             and registration.get('route') == 'baseline_global_water_then_water_only'
             and registration.get('training_roles_per_batch') == fit.GROUPS_PER_ROLE
             and registration.get('updates') == fit.UPDATES and registration.get('learning_rate') == fit.LEARNING_RATE
             and registration.get('kl_coefficient') == fit.KL_WEIGHT and registration.get('clip_norm') == fit.CLIP_NORM
             and registration.get('rng_seed') == binding['rng_seed']
             and registration.get('float32_training_baseline_argmax_must_match') is True
             and registration.get('cpu_gpu_atol') == 0.0001 and registration.get('cpu_gpu_rtol') == 0.00001
             and registration.get('cpu_gpu_first_argmax_must_match') is True)


def _network_matches(network, checkpoint, expected_sha):
    import numpy as np
    from learning.task_water_residual_v1 import policy
    _require(getattr(network, 'sha', None) == expected_sha and type(getattr(network, 'params', None)) is dict
             and set(network.params) == set(checkpoint.params))
    schema = json.loads(checkpoint.schema_json)
    _require(policy._schema(network.metadata) == policy._schema(schema))
    for name in checkpoint.params:
        actual = network.params[name]
        wanted = np.asarray(checkpoint.params[name], dtype=np.float64)
        _require(type(actual) is np.ndarray and actual.dtype == np.dtype('float64')
                 and actual.shape == wanted.shape and actual.tobytes() == wanted.tobytes())
    for name in ('mean', 'scale'):
        actual = getattr(network, name)
        wanted = np.asarray(getattr(checkpoint, name), dtype=np.float64)
        _require(type(actual) is np.ndarray and actual.dtype == np.dtype('float64')
                 and actual.shape == wanted.shape and actual.tobytes() == wanted.tobytes())
    policy._normalization(network.mean, network.scale, policy._schema(schema))
    return schema


def _global_cast_differences(groups, network, metadata, guard):
    """Count full-roster f64/f32 first-argmax changes on selected train groups."""
    import numpy as np
    from learning.task_water_residual_v1 import assessment
    count, changed = 0, 0
    for group in groups:
        guard()
        _raw, water, scores = assessment._scored(group, network, metadata)
        if group.op != 'WATER' or not water[int(np.argmax(scores))] or np.count_nonzero(water) < 2:
            continue
        with np.errstate(over='ignore', invalid='ignore'):
            reduced = scores.astype(np.float32)
        _require(np.isfinite(reduced).all())
        count += 1
        changed += int(int(np.argmax(scores)) != int(np.argmax(reduced)))
    return count, changed


class JaxBackend:
    """Actual JAX backend; no CPU fallback and no device names in results."""
    def __init__(self):
        import jax
        import jax.numpy as jnp
        self.jax, self.jnp = jax, jnp

    def device(self, expected_kind):
        devices = self.jax.devices('gpu')
        _require(len(devices) == 1 and devices[0].platform == 'gpu'
                 and devices[0].device_kind == expected_kind)
        return devices[0]

    def validate(self, device, expected_kind):
        return (isinstance(device, self.jax.Device) and device == self.device(expected_kind))

    def train(self, groups, *, mean, scale, device, guard, rng_seed):
        from learning.task_water_residual_v1 import fit
        return fit.fit(groups, mean=mean, scale=scale, device=device, guard=guard, rng_seed=rng_seed)

    def residual(self, normalized, weights, *, device, guard):
        import numpy as np
        guard()
        # Keep every development transfer and primitive at one fixed shape so
        # distinct roster lengths cannot cause repeated GPU compilations.
        padded, count = _pad_normalized(normalized)
        with self.jax.default_device(device), self.jax.default_matmul_precision('highest'):
            x, w = self.jax.device_put(padded, device), self.jax.device_put(weights, device)
            # Match fit._components: subtract the first candidate before the
            # elementwise multiplication and reduction. The deployment policy
            # uses an uncentred dot product, so compare relative scores below.
            value = self.jnp.sum((x[None, :, :] - x[None, :1, :]) * w[None, None, :], axis=2)[0]
            for leaf in (x, w, value):
                _require(isinstance(leaf, self.jax.Array) and leaf.devices() == {device})
                leaf.block_until_ready()
            full = np.asarray(self.jax.device_get(value))
        guard()
        _require(full.dtype == np.dtype('float32') and full.shape == (256,) and np.isfinite(full).all())
        return full[:count].copy()


def _pad_normalized(normalized):
    import numpy as np
    _require(type(normalized) is np.ndarray and normalized.dtype == np.dtype('float32')
             and normalized.ndim == 2 and normalized.shape[1] == 373
             and 1 <= len(normalized) <= 256 and np.isfinite(normalized).all())
    # Repeat the reference, as fit masks padding to the reference before
    # subtraction. This avoids irrelevant extreme padded residuals.
    result = np.broadcast_to(normalized[:1], (256, 373)).copy()
    result[:len(normalized)] = normalized
    return result, len(normalized)


def _development_parity(groups, network, weights, metadata, rows, *, backend, device, guard, atol, rtol):
    import numpy as np
    from learning.task_choice_v1.features import OPS
    from learning.task_water_residual_v1 import assessment, policy
    schema = policy._schema(metadata)
    mean, scale = policy._normalization(network.mean, network.scale, schema)
    counts = {op: 0 for op in OPS}
    groups_with_water = candidates = flips = residual_disagree = adjusted_disagree = 0
    max_error = 0.0
    _require(len(groups) == len(rows))
    for group, row in zip(groups, rows):
        guard()
        raw, water, scores = assessment._scored(group, network, metadata)
        base = int(np.argmax(scores))
        selected = policy.select(raw, scores, mean, scale, weights, metadata)
        ops = policy._operations(raw, schema)
        op = OPS[int(np.argmax(ops[base]))]
        counts[op] += 1
        changed_op = bool(not np.array_equal(ops[base], ops[selected]))
        flips += int(changed_op)
        _require(row['changed'] == (selected != base) and row['operation_flip'] == changed_op)
        indices = np.flatnonzero(water)
        if not len(indices):
            _require(selected == base)
            continue
        groups_with_water += 1
        candidates += len(indices)
        with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
            normalized = ((raw[indices].astype(np.float64) - mean) / scale).astype(np.float32)
            cpu = normalized @ weights
        _require(np.isfinite(normalized).all() and cpu.dtype == np.dtype('float32') and np.isfinite(cpu).all())
        gpu = backend.residual(normalized, weights, device=device, guard=guard)
        _require(type(gpu) is np.ndarray and gpu.dtype == np.dtype('float32')
                 and gpu.shape == cpu.shape and np.isfinite(gpu).all())
        # A common offset cannot affect ranking and differs by design between
        # centered training and the uncentered deployed model.
        cpu_relative = cpu.astype(np.float64) - np.float64(cpu[0])
        gpu_relative = gpu.astype(np.float64) - np.float64(gpu[0])
        max_error = max(max_error, float(np.max(np.abs(cpu_relative - gpu_relative))))
        _require(np.allclose(cpu_relative, gpu_relative, atol=atol, rtol=rtol))
        residual_disagree += int(int(np.argmax(cpu_relative)) != int(np.argmax(gpu_relative)))
        cpu_adjusted = scores[indices] + cpu.astype(np.float64)
        gpu_adjusted = scores[indices] + gpu_relative
        _require(np.isfinite(cpu_adjusted).all() and np.isfinite(gpu_adjusted).all())
        adjusted_disagree += int(int(np.argmax(cpu_adjusted)) != int(np.argmax(gpu_adjusted)))
        gpu_selected = int(indices[int(np.argmax(gpu_adjusted))]) if water[base] else base
        _require(gpu_selected == selected)
    _require(groups_with_water > 0 and candidates > 0 and flips == 0
             and residual_disagree == 0 and adjusted_disagree == 0)
    return dict(groups=len(groups), water_groups=groups_with_water, water_candidates=candidates,
                max_abs_error=max_error, atol=atol, rtol=rtol, allclose=True,
                residual_argmax_mismatches=residual_disagree, adjusted_argmax_mismatches=adjusted_disagree,
                baseline_operation_counts=counts, operation_flips=flips)


def execute(*, train_raw, dev_raw, baseline_raw, registration, binding, registration_commit,
            guard, device, network_loader, backend):
    """Train/evaluate private byte snapshots; all runtime choices are explicit.

    ``network_loader`` must reopen the authenticated private baseline copy as a
    TaskNetwork in production. Injectable collaborators permit synthetic tests
    only; they cannot themselves prove a GPU update or input provenance.
    """
    import numpy as np
    from learning.task_pair_checkpoint_v1 import snapshot
    from learning.task_water_residual_v1 import assessment, fit, snapshot_data
    _require(callable(guard) and callable(network_loader))
    guard()
    _registration_contract(registration, binding, registration_commit)
    train = snapshot_data.read_split(train_raw, binding['train_sha256'], 'train', guard=guard)
    dev = snapshot_data.read_split(dev_raw, binding['dev_sha256'], 'dev', guard=guard)
    _require((len(train), len(dev)) == (59, 62))
    checkpoint = snapshot.load_checkpoint(baseline_raw, expected_sha256=binding['baseline_sha256'],
                                          expected_schema_sha256=binding['schema_sha256'])
    guard()
    network = network_loader(baseline_raw, binding['baseline_sha256'])
    metadata = _network_matches(network, checkpoint, binding['baseline_sha256'])
    from decision.api import PRODUCTS
    from learning.task_choice_v1.features import OPS
    start = 5 + len(PRODUCTS)
    _require(metadata['feature_names'][start:start + len(OPS)] == ['task.op.' + op for op in OPS])
    selected, report = assessment.prepare_training(train, network, metadata, guard=guard)
    count, global_differences = _global_cast_differences(train, network, metadata, guard)
    water_differences = report['float32_baseline_argmax_differences']
    _require(count == len(selected) == report['selected_groups']
             and all(report['selected_by_role'][role] >= fit.GROUPS_PER_ROLE for role in assessment.TRAIN_ROLES)
             and type(global_differences) is int and type(water_differences) is int
             and global_differences == 0 and water_differences == 0)
    guard()
    _require(backend.validate(device, binding['device_kind']) is True)
    trained = backend.train(selected, mean=network.mean, scale=network.scale,
                            device=device, guard=guard, rng_seed=binding['rng_seed'])
    weights = trained.weights
    _require(type(weights) is np.ndarray and weights.dtype == np.dtype('float32')
             and weights.shape == (fit.DIM,) and np.isfinite(weights).all()
             and type(trained.updates_completed) is int and trained.updates_completed == fit.UPDATES
             and trained.nonzero_gradient_seen is True and trained.weights_changed is True
             and np.any(weights != 0))
    guard()
    development, rows = assessment.assess_development(dev, network, weights, metadata, guard=guard)
    _require(type(development) is dict and development.get('screen') in ('passed', 'failed', 'unknown')
             and development.get('groups') == 62 and development.get('operation_flips') == 0)
    parity = _development_parity(dev, network, weights, metadata, rows, backend=backend, device=device,
                                 guard=guard, atol=registration['cpu_gpu_atol'], rtol=registration['cpu_gpu_rtol'])
    summary = dict(schema='water-residual-gpu-worker-result-v1', registration_commit=registration_commit,
        training=dict(input_groups=len(train), selected_groups=len(selected), selected_by_role=report['selected_by_role'],
                      float32_global_argmax_differences=global_differences,
                      float32_water_argmax_differences=water_differences, device_verified=True,
                      updates_completed=trained.updates_completed, nonzero_gradient_seen=True, weights_changed=True),
        development=development, parity=parity, official_matches=0, independent_competitive_evaluation=False)
    # Verify serializability without emitting any private arrays or device identity.
    json.dumps(summary, allow_nan=False)
    guard()
    return np.frombuffer(weights.tobytes(order='C'), dtype=np.float32), summary


def _encode_weights(weights):
    import numpy as np
    _require(type(weights) is np.ndarray and weights.dtype == np.dtype('float32')
             and weights.shape == (373,) and np.isfinite(weights).all())
    output = io.BytesIO()
    np.savez_compressed(output, weights=weights)
    raw = output.getvalue()
    _require(0 < len(raw) <= MAX_WEIGHT_BYTES)
    return raw


def _decode_weights(raw):
    import numpy as np
    _require(type(raw) is bytes and 0 < len(raw) <= MAX_WEIGHT_BYTES)
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        infos = archive.infolist()
        _require(len(infos) == 1 and infos[0].filename == 'weights.npy'
                 and 0 < infos[0].file_size <= MAX_WEIGHT_BYTES)
        with archive.open(infos[0]) as source:
            version = np.lib.format.read_magic(source)
            _require(version in ((1, 0), (2, 0)))
            reader = np.lib.format.read_array_header_1_0 if version == (1, 0) else np.lib.format.read_array_header_2_0
            shape, fortran, dtype = reader(source)
            _require(shape == (373,) and fortran is False and dtype == np.dtype('float32')
                     and infos[0].file_size == source.tell() + 373 * dtype.itemsize)
    with np.load(io.BytesIO(raw), allow_pickle=False) as archive:
        _require(archive.files == ['weights'])
        result = archive['weights']
    _require(result.shape == (373,) and result.dtype == np.dtype('float32') and np.isfinite(result).all())
    return np.frombuffer(result.tobytes(order='C'), dtype=np.float32)


def save_outputs(stage, weights, summary, baseline_raw, *, guard, disk_guard=_disk_guard,
                 read_file=_bounded_read, write_once=_write_exclusive):
    """Exclusively save and re-read the candidate before the final receipt."""
    import numpy as np
    stage = Path(stage)
    guard()
    disk_guard(stage)
    raw = _encode_weights(weights)
    weight_path = stage / 'residual.local.npz'
    write_once(weight_path, raw)
    reread = read_file(weight_path, MAX_WEIGHT_BYTES, guard)
    _require(reread == raw and np.array_equal(_decode_weights(reread), weights))
    _require(read_file(stage / 'baseline.npz', MAX_MODEL_BYTES, guard) == baseline_raw)
    guard()
    disk_guard(stage)
    result = dict(summary, residual_sha256=_sha(raw), baseline_unchanged=True)
    encoded = (json.dumps(result, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n').encode()
    _require(len(encoded) <= MAX_METADATA_BYTES)
    write_once(stage / 'gpu-worker-result.local.json', encoded)
    guard()
    return result


def _guest_available():
    with open('/proc/meminfo', encoding='ascii') as source:
        line = next((item for item in source if item.startswith('MemAvailable:')), None)
    _require(line is not None and len(line.split()) >= 2 and line.split()[1].isdigit())
    return int(line.split()[1]) * 1024


def _verify_unchanged(verify, commit, stage, registration, binding, guard):
    guard()
    current_registration, current_binding = verify(commit, stage)
    _require(current_registration == registration and current_binding == binding)
    guard()


class _PrivateArgumentParser(argparse.ArgumentParser):
    def error(self, _message):
        # argparse normally echoes the full command line, including private paths.
        raise ValueError('Invalid private worker arguments')


def main(argv=None):
    started = time.monotonic()
    parser = _PrivateArgumentParser(add_help=False)
    parser.add_argument('--stage', required=True)
    parser.add_argument('--registration-commit', required=True)
    parser.add_argument('--nonce', required=True)
    args = parser.parse_args(argv)
    _require(os.name == 'posix' and sys.flags.isolated and sys.flags.dont_write_bytecode and __debug__)
    stage = Path(args.stage)
    _require(stage.is_absolute() and stage.resolve(strict=True) == stage and stage.is_dir())

    owner_check = [None]
    def guard():
        if time.monotonic() - started >= WORK_SECONDS or os.path.lexists(stage / 'gpu-cancel.local.json'):
            raise TimeoutError('Private worker deadline or cancellation')
        if _guest_available() < 2 * 1024**3:
            raise MemoryError('Private worker guest memory floor')
        if owner_check[0] is not None:
            owner_check[0]()

    guard()
    _disk_guard(stage)
    # Set before importing numpy/JAX through the registration/bootstrap graph.
    os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
    for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[key] = '1'
    root = Path(__file__).resolve().parents[2]
    bootstrap = runpy.run_path(str(root / 'learning' / 'task_equiv_bc_v1' / 'gpu_bootstrap.py'))
    imports, checked_root = bootstrap['bind']()
    _require(checked_root == root)
    from learning.task_water_residual_v1.experiment import verify_registration
    registration, binding = verify_registration(args.registration_commit, stage)
    owner_check[0] = _claim(stage, args.registration_commit, args.nonce, guard=guard)
    from learning.task_teacher_transfer_dataset_v1.experiment import FINAL_BYTES
    train_raw = _bounded_read(stage / 'train.npz', FINAL_BYTES, guard)
    dev_raw = _bounded_read(stage / 'dev.npz', FINAL_BYTES, guard)
    baseline_raw = _bounded_read(stage / 'baseline.npz', MAX_MODEL_BYTES, guard)
    from learning.task_bc_v1.model import TaskNetwork
    backend = JaxBackend()
    device = backend.device(binding['device_kind'])
    weights, summary = execute(train_raw=train_raw, dev_raw=dev_raw, baseline_raw=baseline_raw,
        registration=registration, binding=binding, registration_commit=args.registration_commit,
        guard=guard, device=device, backend=backend,
        network_loader=lambda _raw, digest: TaskNetwork(stage / 'baseline.npz', digest))
    imports.verify_origins(root)
    _verify_unchanged(verify_registration, args.registration_commit, stage, registration, binding, guard)
    save_outputs(stage, weights, summary, baseline_raw, guard=guard)
    _verify_unchanged(verify_registration, args.registration_commit, stage, registration, binding, guard)
    guard()
    print(json.dumps(dict(phase='gpu_worker', status='passed')), flush=True)
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except BaseException as exc:
        category = ('memory' if isinstance(exc, MemoryError) else 'timeout' if isinstance(exc, TimeoutError)
                    else 'io' if isinstance(exc, OSError) else 'invalid_or_runtime')
        print(json.dumps(dict(phase='gpu_worker', status='failed', category=category)), file=sys.stderr, flush=True)
        raise SystemExit(1) from None
