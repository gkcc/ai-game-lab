"""Bounded, aggregate-only parity diagnostic for already loaded inputs."""
import numpy as np

from learning.task_bc_v1.model import DIMENSIONS, TaskNetwork
from learning.task_completion_bc_v1 import numeric
from learning.task_completion_bc_v1.training import inputs, as_groups

MAX_GROUPS = 16
MAX_ROWS = 8192


def summarize(pairs):
    """Compare ordered one-dimensional logits without retaining per-row results."""
    if not isinstance(pairs, (list, tuple)) or not pairs or len(pairs) > MAX_GROUPS:
        raise ValueError('Invalid comparison group count')
    rows = failures = argmax_failures = 0
    max_error = 0.0
    for gpu, cpu in pairs:
        gpu, cpu = np.asarray(gpu, dtype=np.float64), np.asarray(cpu, dtype=np.float64)
        if gpu.ndim != 1 or not len(gpu) or gpu.shape != cpu.shape:
            raise ValueError('Invalid comparison shape')
        rows += len(gpu)
        if rows > MAX_ROWS:
            raise ValueError('Comparison row budget exceeded')
        if not np.isfinite(gpu).all() or not np.isfinite(cpu).all():
            raise ValueError('Nonfinite comparison')
        diff = np.abs(gpu - cpu)
        failures += int(np.count_nonzero(diff > 1e-4 + 1e-5 * np.abs(cpu)))
        argmax_failures += int(np.argmax(gpu) != np.argmax(cpu))
        max_error = max(max_error, float(diff.max()))
    return dict(groups=len(pairs), candidate_rows=rows, finite=True,
                out_of_tolerance_candidates=failures, argmax_different_groups=argmax_failures,
                max_absolute_error=max_error, parity_passed=(failures == 0 and argmax_failures == 0))


def validate_device(device, expected_device, expected_device_kind):
    if (not isinstance(expected_device, str) or not expected_device
            or not isinstance(expected_device_kind, str) or not expected_device_kind
            or getattr(device, 'platform', None) != 'gpu'
            or str(device) != expected_device or getattr(device, 'device_kind', None) != expected_device_kind):
        raise RuntimeError('GPU identity validation failed')


def prepare(groups, model):
    if not isinstance(groups, (list, tuple)):
        raise ValueError('Invalid groups')
    selected = []
    for group in groups:
        if not isinstance(group, dict) or group.get('status') not in ('completed_observed', 'unknown'):
            raise ValueError('Invalid group status')
        if group['status'] == 'completed_observed':
            selected.append(group)
            if len(selected) == MAX_GROUPS:
                break
    if len(selected) != MAX_GROUPS:
        raise ValueError('Insufficient completed groups')
    numeric.pack(selected, MAX_GROUPS, MAX_ROWS)
    if not isinstance(model, TaskNetwork):
        raise ValueError('Invalid model')
    for name in ('mean', 'scale'):
        a = np.asarray(getattr(model, name))
        if a.shape != (373,) or not np.isfinite(a).all() or (name == 'scale' and np.any(a < 1)):
            raise ValueError('Invalid normalization')
    if set(model.params) != {f'{p}{i}' for i in range(3) for p in ('w', 'b')}:
        raise ValueError('Invalid model parameters')
    for i in range(3):
        for key, shape in ((f'w{i}', (DIMENSIONS[i], DIMENSIONS[i+1])), (f'b{i}', (DIMENSIONS[i+1],))):
            a = np.asarray(model.params[key])
            if a.shape != shape or not np.isfinite(a).all() or not np.isfinite(a.astype(np.float32)).all():
                raise ValueError('Invalid model parameters')
    return as_groups(selected)


def compare(groups, model, expected_device, expected_device_kind, progress=None):
    if progress is not None and not callable(progress):
        raise ValueError('Invalid progress callback')
    def report(stage, started, completed, rows):
        if progress is not None:
            progress(dict(stage=stage, started_groups=started, completed_groups=completed,
                          candidate_rows=rows))
    selected = prepare(groups, model)  # All group and row budgets checked before any forward.
    report('prepared', 0, 0, 0)
    import jax
    try:
        device = jax.devices('gpu')[0]
    except (RuntimeError, IndexError) as exc:
        raise RuntimeError('GPU required') from exc
    validate_device(device, expected_device, expected_device_kind)
    report('device_validated', 0, 0, 0)
    params = {k: jax.device_put(np.asarray(v, dtype=np.float32), device) for k, v in model.params.items()}
    pairs = []
    rows = 0
    for g in selected:
        normalized = ((inputs(g) - model.mean) / model.scale).astype(np.float32)
        if not np.isfinite(normalized).all():
            raise ValueError('Invalid normalized inputs')
        report('forward_started', len(pairs) + 1, len(pairs), rows)
        x = jax.device_put(normalized, device)
        for i in range(3):
            x = x @ params[f'w{i}'] + params[f'b{i}']
            if i < 2:
                x = jax.nn.relu(x)
        x.block_until_ready()
        if x.device != device:
            raise RuntimeError('Inference device validation failed')
        rows += len(g.candidates)
        report('forward_completed', len(pairs) + 1, len(pairs) + 1, rows)
        pairs.append((np.asarray(x[..., 0]), model.score(g.shared, g.candidates)))
        # Illegal numeric values stop immediately; ordinary differences do not.
        if not np.isfinite(pairs[-1][0]).all() or not np.isfinite(pairs[-1][1]).all():
            raise ValueError('Nonfinite comparison')
    result = summarize(pairs)
    result.update(gpu_device_validated=True, completed_gpu_forwards=len(pairs))
    return result
