"""Two precision settings against one CPU reference, aggregate output only."""
import numpy as np

from learning.task_completion_bc_v1.training import inputs
from learning.task_completion_parity_v1.compare import prepare, summarize, validate_device


def compare(groups, model, expected_device, expected_device_kind, progress=None):
    if progress is not None and not callable(progress):
        raise ValueError('Invalid progress callback')
    cpu_started = cpu_count = cpu_rows = gpu_started = gpu_count = gpu_rows = 0
    def report(stage, setting=None):
        if progress is not None:
            progress(dict(stage=stage, setting=setting, started_cpu_groups=cpu_started,
                          completed_cpu_groups=cpu_count, completed_cpu_rows=cpu_rows,
                          started_gpu_groups=gpu_started, completed_gpu_groups=gpu_count,
                          completed_gpu_rows=gpu_rows))
    selected = prepare(groups, model)
    if sum(len(g.candidates) for g in selected) != 1741:
        raise ValueError('Unexpected candidate budget')
    report('prepared')
    import jax
    try:
        device = jax.devices('gpu')[0]
    except (RuntimeError, IndexError) as exc:
        raise RuntimeError('GPU required') from exc
    validate_device(device, expected_device, expected_device_kind)
    default_precision = jax.config.jax_default_matmul_precision
    if default_precision is not None and not isinstance(default_precision, str):
        raise ValueError('Invalid default precision setting')
    report('device_validated')
    params = {k: jax.device_put(np.asarray(v, dtype=np.float32), device) for k, v in model.params.items()}
    cpu = []
    normalized = []
    for g in selected:
        x = ((inputs(g) - model.mean) / model.scale).astype(np.float32)
        if not np.isfinite(x).all():
            raise ValueError('Invalid normalized inputs')
        cpu_started += 1
        report('cpu_started')
        scores = np.asarray(model.score(g.shared, g.candidates))
        if scores.shape != (len(g.candidates),) or not np.isfinite(scores).all():
            raise ValueError('Invalid CPU scores')
        normalized.append(x)
        cpu.append(scores)
        cpu_count += 1
        cpu_rows += len(scores)
        report('cpu_completed')
    def run(setting):
        nonlocal gpu_started, gpu_count, gpu_rows
        pairs = []
        for x, reference in zip(normalized, cpu):
            gpu_started += 1
            report('gpu_started', setting)
            y = jax.device_put(x, device)
            for i in range(3):
                y = y @ params[f'w{i}'] + params[f'b{i}']
                if i < 2:
                    y = jax.nn.relu(y)
            y.block_until_ready()
            if y.device != device:
                raise RuntimeError('Inference device validation failed')
            gpu_count += 1
            gpu_rows += len(x)
            report('gpu_completed', setting)
            scores = np.asarray(y[..., 0])
            if scores.shape != reference.shape or not np.isfinite(scores).all():
                raise ValueError('Invalid GPU scores')
            pairs.append((scores, reference))
        return summarize(pairs)
    default = run('default')
    with jax.default_matmul_precision('highest'):
        highest = run('highest')
    return dict(default_precision=default_precision, default=default, highest=highest,
                gpu_device_validated=True, completed_gpu_forwards=gpu_count,
                completed_gpu_rows=gpu_rows, completed_cpu_forwards=cpu_count,
                completed_cpu_rows=cpu_rows)
