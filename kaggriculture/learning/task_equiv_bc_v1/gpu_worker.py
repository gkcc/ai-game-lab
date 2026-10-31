"""Private bounded GPU child; parent authenticates source and input provenance."""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import runpy
import sys
import time
import zipfile


def require(value):
    if not value:
        raise ValueError('GPU input contract failed')


def read_bounded(path, limit):
    require(path.is_file() and not path.is_symlink() and 0 < path.stat().st_size <= limit)
    with path.open('rb') as stream:
        raw = stream.read(limit + 1)
    require(0 < len(raw) <= limit)
    return raw


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def write_new(path, raw):
    require(type(raw) is bytes and raw)
    with path.open('xb') as stream:
        require(stream.write(raw) == len(raw))


def validate_contract(contract, commit):
    require(type(contract) is dict
            and set(contract) == {'schema', 'registration_commit', 'groups_sha256',
                                  'checkpoint_sha256', 'schema_sha256', 'device_kind'}
            and contract['schema'] == 'task-equiv-bc-gpu-contract-v1'
            and contract['registration_commit'] == commit
            and type(contract['device_kind']) is str
            and 1 <= len(contract['device_kind']) <= 128
            and contract['device_kind'] == contract['device_kind'].strip()
            and all(32 <= ord(char) != 127 for char in contract['device_kind']))


def _archive_headers(raw):
    """Check ZIP/NPY budgets and array shapes before NumPy allocates arrays."""
    import numpy as np
    names = {'features', 'offsets', 'teacher', 'positive', 'verified', 'roles'}
    shapes, types = {}, {}
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        infos = archive.infolist()
        require(len(infos) == len(names) and {item.filename for item in infos} == {n + '.npy' for n in names})
        require(sum(item.file_size for item in infos) <= 24 * 1024**2
                and all(item.compress_type in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED)
                        and 0 < item.file_size <= 24 * 1024**2 for item in infos))
        for item in infos:
            name = item.filename[:-4]
            with archive.open(item) as stream:
                version = np.lib.format.read_magic(stream)
                if version == (1, 0):
                    shape, fortran, dtype = np.lib.format.read_array_header_1_0(stream)
                elif version == (2, 0):
                    shape, fortran, dtype = np.lib.format.read_array_header_2_0(stream)
                else:
                    raise ValueError('Unsupported NPY header')
            require(not fortran and not dtype.hasobject)
            shapes[name], types[name] = shape, dtype
    shape = shapes['features']
    require(len(shape) == 2 and 96 <= shape[0] <= 48 * 256 and shape[1] == 373
            and types['features'] == np.dtype('float32'))
    require(shapes['offsets'] == (49,) and types['offsets'] == np.dtype('int32')
            and shapes['teacher'] == shapes['roles'] == (48,)
            and types['teacher'] == types['roles'] == np.dtype('int32')
            and shapes['positive'] == shapes['verified'] == (shape[0],)
            and types['positive'] == types['verified'] == np.dtype('bool'))


def unpack_groups(raw):
    """Decode a bounded, non-pickle training-only matrix with two seed roles."""
    import numpy as np
    _archive_headers(raw)
    with np.load(io.BytesIO(raw), allow_pickle=False) as data:
        require(set(data.files) == {'features', 'offsets', 'teacher', 'positive', 'verified', 'roles'})
        arrays = {name: np.array(data[name], copy=True) for name in data.files}
    x, offsets, teacher = (arrays[k] for k in ('features', 'offsets', 'teacher'))
    positive, verified, roles = (arrays[k] for k in ('positive', 'verified', 'roles'))
    require(x.dtype == np.float32 and x.ndim == 2 and x.shape[1] == 373 and np.isfinite(x).all())
    require(offsets.dtype == np.int32 and offsets.shape == (49,) and offsets[0] == 0
            and offsets[-1] == len(x) and np.all((np.diff(offsets) >= 2) & (np.diff(offsets) <= 256)))
    require(teacher.dtype == np.int32 and teacher.shape == (48,)
            and roles.dtype == np.int32 and roles.shape == (48,)
            and np.count_nonzero(roles == 0) == 24 and np.count_nonzero(roles == 1) == 24)
    require(positive.dtype == np.bool_ and verified.dtype == np.bool_
            and positive.shape == verified.shape == (len(x),))
    groups, eligible = [], [0, 0]
    for i in range(48):
        lo, hi = int(offsets[i]), int(offsets[i + 1])
        label = int(teacher[i])
        p, u = positive[lo:hi], verified[lo:hi]
        require(0 <= label < hi - lo and p[label] and not np.any(p & ~u))
        if np.count_nonzero(p) >= 2 and np.any(u & ~p):
            eligible[int(roles[i])] += 1
        groups.append({'features': x[lo:hi], 'teacher_index': label,
                       'positive_mask': p, 'verified_mask': u, 'weight': 1.0})
    require(sum(eligible) >= 4 and min(eligible) >= 1)
    return groups, eligible


def main(argv=None):
    started, phase = time.monotonic(), 'arguments'
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument('--stage', required=True)
        parser.add_argument('--registration-commit', required=True)
        args = parser.parse_args(argv)
        require(os.name == 'posix' and sys.flags.isolated and sys.flags.dont_write_bytecode)
        bootstrap = runpy.run_path(str(Path(__file__).with_name('gpu_bootstrap.py')))
        imports, project_root = bootstrap['bind']()
        stage = Path(args.stage)
        require(stage.is_absolute() and stage.resolve(strict=True) == stage
                and stage.name == 'task-equivalent-bc-v3-20261031' and stage.parent.name == 'artifacts')
        cancel = stage / 'gpu-cancel.local.json'

        def guard():
            if cancel.exists() or time.monotonic() - started >= 880:
                raise TimeoutError('GPU stage cancelled or timed out')
            with open('/proc/meminfo', encoding='ascii') as stream:
                available = next((line for line in stream if line.startswith('MemAvailable:')), None)
            require(available is not None and available.split()[1].isdigit())
            if int(available.split()[1]) * 1024 < 4 * 1024**3:
                raise MemoryError('Guest memory floor')
            return True

        phase = 'binding'
        guard()
        contract = json.loads(read_bounded(stage / 'gpu-contract.local.json', 8192))
        validate_contract(contract, args.registration_commit)
        groups_raw = read_bounded(stage / 'training-groups.local.npz', 32 * 1024**2)
        guard()
        model_raw = read_bounded(stage / 'initial.local.npz', 8 * 1024**2)
        require(sha(groups_raw) == contract['groups_sha256']
                and sha(model_raw) == contract['checkpoint_sha256'])
        groups, eligible = unpack_groups(groups_raw)
        del groups_raw
        guard()
        from learning.task_pair_checkpoint_v1.snapshot import load_checkpoint
        from learning.task_equiv_bc_v1.train import fit_paired, export_pair
        import jax
        import numpy as np
        import optax
        imports.verify_origins(project_root)
        require(sys.version_info[:3] == (3, 12, 3) and jax.__version__ == '0.4.38'
                and optax.__version__ == '0.2.4' and np.__version__ == '2.5.4')
        guard()

        checkpoint = load_checkpoint(model_raw, expected_sha256=contract['checkpoint_sha256'],
                                     expected_schema_sha256=contract['schema_sha256'])
        del model_raw
        guard()
        phase = 'device'
        require(os.environ.get('XLA_PYTHON_CLIENT_PREALLOCATE') == 'false')
        device = jax.devices('gpu')[0]
        require(device.platform == 'gpu' and contract['device_kind'] == device.device_kind)
        guard()
        phase = 'training'
        batches = np.tile(np.arange(48, dtype=np.int32), (32, 1))
        result = fit_paired(groups, checkpoint=checkpoint, batches=batches, device=device, guard=guard)
        imports.verify_origins(project_root)
        require(result.completed_updates == 32 and result.control_changed
                and result.experiment_changed and result.arms_differ
                and result.records[0]['gradient_gap_at_current_arm_parameters'] > 1e-8
                and all(item['eligible_groups'] == sum(eligible) for item in result.records))
        control, experiment = export_pair(result, checkpoint=checkpoint,
            registration_commit=args.registration_commit, source_kind='validated_counterfactual_requests',
            expected_device_kind=contract['device_kind'])
        guard()
        phase = 'output'
        write_new(stage / 'control.local.npz', control)
        write_new(stage / 'experiment.local.npz', experiment)
        output = {'schema': 'task-equiv-bc-gpu-result-v1', 'status': 'passed',
                  'device_platform': device.platform, 'device_kind': device.device_kind,
                  'python_version': '3.12.3', 'jax_version': jax.__version__,
                  'optax_version': optax.__version__, 'numpy_version': np.__version__,
                  'updates_per_arm': result.completed_updates, 'eligible_training_groups': sum(eligible),
                  'eligible_per_training_role': eligible, 'both_arms_changed': True,
                  'arms_differ': True, 'records': list(result.records),
                  'control_sha256': sha(control), 'experiment_sha256': sha(experiment)}
        write_new(stage / 'gpu-result.local.json', (json.dumps(output, allow_nan=False) + '\n').encode())
        guard()
        print(json.dumps({'phase': 'gpu_complete', 'status': 'passed', 'updates_per_arm': 32}), flush=True)
        return 0
    except BaseException as exc:
        category = ('timeout' if isinstance(exc, TimeoutError) else
                    'memory' if isinstance(exc, MemoryError) else 'invalid_or_runtime')
        print(json.dumps({'phase': phase, 'status': 'failed', 'category': category}), file=sys.stderr, flush=True)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
