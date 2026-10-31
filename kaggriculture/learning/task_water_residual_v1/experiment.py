"""Register and supervise one private WATER-residual GPU feasibility probe.

Historical split identities and seed roles remain private. No original model,
default bot, prior data or unfinished holdout is modified by this module.
"""
import argparse
import hashlib
import io
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import secrets
import shutil
import subprocess
import sys
import time
import zipfile

from learning.task_student_continuation_v1 import experiment as previous
from learning.task_teacher_transfer_dataset_v1 import experiment as transfer


REPO = previous.REPO
GAME = REPO.parent.parent / 'kaggriculture'
STAGE_NAME = 'task-water-residual-v1-20261031'
STAGE = GAME / 'artifacts' / STAGE_NAME
REGISTRATION = REPO / 'review/evidence/task-water-residual-v1-20261031/registration.json'
BASE_COMMIT = 'ca25725f7db9182ab5fe77b7eea708f172aeb64c'
DATA_COMMIT = '7a2389ea77822a9bf4f11d3f9dc244765d270041'
RNG_SEED = 314159265
MAX_MODEL_BYTES = 2 * 1024**2
MAX_METADATA = 1024**2
MEMORY_INITIAL = 5 * 1024**3
MEMORY_RUNNING = 4 * 1024**3
DISK_FLOOR = 4 * 1024**3
SOURCE_FILES = tuple('kaggriculture/learning/' + name for name in (
    'task_water_residual_v1/__init__.py', 'task_water_residual_v1/policy.py',
    'task_water_residual_v1/fit.py', 'task_water_residual_v1/assessment.py',
    'task_water_residual_v1/snapshot_data.py', 'task_water_residual_v1/experiment.py',
    'task_water_residual_v1/gpu_supervisor.py', 'task_water_residual_v1/gpu_worker.py',
    'task_equiv_bc_v1/host_gpu.py', 'task_equiv_bc_v1/gpu_bootstrap.py',
    'task_equiv_bc_v1/gpu_supervisor.py', 'task_pair_gpu_smoke_v1/import_guard.py',
    'task_pair_pidfd_guard_v1/guard.py', 'task_pair_pidfd_guard_v1/gate.py',
    'task_pair_checkpoint_v1/snapshot.py', 'task_bc_v1/model.py',
    'task_choice_v1/features.py', 'bc_crop/features.py', 'panel_eval_v3/bootstrap.py',
    'task_student_collection_v1/dataset.py', 'task_student_continuation_v1/experiment.py',
    'task_student_anchor_collection_v1/sources.py', 'task_student_dagger_collection_v1/sources.py',
    'task_multi_teacher_probe_v1/experiment.py', 'task_teacher_support_probe_v1/experiment.py',
    'task_pair_dev_games_v1/panel.py', 'task_pair_wsl_launch_v1/host.py',
    'task_teacher_transfer_dataset_v1/experiment.py', 'task_teacher_transfer_dataset_v1/dataset.py')) + (
        'kaggriculture/decision/api.py',)


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _read(path, limit):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError('Invalid private or source file')
    with path.open('rb') as stream:
        raw = stream.read(limit + 1)
    if not raw or len(raw) > limit:
        raise ValueError('Private or source file exceeds its bound')
    return raw


def _json(raw):
    return previous._document(raw)


def _write(path, raw):
    with Path(path).open('xb') as stream:
        if stream.write(raw) != len(raw):
            raise OSError('Incomplete private snapshot write')


def _git(*args):
    return subprocess.check_output(['git', '-C', str(REPO), *args],
                                   stderr=subprocess.DEVNULL, timeout=30).decode('utf-8').strip()


def _windows():
    return os.name == 'nt'


def _source_hashes():
    return {relative: _sha(_read(REPO / relative, MAX_METADATA)) for relative in SOURCE_FILES}


def _registration(source_hashes, binding_sha):
    return dict(schema='water-residual-gpu-prereg-v1', base_commit=BASE_COMMIT,
        data_registration_commit=DATA_COMMIT, source_sha256=source_hashes,
        private_binding_sha256=binding_sha, train_roles=['A01', 'A02'], dev_roles=['A07', 'A08'],
        train_source_groups=59, dev_source_groups=62, training_operations=['WATER'],
        worker_baseline_group_scores_max=242, host_recheck_group_scores_max=62,
        route='baseline_global_water_then_water_only', training_roles_per_batch=8,
        updates=64, learning_rate=0.0001, kl_coefficient=0.1, clip_norm=1.0, rng_seed=RNG_SEED,
        float32_training_baseline_argmax_must_match=True, cpu_gpu_atol=0.0001,
        cpu_gpu_rtol=0.00001, cpu_gpu_first_argmax_must_match=True,
        initial_available_bytes=MEMORY_INITIAL, running_available_bytes=MEMORY_RUNNING,
        linux_available_bytes=2 * 1024**3, linux_worker_seconds=570,
        linux_supervision_seconds=600, windows_supervision_seconds=900,
        disk_free_floor_bytes=DISK_FLOOR, attempts=1, new_complete_games=0,
        independent_game_evaluation=False, promotion=False,
        dev_screen=dict(each_role_eligible=True, each_role_no_accuracy_drop=True,
                        min_paired_net_improvements=2, equal_role_nll_increase_max=0.0000001,
                        at_least_one_change=True, operation_flips=0))


def _older_wsl_contract():
    path = GAME / 'artifacts' / 'task-pair-wsl-contract-v1-20261030'
    host = _json(_read(path / 'host-contract.local.json', MAX_METADATA))
    training = _json(_read(path / 'training-contract.local.json', MAX_METADATA))
    names = ('distribution', 'wsl_venv_executable', 'wsl_game_root', 'wsl_repo_root')
    if (any(type(host.get(key)) is not str or not host[key] or '\x00' in host[key] for key in names)
            or any(not PurePosixPath(host[key]).is_absolute() for key in names[1:])
            or type(training.get('target_device_kind')) is not str or not training['target_device_kind']):
        raise ValueError('Existing private WSL identity is unavailable')
    if os.name == 'nt':
        for key, target in (('wsl_game_root', GAME), ('wsl_repo_root', REPO)):
            parts = PurePosixPath(host[key]).parts
            if len(parts) < 4 or parts[1] != 'mnt' or len(parts[2]) != 1 or not parts[2].isalpha():
                raise ValueError('WSL path is not a mapped Windows drive')
            mapped = Path(parts[2].upper() + ':/').joinpath(*parts[3:])
            if mapped.resolve(strict=True) != target.resolve(strict=True):
                raise ValueError('WSL path maps to a different project')
    return {key: host[key] for key in names}, training['target_device_kind']


def prepare():
    """Create private source snapshots and public preregistration, without inference."""
    expected_change = {'kaggriculture/learning/task_water_residual_v1/assessment.py'}
    changed = set(filter(None, _git('diff', '--name-only').splitlines()))
    if (not _windows() or _git('rev-parse', 'HEAD') != BASE_COMMIT
            or changed != expected_change or _git('diff', '--cached', '--name-only')
            or REGISTRATION.exists() or STAGE.exists()):
        raise ValueError('Unexpected residual preparation state')
    if previous.physical_available() < MEMORY_INITIAL:
        raise MemoryError('Initial physical-memory floor not met')
    if shutil.disk_usage(GAME / 'artifacts').free < DISK_FLOOR + 2 * transfer.FINAL_BYTES + MAX_MODEL_BYTES:
        raise OSError('Insufficient space for private residual snapshots')
    deadline = time.monotonic() + 300
    guard = lambda: previous._guard(deadline)
    train, dev, contract = transfer.load_trainable(guard=guard)
    if (len(train), len(dev), contract['registration_commit']) != (59, 62, DATA_COMMIT):
        raise ValueError('Completed transfer dataset differs')
    from learning.task_student_anchor_collection_v1 import sources
    from learning.task_pair_checkpoint_v1 import snapshot
    from . import policy
    import numpy as np
    ctx = sources.open_context(guard=guard)
    mean, scale, metadata = ctx.model_normalization()
    policy._schema(metadata)
    policy._normalization(mean, scale, policy._schema(metadata))
    baseline = _read(ctx._task_path, MAX_MODEL_BYTES)
    snap = snapshot.load_checkpoint(baseline, expected_sha256=ctx.task_sha256,
                                    expected_schema_sha256=metadata['schema_sha256'])
    if not np.array_equal(snap.mean, mean) or not np.array_equal(snap.scale, scale):
        raise ValueError('Baseline normalization differs')
    raw_train = _read(transfer.PRIVATE / 'train.local.npz', transfer.FINAL_BYTES)
    raw_dev = _read(transfer.PRIVATE / 'dev.local.npz', transfer.FINAL_BYTES)
    if (_sha(raw_train) != contract['train_sha256'] or _sha(raw_dev) != contract['dev_sha256']):
        raise ValueError('Transfer split snapshots differ')
    wsl, device_kind = _older_wsl_contract()
    binding = dict(schema='water-residual-private-binding-v1', salt=secrets.token_hex(32),
        baseline_sha256=ctx.task_sha256, train_sha256=contract['train_sha256'],
        dev_sha256=contract['dev_sha256'], schema_sha256=metadata['schema_sha256'],
        data_registration_commit=DATA_COMMIT, device_kind=device_kind, rng_seed=RNG_SEED, wsl=wsl)
    binding_raw = previous._encoded(binding)
    registration = _registration(_source_hashes(), _sha(binding_raw))
    STAGE.mkdir(parents=False, exist_ok=False)
    for name, raw in (('binding.local.json', binding_raw), ('baseline.npz', baseline),
                      ('train.npz', raw_train), ('dev.npz', raw_dev)):
        guard()
        _write(STAGE / name, raw)
    REGISTRATION.parent.mkdir(parents=True, exist_ok=True)
    _write(REGISTRATION, previous._encoded(registration))
    print('Private residual snapshots and public registration prepared.')


def verify_registration(commit, stage):
    """Bind current committed sources and a private stage on Windows or WSL."""
    if (type(commit) is not str or re.fullmatch(r'[0-9a-f]{40}', commit) is None
            or _git('rev-parse', 'HEAD') != commit or _git('rev-parse', 'HEAD^') != BASE_COMMIT
            or _git('status', '--porcelain')):
        raise ValueError('Residual registration or workspace differs')
    stage = Path(stage)
    if (stage != STAGE or stage.is_symlink() or not stage.is_dir()
            or stage.resolve(strict=True) != stage):
        raise ValueError('Residual private stage differs')
    raw = _read(REGISTRATION, MAX_METADATA)
    committed = subprocess.check_output(['git', '-C', str(REPO), 'show',
        commit + ':' + REGISTRATION.relative_to(REPO).as_posix()], stderr=subprocess.DEVNULL, timeout=30)
    reg = _json(raw)
    binding_raw = _read(stage / 'binding.local.json', MAX_METADATA)
    if (raw != committed or reg != _registration(_source_hashes(), _sha(binding_raw))):
        raise ValueError('Residual sources or binding differ from registration')
    binding = _json(binding_raw)
    if (binding.get('schema') != 'water-residual-private-binding-v1'
            or binding.get('data_registration_commit') != DATA_COMMIT or binding.get('rng_seed') != RNG_SEED
            or any(type(binding.get(key)) is not str or re.fullmatch(r'[0-9a-f]{64}', binding[key]) is None
                   for key in ('baseline_sha256', 'train_sha256', 'dev_sha256', 'schema_sha256'))
            or type(binding.get('device_kind')) is not str or not binding['device_kind']
            or type(binding.get('wsl')) is not dict):
        raise ValueError('Residual private binding fields differ')
    return reg, binding


def _asset_check(stage, binding):
    for name, key, limit in (('baseline.npz', 'baseline_sha256', MAX_MODEL_BYTES),
                             ('train.npz', 'train_sha256', transfer.FINAL_BYTES),
                             ('dev.npz', 'dev_sha256', transfer.FINAL_BYTES)):
        if _sha(_read(stage / name, limit)) != binding[key]:
            raise ValueError('Residual private asset differs')


def _verify_host_gpu(expected_kind):
    result = subprocess.run(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'],
                            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, timeout=30, check=False)
    names = [line.strip() for line in result.stdout.decode('utf-8').splitlines() if line.strip()]
    if result.returncode != 0 or names != [expected_kind]:
        raise ValueError('The registered host GPU is unavailable or ambiguous')


def _weights(raw):
    """Decode the one small residual array without trusting an NPZ header."""
    import numpy as np
    if type(raw) is not bytes or not 0 < len(raw) <= 65536:
        raise ValueError('Residual weights archive size differs')
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        infos = archive.infolist()
        if len(infos) != 1 or infos[0].filename != 'weights.npy' or infos[0].file_size > 8192:
            raise ValueError('Residual weights archive differs')
        with archive.open(infos[0]) as source:
            version = np.lib.format.read_magic(source)
            if version == (1, 0):
                shape, fortran, dtype = np.lib.format.read_array_header_1_0(source)
            elif version == (2, 0):
                shape, fortran, dtype = np.lib.format.read_array_header_2_0(source)
            else:
                raise ValueError('Residual weights header version differs')
            if (shape != (373,) or fortran is not False or dtype != np.dtype('float32')
                    or infos[0].file_size != source.tell() + 373 * 4):
                raise ValueError('Residual weights header differs')
            payload = source.read(373 * 4 + 1)
    if len(payload) != 373 * 4:
        raise ValueError('Residual weights payload differs')
    weights = np.frombuffer(payload, dtype=np.float32)
    if not np.isfinite(weights).all() or not np.any(weights):
        raise ValueError('Residual weights are empty or nonfinite')
    return weights


def _equivalent(left, right):
    """Exact fields/counts and tolerant finite aggregate floating-point values."""
    if type(left) is not type(right):
        return False
    if type(left) is dict:
        return set(left) == set(right) and all(_equivalent(left[k], right[k]) for k in left)
    if type(left) is list:
        return len(left) == len(right) and all(_equivalent(a, b) for a, b in zip(left, right))
    if type(left) is float:
        return math.isfinite(left) and math.isfinite(right) and math.isclose(left, right, rel_tol=1e-5, abs_tol=1e-6)
    return left == right


def _validate_worker_output(head, binding, *, guard):
    """Authenticate output and independently re-evaluate consumed dev groups."""
    import numpy as np
    from learning.task_bc_v1.model import TaskNetwork
    from learning.task_choice_v1.features import OPS
    from learning.task_pair_checkpoint_v1 import snapshot
    from . import assessment, snapshot_data
    guard()
    result = _json(_read(STAGE / 'gpu-worker-result.local.json', MAX_METADATA))
    fields = {'schema', 'registration_commit', 'residual_sha256', 'baseline_unchanged', 'training',
              'development', 'parity', 'official_matches', 'independent_competitive_evaluation'}
    training, parity = result.get('training'), result.get('parity')
    if (set(result) != fields or result.get('schema') != 'water-residual-gpu-worker-result-v1'
            or result.get('registration_commit') != head or result.get('baseline_unchanged') is not True
            or type(result.get('official_matches')) is not int or result['official_matches'] != 0
            or result.get('independent_competitive_evaluation') is not False
            or type(training) is not dict or type(parity) is not dict):
        raise ValueError('Residual worker result differs')
    train_fields = {'input_groups', 'selected_groups', 'selected_by_role', 'float32_global_argmax_differences',
                    'float32_water_argmax_differences', 'updates_completed', 'nonzero_gradient_seen',
                    'weights_changed', 'device_verified'}
    roles = training.get('selected_by_role')
    if (set(training) != train_fields or type(training.get('input_groups')) is not int
            or training['input_groups'] != 59 or type(training.get('selected_groups')) is not int
            or not 16 <= training['selected_groups'] <= 58 or type(roles) is not dict
            or set(roles) != {'A01', 'A02'} or any(type(roles[k]) is not int or roles[k] < 8 for k in roles)
            or sum(roles.values()) != training['selected_groups']
            or any(type(training.get(key)) is not int or training[key] != expected for key, expected in
                   (('float32_global_argmax_differences', 0), ('float32_water_argmax_differences', 0),
                    ('updates_completed', 64)))
            or any(training.get(key) is not True for key in
                   ('nonzero_gradient_seen', 'weights_changed', 'device_verified'))):
        raise ValueError('Residual training evidence differs')
    parity_fields = {'groups', 'water_groups', 'water_candidates', 'max_abs_error', 'atol', 'rtol', 'allclose',
                     'residual_argmax_mismatches', 'adjusted_argmax_mismatches',
                     'baseline_operation_counts', 'operation_flips'}
    counts = parity.get('baseline_operation_counts')
    if (set(parity) != parity_fields or type(parity.get('groups')) is not int or parity['groups'] != 62
            or any(type(parity.get(k)) is not int or parity[k] < 0 for k in ('water_groups', 'water_candidates'))
            or type(counts) is not dict or set(counts) != set(OPS)
            or any(type(value) is not int or value < 0 for value in counts.values()) or sum(counts.values()) != 62
            or type(parity.get('max_abs_error')) is not float or not math.isfinite(parity['max_abs_error'])
            or parity['max_abs_error'] < 0 or parity.get('atol') != 0.0001 or parity.get('rtol') != 0.00001
            or parity.get('allclose') is not True
            or any(type(parity.get(k)) is not int or parity[k] != 0 for k in
                   ('residual_argmax_mismatches', 'adjusted_argmax_mismatches', 'operation_flips'))):
        raise ValueError('Residual numerical parity evidence differs')
    raw_weights = _read(STAGE / 'residual.local.npz', 65536)
    if _sha(raw_weights) != result.get('residual_sha256'):
        raise ValueError('Residual model identity differs')
    weights = _weights(raw_weights)
    _asset_check(STAGE, binding)
    raw_model = _read(STAGE / 'baseline.npz', MAX_MODEL_BYTES)
    snap = snapshot.load_checkpoint(raw_model, expected_sha256=binding['baseline_sha256'],
                                    expected_schema_sha256=binding['schema_sha256'])
    network = TaskNetwork(STAGE / 'baseline.npz', binding['baseline_sha256'])
    if (network.metadata['schema_sha256'] != binding['schema_sha256']
            or not np.array_equal(network.mean, snap.mean) or not np.array_equal(network.scale, snap.scale)
            or not all(np.array_equal(network.params[k], snap.params[k]) for k in snap.params)):
        raise ValueError('Frozen baseline reload differs')
    dev = snapshot_data.read_split(_read(STAGE / 'dev.npz', transfer.FINAL_BYTES), binding['dev_sha256'],
                                   'dev', guard=guard)
    development, rows = assessment.assess_development(dev, network, weights, network.metadata, guard=guard)
    if not _equivalent(result.get('development'), development):
        raise ValueError('Consumed development assessment differs')
    actual_counts = {op: sum(row['baseline_operation'] == op for row in rows) for op in OPS}
    actual_water_groups = sum(row['water_candidates'] > 0 for row in rows)
    actual_water_candidates = sum(row['water_candidates'] for row in rows)
    if (actual_water_groups < 1 or parity['baseline_operation_counts'] != actual_counts
            or parity['water_groups'] != actual_water_groups
            or parity['water_candidates'] != actual_water_candidates):
        raise ValueError('Residual numerical comparison population differs')
    _asset_check(STAGE, binding)
    return dict(development_screen=development['screen'], development=development,
                training=training, parity=parity, official_matches=0)


def run():
    """Start one supervised WSL attempt; unknown cleanup never becomes success."""
    if not _windows() or previous.physical_available() < MEMORY_INITIAL:
        raise MemoryError('Initial physical-memory floor not met')
    head = _git('rev-parse', 'HEAD')
    if _git('rev-parse', 'origin/main') != head:
        raise ValueError('Residual registration is not published')
    _reg, binding = verify_registration(head, STAGE)
    deadline = time.monotonic() + 300
    guard = lambda: previous._guard(deadline)
    _train, _dev, contract = transfer.load_trainable(guard=guard)
    if (contract['train_sha256'] != binding['train_sha256']
            or contract['dev_sha256'] != binding['dev_sha256']):
        raise ValueError('Residual snapshots are no longer authorized')
    _asset_check(STAGE, binding)
    wsl, expected_device = _older_wsl_contract()
    if wsl != binding['wsl'] or expected_device != binding['device_kind']:
        raise ValueError('Residual WSL runtime identity differs')
    if shutil.disk_usage(STAGE).free < DISK_FLOOR:
        raise OSError('Residual output disk floor not met')
    _verify_host_gpu(expected_device)
    _write(STAGE / 'host-attempt.local.json', previous._encoded(dict(
        schema='water-residual-host-attempt-v1', commit=head, retry_allowed=False)))
    from learning.task_pair_wsl_launch_v1.host import _system_wsl
    from learning.task_equiv_bc_v1 import host_gpu
    posix_stage = str(PurePosixPath(wsl['wsl_game_root']) / 'artifacts' / STAGE_NAME)
    supervisor = str(PurePosixPath(wsl['wsl_repo_root']) / 'kaggriculture' / 'learning'
                     / 'task_water_residual_v1' / 'gpu_supervisor.py')
    command = [str(_system_wsl()), '-d', wsl['distribution'], '--exec', '/usr/bin/env',
               'XLA_PYTHON_CLIENT_PREALLOCATE=false', wsl['wsl_venv_executable'], '-I', '-B', supervisor,
               '--stage', posix_stage, '--registration-commit', head]
    # This adapter specializes only this process; it never changes the old
    # module or its historic stage. The existing monitor owns its WSL handle.
    host_gpu.STAGE_NAME = STAGE_NAME
    result = host_gpu.supervise(STAGE, command, head, available=previous.physical_available)
    try:
        if result.status == 'passed':
            try:
                post_deadline = time.monotonic() + 300
                verified = _validate_worker_output(head, binding, guard=lambda: previous._guard(post_deadline))
                _registration_again, final_binding = verify_registration(head, STAGE)
                if final_binding != binding:
                    raise ValueError('Residual binding changed during execution')
            except Exception:
                _write(STAGE / 'host-result.local.json', previous._encoded(dict(status='failed',
                    reason='worker_validation', cleanup_confirmed=True, retry_allowed=False)))
                raise
            status = dict(status='passed', reason='worker_validated', cleanup_confirmed=True,
                          retry_allowed=False, verification=verified)
        else:
            status = dict(status=result.status, reason=result.reason,
                          cleanup_confirmed=result.cleanup_confirmed, retry_allowed=False)
        _write(STAGE / 'host-result.local.json', previous._encoded(status))
        print(json.dumps(dict(status=status['status'], cleanup_confirmed=status['cleanup_confirmed'])), flush=True)
        if result.status != 'passed':
            raise RuntimeError('Residual GPU attempt failed or cleanup is unconfirmed')
    finally:
        if result.status == 'cleanup_unconfirmed':
            # A write or validation error must not discard the owned handle.
            # Wait for transport exit; it still does not prove Linux cleanup.
            if result.child is not None:
                result.child.wait()
            if result.stdout_stream is not None:
                result.stdout_stream.close()
            if result.stderr_stream is not None:
                result.stderr_stream.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare', 'run'))
    args = parser.parse_args()
    prepare() if args.command == 'prepare' else run()


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        category = 'memory' if isinstance(exc, MemoryError) else 'io' if isinstance(exc, OSError) else 'invalid_or_runtime'
        print(json.dumps(dict(status='failed', category=category)), file=sys.stderr)
        raise SystemExit(1) from None
