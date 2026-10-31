"""Single preregistered CPU state-residual experiment on consumed data.

The diagnostic sets have already been observed and cannot prove independent
generalization or game reward. All snapshots, inputs, weights and identities
stay private. Importing this module never reads a sample or starts training.
"""
import argparse
from collections import Counter
from dataclasses import dataclass
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import secrets
import shutil
import subprocess
import sys
import time
import zipfile

from learning.task_student_continuation_v1 import experiment as previous
from learning.task_multi_teacher_probe_v1 import experiment as supervisor
from learning.task_teacher_transfer_dataset_v1 import experiment as transfer
from learning.task_teacher_cross_source_v1 import experiment as cross
from learning.task_multisource_rank_v1 import experiment as old
from learning.task_disagreement_trace_v1 import experiment as new
from learning.task_pair_dev_games_v1.panel import ENGINE_COMMIT, ENGINE_SHA


REPO = previous.REPO
PRIVATE = REPO.parent / 'private' / 'task-state-residual-v1-20261031'
REGISTRATION = REPO / 'review/evidence/task-state-residual-v1-20261031/registration.json'
BASE_COMMIT = 'cb85cb1cf80f9da1a65ba572391ab0baa40b4d78'
OLD_COMMIT = 'cce3eaa8edd8cbb710554e3e085a0d2b557aaf4e'
NEW_COMMIT = '384bc5f2605b157e160ba3b1910151d2eeaf53ee'
JOB = dict(supervisor.JOB)
INITIAL_MEMORY, MIN_DISK, OUTPUT_RESERVE = 5 * 1024**3, 4 * 1024**3, 4 * 1024**2
MAX_PARAMS = 2 * 1024**2
POOLS = ('old_train', 'new_train', 'old_diag', 'new_dev')
NEW_ROLES = {'new_train': ('A03', 'A04'), 'new_dev': ('A09', 'A10')}
OPS = ('DIG', 'FERTILIZE', 'HARVEST', 'PLANT', 'WATER')
RNG_DOMAIN = b'kaggriculture-state-residual-v1/optimizer'
SOURCE_NAMES = tuple(dict.fromkeys(old.SOURCE_NAMES + new.SOURCE_NAMES + tuple(
    'kaggriculture/learning/task_state_residual_v1/' + name for name in (
        '__init__.py', 'core.py', 'evaluate.py', 'split_dedup.py', 'experiment.py',
        'test_core.py', 'test_evaluate.py', 'test_split_dedup.py', 'test_experiment.py'))
    + ('review/evidence/task-state-residual-v1-20261031/REGISTRATION.zh-CN.md',)))


def _git(*args):
    return subprocess.check_output(['git', '-C', str(REPO), *args],
        stderr=subprocess.DEVNULL, timeout=30).decode('utf-8').strip()


def _read(path, limit=1024**2):
    return cross._read(path, limit=limit)


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _source_hashes():
    return {name: _sha(_read(REPO / name)) for name in SOURCE_NAMES}


def _registration(source_sha, binding_sha):
    return dict(schema='state-residual-prereg-v1', base_commit=BASE_COMMIT,
        old_registration_commit=OLD_COMMIT, new_registration_commit=NEW_COMMIT,
        source_sha256=source_sha, private_binding_sha256=binding_sha,
        official_rule_commit=ENGINE_COMMIT, official_rule_sha256=ENGINE_SHA,
        old_train=512, old_diagnostic=256, new_train=42, new_diagnostic=48,
        dedup='train_only_graph_then_all_feature_graph_diagnostic_excluded_if_train_connected',
        train_priority=['new_train', 'old_train'], diagnostic_priority=['new_dev', 'old_diag'],
        filter='single_or_identical_normalized_rows_or_new_train_teacher_student_identical',
        minimum=dict(old_train=16, new_train=16, old_diag=32, new_dev=16,
                     each_new_role=4, each_new_role_windows=3),
        verify_recorded_students='all_42_train_before_fit_all_48_diagnostic_after_fit',
        baseline_cast='subtract_float64_max_then_float32',
        probability_max_abs_error=1e-6, target_nll_atol=1e-4, target_nll_rtol=1e-6,
        hidden=16, updates=64, learning_rate=0.001, gradient_clip=1.0, kl_weight=0.1,
        batch_old=8, batch_new=8, per_role_batch_balance=False,
        optimizer_rng_domain=RNG_DOMAIN.decode('ascii'),
        optimizer_rng_encoding='sha256_first_four_bytes_little_endian',
        parity='all_retained_fixed_cpu_jax_batch_vs_numpy_deployment',
        parity_atol=1e-4, parity_rtol=1e-5,
        screening=dict(new_dev_net_correct_at_least=2, each_new_role_no_drop=True,
            old_non_water_minimum_groups=20, old_non_water_max_drop_fraction=0.05,
            at_least_one_changed_choice=True),
        job=dict(JOB), memory_initial_bytes=INITIAL_MEMORY, disk_floor_bytes=MIN_DISK,
        output_reserve_bytes=OUTPUT_RESERVE, max_attempts=1, device='cpu', gpu=False,
        new_games=0, independent_evaluation=False, promotion=False)


def _source_contracts(guard):
    """Authenticate historical seals without scoring or running a game."""
    from learning.task_student_anchor_collection_v1 import sources
    guard()
    completed = new.load_completed(guard=guard)
    if (completed['registration_commit'] != NEW_COMMIT or completed['summary']['data_gate'] != 'passed'
            or sum(completed['summary']['retained_by_role'][r] for r in NEW_ROLES['new_train']) != 42
            or sum(completed['summary']['retained_by_role'][r] for r in NEW_ROLES['new_dev']) != 48):
        raise ValueError('Consumed disagreement source differs')
    old_binding, old_sha = old._registered_binding(OLD_COMMIT)
    new_binding, new_sha = new._registered_binding(NEW_COMMIT)
    context = sources.open_context(guard=guard)
    if new._context_binding(context) != {k: v for k, v in new_binding.items() if k not in ('schema', 'salt')}:
        raise ValueError('Consumed source context differs')
    _mean, _scale, metadata = context.model_normalization()
    prior = old_binding['earlier']
    if (context.task_sha256 != prior['baseline_sha256'] or context.task_sha256 != new_binding['baseline_sha256']
            or context.crop_sha256 != new_binding['crop_sha256']
            or prior['schema_sha256'] != new_binding['schema_sha256']
            or cross._schema_identity(prior['historical']['schema_sha256'], metadata) != prior['schema_sha256']):
        raise ValueError('Frozen model or schema identity differs')
    identity = dict(old_binding_sha256=old_sha, new_binding_sha256=new_sha,
        baseline_sha256=context.task_sha256, crop_sha256=context.crop_sha256,
        schema_sha256=prior['schema_sha256'], metadata_sha256=_sha(previous._encoded(metadata)),
        train_sha256=completed['train_sha256'], diagnostic_sha256=completed['diagnostic_sha256'])
    guard()
    return identity, context, old_binding, new_binding


def prepare():
    if (os.name != 'nt' or _git('rev-parse', 'HEAD') != BASE_COMMIT
            or _git('diff', '--name-only') or _git('diff', '--cached', '--name-only')
            or REGISTRATION.exists() or PRIVATE.exists()):
        raise ValueError('Unexpected registration state')
    if previous.physical_available() < INITIAL_MEMORY:
        raise MemoryError('Initial memory floor')
    deadline = time.monotonic() + 300
    guard = lambda: previous._guard(deadline)
    identity, _context, _old, _new = _source_contracts(guard)
    binding = dict(schema='state-residual-private-binding-v1', salt=secrets.token_hex(32), **identity)
    registration = _registration(_source_hashes(), _sha(previous._encoded(binding)))
    PRIVATE.mkdir(parents=False, exist_ok=False)
    previous._write_new(PRIVATE / 'binding.local.json', binding)
    REGISTRATION.parent.mkdir(parents=True, exist_ok=True)
    previous._write_new(REGISTRATION, registration)
    print(json.dumps(dict(status='prepared')), flush=True)


def _registered_binding(head):
    if type(head) is not str or re.fullmatch(r'[0-9a-f]{40}', head) is None:
        raise ValueError('Invalid registration commit')
    parent = _git('rev-parse', head + '^')
    raw = subprocess.check_output(['git', '-C', str(REPO), 'show',
        head + ':' + REGISTRATION.relative_to(REPO).as_posix()], stderr=subprocess.DEVNULL, timeout=30)
    binding_raw = _read(PRIVATE / 'binding.local.json')
    if parent != BASE_COMMIT or previous._document(raw) != _registration(_source_hashes(), _sha(binding_raw)):
        raise ValueError('Registration differs')
    binding = previous._document(binding_raw)
    fields = {'schema','salt','old_binding_sha256','new_binding_sha256','baseline_sha256',
              'crop_sha256','schema_sha256','metadata_sha256','train_sha256','diagnostic_sha256'}
    if (set(binding) != fields or binding.get('schema') != 'state-residual-private-binding-v1'
            or any(type(binding[key]) is not str or re.fullmatch(r'[0-9a-f]{64}', binding[key]) is None
                   for key in fields - {'schema'})):
        raise ValueError('Private source binding differs')
    return binding, _sha(binding_raw)


def _runtime_binding():
    head = _git('rev-parse', 'HEAD')
    if _git('status', '--porcelain') or _git('rev-parse', 'origin/main') != head:
        raise ValueError('Registration is not clean and published')
    binding, digest = _registered_binding(head)
    return head, binding, digest


@dataclass(frozen=True, slots=True, repr=False)
class _Example:
    item: object
    step: int | None
    ordinal: int | None
    student: int | None

    def __repr__(self):
        return '<PrivateStateExample>'


def _snapshot(array):
    import numpy as np
    if type(array) is not np.ndarray or array.dtype != np.dtype('float32'):
        raise ValueError('Private feature dtype differs')
    return np.frombuffer(array.tobytes(order='C'), dtype=np.float32).reshape(array.shape)


def _examples(old_train, old_diag, old_roles, arrays, guard):
    """Map authenticated groups to stable private items in fixed priority order."""
    from learning.task_multisource_rank_v1 import dedup
    if len(old_train) != 512 or len(old_diag) != 256 or len(old_roles) != 256 or len(arrays) != 2:
        raise ValueError('Source population differs')
    output = []
    for pool, data, roles in (('new_train', arrays[0], NEW_ROLES['new_train']),
                              ('new_dev', arrays[1], NEW_ROLES['new_dev'])):
        if len(data['teacher_target']) != (42 if pool == 'new_train' else 48):
            raise ValueError('Consumed group population differs')
        for i in range(len(data['teacher_target'])):
            guard()
            role_index = int(data['role'][i])
            if not 0 <= role_index < len(new.ROLES) or new.ROLES[role_index] not in roles:
                raise ValueError('Consumed group role differs')
            lo, hi = int(data['offsets'][i]), int(data['offsets'][i + 1])
            item = dedup.Item(_snapshot(data['shared'][i]), _snapshot(data['candidates'][lo:hi]),
                              int(data['teacher_target'][i]), pool, new.ROLES[role_index])
            output.append(_Example(item, int(data['step'][i]), int(data['ordinal'][i]),
                                   int(data['student_index'][i])))
    for index, group in enumerate(old_train):
        guard()
        role = 'T01' if index < old.TRAIN_PER_OLD else 'T02'
        output.append(_Example(dedup.Item(_snapshot(group.shared), _snapshot(group.candidates),
                                         group.target_index, 'old_train', role), None, None, None))
    for group, role in zip(old_diag, old_roles):
        guard()
        if role not in ('T01', 'T02'):
            raise ValueError('Old diagnostic role differs')
        output.append(_Example(dedup.Item(_snapshot(group.shared), _snapshot(group.candidates),
                                         group.target_index, 'old_diag', role), None, None, None))
    return output


def _filter_pools(examples, mean, scale, guard):
    """Use training-only decisions before any diagnostic target comparison."""
    import numpy as np
    from . import split_dedup
    if type(examples) is not list or len(examples) != 858:
        raise ValueError('Private input population differs')
    lookup = {id(ex.item): ex for ex in examples}
    if len(lookup) != len(examples):
        raise ValueError('Duplicate private item identity')
    retained, split_summary = split_dedup.deduplicate([ex.item for ex in examples],
                                                       mean=mean, scale=scale, guard=guard)
    pools = {pool: [] for pool in POOLS}
    filtered = {pool: dict(single=0, identical_rows=0, student_same=0) for pool in POOLS}
    for item in retained:
        guard()
        ex = lookup[id(item)]
        if len(item.candidates) == 1:
            filtered[item.pool]['single'] += 1
            continue
        with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
            normalized = ((item.candidates.astype(np.float64) - mean[275:]) / scale[275:]).astype(np.float32)
        if not np.isfinite(normalized).all():
            raise ValueError('Nonfinite normalized candidate')
        normalized[normalized == 0] = np.float32(0)
        if np.array_equal(normalized, np.broadcast_to(normalized[:1], normalized.shape)):
            filtered[item.pool]['identical_rows'] += 1
            continue
        if item.pool == 'new_train':
            if type(ex.student) is not int or not 0 <= ex.student < len(item.candidates):
                raise ValueError('Invalid private student index')
            if np.array_equal(normalized[item.target], normalized[ex.student]):
                filtered[item.pool]['student_same'] += 1
                continue
        pools[item.pool].append(ex)
    roles = {role: sum(ex.item.role == role for pool in NEW_ROLES for ex in pools[pool])
             for role in ('A03','A04','A09','A10')}
    windows = {role: len({ex.step // 120 for pool in NEW_ROLES for ex in pools[pool]
                          if ex.item.role == role}) for role in roles}
    checks = dict(old_train=len(pools['old_train']) >= 16, new_train=len(pools['new_train']) >= 16,
                  old_diag=len(pools['old_diag']) >= 32, new_dev=len(pools['new_dev']) >= 16,
                  each_new_role=all(roles[r] >= 4 for r in roles),
                  each_new_role_windows=all(windows[r] >= 3 for r in roles))
    summary = dict(split=split_summary, filtered=filtered,
                   eligible_by_pool={pool: len(pools[pool]) for pool in POOLS},
                   new_roles=roles, new_role_windows=windows, checks=checks,
                   passed=all(checks.values()))
    return pools, summary


def _model(context, binding, guard):
    """Make the new stage's exclusive immutable-by-convention baseline copy."""
    import numpy as np
    from learning.task_bc_v1.model import TaskNetwork
    from learning.task_pair_checkpoint_v1 import snapshot
    guard()
    raw = _read(context._task_path, limit=snapshot.MAX_ARCHIVE_BYTES)
    checkpoint = snapshot.load_checkpoint(raw, expected_sha256=binding['baseline_sha256'],
                                          expected_schema_sha256=binding['schema_sha256'])
    path = PRIVATE / 'baseline.local.npz'
    with path.open('xb') as output:
        if output.write(raw) != len(raw):
            raise OSError('Incomplete private baseline write')
        output.flush()
        os.fsync(output.fileno())
    if _read(path, limit=snapshot.MAX_ARCHIVE_BYTES) != raw:
        raise ValueError('Private baseline snapshot differs')
    network = TaskNetwork(path, binding['baseline_sha256'])
    if (_sha(previous._encoded(network.metadata)) != binding['metadata_sha256']
            or not np.array_equal(network.mean, checkpoint.mean)
            or not np.array_equal(network.scale, checkpoint.scale)
            or any(not np.array_equal(network.params[key], checkpoint.params[key]) for key in checkpoint.params)):
        raise ValueError('Frozen model snapshot differs')
    guard()
    return network


def _score(examples, network, *, verify_students, guard):
    """No label or stored student index is ever passed to the scorer."""
    import numpy as np
    from learning.task_multisource_rank_v1 import core as rank
    scored = {}
    for ex in examples:
        guard()
        item = ex.item
        scores = network.score(item.shared, item.candidates)
        if (type(scores) is not np.ndarray or scores.dtype != np.dtype('float64')
                or scores.shape != (len(item.candidates),) or not np.isfinite(scores).all()):
            raise ValueError('Frozen baseline scoring differs')
        if verify_students and (type(ex.student) is not int or int(np.argmax(scores)) != ex.student):
            raise ValueError('Recorded student index differs')
        score = np.frombuffer(scores.tobytes(), dtype=np.float64)
        scored[id(item)] = rank.Group(item.shared, item.candidates, score, item.target,
                                      'new' if item.pool.startswith('new') else 'old')
    return scored


class PrecisionError(ValueError):
    def __init__(self, reason):
        super().__init__('Private baseline precision gate failed')
        self.reason = reason


def _precision(groups, guard):
    """Compare centered input representations, before JAX computation."""
    import numpy as np
    from . import core
    if not groups:
        raise PrecisionError('empty')
    max_probability = max_nll = max_ratio = 0.0
    for group in groups:
        guard()
        try:
            f64, f32 = core._training_baseline(group.baseline)
        except core.StateResidualError:
            raise PrecisionError('centered_cast_or_argmax') from None
        a, b = np.asarray(f64, np.float64), np.asarray(f32, np.float64)
        p, q = np.exp(a), np.exp(b)
        p /= p.sum(); q /= q.sum()
        with np.errstate(over='ignore', invalid='ignore'):
            nll64 = float(np.log(np.exp(a).sum()) - a[group.target])
            nll32 = float(np.log(np.exp(b).sum()) - b[group.target])
            diff = float(np.max(np.abs(p - q)))
            ndiff = abs(nll64 - nll32)
            limit = 1e-4 + 1e-6 * abs(nll64)
        if not all(math.isfinite(value) for value in (diff, nll64, nll32, ndiff, limit)):
            raise PrecisionError('nonfinite')
        if diff > 1e-6:
            raise PrecisionError('probability')
        if ndiff > limit:
            raise PrecisionError('target_nll')
        max_probability, max_nll, max_ratio = max(max_probability, diff), max(max_nll, ndiff), max(max_ratio, ndiff / limit)
    return dict(groups=len(groups), probability_max_abs_error=max_probability,
                target_nll_max_abs_error=max_nll, target_nll_max_limit_ratio=max_ratio,
                argmax_stable=True, passed=True)


def _decode_params(raw):
    import numpy as np
    from . import core
    shapes = {'ws': (275, 16), 'wc': (98, 16), 'b': (16,), 'v': (16,)}
    if type(raw) is not bytes or not 0 < len(raw) <= MAX_PARAMS:
        raise ValueError('Private parameter archive differs')
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        infos = archive.infolist()
        if (len(infos) != len(shapes) or {v.filename for v in infos} != {k + '.npy' for k in shapes}
                or sum(v.file_size for v in infos) > MAX_PARAMS):
            raise ValueError('Private parameter archive differs')
        for info in infos:
            with archive.open(info) as src:
                shape, fortran, dtype = new._bounded_header(src, info.file_size)
                if (shape != shapes[info.filename[:-4]] or fortran is not False
                        or dtype != np.dtype('float32') or dtype.hasobject
                        or info.file_size != src.tell() + math.prod(shape) * dtype.itemsize):
                    raise ValueError('Private parameter header differs')
    with np.load(io.BytesIO(raw), allow_pickle=False) as source:
        params = core.Params(**{name: source[name] for name in shapes})
    params = core._checked_params(params)
    if not np.any(params.v):
        raise ValueError('Output layer did not change')
    return params


def _write_params(params):
    import numpy as np
    from . import core
    params = core._checked_params(params)
    output = io.BytesIO()
    np.savez_compressed(output, **{name: getattr(params, name) for name in ('ws','wc','b','v')})
    raw = output.getvalue()
    restored = _decode_params(raw)
    if any(not np.array_equal(getattr(params, name), getattr(restored, name)) for name in ('ws','wc','b','v')):
        raise ValueError('Private parameters did not round trip')
    with (PRIVATE / 'params.local.npz').open('xb') as dst:
        if dst.write(raw) != len(raw):
            raise OSError('Incomplete private parameter write')
        dst.flush(); os.fsync(dst.fileno())
    if _read(PRIVATE / 'params.local.npz', limit=MAX_PARAMS) != raw:
        raise ValueError('Private parameters changed after writing')
    return restored, _sha(raw)


def _screen(new_dev, old_diag):
    overall = new_dev['overall']
    per_role = []
    for role in ('A09','A10'):
        row = new_dev['by_role'][role]['overall']
        per_role.append((row['residual_correct'] >= row['baseline_correct']) if row['groups'] else None)
    role_ok = False if False in per_role else None if None in per_role else True
    nonwater = [old_diag['by_target_operation'][op] for op in OPS if op != 'WATER']
    count = sum(row['groups'] for row in nonwater)
    before = sum(row['baseline_correct'] for row in nonwater)
    after = sum(row['residual_correct'] for row in nonwater)
    checks = dict(new_dev_net_correct_at_least_two=overall['residual_correct'] - overall['baseline_correct'] >= 2,
        each_new_role_no_drop=role_ok, new_dev_at_least_one_change=overall['changed'] > 0,
        old_non_water_coverage_at_least_twenty=count >= 20,
        old_non_water_drop_at_most_five_points=(after - before >= -0.05 * count) if count >= 20 else None)
    failed = any(value is False for key, value in checks.items() if key != 'old_non_water_coverage_at_least_twenty')
    incomplete = not checks['old_non_water_coverage_at_least_twenty'] or any(v is None for v in checks.values())
    return dict(checks=checks, screen='failed' if failed else 'unknown' if incomplete else 'passed',
                old_non_water_groups=count, old_non_water_baseline_correct=before,
                old_non_water_residual_correct=after)


def _validate_coverage(value):
    if (type(value) is not dict or set(value) != {'split','filtered','eligible_by_pool','new_roles',
                                                  'new_role_windows','checks','passed'}
            or type(value['eligible_by_pool']) is not dict or set(value['eligible_by_pool']) != set(POOLS)
            or type(value['filtered']) is not dict or set(value['filtered']) != set(POOLS)
            or type(value['split']) is not dict):
        raise ValueError('Private coverage differs')
    split = value['split']
    split_fields = {'input_count','retained_count','input_by_pool','retained_by_pool',
        'duplicate_removed_by_pool','conflict_removed_by_pool',
        'diagnostic_excluded_train_connected_by_pool','train_component_count','all_feature_component_count',
        'train_conflict_components','diagnostic_only_components','diagnostic_conflict_components',
        'diagnostic_train_connected_components'}
    maps = ('input_by_pool','retained_by_pool','duplicate_removed_by_pool',
            'conflict_removed_by_pool','diagnostic_excluded_train_connected_by_pool')
    counts = split_fields - set(maps)
    if (set(split) != split_fields or any(type(split[key]) is not int or not 0 <= split[key] <= 858 for key in counts)
            or split['train_conflict_components'] > split['train_component_count']
            or split['diagnostic_conflict_components'] > split['diagnostic_only_components']
            or split['diagnostic_only_components'] + split['diagnostic_train_connected_components'] > split['all_feature_component_count']
            or any(type(split.get(key)) is not dict or set(split[key]) != set(POOLS)
            or any(type(v) is not int or v < 0 for v in split[key].values()) for key in maps)
            or split['input_by_pool'] != {'old_train':512,'new_train':42,'old_diag':256,'new_dev':48}
            or split.get('input_count') != 858
            or split.get('retained_count') != sum(split['retained_by_pool'].values())
            or split['retained_by_pool']['old_train'] + split['retained_by_pool']['new_train']
               != split['train_component_count'] - split['train_conflict_components']
            or split['retained_by_pool']['old_diag'] + split['retained_by_pool']['new_dev']
               != split['diagnostic_only_components'] - split['diagnostic_conflict_components']):
        raise ValueError('Private source accounting differs')
    for pool in POOLS:
        row, eligible = value['filtered'][pool], value['eligible_by_pool'][pool]
        if (type(row) is not dict or set(row) != {'single','identical_rows','student_same'}
                or any(type(v) is not int or v < 0 for v in row.values())
                or type(eligible) is not int or eligible < 0
                or (pool != 'new_train' and row['student_same'] != 0)
                or split['input_by_pool'][pool] != split['retained_by_pool'][pool]
                   + split['duplicate_removed_by_pool'][pool] + split['conflict_removed_by_pool'][pool]
                   + split['diagnostic_excluded_train_connected_by_pool'][pool]
                or split['retained_by_pool'][pool] != eligible + sum(row.values())
                or (pool.endswith('train') and split['diagnostic_excluded_train_connected_by_pool'][pool] != 0)):
            raise ValueError('Private filter accounting differs')
    roles, windows = value['new_roles'], value['new_role_windows']
    vocabulary = {'A03','A04','A09','A10'}
    role_limits = {'A03':21,'A04':21,'A09':24,'A10':24}
    if (type(roles) is not dict or set(roles) != vocabulary or type(windows) is not dict or set(windows) != vocabulary
            or any(type(roles[r]) is not int or not 0 <= roles[r] <= role_limits[r] or type(windows[r]) is not int
                   or not 0 <= windows[r] <= min(6, roles[r]) or roles[r] > 4 * windows[r] for r in vocabulary)
            or sum(roles[r] for r in NEW_ROLES['new_train']) != value['eligible_by_pool']['new_train']
            or sum(roles[r] for r in NEW_ROLES['new_dev']) != value['eligible_by_pool']['new_dev']):
        raise ValueError('Private role accounting differs')
    expected = dict(old_train=value['eligible_by_pool']['old_train'] >= 16,
        new_train=value['eligible_by_pool']['new_train'] >= 16,
        old_diag=value['eligible_by_pool']['old_diag'] >= 32, new_dev=value['eligible_by_pool']['new_dev'] >= 16,
        each_new_role=all(roles[r] >= 4 for r in vocabulary),
        each_new_role_windows=all(windows[r] >= 3 for r in vocabulary))
    if value['checks'] != expected or value['passed'] is not all(expected.values()):
        raise ValueError('Private coverage gate differs')


def _validate_assessment(value, roles, source):
    from learning.task_multisource_rank_v1 import evaluate as prior
    if type(value) is not dict or set(value) != {'overall','by_target_operation','by_role','independent_evaluation'}:
        raise ValueError('Private assessment differs')
    empty = prior._summary([])
    by_source = {'old': value['overall'] if source == 'old' else empty,
                 'new': value['overall'] if source == 'new' else empty}
    old._validate_summary(dict(value, by_source=by_source), roles, source)
    buckets = [value['overall'], *value['by_target_operation'].values()]
    for role in roles:
        section = value['by_role'][role]
        buckets.extend((section['overall'], *section['by_target_operation'].values()))
    if any(row['regressed'] > row['baseline_correct']
           or row['improved'] > row['groups'] - row['baseline_correct'] for row in buckets):
        raise ValueError('Private assessment outcome accounting differs')


def _validate_precision(value, groups):
    if (type(value) is not dict or set(value) != {'groups','probability_max_abs_error',
            'target_nll_max_abs_error','target_nll_max_limit_ratio','argmax_stable','passed'}
            or type(value['groups']) is not int or value['groups'] != groups
            or value['argmax_stable'] is not True or value['passed'] is not True):
        raise ValueError('Precision receipt differs')
    for key in ('probability_max_abs_error','target_nll_max_abs_error','target_nll_max_limit_ratio'):
        if type(value[key]) not in (int,float) or not math.isfinite(value[key]) or value[key] < 0:
            raise ValueError('Precision receipt differs')
    if value['probability_max_abs_error'] > 1e-6 or value['target_nll_max_limit_ratio'] > 1:
        raise ValueError('Precision receipt differs')


def _result(raw, head):
    value = previous._document(raw)
    fields = {'schema','registration_commit','coverage','train_student_checked','diagnostic_student_checked',
              'train_precision','diagnostic_precision','parity','new_dev','old_diag','screen',
              'baseline_sha256','params_sha256','updates_completed','nonzero_gradient_seen',
              'output_layer_changed','device','new_games','gpu','independent_evaluation','promotion','software'}
    if (set(value) != fields or value.get('schema') != 'state-residual-result-v1'
            or value.get('registration_commit') != head or value.get('device') != 'cpu'
            or value.get('gpu') is not False or type(value.get('new_games')) is not int or value['new_games'] != 0
            or value.get('independent_evaluation') is not False or value.get('promotion') is not False
            or type(value.get('updates_completed')) is not int or value['updates_completed'] != 64
            or value.get('nonzero_gradient_seen') is not True or value.get('output_layer_changed') is not True
            or value.get('train_student_checked') != dict(groups=42, mismatches=0)
            or value.get('diagnostic_student_checked') != dict(groups=48, mismatches=0)
            or any(type(value.get(key)) is not str or re.fullmatch(r'[0-9a-f]{64}', value[key]) is None
                   for key in ('baseline_sha256','params_sha256'))
            or type(value.get('software')) is not dict or set(value['software']) != {'python','numpy','jax','optax'}
            or any(type(v) is not str or re.fullmatch(r'[A-Za-z0-9.+_-]{1,64}', v) is None
                   for v in value['software'].values())):
        raise ValueError('Private experiment result differs')
    _validate_coverage(value['coverage'])
    if value['coverage']['passed'] is not True:
        raise ValueError('Insufficient retained coverage')
    counts = value['coverage']['eligible_by_pool']
    train_count, diagnostic_count = counts['old_train'] + counts['new_train'], counts['old_diag'] + counts['new_dev']
    _validate_precision(value['train_precision'], train_count)
    _validate_precision(value['diagnostic_precision'], diagnostic_count)
    _validate_assessment(value['new_dev'], ('A09','A10'), 'new')
    _validate_assessment(value['old_diag'], ('T01','T02'), 'old')
    if (value['new_dev']['overall']['groups'] != counts['new_dev']
            or value['old_diag']['overall']['groups'] != counts['old_diag']
            or any(value['new_dev']['by_role'][role]['overall']['groups'] != value['coverage']['new_roles'][role]
                   for role in ('A09','A10'))
            or value['screen'] != _screen(value['new_dev'], value['old_diag'])):
        raise ValueError('Private assessment accounting differs')
    parity = value['parity']
    total = train_count + diagnostic_count
    if (type(parity) is not dict or set(parity) != {'groups','allclose','baseline_argmax_mismatches',
            'training_choice_mismatches','maximum_absolute_error','fixed_batch','fixed_width',
            'train_width','train_width_groups','full_width_groups','groups_by_width','device'}
            or type(parity['groups']) is not int or parity['groups'] != total or parity['allclose'] is not True
            or type(parity['baseline_argmax_mismatches']) is not int or parity['baseline_argmax_mismatches'] != 0
            or type(parity['training_choice_mismatches']) is not int or parity['training_choice_mismatches'] != 0
            or parity['fixed_batch'] != 16 or parity['fixed_width'] != 256 or parity['device'] != 'cpu'
            or type(parity['train_width']) is not int or not 2 <= parity['train_width'] <= 256
            or type(parity['train_width_groups']) is not int or not train_count <= parity['train_width_groups'] <= total
            or parity['full_width_groups'] != total or type(parity['groups_by_width']) is not dict
            or set(parity['groups_by_width']) != {str(parity['train_width']), '256'}
            or parity['groups_by_width'][str(parity['train_width'])] != parity['train_width_groups']
            or parity['groups_by_width']['256'] != total
            or type(parity['maximum_absolute_error']) not in (int,float)
            or not math.isfinite(parity['maximum_absolute_error']) or parity['maximum_absolute_error'] < 0):
        raise ValueError('Private parity receipt differs')
    return value


def _validated_outputs(head, binding, guard):
    result = _result(_read(PRIVATE / 'worker-result.local.json'), head)
    if previous._document(_read(PRIVATE / 'coverage.local.json')) != result['coverage']:
        raise ValueError('Private coverage receipt differs')
    raw = _read(PRIVATE / 'params.local.npz', limit=MAX_PARAMS)
    _decode_params(raw)
    baseline = _read(PRIVATE / 'baseline.local.npz', limit=2 * 1024**2)
    if (_sha(raw) != result['params_sha256'] or _sha(baseline) != result['baseline_sha256']
            or result['baseline_sha256'] != binding['baseline_sha256']):
        raise ValueError('Private model identity differs')
    guard()
    return result


def child():
    head, binding, binding_sha = _runtime_binding()
    claim = previous._document(_read(PRIVATE / 'claim.local.json'))
    attempt = previous._document(_read(PRIVATE / 'attempt.local.json'))
    if (claim.get('schema') != 'state-residual-claim-v1' or claim.get('commit') != head
            or type(claim.get('attempt')) is not int or claim['attempt'] != 1
            or claim.get('freeze_sha256') != binding_sha or claim.get('retry_allowed') is not False
            or attempt != dict(schema='state-residual-attempt-v1', commit=head, retry_allowed=False)
            or (PRIVATE / 'failed.local.json').exists() or (PRIVATE / 'closed.local.json').exists()):
        raise ValueError('Child is not uniquely claimed')
    from learning.task_teacher_dev_deviation_v1.membership import ENV_KEY, verify_worker
    from learning.task_pair_fresh_multi_v1.entry import verify_live_owner
    import psutil
    owner = previous._document(_read(PRIVATE / 'owner.local.json'))
    verify_worker(os.environ.get(ENV_KEY), claim, binding_sha)
    verify_live_owner(owner, commit=head, binding_sha=binding_sha, current_pid=os.getpid(),
                      process=psutil.Process, executable=sys.executable, samefile=os.path.samefile,
                      now=time.monotonic())
    previous._write_new(PRIVATE / 'child-attempt.local.json', dict(
        schema='state-residual-child-attempt-v1', commit=head, retry_allowed=False))
    deadline = time.monotonic() + JOB['wall_seconds']
    guard = lambda: previous._guard(deadline)
    phase = 'source'
    try:
        import numpy as np
        from . import core, evaluate
        identity, context, old_binding, new_binding = _source_contracts(guard)
        if identity != {k: v for k, v in binding.items() if k not in ('schema','salt')}:
            raise ValueError('Source binding changed')
        source_result, arrays = new._validated_outputs(NEW_COMMIT, guard, new_binding['operation_columns'])
        if (source_result['summary']['data_gate'] != 'passed'
                or source_result['train_sha256'] != binding['train_sha256']
                or source_result['diagnostic_sha256'] != binding['diagnostic_sha256']):
            raise ValueError('Consumed source differs')
        old_train, old_diag, old_roles = old._collect_old(old_binding, guard)
        examples = _examples(old_train, old_diag, old_roles, arrays, guard)
        del old_train, old_diag, old_roles, arrays
        mean, scale, metadata = context.model_normalization()
        if _sha(previous._encoded(metadata)) != binding['metadata_sha256']:
            raise ValueError('Model metadata differs')
        phase = 'coverage'
        pools, coverage = _filter_pools(examples, mean, scale, guard)
        previous._write_new(PRIVATE / 'coverage.local.json', coverage)
        if coverage['passed'] is not True:
            raise ValueError('Insufficient retained coverage')
        _validate_coverage(coverage)
        # Both source archives have completed their final hash at this point.
        phase = 'baseline_snapshot'
        network = _model(context, binding, guard)
        original_new_train = [ex for ex in examples if ex.item.pool == 'new_train']
        original_new_dev = [ex for ex in examples if ex.item.pool == 'new_dev']
        if (len(original_new_train), len(original_new_dev)) != (42, 48):
            raise ValueError('Consumed student population differs')
        phase = 'train_student_identity'
        new_scores = _score(original_new_train, network, verify_students=True, guard=guard)
        old_scores = _score(pools['old_train'], network, verify_students=False, guard=guard)
        train_examples = pools['old_train'] + pools['new_train']
        train_groups = [(new_scores if ex.item.pool == 'new_train' else old_scores)[id(ex.item)]
                        for ex in train_examples]
        phase = 'train_precision'
        train_precision = _precision(train_groups, guard)
        train_width = max(len(group.candidates) for group in train_groups)
        if not 2 <= train_width <= 256 or any(len(group.candidates) > train_width for group in train_groups):
            raise ValueError('Training width differs')
        phase = 'cpu_training'
        import jax
        import optax
        devices = jax.devices('cpu')
        if len(devices) != 1 or devices[0].platform != 'cpu':
            raise ValueError('Registered CPU backend unavailable')
        rng_seed = int.from_bytes(hashlib.sha256(RNG_DOMAIN).digest()[:4], 'little')
        trained = core.fit(train_groups, mean=mean, scale=scale, device=devices[0], guard=guard,
                           rng_seed=rng_seed)
        if (trained.updates_completed != 64 or not trained.nonzero_gradient_seen
                or not trained.output_layer_changed):
            raise ValueError('Training did not change the output layer')
        params, params_sha = _write_params(trained.params)
        # No diagnostic scores or recorded-index comparisons occur before fit.
        phase = 'diagnostic_student_identity'
        new_diag_scores = _score(original_new_dev, network, verify_students=True, guard=guard)
        old_diag_scores = _score(pools['old_diag'], network, verify_students=False, guard=guard)
        old_diag_groups = [old_diag_scores[id(ex.item)] for ex in pools['old_diag']]
        new_dev_groups = [new_diag_scores[id(ex.item)] for ex in pools['new_dev']]
        diagnostic_groups = old_diag_groups + new_dev_groups
        phase = 'diagnostic_precision'
        diagnostic_precision = _precision(diagnostic_groups, guard)
        phase = 'parity'
        all_groups = train_groups + diagnostic_groups
        parity = evaluate.parity(all_groups, mean, scale, params, device=devices[0],
                                 guard=guard, train_width=train_width)
        expected_width_count = sum(len(group.candidates) <= train_width for group in all_groups)
        if (parity['train_width_groups'] != expected_width_count
                or parity['train_width_groups'] < len(train_groups)
                or parity['full_width_groups'] != len(all_groups)):
            raise ValueError('Parity population differs')
        phase = 'assessment'
        old_summary = evaluate.assess(old_diag_groups, mean, scale, params, metadata,
            roles=[ex.item.role for ex in pools['old_diag']], role_labels=('T01','T02'), guard=guard)
        new_summary = evaluate.assess(new_dev_groups, mean, scale, params, metadata,
            roles=[ex.item.role for ex in pools['new_dev']], role_labels=('A09','A10'), guard=guard)
        screen = _screen(new_summary, old_summary)
        current, _context, _old, _new = _source_contracts(guard)
        if current != identity or _runtime_binding() != (head, binding, binding_sha):
            raise ValueError('Source or registration changed')
        if _sha(_read(PRIVATE / 'baseline.local.npz', limit=2 * 1024**2)) != binding['baseline_sha256']:
            raise ValueError('Private baseline changed')
        result = dict(schema='state-residual-result-v1', registration_commit=head, coverage=coverage,
            train_student_checked=dict(groups=42, mismatches=0),
            diagnostic_student_checked=dict(groups=48, mismatches=0),
            train_precision=train_precision, diagnostic_precision=diagnostic_precision, parity=parity,
            new_dev=new_summary, old_diag=old_summary, screen=screen,
            baseline_sha256=binding['baseline_sha256'], params_sha256=params_sha,
            updates_completed=trained.updates_completed, nonzero_gradient_seen=trained.nonzero_gradient_seen,
            output_layer_changed=trained.output_layer_changed, device='cpu', new_games=0, gpu=False,
            independent_evaluation=False, promotion=False,
            software=dict(python='.'.join(map(str, sys.version_info[:3])), numpy=np.__version__,
                          jax=jax.__version__, optax=optax.__version__))
        _result(previous._encoded(result), head)
        guard()
        previous._write_new(PRIVATE / 'worker-result.local.json', result)
    except BaseException as exc:
        reason = exc.reason if isinstance(exc, PrecisionError) else type(exc).__name__
        previous._write_new(PRIVATE / 'worker-failed.local.json', dict(
            phase=phase, category=reason, retry_allowed=False))
        raise


def run():
    if os.name != 'nt' or previous.physical_available() < INITIAL_MEMORY:
        raise MemoryError('Initial memory floor')
    if shutil.disk_usage(PRIVATE).free < MIN_DISK + OUTPUT_RESERVE:
        raise OSError('Private disk floor')
    head, binding, binding_sha = _runtime_binding()
    for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
        os.environ[key] = '1'
    os.environ['JAX_PLATFORMS'] = 'cpu'
    from learning.task_teacher_dev_deviation_v1.membership import token_hash
    from learning.task_pair_fresh_multi_v1.entry import owner_marker
    import psutil
    deadline = time.monotonic() + JOB['wall_seconds'] + 2 * JOB['cleanup_seconds']
    previous._write_new(PRIVATE / 'attempt.local.json', dict(
        schema='state-residual-attempt-v1', commit=head, retry_allowed=False))
    try:
        process = psutil.Process(os.getpid())
        owner = previous._document(owner_marker(commit=head, binding_sha=binding_sha, pid=os.getpid(),
            created_us=int(process.create_time() * 1_000_000), deadline=deadline))
        previous._write_new(PRIVATE / 'owner.local.json', owner)
        job_name = 'Local\\TeacherDeviation-' + secrets.token_hex(16)
        previous._write_new(PRIVATE / 'claim.local.json', dict(schema='state-residual-claim-v1',
            commit=head, attempt=1, freeze_sha256=binding_sha, job_token_sha256=token_hash(job_name),
            retry_allowed=False))
        command = [sys.executable, '-B', '-m', 'learning.task_state_residual_v1.experiment', 'child']
        job = supervisor._supervise_owned(command, str(REPO / 'kaggriculture'), JOB, job_name)
        previous._write_new(PRIVATE / 'job.local.json', job)
        if not transfer._job_ok(job):
            raise RuntimeError('Owned Job did not close cleanly')
        guard = lambda: previous._guard(deadline)
        result = _validated_outputs(head, binding, guard)
        if _registered_binding(head)[1] != binding_sha:
            raise ValueError('Registration changed')
        names = ('worker-result','coverage','job','attempt','owner','claim','child-attempt')
        receipts = {name: _sha(_read(PRIVATE / (name + '.local.json'))) for name in names}
        previous._write_new(PRIVATE / 'closed.local.json', dict(schema='state-residual-closed-v1',
            registration_commit=head, binding_sha256=binding_sha, receipts=receipts,
            baseline_sha256=result['baseline_sha256'], params_sha256=result['params_sha256'], retry_allowed=False))
        print(json.dumps(dict(status='completed', screen=result['screen']['screen']), sort_keys=True), flush=True)
    except BaseException as exc:
        previous._write_new(PRIVATE / 'failed.local.json', dict(
            status='failed', category=type(exc).__name__, retry_allowed=False))
        raise


def load_completed(*, guard=lambda: None):
    """Authenticate a technical result; never authorize deployment or games."""
    if (PRIVATE / 'failed.local.json').exists() or (PRIVATE / 'worker-failed.local.json').exists():
        raise ValueError('Failed attempt cannot be consumed')
    closed = previous._document(_read(PRIVATE / 'closed.local.json'))
    head = closed.get('registration_commit')
    binding, binding_sha = _registered_binding(head)
    names = ('worker-result','coverage','job','attempt','owner','claim','child-attempt')
    if (set(closed) != {'schema','registration_commit','binding_sha256','receipts',
                        'baseline_sha256','params_sha256','retry_allowed'}
            or closed['schema'] != 'state-residual-closed-v1' or closed['binding_sha256'] != binding_sha
            or closed['retry_allowed'] is not False or type(closed['receipts']) is not dict
            or set(closed['receipts']) != set(names)):
        raise ValueError('Completion seal differs')
    for name in names:
        guard()
        if _sha(_read(PRIVATE / (name + '.local.json'))) != closed['receipts'][name]:
            raise ValueError('Completion receipt differs')
    if not transfer._job_ok(previous._document(_read(PRIVATE / 'job.local.json'))):
        raise ValueError('Owned Job receipt differs')
    attempt = previous._document(_read(PRIVATE / 'attempt.local.json'))
    marker = previous._document(_read(PRIVATE / 'child-attempt.local.json'))
    claim = previous._document(_read(PRIVATE / 'claim.local.json'))
    owner = previous._document(_read(PRIVATE / 'owner.local.json'))
    if (attempt != dict(schema='state-residual-attempt-v1', commit=head, retry_allowed=False)
            or marker != dict(schema='state-residual-child-attempt-v1', commit=head, retry_allowed=False)
            or claim.get('schema') != 'state-residual-claim-v1' or claim.get('commit') != head
            or type(claim.get('attempt')) is not int or claim['attempt'] != 1
            or claim.get('freeze_sha256') != binding_sha or claim.get('retry_allowed') is not False
            or type(claim.get('job_token_sha256')) is not str
            or re.fullmatch(r'[0-9a-f]{64}', claim['job_token_sha256']) is None
            or type(owner) is not dict or set(owner) != {'schema','registration_commit','binding_sha256',
                                                         'pid','created_us','deadline'}
            or owner.get('schema') != 'task-pair-fresh-multi-owner-v1'
            or owner.get('registration_commit') != head or owner.get('binding_sha256') != binding_sha
            or type(owner.get('pid')) is not int or owner['pid'] <= 0
            or type(owner.get('created_us')) is not int or owner['created_us'] <= 0
            or type(owner.get('deadline')) not in (int,float) or not math.isfinite(owner['deadline'])):
        raise ValueError('Completion ownership differs')
    result = _validated_outputs(head, binding, guard)
    if (closed['baseline_sha256'] != result['baseline_sha256']
            or closed['params_sha256'] != result['params_sha256']):
        raise ValueError('Completion model identity differs')
    return result


class _PrivateArgumentParser(argparse.ArgumentParser):
    def error(self, _message):
        raise ValueError('Invalid command')


def main():
    parser = _PrivateArgumentParser()
    parser.add_argument('command', choices=('prepare','run','child'))
    args = parser.parse_args()
    {'prepare':prepare, 'run':run, 'child':child}[args.command]()


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        category = ('memory' if isinstance(exc, MemoryError) else 'timeout' if isinstance(exc, TimeoutError)
                    else 'io' if isinstance(exc, OSError) else 'invalid_or_runtime')
        print(json.dumps(dict(status='failed', category=category)), file=sys.stderr)
        raise SystemExit(1) from None
