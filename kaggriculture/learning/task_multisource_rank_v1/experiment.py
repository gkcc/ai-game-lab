"""One CPU-only, preregistered cross-operation teacher-ranking experiment.

This is a different research question from the failed WATER-only GPU attempt.
The registered sources are already-consumed data; no game, holdout inventory,
or default bot is opened. Private model and sample data never enter Git.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import secrets
import shutil
import subprocess
import sys
import time

from learning.task_student_continuation_v1 import experiment as previous
from learning.task_multi_teacher_probe_v1 import experiment as supervisor
from learning.task_teacher_transfer_dataset_v1 import experiment as transfer
from learning.task_teacher_cross_source_v1 import experiment as cross


REPO = previous.REPO
PRIVATE = REPO.parent / 'private' / 'task-multisource-rank-v1-20261031'
REGISTRATION = REPO / 'review/evidence/task-multisource-rank-v1-20261031/registration.json'
BASE_COMMIT = '73c776e690e3a5f1a4877f11dabb126938fc9a08'
CROSS_COMMIT = 'a816e60d02c8b793a8b0e27634db2282bbc23b44'
JOB = dict(supervisor.JOB)
INITIAL_MEMORY, MIN_DISK = 5 * 1024**3, 4 * 1024**3
TRAIN_PER_OLD, DIAG_PER_OLD = 256, 128
SOURCE_NAMES = tuple(dict.fromkeys(cross.SOURCES + (
    'kaggriculture/learning/task_multisource_rank_v1/__init__.py',
    'kaggriculture/learning/task_multisource_rank_v1/core.py',
    'kaggriculture/learning/task_multisource_rank_v1/dedup.py',
    'kaggriculture/learning/task_multisource_rank_v1/evaluate.py',
    'kaggriculture/learning/task_multisource_rank_v1/experiment.py',
    'kaggriculture/learning/task_pair_checkpoint_v1/snapshot.py',
    'kaggriculture/learning/task_water_residual_v1/policy.py',
    'kaggriculture/learning/task_water_residual_v1/fit.py',
)))


def _git(*args):
    return subprocess.check_output(['git', '-C', str(REPO), *args],
                                   stderr=subprocess.DEVNULL, timeout=30).decode('utf-8').strip()


def _read(path, limit=1024**2):
    return cross._read(path, limit=limit)


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _source_hashes():
    return {name: _sha(_read(REPO / name)) for name in SOURCE_NAMES}


def _indices(name, count):
    """Index-only fixed selection, determined without reading group contents."""
    if name not in ('old', 'new') or type(count) is not int or count < TRAIN_PER_OLD + DIAG_PER_OLD:
        raise ValueError('Invalid old source population')
    domain = ('kaggriculture-multisource-rank-v1/' + name + '/').encode('ascii')
    ordered = sorted(range(count), key=lambda i: (hashlib.sha256(domain + str(i).encode('ascii')).digest(), i))
    train = frozenset(ordered[:TRAIN_PER_OLD])
    diag = frozenset(ordered[TRAIN_PER_OLD:TRAIN_PER_OLD + DIAG_PER_OLD])
    if len(train) != TRAIN_PER_OLD or len(diag) != DIAG_PER_OLD or train & diag:
        raise ValueError('Old source index split differs')
    return train, diag


def _registration(source_sha, binding_sha):
    return dict(schema='task-multisource-rank-prereg-v1', base_commit=BASE_COMMIT,
        cross_source_registration_commit=CROSS_COMMIT, source_sha256=source_sha,
        private_binding_sha256=binding_sha, device='cpu', max_attempts=1,
        old_sources=['old', 'new'], old_train_per_source=TRAIN_PER_OLD,
        old_diagnostic_per_source=DIAG_PER_OLD, selection='sha256_source_and_group_index',
        selection_domain='kaggriculture-multisource-rank-v1/<source>/',
        optimizer_rng_domain='kaggriculture-multisource-rank-v1/optimizer',
        optimizer_rng_encoding='sha256_first_four_bytes_little_endian',
        new_train_groups=59, new_dev_groups=62, updates=64,
        dedup='raw_and_normalized_exact_bytes_transitive_conflicts_removed_train_priority',
        minimum_candidates_per_group=2,
        require_normalized_candidate_variation=True,
        minimum_after_dedup=dict(old_train=16, new_train=16, old_diag=32, new_dev=32,
                                 each_new_dev_role=8),
        per_batch_old=8, per_batch_new=8, learning_rate=0.0001,
        gradient_clip=1.0, kl_weight=0.1,
        model='frozen_task_logits_plus_cross_operation_linear_residual',
        train_first_choice_float32_stability_required=True,
        parity_atol=0.0001, parity_rtol=0.00001,
        screening=dict(new_dev_net_correct_at_least=2, each_new_role_no_drop=True,
                       old_non_water_minimum_groups=20, old_non_water_max_drop_fraction=0.05,
                       at_least_one_changed_choice=True),
        memory_initial_bytes=INITIAL_MEMORY, disk_floor_bytes=MIN_DISK, job=dict(JOB),
        independent_evaluation=False, new_games=0, default_bot_modified=False)


def prepare():
    if (os.name != 'nt' or _git('rev-parse', 'HEAD') != BASE_COMMIT
            or _git('diff', '--name-only') or _git('diff', '--cached', '--name-only')
            or REGISTRATION.exists() or PRIVATE.exists()):
        raise ValueError('Unexpected multisource registration state')
    if previous.physical_available() < INITIAL_MEMORY:
        raise MemoryError('Initial multisource memory floor not met')
    deadline = time.monotonic() + 300
    guard = lambda: previous._guard(deadline)
    completed = cross.load_completed()
    if completed['registration_commit'] != CROSS_COMMIT or completed['passed'] is not True:
        raise ValueError('Cross-source comparison not authenticated')
    earlier, _ = cross._registered_binding(CROSS_COMMIT)
    train, dev, contract = transfer.load_trainable(guard=guard)
    if (len(train), len(dev), contract['registration_commit']) != (59, 62, cross.DATA_COMMIT):
        raise ValueError('New data identity differs')
    from learning.task_student_anchor_collection_v1 import sources
    context = sources.open_context(guard=guard)
    _mean, _scale, metadata = context.model_normalization()
    semantic = cross._schema_identity(earlier['historical']['schema_sha256'], metadata)
    if (contract['train_sha256'] != earlier['train_sha256']
            or contract['dev_sha256'] != earlier['dev_sha256']
            or context.task_sha256 != earlier['baseline_sha256'] or semantic != earlier['schema_sha256']):
        raise ValueError('Multisource identities differ')
    binding = dict(schema='task-multisource-rank-private-v1', salt=secrets.token_hex(32),
                   earlier=earlier, cross_commit=CROSS_COMMIT)
    binding_raw = previous._encoded(binding)
    registration = _registration(_source_hashes(), _sha(binding_raw))
    PRIVATE.mkdir(parents=False, exist_ok=False)
    previous._write_new(PRIVATE / 'binding.local.json', binding)
    REGISTRATION.parent.mkdir(parents=True, exist_ok=True)
    previous._write_new(REGISTRATION, registration)
    print('Private source identities and CPU experiment registration prepared.')


def _registered_binding(head):
    if type(head) is not str or re.fullmatch(r'[0-9a-f]{40}', head) is None:
        raise ValueError('Invalid registration commit')
    parent = _git('rev-parse', head + '^')
    raw = subprocess.check_output(['git', '-C', str(REPO), 'show',
        head + ':' + REGISTRATION.relative_to(REPO).as_posix()], stderr=subprocess.DEVNULL, timeout=30)
    binding_raw = _read(PRIVATE / 'binding.local.json')
    registration = previous._document(raw)
    if parent != BASE_COMMIT or registration != _registration(_source_hashes(), _sha(binding_raw)):
        raise ValueError('Multisource source or registration identity differs')
    binding = previous._document(binding_raw)
    earlier, _ = cross._registered_binding(CROSS_COMMIT)
    if (binding.get('schema') != 'task-multisource-rank-private-v1'
            or binding.get('cross_commit') != CROSS_COMMIT or binding.get('earlier') != earlier):
        raise ValueError('Multisource private source binding differs')
    return binding, _sha(binding_raw)


def _runtime_binding():
    head = _git('rev-parse', 'HEAD')
    if _git('status', '--porcelain') or _git('rev-parse', 'origin/main') != head:
        raise ValueError('Multisource registration not clean and published')
    binding, digest = _registered_binding(head)
    raw = subprocess.check_output(['git', '-C', str(REPO), 'show',
        head + ':' + REGISTRATION.relative_to(REPO).as_posix()], stderr=subprocess.DEVNULL, timeout=30)
    if _read(REGISTRATION) != raw:
        raise ValueError('Multisource registration file differs')
    return head, binding, digest


def _collect_old(binding, guard):
    """Retain only the fixed 384 index-selected groups per source."""
    from learning.task_teacher_cross_source_v1 import archive, overlap
    train, diagnostic, roles = [], [], []
    for name, role in (('old', 'T01'), ('new', 'T02')):
        selected, measured = _indices(name, cross.GROUPS[name])
        seen = set()
        def consume(index, shared, candidates, target):
            if index in selected or index in measured:
                guard()
                group = overlap.OldGroup(shared, candidates, target)
                if index in selected:
                    train.append(group)
                else:
                    diagnostic.append(group)
                    roles.append(role)
                seen.add(index)
        receipt = archive.scan_archive(cross.OLD_PATHS[name],
            expected_sha256=binding['earlier']['historical']['archive_sha256'][name],
            expected_groups=cross.GROUPS[name], expected_candidates=cross.CANDIDATES[name],
            guard=guard, consume=consume)
        if (receipt != dict(groups=cross.GROUPS[name], candidates=cross.CANDIDATES[name],
                            verified_after_scan=True) or seen != selected | measured):
            raise ValueError('Selected old groups or archive seal differs')
    if len(train) != 512 or len(diagnostic) != 256 or len(roles) != 256:
        raise ValueError('Old selected population differs')
    return train, diagnostic, roles


def _model(context, binding, guard):
    from learning.task_bc_v1.model import TaskNetwork
    from learning.task_pair_checkpoint_v1 import snapshot
    guard()
    raw = _read(context._task_path, limit=snapshot.MAX_ARCHIVE_BYTES)
    checkpoint = snapshot.load_checkpoint(raw, expected_sha256=binding['earlier']['baseline_sha256'],
                                          expected_schema_sha256=binding['earlier']['schema_sha256'])
    if context.task_sha256 != binding['earlier']['baseline_sha256']:
        raise ValueError('Frozen task model differs')
    path = PRIVATE / 'baseline.local.npz'
    with path.open('xb') as output:
        output.write(raw)
        output.flush()
        os.fsync(output.fileno())
    if _read(path, limit=snapshot.MAX_ARCHIVE_BYTES) != raw:
        raise ValueError('Frozen model snapshot differs')
    network = TaskNetwork(path, context.task_sha256)
    import numpy as np
    if (cross._schema_identity(binding['earlier']['historical']['schema_sha256'], network.metadata)
            != binding['earlier']['schema_sha256']
            or not np.array_equal(network.mean, checkpoint.mean)
            or not np.array_equal(network.scale, checkpoint.scale)
            or any(not np.array_equal(network.params[name], checkpoint.params[name]) for name in checkpoint.params)):
        raise ValueError('Frozen model snapshot semantics differ')
    return network


def _group(shared, candidates, target, source, network):
    import numpy as np
    from . import core
    scores = network.score(shared, candidates)
    if (type(scores) is not np.ndarray or scores.dtype != np.dtype('float64')
            or scores.shape != (len(candidates),) or not np.isfinite(scores).all()):
        raise ValueError('Frozen task scores differ')
    return core.Group(shared, candidates, scores, target, source)


def _new_group(group, network):
    raw = group.features
    return _group(raw[0, :275].copy(), raw[:, 275:].copy(), group.target_index, 'new', network)


def _deduplicate(old_train, old_diag, old_roles, new_train, new_dev, mean, scale, guard, on_report=None):
    from . import dedup
    import numpy as np
    def snapshot(array):
        if type(array) is not np.ndarray or array.dtype != np.dtype('float32'):
            raise ValueError('Multisource snapshot dtype differs')
        return np.frombuffer(array.tobytes(order='C'), dtype=np.float32).reshape(array.shape)
    items = []
    for index, item in enumerate(old_train):
        items.append(dedup.Item(snapshot(item.shared), snapshot(item.candidates), item.target_index, 'old_train',
                                'T01' if index < TRAIN_PER_OLD else 'T02'))
    for item in new_train:
        raw = item.features
        items.append(dedup.Item(snapshot(raw[0, :275]), snapshot(raw[:, 275:]), item.target_index,
                                'new_train', item.role))
    for item, role in zip(old_diag, old_roles):
        items.append(dedup.Item(snapshot(item.shared), snapshot(item.candidates), item.target_index, 'old_diag', role))
    for item in new_dev:
        raw = item.features
        items.append(dedup.Item(snapshot(raw[0, :275]), snapshot(raw[:, 275:]), item.target_index,
                                'new_dev', item.role))
    if len(items) != 889:
        raise ValueError('Pre-dedup population differs')
    kept, report = dedup.deduplicate(items, mean=mean, scale=scale, guard=guard)
    pool_names = ('old_train', 'new_train', 'old_diag', 'new_dev')
    pools = {key: [] for key in pool_names}
    single = {key: 0 for key in pool_names}
    constant = {key: 0 for key in pool_names}
    for item in kept:
        guard()
        if len(item.candidates) == 1:
            single[item.pool] += 1
            continue
        with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
            local = ((item.candidates.astype(np.float64) - mean[275:]) / scale[275:]).astype(np.float32)
        if not np.isfinite(local).all():
            raise ValueError('Nonfinite normalized candidate features')
        if np.array_equal(local, np.broadcast_to(local[:1], local.shape)):
            constant[item.pool] += 1
            continue
        pools[item.pool].append(item)
    report['single_candidate_excluded_by_pool'] = single
    report['constant_normalized_excluded_by_pool'] = constant
    report['eligible_by_pool'] = {key: len(pools[key]) for key in pool_names}
    role_vocabulary = {'old_train': ('T01', 'T02'), 'new_train': ('A01', 'A02'),
                       'old_diag': ('T01', 'T02'), 'new_dev': ('A07', 'A08')}
    if any(item.role not in role_vocabulary[key] for key in pool_names for item in pools[key]):
        raise ValueError('Unexpected multisource role')
    report['eligible_by_role'] = {
        key: {role: sum(item.role == role for item in pools[key]) for role in role_vocabulary[key]}
        for key in pool_names}
    if on_report is not None:
        if not callable(on_report):
            raise ValueError('Invalid multisource coverage reporter')
        on_report(report)
    minima = {'old_train': 16, 'new_train': 16, 'old_diag': 32, 'new_dev': 32}
    if (any(len(pools[key]) < count for key, count in minima.items())
            or any(sum(item.role == role for item in pools['new_dev']) < 8 for role in ('A07', 'A08'))):
        raise ValueError('Multisource post-dedup coverage insufficient')
    return pools, report


def _selected_groups(pools, network, guard):
    result = {}
    for pool in ('old_train', 'new_train', 'old_diag', 'new_dev'):
        groups = []
        for item in pools[pool]:
            guard()
            groups.append(_group(item.shared, item.candidates, item.target,
                                 'new' if pool.startswith('new') else 'old', network))
        result[pool] = groups
    return result


def _baseline_cast_report(groups, guard):
    """Count training precision hazards without exposing model logits."""
    import numpy as np
    counts = {source: dict(groups=0, nonfinite=0, argmax_changed=0) for source in ('old', 'new')}
    for item in groups:
        guard()
        if item.source not in counts:
            raise ValueError('Invalid training source')
        bucket = counts[item.source]
        bucket['groups'] += 1
        with np.errstate(over='ignore', invalid='ignore'):
            cast = item.baseline.astype(np.float32)
        if not np.isfinite(cast).all():
            bucket['nonfinite'] += 1
        elif int(np.argmax(item.baseline)) != int(np.argmax(cast)):
            bucket['argmax_changed'] += 1
    return counts


def _validate_cast_report(value, dedup):
    if type(value) is not dict or set(value) != {'old', 'new'}:
        raise ValueError('Multisource baseline cast vocabulary differs')
    for source, pool in (('old', 'old_train'), ('new', 'new_train')):
        row = value[source]
        if (type(row) is not dict or set(row) != {'groups', 'nonfinite', 'argmax_changed'}
                or any(type(row[key]) is not int or row[key] < 0 for key in row)
                or row['groups'] != dedup['eligible_by_pool'][pool]
                or row['nonfinite'] != 0 or row['argmax_changed'] != 0):
            raise ValueError('Multisource baseline cast report differs')


def _parity(groups, *, mean, scale, weights, device, guard):
    """Compare centered JAX CPU fitting scores with deployed NumPy choices."""
    import numpy as np
    import jax
    import jax.numpy as jnp
    from . import core
    if device.platform != 'cpu' or not groups:
        raise ValueError('CPU parity device or population differs')
    centered = jax.jit(lambda x, w: jnp.sum((x - x[:1]) * w[None, :], axis=1))
    max_error = 0.0
    zero = np.zeros(core.DIM, dtype=np.float32)
    with jax.default_device(device), jax.default_matmul_precision('highest'):
        placed_weights = jax.device_put(weights, device)
        for group in groups:
            guard()
            if core.select(group, mean, scale, zero) != int(np.argmax(group.baseline)):
                raise ValueError('Zero residual does not preserve baseline')
            raw = np.concatenate((np.broadcast_to(group.shared, (len(group.candidates), 275)),
                                  group.candidates), axis=1)
            normalized = ((raw.astype(np.float64) - mean) / scale).astype(np.float32)
            if not np.isfinite(normalized).all():
                raise ValueError('Nonfinite CPU parity input')
            padded = np.broadcast_to(normalized[:1], (256, core.DIM)).copy()
            padded[:len(normalized)] = normalized
            placed = jax.device_put(padded, device)
            prediction = centered(placed, placed_weights)
            if placed.devices() != {device} or placed_weights.devices() != {device} or prediction.devices() != {device}:
                raise ValueError('CPU parity left the registered device')
            prediction.block_until_ready()
            actual = np.asarray(jax.device_get(prediction))[:len(normalized)].astype(np.float64)
            numpy_residual = normalized @ weights
            expected = numpy_residual.astype(np.float64) - np.float64(numpy_residual[0])
            if not np.isfinite(actual).all() or not np.isfinite(expected).all():
                raise ValueError('Nonfinite CPU parity result')
            max_error = max(max_error, float(np.max(np.abs(actual - expected))))
            if (not np.allclose(actual, expected, atol=1e-4, rtol=1e-5)
                    or int(np.argmax(group.baseline + actual)) != core.select(group, mean, scale, weights)):
                raise ValueError('Training and deployment choices differ')
    return dict(groups=len(groups), allclose=True, adjusted_argmax_mismatches=0,
                zero_residual_mismatches=0, maximum_absolute_error=max_error)


def _screen(new_dev, old_diag):
    """Predeclared descriptive gates; neither input is a fresh match."""
    from learning.task_choice_v1.features import OPS
    new = new_dev['overall']
    role_checks = []
    for role in ('A07', 'A08'):
        bucket = new_dev['by_role'][role]['overall']
        role_checks.append((bucket['residual_correct'] >= bucket['baseline_correct']) if bucket['groups'] else None)
    roles_ok = False if False in role_checks else None if None in role_checks else True
    nonwater = [old_diag['by_target_operation'][op] for op in OPS if op != 'WATER']
    count = sum(bucket['groups'] for bucket in nonwater)
    old_base = sum(bucket['baseline_correct'] for bucket in nonwater)
    old_new = sum(bucket['residual_correct'] for bucket in nonwater)
    checks = dict(new_dev_net_correct_at_least_two=new['residual_correct'] - new['baseline_correct'] >= 2,
                  each_new_dev_role_no_accuracy_drop=roles_ok,
                  old_non_water_coverage_at_least_twenty=count >= 20,
                  old_non_water_drop_at_most_five_points=(old_new - old_base >= -0.05 * count) if count >= 20 else None,
                  new_dev_at_least_one_change=new['changed'] > 0)
    failed = any(value is False for key, value in checks.items()
                 if key != 'old_non_water_coverage_at_least_twenty')
    incomplete = (not checks['old_non_water_coverage_at_least_twenty']
                  or any(value is None for value in checks.values()))
    return dict(checks=checks, screen='failed' if failed else 'unknown' if incomplete else 'passed',
                old_non_water_groups=count, old_non_water_baseline_correct=old_base,
                old_non_water_residual_correct=old_new)


def _write_weights(weights):
    import io
    import numpy as np
    if (type(weights) is not np.ndarray or weights.dtype != np.dtype('float32')
            or weights.shape != (373,) or not np.isfinite(weights).all()):
        raise ValueError('Residual weights differ')
    path = PRIVATE / 'residual.local.npy'
    with path.open('xb') as output:
        np.save(output, weights, allow_pickle=False)
        output.flush()
        os.fsync(output.fileno())
    raw = _read(path)
    restored = np.load(io.BytesIO(raw), allow_pickle=False)
    if (type(restored) is not np.ndarray or restored.dtype != np.dtype('float32')
            or restored.shape != (373,) or not np.array_equal(restored, weights)):
        raise ValueError('Residual snapshot did not round trip')
    return restored, _sha(raw)


def _validate_summary(value, roles, source):
    """Check aggregate accounting before allowing a sealed screen result."""
    import math
    from learning.task_choice_v1.features import OPS
    names = ('groups', 'baseline_correct', 'residual_correct', 'changed', 'op_flips', 'improved', 'regressed')
    fields = set(names) | {'coverage', 'baseline_nll', 'residual_nll'}
    if (type(value) is not dict or set(value) != {'overall', 'by_target_operation', 'by_source',
                                                  'by_role', 'independent_evaluation'}
            or value['independent_evaluation'] is not False or set(value['by_target_operation']) != set(OPS)
            or set(value['by_source']) != {'old', 'new'} or set(value['by_role']) != set(roles)):
        raise ValueError('Multisource summary vocabulary differs')
    def bucket(item):
        if (type(item) is not dict or set(item) != fields
                or any(type(item[name]) is not int or item[name] < 0 for name in names)
                or any(item[name] > item['groups'] for name in names if name != 'groups')
                or item['op_flips'] > item['changed'] or item['improved'] + item['regressed'] > item['changed']
                or item['residual_correct'] - item['baseline_correct'] != item['improved'] - item['regressed']
                or item['coverage'] != ('observed' if item['groups'] else 'unknown')):
            raise ValueError('Multisource aggregate counts differ')
        for key in ('baseline_nll', 'residual_nll'):
            loss = item[key]
            if not item['groups'] and loss is not None:
                raise ValueError('Multisource aggregate loss differs')
            if item['groups'] and (type(loss) not in (int, float) or not math.isfinite(loss) or loss < -1e-10):
                raise ValueError('Multisource aggregate loss differs')
    def partition(total, pieces):
        bucket(total)
        for part in pieces:
            bucket(part)
        if any(total[name] != sum(part[name] for part in pieces) for name in names):
            raise ValueError('Multisource aggregate partition differs')
        for key in ('baseline_nll', 'residual_nll'):
            if total['groups']:
                combined = math.fsum(part[key] * (part['groups'] / total['groups'])
                                     for part in pieces if part['groups'])
                if not math.isclose(total[key], combined, rel_tol=1e-10, abs_tol=1e-10):
                    raise ValueError('Multisource aggregate loss partition differs')
    overall = value['overall']
    partition(overall, list(value['by_target_operation'].values()))
    partition(overall, list(value['by_source'].values()))
    if value['by_source'][source]['groups'] != overall['groups']:
        raise ValueError('Multisource diagnostic source differs')
    role_parts = []
    for role in roles:
        item = value['by_role'][role]
        if type(item) is not dict or set(item) != {'overall', 'by_target_operation'} or set(item['by_target_operation']) != set(OPS):
            raise ValueError('Multisource role vocabulary differs')
        partition(item['overall'], list(item['by_target_operation'].values()))
        role_parts.append(item['overall'])
    partition(overall, role_parts)
    for op in OPS:
        partition(value['by_target_operation'][op],
                  [value['by_role'][role]['by_target_operation'][op] for role in roles])
    return overall['groups']


def _result(raw, head):
    value = previous._document(raw)
    fields = {'schema', 'registration_commit', 'new_dev', 'old_diag', 'dedup', 'parity',
              'screen', 'weights_sha256', 'baseline_sha256', 'updates_completed',
              'nonzero_gradient_seen', 'weights_changed', 'device', 'new_games',
              'independent_evaluation', 'software'}
    if (set(value) != fields or value.get('schema') != 'task-multisource-rank-result-v1'
            or value.get('registration_commit') != head or value.get('device') != 'cpu'
            or type(value.get('new_games')) is not int or value['new_games'] != 0
            or value.get('independent_evaluation') is not False
            or type(value.get('updates_completed')) is not int or value['updates_completed'] != 64
            or value.get('nonzero_gradient_seen') is not True or value.get('weights_changed') is not True
            or value.get('screen') != _screen(value['new_dev'], value['old_diag'])
            or type(value.get('parity')) is not dict or value['parity'].get('allclose') is not True
            or value['parity'].get('adjusted_argmax_mismatches') != 0
            or value['parity'].get('zero_residual_mismatches') != 0
            or any(type(value.get(key)) is not str or re.fullmatch(r'[0-9a-f]{64}', value[key]) is None
                   for key in ('weights_sha256', 'baseline_sha256'))):
        raise ValueError('Multisource result contract differs')
    old_count = _validate_summary(value['old_diag'], ('T01', 'T02'), 'old')
    new_count = _validate_summary(value['new_dev'], ('A07', 'A08'), 'new')
    dedup = value['dedup']
    pools = ('old_train', 'new_train', 'old_diag', 'new_dev')
    pool_maps = ('input_by_pool', 'retained_by_pool', 'duplicate_removed_by_pool',
                 'conflict_removed_by_pool', 'single_candidate_excluded_by_pool',
                 'constant_normalized_excluded_by_pool', 'eligible_by_pool')
    minima = {'old_train': 16, 'new_train': 16, 'old_diag': 32, 'new_dev': 32}
    role_vocabulary = {'old_train': ('T01', 'T02'), 'new_train': ('A01', 'A02'),
                       'old_diag': ('T01', 'T02'), 'new_dev': ('A07', 'A08')}
    if (type(dedup) is not dict or type(dedup.get('input_count')) is not int or dedup['input_count'] != 889
            or any(type(dedup.get(key)) is not dict or set(dedup[key]) != set(pools)
                   or any(type(dedup[key][pool]) is not int or dedup[key][pool] < 0 for pool in pools)
                   for key in pool_maps)
            or dedup['input_by_pool'] != {'old_train': 512, 'new_train': 59, 'old_diag': 256, 'new_dev': 62}
            or sum(dedup['retained_by_pool'].values()) != dedup.get('retained_count')
            or any(dedup['input_by_pool'][pool] != dedup['retained_by_pool'][pool]
                   + dedup['duplicate_removed_by_pool'][pool] + dedup['conflict_removed_by_pool'][pool]
                   or dedup['retained_by_pool'][pool] != dedup['eligible_by_pool'][pool]
                   + dedup['single_candidate_excluded_by_pool'][pool]
                   + dedup['constant_normalized_excluded_by_pool'][pool] for pool in pools)
            or any(dedup['eligible_by_pool'][pool] < minimum for pool, minimum in minima.items())
            or type(dedup.get('eligible_by_role')) is not dict or set(dedup['eligible_by_role']) != set(pools)
            or any(type(dedup['eligible_by_role'][pool]) is not dict
                   or set(dedup['eligible_by_role'][pool]) != set(role_vocabulary[pool])
                   or any(type(count) is not int or count < 0 for count in dedup['eligible_by_role'][pool].values())
                   or sum(dedup['eligible_by_role'][pool].values()) != dedup['eligible_by_pool'][pool]
                   for pool in pools)
            or any(dedup['eligible_by_role']['old_diag'][role] != value['old_diag']['by_role'][role]['overall']['groups']
                   for role in ('T01', 'T02'))
            or any(dedup['eligible_by_role']['new_dev'][role] != value['new_dev']['by_role'][role]['overall']['groups']
                   for role in ('A07', 'A08'))
            or any(value['new_dev']['by_role'][role]['overall']['groups'] < 8 for role in ('A07', 'A08'))
            or dedup['eligible_by_pool'].get('old_diag') != old_count
            or dedup['eligible_by_pool'].get('new_dev') != new_count
            or type(value['parity'].get('groups')) is not int
            or value['parity']['groups'] != old_count + new_count):
        raise ValueError('Multisource retained counts differ')
    return value


def child():
    head, binding, binding_sha = _runtime_binding()
    claim = previous._document(_read(PRIVATE / 'claim.local.json'))
    attempt = previous._document(_read(PRIVATE / 'attempt.local.json'))
    if (claim.get('schema') != 'task-multisource-rank-claim-v1' or claim.get('commit') != head
            or type(claim.get('attempt')) is not int or claim['attempt'] != 1
            or claim.get('freeze_sha256') != binding_sha or claim.get('retry_allowed') is not False
            or attempt != dict(schema='task-multisource-rank-attempt-v1', commit=head, retry_allowed=False)
            or (PRIVATE / 'failed.local.json').exists() or (PRIVATE / 'closed.local.json').exists()):
        raise ValueError('Multisource child is not uniquely claimed')
    from learning.task_teacher_dev_deviation_v1.membership import ENV_KEY, verify_worker
    from learning.task_pair_fresh_multi_v1.entry import verify_live_owner
    import psutil
    owner = previous._document(_read(PRIVATE / 'owner.local.json'))
    verify_worker(os.environ.get(ENV_KEY), claim, binding_sha)
    verify_live_owner(owner, commit=head, binding_sha=binding_sha, current_pid=os.getpid(),
                      process=psutil.Process, executable=sys.executable, samefile=os.path.samefile,
                      now=time.monotonic())
    previous._write_new(PRIVATE / 'child-attempt.local.json', dict(
        schema='task-multisource-rank-child-attempt-v1', commit=head, retry_allowed=False))
    deadline = time.monotonic() + JOB['wall_seconds']
    guard = lambda: previous._guard(deadline)
    phase = 'source'
    try:
        from learning.task_student_anchor_collection_v1 import sources
        from . import core, evaluate
        import numpy as np
        train, dev, contract = transfer.load_trainable(guard=guard)
        if ((len(train), len(dev), contract['registration_commit']) != (59, 62, cross.DATA_COMMIT)
                or contract['train_sha256'] != binding['earlier']['train_sha256']
                or contract['dev_sha256'] != binding['earlier']['dev_sha256']):
            raise ValueError('Authenticated new sample differs')
        context = sources.open_context(guard=guard)
        mean, scale, metadata = context.model_normalization()
        if (context.task_sha256 != binding['earlier']['baseline_sha256']
                or cross._schema_identity(binding['earlier']['historical']['schema_sha256'], metadata)
                    != binding['earlier']['schema_sha256']):
            raise ValueError('Authenticated baseline differs')
        old_train, old_diag, old_roles = _collect_old(binding, guard)
        phase = 'dedup'
        pools, dedup_report = _deduplicate(old_train, old_diag, old_roles, train, dev, mean, scale, guard,
            on_report=lambda report: previous._write_new(PRIVATE / 'coverage.local.json', report))
        del old_train, old_diag, old_roles, train, dev
        phase = 'baseline_forward'
        network = _model(context, binding, guard)
        groups = _selected_groups(pools, network, guard)
        training_groups = groups['old_train'] + groups['new_train']
        phase = 'baseline_cast'
        cast_report = _baseline_cast_report(training_groups, guard)
        previous._write_new(PRIVATE / 'baseline-cast.local.json', cast_report)
        if any(item['nonfinite'] or item['argmax_changed'] for item in cast_report.values()):
            raise ValueError('Training baseline float32 cast changed a first choice')
        phase = 'cpu_training'
        import jax
        import optax
        devices = jax.devices('cpu')
        if len(devices) != 1 or devices[0].platform != 'cpu':
            raise ValueError('Registered CPU backend unavailable')
        rng_seed = int.from_bytes(hashlib.sha256(b'kaggriculture-multisource-rank-v1/optimizer').digest()[:4], 'little')
        trained = core.fit(training_groups, mean=mean, scale=scale, device=devices[0], guard=guard, rng_seed=rng_seed)
        if (trained.updates_completed != 64 or not trained.nonzero_gradient_seen or not trained.weights_changed):
            raise ValueError('CPU training did not update the residual')
        phase = 'assessment'
        weights, weights_sha = _write_weights(trained.weights)
        diagnostic_groups = groups['old_diag'] + groups['new_dev']
        parity = _parity(diagnostic_groups, mean=mean, scale=scale, weights=weights,
                         device=devices[0], guard=guard)
        old_roles = [item.role for item in pools['old_diag']]
        new_roles = [item.role for item in pools['new_dev']]
        old_summary = evaluate.assess(groups['old_diag'], mean, scale, weights, metadata,
                                      roles=old_roles, role_labels=('T01', 'T02'), guard=guard)
        new_summary = evaluate.assess(groups['new_dev'], mean, scale, weights, metadata,
                                      roles=new_roles, role_labels=('A07', 'A08'), guard=guard)
        screen = _screen(new_summary, old_summary)
        if _sha(_read(PRIVATE / 'baseline.local.npz', limit=2 * 1024**2)) != binding['earlier']['baseline_sha256']:
            raise ValueError('Frozen model snapshot changed')
        result = dict(schema='task-multisource-rank-result-v1', registration_commit=head,
            new_dev=new_summary, old_diag=old_summary, dedup=dedup_report, parity=parity, screen=screen,
            weights_sha256=weights_sha, baseline_sha256=context.task_sha256,
            updates_completed=trained.updates_completed, nonzero_gradient_seen=trained.nonzero_gradient_seen,
            weights_changed=trained.weights_changed, device='cpu', new_games=0, independent_evaluation=False,
            software=dict(python='.'.join(map(str, sys.version_info[:3])), numpy=np.__version__,
                          jax=jax.__version__, optax=optax.__version__))
        _result(previous._encoded(result), head)
        guard()
        previous._write_new(PRIVATE / 'worker-result.local.json', result)
    except BaseException as exc:
        previous._write_new(PRIVATE / 'worker-failed.local.json', dict(
            phase=phase, category=type(exc).__name__, retry_allowed=False))
        raise


def run():
    if os.name != 'nt' or previous.physical_available() < INITIAL_MEMORY:
        raise MemoryError('Multisource initial memory floor not met')
    if shutil.disk_usage(PRIVATE).free < MIN_DISK:
        raise OSError('Multisource disk floor not met')
    head, _binding, binding_sha = _runtime_binding()
    for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[key] = '1'
    os.environ['JAX_PLATFORMS'] = 'cpu'
    from learning.task_teacher_dev_deviation_v1.membership import token_hash
    from learning.task_pair_fresh_multi_v1.entry import owner_marker
    import psutil
    deadline = time.monotonic() + JOB['wall_seconds'] + 2 * JOB['cleanup_seconds']
    previous._write_new(PRIVATE / 'attempt.local.json', dict(
        schema='task-multisource-rank-attempt-v1', commit=head, retry_allowed=False))
    try:
        process = psutil.Process(os.getpid())
        owner = previous._document(owner_marker(commit=head, binding_sha=binding_sha, pid=os.getpid(),
            created_us=int(process.create_time() * 1_000_000), deadline=deadline))
        previous._write_new(PRIVATE / 'owner.local.json', owner)
        job_name = 'Local\\TeacherDeviation-' + secrets.token_hex(16)
        previous._write_new(PRIVATE / 'claim.local.json', dict(schema='task-multisource-rank-claim-v1',
            commit=head, attempt=1, freeze_sha256=binding_sha, job_token_sha256=token_hash(job_name),
            retry_allowed=False))
        command = [sys.executable, '-B', '-m', 'learning.task_multisource_rank_v1.experiment', 'child']
        job = supervisor._supervise_owned(command, str(REPO / 'kaggriculture'), JOB, job_name)
        previous._write_new(PRIVATE / 'job.local.json', job)
        if not transfer._job_ok(job):
            raise RuntimeError('Multisource owned Job did not close cleanly')
        worker_raw = _read(PRIVATE / 'worker-result.local.json')
        result = _result(worker_raw, head)
        coverage_raw = _read(PRIVATE / 'coverage.local.json')
        cast_raw = _read(PRIVATE / 'baseline-cast.local.json')
        if previous._document(coverage_raw) != result['dedup']:
            raise ValueError('Multisource coverage receipt differs')
        _validate_cast_report(previous._document(cast_raw), result['dedup'])
        weight_raw = _read(PRIVATE / 'residual.local.npy')
        baseline_raw = _read(PRIVATE / 'baseline.local.npz', limit=2 * 1024**2)
        if (_sha(weight_raw) != result['weights_sha256'] or _sha(baseline_raw) != result['baseline_sha256']):
            raise ValueError('Multisource artifact identities differ')
        closed = dict(schema='task-multisource-rank-closed-v1', registration_commit=head,
            worker_sha256=_sha(worker_raw), job_sha256=_sha(_read(PRIVATE / 'job.local.json')),
            attempt_sha256=_sha(_read(PRIVATE / 'attempt.local.json')),
            owner_sha256=_sha(_read(PRIVATE / 'owner.local.json')),
            claim_sha256=_sha(_read(PRIVATE / 'claim.local.json')),
            child_attempt_sha256=_sha(_read(PRIVATE / 'child-attempt.local.json')),
            coverage_sha256=_sha(coverage_raw), cast_sha256=_sha(cast_raw),
            binding_sha256=binding_sha, weights_sha256=_sha(weight_raw),
            baseline_sha256=_sha(baseline_raw), retry_allowed=False)
        previous._write_new(PRIVATE / 'closed.local.json', closed)
        print(json.dumps(dict(status='completed', screen=result['screen']['screen']), sort_keys=True), flush=True)
    except BaseException as exc:
        previous._write_new(PRIVATE / 'failed.local.json', dict(
            status='failed', category=type(exc).__name__, retry_allowed=False))
        raise


def load_completed():
    if (PRIVATE / 'failed.local.json').exists() or (PRIVATE / 'worker-failed.local.json').exists():
        raise ValueError('Failed multisource attempt cannot be consumed')
    closed = previous._document(_read(PRIVATE / 'closed.local.json'))
    head = closed.get('registration_commit')
    binding, binding_sha = _registered_binding(head)
    if (closed.get('schema') != 'task-multisource-rank-closed-v1'
            or closed.get('retry_allowed') is not False or closed.get('binding_sha256') != binding_sha):
        raise ValueError('Multisource closed seal differs')
    for name, key, limit in (
        ('worker-result.local.json', 'worker_sha256', 1024**2),
        ('job.local.json', 'job_sha256', 1024**2),
        ('attempt.local.json', 'attempt_sha256', 1024**2),
        ('owner.local.json', 'owner_sha256', 1024**2),
        ('claim.local.json', 'claim_sha256', 1024**2),
        ('child-attempt.local.json', 'child_attempt_sha256', 1024**2),
        ('coverage.local.json', 'coverage_sha256', 1024**2),
        ('baseline-cast.local.json', 'cast_sha256', 1024**2),
        ('residual.local.npy', 'weights_sha256', 1024**2),
        ('baseline.local.npz', 'baseline_sha256', 2 * 1024**2)):
        if _sha(_read(PRIVATE / name, limit=limit)) != closed.get(key):
            raise ValueError('Multisource receipt identity differs')
    if not transfer._job_ok(previous._document(_read(PRIVATE / 'job.local.json'))):
        raise ValueError('Multisource job receipt differs')
    if previous._document(_read(PRIVATE / 'attempt.local.json')) != dict(
            schema='task-multisource-rank-attempt-v1', commit=head, retry_allowed=False):
        raise ValueError('Multisource attempt differs')
    claim = previous._document(_read(PRIVATE / 'claim.local.json'))
    marker = previous._document(_read(PRIVATE / 'child-attempt.local.json'))
    owner = previous._document(_read(PRIVATE / 'owner.local.json'))
    if (claim.get('schema') != 'task-multisource-rank-claim-v1' or claim.get('commit') != head
            or type(claim.get('attempt')) is not int or claim['attempt'] != 1
            or claim.get('freeze_sha256') != binding_sha or claim.get('retry_allowed') is not False
            or type(claim.get('job_token_sha256')) is not str
            or re.fullmatch(r'[0-9a-f]{64}', claim['job_token_sha256']) is None
            or marker != dict(schema='task-multisource-rank-child-attempt-v1', commit=head, retry_allowed=False)
            or type(owner) is not dict or set(owner) != {'schema', 'registration_commit', 'binding_sha256',
                                                         'pid', 'created_us', 'deadline'}
            or owner.get('schema') != 'task-pair-fresh-multi-owner-v1'
            or owner.get('registration_commit') != head or owner.get('binding_sha256') != binding_sha
            or type(owner.get('pid')) is not int or owner['pid'] <= 0
            or type(owner.get('created_us')) is not int or owner['created_us'] <= 0
            or type(owner.get('deadline')) not in (int, float)):
        raise ValueError('Multisource claim or owner differs')
    result = _result(_read(PRIVATE / 'worker-result.local.json'), head)
    if (result['weights_sha256'] != closed['weights_sha256']
            or result['baseline_sha256'] != closed['baseline_sha256']
            or result['baseline_sha256'] != binding['earlier']['baseline_sha256']
            or previous._document(_read(PRIVATE / 'coverage.local.json')) != result['dedup']):
        raise ValueError('Multisource model identity differs')
    _validate_cast_report(previous._document(_read(PRIVATE / 'baseline-cast.local.json')), result['dedup'])
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=('prepare', 'run', 'child'))
    args = parser.parse_args()
    {'prepare': prepare, 'run': run, 'child': child}[args.command]()


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        kind = ('memory' if isinstance(exc, MemoryError) else 'timeout' if isinstance(exc, TimeoutError)
                else 'io' if isinstance(exc, OSError) else 'invalid_or_runtime')
        print(json.dumps(dict(status='failed', category=kind)), file=sys.stderr)
        raise SystemExit(1) from None
