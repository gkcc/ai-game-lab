"""A single bounded extraction of previously consumed student traces.

Only the owner's Windows Job may read authenticated game sources.  Labels are
conditional teacher choices, not executed actions, payoffs or independent test
evidence.  The private NPZ is not permission to train if the data gate fails.
"""
import argparse
from collections import Counter
import hashlib
import io
import json
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
from learning.task_pair_dev_games_v1.panel import ENGINE_COMMIT, ENGINE_SHA

REPO = previous.REPO
PRIVATE = REPO.parent / 'private' / 'task-disagreement-trace-v1-20261031'
REGISTRATION = REPO / 'review/evidence/task-disagreement-trace-v1-20261031/registration.json'
BASE_COMMIT = 'd6227dcc403ac657b3ab6bfbc729c1da75f8d216'
JOB = dict(supervisor.JOB)
INITIAL_MEMORY, MIN_DISK = 5 * 1024**3, 4 * 1024**3
MAX_ARCHIVE, MAX_GROUPS = 32 * 1024**2, 96
ROLES = ('A03', 'A04', 'A09', 'A10')
CASES = (
    ('A03', 'train', 1, 'T03', 'market', 0),
    ('A04', 'train', 1, 'T04', 'round2', 1),
    ('A09', 'diagnostic', 2, 'V01', 'market', 1),
    ('A10', 'diagnostic', 2, 'V02', 'round2', 0),
)
OPS = ('DIG', 'FERTILIZE', 'HARVEST', 'PLANT', 'WATER')
SOURCE_NAMES = tuple(dict.fromkeys(cross.SOURCES + (
    'kaggriculture/learning/task_disagreement_trace_v1/__init__.py',
    'kaggriculture/learning/task_disagreement_trace_v1/experiment.py',
    'kaggriculture/learning/task_disagreement_trace_v1/sampling.py',
    'kaggriculture/learning/task_disagreement_trace_v1/extract.py',
    'kaggriculture/learning/task_disagreement_trace_v1/test_experiment.py',
    'kaggriculture/learning/task_disagreement_trace_v1/test_sampling.py',
    'kaggriculture/learning/task_disagreement_trace_v1/test_extract.py',
    'kaggriculture/learning/task_disagreement_trace_v1/test_seal.py',
    'kaggriculture/learning/task_multisource_rank_v1/dedup.py',
    'kaggriculture/learning/task_student_continuation_v1/samples.py',
    'kaggriculture/learning/task_pair_dev_games_v1/panel.py',
    'kaggriculture/learning/task_water_residual_v1/policy.py',
    'review/evidence/task-disagreement-trace-v1-20261031/REGISTRATION.zh-CN.md',
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


def _registration(source_sha, binding_sha):
    return dict(schema='task-disagreement-trace-prereg-v1', base_commit=BASE_COMMIT,
        source_sha256=source_sha, private_binding_sha256=binding_sha,
        official_rule_commit=ENGINE_COMMIT, official_rule_sha256=ENGINE_SHA,
        cases=[dict(role=a, split=b, batch=c, historical_role=d, opponent=e, side=f)
               for a, b, c, d, e, f in CASES],
        turns_per_case=719, windows=[[0,119],[120,239],[240,359],
                                    [360,479],[480,599],[600,718]],
        max_per_window=4, max_groups=MAX_GROUPS, max_candidates=256,
        train_selection='fixed_coordinate_rank_on_teacher_student_disagreements',
        diagnostic_selection='fixed_coordinate_rank_blind_to_label_and_outcome',
        dedup='ordered_raw_or_normalized_f32_transitive_teacher_conflict_train_priority',
        filter='single_candidate_or_identical_normalized_rows_or_train_teacher_student_equal',
        thresholds=dict(train=16, diagnostic=16, per_role=4, windows_per_role=3,
                        train_teacher_operations=2),
        job=dict(JOB), memory_initial_bytes=INITIAL_MEMORY, disk_floor_bytes=MIN_DISK,
        max_attempts=1, training=False, model_forward=False, gpu=False,
        new_games=0, independent_evaluation=False)


def _context_binding(context):
    """Run only after an authorized prepare or child entry, never on import."""
    mapping = context.private_role_binding()
    selected = {case[0]: mapping[case[0]] for case in CASES}
    if len(set(selected.values())) != len(CASES):
        raise ValueError('Source roles are not distinct')
    for alias, _split, batch, role, _opponent, _side in CASES:
        if context.source_role(batch=batch, role=role) != alias:
            raise ValueError('Source role differs')
    _mean, _scale, metadata = context.model_normalization()
    names = metadata['feature_names']
    if type(names) is not list or len(names) != 98 or any(names.count('task.op.' + op) != 1 for op in OPS):
        raise ValueError('Task operation schema differs')
    return dict(roles=selected, baseline_sha256=context.task_sha256,
                crop_sha256=context.crop_sha256,
                metadata_sha256=_sha(previous._encoded(metadata)),
                schema_sha256=metadata['schema_sha256'],
                operation_columns=[names.index('task.op.' + op) for op in OPS])


def prepare():
    """Freeze source metadata only. Do not access games or run inference."""
    if (os.name != 'nt' or _git('rev-parse', 'HEAD') != BASE_COMMIT
            or _git('diff', '--name-only') or _git('diff', '--cached', '--name-only')
            or REGISTRATION.exists() or PRIVATE.exists()):
        raise ValueError('Unexpected preparation state')
    if previous.physical_available() < INITIAL_MEMORY:
        raise MemoryError('Initial memory floor')
    from learning.task_student_anchor_collection_v1 import sources
    deadline = time.monotonic() + 300
    guard = lambda: previous._guard(deadline)
    context = sources.open_context(guard=guard)
    identity = _context_binding(context)
    binding = dict(schema='disagreement-trace-binding-v1', salt=secrets.token_hex(32), **identity)
    raw = previous._encoded(binding)
    registration = _registration(_source_hashes(), _sha(raw))
    PRIVATE.mkdir(parents=False, exist_ok=False)
    previous._write_new(PRIVATE / 'binding.local.json', binding)
    REGISTRATION.parent.mkdir(parents=True, exist_ok=True)
    previous._write_new(REGISTRATION, registration)
    print(json.dumps(dict(status='prepared')), flush=True)


def _registered_binding(head):
    if type(head) is not str or re.fullmatch(r'[0-9a-f]{40}', head) is None:
        raise ValueError('Invalid registration commit')
    parent = subprocess.check_output(['git', '-C', str(REPO), 'rev-parse', head + '^'],
                                     stderr=subprocess.DEVNULL, timeout=30).decode('ascii').strip()
    raw = subprocess.check_output(['git', '-C', str(REPO), 'show',
        head + ':' + REGISTRATION.relative_to(REPO).as_posix()], stderr=subprocess.DEVNULL, timeout=30)
    binding_raw = _read(PRIVATE / 'binding.local.json')
    if parent != BASE_COMMIT or previous._document(raw) != _registration(_source_hashes(), _sha(binding_raw)):
        raise ValueError('Registration differs')
    binding = previous._document(binding_raw)
    if (set(binding) != {'schema', 'salt', 'roles', 'baseline_sha256', 'crop_sha256',
                         'metadata_sha256', 'schema_sha256', 'operation_columns'}
            or binding['schema'] != 'disagreement-trace-binding-v1'
            or type(binding['salt']) is not str or re.fullmatch(r'[0-9a-f]{64}', binding['salt']) is None
            or type(binding['roles']) is not dict or set(binding['roles']) != set(ROLES)
            or type(binding['operation_columns']) is not list or len(binding['operation_columns']) != len(OPS)
            or any(type(n) is not int or not 0 <= n < 98 for n in binding['operation_columns'])
            or len(set(binding['operation_columns'])) != len(OPS)):
        raise ValueError('Private binding differs')
    return binding, _sha(binding_raw)


def _runtime_binding():
    head = _git('rev-parse', 'HEAD')
    if _git('status', '--porcelain') or _git('rev-parse', 'origin/main') != head:
        raise ValueError('Registration is not clean and published')
    binding, digest = _registered_binding(head)
    return head, binding, digest


def _collect_case(context, case, metadata, guard, *, read_game, sampler_type, extractor):
    """Accept samples only after both game and same-file trace contexts close."""
    role, mode, batch, old_role, opponent, side = case
    with read_game(context, batch=batch, role=old_role, opponent=opponent,
                   side=side, guard=guard) as game:
        identity = game.identity
        if any(identity.get(key) != value for key, value in
               (('source_role', role), ('batch', batch), ('role', old_role),
                ('opponent', opponent), ('side', side))):
            raise ValueError('Source game identity differs')
        with game.open_trace(guard=guard) as stream:
            sampler = sampler_type(stream, mode=mode, role=role, opponent=opponent,
                                   side=side, guard=guard)
            accounting = sampler.scan(task_sha256=game.task_sha256, crop_sha256=game.crop_sha256)
            if type(accounting) is not dict or accounting.get('lines') != 719:
                raise ValueError('Source scan is incomplete')
    # The compressed trace's final same-descriptor hash has passed here.
    windows = sampler.samples()
    if type(windows) is not list or len(windows) != 6:
        raise ValueError('Selected windows differ')
    examples, seen = [], set()
    for window, choices in enumerate(windows):
        if type(choices) is not list or len(choices) > 4:
            raise ValueError('Window budget exceeded')
        for sample in choices:
            guard()
            if (type(sample) is not dict or set(sample) != {'role','opponent','side','step','ordinal',
                    'inputs','teacher','chosen'} or sample['role'] != role or sample['opponent'] != opponent
                    or type(sample['side']) is not int or sample['side'] != side
                    or type(sample['step']) is not int or not 0 <= sample['step'] < 719
                    or sample['step'] // 120 != window or sample['step'] in seen
                    or type(sample['ordinal']) is not int or not 0 <= sample['ordinal'] < 32):
                raise ValueError('Selected sample coordinate differs')
            seen.add(sample['step'])
            example = extractor(role=role, step=sample['step'], ordinal=sample['ordinal'],
                inputs=sample['inputs'], teacher_label=sample['teacher'], chosen=sample['chosen'],
                metadata=metadata, guard=guard)
            examples.append(example)
    return examples


def _build_dataset(examples, *, mean, scale, metadata, guard):
    import numpy as np
    from learning.task_multisource_rank_v1 import dedup
    from .extract import PrivateExample
    if type(examples) is not list or len(examples) > MAX_GROUPS:
        raise ValueError('Collection group budget exceeded')
    if type(metadata) is not dict or type(metadata.get('feature_names')) is not list:
        raise ValueError('Invalid task feature schema')
    names = metadata['feature_names']
    op_columns = [names.index('task.op.' + name) for name in OPS]
    if (len(names) != 98 or len(set(op_columns)) != len(OPS)
            or any(names.count('task.op.' + name) != 1 for name in OPS)):
        raise ValueError('Invalid task operation schema')
    by_item, coordinates = {}, set()
    for example in examples:
        guard()
        if (type(example) is not PrivateExample or example.role not in ROLES
                or example.item.pool != ('new_train' if example.role in ROLES[:2] else 'new_dev')
                or type(example.step) is not int or not 0 <= example.step < 719
                or type(example.ordinal) is not int or not 0 <= example.ordinal < 32
                or type(example.student_index) is not int
                or not 0 <= example.student_index < len(example.item.candidates)
                or (example.role in ROLES[:2] and example.item.target == example.student_index)
                or (example.role, example.step) in coordinates or id(example.item) in by_item):
            raise ValueError('Invalid private example identity')
        coordinates.add((example.role, example.step))
        by_item[id(example.item)] = example
    retained, dedup_summary = dedup.deduplicate([x.item for x in examples], mean=mean, scale=scale, guard=guard)
    counts = {role: dict(single=0, identical_rows=0, indistinguishable_disagreement=0,
                         diagnostic_indistinguishable_disagreement=0) for role in ROLES}
    accepted = []
    for item in retained:
        guard()
        ex = by_item[id(item)]
        if len(item.candidates) == 1:
            counts[item.role]['single'] += 1
            continue
        with np.errstate(over='ignore', divide='ignore', invalid='ignore'):
            normalized = ((item.candidates.astype(np.float64) - mean[275:]) / scale[275:]).astype('<f4')
        if not np.isfinite(normalized).all():
            raise ValueError('Normalized candidates are not finite')
        normalized[normalized == 0] = np.float32(0)
        if all(np.array_equal(normalized[0], row) for row in normalized[1:]):
            counts[item.role]['identical_rows'] += 1
            continue
        indistinguishable = (item.target != ex.student_index and
                             np.array_equal(normalized[item.target], normalized[ex.student_index]))
        if indistinguishable and item.role in ROLES[:2]:
            counts[item.role]['indistinguishable_disagreement'] += 1
            continue
        if indistinguishable:
            counts[item.role]['diagnostic_indistinguishable_disagreement'] += 1
        accepted.append(ex)
    by_role = {role: sum(ex.role == role for ex in accepted) for role in ROLES}
    by_window = {role: len({ex.step // 120 for ex in accepted if ex.role == role}) for role in ROLES}
    target_ops = Counter()
    for ex in accepted:
        if ex.role not in ROLES[:2]:
            continue
        values = ex.item.candidates[ex.item.target, op_columns]
        if not all(v in (0.0, 1.0) for v in values) or sum(values) != 1:
            raise ValueError('Teacher operation encoding differs')
        target_ops[OPS[int(np.argmax(values))]] += 1
    train = [ex for ex in accepted if ex.role in ROLES[:2]]
    diagnostic = [ex for ex in accepted if ex.role in ROLES[2:]]
    checks = dict(train_min=len(train) >= 16, diagnostic_min=len(diagnostic) >= 16,
                  per_role=all(by_role[r] >= 4 for r in ROLES),
                  windows_per_role=all(by_window[r] >= 3 for r in ROLES),
                  train_teacher_operations=len(target_ops) >= 2)
    summary = dict(selected_by_role={r: sum(ex.role == r for ex in examples) for r in ROLES},
        retained_by_role=by_role, windows_by_role=by_window,
        train_target_operations={op: target_ops[op] for op in OPS},
        dedup=dedup_summary, filtered_by_role=counts, checks=checks,
        data_gate='passed' if all(checks.values()) else 'failed')
    return train, diagnostic, summary


def _split_bytes(examples, guard):
    """Serialize only private bounded numeric inputs and non-feature indexes."""
    import numpy as np
    if type(examples) is not list or len(examples) > MAX_GROUPS:
        raise ValueError('Invalid split size')
    offsets = [0]
    for example in examples:
        guard()
        width = len(example.item.candidates)
        if not 1 <= width <= 256:
            raise ValueError('Candidate width differs')
        offsets.append(offsets[-1] + width)
    arrays = dict(
        shared=np.asarray([x.item.shared for x in examples], dtype=np.float32).reshape(-1, 275),
        candidates=(np.concatenate([x.item.candidates for x in examples], axis=0)
                    if examples else np.empty((0, 98), dtype=np.float32)),
        offsets=np.asarray(offsets, dtype=np.int32),
        teacher_target=np.asarray([x.item.target for x in examples], dtype=np.int32),
        student_index=np.asarray([x.student_index for x in examples], dtype=np.int32),
        role=np.asarray([ROLES.index(x.role) for x in examples], dtype=np.int8),
        step=np.asarray([x.step for x in examples], dtype=np.int16),
        ordinal=np.asarray([x.ordinal for x in examples], dtype=np.int16))
    output = io.BytesIO()
    np.savez_compressed(output, **arrays)
    raw = output.getvalue()
    if not 0 < len(raw) <= MAX_ARCHIVE:
        raise ValueError('Private split byte budget exceeded')
    guard()
    return raw


def _bounded_header(source, member_size):
    """Bound the declared header length before reading or asking NumPy to parse."""
    import numpy as np
    limit = 4096
    try:
        version = np.lib.format.read_magic(source)
        if version not in ((1, 0), (2, 0)):
            raise ValueError
        length_size = 2 if version == (1, 0) else 4
        length_bytes = source.read(length_size)
        if len(length_bytes) != length_size:
            raise ValueError
        length = int.from_bytes(length_bytes, 'little')
        if not 0 < length <= min(limit - source.tell(), member_size - source.tell()):
            raise ValueError
        header = source.read(length)
        if len(header) != length:
            raise ValueError
        bounded = io.BytesIO(np.lib.format.magic(*version) + length_bytes + header)
        if np.lib.format.read_magic(bounded) != version:
            raise ValueError
        reader = (np.lib.format.read_array_header_1_0 if version == (1, 0)
                  else np.lib.format.read_array_header_2_0)
        return reader(bounded, max_header_size=limit)
    except (ValueError, EOFError, TypeError, OverflowError, UnicodeError, SyntaxError):
        raise ValueError('Invalid private split header') from None


def _load_split_bytes(raw, split, guard):
    """Check zip and NPY headers before allowing NumPy to allocate."""
    import math
    import numpy as np
    dtypes = {'shared':np.dtype('float32'), 'candidates':np.dtype('float32'),
              'offsets':np.dtype('int32'), 'teacher_target':np.dtype('int32'),
              'student_index':np.dtype('int32'), 'role':np.dtype('int8'),
              'step':np.dtype('int16'), 'ordinal':np.dtype('int16')}
    if (type(raw) is not bytes or not 0 < len(raw) <= MAX_ARCHIVE
            or split not in ('train', 'diagnostic')):
        raise ValueError('Invalid private split')
    shapes = {}
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        infos = archive.infolist()
        if (len(infos) != len(dtypes) or {x.filename for x in infos} != {k + '.npy' for k in dtypes}
                or sum(x.file_size for x in infos) > MAX_ARCHIVE):
            raise ValueError('Private split archive differs')
        for info in infos:
            guard()
            with archive.open(info) as src:
                shape, fortran, dtype = _bounded_header(src, info.file_size)
                key = info.filename[:-4]
                if (type(shape) is not tuple or any(type(n) is not int or n < 0 for n in shape)
                        or fortran is not False or dtype != dtypes[key] or dtype.hasobject
                        or info.file_size != src.tell() + math.prod(shape) * dtype.itemsize):
                    raise ValueError('Invalid private split header')
                shapes[key] = shape
    n = shapes['teacher_target'][0] if len(shapes['teacher_target']) == 1 else -1
    if (not 0 <= n <= MAX_GROUPS or shapes['shared'] != (n, 275)
            or len(shapes['candidates']) != 2 or shapes['candidates'][1] != 98
            or shapes['candidates'][0] > MAX_GROUPS * 256 or shapes['offsets'] != (n + 1,)
            or any(shapes[k] != (n,) for k in ('student_index','role','step','ordinal'))):
        raise ValueError('Invalid private split shape')
    with np.load(io.BytesIO(raw), allow_pickle=False) as source:
        arrays = {key: source[key] for key in dtypes}
    offsets, candidates = arrays['offsets'], arrays['candidates']
    if (not np.isfinite(arrays['shared']).all() or not np.isfinite(candidates).all()
            or offsets[0] != 0 or offsets[-1] != len(candidates)):
        raise ValueError('Invalid private split values')
    seen, per_window = set(), Counter()
    for i in range(n):
        guard()
        width = int(offsets[i+1]) - int(offsets[i])
        role, step, ordinal = (int(arrays[key][i]) for key in ('role','step','ordinal'))
        target, student = (int(arrays[key][i]) for key in ('teacher_target','student_index'))
        if (not 2 <= width <= 256 or not 0 <= target < width or not 0 <= student < width
                or role not in ((0,1) if split == 'train' else (2,3))
                or not 0 <= step < 719 or not 0 <= ordinal < 32 or (role,step) in seen
                or (split == 'train' and target == student)):
            raise ValueError('Private split group differs')
        seen.add((role,step))
        per_window[(role,step // 120)] += 1
        if per_window[(role,step // 120)] > 4:
            raise ValueError('Private split window budget exceeded')
    return arrays


def _write_bytes_new(path, raw):
    with Path(path).open('xb') as out:
        if out.write(raw) != len(raw):
            raise OSError('Incomplete private data write')


def _counts_match(summary, train, diagnostic, operation_columns):
    """Cross-check retained counts and windows against the sealed arrays."""
    role_counts, window_sets = {r:0 for r in ROLES}, {r:set() for r in ROLES}
    for arrays in (train, diagnostic):
        for role, step in zip(arrays['role'], arrays['step']):
            alias = ROLES[int(role)]
            role_counts[alias] += 1
            window_sets[alias].add(int(step) // 120)
    if (type(summary) is not dict or set(summary) != {'selected_by_role','retained_by_role','windows_by_role',
            'train_target_operations','dedup','filtered_by_role','checks','data_gate'}
            or type(summary['train_target_operations']) is not dict
            or set(summary['train_target_operations']) != set(OPS)
            or type(operation_columns) is not list or len(operation_columns) != len(OPS)
            or any(type(n) is not int or not 0 <= n < 98 for n in operation_columns)
            or len(set(operation_columns)) != len(OPS)):
        raise ValueError('Private summary fields differ')
    op_counts = {op:0 for op in OPS}
    for i, target in enumerate(train['teacher_target']):
        row = train['candidates'][int(train['offsets'][i]) + int(target), operation_columns]
        if any(v not in (0.0,1.0) for v in row) or sum(row) != 1:
            raise ValueError('Private teacher operation differs')
        op_counts[OPS[int(row.argmax())]] += 1
    selected, filtered, dedup = summary['selected_by_role'], summary['filtered_by_role'], summary['dedup']
    if (type(selected) is not dict or set(selected) != set(ROLES)
            or any(type(selected[r]) is not int or not 0 <= selected[r] <= 24 for r in ROLES)
            or type(filtered) is not dict or set(filtered) != set(ROLES) or type(dedup) is not dict
            or set(summary['train_target_operations']) != set(OPS)
            or summary['train_target_operations'] != op_counts):
        raise ValueError('Private selection accounting differs')
    for role in ROLES:
        row = filtered[role]
        if (type(row) is not dict or set(row) != {'single','identical_rows','indistinguishable_disagreement',
                                                   'diagnostic_indistinguishable_disagreement'}
                or any(type(v) is not int or v < 0 for v in row.values())
                or selected[role] < role_counts[role]
                or row['diagnostic_indistinguishable_disagreement'] > role_counts[role]
                or (row['single'] + row['identical_rows'] + row['indistinguishable_disagreement']
                    > selected[role] - role_counts[role])
                or (role in ROLES[:2] and row['diagnostic_indistinguishable_disagreement'] != 0)
                or (role in ROLES[2:] and row['indistinguishable_disagreement'] != 0)):
            raise ValueError('Private filter accounting differs')
    for split, roles, pool in (('train',ROLES[:2],'new_train'), ('diagnostic',ROLES[2:],'new_dev')):
        try:
            incoming = dedup['input_by_pool'][pool]
            kept = dedup['retained_by_pool'][pool]
            removed = dedup['duplicate_removed_by_pool'][pool] + dedup['conflict_removed_by_pool'][pool]
            discarded = sum(filtered[r]['single'] + filtered[r]['identical_rows'] +
                            filtered[r]['indistinguishable_disagreement'] for r in roles)
        except (TypeError, KeyError):
            raise ValueError('Private deduplication accounting differs') from None
        if (any(type(x) is not int or x < 0 for x in (incoming,kept,removed))
                or incoming != sum(selected[r] for r in roles) or incoming != kept + removed
                or kept != sum(role_counts[r] for r in roles) + discarded):
            raise ValueError('Private split counts are unbalanced')
    if (dedup.get('input_count') != sum(selected.values()) or dedup.get('retained_count') !=
            dedup['retained_by_pool']['new_train'] + dedup['retained_by_pool']['new_dev']):
        raise ValueError('Private deduplication totals differ')
    checks = summary['checks']
    inferred = dict(train_min=len(train['role']) >= 16, diagnostic_min=len(diagnostic['role']) >= 16,
        per_role=all(role_counts[r] >= 4 for r in ROLES),
        windows_per_role=all(len(window_sets[r]) >= 3 for r in ROLES),
        train_teacher_operations=sum(value > 0 for value in summary['train_target_operations'].values()) >= 2)
    if (summary.get('retained_by_role') != role_counts
            or summary.get('windows_by_role') != {r:len(window_sets[r]) for r in ROLES}
            or checks != inferred or summary.get('data_gate') != ('passed' if all(checks.values()) else 'failed')
            or sum(summary['train_target_operations'].values()) != len(train['role'])):
        raise ValueError('Private split aggregate differs')


def _worker_result(raw, head):
    result = previous._document(raw)
    if (type(result) is not dict or set(result) != {'schema','registration_commit','summary','source_games',
            'trace_turns','train_sha256','diagnostic_sha256','training','model_forward','new_games'}
            or result['schema'] != 'disagreement-trace-worker-v1' or result['registration_commit'] != head
            or type(result['source_games']) is not int or result['source_games'] != 4
            or type(result['trace_turns']) is not int or result['trace_turns'] != 4 * 719
            or result['training'] is not False or result['model_forward'] is not False
            or type(result['new_games']) is not int or result['new_games'] != 0
            or type(result['summary']) is not dict
            or result['summary'].get('data_gate') not in ('passed','failed')
            or any(type(result[k]) is not str or re.fullmatch(r'[0-9a-f]{64}', result[k]) is None
                   for k in ('train_sha256','diagnostic_sha256'))):
        raise ValueError('Worker result differs')
    return result


def _dataset_contract(head, result):
    gate = result['summary']['data_gate']
    return dict(schema='disagreement-trace-private-contract-v1', registration_commit=head,
        data_gate=gate, train_sha256=result['train_sha256'],
        diagnostic_sha256=result['diagnostic_sha256'], role_vocabulary=list(ROLES),
        eligible_for_future_review=(gate == 'passed'), training_started=False,
        independent_evaluation=False)


def _validated_outputs(head, guard, operation_columns):
    result = _worker_result(_read(PRIVATE / 'worker-result.local.json'), head)
    if previous._document(_read(PRIVATE / 'dataset-contract.local.json')) != _dataset_contract(head, result):
        raise ValueError('Private dataset contract differs')
    arrays = []
    for split in ('train', 'diagnostic'):
        guard()
        raw = _read(PRIVATE / (split + '.local.npz'), MAX_ARCHIVE)
        if _sha(raw) != result[split + '_sha256']:
            raise ValueError('Private dataset digest differs')
        arrays.append(_load_split_bytes(raw, split, guard))
    _counts_match(result['summary'], *arrays, operation_columns)
    return result, arrays


def child():
    """Accept one claimed Job; sources and private results stay in this process."""
    head, binding, binding_sha = _runtime_binding()
    claim = previous._document(_read(PRIVATE / 'claim.local.json'))
    attempt = previous._document(_read(PRIVATE / 'attempt.local.json'))
    if (claim.get('schema') != 'disagreement-trace-claim-v1' or claim.get('commit') != head
            or type(claim.get('attempt')) is not int or claim['attempt'] != 1
            or claim.get('freeze_sha256') != binding_sha or claim.get('retry_allowed') is not False
            or attempt != dict(schema='disagreement-trace-attempt-v1', commit=head, retry_allowed=False)
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
        schema='disagreement-trace-child-attempt-v1', commit=head, retry_allowed=False))
    for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
        os.environ[key] = '1'
    deadline = time.monotonic() + JOB['wall_seconds']
    guard = lambda: previous._guard(deadline)
    completed = 0
    try:
        from learning.task_student_anchor_collection_v1 import sources
        from .sampling import RankedStream
        from .extract import extract_example
        context = sources.open_context(guard=guard)
        if _context_binding(context) != {k:v for k,v in binding.items() if k not in ('schema','salt')}:
            raise ValueError('Source binding changed')
        mean, scale, metadata = context.model_normalization()
        examples = []
        for case in CASES:
            guard()
            examples.extend(_collect_case(context, case, metadata, guard, read_game=sources.read_game,
                                           sampler_type=RankedStream, extractor=extract_example))
            completed += 1
        if completed != 4:
            raise ValueError('Source population incomplete')
        train, diagnostic, summary = _build_dataset(examples, mean=mean, scale=scale,
                                                     metadata=metadata, guard=guard)
        if _context_binding(context) != {k:v for k,v in binding.items() if k not in ('schema','salt')}:
            raise ValueError('Source binding changed during scan')
        if _runtime_binding() != (head, binding, binding_sha):
            raise ValueError('Registration changed during scan')
        train_raw, diagnostic_raw = _split_bytes(train, guard), _split_bytes(diagnostic, guard)
        _counts_match(summary, _load_split_bytes(train_raw, 'train', guard),
                      _load_split_bytes(diagnostic_raw, 'diagnostic', guard), binding['operation_columns'])
        if shutil.disk_usage(PRIVATE).free < MIN_DISK + len(train_raw) + len(diagnostic_raw):
            raise OSError('Private disk floor')
        result = dict(schema='disagreement-trace-worker-v1', registration_commit=head,
            summary=summary, source_games=4, trace_turns=4*719, train_sha256=_sha(train_raw),
            diagnostic_sha256=_sha(diagnostic_raw), training=False, model_forward=False, new_games=0)
        _worker_result(previous._encoded(result), head)
        _write_bytes_new(PRIVATE / 'train.local.npz', train_raw)
        _write_bytes_new(PRIVATE / 'diagnostic.local.npz', diagnostic_raw)
        guard()
        previous._write_new(PRIVATE / 'dataset-contract.local.json', _dataset_contract(head, result))
        previous._write_new(PRIVATE / 'worker-result.local.json', result)
    except BaseException as exc:
        previous._write_new(PRIVATE / 'worker-failed.local.json', dict(
            category=type(exc).__name__, completed_sources=completed, retry_allowed=False))
        raise


def run():
    if os.name != 'nt' or previous.physical_available() < INITIAL_MEMORY:
        raise MemoryError('Initial memory floor')
    if shutil.disk_usage(PRIVATE).free < MIN_DISK + 2 * MAX_ARCHIVE:
        raise OSError('Private disk floor')
    head, binding, binding_sha = _runtime_binding()
    for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
        os.environ[key] = '1'
    from learning.task_teacher_dev_deviation_v1.membership import token_hash
    from learning.task_pair_fresh_multi_v1.entry import owner_marker
    import psutil
    deadline = time.monotonic() + JOB['wall_seconds'] + 2 * JOB['cleanup_seconds']
    previous._write_new(PRIVATE / 'attempt.local.json', dict(
        schema='disagreement-trace-attempt-v1', commit=head, retry_allowed=False))
    try:
        proc = psutil.Process(os.getpid())
        owner = previous._document(owner_marker(commit=head, binding_sha=binding_sha, pid=os.getpid(),
            created_us=int(proc.create_time() * 1_000_000), deadline=deadline))
        previous._write_new(PRIVATE / 'owner.local.json', owner)
        job_name = 'Local\\TeacherDeviation-' + secrets.token_hex(16)
        previous._write_new(PRIVATE / 'claim.local.json', dict(schema='disagreement-trace-claim-v1',
            commit=head, attempt=1, freeze_sha256=binding_sha, job_token_sha256=token_hash(job_name), retry_allowed=False))
        command = [sys.executable, '-B', '-m', 'learning.task_disagreement_trace_v1.experiment', 'child']
        job = supervisor._supervise_owned(command, str(REPO / 'kaggriculture'), JOB, job_name)
        previous._write_new(PRIVATE / 'job.local.json', job)
        if not transfer._job_ok(job):
            raise RuntimeError('Owned Job did not close cleanly')
        guard = lambda: previous._guard(deadline)
        result, _arrays = _validated_outputs(head, guard, binding['operation_columns'])
        if _registered_binding(head)[1] != binding_sha:
            raise ValueError('Source registration changed')
        names = ('worker-result','dataset-contract','job','attempt','owner','claim','child-attempt')
        receipts = {name: _sha(_read(PRIVATE / (name + '.local.json'))) for name in names}
        previous._write_new(PRIVATE / 'closed.local.json', dict(schema='disagreement-trace-closed-v1',
            registration_commit=head, binding_sha256=binding_sha, receipts=receipts, retry_allowed=False))
        print(json.dumps(dict(status='completed', summary=result['summary']), sort_keys=True), flush=True)
    except BaseException as exc:
        previous._write_new(PRIVATE / 'failed.local.json', dict(
            status='failed', category=type(exc).__name__, retry_allowed=False))
        raise


def load_completed(*, guard=lambda: None):
    """Read technical completion for reporting; this is not training approval."""
    if (PRIVATE / 'failed.local.json').exists() or (PRIVATE / 'worker-failed.local.json').exists():
        raise ValueError('Failed attempt cannot be consumed')
    closed = previous._document(_read(PRIVATE / 'closed.local.json'))
    head = closed.get('registration_commit')
    binding, binding_sha = _registered_binding(head)
    names = ('worker-result','dataset-contract','job','attempt','owner','claim','child-attempt')
    if (set(closed) != {'schema','registration_commit','binding_sha256','receipts','retry_allowed'}
            or closed['schema'] != 'disagreement-trace-closed-v1' or closed['binding_sha256'] != binding_sha
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
    if (attempt != dict(schema='disagreement-trace-attempt-v1', commit=head, retry_allowed=False)
            or marker != dict(schema='disagreement-trace-child-attempt-v1', commit=head, retry_allowed=False)
            or claim.get('schema') != 'disagreement-trace-claim-v1' or claim.get('commit') != head
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
            or type(owner.get('deadline')) not in (int,float)):
        raise ValueError('Completion ownership differs')
    return _validated_outputs(head, guard, binding['operation_columns'])[0]


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
