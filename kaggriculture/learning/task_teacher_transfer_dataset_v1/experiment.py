"""Collect bounded multi-teacher task examples from completed development games.

Selection is fixed before support is inspected. Only terminal candidate
features and a mapped target index are retained; this stage does not train.
"""
import argparse
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
from learning.task_multi_teacher_probe_v1 import experiment as replay_probe
from learning.task_teacher_support_probe_v1 import experiment as support_probe
from learning.task_pair_dev_games_v1.panel import ENGINE_COMMIT, ENGINE_SHA


REPO = previous.REPO
PRIVATE = REPO.parent / 'private' / 'task-teacher-transfer-dataset-v1-20261031'
REGISTRATION = REPO / 'review/evidence/task-teacher-transfer-dataset-v1-20261031/registration.json'
PRIOR_LEDGER = REPO / 'review/evidence/task-teacher-support-probe-v1-result-20261031/ledger.json'
BASE_COMMIT = 'aa4af583fc86cfd6be0dd171bfccaa21e0417160'
PRIOR_COMMIT = '7da1eedbd1a5c75653bac072afc2455962134c83'
ROLES = ('A01', 'A02', 'A07', 'A08')
SLOTS = dict(A01='T01', A02='T02', A07='V01', A08='V02')
OPPONENTS, SIDES = ('market', 'round2'), (0, 1)
STEPS, HOURS = (48, 172, 296, 420, 544, 668), (0, 4, 8, 12, 16, 20)
CASES = tuple((role, opponent, side) for role in ROLES for opponent in OPPONENTS for side in SIDES)
JOB, WALL_SECONDS = replay_probe.JOB, 3600
CASE_BYTES, FINAL_BYTES, MIN_FREE_BYTES = 16 * 1024**2, 128 * 1024**2, 4 * 1024**3
MAX_GROUPS, MAX_GROUPS_PER_TURN = 384, 4
OPS = ('DIG', 'FERTILIZE', 'HARVEST', 'PLANT', 'WATER')
CASE_FIELDS = frozenset(('features', 'offsets', 'target', 'operation', 'step', 'teacher_ordinal', 'terminal_ordinal'))
SPLIT_FIELDS = CASE_FIELDS | {'role', 'opponent', 'side'}


def _registration():
    return dict(schema='teacher-transfer-dataset-prereg-v1', base_commit=BASE_COMMIT,
        prior_registration_commit=PRIOR_COMMIT, engine_commit=ENGINE_COMMIT, engine_sha256=ENGINE_SHA,
        train_roles=['A01', 'A02'], dev_roles=['A07', 'A08'], source_batch=1, role_slots=dict(SLOTS),
        opponents=list(OPPONENTS), learner_sides=list(SIDES), games=16,
        steps=list(STEPS), hours=list(HOURS), turns_per_day=24, sampled_turns=96,
        max_rule_calls=192, max_teacher_callbacks_per_turn=32, max_candidates_per_group=256,
        max_selected_ordinals_per_turn=MAX_GROUPS_PER_TURN, max_groups=MAX_GROUPS,
        case_file_max_bytes=CASE_BYTES, final_file_max_bytes=FINAL_BYTES,
        initial_available_bytes=previous.INITIAL_MEMORY, run_available_bytes=previous.RUN_MEMORY,
        disk_free_floor_bytes=MIN_FREE_BYTES, case_reserve_bytes=len(CASES) * CASE_BYTES,
        final_reserve_bytes=2 * FINAL_BYTES, total_wall_seconds=WALL_SECONDS, job=dict(JOB), max_attempts=1,
        data_gate=dict(train_min=32, dev_min=16, train_each_role_min=8, dev_each_role_min=4,
            train_each_opponent_min=6, dev_each_opponent_min=3, train_each_side_min=4, dev_each_side_min=2,
            hours_each_split_min=4, hours_each_opponent_min=2, operations_each_split_min=2),
        training=False, new_complete_games=0, new_official_transitions=0,
        independent_evaluation=False, promotion=False)


def _prior_snapshot():
    prior = support_probe.load_completed()
    public = previous._document(previous._read(PRIOR_LEDGER))
    if (prior.get('summary') != public.get('summary')
            or prior.get('summary', {}).get('registration_commit') != PRIOR_COMMIT
            or prior.get('summary', {}).get('status') != 'completed'
            or prior.get('summary', {}).get('sampled_turns') != 24):
        raise ValueError('Completed support probe differs from published evidence')
    fields = ('role', 'opponent', 'learner_side', 'teacher_side', 'replay_states', 'sampled_turns',
              'rule_calls', 'rows', 'totals', 'official_transitions', 'training', 'passed')
    if public.get('cases') != [{key: case[key] for key in fields} for case in prior['cases']]:
        raise ValueError('Published support case projection differs')
    return {name + '_sha256': previous._sha(previous._read(support_probe.PRIVATE / filename))
            for name, filename in (('result', 'result.provisional.local.json'),
                                   ('evidence', 'evidence.local.json'), ('closed', 'closed.local.json'))}


def prepare():
    if (previous._git('rev-parse', 'HEAD') != BASE_COMMIT or previous._git('diff', '--name-only')
            or previous._git('diff', '--cached', '--name-only') or REGISTRATION.exists() or PRIVATE.exists()):
        raise ValueError('Unexpected transfer dataset preparation state')
    prior = _prior_snapshot()
    from learning.task_student_anchor_collection_v1 import sources
    context = sources.open_context(guard=lambda: None)
    role_seeds = context.private_role_binding()
    selected = {role: role_seeds[role] for role in ROLES}
    if len(set(selected.values())) != len(ROLES) or any(
            context.source_role(batch=1, role=SLOTS[role]) != role for role in ROLES):
        raise ValueError('Completed source role split differs')
    binding = dict(schema='teacher-transfer-dataset-private-binding-v1', salt=secrets.token_hex(32),
                   prior=prior, source_roles=selected,
                   task_sha256=context.task_sha256, crop_sha256=context.crop_sha256)
    raw = previous._encoded(binding)
    PRIVATE.mkdir(parents=False, exist_ok=False)
    previous._write_new(PRIVATE / 'binding.local.json', binding)
    previous._write_new(REGISTRATION, {**_registration(), 'binding_commitment_sha256': previous._sha(raw)})
    print('Private transfer binding and public registration prepared.')


def _runtime_binding():
    head = previous._git('rev-parse', 'HEAD')
    if (previous._git('status', '--porcelain') or previous._git('rev-parse', 'origin/main') != head
            or previous._git('rev-parse', 'HEAD^') != BASE_COMMIT
            or previous._git('merge-base', '--is-ancestor', BASE_COMMIT, head)):
        raise ValueError('The exact transfer registration must be clean and published')
    raw = previous._read(REGISTRATION)
    committed = subprocess.check_output(['git', '-C', str(REPO), 'show',
        head + ':' + REGISTRATION.relative_to(REPO).as_posix()])
    reg = previous._document(raw)
    if raw != committed or any(reg.get(k) != v for k, v in _registration().items()):
        raise ValueError('Transfer registration differs')
    binding_raw = previous._read(PRIVATE / 'binding.local.json')
    if previous._sha(binding_raw) != reg.get('binding_commitment_sha256'):
        raise ValueError('Transfer private binding differs')
    binding = previous._document(binding_raw)
    if binding.get('schema') != 'teacher-transfer-dataset-private-binding-v1' or binding.get('prior') != _prior_snapshot():
        raise ValueError('Prior support evidence changed')
    return head, binding


def _registered_binding(head):
    """Reauthenticate the historical preregistration independently of HEAD."""
    if type(head) is not str or re.fullmatch(r'[0-9a-f]{40}', head) is None:
        raise ValueError('Invalid transfer registration commit')
    relative = REGISTRATION.relative_to(REPO).as_posix()
    raw = subprocess.check_output(['git', '-C', str(REPO), 'show', head + ':' + relative],
                                  stderr=subprocess.DEVNULL, timeout=30)
    parent = subprocess.check_output(['git', '-C', str(REPO), 'rev-parse', head + '^'],
                                     stderr=subprocess.DEVNULL, timeout=30).decode('ascii').strip()
    if not raw or len(raw) > 65536 or parent != BASE_COMMIT:
        raise ValueError('Transfer registration history differs')
    reg = previous._document(raw)
    binding_raw = previous._read(PRIVATE / 'binding.local.json')
    if (any(reg.get(k) != v for k, v in _registration().items())
            or reg.get('binding_commitment_sha256') != previous._sha(binding_raw)):
        raise ValueError('Historical transfer binding differs')
    return previous._sha(binding_raw)


def _case_name(role, opponent, side):
    if (role, opponent, side) not in CASES or type(side) is not int:
        raise ValueError('Unregistered transfer case')
    return role + '-' + opponent + '-learner' + str(side)


def _model_schema(metadata):
    from decision.api import PRODUCTS
    from learning.task_choice_v1.features import SCHEMA_VERSION, schema_hash
    keys = ('schema_version', 'schema_sha256', 'feature_names', 'shared_feature_names')
    if type(metadata) is not dict or any(key not in metadata for key in keys):
        raise ValueError('Transfer model schema missing')
    schema = {key: metadata[key] for key in keys}
    names, shared = schema['feature_names'], schema['shared_feature_names']
    start = 5 + len(PRODUCTS)
    if (schema['schema_version'] != SCHEMA_VERSION or type(names) is not list or len(names) != 98
            or type(shared) is not list or len(shared) != 275
            or any(type(name) is not str or not name for name in names + shared)
            or len(set(names)) != len(names) or len(set(shared)) != len(shared)
            or names[start:start + len(OPS)] != ['task.op.' + op for op in OPS]
            or schema['schema_sha256'] != schema_hash(names, shared)):
        raise ValueError('Transfer model schema or operation columns differ')
    return schema


def _validate_turn(row, *, role, opponent, side, step, hour):
    from .dataset import rank_ordinals
    fields = {'step', 'hour', 'teacher_request_matched', 'terminal_request_equal',
              'selected_ordinals', 'retained_groups', *support_probe.COUNT_FIELDS}
    if (type(row) is not dict or set(row) != fields or type(row.get('step')) is not int or row['step'] != step
            or type(row.get('hour')) is not int or row['hour'] != hour
            or row.get('teacher_request_matched') is not True or type(row.get('terminal_request_equal')) is not bool):
        raise ValueError('Transfer turn fields differ')
    support_row = {key: row[key] for key in ('step', 'teacher_request_matched',
                   'terminal_request_equal', *support_probe.COUNT_FIELDS)}
    support_probe._validate_row(support_row, step)
    selected = row['selected_ordinals']
    if (type(selected) is not list or any(type(ordinal) is not int for ordinal in selected)
            or selected != list(rank_ordinals(role, opponent, side, step,
                                                                     row['teacher_callbacks']))
            or type(row['retained_groups']) is not int or not 0 <= row['retained_groups'] <= len(selected)
            or row['retained_groups'] > row['supported']):
        raise ValueError('Transfer fixed ordinal selection differs')


def _save_case(path, groups, guard):
    import numpy as np
    guard()
    if len(groups) > len(STEPS) * MAX_GROUPS_PER_TURN:
        raise ValueError('Transfer case group limit exceeded')
    offsets = [0]
    for _step, group in groups:
        offsets.append(offsets[-1] + group.candidate_count)
    features = (np.concatenate([group.features for _step, group in groups], axis=0) if groups
                else np.empty((0, 373), dtype=np.float32))
    arrays = dict(features=features, offsets=np.asarray(offsets, dtype=np.int32),
        target=np.asarray([g.label for _step, g in groups], dtype=np.int32),
        operation=np.asarray([OPS.index(g.operation) for _step, g in groups], dtype=np.int8),
        step=np.asarray([step for step, _g in groups], dtype=np.int16),
        teacher_ordinal=np.asarray([g.teacher_ordinal for _step, g in groups], dtype=np.int16),
        terminal_ordinal=np.asarray([g.terminal_ordinal for _step, g in groups], dtype=np.int16))
    out = io.BytesIO()
    np.savez_compressed(out, **arrays)
    raw = out.getvalue()
    if not raw or len(raw) > CASE_BYTES:
        raise ValueError('Transfer case file limit exceeded')
    with Path(path).open('xb') as target:
        if target.write(raw) != len(raw):
            raise OSError('Incomplete transfer case write')
    guard()
    return previous._sha(raw)


def _case_result(raw, *, head, role, opponent, side):
    value = previous._document(raw)
    fields = {'schema', 'registration_commit', 'role', 'opponent', 'learner_side', 'teacher_side',
              'replay_states', 'sampled_turns', 'rule_calls', 'rows', 'data_sha256',
              'official_transitions', 'training', 'passed'}
    rows = value.get('rows')
    if (set(value) != fields or value.get('schema') != 'teacher-transfer-dataset-private-case-v1'
            or value.get('registration_commit') != head or value.get('role') != role
            or value.get('opponent') != opponent or type(value.get('learner_side')) is not int
            or value['learner_side'] != side or type(value.get('teacher_side')) is not int
            or value['teacher_side'] != 1 - side or type(value.get('replay_states')) is not int
            or value['replay_states'] != 720 or type(value.get('sampled_turns')) is not int
            or value['sampled_turns'] != len(STEPS) or type(value.get('rule_calls')) is not int
            or value['rule_calls'] != 2 * len(STEPS) or type(value.get('official_transitions')) is not int
            or value['official_transitions'] != 0 or value.get('training') is not False
            or value.get('passed') is not True or type(rows) is not list or len(rows) != len(STEPS)
            or type(value.get('data_sha256')) is not str or len(value['data_sha256']) != 64):
        raise ValueError('Transfer case result differs')
    for step, hour, row in zip(STEPS, HOURS, rows):
        _validate_turn(row, role=role, opponent=opponent, side=side, step=step, hour=hour)
    return value


def _validate_array_headers(archive, infos, dtypes, max_groups):
    """Bound declared NPY shapes before NumPy allocates from a ZIP stream."""
    import math
    import numpy as np
    shapes = {}
    for item in infos:
        key = item.filename[:-4]
        if key not in dtypes:
            raise ValueError('Transfer array header is unexpected')
        with archive.open(item, 'r') as source:
            version = np.lib.format.read_magic(source)
            if version == (1, 0):
                shape, fortran, dtype = np.lib.format.read_array_header_1_0(source)
            elif version == (2, 0):
                shape, fortran, dtype = np.lib.format.read_array_header_2_0(source)
            else:
                raise ValueError('Transfer array header version differs')
            if (type(shape) is not tuple or any(type(n) is not int or n < 0 for n in shape)
                    or fortran is not False or dtype != dtypes[key] or dtype.hasobject
                    or item.file_size != source.tell() + math.prod(shape) * dtype.itemsize):
                raise ValueError('Transfer array header differs from its data')
            shapes[key] = shape
    if set(shapes) != set(dtypes) or len(shapes['features']) != 2 or shapes['features'][1] != 373:
        raise ValueError('Transfer feature header differs')
    n = shapes['target'][0] if len(shapes['target']) == 1 else -1
    if (not 0 <= n <= max_groups
            or shapes['features'][0] > max_groups * 256
            or shapes['offsets'] != (n + 1,)
            or any(shapes[key] != (n,) for key in set(dtypes) - {'features', 'offsets'})):
        raise ValueError('Transfer array header shape exceeds its budget')


def _array_dtypes(*, split):
    import numpy as np
    result = dict(features=np.dtype('float32'), offsets=np.dtype('int32'), target=np.dtype('int32'),
                  operation=np.dtype('int8'), step=np.dtype('int16'),
                  teacher_ordinal=np.dtype('int16'), terminal_ordinal=np.dtype('int16'))
    if split:
        result.update(role=np.dtype('int8'), opponent=np.dtype('int8'), side=np.dtype('int8'))
    return result


def _load_case(path, case, guard):
    """Authenticate one bounded ragged NPZ and map it to private groups."""
    import numpy as np
    from .dataset import Group
    guard()
    raw = previous._read(path)
    if len(raw) > CASE_BYTES or previous._sha(raw) != case['data_sha256']:
        raise ValueError('Transfer case data changed')
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        infos = archive.infolist()
        if ({item.filename for item in infos} != {name + '.npy' for name in CASE_FIELDS}
                or len(infos) != len(CASE_FIELDS) or sum(item.file_size for item in infos) > CASE_BYTES):
            raise ValueError('Transfer case archive differs')
        _validate_array_headers(archive, infos, _array_dtypes(split=False), len(STEPS) * MAX_GROUPS_PER_TURN)
    with np.load(io.BytesIO(raw), allow_pickle=False) as data:
        if set(data.files) != CASE_FIELDS:
            raise ValueError('Transfer case fields differ')
        arrays = {key: data[key] for key in CASE_FIELDS}
    features, offsets = arrays['features'], arrays['offsets']
    n = len(arrays['target'])
    if (features.dtype != np.dtype('float32') or features.ndim != 2 or features.shape[1] != 373
            or n > len(STEPS) * MAX_GROUPS_PER_TURN or not np.isfinite(features).all()
            or offsets.dtype != np.dtype('int32') or offsets.shape != (n + 1,)
            or offsets[0] != 0 or offsets[-1] != len(features)
            or any(value.shape != (n,) or value.dtype.kind != 'i' for key, value in arrays.items()
                   if key not in ('features', 'offsets'))):
        raise ValueError('Transfer case array shape differs')
    rows = {row['step']: row for row in case['rows']}
    counts, groups, seen = {step: 0 for step in STEPS}, [], set()
    for i in range(n):
        guard()
        start, end = int(offsets[i]), int(offsets[i + 1])
        step, teacher_ord = int(arrays['step'][i]), int(arrays['teacher_ordinal'][i])
        terminal_ord, op_index, label = (int(arrays[k][i]) for k in ('terminal_ordinal', 'operation', 'target'))
        if (not 0 <= start < end <= len(features) or not 1 <= end - start <= 256
                or step not in rows or not 0 <= op_index < len(OPS)
                or teacher_ord not in rows[step]['selected_ordinals']
                or not 0 <= terminal_ord < rows[step]['terminal_callbacks']
                or terminal_ord != teacher_ord
                or not 0 <= label < end - start or (step, teacher_ord) in seen):
            raise ValueError('Transfer case group coordinate differs')
        seen.add((step, teacher_ord))
        counts[step] += 1
        immutable = np.frombuffer(features[start:end].tobytes(order='C'), dtype=np.float32).reshape(end - start, 373)
        groups.append(Group(case['role'], case['opponent'], case['learner_side'], step,
                            teacher_ord, terminal_ord, OPS[op_index], immutable, label))
    if any(counts[step] != rows[step]['retained_groups'] for step in STEPS):
        raise ValueError('Transfer case group counts differ')
    return groups


def child(role, opponent, side):
    name = _case_name(role, opponent, side)
    head, binding = _runtime_binding()
    if (PRIVATE / 'failed.local.json').exists() or (PRIVATE / 'closed.local.json').exists():
        raise ValueError('Transfer parent stage is already closed')
    claim = previous._document(previous._read(PRIVATE / ('claim-' + name + '.local.json')))
    attempt = previous._document(previous._read(PRIVATE / 'attempt.local.json'))
    binding_sha = _registered_binding(head)
    if (claim.get('schema') != 'teacher-transfer-dataset-claim-v1' or claim.get('commit') != head
            or claim.get('case') != name or claim.get('attempt') != 1
            or claim.get('retry_allowed') is not False or claim.get('freeze_sha256') != binding_sha
            or attempt != dict(schema='teacher-transfer-dataset-attempt-v1', commit=head, retry_allowed=False)
            or (PRIVATE / ('case-' + name + '.local.json')).exists()
            or (PRIVATE / ('case-' + name + '.failed.local.json')).exists()):
        raise ValueError('Transfer child is not uniquely claimed')
    from learning.task_teacher_dev_deviation_v1.membership import ENV_KEY, verify_worker
    from learning.task_pair_fresh_multi_v1.entry import verify_live_owner
    import psutil
    owner = previous._document(previous._read(PRIVATE / 'owner.local.json'))
    verify_worker(os.environ.get(ENV_KEY), claim, binding_sha)
    verify_live_owner(owner, commit=head, binding_sha=binding_sha, current_pid=os.getpid(),
                      process=psutil.Process, executable=sys.executable, samefile=os.path.samefile,
                      now=time.monotonic())
    previous._write_new(PRIVATE / ('child-attempt-' + name + '.local.json'), dict(
        schema='teacher-transfer-dataset-child-attempt-v1', commit=head, case=name, retry_allowed=False))
    for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[key] = '1'
    deadline = time.monotonic() + JOB['wall_seconds']
    guard = lambda: previous._guard(deadline)
    rows, selected_groups, calls = [], [], 0
    try:
        from decision.api import observe
        from decision.audit_market_teacher import observation
        from learning.task_request_gate_v1.policy import same_request
        from learning.task_student_anchor_collection_v1 import sources
        from learning.task_teacher_support_probe_v1.opponent import SupportTeacher
        from learning.task_teacher_support_probe_v1.terminal import TerminalTeacherProbe
        from .dataset import rank_ordinals
        from .extract import extract_turn
        context = sources.open_context(guard=guard)
        if (context.source_role(batch=1, role=SLOTS[role]) != role
                or context.private_role_binding()[role] != binding.get('source_roles', {}).get(role)
                or context.task_sha256 != binding.get('task_sha256')
                or context.crop_sha256 != binding.get('crop_sha256')):
            raise ValueError('Completed transfer source binding differs')
        _mean, _scale, metadata = context.model_normalization()
        schema = _model_schema(metadata)
        with sources.read_game(context, batch=1, role=SLOTS[role], opponent=opponent, side=side,
                               guard=guard) as game:
            raw, expected = replay_probe._rival_source(game, guard)
            replay = game.replay
            teacher_side = 1 - side
            if (type(replay.get('steps')) is not list or len(replay['steps']) != 720
                    or type(replay.get('configuration', {}).get('turnsPerDay')) is not int
                    or replay['configuration']['turnsPerDay'] != 24):
                raise ValueError('Completed replay configuration differs')
            for step, hour in zip(STEPS, HOURS):
                guard()
                if calls + 2 > len(STEPS) * 2:
                    raise ValueError('Transfer rule call limit exceeded')
                view = observe(observation(replay['steps'][step], teacher_side), replay['configuration'])
                if ('seed' in view.configuration or 'info' in view.observation
                        or type(view.observation.get('hour')) is not int or view.observation['hour'] != hour):
                    raise ValueError('Policy view or registered hour differs')
                teacher = SupportTeacher(raw, expected, opponent, guard=guard)
                terminal = TerminalTeacherProbe(guard=guard)
                calls += 1
                teacher_request = teacher(view.observation, view.configuration)
                if not same_request(teacher_request, replay['steps'][step + 1][teacher_side]['action']):
                    raise ValueError('Teacher request does not reproduce the replay')
                calls += 1
                terminal_request = terminal(view.observation, view.configuration)
                teacher_records = teacher.last.get('records') if type(teacher.last) is dict else None
                terminal_records = terminal.last.get('records') if type(terminal.last) is dict else None
                if type(teacher_records) is not list or type(terminal_records) is not list:
                    raise ValueError('Rule callback records missing')
                selected = rank_ordinals(role, opponent, side, step, len(teacher_records))
                groups, counts = extract_turn(view.observation, view.configuration, teacher_records,
                                              terminal_records, schema=schema, guard=guard)
                retained = [group for group in groups if group.teacher_ordinal in selected]
                if len(retained) > MAX_GROUPS_PER_TURN or len({g.teacher_ordinal for g in retained}) != len(retained):
                    raise ValueError('Transfer selected groups differ')
                row = dict(step=step, hour=hour, teacher_request_matched=True,
                           terminal_request_equal=same_request(teacher_request, terminal_request),
                           selected_ordinals=list(selected), retained_groups=len(retained), **counts)
                _validate_turn(row, role=role, opponent=opponent, side=side, step=step, hour=hour)
                selected_groups.extend((step, group) for group in retained)
                rows.append(row)
                teacher.clear()
                terminal.clear()
                del teacher, terminal, teacher_records, terminal_records, view, teacher_request, terminal_request
                del groups, retained
        data_path = PRIVATE / ('data-' + name + '.local.npz')
        data_sha = _save_case(data_path, selected_groups, guard)
        previous._write_new(PRIVATE / ('case-' + name + '.local.json'), dict(
            schema='teacher-transfer-dataset-private-case-v1', registration_commit=head, role=role,
            opponent=opponent, learner_side=side, teacher_side=teacher_side, replay_states=720,
            sampled_turns=len(rows), rule_calls=calls, rows=rows, data_sha256=data_sha,
            official_transitions=0, training=False, passed=True))
    except BaseException as exc:
        previous._write_new(PRIVATE / ('case-' + name + '.failed.local.json'), dict(
            outcome='failed', error_type=type(exc).__name__, completed_turns=len(rows),
            rule_calls_started=calls, retry_allowed=False))
        raise


def _split_arrays(groups, guard):
    import numpy as np
    guard()
    if len(groups) > MAX_GROUPS:
        raise ValueError('Transfer split group limit exceeded')
    offsets = [0]
    for group in groups:
        offsets.append(offsets[-1] + len(group.features))
    features = (np.concatenate([g.features for g in groups], axis=0) if groups
                else np.empty((0, 373), dtype=np.float32))
    return dict(features=features, offsets=np.asarray(offsets, dtype=np.int32),
        target=np.asarray([g.target_index for g in groups], dtype=np.int32),
        role=np.asarray([ROLES.index(g.role) for g in groups], dtype=np.int8),
        opponent=np.asarray([OPPONENTS.index(g.opponent) for g in groups], dtype=np.int8),
        side=np.asarray([g.side for g in groups], dtype=np.int8),
        operation=np.asarray([OPS.index(g.op) for g in groups], dtype=np.int8),
        step=np.asarray([g.step for g in groups], dtype=np.int16),
        teacher_ordinal=np.asarray([g.teacher_ordinal for g in groups], dtype=np.int16),
        terminal_ordinal=np.asarray([g.terminal_ordinal for g in groups], dtype=np.int16))


def _save_split(path, groups, guard):
    import numpy as np
    arrays = _split_arrays(groups, guard)
    output = io.BytesIO()
    np.savez_compressed(output, **arrays)
    raw = output.getvalue()
    if not raw or len(raw) > FINAL_BYTES:
        raise ValueError('Transfer final file limit exceeded')
    with Path(path).open('xb') as target:
        if target.write(raw) != len(raw):
            raise OSError('Incomplete transfer split write')
    guard()
    return previous._sha(raw)


def _file_sha(path, maximum, guard):
    import hashlib
    guard()
    digest, size = hashlib.sha256(), 0
    with Path(path).open('rb') as source:
        while True:
            block = source.read(min(4 * 1024**2, maximum - size + 1))
            if not block:
                break
            size += len(block)
            if size > maximum:
                raise ValueError('Transfer data file exceeds registered limit')
            digest.update(block)
            guard()
    if size == 0:
        raise ValueError('Empty transfer data file')
    return digest.hexdigest()


def _contract(head, train_sha, dev_sha):
    return dict(schema='teacher-transfer-dataset-private-contract-v1', registration_commit=head,
        train_sha256=train_sha, dev_sha256=dev_sha, role_vocabulary=list(ROLES),
        opponent_vocabulary=list(OPPONENTS), operation_vocabulary=list(OPS),
        model_role='baseline', training_started=False)


def _job_ok(job):
    required = ('passed', 'assigned', 'resumed', 'launcher_reaped', 'job_members_zero', 'exit_code',
                'acceptance_error_kind', 'retry_allowed', 'stop_reason', 'error_kind',
                'cleanup_error_kind', 'termination_error_kind', 'wait_error_kind', 'membership_error_kind',
                'close_error_kind', 'rss_complete_samples', 'peak_sampled_rss_bytes',
                'peak_sampled_job_commit_bytes', 'max_sampled_members', 'rss_threshold_sample_members')
    return (type(job) is dict and all(key in job for key in required)
            and job.get('passed') is True and job.get('launcher_reaped') is True
            and job.get('assigned') is True and job.get('resumed') is True
            and job.get('job_members_zero') is True and type(job.get('exit_code')) is int
            and job['exit_code'] == 0 and job.get('acceptance_error_kind') is None
            and job.get('retry_allowed') is False and job.get('stop_reason') is None
            and job.get('error_kind') is None and job.get('cleanup_error_kind') is None
            and all(job[key] is None for key in ('termination_error_kind', 'wait_error_kind',
                                                  'membership_error_kind', 'close_error_kind',
                                                  'rss_threshold_sample_members'))
            and type(job.get('peak_sampled_rss_bytes')) is int
            and 0 <= job['peak_sampled_rss_bytes'] <= JOB['max_rss_bytes']
            and type(job.get('peak_sampled_job_commit_bytes')) is int
            and 0 <= job['peak_sampled_job_commit_bytes'] <= JOB['max_commit_bytes']
            and type(job.get('max_sampled_members')) is int and job['max_sampled_members'] >= 1
            and type(job.get('rss_complete_samples')) is int and job['rss_complete_samples'] >= 1)


def load_completed(*, guard=lambda: None):
    """Load a technical seal for reporting; it does not authorize training."""
    if (PRIVATE / 'failed.local.json').exists():
        raise ValueError('A failed transfer attempt cannot be consumed')
    closed = previous._document(previous._read(PRIVATE / 'closed.local.json'))
    attempt = previous._document(previous._read(PRIVATE / 'attempt.local.json'))
    evidence_raw = previous._read(PRIVATE / 'evidence.local.json')
    result_raw = previous._read(PRIVATE / 'result.provisional.local.json')
    contract_raw = previous._read(PRIVATE / 'dataset-contract.local.json')
    evidence, result, contract = map(previous._document, (evidence_raw, result_raw, contract_raw))
    head = closed.get('registration_commit')
    binding_sha = _registered_binding(head)
    if (type(head) is not str or len(head) != 40
            or closed.get('schema') != 'teacher-transfer-dataset-closed-v1'
            or closed.get('outcome') != 'completed' or closed.get('retry_allowed') is not False
            or closed.get('result_sha256') != previous._sha(result_raw)
            or closed.get('evidence_sha256') != previous._sha(evidence_raw)
            or closed.get('dataset_contract_sha256') != previous._sha(contract_raw)
            or attempt != dict(schema='teacher-transfer-dataset-attempt-v1', commit=head, retry_allowed=False)
            or evidence.get('schema') != 'teacher-transfer-dataset-private-evidence-v1'
            or evidence.get('registration_commit') != head or type(evidence.get('cases')) is not dict
            or set(evidence['cases']) != {_case_name(r, o, s) for r, o, s in CASES}
            or contract.get('schema') != 'teacher-transfer-dataset-private-contract-v1'
            or contract.get('registration_commit') != head or contract.get('role_vocabulary') != list(ROLES)
            or contract.get('opponent_vocabulary') != list(OPPONENTS) or contract.get('operation_vocabulary') != list(OPS)
            or contract.get('model_role') != 'baseline' or contract.get('training_started') is not False
            or any(type(contract.get(key)) is not str or len(contract[key]) != 64
                   for key in ('train_sha256', 'dev_sha256'))):
        raise ValueError('Transfer completion evidence differs')
    reconstructed = []
    for role, opponent, side in CASES:
        guard()
        name = _case_name(role, opponent, side)
        if (PRIVATE / ('case-' + name + '.failed.local.json')).exists():
            raise ValueError('A transfer case has a failure marker')
        item = evidence['cases'][name]
        case_raw = previous._read(PRIVATE / ('case-' + name + '.local.json'))
        job_raw = previous._read(PRIVATE / ('job-' + name + '.local.json'))
        claim_raw = previous._read(PRIVATE / ('claim-' + name + '.local.json'))
        child_attempt_raw = previous._read(PRIVATE / ('child-attempt-' + name + '.local.json'))
        claim, child_attempt = previous._document(claim_raw), previous._document(child_attempt_raw)
        if (type(item) is not dict or item.get('case_sha256') != previous._sha(case_raw)
                or item.get('job_sha256') != previous._sha(job_raw)
                or item.get('claim_sha256') != previous._sha(claim_raw)
                or item.get('child_attempt_sha256') != previous._sha(child_attempt_raw)
                or claim.get('schema') != 'teacher-transfer-dataset-claim-v1' or claim.get('commit') != head
                or claim.get('case') != name or claim.get('attempt') != 1
                or claim.get('retry_allowed') is not False or claim.get('freeze_sha256') != binding_sha
                or type(claim.get('job_token_sha256')) is not str or len(claim['job_token_sha256']) != 64
                or child_attempt != dict(schema='teacher-transfer-dataset-child-attempt-v1',
                                         commit=head, case=name, retry_allowed=False)):
            raise ValueError('Transfer case receipt differs')
        if not _job_ok(previous._document(job_raw)):
            raise ValueError('Transfer job did not close cleanly')
        case = _case_result(case_raw, head=head, role=role, opponent=opponent, side=side)
        data_sha = _file_sha(PRIVATE / ('data-' + name + '.local.npz'), CASE_BYTES, guard)
        if data_sha != case['data_sha256'] or data_sha != item.get('data_sha256'):
            raise ValueError('Transfer case data differs')
        reconstructed.append(case)
    for split in ('train', 'dev'):
        if _file_sha(PRIVATE / (split + '.local.npz'), FINAL_BYTES, guard) != contract[split + '_sha256']:
            raise ValueError('Transfer final dataset differs')
    summary = result.get('summary')
    if (result.get('cases') != reconstructed or type(summary) is not dict
            or summary.get('registration_commit') != head or summary.get('status') not in ('passed', 'failed')
            or summary.get('training') is not False or summary.get('independent_evaluation') is not False
            or summary.get('source_games') != 16 or summary.get('sampled_turns') != 96
            or summary.get('rule_calls') != 192 or summary.get('official_transitions') != 0
            or summary.get('source_groups') != sum(row['retained_groups'] for case in reconstructed for row in case['rows'])):
        raise ValueError('Transfer sealed result differs')
    return result, contract


def _verify_split(path, expected_groups, expected_sha, guard):
    """Check a bounded final NPZ byte-for-byte against the rederived groups."""
    import numpy as np
    guard()
    blocks, size = [], 0
    with Path(path).open('rb') as source:
        while True:
            block = source.read(min(4 * 1024**2, FINAL_BYTES - size + 1))
            if not block:
                break
            size += len(block)
            if size > FINAL_BYTES:
                raise ValueError('Transfer final dataset exceeds its budget')
            blocks.append(block)
            guard()
    raw = b''.join(blocks)
    if not raw or previous._sha(raw) != expected_sha:
        raise ValueError('Transfer final dataset hash differs')
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        infos = archive.infolist()
        if ({item.filename for item in infos} != {name + '.npy' for name in SPLIT_FIELDS}
                or len(infos) != len(SPLIT_FIELDS)
                or sum(item.file_size for item in infos) > MAX_GROUPS * 256 * 373 * 4 + 2 * 1024**2):
            raise ValueError('Transfer final archive differs')
        _validate_array_headers(archive, infos, _array_dtypes(split=True), MAX_GROUPS)
    expected = _split_arrays(expected_groups, guard)
    with np.load(io.BytesIO(raw), allow_pickle=False) as data:
        if set(data.files) != SPLIT_FIELDS:
            raise ValueError('Transfer final fields differ')
        for key in SPLIT_FIELDS:
            guard()
            actual = data[key]
            target = expected[key]
            if (actual.dtype != target.dtype or actual.shape != target.shape
                    or actual.tobytes(order='C') != target.tobytes(order='C')):
                raise ValueError('Transfer final arrays differ from rederived groups')
    guard()


def load_trainable(*, guard=lambda: None):
    """Recompute the data gate and splits before permitting a future training stage.

    Source callbacks themselves were not retained, so this checks sealed case
    arrays and their registered receipts, not a re-execution of each teacher.
    """
    from learning.task_student_anchor_collection_v1 import sources
    from . import dataset
    result, contract = load_completed(guard=guard)
    if result['summary']['status'] != 'passed':
        raise ValueError('Transfer data gate did not pass')
    binding_raw = previous._read(PRIVATE / 'binding.local.json')
    if previous._sha(binding_raw) != _registered_binding(contract['registration_commit']):
        raise ValueError('Transfer training source binding differs from registration')
    binding = previous._document(binding_raw)
    context = sources.open_context(guard=guard)
    roles = context.private_role_binding()
    if ({role: roles[role] for role in ROLES} != binding.get('source_roles')
            or context.task_sha256 != binding.get('task_sha256')
            or context.crop_sha256 != binding.get('crop_sha256')
            or any(context.source_role(batch=1, role=SLOTS[role]) != role for role in ROLES)):
        raise ValueError('Transfer source role or model binding differs')
    mean, scale, metadata = context.model_normalization()
    _model_schema(metadata)
    groups = []
    for case in result['cases']:
        guard()
        name = _case_name(case['role'], case['opponent'], case['learner_side'])
        groups.extend(_load_case(PRIVATE / ('data-' + name + '.local.npz'), case, guard))
        if len(groups) > MAX_GROUPS:
            raise ValueError('Transfer source group limit exceeded')
    train, dev, summary = dataset.merge(groups, mean=mean, scale=scale, guard=guard)
    rows = [row for case in result['cases'] for row in case['rows']]
    summary.update(registration_commit=contract['registration_commit'], source_games=len(result['cases']),
        sampled_turns=len(rows), rule_calls=sum(case['rule_calls'] for case in result['cases']),
        teacher_requests_matched=sum(int(row['teacher_request_matched']) for row in rows),
        terminal_full_requests_equal=sum(int(row['terminal_request_equal']) for row in rows),
        selected_ordinals=sum(len(row['selected_ordinals']) for row in rows),
        supported_callbacks=sum(row['supported'] for row in rows),
        unsupported_callbacks=sum(row['unsupported'] for row in rows),
        unknown_callbacks=sum(row['unknown'] for row in rows),
        official_transitions=0, training=False, independent_evaluation=False)
    if summary != result['summary'] or summary['status'] != 'passed':
        raise ValueError('Transfer data gate or aggregate cannot be reproduced')
    for name, selected in (('train', train), ('dev', dev)):
        _verify_split(PRIVATE / (name + '.local.npz'), selected, contract[name + '_sha256'], guard)
    return train, dev, contract


def run():
    if os.name != 'nt' or previous.physical_available() < previous.INITIAL_MEMORY:
        raise MemoryError('Windows initial memory floor not met')
    if shutil.disk_usage(PRIVATE).free < MIN_FREE_BYTES + len(CASES) * CASE_BYTES + 2 * FINAL_BYTES:
        raise OSError('Insufficient free space for transfer datasets')
    head, binding = _runtime_binding()
    for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[key] = '1'
    from learning.task_teacher_dev_deviation_v1.membership import token_hash
    from learning.task_pair_fresh_multi_v1.entry import owner_marker
    from learning.task_student_anchor_collection_v1 import sources
    from . import dataset
    import psutil
    deadline = time.monotonic() + WALL_SECONDS
    guard = lambda: previous._guard(deadline)
    guard()
    previous._write_new(PRIVATE / 'attempt.local.json', dict(
        schema='teacher-transfer-dataset-attempt-v1', commit=head, retry_allowed=False))
    completed, groups, evidence = [], [], {}
    active_case = None
    try:
        binding_sha = previous._sha(previous._read(PRIVATE / 'binding.local.json'))
        owner_process = psutil.Process(os.getpid())
        owner = previous._document(owner_marker(commit=head, binding_sha=binding_sha, pid=os.getpid(),
            created_us=int(owner_process.create_time() * 1_000_000), deadline=deadline))
        previous._write_new(PRIVATE / 'owner.local.json', owner)
        context = sources.open_context(guard=guard)
        actual_roles = context.private_role_binding()
        if ({role: actual_roles[role] for role in ROLES} != binding.get('source_roles')
                or context.task_sha256 != binding.get('task_sha256')
                or context.crop_sha256 != binding.get('crop_sha256')
                or any(context.source_role(batch=1, role=SLOTS[role]) != role for role in ROLES)):
            raise ValueError('Transfer source role or model binding differs')
        mean, scale, metadata = context.model_normalization()
        _model_schema(metadata)
        if (tuple(dataset.ROLES) != ROLES or tuple(dataset.OPPONENTS) != OPPONENTS
                or tuple(dataset.STEPS) != STEPS or tuple(dataset.HOURS) != HOURS
                or tuple(dataset.OPS) != OPS or dataset.MAX_GROUPS != MAX_GROUPS
                or dataset.MAX_SELECTED != MAX_GROUPS_PER_TURN):
            raise ValueError('Transfer data vocabulary differs')
        for role, opponent, side in CASES:
            guard()
            if previous.physical_available() < previous.INITIAL_MEMORY:
                raise MemoryError('Transfer between-case memory floor not met')
            active_case = _case_name(role, opponent, side)
            job_name = 'Local\\TeacherDeviation-' + secrets.token_hex(16)
            previous._write_new(PRIVATE / ('claim-' + active_case + '.local.json'),
                dict(schema='teacher-transfer-dataset-claim-v1', commit=head, case=active_case,
                     attempt=1, retry_allowed=False, freeze_sha256=binding_sha,
                     job_token_sha256=token_hash(job_name)))
            command = [sys.executable, '-B', '-m', 'learning.task_teacher_transfer_dataset_v1.experiment',
                       'child', role, opponent, str(side)]
            remaining = deadline - time.monotonic() - 2 * JOB['cleanup_seconds']
            if remaining <= 0:
                raise TimeoutError('Transfer parent deadline exhausted')
            limits = dict(JOB, wall_seconds=min(JOB['wall_seconds'], remaining))
            job = replay_probe._supervise_owned(command, str(REPO / 'kaggriculture'), limits, job_name)
            job_path = PRIVATE / ('job-' + active_case + '.local.json')
            previous._write_new(job_path, job)
            if not _job_ok(job):
                raise RuntimeError('Owned transfer case did not finish cleanly')
            case_path = PRIVATE / ('case-' + active_case + '.local.json')
            raw = previous._read(case_path)
            case = _case_result(raw, head=head, role=role, opponent=opponent, side=side)
            data_path = PRIVATE / ('data-' + active_case + '.local.npz')
            part = _load_case(data_path, case, guard)
            if len(groups) + len(part) > MAX_GROUPS:
                raise ValueError('Transfer source group limit exceeded')
            groups.extend(part)
            evidence[active_case] = dict(case_sha256=previous._sha(raw),
                job_sha256=previous._sha(previous._read(job_path)), data_sha256=case['data_sha256'],
                claim_sha256=previous._sha(previous._read(PRIVATE / ('claim-' + active_case + '.local.json'))),
                child_attempt_sha256=previous._sha(previous._read(
                    PRIVATE / ('child-attempt-' + active_case + '.local.json'))))
            completed.append(case)
            active_case = None
        guard()
        if (len(completed) != 16 or len(groups) !=
                sum(row['retained_groups'] for case in completed for row in case['rows'])):
            raise ValueError('Transfer source cases or counts are incomplete')
        train, dev, summary = dataset.merge(groups, mean=mean, scale=scale, guard=guard)
        if shutil.disk_usage(PRIVATE).free < MIN_FREE_BYTES + 2 * FINAL_BYTES:
            raise OSError('Insufficient free space for final transfer splits')
        train_sha = _save_split(PRIVATE / 'train.local.npz', train, guard)
        dev_sha = _save_split(PRIVATE / 'dev.local.npz', dev, guard)
        contract_path = PRIVATE / 'dataset-contract.local.json'
        previous._write_new(contract_path, _contract(head, train_sha, dev_sha))
        rows = [row for case in completed for row in case['rows']]
        summary.update(registration_commit=head, source_games=16, sampled_turns=len(rows),
            rule_calls=sum(case['rule_calls'] for case in completed),
            teacher_requests_matched=sum(int(row['teacher_request_matched']) for row in rows),
            terminal_full_requests_equal=sum(int(row['terminal_request_equal']) for row in rows),
            selected_ordinals=sum(len(row['selected_ordinals']) for row in rows),
            supported_callbacks=sum(row['supported'] for row in rows),
            unsupported_callbacks=sum(row['unsupported'] for row in rows),
            unknown_callbacks=sum(row['unknown'] for row in rows),
            official_transitions=0, training=False, independent_evaluation=False)
        if (summary['sampled_turns'] != 96 or summary['rule_calls'] != 192
                or summary['teacher_requests_matched'] != 96 or summary['source_groups'] != len(groups)):
            raise ValueError('Transfer aggregate source counts differ')
        evidence_path = PRIVATE / 'evidence.local.json'
        previous._write_new(evidence_path, dict(schema='teacher-transfer-dataset-private-evidence-v1',
                                                registration_commit=head, cases=evidence))
        result_path = PRIVATE / 'result.provisional.local.json'
        previous._write_new(result_path, dict(summary=summary, cases=completed))
        print(json.dumps(summary, ensure_ascii=False, sort_keys=True), flush=True)
        guard()
        previous._write_new(PRIVATE / 'closed.local.json', dict(schema='teacher-transfer-dataset-closed-v1',
            outcome='completed', retry_allowed=False, registration_commit=head,
            evidence_sha256=previous._sha(previous._read(evidence_path)),
            result_sha256=previous._sha(previous._read(result_path)),
            dataset_contract_sha256=previous._sha(previous._read(contract_path))))
    except BaseException as exc:
        previous._write_new(PRIVATE / 'failed.local.json', dict(outcome='failed',
            error_type=type(exc).__name__, completed_cases=len(completed),
            active_case=active_case, retained_groups=len(groups), retry_allowed=False))
        raise


class _PrivateArgumentParser(argparse.ArgumentParser):
    def error(self, message):
        raise ValueError('invalid command line arguments')


def main():
    parser = _PrivateArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare', 'run', 'child'))
    parser.add_argument('role', nargs='?', choices=ROLES)
    parser.add_argument('opponent', nargs='?', choices=OPPONENTS)
    parser.add_argument('side', nargs='?', type=int, choices=SIDES)
    args = parser.parse_args()
    if args.command == 'child':
        if args.role is None or args.opponent is None or args.side is None:
            raise ValueError('Transfer child coordinates required')
        child(args.role, args.opponent, args.side)
    else:
        if args.role is not None or args.opponent is not None or args.side is not None:
            raise ValueError('Unexpected transfer coordinates')
        prepare() if args.command == 'prepare' else run()


def cli():
    try:
        main()
    except KeyboardInterrupt:
        print(json.dumps(dict(status='failed', category='interrupted')), file=sys.stderr)
        raise SystemExit(130) from None
    except Exception as exc:
        category = ('memory' if isinstance(exc, MemoryError) else 'timeout' if isinstance(exc, TimeoutError)
                    else 'io' if isinstance(exc, OSError) else 'invalid_or_runtime')
        print(json.dumps(dict(status='failed', category=category)), file=sys.stderr)
        raise SystemExit(1) from None


if __name__ == '__main__':
    cli()
