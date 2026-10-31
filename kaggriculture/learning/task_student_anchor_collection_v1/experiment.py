"""Collect and merge consumed student demonstrations; never train here.

All original games are completed historical development sources.  The two
Anchor plans are frozen before any comparison.  Existing results remain
immutable; model files, source coordinates and features stay on this machine.
"""
import argparse
import io
import json
import os
from pathlib import Path
import secrets
import shutil
import subprocess
import sys
import time
import zipfile

from learning.task_student_continuation_v1 import experiment as previous
from learning.task_student_continuation_v1 import samples as fresh_samples
from learning.task_student_collection_v1 import experiment as fresh_collection
from learning.task_student_dagger_collection_v1 import experiment as dagger
from learning.task_student_dagger_collection_v1 import sampling
from learning.task_pair_dev_games_v1.panel import ENGINE_COMMIT, ENGINE_SHA


REPO = previous.REPO
PRIVATE = REPO.parent / 'private' / 'task-student-anchor-mixed-v1-20261031'
REGISTRATION = REPO / 'review/evidence/task-student-anchor-mixed-v1-20261031/registration.json'
FRESH_LEDGER = REPO / 'review/evidence/task-student-collection-v1-result-20261031/ledger.json'
DAGGER_LEDGER = REPO / 'review/evidence/task-student-dagger-collection-v1-result-20261031/ledger.json'
BASE_COMMIT = '64c3855b2029ac0f549b739f9f5f1539c6911906'
FRESH_COMMIT = '2da14f9d9eafb8b8a8aa90bb11a9ae3558c1fdca'
DAGGER_COMMIT = 'b706506b5fbf7e459188d096cc8d2c5dcd2dee96'
ANCHOR_RESULT_COMMIT = '1610ee975de25c0c1afa7a34facd54f1904ee331'
MAX_DATA_BYTES = 64 * 1024**2
MIN_FREE_BYTES = 4 * 1024**3
WALL_SECONDS = 3600
MAX_COMPARISONS, MAX_CALLS, MAX_TRANSITIONS = 480, 960, 960
GROUP_LIMIT = 268
FIELDS = frozenset(('features', 'offsets', 'teacher', 'student', 'third', 'role',
                    'opponent', 'side', 'window', 'step', 'ordinal'))
FRESH_ROLES = ('F01', 'F02', 'F03', 'F04')
DAGGER_ROLES = ('D01', 'D02', 'D03', 'D04', 'D05', 'D06')
ANCHOR_ROLES = tuple('A%02d' % i for i in range(1, 11))
# This order is both the merge contract and the serialized NPZ role vocabulary.
ALL_ROLES = (FRESH_ROLES[:2] + DAGGER_ROLES[:4] + ANCHOR_ROLES[:6]
             + FRESH_ROLES[2:] + DAGGER_ROLES[4:] + ANCHOR_ROLES[6:])


def _registration():
    return dict(schema='student-anchor-mixed-prereg-v1', base_commit=BASE_COMMIT,
        fresh_registration_commit=FRESH_COMMIT, dagger_registration_commit=DAGGER_COMMIT,
        anchor_result_commit=ANCHOR_RESULT_COMMIT, engine_commit=ENGINE_COMMIT,
        engine_sha256=ENGINE_SHA, source_games=40, source_passes=2,
        real_windows=240, planning_cells=288, max_comparisons=MAX_COMPARISONS,
        max_controller_calls=MAX_CALLS, max_official_transitions=MAX_TRANSITIONS,
        candidate_steps_per_window=4, max_wall_seconds=WALL_SECONDS,
        initial_memory_bytes=previous.INITIAL_MEMORY, run_memory_bytes=previous.RUN_MEMORY,
        min_free_bytes=MIN_FREE_BYTES, max_bytes_per_split=MAX_DATA_BYTES,
        max_merged_groups=GROUP_LIMIT, candidate_width_max=256, max_attempts=1,
        roles=list(ALL_ROLES), opponents=list(sampling.OPPONENTS),
        windows=[list(w) for w in sampling.WINDOWS],
        anchor_batches=[dict(T01='A01', T02='A02', T03='A03', T04='A04', V01='A07', V02='A08'),
                        dict(T01='A05', T02='A06', T03=None, T04=None, V01='A09', V02='A10')],
        data_gate=dict(train_min=32, dev_min=16, train_seed_roles_min=6, dev_seed_roles_min=4,
                       examples_per_counted_role_min=2, max_role_fraction_numerator=1,
                       max_role_fraction_denominator=3, examples_per_opponent_min=2,
                       examples_per_side_min=2, windows_each_split_min=3),
        training=False, new_complete_games=0, independent_evaluation=False, promotion=False)


def _sha_file(path):
    raw = previous._read(path)
    if not raw or len(raw) > MAX_DATA_BYTES:
        raise ValueError('Prior data size outside registered limit')
    return previous._sha(raw)


def _prior_snapshot():
    """Authenticate old outcome receipts and capture current private data identities.

    The older fresh completion did not bind its dataset contract in a closed
    receipt; its current contract is separately frozen by this new binding.
    """
    fdir, ddir = fresh_collection.PRIVATE, dagger.PRIVATE
    if (fdir / 'failed.local.json').exists() or (ddir / 'failed.local.json').exists():
        raise ValueError('Prior attempt has a failure marker')
    fresult_raw = previous._read(fdir / 'result.local.json')
    fresult = previous._document(fresult_raw)
    fclosed_raw = previous._read(fdir / 'closed.local.json')
    fclosed = previous._document(fclosed_raw)
    fcontract_raw = previous._read(fdir / 'dataset-contract.local.json')
    fcontract = previous._document(fcontract_raw)
    if (fresult != previous._document(previous._read(FRESH_LEDGER))
            or fresult.get('summary', {}).get('registration_commit') != FRESH_COMMIT
            or fresult.get('summary', {}).get('status') != 'failed'
            or fclosed != {'outcome': 'completed', 'retry_allowed': False}
            or fcontract.get('schema') != 'student-collection-private-dataset-v1'
            or fcontract.get('registration_commit') != FRESH_COMMIT
            or fcontract.get('model_role') != 'baseline' or fcontract.get('training_started') is not False):
        raise ValueError('Prior fresh collection evidence differs')
    dresult, dcontract = dagger.load_completed()
    if (dresult.get('summary') != previous._document(previous._read(DAGGER_LEDGER)).get('summary')
            or dresult.get('summary', {}).get('registration_commit') != DAGGER_COMMIT
            or dresult.get('summary', {}).get('status') != 'failed'):
        raise ValueError('Prior DAgger collection evidence differs')
    manifest = {}
    for name, directory, contract in (('fresh', fdir, fcontract), ('dagger', ddir, dcontract)):
        item = dict(contract_sha256=previous._sha(previous._read(directory / 'dataset-contract.local.json')),
                    closed_sha256=previous._sha(previous._read(directory / 'closed.local.json')),
                    result_sha256=previous._sha(previous._read(directory /
                        ('result.local.json' if name == 'fresh' else 'result.provisional.local.json'))))
        for split in ('train', 'dev'):
            item[split + '_sha256'] = _sha_file(directory / (split + '.local.npz'))
            if item[split + '_sha256'] != contract.get(split + '_sha256'):
                raise ValueError('Prior dataset does not match its contract')
        manifest[name] = item
    return manifest, fresult, dresult


def prepare():
    if (previous._git('rev-parse', 'HEAD') != BASE_COMMIT or previous._git('diff', '--name-only')
            or previous._git('diff', '--cached', '--name-only') or REGISTRATION.exists() or PRIVATE.exists()):
        raise ValueError('Unexpected mixed collection preparation state')
    manifest, _, _ = _prior_snapshot()
    from . import sources
    context = sources.open_context(guard=lambda: None)
    roles = context.private_role_binding()
    if set(roles) != set(ANCHOR_ROLES) or len(set(roles.values())) != 10:
        raise ValueError('Anchor completed seed roles differ')
    binding = dict(schema='student-anchor-mixed-private-binding-v1', salt=secrets.token_hex(32),
                   prior=manifest, anchor_roles=roles, task_sha256=context.task_sha256,
                   crop_sha256=context.crop_sha256)
    raw = previous._encoded(binding)
    PRIVATE.mkdir(parents=False, exist_ok=False)
    previous._write_new(PRIVATE / 'binding.local.json', binding)
    previous._write_new(REGISTRATION, {**_registration(), 'binding_commitment_sha256': previous._sha(raw)})
    print('Private mixed binding and public registration prepared.')


def _runtime_binding():
    head = previous._git('rev-parse', 'HEAD')
    if (previous._git('status', '--porcelain') or previous._git('rev-parse', 'origin/main') != head
            or previous._git('rev-parse', 'HEAD^') != BASE_COMMIT
            or previous._git('merge-base', '--is-ancestor', BASE_COMMIT, head)):
        raise ValueError('The exact mixed registration commit must be clean and published')
    raw = previous._read(REGISTRATION)
    committed = subprocess.check_output(['git', '-C', str(REPO), 'show',
                                         head + ':' + REGISTRATION.relative_to(REPO).as_posix()])
    reg = previous._document(raw)
    if raw != committed or any(reg.get(k) != v for k, v in _registration().items()):
        raise ValueError('Mixed collection registration differs')
    binding_raw = previous._read(PRIVATE / 'binding.local.json')
    if previous._sha(binding_raw) != reg.get('binding_commitment_sha256'):
        raise ValueError('Mixed private binding differs')
    binding = previous._document(binding_raw)
    if binding.get('schema') != 'student-anchor-mixed-private-binding-v1':
        raise ValueError('Mixed private binding schema differs')
    return head, binding


def _load_prior_groups(directory, split, expected_sha, role_map, ledger):
    """Read a bounded, authenticated NPZ and tie every retained group to its row."""
    import numpy as np
    from .merge import DEV_ROLES, TRAIN_ROLES, Group
    if split not in ('train', 'dev'):
        raise ValueError('Prior dataset split differs')
    raw = previous._read(directory / (split + '.local.npz'))
    if len(raw) > MAX_DATA_BYTES or previous._sha(raw) != expected_sha:
        raise ValueError('Prior dataset changed')
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        infos = archive.infolist()
        if ({item.filename for item in infos} != {name + '.npy' for name in FIELDS}
                or len(infos) != len(FIELDS) or sum(item.file_size for item in infos) > 128 * 1024**2):
            raise ValueError('Prior dataset archive differs')
    with np.load(io.BytesIO(raw), allow_pickle=False) as data:
        if set(data.files) != FIELDS:
            raise ValueError('Prior dataset fields differ')
        arrays = {key: data[key] for key in FIELDS}
    x, offsets = arrays['features'], arrays['offsets']
    n = len(arrays['teacher'])
    if (x.dtype != np.dtype('float32') or x.ndim != 2 or x.shape[1] != 373
            or n > 144 or not np.isfinite(x).all() or offsets.dtype != np.dtype('int32')
            or offsets.shape != (n + 1,) or offsets[0] != 0 or offsets[-1] != len(x)
            or any(value.shape != (n,) or value.dtype.kind != 'i' for key, value in arrays.items()
                   if key not in ('features', 'offsets'))):
        raise ValueError('Prior dataset shape differs')
    rows = {(row['role'], row['opponent'], row['side'], row['window']): row
            for row in ledger['rows']}
    if len(rows) != len(ledger['rows']):
        raise ValueError('Prior ledger coordinates repeat')
    groups = []
    for i in range(n):
        start, end = int(offsets[i]), int(offsets[i + 1])
        role_index, opponent_index = int(arrays['role'][i]), int(arrays['opponent'][i])
        if (not 0 <= start < end <= len(x) or not 3 <= end - start <= 256
                or not 0 <= role_index < len(role_map) or not 0 <= opponent_index < len(sampling.OPPONENTS)):
            raise ValueError('Prior group indices differ')
        old_role, new_role = role_map[role_index]
        if new_role not in (TRAIN_ROLES if split == 'train' else DEV_ROLES):
            raise ValueError('Prior dataset role is in the wrong split')
        opponent = sampling.OPPONENTS[opponent_index]
        side, window = int(arrays['side'][i]), int(arrays['window'][i])
        step, ordinal = int(arrays['step'][i]), int(arrays['ordinal'][i])
        row = rows.get((old_role, opponent, side, window))
        if (row is None or row.get('eligible') is not True or row.get('step') != step
                or row.get('ordinal') != ordinal or row.get('candidate_width') != end - start):
            raise ValueError('Prior retained group does not match its ledger')
        indices = tuple(int(arrays[key][i]) for key in ('teacher', 'student', 'third'))
        groups.append(Group(new_role, opponent, side, window, step, ordinal, x[start:end].copy(), *indices))
    return groups


def _save_dataset(path, groups, guard):
    import numpy as np
    guard()
    if len(groups) > GROUP_LIMIT:
        raise ValueError('Merged dataset group limit exceeded')
    offsets = [0]
    for group in groups:
        offsets.append(offsets[-1] + len(group.features))
    features = (np.concatenate([g.features for g in groups], axis=0) if groups
                else np.empty((0, 373), dtype=np.float32))
    arrays = dict(features=features, offsets=np.asarray(offsets, dtype=np.int32),
        teacher=np.asarray([g.teacher_index for g in groups], dtype=np.int32),
        student=np.asarray([g.student_index for g in groups], dtype=np.int32),
        third=np.asarray([g.third_index for g in groups], dtype=np.int32),
        role=np.asarray([ALL_ROLES.index(g.role) for g in groups], dtype=np.int8),
        opponent=np.asarray([sampling.OPPONENTS.index(g.opponent) for g in groups], dtype=np.int8),
        side=np.asarray([g.side for g in groups], dtype=np.int8),
        window=np.asarray([g.window for g in groups], dtype=np.int8),
        step=np.asarray([g.step for g in groups], dtype=np.int16),
        ordinal=np.asarray([g.ordinal for g in groups], dtype=np.int16))
    output = io.BytesIO()
    np.savez_compressed(output, **arrays)
    raw = output.getvalue()
    if not raw or len(raw) > MAX_DATA_BYTES:
        raise ValueError('Merged dataset byte limit exceeded')
    with Path(path).open('xb') as target:
        if target.write(raw) != len(raw):
            raise OSError('Incomplete private dataset write')
    guard()
    return previous._sha(raw)


def _dataset_contract(commit, plan_sha, train_sha, dev_sha):
    return dict(schema='student-anchor-mixed-private-dataset-v1', registration_commit=commit,
                plan_sha256=plan_sha, train_sha256=train_sha, dev_sha256=dev_sha,
                role_vocabulary=list(ALL_ROLES), model_role='baseline', training_started=False)


def _empty_windows():
    """Represent the two absent second-batch roles in the fixed plan grid."""
    return [[] for _ in sampling.WINDOWS]


def _verify_empty_entry(entry, *, schema, mean, scale, guard):
    from learning.task_student_dagger_collection_v1 import planning
    return planning.verify_entry(entry, [], schema=schema, mean=mean, scale=scale, guard=guard)


def load_completed():
    """Return only sealed private results; callers must rehash actual NPZs."""
    if (PRIVATE / 'failed.local.json').exists():
        raise ValueError('A failed mixed attempt cannot be consumed')
    closed = previous._document(previous._read(PRIVATE / 'closed.local.json'))
    attempt = previous._document(previous._read(PRIVATE / 'attempt.local.json'))
    frozen = previous._document(previous._read(PRIVATE / 'plan-frozen.local.json'))
    plan_raw = previous._read(PRIVATE / 'plan.local.json')
    result_raw = previous._read(PRIVATE / 'result.provisional.local.json')
    contract_raw = previous._read(PRIVATE / 'dataset-contract.local.json')
    plan, result, contract = map(previous._document, (plan_raw, result_raw, contract_raw))
    commit = closed.get('registration_commit')
    if (closed.get('schema') != 'student-anchor-mixed-closed-v1' or closed.get('outcome') != 'completed'
            or closed.get('retry_allowed') is not False or type(commit) is not str or len(commit) != 40
            or attempt.get('schema') != 'student-anchor-mixed-attempt-v1' or attempt.get('commit') != commit
            or attempt.get('retry_allowed') is not False
            or frozen.get('schema') != 'student-anchor-mixed-plan-freeze-v1'
            or frozen.get('registration_commit') != commit
            or frozen.get('plan_sha256') != previous._sha(plan_raw)
            or closed.get('plan_sha256') != previous._sha(plan_raw)
            or closed.get('result_sha256') != previous._sha(result_raw)
            or closed.get('dataset_contract_sha256') != previous._sha(contract_raw)
            or plan.get('schema') != 'student-anchor-mixed-private-plan-v1'
            or any(type(plan.get(key, {}).get('entries')) is not list
                   or len(plan[key]['entries']) != 144 for key in ('batch1', 'batch2'))
            or contract.get('schema') != 'student-anchor-mixed-private-dataset-v1'
            or contract.get('registration_commit') != commit
            or contract.get('plan_sha256') != previous._sha(plan_raw)
            or contract.get('role_vocabulary') != list(ALL_ROLES)
            or contract.get('model_role') != 'baseline' or contract.get('training_started') is not False
            or any(type(contract.get(key)) is not str or len(contract[key]) != 64
                   for key in ('train_sha256', 'dev_sha256'))
            or result.get('summary', {}).get('registration_commit') != commit
            or result.get('summary', {}).get('training') is not False
            or result.get('summary', {}).get('independent_evaluation') is not False
            or result.get('summary', {}).get('status') not in ('passed', 'failed')):
        raise ValueError('Mixed completion evidence differs')
    return result, contract


def run():
    if os.name != 'nt' or previous.physical_available() < previous.INITIAL_MEMORY:
        raise MemoryError('Windows initial memory floor not met')
    if shutil.disk_usage(PRIVATE).free < MIN_FREE_BYTES + 2 * MAX_DATA_BYTES:
        raise OSError('Insufficient free space for mixed datasets')
    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[name] = '1'
    head, binding = _runtime_binding()
    deadline = time.monotonic() + WALL_SECONDS
    guard = lambda: previous._guard(deadline)
    from decision.api import observe
    from decision.audit_market_teacher import observation
    from learning.task_student_continuation_v1 import compare, official_step
    from learning.task_student_dagger_collection_v1 import dataset, planning
    from . import merge, sources

    if ALL_ROLES != merge.ROLES:
        raise ValueError('Mixed serialized role vocabulary differs')
    guard()
    previous._write_new(PRIVATE / 'attempt.local.json', dict(schema='student-anchor-mixed-attempt-v1',
        commit=head, retry_allowed=False, max_comparisons=MAX_COMPARISONS))
    rows, examples = [], []
    comparisons = calls = transitions = 0
    try:
        manifest, fresult, dresult = _prior_snapshot()
        if manifest != binding.get('prior'):
            raise ValueError('Prior data identities changed after registration')
        context = sources.open_context(guard=guard)
        if (context.private_role_binding() != binding.get('anchor_roles')
                or context.task_sha256 != binding.get('task_sha256')
                or context.crop_sha256 != binding.get('crop_sha256')):
            raise ValueError('Completed source roles or model identities changed')
        mean, scale, schema = context.model_normalization()
        f_roles = tuple(zip(fresh_samples.ROLES, FRESH_ROLES))
        d_roles = tuple(zip(sampling.ROLES, DAGGER_ROLES))
        prior_groups = []
        for name, directory, role_map, result in (
                ('fresh', fresh_collection.PRIVATE, f_roles, fresult),
                ('dagger', dagger.PRIVATE, d_roles, dresult)):
            for split in ('train', 'dev'):
                guard()
                part = _load_prior_groups(directory, split,
                    manifest[name][split + '_sha256'], role_map, result)
                if len(part) != {('fresh', 'train'): 7, ('fresh', 'dev'): 2,
                                 ('dagger', 'train'): 15, ('dagger', 'dev'): 4}[(name, split)]:
                    raise ValueError('Prior split group count differs')
                prior_groups.extend(part)
        if len(prior_groups) != 28:
            raise ValueError('Prior retained group count differs')

        def cells(batch):
            for role in sampling.ROLES:
                global_role = context.source_role(batch=batch, role=role)
                for opponent in sampling.OPPONENTS:
                    for side in (0, 1):
                        guard()
                        if global_role is None:
                            windows = _empty_windows()
                            for window, candidates in enumerate(windows):
                                yield dict(role=role, opponent=opponent, side=side,
                                           window=window, candidates=candidates)
                            continue
                        with sources.read_game(context, batch=batch, role=role, opponent=opponent,
                                               side=side, guard=guard) as game:
                            if game.identity.get('source_role') != global_role:
                                raise ValueError('Completed source alias differs')
                            _accounting, windows = game.scan_candidates(guard=guard)
                            for window, candidates in enumerate(windows):
                                yield dict(role=role, opponent=opponent, side=side,
                                           window=window, candidates=candidates)

        plans = {f'batch{batch}': planning.create_plan(cells(batch), schema=schema,
                    mean=mean, scale=scale, guard=guard) for batch in (1, 2)}
        plan = dict(schema='student-anchor-mixed-private-plan-v1', **plans)
        plan_path = PRIVATE / 'plan.local.json'
        previous._write_new(plan_path, plan)
        plan_raw = previous._read(plan_path)
        plan_sha = previous._sha(plan_raw)
        if previous._document(plan_raw) != plan:
            raise ValueError('Mixed private plan readback differs')
        previous._write_new(PRIVATE / 'plan-frozen.local.json', dict(
            schema='student-anchor-mixed-plan-freeze-v1', registration_commit=head, plan_sha256=plan_sha))

        for batch in (1, 2):
            entries = iter(plans[f'batch{batch}']['entries'])
            for role in sampling.ROLES:
                global_role = context.source_role(batch=batch, role=role)
                for opponent in sampling.OPPONENTS:
                    for side in (0, 1):
                        guard()
                        if previous._sha(previous._read(plan_path)) != plan_sha:
                            raise ValueError('Frozen mixed plan changed')
                        if global_role is None:
                            for window in range(len(sampling.WINDOWS)):
                                entry = next(entries)
                                if tuple(entry.get(k) for k in ('role', 'opponent', 'side', 'window')) != (
                                        role, opponent, side, window):
                                    raise ValueError('Empty plan coordinate differs')
                                if _verify_empty_entry(entry, schema=schema, mean=mean,
                                                       scale=scale, guard=guard) is not None:
                                    raise ValueError('An empty role cannot select a game')
                            continue
                        with sources.read_game(context, batch=batch, role=role, opponent=opponent,
                                               side=side, guard=guard) as game:
                            if game.identity.get('source_role') != global_role:
                                raise ValueError('Completed source alias changed')
                            _accounting, windows = game.scan_candidates(guard=guard)
                            for window, candidates in enumerate(windows):
                                guard()
                                entry = next(entries)
                                if tuple(entry.get(k) for k in ('role', 'opponent', 'side', 'window')) != (
                                        role, opponent, side, window):
                                    raise ValueError('Mixed plan coordinate differs')
                                selected = planning.verify_entry(entry, candidates, schema=schema,
                                    mean=mean, scale=scale, guard=guard)
                                row = dict(role=global_role, batch=batch, opponent=opponent, side=side,
                                           window=window, selection=entry['status'],
                                           teacher=None, third=None, eligible=False)
                                if selected is None:
                                    rows.append(row)
                                    continue
                                record = selected['document']['records'][selected['ordinal']]
                                inputs, step = record['inputs'], selected['step']
                                row.update(step=step, ordinal=selected['ordinal'],
                                           candidate_width=len(inputs['candidates']))
                                original = game.replay['steps'][step + 1][side]['action']
                                view = observe(observation(game.replay['steps'][step], side),
                                               game.replay['configuration'])
                                if 'seed' in view.configuration or 'info' in view.observation:
                                    raise ValueError('Policy view contains privileged metadata')
                                for label, alternative in (('teacher', 'teacher'),
                                                           ('third', selected['third_candidate'])):
                                    if alternative is None:
                                        continue
                                    if (comparisons >= MAX_COMPARISONS or calls + 2 > MAX_CALLS
                                            or transitions + 2 > MAX_TRANSITIONS):
                                        raise ValueError('Mixed collection work budget exceeded')
                                    comparisons += 1
                                    calls += 2
                                    outcome = compare.compare_turn(selected['document'], view.observation,
                                        view.configuration, original, selected['ordinal'],
                                        task_sha256=game.task_sha256, crop_sha256=game.crop_sha256,
                                        game_limits=game.game_limits, agent_factory=game.make_agent, guard=guard,
                                        previous_document=selected['previous_document'], alternative=alternative)
                                    if outcome.reference_action is None or outcome.alternative_action is None:
                                        raise ValueError('Comparison lacks verified complete actions')
                                    transitions += 2
                                    official = official_step.compare_step(game.replay, step=step, side=side,
                                        baseline_request=outcome.reference_action,
                                        intervention_request=outcome.alternative_action, guard=guard)
                                    public = outcome.public_summary()
                                    public['official'] = official
                                    row[label] = public
                                example = dataset.make_example(role=role, opponent=opponent, side=side,
                                    window=window, step=step, ordinal=selected['ordinal'], inputs=inputs,
                                    teacher_index=entry['teacher_index'], student_index=entry['student_index'],
                                    third_index=entry['third_index'], teacher_summary=row['teacher'],
                                    third_summary=row['third'])
                                if example is not None:
                                    examples.append(merge.Group(global_role, opponent, side, window, step,
                                        selected['ordinal'], example.features, example.teacher_index,
                                        example.student_index, example.third_index))
                                    row['eligible'] = True
                                rows.append(row)
                                del record, selected, inputs, original, view, outcome, example
                        del game
            try:
                next(entries)
            except StopIteration:
                pass
            else:
                raise ValueError('Mixed plan has extra entries')
        guard()
        if len(rows) != 240 or previous._sha(previous._read(plan_path)) != plan_sha:
            raise ValueError('Mixed plan or real window grid differs')
        if len(examples) != sum(row['eligible'] is True for row in rows):
            raise ValueError('New eligible group count differs')
        train, dev, summary = merge.merge(prior_groups + examples, mean=mean, scale=scale)
        if shutil.disk_usage(PRIVATE).free < MIN_FREE_BYTES + 2 * MAX_DATA_BYTES:
            raise OSError('Insufficient free space for mixed datasets')
        train_sha = _save_dataset(PRIVATE / 'train.local.npz', train, guard)
        dev_sha = _save_dataset(PRIVATE / 'dev.local.npz', dev, guard)
        contract_path = PRIVATE / 'dataset-contract.local.json'
        previous._write_new(contract_path, _dataset_contract(head, plan_sha, train_sha, dev_sha))
        contract_sha = previous._sha(previous._read(contract_path))
        summary.update(anchor_source_games=40, anchor_source_passes=2, real_windows=len(rows),
            selected=sum(row['selection'] == 'selected' for row in rows), anchor_eligible=len(examples),
            prior_retained_groups=len(prior_groups), comparisons=comparisons,
            controller_call_limit_reserved=calls, official_transition_limit_reserved=transitions,
            plan_by_batch={key: value['summary'] for key, value in plans.items()},
            registration_commit=head, training=False, independent_evaluation=False)
        result_path = PRIVATE / 'result.provisional.local.json'
        previous._write_new(result_path, dict(summary=summary, rows=rows))
        result_sha = previous._sha(previous._read(result_path))
        print(json.dumps(summary, ensure_ascii=False, sort_keys=True), flush=True)
        guard()
        previous._write_new(PRIVATE / 'closed.local.json', dict(schema='student-anchor-mixed-closed-v1',
            outcome='completed', retry_allowed=False, registration_commit=head, plan_sha256=plan_sha,
            result_sha256=result_sha, dataset_contract_sha256=contract_sha))
    except BaseException as exc:
        previous._write_new(PRIVATE / 'failed.local.json', dict(outcome='failed', error_type=type(exc).__name__,
            comparisons_started=comparisons, controller_calls_reserved=calls,
            official_transitions_reserved=transitions, retry_allowed=False))
        raise


class _PrivateArgumentParser(argparse.ArgumentParser):
    def error(self, message):
        raise ValueError('invalid command line arguments')


def main():
    parser = _PrivateArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare', 'run'))
    args = parser.parse_args()
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
