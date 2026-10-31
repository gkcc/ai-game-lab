"""Freeze a diverse plan, then inspect consumed DAgger student trajectories.

The trusted process reads only the 24 completed source games. It never parses
the old unused-seed lists, runs a new complete game, or trains a model. Raw
observations, model identities, feature fingerprints and plans remain private.
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

from learning.task_student_continuation_v1 import experiment as previous
from . import sampling


REPO = previous.REPO
PRIVATE = REPO.parent / 'private' / 'task-student-dagger-collection-v1-20261031'
PRIOR = REPO.parent / 'private' / 'task-student-collection-v1-20261031'
PRIOR_REG = REPO / 'review/evidence/task-student-collection-v1-20261031/registration.json'
PRIOR_LEDGER = REPO / 'review/evidence/task-student-collection-v1-result-20261031/ledger.json'
REGISTRATION = REPO / 'review/evidence/task-student-dagger-collection-v1-20261031/registration.json'
BASE_COMMIT = '2b5af446d5f889d9cfdccf44eb405c03d1f70530'
PRIOR_COMMIT = '2da14f9d9eafb8b8a8aa90bb11a9ae3558c1fdca'
ROOT_SHA256 = '98142750672977a458655f3fb953789e607108cd352ba3be117d647f0f402279'
MAX_COMPARISONS, MAX_CALLS, MAX_TRANSITIONS = 288, 576, 576
WALL_SECONDS, MAX_DATA_BYTES, MIN_FREE_BYTES = 3600, 64 * 1024**2, 4 * 1024**3


def _expected_registration():
    return dict(schema='student-dagger-collection-prereg-v1', base_commit=BASE_COMMIT,
        prior_registration_commit=PRIOR_COMMIT, completed_root_sha256=ROOT_SHA256,
        roles=list(sampling.ROLES), opponents=list(sampling.OPPONENTS), sides=[0, 1],
        windows=[list(w) for w in sampling.WINDOWS], candidate_steps_per_window=4,
        source_games=24, source_passes=2, max_windows=144, max_comparisons=MAX_COMPARISONS,
        max_controller_calls=MAX_CALLS, max_official_transitions=MAX_TRANSITIONS,
        max_wall_seconds=WALL_SECONDS, initial_memory_bytes=previous.INITIAL_MEMORY,
        run_memory_bytes=previous.RUN_MEMORY, min_free_bytes=MIN_FREE_BYTES,
        max_bytes_per_split=MAX_DATA_BYTES, candidate_width_max=256, max_attempts=1,
        data_gate=dict(train_min=16, train_each_role_min=4, dev_min=8, dev_each_role_min=2,
                       min_windows_each_split=2, min_opponents_each_split=2),
        training=False, new_complete_games=0, independent_evaluation=False, promotion=False)


def prepare():
    if (previous._git('rev-parse', 'HEAD') != BASE_COMMIT or previous._git('diff', '--name-only')
            or previous._git('diff', '--cached', '--name-only') or REGISTRATION.exists() or PRIVATE.exists()):
        raise ValueError('Unexpected DAgger collection preparation state')
    prior_reg = previous._document(previous._read(PRIOR_REG))
    prior_binding_raw = previous._read(PRIOR / 'binding.local.json')
    prior_result_raw = previous._read(PRIOR / 'result.local.json')
    prior_closed_raw = previous._read(PRIOR / 'closed.local.json')
    prior_result, closed = previous._document(prior_result_raw), previous._document(prior_closed_raw)
    if (prior_result != previous._document(previous._read(PRIOR_LEDGER))
            or prior_reg.get('binding_commitment_sha256') != previous._sha(prior_binding_raw)
            or prior_result.get('summary', {}).get('registration_commit') != PRIOR_COMMIT
            or prior_result.get('summary', {}).get('status') != 'failed'
            or closed.get('outcome') != 'completed' or closed.get('retry_allowed') is not False):
        raise ValueError('Prior failed data gate evidence differs')
    binding = dict(schema='student-dagger-collection-private-binding-v1', salt=secrets.token_hex(32),
        prior_binding_sha256=previous._sha(prior_binding_raw), prior_result_sha256=previous._sha(prior_result_raw),
        prior_closed_sha256=previous._sha(prior_closed_raw), completed_root_sha256=ROOT_SHA256)
    raw = previous._encoded(binding)
    registration = {**_expected_registration(), 'binding_commitment_sha256': previous._sha(raw)}
    PRIVATE.mkdir(parents=False, exist_ok=False)
    with (PRIVATE / 'binding.local.json').open('xb') as target:
        if target.write(raw) != len(raw):
            raise OSError('Incomplete private binding write')
    previous._write_new(REGISTRATION, registration)
    print('Private DAgger binding and public registration prepared.')


def _runtime_binding():
    head = previous._git('rev-parse', 'HEAD')
    if (previous._git('status', '--porcelain') or previous._git('rev-parse', 'origin/main') != head
            or previous._git('rev-parse', 'HEAD^') != BASE_COMMIT
            or previous._git('merge-base', '--is-ancestor', BASE_COMMIT, head)):
        raise ValueError('The exact preregistration commit must be clean and published')
    raw = previous._read(REGISTRATION)
    committed = subprocess.check_output(['git', '-C', str(REPO), 'show',
                                         head + ':' + REGISTRATION.relative_to(REPO).as_posix()])
    reg = previous._document(raw)
    if raw != committed or any(reg.get(k) != v for k, v in _expected_registration().items()):
        raise ValueError('DAgger collection registration differs')
    binding_raw = previous._read(PRIVATE / 'binding.local.json')
    if previous._sha(binding_raw) != reg.get('binding_commitment_sha256'):
        raise ValueError('Private collection commitment differs')
    binding = previous._document(binding_raw)
    if (binding.get('schema') != 'student-dagger-collection-private-binding-v1'
            or binding.get('completed_root_sha256') != ROOT_SHA256
            or binding.get('prior_binding_sha256') != previous._sha(previous._read(PRIOR / 'binding.local.json'))
            or binding.get('prior_result_sha256') != previous._sha(previous._read(PRIOR / 'result.local.json'))
            or binding.get('prior_closed_sha256') != previous._sha(previous._read(PRIOR / 'closed.local.json'))):
        raise ValueError('Historical private binding differs')
    return head


def _save_dataset(path, examples, guard):
    """Write one private ragged split with no pickle or public feature output."""
    import numpy as np
    guard()
    if len(examples) > 144:
        raise ValueError('Dataset group budget exceeded')
    offsets = [0]
    for item in examples:
        offsets.append(offsets[-1] + len(item.features))
    if offsets[-1] > 144 * 256:
        raise ValueError('Dataset candidate budget exceeded')
    features = (np.concatenate([x.features for x in examples], axis=0) if examples
                else np.empty((0, 373), dtype=np.float32))
    arrays = dict(features=features, offsets=np.asarray(offsets, dtype=np.int32),
        teacher=np.asarray([x.teacher_index for x in examples], dtype=np.int32),
        student=np.asarray([x.student_index for x in examples], dtype=np.int32),
        third=np.asarray([x.third_index for x in examples], dtype=np.int32),
        role=np.asarray([sampling.ROLES.index(x.role) for x in examples], dtype=np.int8),
        opponent=np.asarray([sampling.OPPONENTS.index(x.opponent) for x in examples], dtype=np.int8),
        side=np.asarray([x.side for x in examples], dtype=np.int8),
        window=np.asarray([x.window for x in examples], dtype=np.int8),
        step=np.asarray([x.step for x in examples], dtype=np.int16),
        ordinal=np.asarray([x.ordinal for x in examples], dtype=np.int16))
    output = io.BytesIO()
    np.savez_compressed(output, **arrays)
    raw = output.getvalue()
    if not raw or len(raw) > MAX_DATA_BYTES:
        raise ValueError('Private dataset byte budget exceeded')
    with Path(path).open('xb') as target:
        if target.write(raw) != len(raw):
            raise OSError('Incomplete private dataset write')
    guard()
    return previous._sha(raw)


def load_completed():
    """Load only a sealed success; all returned content remains private.

    A provisional result or stdout alone is never a completion certificate.
    This check binds the plan, result and dataset contract; a training caller
    must separately hash the actual dataset bytes against that contract.
    """
    if (PRIVATE / 'failed.local.json').exists():
        raise ValueError('A failed attempt cannot be consumed')
    closed = previous._document(previous._read(PRIVATE / 'closed.local.json'))
    attempt = previous._document(previous._read(PRIVATE / 'attempt.local.json'))
    frozen = previous._document(previous._read(PRIVATE / 'plan-frozen.local.json'))
    result_raw = previous._read(PRIVATE / 'result.provisional.local.json')
    plan_raw = previous._read(PRIVATE / 'plan.local.json')
    contract_raw = previous._read(PRIVATE / 'dataset-contract.local.json')
    result = previous._document(result_raw)
    contract = previous._document(contract_raw)
    plan = previous._document(plan_raw)
    commit = closed.get('registration_commit')
    digests_valid = all(type(contract.get(k)) is str and len(contract[k]) == 64
                        and all(c in '0123456789abcdef' for c in contract[k])
                        for k in ('train_sha256', 'dev_sha256'))
    if (closed.get('schema') != 'student-dagger-closed-v1' or closed.get('outcome') != 'completed'
            or closed.get('retry_allowed') is not False
            or type(commit) is not str or len(commit) != 40
            or any(c not in '0123456789abcdef' for c in commit) or not digests_valid
            or closed.get('result_sha256') != previous._sha(result_raw)
            or closed.get('plan_sha256') != previous._sha(plan_raw)
            or closed.get('dataset_contract_sha256') != previous._sha(contract_raw)
            or attempt.get('schema') != 'student-dagger-attempt-v1' or attempt.get('retry_allowed') is not False
            or attempt.get('commit') != commit
            or frozen.get('schema') != 'student-dagger-plan-freeze-v1'
            or frozen.get('registration_commit') != commit or frozen.get('plan_sha256') != closed['plan_sha256']
            or plan.get('schema') != 'student-dagger-private-plan-v1'
            or type(plan.get('entries')) is not list or len(plan['entries']) != 144
            or contract.get('schema') != 'student-dagger-private-dataset-v1'
            or contract.get('model_role') != 'baseline'
            or contract.get('plan_sha256') != closed['plan_sha256']
            or contract.get('registration_commit') != closed.get('registration_commit')
            or contract.get('training_started') is not False
            or result.get('summary', {}).get('registration_commit') != commit
            or result.get('summary', {}).get('training') is not False
            or result.get('summary', {}).get('independent_evaluation') is not False
            or result.get('summary', {}).get('status') not in ('passed', 'failed')):
        raise ValueError('Completed collection evidence differs')
    return result, contract


def run():
    if os.name != 'nt' or previous.physical_available() < previous.INITIAL_MEMORY:
        raise MemoryError('Windows initial memory floor not met')
    if shutil.disk_usage(PRIVATE).free < MIN_FREE_BYTES + 2 * MAX_DATA_BYTES:
        raise OSError('Insufficient free space for private datasets')
    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[name] = '1'
    head = _runtime_binding()
    deadline = time.monotonic() + WALL_SECONDS
    guard = lambda: previous._guard(deadline)
    from decision.api import observe
    from decision.audit_market_teacher import observation
    from learning.task_student_continuation_v1 import compare, official_step
    from . import dataset, planning, sources

    guard()
    previous._write_new(PRIVATE / 'attempt.local.json', dict(schema='student-dagger-attempt-v1',
        commit=head, max_comparisons=MAX_COMPARISONS, retry_allowed=False))
    rows, examples = [], []
    comparisons = calls = transitions = 0
    try:
        context = sources.open_context(guard=guard)
        mean, scale, schema = context.model_normalization()
        guard()

        def cells():
            for role in sampling.ROLES:
                for opponent in sampling.OPPONENTS:
                    for side in (0, 1):
                        guard()
                        with sources.read_game(context, role=role, opponent=opponent, side=side, guard=guard) as game:
                            _accounting, windows = game.scan_candidates(guard=guard)
                            for window, candidates in enumerate(windows):
                                yield dict(role=role, opponent=opponent, side=side,
                                           window=window, candidates=candidates)

        plan = planning.create_plan(cells(), schema=schema, mean=mean, scale=scale, guard=guard)
        plan_path = PRIVATE / 'plan.local.json'
        previous._write_new(plan_path, plan)
        plan_raw = previous._read(plan_path)
        plan_sha = previous._sha(plan_raw)
        if previous._document(plan_raw) != plan:
            raise ValueError('Private plan readback differs')
        previous._write_new(PRIVATE / 'plan-frozen.local.json', dict(schema='student-dagger-plan-freeze-v1',
            registration_commit=head, plan_sha256=plan_sha))
        entries = iter(plan['entries'])

        for role in sampling.ROLES:
            for opponent in sampling.OPPONENTS:
                for side in (0, 1):
                    guard()
                    if previous._sha(previous._read(plan_path)) != plan_sha:
                        raise ValueError('Frozen plan changed')
                    with sources.read_game(context, role=role, opponent=opponent, side=side, guard=guard) as game:
                        _accounting, windows = game.scan_candidates(guard=guard)
                        for window, candidates in enumerate(windows):
                            guard()
                            entry = next(entries)
                            if tuple(entry.get(k) for k in ('role', 'opponent', 'side', 'window')) != (
                                    role, opponent, side, window):
                                raise ValueError('Frozen plan coordinate differs')
                            selected = planning.verify_entry(entry, candidates, schema=schema,
                                                             mean=mean, scale=scale, guard=guard)
                            row = dict(role=role, opponent=opponent, side=side, window=window,
                                       selection=entry['status'], teacher=None, third=None, eligible=False)
                            if selected is None:
                                rows.append(row)
                                continue
                            record = selected['document']['records'][selected['ordinal']]
                            inputs, step = record['inputs'], selected['step']
                            row.update(step=step, ordinal=selected['ordinal'],
                                       candidate_width=len(inputs['candidates']))
                            original = game.replay['steps'][step + 1][side]['action']
                            view = observe(observation(game.replay['steps'][step], side), game.replay['configuration'])
                            if 'seed' in view.configuration or 'info' in view.observation:
                                raise ValueError('Policy view contains privileged metadata')
                            for label, alternative in (('teacher', 'teacher'),
                                                       ('third', selected['third_candidate'])):
                                if alternative is None:
                                    continue
                                if comparisons >= MAX_COMPARISONS or calls + 2 > MAX_CALLS or transitions + 2 > MAX_TRANSITIONS:
                                    raise ValueError('DAgger collection work budget exceeded')
                                comparisons += 1
                                calls += 2
                                result = compare.compare_turn(selected['document'], view.observation,
                                    view.configuration, original, selected['ordinal'],
                                    task_sha256=game.task_sha256, crop_sha256=game.crop_sha256,
                                    game_limits=game.game_limits, agent_factory=game.make_agent, guard=guard,
                                    previous_document=selected['previous_document'], alternative=alternative)
                                if result.reference_action is None or result.alternative_action is None:
                                    raise ValueError('Comparison lacks verified complete actions')
                                transitions += 2
                                official = official_step.compare_step(game.replay, step=step, side=side,
                                    baseline_request=result.reference_action,
                                    intervention_request=result.alternative_action, guard=guard)
                                public = result.public_summary()
                                public['official'] = official
                                row[label] = public
                            example = dataset.make_example(role=role, opponent=opponent, side=side, window=window,
                                step=step, ordinal=selected['ordinal'], inputs=inputs,
                                teacher_index=entry['teacher_index'], student_index=entry['student_index'],
                                third_index=entry['third_index'], teacher_summary=row['teacher'],
                                third_summary=row['third'])
                            examples.append(example)
                            if example is not None:
                                row['eligible'] = True
                            rows.append(row)
                            del record, selected, inputs, original, view, result, example
                    del game
        try:
            next(entries)
        except StopIteration:
            pass
        else:
            raise ValueError('Frozen plan has extra entries')
        guard()
        if len(rows) != 144 or previous._sha(previous._read(plan_path)) != plan_sha:
            raise ValueError('Frozen plan or outcome grid differs')
        train, dev, summary = dataset.build_dataset(examples, mean=mean, scale=scale)
        selected_count = sum(row['selection'] == 'selected' for row in rows)
        eligible_count = sum(row['eligible'] is True for row in rows)
        if (summary.get('source_count') != selected_count or summary.get('eligible_count') != eligible_count
                or summary.get('ineligible_count') != selected_count - eligible_count):
            raise ValueError('Dataset counts do not match selected window records')
        if shutil.disk_usage(PRIVATE).free < MIN_FREE_BYTES + 2 * MAX_DATA_BYTES:
            raise OSError('Insufficient free space for private datasets')
        train_sha = _save_dataset(PRIVATE / 'train.local.npz', train, guard)
        dev_sha = _save_dataset(PRIVATE / 'dev.local.npz', dev, guard)
        contract_path = PRIVATE / 'dataset-contract.local.json'
        previous._write_new(contract_path, dict(
            schema='student-dagger-private-dataset-v1', registration_commit=head,
            plan_sha256=plan_sha, train_sha256=train_sha, dev_sha256=dev_sha,
            model_role='baseline', training_started=False))
        contract_sha = previous._sha(previous._read(contract_path))
        summary.update(source_games=24, source_passes=2, windows=len(rows), plan=plan['summary'],
            comparisons=comparisons, controller_call_limit_reserved=calls,
            official_transition_limit_reserved=transitions, registration_commit=head,
            training=False, independent_evaluation=False)
        result_path = PRIVATE / 'result.provisional.local.json'
        previous._write_new(result_path, dict(summary=summary, rows=rows))
        result_sha = previous._sha(previous._read(result_path))
        print(json.dumps(summary, ensure_ascii=False, sort_keys=True), flush=True)
        guard()
        previous._write_new(PRIVATE / 'closed.local.json', dict(schema='student-dagger-closed-v1',
            outcome='completed', retry_allowed=False, registration_commit=head, plan_sha256=plan_sha,
            result_sha256=result_sha, dataset_contract_sha256=contract_sha))
    except BaseException as exc:
        previous._write_new(PRIVATE / 'failed.local.json', dict(outcome='failed', error_type=type(exc).__name__,
            comparisons_started=comparisons, controller_calls_reserved=calls,
            official_transitions_reserved=transitions, retry_allowed=False))
        raise


class _PrivateArgumentParser(argparse.ArgumentParser):
    def error(self, message):
        # argparse normally echoes invalid arguments, which can contain private paths.
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
