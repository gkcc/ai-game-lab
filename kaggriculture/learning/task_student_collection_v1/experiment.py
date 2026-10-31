"""One bounded collection of conditional demonstrations from consumed games.

Raw observations, feature arrays, model identities and seeds remain private.
The old binding validation indirectly reads the private exclusion inventory;
reserved seeds never select these samples or enter the model. This stage does
not train a model or run a new complete game.
"""
import argparse
import io
import json
import os
from pathlib import Path
import secrets
import shutil
import subprocess
import time

from learning.task_student_continuation_v1 import experiment as previous
from learning.task_student_continuation_v1 import samples as prior_samples
from . import sampling


REPO = previous.REPO
PRIVATE = REPO.parent / 'private' / 'task-student-collection-v1-20261031'
PREVIOUS = REPO.parent / 'private' / 'task-student-continuation-v1-20261031'
FRESH = previous.FRESH
REGISTRATION = REPO / 'review/evidence/task-student-collection-v1-20261031/registration.json'
PREVIOUS_REGISTRATION = REPO / 'review/evidence/task-student-continuation-v1-20261031/registration.json'
PREVIOUS_LEDGER = REPO / 'review/evidence/task-student-continuation-v1-result-20261031/ledger.json'
BASE_COMMIT = 'a12c570f4018af4cfebbad5a74a20434b8901703'
PREVIOUS_COMMIT = '7d8001422c18359ab9e1a44b055758a322b420b2'
WALL_SECONDS = 3600
MAX_COMPARISONS, MAX_CALLS, MAX_TRANSITIONS = 192, 384, 384
MAX_DATA_BYTES = 64 * 1024 * 1024
MIN_FREE_BYTES = 4 * 1024**3


def _expected_registration():
    return dict(schema='student-collection-prereg-v1', base_commit=BASE_COMMIT,
        previous_registration_commit=PREVIOUS_COMMIT, roles=list(prior_samples.ROLES),
        opponents=list(prior_samples.OPPONENTS), sides=[0, 1], model='baseline',
        windows=[list(w) for w in sampling.WINDOWS], source_games=16, max_windows=96,
        max_comparisons=MAX_COMPARISONS, max_controller_calls=MAX_CALLS,
        max_official_transitions=MAX_TRANSITIONS, max_wall_seconds=WALL_SECONDS,
        initial_memory_bytes=previous.INITIAL_MEMORY, run_memory_bytes=previous.RUN_MEMORY,
        min_free_bytes=MIN_FREE_BYTES, max_bytes_per_split=MAX_DATA_BYTES,
        candidate_width_max=256, max_attempts=1, training=False, new_complete_games=0,
        independent_evaluation=False, promotion=False,
        data_gate=dict(train_min=16, train_each_role_min=4, dev_min=8, dev_each_role_min=2,
                       min_windows_each_split=2, min_opponents_each_split=2))


def prepare():
    if (previous._git('rev-parse', 'HEAD') != BASE_COMMIT or previous._git('diff', '--name-only')
            or previous._git('diff', '--cached', '--name-only')
            or REGISTRATION.exists() or PRIVATE.exists()):
        raise ValueError('Unexpected collection preparation state')
    prior_reg = previous._document(previous._read(PREVIOUS_REGISTRATION))
    prior_binding_raw = previous._read(PREVIOUS / 'binding.local.json')
    prior_result_raw = previous._read(PREVIOUS / 'result.local.json')
    prior_closed_raw = previous._read(PREVIOUS / 'closed.local.json')
    prior_result, closed = previous._document(prior_result_raw), previous._document(prior_closed_raw)
    if (prior_result != previous._document(previous._read(PREVIOUS_LEDGER))
            or prior_reg.get('binding_commitment_sha256') != previous._sha(prior_binding_raw)
            or prior_result.get('summary', {}).get('registration_commit') != PREVIOUS_COMMIT
            or prior_result.get('summary', {}).get('exploratory_gate') != 'passed'
            or closed.get('outcome') != 'completed' or closed.get('retry_allowed') is not False):
        raise ValueError('Previous diagnostic evidence differs')
    prior_binding = previous._document(prior_binding_raw)
    campaign_raw = previous._read(FRESH / 'campaign-result.local.json')
    if previous._sha(campaign_raw) != prior_binding.get('campaign_result_sha256'):
        raise ValueError('Previous campaign identity differs')
    binding = dict(schema='student-collection-private-binding-v1', salt=secrets.token_hex(32),
        prior_binding_sha256=previous._sha(prior_binding_raw), prior_result_sha256=previous._sha(prior_result_raw),
        prior_closed_sha256=previous._sha(prior_closed_raw), campaign_result_sha256=previous._sha(campaign_raw))
    raw = previous._encoded(binding)
    registration = {**_expected_registration(), 'binding_commitment_sha256': previous._sha(raw)}
    PRIVATE.mkdir(parents=False, exist_ok=False)
    with (PRIVATE / 'binding.local.json').open('xb') as target:
        if target.write(raw) != len(raw):
            raise OSError('Incomplete collection binding write')
    previous._write_new(REGISTRATION, registration)
    print('Private collection binding and public registration prepared.')


def _runtime_binding():
    head = previous._git('rev-parse', 'HEAD')
    if (previous._git('status', '--porcelain') or previous._git('rev-parse', 'origin/main') != head
            or previous._git('rev-parse', 'HEAD^') != BASE_COMMIT
            or previous._git('merge-base', '--is-ancestor', BASE_COMMIT, head)):
        raise ValueError('The exact preregistration commit must be clean and published')
    raw = previous._read(REGISTRATION)
    committed = subprocess.check_output(['git', '-C', str(REPO), 'show',
                                         head + ':' + REGISTRATION.relative_to(REPO).as_posix()])
    registration = previous._document(raw)
    if raw != committed or any(registration.get(k) != v for k, v in _expected_registration().items()):
        raise ValueError('Collection registration differs')
    binding_raw = previous._read(PRIVATE / 'binding.local.json')
    if previous._sha(binding_raw) != registration.get('binding_commitment_sha256'):
        raise ValueError('Collection private commitment differs')
    binding = previous._document(binding_raw)
    if (binding.get('schema') != 'student-collection-private-binding-v1'
            or binding.get('prior_binding_sha256') != previous._sha(previous._read(PREVIOUS / 'binding.local.json'))
            or binding.get('prior_result_sha256') != previous._sha(previous._read(PREVIOUS / 'result.local.json'))
            or binding.get('prior_closed_sha256') != previous._sha(previous._read(PREVIOUS / 'closed.local.json'))
            or binding.get('campaign_result_sha256') != previous._sha(previous._read(FRESH / 'campaign-result.local.json'))):
        raise ValueError('Prior private collection sources differ')
    return head, binding


def _save_dataset(path, examples, guard):
    """Save one split only; features and labels are never published."""
    import numpy as np
    guard()
    if len(examples) > 96:
        raise ValueError('Dataset group budget exceeded')
    offsets = [0]
    for example in examples:
        offsets.append(offsets[-1] + len(example.features))
    if offsets[-1] > 96 * 256:
        raise ValueError('Dataset candidate budget exceeded')
    features = (np.concatenate([x.features for x in examples], axis=0) if examples
                else np.empty((0, 373), dtype=np.float32))
    arrays = dict(features=features, offsets=np.asarray(offsets, dtype=np.int32),
        teacher=np.asarray([x.teacher_index for x in examples], dtype=np.int32),
        student=np.asarray([x.student_index for x in examples], dtype=np.int32),
        third=np.asarray([x.third_index for x in examples], dtype=np.int32),
        role=np.asarray([prior_samples.ROLES.index(x.role) for x in examples], dtype=np.int8),
        opponent=np.asarray([prior_samples.OPPONENTS.index(x.opponent) for x in examples], dtype=np.int8),
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
            raise OSError('Incomplete dataset write')
    guard()
    return previous._sha(raw)


def run():
    if os.name != 'nt' or previous.physical_available() < previous.INITIAL_MEMORY:
        raise MemoryError('Windows initial memory floor not met')
    if shutil.disk_usage(PRIVATE).free < MIN_FREE_BYTES + 2 * MAX_DATA_BYTES:
        raise OSError('Insufficient free space for the private dataset')
    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[name] = '1'
    head, binding = _runtime_binding()
    deadline = time.monotonic() + WALL_SECONDS
    guard = lambda: previous._guard(deadline)
    from decision.api import observe
    from decision.audit_market_teacher import observation
    from learning.task_bc_v1.model import TaskNetwork
    from learning.task_dagger_once_v1.policy import BoundedAgent
    from learning.task_pair_dev_games_v1.panel import GAME_LIMITS
    from learning.task_student_continuation_v1 import compare, official_step, sources as old_sources
    from . import dataset, sources

    guard()
    context = old_sources.open_context(campaign_sha256=binding['campaign_result_sha256'], guard=guard)
    previous._write_new(PRIVATE / 'attempt.local.json', dict(schema='student-collection-attempt-v1',
        commit=head, max_comparisons=MAX_COMPARISONS, retry_allowed=False))
    rows, examples = [], []
    comparisons = calls = transitions = 0
    normalization = None
    model_schema = None
    try:
        for role in prior_samples.ROLES:
            for opponent in prior_samples.OPPONENTS:
                for side in (0, 1):
                    guard()
                    with sources.read_game(context, role=role, opponent=opponent, side=side, guard=guard) as game:
                        identity = game.identity
                        if identity['game_limits'] != GAME_LIMITS:
                            raise ValueError('Original game limits differ')
                        if normalization is None:
                            network = TaskNetwork(identity['task_path'], identity['task_sha256'])
                            normalization = (network.mean.copy(), network.scale.copy())
                            model_schema = {key: network.metadata[key] for key in
                                ('schema_version', 'schema_sha256', 'feature_names', 'shared_feature_names')}
                            del network
                        for window, selected in enumerate(game.selected):
                            guard()
                            row = dict(role=role, opponent=opponent, side=side, window=window,
                                       selection='empty' if selected is None else 'selected',
                                       teacher=None, third=None, eligible=False)
                            if selected is None:
                                rows.append(row)
                                continue
                            record = selected['document']['records'][selected['ordinal']]
                            inputs = record['inputs']
                            if any(inputs.get(k) != v for k, v in model_schema.items()):
                                raise ValueError('Feature schema differs from the frozen checkpoint')
                            candidates = inputs['candidates']
                            width = len(candidates)
                            if width > 256:
                                raise ValueError('Candidate width exceeds registered resource budget')
                            ids = [c['id'] for c in candidates]
                            teacher_index, student_index = ids.index(record['teacher_label']), ids.index(record['chosen'])
                            third_id = selected['third_candidate']
                            third_index = ids.index(third_id) if third_id is not None else None
                            step = selected['step']
                            row.update(step=step, ordinal=selected['ordinal'], candidate_width=width)
                            original = game.replay['steps'][step + 1][side]['action']
                            view = observe(observation(game.replay['steps'][step], side), game.replay['configuration'])
                            if 'seed' in view.configuration or 'info' in view.observation:
                                raise ValueError('Policy view contains privileged metadata')

                            def factory():
                                return BoundedAgent(identity['crop_path'], identity['crop_sha256'],
                                                    identity['task_path'], identity['task_sha256'], limits=GAME_LIMITS)

                            for label, alternative in (('teacher', 'teacher'), ('third', third_id)):
                                if alternative is None:
                                    continue
                                if comparisons >= MAX_COMPARISONS or calls + 2 > MAX_CALLS or transitions + 2 > MAX_TRANSITIONS:
                                    raise ValueError('Collection work budget exceeded')
                                comparisons += 1
                                calls += 2
                                result = compare.compare_turn(selected['document'], view.observation,
                                    view.configuration, original, selected['ordinal'],
                                    task_sha256=identity['task_sha256'], crop_sha256=identity['crop_sha256'],
                                    game_limits=GAME_LIMITS, agent_factory=factory, guard=guard,
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
                                teacher_index=teacher_index, student_index=student_index, third_index=third_index,
                                teacher_summary=row['teacher'], third_summary=row['third'])
                            examples.append(example)
                            if example is not None:
                                row['eligible'] = True
                            rows.append(row)
                            del record, selected, inputs, candidates, ids, original, view, result, factory, example
                        del identity
                    del game
        guard()
        if normalization is None or len(rows) != 96:
            raise ValueError('Collection grid is incomplete')
        train, dev, summary = dataset.build_dataset(examples, mean=normalization[0], scale=normalization[1])
        selected_count = sum(row['selection'] == 'selected' for row in rows)
        eligible_count = sum(row['eligible'] is True for row in rows)
        if (summary.get('source_count') != selected_count or summary.get('eligible_count') != eligible_count
                or summary.get('ineligible_count') != selected_count - eligible_count):
            raise ValueError('Dataset counts do not match selected window records')
        if shutil.disk_usage(PRIVATE).free < MIN_FREE_BYTES + 2 * MAX_DATA_BYTES:
            raise OSError('Insufficient free space for the private dataset')
        train_sha = _save_dataset(PRIVATE / 'train.local.npz', train, guard)
        dev_sha = _save_dataset(PRIVATE / 'dev.local.npz', dev, guard)
        contract = dict(schema='student-collection-private-dataset-v1', registration_commit=head,
                        train_sha256=train_sha, dev_sha256=dev_sha,
                        model_role='baseline', training_started=False)
        previous._write_new(PRIVATE / 'dataset-contract.local.json', contract)
        summary.update(source_games=16, windows=len(rows), comparisons=comparisons,
            controller_call_limit_reserved=calls, official_transition_limit_reserved=transitions,
            registration_commit=head, training=False, independent_evaluation=False)
        previous._write_new(PRIVATE / 'result.local.json', dict(summary=summary, rows=rows))
        previous._write_new(PRIVATE / 'closed.local.json', dict(outcome='completed', retry_allowed=False))
        print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    except BaseException as exc:
        previous._write_new(PRIVATE / 'failed.local.json', dict(outcome='failed', error_type=type(exc).__name__,
            comparisons_started=comparisons, controller_calls_reserved=calls,
            official_transitions_reserved=transitions, retry_allowed=False))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare', 'run'))
    args = parser.parse_args()
    prepare() if args.command == 'prepare' else run()


if __name__ == '__main__':
    main()
