"""Reconstruct two frozen opponents on four consumed official replays.

This is a bounded label-interface probe, not model training or a new game.
Only public aliases and aggregate counts may leave the private local stage.
"""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import secrets
import shutil
import subprocess
import sys
import time

from learning.task_student_continuation_v1 import experiment as previous
from learning.task_pair_dev_games_v1.panel import ENGINE_COMMIT, ENGINE_SHA


REPO = previous.REPO
PRIVATE = REPO.parent / 'private' / 'task-multi-teacher-probe-v1-20261031'
FAILED_STAGE = REPO.parent / 'private' / 'task-student-anchor-mixed-v1-20261031'
REGISTRATION = REPO / 'review/evidence/task-multi-teacher-probe-v1-20261031/registration.json'
BASE_COMMIT = '0cd9d51e4feb75b63873ebc1659ea69385880b76'
ANCHOR_COMMIT = '1610ee975de25c0c1afa7a34facd54f1904ee331'
CASES = tuple((opponent, side) for opponent in ('market', 'round2') for side in (0, 1))
JOB = dict(max_rss_bytes=2 * 1024**3, max_commit_bytes=3 * 1024**3,
           min_available_bytes=4 * 1024**3, wall_seconds=900,
           poll_seconds=1, cleanup_seconds=10)
WALL_SECONDS, MIN_FREE_BYTES = 3600, 4 * 1024**3
MAX_CANDIDATES, MAX_RECORDS_PER_TURN = 256, 32


def _registration():
    return dict(schema='multi-teacher-probe-prereg-v1', base_commit=BASE_COMMIT,
        anchor_result_commit=ANCHOR_COMMIT, engine_commit=ENGINE_COMMIT, engine_sha256=ENGINE_SHA,
        source_role='A01', source_batch=1, source_slot='T01',
        opponents=['market', 'round2'], learner_sides=[0, 1], games=4,
        replay_states_per_game=720, turns_per_game=719, max_teacher_calls=2876,
        max_candidates_per_group=MAX_CANDIDATES, max_records_per_turn=MAX_RECORDS_PER_TURN,
        initial_available_bytes=previous.INITIAL_MEMORY, run_available_bytes=previous.RUN_MEMORY,
        disk_free_floor_bytes=MIN_FREE_BYTES, total_wall_seconds=WALL_SECONDS,
        job=dict(JOB), max_attempts=1, training=False, new_complete_games=0,
        new_official_transitions=0, independent_evaluation=False, promotion=False)


def _failed_source_snapshot():
    if (FAILED_STAGE / 'closed.local.json').exists():
        raise ValueError('Previous mixed attempt unexpectedly completed')
    failed_raw = previous._read(FAILED_STAGE / 'failed.local.json')
    frozen_raw = previous._read(FAILED_STAGE / 'plan-frozen.local.json')
    plan_raw = previous._read(FAILED_STAGE / 'plan.local.json')
    attempt_raw = previous._read(FAILED_STAGE / 'attempt.local.json')
    failed, frozen, attempt = map(previous._document, (failed_raw, frozen_raw, attempt_raw))
    if (failed.get('outcome') != 'failed' or failed.get('error_type') != 'MemoryError'
            or failed.get('retry_allowed') is not False or failed.get('comparisons_started') != 346
            or failed.get('controller_calls_reserved') != 692
            or failed.get('official_transitions_reserved') != 692
            or frozen.get('registration_commit') != '7301c921cfd8ae8e5317cc9b4834165b50dc8dac'
            or frozen.get('plan_sha256') != previous._sha(plan_raw)
            or attempt != dict(schema='student-anchor-mixed-attempt-v1',
                               commit='7301c921cfd8ae8e5317cc9b4834165b50dc8dac',
                               max_comparisons=480, retry_allowed=False)):
        raise ValueError('Previous mixed failure evidence differs')
    return dict(failed_sha256=previous._sha(failed_raw),
                plan_sha256=previous._sha(plan_raw), frozen_sha256=previous._sha(frozen_raw),
                attempt_sha256=previous._sha(attempt_raw))


def prepare():
    if (previous._git('rev-parse', 'HEAD') != BASE_COMMIT or previous._git('diff', '--name-only')
            or previous._git('diff', '--cached', '--name-only') or REGISTRATION.exists() or PRIVATE.exists()):
        raise ValueError('Unexpected teacher probe preparation state')
    failure = _failed_source_snapshot()
    from learning.task_student_anchor_collection_v1 import sources
    context = sources.open_context(guard=lambda: None)
    if context.source_role(batch=1, role='T01') != 'A01':
        raise ValueError('Completed teacher source role differs')
    binding = dict(schema='multi-teacher-probe-private-binding-v1', salt=secrets.token_hex(32),
        failed_source=failure, source_seed=context.private_role_binding()['A01'],
        task_sha256=context.task_sha256, crop_sha256=context.crop_sha256)
    raw = previous._encoded(binding)
    PRIVATE.mkdir(parents=False, exist_ok=False)
    previous._write_new(PRIVATE / 'binding.local.json', binding)
    previous._write_new(REGISTRATION, {**_registration(), 'binding_commitment_sha256': previous._sha(raw)})
    print('Private teacher binding and public registration prepared.')


def _runtime_binding():
    head = previous._git('rev-parse', 'HEAD')
    if (previous._git('status', '--porcelain') or previous._git('rev-parse', 'origin/main') != head
            or previous._git('rev-parse', 'HEAD^') != BASE_COMMIT
            or previous._git('merge-base', '--is-ancestor', BASE_COMMIT, head)):
        raise ValueError('The exact teacher registration must be clean and published')
    raw = previous._read(REGISTRATION)
    committed = subprocess.check_output(['git', '-C', str(REPO), 'show',
                                         head + ':' + REGISTRATION.relative_to(REPO).as_posix()])
    reg = previous._document(raw)
    if raw != committed or any(reg.get(k) != v for k, v in _registration().items()):
        raise ValueError('Teacher registration differs')
    binding_raw = previous._read(PRIVATE / 'binding.local.json')
    if previous._sha(binding_raw) != reg.get('binding_commitment_sha256'):
        raise ValueError('Private teacher binding differs')
    binding = previous._document(binding_raw)
    if (binding.get('schema') != 'multi-teacher-probe-private-binding-v1'
            or binding.get('failed_source') != _failed_source_snapshot()):
        raise ValueError('Previous mixed failure binding differs')
    return head, binding


def _case_name(opponent, side):
    if (opponent, side) not in CASES or type(side) is not int:
        raise ValueError('Unregistered teacher case')
    return opponent + '-learner' + str(side)


def _rival_source(game, guard):
    """Read the rival bytes through the completed row's authenticated digest."""
    from learning.task_student_anchor_collection_v1 import sources
    from learning.task_student_dagger_collection_v1 import sources as dagger
    item = game._item
    stage = sources.old.fresh.ARTIFACTS / sources.STAGE_NAME
    raw = dagger._assert_blob(sources.old.fresh.ARTIFACTS,
        stage / 'job-results' / (item.game + '.row.json'), item.hashes['row'], guard)
    row = dagger._json(raw)
    if row.get('opponent') != item.opponent:
        raise ValueError('Completed opponent identity differs')
    expected = dagger._digest(row.get('rival_sha256'))
    rival = dagger._assert_blob(sources.old.fresh.ARTIFACTS,
        stage / 'sources' / (item.opponent + '.py'), expected, guard)
    return rival, expected


def _check_record(record, view, schema):
    """Validate one teacher callback without persisting its private inputs."""
    import math
    from learning.task_request_gate_v1.policy import same_request
    if type(record) is not dict or type(record.get('inputs')) is not dict:
        raise ValueError('Invalid teacher callback')
    inputs = record['inputs']
    if (any(inputs.get(k) != schema.get(k) for k in
            ('schema_version', 'schema_sha256', 'feature_names', 'shared_feature_names'))
            or not same_request(inputs.get('observation'), view.observation)
            or not same_request(inputs.get('configuration'), view.configuration)
            or 'seed' in view.configuration or 'info' in view.observation):
        raise ValueError('Teacher input visibility or schema differs')
    shared, candidates = inputs.get('shared_x'), inputs.get('candidates')
    if (type(shared) is not list or len(shared) != 275 or type(candidates) is not list
            or not 1 <= len(candidates) <= MAX_CANDIDATES
            or any(type(x) not in (int, float) or not math.isfinite(x) for x in shared)):
        raise ValueError('Teacher input width differs')
    identifiers = []
    for candidate in candidates:
        if (type(candidate) is not dict or type(candidate.get('id')) is not str
                or type(candidate.get('x')) is not list or len(candidate['x']) != 98
                or any(type(x) not in (int, float) or not math.isfinite(x) for x in candidate['x'])):
            raise ValueError('Teacher candidate differs')
        identifiers.append(candidate['id'])
    if (len(set(identifiers)) != len(identifiers) or record.get('teacher_label') not in identifiers
            or record.get('chosen') != record.get('teacher_label')):
        raise ValueError('Teacher label is not the original candidate')
    return len(candidates), record.get('op')


def child(opponent, side):
    name = _case_name(opponent, side)
    head, binding = _runtime_binding()
    if (PRIVATE / 'failed.local.json').exists() or (PRIVATE / 'closed.local.json').exists():
        raise ValueError('Teacher parent stage is already closed')
    claim = previous._document(previous._read(PRIVATE / ('claim-' + name + '.local.json')))
    attempt = previous._document(previous._read(PRIVATE / 'attempt.local.json'))
    binding_sha = previous._sha(previous._read(PRIVATE / 'binding.local.json'))
    if (claim.get('schema') != 'multi-teacher-probe-claim-v1' or claim.get('commit') != head
            or claim.get('case') != name or claim.get('attempt') != 1
            or claim.get('retry_allowed') is not False or claim.get('freeze_sha256') != binding_sha
            or attempt != dict(schema='multi-teacher-probe-attempt-v1', commit=head, retry_allowed=False)
            or (PRIVATE / ('case-' + name + '.local.json')).exists()
            or (PRIVATE / ('case-' + name + '.failed.local.json')).exists()):
        raise ValueError('Teacher child is not uniquely claimed')
    from learning.task_teacher_dev_deviation_v1.membership import ENV_KEY, verify_worker
    from learning.task_pair_fresh_multi_v1.entry import verify_live_owner
    import psutil
    owner = previous._document(previous._read(PRIVATE / 'owner.local.json'))
    verify_worker(os.environ.get(ENV_KEY), claim, binding_sha)
    verify_live_owner(owner, commit=head, binding_sha=binding_sha, current_pid=os.getpid(),
                      process=psutil.Process, executable=sys.executable, samefile=os.path.samefile,
                      now=time.monotonic())
    previous._write_new(PRIVATE / ('child-attempt-' + name + '.local.json'), dict(
        schema='multi-teacher-probe-child-attempt-v1', commit=head, case=name, retry_allowed=False))
    for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[key] = '1'
    deadline = time.monotonic() + JOB['wall_seconds']
    guard = lambda: previous._guard(deadline)
    steps, callbacks, width_max = 0, 0, 0
    operations = Counter()
    try:
        from decision.api import observe
        from decision.audit_market_teacher import observation
        from learning.task_request_gate_v1.policy import same_request
        from learning.task_student_anchor_collection_v1 import sources
        from .teacher import TeacherProbe
        context = sources.open_context(guard=guard)
        if (context.source_role(batch=1, role='T01') != 'A01'
                or context.private_role_binding()['A01'] != binding.get('source_seed')
                or context.task_sha256 != binding.get('task_sha256')
                or context.crop_sha256 != binding.get('crop_sha256')):
            raise ValueError('Completed teacher source binding differs')
        _mean, _scale, metadata = context.model_normalization()
        schema = {key: metadata[key] for key in
                  ('schema_version', 'schema_sha256', 'feature_names', 'shared_feature_names')}
        with sources.read_game(context, batch=1, role='T01', opponent=opponent, side=side,
                               guard=guard) as game:
            raw, expected = _rival_source(game, guard)
            probe = TeacherProbe(raw, expected, opponent, guard=guard)
            replay = game.replay
            teacher_side = 1 - side
            if type(replay.get('steps')) is not list or len(replay['steps']) != 720:
                raise ValueError('Completed replay length differs')
            for step in range(719):
                guard()
                view = observe(observation(replay['steps'][step], teacher_side), replay['configuration'])
                actual = probe(view.observation, view.configuration)
                expected_action = replay['steps'][step + 1][teacher_side]['action']
                if not same_request(actual, expected_action):
                    raise ValueError('Teacher request does not reproduce the replay')
                records = probe.last.get('records') if type(probe.last) is dict else None
                if type(records) is not list or len(records) > MAX_RECORDS_PER_TURN:
                    raise ValueError('Teacher callback count exceeds limit')
                for record in records:
                    width, operation = _check_record(record, view, schema)
                    if operation not in ('DIG', 'FERTILIZE', 'HARVEST', 'PLANT', 'WATER'):
                        raise ValueError('Teacher operation differs')
                    callbacks += 1
                    width_max = max(width_max, width)
                    operations[operation] += 1
                if records:
                    del record
                probe.clear()
                steps += 1
                del view, actual, expected_action, records
            if steps != 719 or callbacks == 0:
                raise ValueError('Teacher callback coverage incomplete')
        previous._write_new(PRIVATE / ('case-' + name + '.local.json'), dict(
            schema='multi-teacher-probe-private-case-v1', registration_commit=head, role='A01',
            opponent=opponent, learner_side=side, teacher_side=teacher_side, replay_states=720,
            turns=steps, requests_matched=steps, callbacks=callbacks, max_candidate_width=width_max,
            operations={key: operations[key] for key in ('DIG', 'FERTILIZE', 'HARVEST', 'PLANT', 'WATER')},
            official_transitions=0, training=False, passed=True))
    except BaseException as exc:
        previous._write_new(PRIVATE / ('case-' + name + '.failed.local.json'), dict(
            outcome='failed', error_type=type(exc).__name__, completed_turns=steps,
            callbacks=callbacks, retry_allowed=False))
        raise


def _case_result(raw, *, head, opponent, side):
    result = previous._document(raw)
    operations = result.get('operations')
    if (result.get('schema') != 'multi-teacher-probe-private-case-v1'
            or result.get('registration_commit') != head or result.get('role') != 'A01'
            or result.get('opponent') != opponent or type(result.get('learner_side')) is not int
            or result['learner_side'] != side or type(result.get('teacher_side')) is not int
            or result['teacher_side'] != 1 - side
            or result.get('replay_states') != 720 or result.get('turns') != 719
            or result.get('requests_matched') != 719 or result.get('official_transitions') != 0
            or result.get('training') is not False or result.get('passed') is not True
            or type(result.get('callbacks')) is not int or not 0 < result['callbacks'] <= 719 * MAX_RECORDS_PER_TURN
            or type(result.get('max_candidate_width')) is not int
            or not 1 <= result['max_candidate_width'] <= MAX_CANDIDATES
            or type(operations) is not dict
            or set(operations) != {'DIG', 'FERTILIZE', 'HARVEST', 'PLANT', 'WATER'}
            or any(type(value) is not int or value < 0 for value in operations.values())
            or sum(operations.values()) != result['callbacks']):
        raise ValueError('Teacher case result differs')
    return result


def load_completed():
    """Load only a sealed four-case result and its actual local receipts."""
    if (PRIVATE / 'failed.local.json').exists():
        raise ValueError('A failed teacher probe cannot be consumed')
    closed = previous._document(previous._read(PRIVATE / 'closed.local.json'))
    attempt = previous._document(previous._read(PRIVATE / 'attempt.local.json'))
    evidence_raw = previous._read(PRIVATE / 'evidence.local.json')
    result_raw = previous._read(PRIVATE / 'result.provisional.local.json')
    evidence, result = previous._document(evidence_raw), previous._document(result_raw)
    head = closed.get('registration_commit')
    binding_sha = previous._sha(previous._read(PRIVATE / 'binding.local.json'))
    if (type(head) is not str or len(head) != 40
            or closed.get('schema') != 'multi-teacher-probe-closed-v1'
            or closed.get('outcome') != 'completed' or closed.get('retry_allowed') is not False
            or closed.get('result_sha256') != previous._sha(result_raw)
            or closed.get('evidence_sha256') != previous._sha(evidence_raw)
            or attempt != dict(schema='multi-teacher-probe-attempt-v1', commit=head, retry_allowed=False)
            or evidence.get('schema') != 'multi-teacher-probe-private-evidence-v1'
            or evidence.get('registration_commit') != head
            or type(evidence.get('cases')) is not dict
            or set(evidence['cases']) != {_case_name(opponent, side) for opponent, side in CASES}
            or result.get('summary', {}).get('registration_commit') != head
            or result.get('summary', {}).get('training') is not False
            or result.get('summary', {}).get('independent_evaluation') is not False
            or result.get('summary', {}).get('status') != 'passed'):
        raise ValueError('Teacher probe completion evidence differs')
    reconstructed = []
    for opponent, side in CASES:
        name = _case_name(opponent, side)
        if (PRIVATE / ('case-' + name + '.failed.local.json')).exists():
            raise ValueError('A teacher case has a failure marker')
        item = evidence['cases'][name]
        case_raw = previous._read(PRIVATE / ('case-' + name + '.local.json'))
        job_raw = previous._read(PRIVATE / ('job-' + name + '.local.json'))
        claim = previous._document(previous._read(PRIVATE / ('claim-' + name + '.local.json')))
        child_attempt = previous._document(previous._read(PRIVATE / ('child-attempt-' + name + '.local.json')))
        if (type(item) is not dict or item.get('case_sha256') != previous._sha(case_raw)
                or item.get('job_sha256') != previous._sha(job_raw)
                or claim.get('schema') != 'multi-teacher-probe-claim-v1' or claim.get('commit') != head
                or claim.get('case') != name or claim.get('attempt') != 1
                or claim.get('retry_allowed') is not False or claim.get('freeze_sha256') != binding_sha
                or type(claim.get('job_token_sha256')) is not str or len(claim['job_token_sha256']) != 64
                or child_attempt != dict(schema='multi-teacher-probe-child-attempt-v1',
                                         commit=head, case=name, retry_allowed=False)):
            raise ValueError('Teacher case receipt differs')
        job = previous._document(job_raw)
        if (job.get('passed') is not True or job.get('launcher_reaped') is not True
                or job.get('job_members_zero') is not True or job.get('exit_code') != 0
                or job.get('retry_allowed') is not False or job.get('stop_reason') is not None
                or job.get('error_kind') is not None or job.get('cleanup_error_kind') is not None
                or type(job.get('rss_complete_samples')) is not int or job['rss_complete_samples'] < 1):
            raise ValueError('Teacher job did not close cleanly')
        reconstructed.append(_case_result(case_raw, head=head, opponent=opponent, side=side))
    expected_summary = dict(status='passed', registration_commit=head, role='A01', games=4,
        turns=sum(item['turns'] for item in reconstructed),
        callbacks=sum(item['callbacks'] for item in reconstructed),
        requests_matched=sum(item['requests_matched'] for item in reconstructed),
        official_transitions=0, training=False, independent_evaluation=False)
    if result.get('cases') != reconstructed or result.get('summary') != expected_summary:
        raise ValueError('Teacher sealed case list differs')
    return result


def _supervise_owned(command, cwd, limits, job_name):
    """Use the existing named Windows Job and owned-process sampler."""
    from learning.task_teacher_dev_deviation_v1.membership import ENV_KEY, NamedWindowsAPI
    from learning.task_pair_dev_campaign_v1.campaign import quiet_api_type
    from learning.task_job_empty_exit_v1.sampling import supervise
    from learning.panel_eval_v3.launch import WindowsJob
    api = quiet_api_type(NamedWindowsAPI, WindowsJob.create_suspended)(job_name)
    prior = os.environ.get(ENV_KEY)
    os.environ[ENV_KEY] = job_name
    try:
        return supervise(api, command, cwd, dict(limits))
    finally:
        if prior is None:
            os.environ.pop(ENV_KEY, None)
        else:
            os.environ[ENV_KEY] = prior


def run():
    if os.name != 'nt' or previous.physical_available() < previous.INITIAL_MEMORY:
        raise MemoryError('Windows initial memory floor not met')
    if shutil.disk_usage(PRIVATE).free < MIN_FREE_BYTES:
        raise OSError('Insufficient free space for teacher receipts')
    head, _binding = _runtime_binding()
    for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[key] = '1'
    from learning.task_teacher_dev_deviation_v1.membership import token_hash
    from learning.task_pair_fresh_multi_v1.entry import owner_marker
    import psutil
    deadline = time.monotonic() + WALL_SECONDS
    guard = lambda: previous._guard(deadline)
    guard()
    previous._write_new(PRIVATE / 'attempt.local.json', dict(
        schema='multi-teacher-probe-attempt-v1', commit=head, retry_allowed=False))
    completed, evidence = [], {}
    active_case = None
    try:
        binding_sha = previous._sha(previous._read(PRIVATE / 'binding.local.json'))
        owner_process = psutil.Process(os.getpid())
        owner = previous._document(owner_marker(commit=head, binding_sha=binding_sha, pid=os.getpid(),
            created_us=int(owner_process.create_time() * 1_000_000), deadline=deadline))
        previous._write_new(PRIVATE / 'owner.local.json', owner)
        for opponent, side in CASES:
            guard()
            if previous.physical_available() < previous.INITIAL_MEMORY:
                raise MemoryError('Teacher between-case memory floor not met')
            active_case = _case_name(opponent, side)
            job_name = 'Local\\TeacherDeviation-' + secrets.token_hex(16)
            previous._write_new(PRIVATE / ('claim-' + active_case + '.local.json'),
                dict(schema='multi-teacher-probe-claim-v1', commit=head, case=active_case,
                     attempt=1, retry_allowed=False, freeze_sha256=binding_sha,
                     job_token_sha256=token_hash(job_name)))
            command = [sys.executable, '-B', '-m', 'learning.task_multi_teacher_probe_v1.experiment',
                       'child', opponent, str(side)]
            # Existing supervisor can spend one cleanup interval waiting for
            # the process and another for Job membership to become empty.
            remaining = deadline - time.monotonic() - 2 * JOB['cleanup_seconds']
            if remaining <= 0:
                raise TimeoutError('Teacher parent deadline exhausted')
            limits = dict(JOB, wall_seconds=min(JOB['wall_seconds'], remaining))
            job = _supervise_owned(command, str(REPO / 'kaggriculture'), limits, job_name)
            job_path = PRIVATE / ('job-' + active_case + '.local.json')
            previous._write_new(job_path, job)
            if (job.get('passed') is not True or job.get('launcher_reaped') is not True
                    or job.get('job_members_zero') is not True or job.get('exit_code') != 0
                    or job.get('retry_allowed') is not False or job.get('stop_reason') is not None
                    or job.get('error_kind') is not None or job.get('cleanup_error_kind') is not None
                    or type(job.get('rss_complete_samples')) is not int or job['rss_complete_samples'] < 1):
                raise RuntimeError('Owned teacher case did not finish cleanly')
            case_path = PRIVATE / ('case-' + active_case + '.local.json')
            raw = previous._read(case_path)
            case = _case_result(raw, head=head, opponent=opponent, side=side)
            evidence[active_case] = dict(case_sha256=previous._sha(raw),
                                         job_sha256=previous._sha(previous._read(job_path)))
            completed.append(case)
            active_case = None
        guard()
        if len(completed) != 4:
            raise ValueError('Teacher case set incomplete')
        summary = dict(status='passed', registration_commit=head, role='A01', games=4,
            turns=sum(item['turns'] for item in completed),
            callbacks=sum(item['callbacks'] for item in completed),
            requests_matched=sum(item['requests_matched'] for item in completed),
            official_transitions=0, training=False, independent_evaluation=False)
        evidence_path = PRIVATE / 'evidence.local.json'
        previous._write_new(evidence_path, dict(schema='multi-teacher-probe-private-evidence-v1',
                                                registration_commit=head, cases=evidence))
        result_path = PRIVATE / 'result.provisional.local.json'
        previous._write_new(result_path, dict(summary=summary, cases=completed))
        print(json.dumps(summary, ensure_ascii=False, sort_keys=True), flush=True)
        guard()
        previous._write_new(PRIVATE / 'closed.local.json', dict(schema='multi-teacher-probe-closed-v1',
            outcome='completed', retry_allowed=False, registration_commit=head,
            evidence_sha256=previous._sha(previous._read(evidence_path)),
            result_sha256=previous._sha(previous._read(result_path))))
    except BaseException as exc:
        previous._write_new(PRIVATE / 'failed.local.json', dict(outcome='failed',
            error_type=type(exc).__name__, completed_cases=len(completed),
            active_case=active_case, retry_allowed=False))
        raise


class _PrivateArgumentParser(argparse.ArgumentParser):
    def error(self, message):
        raise ValueError('invalid command line arguments')


def main():
    parser = _PrivateArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare', 'run', 'child'))
    parser.add_argument('opponent', nargs='?', choices=('market', 'round2'))
    parser.add_argument('side', nargs='?', type=int, choices=(0, 1))
    args = parser.parse_args()
    if args.command == 'child':
        if args.opponent is None or args.side is None:
            raise ValueError('Teacher child coordinates required')
        child(args.opponent, args.side)
    else:
        if args.opponent is not None or args.side is not None:
            raise ValueError('Unexpected teacher coordinates')
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
