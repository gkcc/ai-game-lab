"""Probe teacher target support at matching terminal-controller prefixes.

Only 24 preselected turns from four consumed replays are inspected. No model is
trained, no new complete game is played and no official transition is run.
"""
import argparse
import json
import os
from pathlib import Path
import secrets
import shutil
import subprocess
import sys
import time

from learning.task_student_continuation_v1 import experiment as previous
from learning.task_multi_teacher_probe_v1 import experiment as replay_probe
from learning.task_pair_dev_games_v1.panel import ENGINE_COMMIT, ENGINE_SHA


REPO = previous.REPO
PRIVATE = REPO.parent / 'private' / 'task-teacher-support-probe-v1-20261031'
REGISTRATION = REPO / 'review/evidence/task-teacher-support-probe-v1-20261031/registration.json'
PRIOR_LEDGER = REPO / 'review/evidence/task-multi-teacher-probe-v1-result-20261031/ledger.json'
BASE_COMMIT = 'edeb4473ae19d0abea62b3245ae8be0234f37feb'
PRIOR_COMMIT = '3448fa1fd0948ac6e324629ec8f7717f64984f07'
STEPS = (59, 179, 299, 419, 539, 659)
CASES = replay_probe.CASES
JOB, WALL_SECONDS, MIN_FREE_BYTES = replay_probe.JOB, 3600, 4 * 1024**3


def _registration():
    return dict(schema='teacher-support-probe-prereg-v1', base_commit=BASE_COMMIT,
        prior_registration_commit=PRIOR_COMMIT, engine_commit=ENGINE_COMMIT, engine_sha256=ENGINE_SHA,
        source_role='A01', source_batch=1, source_slot='T01', opponents=['market', 'round2'],
        learner_sides=[0, 1], games=4, steps=list(STEPS), sampled_turns=24, max_rule_calls=48,
        max_candidates_per_group=256, max_records_per_turn=32, initial_available_bytes=previous.INITIAL_MEMORY,
        run_available_bytes=previous.RUN_MEMORY, disk_free_floor_bytes=MIN_FREE_BYTES,
        total_wall_seconds=WALL_SECONDS, job=dict(JOB), max_attempts=1, training=False,
        new_complete_games=0, new_official_transitions=0, independent_evaluation=False, promotion=False)


def _prior_snapshot():
    prior = replay_probe.load_completed()
    public = previous._document(previous._read(PRIOR_LEDGER))
    if (prior.get('summary') != public.get('summary')
            or prior.get('summary', {}).get('registration_commit') != PRIOR_COMMIT
            or prior.get('summary', {}).get('status') != 'passed'
            or prior.get('summary', {}).get('turns') != 2876):
        raise ValueError('Completed teacher probe differs from published evidence')
    allowed = ('role', 'opponent', 'learner_side', 'teacher_side', 'replay_states', 'turns',
               'requests_matched', 'callbacks', 'max_candidate_width', 'operations',
               'official_transitions', 'training', 'passed')
    if public.get('cases') != [{key: case[key] for key in allowed} for case in prior['cases']]:
        raise ValueError('Published teacher case projection differs')
    result_raw = previous._read(replay_probe.PRIVATE / 'result.provisional.local.json')
    evidence_raw = previous._read(replay_probe.PRIVATE / 'evidence.local.json')
    closed_raw = previous._read(replay_probe.PRIVATE / 'closed.local.json')
    return dict(result_sha256=previous._sha(result_raw), evidence_sha256=previous._sha(evidence_raw),
                closed_sha256=previous._sha(closed_raw))


def prepare():
    if (previous._git('rev-parse', 'HEAD') != BASE_COMMIT or previous._git('diff', '--name-only')
            or previous._git('diff', '--cached', '--name-only') or REGISTRATION.exists() or PRIVATE.exists()):
        raise ValueError('Unexpected teacher-support preparation state')
    prior = _prior_snapshot()
    from learning.task_student_anchor_collection_v1 import sources
    context = sources.open_context(guard=lambda: None)
    if context.source_role(batch=1, role='T01') != 'A01':
        raise ValueError('Completed source role differs')
    binding = dict(schema='teacher-support-probe-private-binding-v1', salt=secrets.token_hex(32),
                   prior=prior, source_seed=context.private_role_binding()['A01'],
                   task_sha256=context.task_sha256, crop_sha256=context.crop_sha256)
    raw = previous._encoded(binding)
    PRIVATE.mkdir(parents=False, exist_ok=False)
    previous._write_new(PRIVATE / 'binding.local.json', binding)
    previous._write_new(REGISTRATION, {**_registration(), 'binding_commitment_sha256': previous._sha(raw)})
    print('Private support binding and public registration prepared.')


def _runtime_binding():
    head = previous._git('rev-parse', 'HEAD')
    if (previous._git('status', '--porcelain') or previous._git('rev-parse', 'origin/main') != head
            or previous._git('rev-parse', 'HEAD^') != BASE_COMMIT
            or previous._git('merge-base', '--is-ancestor', BASE_COMMIT, head)):
        raise ValueError('The exact support registration must be clean and published')
    raw = previous._read(REGISTRATION)
    committed = subprocess.check_output(['git', '-C', str(REPO), 'show',
        head + ':' + REGISTRATION.relative_to(REPO).as_posix()])
    reg = previous._document(raw)
    if raw != committed or any(reg.get(k) != v for k, v in _registration().items()):
        raise ValueError('Support registration differs')
    binding_raw = previous._read(PRIVATE / 'binding.local.json')
    if previous._sha(binding_raw) != reg.get('binding_commitment_sha256'):
        raise ValueError('Support private binding differs')
    binding = previous._document(binding_raw)
    if binding.get('schema') != 'teacher-support-probe-private-binding-v1' or binding.get('prior') != _prior_snapshot():
        raise ValueError('Prior teacher evidence changed')
    return head, binding


COUNT_FIELDS = ('teacher_callbacks', 'terminal_callbacks', 'teacher_empty_turn', 'terminal_empty_turn',
                'supported', 'unsupported', 'unknown', 'ambiguous',
                'supported_target_features_equal', 'supported_target_features_differ')


def _validate_row(row, step):
    if (type(row) is not dict or set(row) != {'step', 'teacher_request_matched', 'terminal_request_equal', *COUNT_FIELDS}
            or row.get('step') != step or row.get('teacher_request_matched') is not True
            or type(row.get('terminal_request_equal')) is not bool
            or any(type(row.get(key)) is not int or row[key] < 0 for key in COUNT_FIELDS)
            or row['teacher_callbacks'] > 32 or row['terminal_callbacks'] > 32
            or row['teacher_empty_turn'] != int(row['teacher_callbacks'] == 0)
            or row['terminal_empty_turn'] != int(row['terminal_callbacks'] == 0)
            or row['ambiguous'] != 0
            or row['supported'] + row['unsupported'] + row['unknown'] != row['teacher_callbacks']
            or row['supported'] + row['unsupported'] > row['terminal_callbacks']
            or row['supported_target_features_equal'] + row['supported_target_features_differ'] != row['supported']):
        raise ValueError('Support turn aggregate differs')


def child(opponent, side):
    name = replay_probe._case_name(opponent, side)
    head, binding = _runtime_binding()
    if (PRIVATE / 'failed.local.json').exists() or (PRIVATE / 'closed.local.json').exists():
        raise ValueError('Support parent stage is already closed')
    claim = previous._document(previous._read(PRIVATE / ('claim-' + name + '.local.json')))
    attempt = previous._document(previous._read(PRIVATE / 'attempt.local.json'))
    binding_sha = previous._sha(previous._read(PRIVATE / 'binding.local.json'))
    if (claim.get('schema') != 'teacher-support-probe-claim-v1' or claim.get('commit') != head
            or claim.get('case') != name or claim.get('attempt') != 1
            or claim.get('retry_allowed') is not False or claim.get('freeze_sha256') != binding_sha
            or attempt != dict(schema='teacher-support-probe-attempt-v1', commit=head, retry_allowed=False)
            or (PRIVATE / ('case-' + name + '.local.json')).exists()
            or (PRIVATE / ('case-' + name + '.failed.local.json')).exists()):
        raise ValueError('Support child is not uniquely claimed')
    from learning.task_teacher_dev_deviation_v1.membership import ENV_KEY, verify_worker
    from learning.task_pair_fresh_multi_v1.entry import verify_live_owner
    import psutil
    owner = previous._document(previous._read(PRIVATE / 'owner.local.json'))
    verify_worker(os.environ.get(ENV_KEY), claim, binding_sha)
    verify_live_owner(owner, commit=head, binding_sha=binding_sha, current_pid=os.getpid(),
                      process=psutil.Process, executable=sys.executable, samefile=os.path.samefile,
                      now=time.monotonic())
    previous._write_new(PRIVATE / ('child-attempt-' + name + '.local.json'), dict(
        schema='teacher-support-probe-child-attempt-v1', commit=head, case=name, retry_allowed=False))
    for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[key] = '1'
    deadline = time.monotonic() + JOB['wall_seconds']
    guard = lambda: previous._guard(deadline)
    rows, calls = [], 0
    try:
        from decision.api import observe
        from decision.audit_market_teacher import observation
        from learning.task_request_gate_v1.policy import same_request
        from learning.task_student_anchor_collection_v1 import sources
        from .opponent import SupportTeacher
        from .terminal import TerminalTeacherProbe
        from .alignment import align_turn
        context = sources.open_context(guard=guard)
        if (context.source_role(batch=1, role='T01') != 'A01'
                or context.private_role_binding()['A01'] != binding.get('source_seed')
                or context.task_sha256 != binding.get('task_sha256')
                or context.crop_sha256 != binding.get('crop_sha256')):
            raise ValueError('Completed support source binding differs')
        _mean, _scale, metadata = context.model_normalization()
        schema = {key: metadata[key] for key in
                  ('schema_version', 'schema_sha256', 'feature_names', 'shared_feature_names')}
        with sources.read_game(context, batch=1, role='T01', opponent=opponent, side=side,
                               guard=guard) as game:
            raw, expected = replay_probe._rival_source(game, guard)
            replay = game.replay
            teacher_side = 1 - side
            if type(replay.get('steps')) is not list or len(replay['steps']) != 720:
                raise ValueError('Completed replay length differs')
            for step in STEPS:
                guard()
                if calls + 2 > len(STEPS) * 2:
                    raise ValueError('Support rule call limit exceeded')
                view = observe(observation(replay['steps'][step], teacher_side), replay['configuration'])
                if 'seed' in view.configuration or 'info' in view.observation:
                    raise ValueError('Policy view contains privileged metadata')
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
                aggregate = align_turn(view.observation, view.configuration, teacher_records,
                                       terminal_records, schema=schema, guard=guard)
                row = dict(step=step, teacher_request_matched=True,
                           terminal_request_equal=same_request(teacher_request, terminal_request), **aggregate)
                _validate_row(row, step)
                rows.append(row)
                teacher.clear()
                terminal.clear()
                del teacher, terminal, teacher_records, terminal_records, view, teacher_request, terminal_request
        totals = {key: sum(row[key] for row in rows) for key in COUNT_FIELDS}
        previous._write_new(PRIVATE / ('case-' + name + '.local.json'), dict(
            schema='teacher-support-probe-private-case-v1', registration_commit=head, role='A01',
            opponent=opponent, learner_side=side, teacher_side=teacher_side, replay_states=720,
            sampled_turns=len(rows), rule_calls=calls, rows=rows, totals=totals,
            official_transitions=0, training=False, passed=True))
    except BaseException as exc:
        previous._write_new(PRIVATE / ('case-' + name + '.failed.local.json'), dict(
            outcome='failed', error_type=type(exc).__name__, completed_turns=len(rows),
            rule_calls_started=calls, retry_allowed=False))
        raise


def _case_result(raw, *, head, opponent, side):
    result = previous._document(raw)
    rows = result.get('rows')
    if (set(result) != {'schema', 'registration_commit', 'role', 'opponent', 'learner_side', 'teacher_side',
                        'replay_states', 'sampled_turns', 'rule_calls', 'rows', 'totals',
                        'official_transitions', 'training', 'passed'}
            or result.get('schema') != 'teacher-support-probe-private-case-v1'
            or result.get('registration_commit') != head or result.get('role') != 'A01'
            or result.get('opponent') != opponent or type(result.get('learner_side')) is not int
            or result['learner_side'] != side or type(result.get('teacher_side')) is not int
            or result['teacher_side'] != 1 - side or result.get('replay_states') != 720
            or result.get('sampled_turns') != len(STEPS) or result.get('rule_calls') != 2 * len(STEPS)
            or result.get('official_transitions') != 0 or result.get('training') is not False
            or result.get('passed') is not True or type(rows) is not list or len(rows) != len(STEPS)):
        raise ValueError('Support case result differs')
    for step, row in zip(STEPS, rows):
        _validate_row(row, step)
    if result.get('totals') != {key: sum(row[key] for row in rows) for key in COUNT_FIELDS}:
        raise ValueError('Support case aggregate differs')
    return result


def _summary(head, cases):
    totals = {key: sum(case['totals'][key] for case in cases) for key in COUNT_FIELDS}
    return dict(status='completed', registration_commit=head, role='A01', games=len(cases),
        sampled_turns=sum(case['sampled_turns'] for case in cases),
        teacher_requests_matched=sum(len(case['rows']) for case in cases),
        terminal_full_requests_equal=sum(int(row['terminal_request_equal']) for case in cases for row in case['rows']),
        rule_calls=sum(case['rule_calls'] for case in cases), totals=totals,
        official_transitions=0, training=False, independent_evaluation=False)


def load_completed():
    if (PRIVATE / 'failed.local.json').exists():
        raise ValueError('A failed support probe cannot be consumed')
    closed = previous._document(previous._read(PRIVATE / 'closed.local.json'))
    attempt = previous._document(previous._read(PRIVATE / 'attempt.local.json'))
    evidence_raw = previous._read(PRIVATE / 'evidence.local.json')
    result_raw = previous._read(PRIVATE / 'result.provisional.local.json')
    evidence, result = previous._document(evidence_raw), previous._document(result_raw)
    head = closed.get('registration_commit')
    binding_sha = previous._sha(previous._read(PRIVATE / 'binding.local.json'))
    if (type(head) is not str or len(head) != 40
            or closed.get('schema') != 'teacher-support-probe-closed-v1'
            or closed.get('outcome') != 'completed' or closed.get('retry_allowed') is not False
            or closed.get('result_sha256') != previous._sha(result_raw)
            or closed.get('evidence_sha256') != previous._sha(evidence_raw)
            or attempt != dict(schema='teacher-support-probe-attempt-v1', commit=head, retry_allowed=False)
            or evidence.get('schema') != 'teacher-support-probe-private-evidence-v1'
            or evidence.get('registration_commit') != head or type(evidence.get('cases')) is not dict
            or set(evidence['cases']) != {replay_probe._case_name(o, s) for o, s in CASES}):
        raise ValueError('Support completion evidence differs')
    reconstructed = []
    for opponent, side in CASES:
        name = replay_probe._case_name(opponent, side)
        if (PRIVATE / ('case-' + name + '.failed.local.json')).exists():
            raise ValueError('A support case has a failure marker')
        item = evidence['cases'][name]
        case_raw = previous._read(PRIVATE / ('case-' + name + '.local.json'))
        job_raw = previous._read(PRIVATE / ('job-' + name + '.local.json'))
        claim = previous._document(previous._read(PRIVATE / ('claim-' + name + '.local.json')))
        child_attempt = previous._document(previous._read(PRIVATE / ('child-attempt-' + name + '.local.json')))
        if (type(item) is not dict or item.get('case_sha256') != previous._sha(case_raw)
                or item.get('job_sha256') != previous._sha(job_raw)
                or claim.get('schema') != 'teacher-support-probe-claim-v1' or claim.get('commit') != head
                or claim.get('case') != name or claim.get('attempt') != 1
                or claim.get('retry_allowed') is not False or claim.get('freeze_sha256') != binding_sha
                or type(claim.get('job_token_sha256')) is not str or len(claim['job_token_sha256']) != 64
                or child_attempt != dict(schema='teacher-support-probe-child-attempt-v1',
                                         commit=head, case=name, retry_allowed=False)):
            raise ValueError('Support case receipt differs')
        job = previous._document(job_raw)
        if (job.get('passed') is not True or job.get('launcher_reaped') is not True
                or job.get('job_members_zero') is not True or job.get('exit_code') != 0
                or job.get('retry_allowed') is not False or job.get('stop_reason') is not None
                or job.get('error_kind') is not None or job.get('cleanup_error_kind') is not None
                or type(job.get('rss_complete_samples')) is not int or job['rss_complete_samples'] < 1):
            raise ValueError('Support job did not close cleanly')
        reconstructed.append(_case_result(case_raw, head=head, opponent=opponent, side=side))
    if result.get('cases') != reconstructed or result.get('summary') != _summary(head, reconstructed):
        raise ValueError('Support sealed case list differs')
    return result


def run():
    if os.name != 'nt' or previous.physical_available() < previous.INITIAL_MEMORY:
        raise MemoryError('Windows initial memory floor not met')
    if shutil.disk_usage(PRIVATE).free < MIN_FREE_BYTES:
        raise OSError('Insufficient free space for support receipts')
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
        schema='teacher-support-probe-attempt-v1', commit=head, retry_allowed=False))
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
                raise MemoryError('Support between-case memory floor not met')
            active_case = replay_probe._case_name(opponent, side)
            job_name = 'Local\\TeacherDeviation-' + secrets.token_hex(16)
            previous._write_new(PRIVATE / ('claim-' + active_case + '.local.json'),
                dict(schema='teacher-support-probe-claim-v1', commit=head, case=active_case,
                     attempt=1, retry_allowed=False, freeze_sha256=binding_sha,
                     job_token_sha256=token_hash(job_name)))
            command = [sys.executable, '-B', '-m', 'learning.task_teacher_support_probe_v1.experiment',
                       'child', opponent, str(side)]
            remaining = deadline - time.monotonic() - 2 * JOB['cleanup_seconds']
            if remaining <= 0:
                raise TimeoutError('Support parent deadline exhausted')
            limits = dict(JOB, wall_seconds=min(JOB['wall_seconds'], remaining))
            job = replay_probe._supervise_owned(command, str(REPO / 'kaggriculture'), limits, job_name)
            job_path = PRIVATE / ('job-' + active_case + '.local.json')
            previous._write_new(job_path, job)
            if (job.get('passed') is not True or job.get('launcher_reaped') is not True
                    or job.get('job_members_zero') is not True or job.get('exit_code') != 0
                    or job.get('retry_allowed') is not False or job.get('stop_reason') is not None
                    or job.get('error_kind') is not None or job.get('cleanup_error_kind') is not None
                    or type(job.get('rss_complete_samples')) is not int or job['rss_complete_samples'] < 1):
                raise RuntimeError('Owned support case did not finish cleanly')
            case_path = PRIVATE / ('case-' + active_case + '.local.json')
            raw = previous._read(case_path)
            case = _case_result(raw, head=head, opponent=opponent, side=side)
            evidence[active_case] = dict(case_sha256=previous._sha(raw),
                                         job_sha256=previous._sha(previous._read(job_path)))
            completed.append(case)
            active_case = None
        guard()
        if len(completed) != 4:
            raise ValueError('Support case set incomplete')
        summary = _summary(head, completed)
        evidence_path = PRIVATE / 'evidence.local.json'
        previous._write_new(evidence_path, dict(schema='teacher-support-probe-private-evidence-v1',
                                                registration_commit=head, cases=evidence))
        result_path = PRIVATE / 'result.provisional.local.json'
        previous._write_new(result_path, dict(summary=summary, cases=completed))
        print(json.dumps(summary, ensure_ascii=False, sort_keys=True), flush=True)
        guard()
        previous._write_new(PRIVATE / 'closed.local.json', dict(schema='teacher-support-probe-closed-v1',
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
            raise ValueError('Support child coordinates required')
        child(args.opponent, args.side)
    else:
        if args.opponent is not None or args.side is not None:
            raise ValueError('Unexpected support coordinates')
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
