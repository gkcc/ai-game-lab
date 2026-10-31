"""One privately bound, preregistered, serial paired-game experiment.

Imports do not read previous games or candidates. Only ``prepare`` binds the
already-consumed screening result; only a published registration can run.
Every game, replay and large input operation belongs to one owned Windows Job.
"""
import argparse
from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import re
import secrets
import shutil
import stat
import subprocess
import sys
import time

from learning.task_student_continuation_v1 import experiment as base
from learning.task_teacher_counterfactual_screen_v1 import experiment as screen
from learning.task_multi_teacher_probe_v1 import experiment as supervisor
from learning.task_teacher_transfer_dataset_v1 import experiment as transfer
from learning.task_pair_fresh_multi_v1 import prepare as fresh

REPO = base.REPO
GAME = fresh.GAME
PRIVATE = REPO.parent / 'private' / 'task-teacher-counterfactual-pair-v1-20261031'
STAGE = GAME / 'artifacts' / 'task-teacher-counterfactual-pair-v1-20261031'
PREFIX = 'kaggriculture/learning/task_teacher_counterfactual_pair_v1/'
REGISTRATION = REPO / PREFIX / 'registration.json'
BASE_COMMIT = '4d950a80f599be14d3b067fadc86d101f538a629'
SCREEN_COMMIT = '3c02823a9e1692ef452f7e7d092dd5cd21619b4b'
PAIR_FILES = tuple(PREFIX + x for x in (
    '__init__.py', 'gate.py', 'actor.py', 'checks.py', 'audit.py', 'worker.py',
    'experiment.py', 'test_gate.py', 'test_actor.py', 'test_checks.py',
    'test_audit.py', 'test_worker.py', 'test_experiment.py', 'REGISTRATION.zh-CN.md'))
SOURCE_NAMES = tuple(dict.fromkeys(screen.SOURCE_NAMES + PAIR_FILES + (
    'kaggriculture/run_match.py', 'kaggriculture/main.py',
    'kaggriculture/learning/task_anchor_once_v1/storage.py',
    'kaggriculture/learning/task_smoke_weight_probe_v1/core.py')))
PHASES = ('prepare_case', 'control', 'control_audit', 'control_verify',
          'intervene', 'intervene_audit', 'pair_verify')
MIB = 1024**2
PHASE_BYTES = dict(prepare_case=8*MIB, control=160*MIB, control_audit=MIB,
                   control_verify=MIB, intervene=160*MIB, intervene_audit=MIB,
                   pair_verify=MIB)
JOB = dict(supervisor.JOB)
TOTAL_SECONDS = 8*60*60
MIN_DISK = 4*1024**3
OUTPUT_BYTES = 8*1024**3
PARENT_RESERVE = 16*MIB
INITIAL_MEMORY = 5*1024**3
MAX_METADATA = 8*MIB
MAX_ONE_FILE = 512*MIB
OUTPUT_COUNT = 10000
_HEX = re.compile(r'[0-9a-f]{64}\Z')
_REL = re.compile(r'[A-Za-z0-9_.-]+\Z')


def _require(value):
    if not value:
        raise ValueError('Private paired experiment contract differs')


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _read(path, limit=MAX_METADATA):
    with Path(path).open('rb') as stream:
        raw = stream.read(limit + 1)
    _require(len(raw) <= limit)
    return raw


def _write(path, value):
    raw = base._encoded(value)
    with Path(path).open('xb') as stream:
        if stream.write(raw) != len(raw):
            raise OSError('Incomplete private evidence')
        stream.flush()
        os.fsync(stream.fileno())
    _require(_read(path) == raw)
    return _sha(raw)


def _git(*args):
    return subprocess.check_output(['git', '-C', str(REPO), *args],
                                   stderr=subprocess.DEVNULL, timeout=30).decode('utf-8').strip()


def _source_hashes():
    return {name: _sha(_read(REPO / name, limit=MIB)) for name in SOURCE_NAMES}


def _screen_snapshot(guard):
    """Read the exact previously sealed plan/result bytes; never open a trace."""
    guard()
    closed_path = screen.PRIVATE / 'closed.local.json'
    before = _read(closed_path)
    verified = screen.load_completed(guard=guard)
    after = _read(closed_path)
    _require(before == after)
    closed = base._document(after)
    _require(closed.get('registration_commit') == SCREEN_COMMIT)
    receipts = closed.get('receipts')
    _require(type(receipts) is dict)
    plan_raw = _read(screen.PRIVATE / 'private-plan.local.json')
    result_raw = _read(screen.PRIVATE / 'worker-result.local.json')
    _require(_sha(plan_raw) == receipts.get('private-plan')
             and _sha(result_raw) == receipts.get('worker-result'))
    plan, result = base._document(plan_raw), base._document(result_raw)
    _require(screen._validate_documents(SCREEN_COMMIT, plan, result) == verified
             and result['gate']['status'] == 'sufficient_for_separate_design')
    old_binding, old_binding_sha = screen._registered_binding(SCREEN_COMMIT)
    _require(_read(closed_path) == before)
    guard()
    return plan, old_binding, dict(screen_closed_sha256=_sha(before),
                                   screen_plan_sha256=_sha(plan_raw),
                                   screen_result_sha256=_sha(result_raw),
                                   screen_binding_sha256=old_binding_sha)


def _binding(plan, old, identity, *, salt, default_sha):
    _require(type(salt) is str and _HEX.fullmatch(salt) is not None
             and type(default_sha) is str and _HEX.fullmatch(default_sha) is not None)
    cases = []
    for row in plan['cases']:
        candidate = row['candidate']
        if candidate is None:
            continue
        case = {key: candidate[key] for key in ('role', 'slot', 'opponent', 'side',
            'window', 'rank', 'step', 'ordinal', 'input_sha256', 'original_choice', 'teacher_choice')}
        case.update(seed=old['roles'][candidate['role']], task_sha256=old['baseline_sha256'],
                    crop_sha256=old['crop_sha256'])
        cases.append(case)
    _require(4 <= len(cases) <= 8 and {x['role'] for x in cases} == set(screen.screen.ROLES)
             and len({(x['role'], x['opponent'], x['side']) for x in cases}) == len(cases))
    units = [dict(case_index=i, phase=phase) for i in range(len(cases)) for phase in PHASES]
    return dict(schema='teacher-counterfactual-pair-private-binding-v1', salt=salt,
        screen_commit=SCREEN_COMMIT, screen_identity=identity, software=old['software'],
        default_bot_sha256=default_sha, cases=cases, execution_order=units)


def _registration(sources, commitment, software):
    return dict(schema='teacher-counterfactual-pair-prereg-v1', base_commit=BASE_COMMIT,
        source_sha256=sources, private_binding_sha256=commitment, screen_commit=SCREEN_COMMIT,
        official_rule_commit=screen.ENGINE_COMMIT, official_rule_sha256=screen.ENGINE_SHA,
        official_core_sha256=screen.CORE_SHA, software=screen._validate_software(software),
        candidate_min=4, candidate_max=8, max_games=16, max_official_replays=16,
        max_official_transitions=(16+16)*719, action_turns=719, replay_states=720,
        phases=list(PHASES), output_bytes=OUTPUT_BYTES, parent_reserve_bytes=PARENT_RESERVE,
        phase_write_bytes=dict(PHASE_BYTES), initial_available_bytes=INITIAL_MEMORY,
        minimum_available_bytes=base.RUN_MEMORY, minimum_disk_bytes=MIN_DISK,
        job=dict(JOB), total_wall_seconds=TOTAL_SECONDS, max_attempts=1,
        signal='all_technical_pass_and_mean_own_delta_positive_and_mean_margin_and_score_delta_nonnegative',
        role_summary='private_descriptive_only_not_a_gate_or_independent_test',
        selection='all_sealed_candidates_fixed_order_no_replacement',
        training=False, gpu=False, ppo=False, new_seeds=False,
        independent_evaluation=False, promotion=False)


def _prepare_status():
    return all(line.startswith('?? ') and line[3:].replace('\\', '/') in PAIR_FILES
               for line in _git('status', '--porcelain', '--untracked-files=all').splitlines())


def prepare():
    _require(os.name == 'nt' and _git('rev-parse', 'HEAD') == BASE_COMMIT
             and _prepare_status() and not PRIVATE.exists() and not STAGE.exists()
             and not REGISTRATION.exists())
    if base.physical_available() < INITIAL_MEMORY:
        raise MemoryError('Initial memory floor')
    deadline = time.monotonic() + 300
    guard = lambda: base._guard(deadline)
    plan, old, identity = _screen_snapshot(guard)
    default_sha = _sha(_read(GAME / 'main.py', limit=MIB))
    binding = _binding(plan, old, identity, salt=secrets.token_hex(32), default_sha=default_sha)
    _require(screen._software() == binding['software'])
    registration = _registration(_source_hashes(), _sha(base._encoded(binding)), binding['software'])
    PRIVATE.mkdir(parents=False, exist_ok=False)
    _write(PRIVATE / 'binding.local.json', binding)
    _write(REGISTRATION, registration)
    print(json.dumps(dict(status='prepared')), flush=True)


def _registered_binding(head, guard):
    _require(type(head) is str and re.fullmatch(r'[0-9a-f]{40}', head) is not None
             and _git('rev-parse', head + '^') == BASE_COMMIT)
    raw = subprocess.check_output(['git', '-C', str(REPO), 'show',
        head + ':' + REGISTRATION.relative_to(REPO).as_posix()], stderr=subprocess.DEVNULL, timeout=30)
    binding_raw = _read(PRIVATE / 'binding.local.json')
    value = base._document(binding_raw)
    _require(base._document(raw) == _registration(_source_hashes(), _sha(binding_raw), value['software']))
    plan, old, identity = _screen_snapshot(guard)
    expected = _binding(plan, old, identity, salt=value['salt'],
                        default_sha=_sha(_read(GAME / 'main.py', limit=MIB)))
    _require(value == expected and screen._software() == value['software'])
    return value, _sha(binding_raw)


def _runtime_binding(guard):
    head = _git('rev-parse', 'HEAD')
    _require(not _git('status', '--porcelain') and _git('rev-parse', 'origin/main') == head)
    binding, digest = _registered_binding(head, guard)
    return head, binding, digest


def _unit_name(index):
    _require(type(index) is int and 0 <= index < 8*len(PHASES))
    return f'u{index:03d}'


def _case_dir(index):
    _require(type(index) is int and 0 <= index < 8)
    return STAGE / f'c{index:02d}'


def _safe_path(root, relative):
    _require(type(relative) is str and relative and '\\' not in relative and ':' not in relative)
    p = PurePosixPath(relative)
    _require(not p.is_absolute() and all(x not in ('.', '..') and _REL.fullmatch(x)
             and x.rstrip(' .') == x and not PureWindowsPath(x).is_reserved() for x in p.parts)
             and str(p) == relative)
    root = Path(root)
    for directory in (root.parent, root):
        info = directory.lstat()
        _require(stat.S_ISDIR(info.st_mode) and not stat.S_ISLNK(info.st_mode)
                 and not getattr(info, 'st_file_attributes', 0) & 0x400)
    root = root.resolve(strict=True)
    current = root
    for part in p.parts:
        current = current / part
        info = current.lstat()
        _require(not stat.S_ISLNK(info.st_mode) and not getattr(info, 'st_file_attributes', 0) & 0x400)
    _require(current.resolve(strict=True).is_relative_to(root) and current.is_file())
    return current


def _file_sha(root, relative, guard):
    path = _safe_path(root, relative)
    digest, total = hashlib.sha256(), 0
    with path.open('rb') as stream:
        while True:
            guard()
            block = stream.read(MIB)
            if not block:
                break
            total += len(block)
            _require(total <= MAX_ONE_FILE)
            digest.update(block)
    guard()
    return digest.hexdigest()


def _tree_bytes(root, guard):
    if not root.exists():
        return 0
    total, count, stack = 0, 0, [root]
    while stack:
        guard()
        directory = stack.pop()
        info = directory.lstat()
        _require(stat.S_ISDIR(info.st_mode) and not getattr(info, 'st_file_attributes', 0) & 0x400)
        with os.scandir(directory) as entries:
            for entry in entries:
                count += 1
                _require(count <= OUTPUT_COUNT)
                info = entry.stat(follow_symlinks=False)
                _require(not stat.S_ISLNK(info.st_mode) and not getattr(info, 'st_file_attributes', 0) & 0x400)
                if stat.S_ISDIR(info.st_mode):
                    stack.append(Path(entry.path))
                else:
                    _require(stat.S_ISREG(info.st_mode))
                    total += info.st_size
                    _require(total <= OUTPUT_BYTES)
    return total


def _guard(deadline):
    base._guard(deadline)
    if shutil.disk_usage(GAME / 'artifacts').free < MIN_DISK:
        raise OSError('Disk floor')
    if shutil.disk_usage(PRIVATE).free < MIN_DISK:
        raise OSError('Disk floor')


def _disk_gate(guard):
    used = _tree_bytes(PRIVATE, guard) + _tree_bytes(STAGE, guard)
    _require(used <= OUTPUT_BYTES)
    remaining = OUTPUT_BYTES - used
    if (shutil.disk_usage(GAME / 'artifacts').free < MIN_DISK + remaining
            or shutil.disk_usage(PRIVATE).free < MIN_DISK + remaining):
        raise OSError('Disk reservation')
    return used


def _required_paths(phase):
    if phase == 'prepare_case':
        return {'gate.local.json', 'sources/actor.py', 'sources/gate.py', 'agents/control.py',
                'agents/intervene.py', 'opponents/opponent.py'}
    if phase in ('control', 'intervene'):
        return {f'games/{phase}/replay.json', f'games/{phase}/result.json',
                f'traces/{phase}.jsonl.gz', f'events/{phase}.target.local.json',
                f'events/{phase}.complete.local.json'}
    return set()


def _validate_account(value):
    _require(type(value) is dict and set(value) == {'control', 'intervention', 'delta'})
    fields = {'own', 'opponent', 'margin', 'score'}
    for item in value.values():
        _require(type(item) is dict and set(item) == fields and all(type(x) in (int, float)
                 and math.isfinite(x) for x in item.values()))
    for arm in ('control', 'intervention'):
        margin, score = value[arm]['margin'], value[arm]['score']
        _require(score == (1 if margin > 0 else 0.5 if margin == 0 else 0))
    for key in fields:
        _require(math.isclose(value['delta'][key], value['intervention'][key]-value['control'][key],
                              rel_tol=0, abs_tol=1e-9))
    _require(math.isclose(value['delta']['margin'], value['delta']['own']-value['delta']['opponent'],
                          rel_tol=0, abs_tol=1e-9))
    return value


def _phase_result(value, phase):
    _require(type(value) is dict and set(value) == {'passed', 'phase', 'outputs', 'details'}
             and value['passed'] is True and value['phase'] == phase
             and type(value['outputs']) is dict and type(value['details']) is dict
             and _required_paths(phase) <= set(value['outputs'])
             and len(value['outputs']) <= OUTPUT_COUNT
             and all(type(k) is str and type(v) is str and _HEX.fullmatch(v) is not None
                     for k, v in value['outputs'].items()))
    if phase == 'prepare_case':
        _require(set(value['outputs']) == _required_paths(phase))
    elif phase in ('control', 'intervene'):
        _require(all(path in _required_paths(phase) or path.startswith(f'games/{phase}/')
                     for path in value['outputs']))
    else:
        _require(not value['outputs'])
    detail = value['details']
    if phase == 'prepare_case':
        _require(detail == {'prepared': True})
    elif phase in ('control', 'intervene'):
        _require(set(detail) == {'states','action_turns','both_done','agent_errors'}
                 and type(detail.get('states')) is int and detail['states'] == 720
                 and type(detail.get('action_turns')) is int and detail['action_turns'] == 719
                 and detail.get('both_done') is True and type(detail.get('agent_errors')) is int
                 and detail['agent_errors'] == 0)
    elif phase in ('control_audit', 'intervene_audit'):
        _require(set(detail) == {'action_turns','shed_order_frames_checked','semantic_sha256','semantic_sha256_v2'}
                 and all(type(detail.get(k)) is str and _HEX.fullmatch(detail[k]) is not None
                         for k in ('semantic_sha256','semantic_sha256_v2'))
                 and type(detail.get('action_turns')) is int and detail['action_turns'] == 719
                 and type(detail.get('shed_order_frames_checked')) is int
                 and detail['shed_order_frames_checked'] == 720)
    elif phase == 'control_verify':
        _require(detail == {'control_verified': True})
    elif phase == 'pair_verify':
        _require(set(detail) == {'pair_verified','account'} and detail.get('pair_verified') is True)
        _validate_account(detail.get('account'))
    return value


def _verify_outputs(case_dir, outputs, guard):
    for relative, expected in outputs.items():
        _require(_file_sha(case_dir, relative, guard) == expected)


def _claim_path(sequence):
    return PRIVATE / ('claim-' + _unit_name(sequence) + '.local.json')


def _accepted_path(sequence):
    return PRIVATE / ('accepted-' + _unit_name(sequence) + '.local.json')


def _child_result_path(sequence):
    return PRIVATE / ('child-result-' + _unit_name(sequence) + '.local.json')


def _previous(sequence, claim, *, head, binding):
    if sequence == 0:
        _require(claim['previous_sha256'] is None)
        return
    raw = _read(_accepted_path(sequence-1))
    _require(_sha(raw) == claim['previous_sha256'])
    prior = base._document(raw)
    _require(prior.get('schema') == 'teacher-pair-accepted-v1' and prior.get('commit') == head
             and prior.get('sequence') == sequence-1 and prior.get('passed') is True
             and prior.get('unit') == binding['execution_order'][sequence-1]
             and prior.get('retry_allowed') is False)


@contextmanager
def _owned_writes_with_subprocess_null(directory, limit):
    """Retain the file guard while permitting subprocess to open only NUL.

    Windows subprocess uses os.open(os.devnull, os.O_RDWR) for DEVNULL. The
    sole exception is that exact device and flag combination; ordinary file
    descriptors remain subject to owned_writes' refusal. This is scoped to a
    single-threaded child and does not turn the adapter into a native quota.
    """
    from learning.task_anchor_once_v1.storage import owned_writes
    _require(os.name == 'nt' and os.devnull == 'nul')
    null_path, null_flags = 'nul', os.O_RDWR
    raw_open = os.open
    with owned_writes(directory, limit) as budget:
        guarded_open = os.open
        def open_with_null(path, flags, mode=0o777, *, dir_fd=None):
            if type(path) is str and path == null_path and flags == null_flags and dir_fd is None:
                return raw_open(path, flags, mode)
            return guarded_open(path, flags, mode, dir_fd=dir_fd)
        os.open = open_with_null
        try:
            yield budget
        finally:
            os.open = guarded_open


def child(sequence):
    deadline = time.monotonic() + JOB['wall_seconds']
    guard = lambda: _guard(deadline)
    head, binding, digest = _runtime_binding(guard)
    _require(type(sequence) is int and 0 <= sequence < len(binding['execution_order']))
    unit = binding['execution_order'][sequence]
    claim = base._document(_read(_claim_path(sequence)))
    _require(claim.get('schema') == 'teacher-pair-claim-v1' and claim.get('commit') == head
        and claim.get('sequence') == sequence and claim.get('unit') == unit
        and claim.get('attempt') == 1 and claim.get('retry_allowed') is False
        and claim.get('freeze_sha256') == digest and type(claim.get('limit_bytes')) is int
        and 0 <= claim['limit_bytes'] <= PHASE_BYTES[unit['phase']]
        and type(claim.get('deadline')) in (int, float) and math.isfinite(claim['deadline'])
        and time.monotonic() < claim['deadline']
        and not (PRIVATE / 'failed.local.json').exists() and not (PRIVATE / 'closed.local.json').exists())
    _previous(sequence, claim, head=head, binding=binding)
    from learning.task_teacher_dev_deviation_v1.membership import ENV_KEY, verify_worker
    from learning.task_pair_fresh_multi_v1.entry import verify_live_owner
    import psutil
    owner = base._document(_read(PRIVATE / 'owner.local.json'))
    verify_worker(os.environ.get(ENV_KEY), claim, digest)
    verify_live_owner(owner, commit=head, binding_sha=digest, current_pid=os.getpid(), process=psutil.Process,
                      executable=sys.executable, samefile=os.path.samefile, now=time.monotonic())
    _write(PRIVATE / ('child-attempt-' + _unit_name(sequence) + '.local.json'),
           dict(schema='teacher-pair-child-attempt-v1', commit=head, sequence=sequence, retry_allowed=False))
    deadline = min(deadline, claim['deadline'])
    guard = lambda: _guard(deadline)
    budget = None
    try:
        from learning.task_student_continuation_v1 import official_step
        from . import worker
        screen._engine_identity(official_step, guard)
        directory = _case_dir(unit['case_index'])
        with _owned_writes_with_subprocess_null(directory, claim['limit_bytes']) as budget:
            result = worker.run_phase(stage=directory, phase=unit['phase'],
                                      case=binding['cases'][unit['case_index']], guard=guard)
        storage = budget.snapshot()
        _require(type(storage) is dict and storage.get('limit_bytes') == claim['limit_bytes']
                 and type(storage.get('used_bytes')) is int and 0 <= storage['used_bytes'] <= claim['limit_bytes']
                 and storage.get('refused') == 0)
        _phase_result(result, unit['phase'])
        screen._engine_identity(official_step, guard)
        _require(_runtime_binding(guard) == (head, binding, digest))
        _write(_child_result_path(sequence), dict(schema='teacher-pair-child-result-v1', commit=head,
            sequence=sequence, unit=unit, result=result, storage=storage, retry_allowed=False))
    except BaseException as exc:
        try:
            _write(PRIVATE / ('child-failed-' + _unit_name(sequence) + '.local.json'),
                   dict(schema='teacher-pair-child-failed-v1', commit=head, sequence=sequence,
                        category=type(exc).__name__, storage=(budget.snapshot() if budget else None),
                        retry_allowed=False))
        finally:
            raise


def _signal(accounts):
    _require(type(accounts) is list and 4 <= len(accounts) <= 8)
    for account in accounts:
        _validate_account(account)
    means = {key: math.fsum(x['delta'][key] for x in accounts)/len(accounts)
             for key in ('own', 'opponent', 'margin', 'score')}
    return dict(means=means, descriptive_signal=(means['own'] > 0 and means['margin'] >= 0
                                                and means['score'] >= 0))


def _role_summary(accounts, cases):
    _require(type(accounts) is list and type(cases) is list and len(accounts) == len(cases)
             and {case['role'] for case in cases} == set(screen.screen.ROLES))
    result = {}
    for role in screen.screen.ROLES:
        selected = [account for account, case in zip(accounts, cases) if case['role'] == role]
        _require(bool(selected))
        for account in selected:
            _validate_account(account)
        result[role] = {key: math.fsum(x['delta'][key] for x in selected)/len(selected)
                        for key in ('own', 'opponent', 'margin', 'score')}
    return result


def run():
    if os.name != 'nt' or base.physical_available() < INITIAL_MEMORY:
        raise MemoryError('Initial memory floor')
    deadline = time.monotonic() + TOTAL_SECONDS
    guard = lambda: _guard(deadline)
    head, binding, digest = _runtime_binding(guard)
    _require(not STAGE.exists() and not (PRIVATE/'attempt.local.json').exists()
             and not (PRIVATE/'failed.local.json').exists() and not (PRIVATE/'closed.local.json').exists())
    _disk_gate(guard)
    for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[key] = '1'
    from learning.task_teacher_dev_deviation_v1.membership import token_hash
    from learning.task_pair_fresh_multi_v1.entry import owner_marker
    import psutil
    _write(PRIVATE/'attempt.local.json', dict(schema='teacher-pair-attempt-v1', commit=head,
                                             retry_allowed=False))
    completed = []; accounts = []; charged = 0; previous_sha = None; outputs = {}
    active = None
    try:
        STAGE.mkdir(parents=False, exist_ok=False)
        process = psutil.Process(os.getpid())
        owner = base._document(owner_marker(commit=head, binding_sha=digest, pid=os.getpid(),
            created_us=int(process.create_time()*1_000_000), deadline=deadline))
        _write(PRIVATE/'owner.local.json', owner)
        for sequence, unit in enumerate(binding['execution_order']):
            guard()
            if base.physical_available() < INITIAL_MEMORY:
                raise MemoryError('Between-unit memory floor')
            _disk_gate(guard)
            phase, index = unit['phase'], unit['case_index']
            directory = _case_dir(index)
            if phase == 'prepare_case':
                directory.mkdir(parents=False, exist_ok=False)
                outputs[index] = {}
            _verify_outputs(directory, outputs[index], guard)
            limit = min(PHASE_BYTES[phase], OUTPUT_BYTES-PARENT_RESERVE-charged)
            _require(limit > 0)
            remaining = deadline-time.monotonic()-2*JOB['cleanup_seconds']
            if remaining <= 0:
                raise TimeoutError('Total deadline')
            wall = min(JOB['wall_seconds'], remaining)
            job_name = 'Local\\TeacherDeviation-' + secrets.token_hex(16)
            claim = dict(schema='teacher-pair-claim-v1', commit=head, sequence=sequence, unit=unit,
                attempt=1, retry_allowed=False, freeze_sha256=digest, previous_sha256=previous_sha,
                limit_bytes=limit, deadline=min(deadline, time.monotonic()+wall),
                job_token_sha256=token_hash(job_name))
            claim_sha = _write(_claim_path(sequence), claim)
            active = sequence
            command = [sys.executable, '-B', '-m', 'learning.task_teacher_counterfactual_pair_v1.experiment',
                       'child', str(sequence)]
            job = supervisor._supervise_owned(command, str(REPO/'kaggriculture'),
                                               dict(JOB, wall_seconds=wall), job_name)
            job_sha = _write(PRIVATE/('job-'+_unit_name(sequence)+'.local.json'), job)
            if not transfer._job_ok(job):
                raise RuntimeError('Owned Job did not finish cleanly')
            raw = _read(_child_result_path(sequence))
            receipt = base._document(raw)
            _require(receipt.get('schema') == 'teacher-pair-child-result-v1' and receipt.get('commit') == head
                and receipt.get('sequence') == sequence and receipt.get('unit') == unit
                and receipt.get('retry_allowed') is False)
            result = _phase_result(receipt['result'], phase)
            storage = receipt['storage']
            _require(type(storage) is dict and storage.get('limit_bytes') == limit
                and type(storage.get('used_bytes')) is int and 0 <= storage['used_bytes'] <= limit
                and storage.get('refused') == 0)
            charged += storage['used_bytes']
            _require(charged <= OUTPUT_BYTES-PARENT_RESERVE)
            # Previously accepted files (especially the frozen opponent and
            # gate) remain immutable across each subsequent worker phase.
            _verify_outputs(directory, outputs[index], guard)
            _verify_outputs(directory, result['outputs'], guard)
            for relative, sha in result['outputs'].items():
                _require(relative not in outputs[index] or outputs[index][relative] == sha)
                outputs[index][relative] = sha
            if phase == 'pair_verify':
                accounts.append(result['details']['account'])
            marker_raw = _read(PRIVATE/('child-attempt-'+_unit_name(sequence)+'.local.json'))
            marker = base._document(marker_raw)
            _require(marker == dict(schema='teacher-pair-child-attempt-v1', commit=head,
                                    sequence=sequence, retry_allowed=False))
            accepted = dict(schema='teacher-pair-accepted-v1', passed=True, commit=head,
                sequence=sequence, unit=unit, previous_sha256=previous_sha, claim_sha256=claim_sha,
                child_attempt_sha256=_sha(marker_raw), child_result_sha256=_sha(raw), job_sha256=job_sha,
                charged_bytes=charged, retry_allowed=False)
            previous_sha = _write(_accepted_path(sequence), accepted)
            completed.append(sequence)
            active = None
        guard()
        _require(len(completed) == len(binding['execution_order']) and len(accounts) == len(binding['cases']))
        for index, files in outputs.items():
            _verify_outputs(_case_dir(index), files, guard)
        _disk_gate(guard)
        _require(_runtime_binding(guard) == (head, binding, digest))
        result = dict(schema='teacher-pair-private-result-v1', commit=head, accounts=accounts,
            signal=_signal(accounts), role_summary=_role_summary(accounts, binding['cases']),
            technical_pass=True, training=False, gpu=False, ppo=False,
            new_seeds=False, independent_evaluation=False, promotion=False)
        result_sha = _write(PRIVATE/'result.provisional.local.json', result)
        _write(PRIVATE/'closed.local.json', dict(schema='teacher-pair-closed-v1', commit=head,
            binding_sha256=digest, result_sha256=result_sha, final_accepted_sha256=previous_sha,
            retry_allowed=False))
        print(json.dumps(dict(status='completed')), flush=True)
    except BaseException as exc:
        _write(PRIVATE/'failed.local.json', dict(schema='teacher-pair-failed-v1', commit=head,
            category=type(exc).__name__, completed_units=len(completed), active_sequence=active,
            active_unit_progress=('unknown' if active is not None else 'none'),
            confirmed_charged_bytes=charged, retry_allowed=False))
        raise


def load_completed(*, guard=lambda: None):
    """Revalidate the private completion chain; never publish returned data."""
    _require(callable(guard) and not (PRIVATE/'failed.local.json').exists())
    guard()
    closed = base._document(_read(PRIVATE/'closed.local.json'))
    _require(set(closed) == {'schema', 'commit', 'binding_sha256', 'result_sha256',
                             'final_accepted_sha256', 'retry_allowed'}
             and closed['schema'] == 'teacher-pair-closed-v1' and closed['retry_allowed'] is False)
    head = closed['commit']
    binding, digest = _registered_binding(head, guard)
    _require(closed['binding_sha256'] == digest)
    attempt = base._document(_read(PRIVATE/'attempt.local.json'))
    _require(attempt == dict(schema='teacher-pair-attempt-v1', commit=head, retry_allowed=False))
    owner = base._document(_read(PRIVATE/'owner.local.json'))
    _require(set(owner) == {'schema','registration_commit','binding_sha256','pid','created_us','deadline'}
             and owner['schema'] == 'task-pair-fresh-multi-owner-v1' and owner['registration_commit'] == head
             and owner['binding_sha256'] == digest and type(owner['pid']) is int and owner['pid'] > 0
             and type(owner['created_us']) is int and owner['created_us'] > 0
             and type(owner['deadline']) in (int, float) and math.isfinite(owner['deadline']))
    previous_sha, charged, accounts, outputs = None, 0, [], {}
    for sequence, unit in enumerate(binding['execution_order']):
        guard()
        _require(not (PRIVATE/('child-failed-'+_unit_name(sequence)+'.local.json')).exists())
        phase, index = unit['phase'], unit['case_index']
        directory = _case_dir(index)
        if phase == 'prepare_case':
            _require(index not in outputs)
            outputs[index] = {}
        claim_raw = _read(_claim_path(sequence))
        claim = base._document(claim_raw)
        limit = min(PHASE_BYTES[phase], OUTPUT_BYTES-PARENT_RESERVE-charged)
        _require(type(claim) is dict and set(claim) == {'schema','commit','sequence','unit','attempt',
                 'retry_allowed','freeze_sha256','previous_sha256','limit_bytes','deadline','job_token_sha256'}
                 and claim['schema'] == 'teacher-pair-claim-v1' and claim['commit'] == head
                 and claim['sequence'] == sequence and claim['unit'] == unit and claim['attempt'] == 1
                 and claim['retry_allowed'] is False and claim['freeze_sha256'] == digest
                 and claim['previous_sha256'] == previous_sha and type(claim['limit_bytes']) is int
                 and claim['limit_bytes'] == limit and type(claim['deadline']) in (int, float)
                 and math.isfinite(claim['deadline']) and claim['deadline'] <= owner['deadline']
                 and type(claim['job_token_sha256']) is str and _HEX.fullmatch(claim['job_token_sha256']))
        job_raw = _read(PRIVATE/('job-'+_unit_name(sequence)+'.local.json'))
        _require(transfer._job_ok(base._document(job_raw)))
        marker_raw = _read(PRIVATE/('child-attempt-'+_unit_name(sequence)+'.local.json'))
        _require(base._document(marker_raw) == dict(schema='teacher-pair-child-attempt-v1', commit=head,
                                                    sequence=sequence, retry_allowed=False))
        result_raw = _read(_child_result_path(sequence))
        child_result = base._document(result_raw)
        _require(type(child_result) is dict and set(child_result) == {'schema','commit','sequence','unit',
                 'result','storage','retry_allowed'} and child_result['schema'] == 'teacher-pair-child-result-v1'
                 and child_result['commit'] == head and child_result['sequence'] == sequence
                 and child_result['unit'] == unit and child_result['retry_allowed'] is False)
        value = _phase_result(child_result['result'], phase)
        storage = child_result['storage']
        _require(type(storage) is dict and storage.get('limit_bytes') == limit
                 and type(storage.get('used_bytes')) is int and 0 <= storage['used_bytes'] <= limit
                 and storage.get('refused') == 0)
        charged += storage['used_bytes']
        _require(charged <= OUTPUT_BYTES-PARENT_RESERVE)
        _verify_outputs(directory, outputs[index], guard)
        _verify_outputs(directory, value['outputs'], guard)
        for relative, sha in value['outputs'].items():
            _require(relative not in outputs[index] or outputs[index][relative] == sha)
            outputs[index][relative] = sha
        if phase == 'pair_verify':
            accounts.append(value['details']['account'])
        accepted_raw = _read(_accepted_path(sequence))
        accepted = base._document(accepted_raw)
        _require(accepted == dict(schema='teacher-pair-accepted-v1', passed=True, commit=head,
            sequence=sequence, unit=unit, previous_sha256=previous_sha, claim_sha256=_sha(claim_raw),
            child_attempt_sha256=_sha(marker_raw), child_result_sha256=_sha(result_raw), job_sha256=_sha(job_raw),
            charged_bytes=charged, retry_allowed=False))
        previous_sha = _sha(accepted_raw)
    _require(len(accounts) == len(binding['cases']) and previous_sha == closed['final_accepted_sha256'])
    raw = _read(PRIVATE/'result.provisional.local.json')
    _require(_sha(raw) == closed['result_sha256'])
    result = base._document(raw)
    expected = dict(schema='teacher-pair-private-result-v1', commit=head, accounts=accounts,
        signal=_signal(accounts), role_summary=_role_summary(accounts, binding['cases']),
        technical_pass=True, training=False, gpu=False, ppo=False,
        new_seeds=False, independent_evaluation=False, promotion=False)
    _require(result == expected)
    guard()
    _require(not (PRIVATE/'failed.local.json').exists())
    return result


class _PrivateParser(argparse.ArgumentParser):
    def error(self, _message):
        raise ValueError('Invalid command')


def main():
    parser = _PrivateParser()
    parser.add_argument('command', choices=('prepare', 'run', 'child'))
    parser.add_argument('sequence', nargs='?', type=int)
    args = parser.parse_args()
    if args.command == 'child':
        _require(args.sequence is not None)
        child(args.sequence)
    else:
        _require(args.sequence is None)
        (prepare if args.command == 'prepare' else run)()


if __name__ == '__main__':
    try:
        main()
    except (Exception, KeyboardInterrupt) as exc:
        kind = ('memory' if isinstance(exc, MemoryError) else 'timeout' if isinstance(exc, TimeoutError)
                else 'io' if isinstance(exc, OSError) else 'invalid_or_runtime')
        print(json.dumps(dict(status='failed', category=kind)), file=sys.stderr)
        raise SystemExit(1) from None
