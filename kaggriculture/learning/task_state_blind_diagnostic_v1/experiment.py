"""One preregistered, CPU-only diagnostic of consumed official traces.

Importing this module performs no source access. ``prepare`` binds metadata;
``run`` only accepts a subsequently committed and published registration.
Neither command trains a model or plays a game. Results remain private until
separately reviewed and projected for publication.
"""
import argparse
import hashlib
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

from learning.task_student_continuation_v1 import experiment as previous
from learning.task_multi_teacher_probe_v1 import experiment as supervisor
from learning.task_teacher_transfer_dataset_v1 import experiment as transfer
from learning.task_state_residual_v1 import experiment as state
from learning.task_state_train_fit_v1 import experiment as train_fit
from learning.task_pair_dev_games_v1.panel import ENGINE_COMMIT, ENGINE_SHA
from .sampling import OPPONENTS, ROLES, WINDOWS

REPO = previous.REPO
PRIVATE = REPO.parent / 'private' / 'task-state-blind-diagnostic-v1-20261031'
PREFIX = 'kaggriculture/learning/task_state_blind_diagnostic_v1/'
REGISTRATION = REPO / PREFIX / 'registration.json'
BASE_COMMIT = '5034737a8abcb6853d8cdbeabbd53211d64265df'
STATE_COMMIT = train_fit.STATE_COMMIT
JOB = dict(supervisor.JOB)
INITIAL_MEMORY, MIN_DISK, OUTPUT_RESERVE = 5 * 1024**3, 4 * 1024**3, 4 * 1024**2
CASES = tuple((role, slot, opponent, side) for role, slot in (('A05','T01'), ('A06','T02'))
              for opponent in OPPONENTS for side in (0, 1))
NEW_FILES = tuple(PREFIX + name for name in ('__init__.py','sampling.py','extract.py',
    'diagnostic.py','experiment.py','test_core.py','test_experiment.py','REGISTRATION.zh-CN.md'))
SOURCE_NAMES = tuple(dict.fromkeys(train_fit.SOURCE_NAMES + NEW_FILES + (
    'kaggriculture/learning/task_bc_error_diagnostics_v1/shadow.py',
    'kaggriculture/decision/audit_market_teacher.py',)))


def _git(*args):
    return subprocess.check_output(['git','-C',str(REPO),*args],
        stderr=subprocess.DEVNULL, timeout=30).decode('utf-8').strip()


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _read(path, limit=1024**2):
    return train_fit._read(path, limit=limit)


def _source_hashes():
    return {name: _sha(_read(REPO / name)) for name in SOURCE_NAMES}


def _registration(sources, binding_sha):
    return dict(schema='state-blind-diagnostic-prereg-v1', base_commit=BASE_COMMIT,
        state_registration_commit=STATE_COMMIT, source_sha256=sources,
        private_binding_sha256=binding_sha, official_rule_commit=ENGINE_COMMIT,
        official_rule_sha256=ENGINE_SHA, cases=[dict(role=r,batch=2,slot=t,opponent=o,side=s)
                                               for r,t,o,s in CASES],
        windows=[list(x) for x in WINDOWS], max_steps_per_window=4,
        max_callbacks_per_turn=32, max_candidates_per_group=256, max_groups=192,
        selection='eligible_step_hash_then_independent_ordinal_hash_blind_to_labels_and_outcomes',
        replay_view='typed_ordered_observation_configuration_and_inventory',
        duplicate_scope='within_sample_raw_and_normalized_ordered_groups_no_removal',
        assessor_passes_per_group=4, baseline_forward_passes_per_group=1,
        residual_recomputed_within_each_assessor_pass=True,
        historical_screen='failed', independent_evaluation=False, promotion=False,
        training=False, model_updates=0, new_games=0, gpu=False, device='cpu',
        job=dict(JOB), initial_memory_bytes=INITIAL_MEMORY, disk_floor_bytes=MIN_DISK,
        output_reserve_bytes=OUTPUT_RESERVE, max_attempts=1)


def _identity(context, guard):
    """Authenticate completed model and source metadata, without model scoring."""
    guard()
    prior = state.load_completed(guard=guard)
    old_binding, old_sha = state._registered_binding(STATE_COMMIT)
    roles = context.private_role_binding()
    selected = {role: roles[role] for role in ROLES}
    if (len(set(selected.values())) != 2 or context.source_role(batch=2,role='T01') != 'A05'
            or context.source_role(batch=2,role='T02') != 'A06'
            or prior.get('screen',{}).get('screen') != 'failed'
            or prior.get('registration_commit') != STATE_COMMIT
            or prior.get('baseline_sha256') != old_binding.get('baseline_sha256')
            or context.task_sha256 != old_binding.get('baseline_sha256')
            or context.crop_sha256 != old_binding.get('crop_sha256')):
        raise ValueError('Historical source identities differ')
    _mean, _scale, metadata = context.model_normalization()
    if (_sha(previous._encoded(metadata)) != old_binding.get('metadata_sha256')
            or metadata.get('schema_sha256') != old_binding.get('schema_sha256')):
        raise ValueError('Historical model schema differs')
    guard()
    return dict(roles=selected, state_binding_sha256=old_sha,
        baseline_sha256=prior['baseline_sha256'], params_sha256=prior['params_sha256'],
        crop_sha256=context.crop_sha256, schema_sha256=metadata['schema_sha256'],
        metadata_sha256=old_binding['metadata_sha256'])


def _prepare_status():
    """Permit only this untracked source bundle at the frozen parent commit."""
    lines = _git('status','--porcelain','--untracked-files=all').splitlines()
    return all(line.startswith('?? ') and line[3:].replace('\\','/') in NEW_FILES for line in lines)


def prepare():
    if (os.name != 'nt' or _git('rev-parse','HEAD') != BASE_COMMIT or not _prepare_status()
            or PRIVATE.exists() or REGISTRATION.exists()):
        raise ValueError('Unexpected preparation state')
    if previous.physical_available() < INITIAL_MEMORY:
        raise MemoryError('Initial memory floor')
    from learning.task_student_anchor_collection_v1 import sources
    deadline = time.monotonic() + 300
    guard = lambda: previous._guard(deadline)
    context = sources.open_context(guard=guard)
    identity = _identity(context, guard)
    binding = dict(schema='state-blind-diagnostic-binding-v1', salt=secrets.token_hex(32), **identity)
    registration = _registration(_source_hashes(), _sha(previous._encoded(binding)))
    PRIVATE.mkdir(parents=False, exist_ok=False)
    previous._write_new(PRIVATE / 'binding.local.json', binding)
    previous._write_new(REGISTRATION, registration)
    print(json.dumps(dict(status='prepared')), flush=True)


def _registered_binding(head):
    if type(head) is not str or re.fullmatch(r'[0-9a-f]{40}', head) is None:
        raise ValueError('Invalid registration commit')
    parent = _git('rev-parse', head + '^')
    raw = subprocess.check_output(['git','-C',str(REPO),'show',
        head + ':' + REGISTRATION.relative_to(REPO).as_posix()], stderr=subprocess.DEVNULL, timeout=30)
    local = _read(PRIVATE / 'binding.local.json')
    if parent != BASE_COMMIT or previous._document(raw) != _registration(_source_hashes(), _sha(local)):
        raise ValueError('Frozen registration differs')
    binding = previous._document(local)
    keys = {'schema','salt','roles','state_binding_sha256','baseline_sha256','params_sha256',
            'crop_sha256','schema_sha256','metadata_sha256'}
    if (set(binding) != keys or binding.get('schema') != 'state-blind-diagnostic-binding-v1'
            or type(binding.get('roles')) is not dict or set(binding['roles']) != set(ROLES)
            or any(type(value) is not int or not 0 <= value < 2**32 for value in binding['roles'].values())
            or len(set(binding['roles'].values())) != 2
            or any(type(binding[key]) is not str or re.fullmatch(r'[0-9a-f]{64}',binding[key]) is None
                   for key in keys - {'schema','roles'})):
        raise ValueError('Private binding differs')
    return binding, _sha(local)


def _runtime_binding():
    head = _git('rev-parse','HEAD')
    if _git('status','--porcelain') or _git('rev-parse','origin/main') != head:
        raise ValueError('Registration not clean and published')
    binding, digest = _registered_binding(head)
    return head, binding, digest


def _same_view(sample, replay, side):
    """Compare exactly typed, ordered single-player data to this replay step."""
    from decision.api import observe
    from decision.audit_market_teacher import observation
    from learning.task_bc_error_diagnostics_v1.shadow import first_difference
    step, inputs = sample['step'], sample['inputs']
    if (type(replay) is not dict or type(replay.get('steps')) is not list
            or len(replay['steps']) != 720 or type(inputs) is not dict):
        raise ValueError('Private replay view differs')
    view = observe(observation(replay['steps'][step], side), replay['configuration'])
    if (first_difference(inputs.get('observation'), view.observation) is not None
            or first_difference(inputs.get('configuration'), view.configuration) is not None
            or 'seed' in view.configuration or 'info' in view.observation):
        raise ValueError('Private replay view differs')
    private = view.observation['private']
    if (first_difference(inputs.get('inventory_item_order'), [list(x) for x in private['inventories']]) is not None
            or first_difference(inputs.get('shed_item_order'), list(private['shed'])) is not None):
        raise ValueError('Private inventory order differs')


def _collect_case(context, case, metadata, guard, *, read_game):
    """Use a single authenticated game; release its replay on scope exit."""
    from .sampling import RankedStream
    from .extract import extract_example
    from .diagnostic import PrivateRecord
    role, slot, opponent, side = case
    result, seen = [], set()
    with read_game(context, batch=2, role=slot, opponent=opponent, side=side, guard=guard) as game:
        expected = dict(source_role=role,batch=2,role=slot,opponent=opponent,side=side)
        if any(game.identity.get(key) != value for key,value in expected.items()):
            raise ValueError('Authenticated source coordinate differs')
        with game.open_trace(guard=guard) as stream:
            sampler = RankedStream(stream, role=role, opponent=opponent, side=side, guard=guard)
            accounting = sampler.scan(task_sha256=game.task_sha256, crop_sha256=game.crop_sha256)
            if accounting.get('lines') != 719:
                raise ValueError('Source scan incomplete')
        # Source trace's final same-descriptor hash has passed before sampling.
        windows = sampler.samples()
        if type(windows) is not list or len(windows) != len(WINDOWS):
            raise ValueError('Sampling windows differ')
        for window, entries in enumerate(windows):
            if type(entries) is not list or len(entries) > 4:
                raise ValueError('Sample window exceeds budget')
            for sample in entries:
                guard()
                if (type(sample) is not dict or set(sample) != {'role','opponent','side','step',
                        'ordinal','inputs','teacher','chosen'} or sample['role'] != role
                        or sample['opponent'] != opponent or type(sample['side']) is not int
                        or sample['side'] != side or type(sample['step']) is not int
                        or sample['step'] // 120 != window or sample['step'] in seen):
                    raise ValueError('Selected sample coordinate differs')
                seen.add(sample['step'])
                _same_view(sample, game.replay, side)
                ex = extract_example(role=role, step=sample['step'], ordinal=sample['ordinal'],
                    inputs=sample['inputs'], teacher_label=sample['teacher'], chosen=sample['chosen'],
                    metadata=metadata, guard=guard)
                result.append(PrivateRecord(role, opponent, side, window, ex.item, ex.student_index))
    return result


def _validate_report(report):
    """Check result accounting without inspecting or exposing private groups."""
    from learning.task_choice_v1.features import OPS
    fields = {'overall','by_role','by_opponent','by_side','by_window','by_target_operation','coverage_by_case',
              'single_candidate_groups','duplicates','independent_evaluation','promotion','historical_screen'}
    if (type(report) is not dict or set(report) != fields or report['independent_evaluation'] is not False
            or report['promotion'] is not False or report['historical_screen'] != 'failed'):
        raise ValueError('Private report contract differs')
    count_keys = ('groups','baseline_correct','residual_correct','changed','op_flips','improved','regressed')
    keys = set(count_keys) | {'coverage','baseline_nll','residual_nll','teacher_pair'}
    def bucket(value):
        if (type(value) is not dict or set(value) != keys
                or any(type(value[k]) is not int or not 0 <= value[k] <= 192 for k in count_keys)):
            raise ValueError('Private bucket differs')
        n = value['groups']
        if (value['coverage'] != ('observed' if n else 'unknown')
                or any(value[k] > n for k in count_keys[1:])
                or value['op_flips'] > value['changed']
                or value['improved'] + value['regressed'] > value['changed']
                or value['regressed'] > value['baseline_correct']
                or value['improved'] > n - value['baseline_correct']
                or value['residual_correct'] != value['baseline_correct'] + value['improved'] - value['regressed']):
            raise ValueError('Private paired counts differ')
        pair = dict(both_match=value['baseline_correct']-value['regressed'], improved=value['improved'],
                    regressed=value['regressed'], neither_match=n-value['baseline_correct']-value['improved'])
        if value['teacher_pair'] != pair or min(pair.values()) < 0 or sum(pair.values()) != n:
            raise ValueError('Private paired grid differs')
        for key in ('baseline_nll','residual_nll'):
            number = value[key]
            if ((n == 0 and number is not None) or (n > 0 and
                (type(number) not in (int,float) or not math.isfinite(number) or number < 0))):
                raise ValueError('Private conditional loss differs')
    bucket(report['overall'])
    total = report['overall']
    vocabularies = dict(by_role=set(ROLES), by_opponent=set(OPPONENTS), by_side={'0','1'},
        by_window={str(i) for i in range(len(WINDOWS))}, by_target_operation=set(OPS))
    for field, vocabulary in vocabularies.items():
        partition = report[field]
        if type(partition) is not dict or set(partition) != vocabulary:
            raise ValueError('Private aggregate partition differs')
        for value in partition.values():
            bucket(value)
        if any(sum(value[k] for value in partition.values()) != total[k] for k in count_keys):
            raise ValueError('Private aggregate partition counts differ')
        if total['groups']:
            for key in ('baseline_nll','residual_nll'):
                mean = math.fsum(value[key] * (value['groups'] / total['groups'])
                                 for value in partition.values() if value['groups'])
                if not math.isclose(mean,total[key],rel_tol=1e-10,abs_tol=1e-10):
                    raise ValueError('Private aggregate loss differs')
    if (any(value['groups'] > 96 for value in report['by_role'].values())
            or any(value['groups'] > 32 for value in report['by_window'].values())
            or type(report['single_candidate_groups']) is not int
            or not 0 <= report['single_candidate_groups'] <= total['groups']):
        raise ValueError('Private sample budget differs')
    cases = report['coverage_by_case']
    if type(cases) is not dict or set(cases) != set(ROLES):
        raise ValueError('Private case coverage differs')
    entries = []
    for role in ROLES:
        if type(cases[role]) is not dict or set(cases[role]) != set(OPPONENTS):
            raise ValueError('Private case coverage differs')
        for opponent in OPPONENTS:
            sides = cases[role][opponent]
            if type(sides) is not dict or set(sides) != {'0','1'}:
                raise ValueError('Private case coverage differs')
            for side in ('0','1'):
                value = sides[side]
                if (type(value) is not dict or set(value) != {'groups','windows'}
                        or type(value['groups']) is not int or not 0 <= value['groups'] <= 24
                        or type(value['windows']) is not dict
                        or set(value['windows']) != {str(i) for i in range(len(WINDOWS))}
                        or any(type(n) is not int or not 0 <= n <= 4 for n in value['windows'].values())
                        or sum(value['windows'].values()) != value['groups']):
                    raise ValueError('Private case coverage differs')
                entries.append((role,opponent,side,value))
    if (sum(value['groups'] for _,_,_,value in entries) != total['groups']
            or any(sum(value['groups'] for r,_,_,value in entries if r == role)
                   != report['by_role'][role]['groups'] for role in ROLES)
            or any(sum(value['groups'] for _,o,_,value in entries if o == opponent)
                   != report['by_opponent'][opponent]['groups'] for opponent in OPPONENTS)
            or any(sum(value['groups'] for _,_,s,value in entries if s == side)
                   != report['by_side'][side]['groups'] for side in ('0','1'))
            or any(sum(value['windows'][str(i)] for _,_,_,value in entries)
                   != report['by_window'][str(i)]['groups'] for i in range(len(WINDOWS)))):
        raise ValueError('Private case coverage margins differ')
    duplicates = report['duplicates']
    if (type(duplicates) is not dict or set(duplicates) != {'raw','normalized','scope'}
            or duplicates['scope'] != 'within_selected_ordered_groups_only'):
        raise ValueError('Private duplicate scope differs')
    for key in ('raw','normalized'):
        values = duplicates[key]
        if (type(values) is not dict or set(values) != {'unique','repeated_groups','excess'}
                or any(type(x) is not int or not 0 <= x <= total['groups'] for x in values.values())
                or values['excess'] != total['groups'] - values['unique']
                or (values['excess'] == 0 and values['repeated_groups'] != 0)
                or (values['excess'] > 0 and values['repeated_groups'] <= values['excess'])):
            raise ValueError('Private duplicate counts differ')
    if (duplicates['normalized']['unique'] > duplicates['raw']['unique']
            or duplicates['normalized']['repeated_groups'] < duplicates['raw']['repeated_groups']):
        raise ValueError('Private duplicate representation differs')
    return True


def _result(raw, head):
    value = previous._document(raw)
    keys = {'schema','registration_commit','source_games','trace_turns','sampled_groups','report',
            'baseline_sha256','params_sha256','device','training','model_updates','new_games',
            'gpu','independent_evaluation','promotion'}
    if (set(value) != keys or value.get('schema') != 'state-blind-diagnostic-result-v1'
            or value.get('registration_commit') != head or value.get('source_games') != 8
            or value.get('trace_turns') != 8 * 719 or value.get('device') != 'cpu'
            or value.get('training') is not False or type(value.get('model_updates')) is not int
            or value['model_updates'] != 0 or type(value.get('new_games')) is not int or value['new_games'] != 0
            or value.get('gpu') is not False or value.get('independent_evaluation') is not False
            or value.get('promotion') is not False or type(value.get('sampled_groups')) is not int
            or not 0 <= value['sampled_groups'] <= 192
            or any(type(value.get(k)) is not str or re.fullmatch(r'[0-9a-f]{64}',value[k]) is None
                   for k in ('baseline_sha256','params_sha256'))):
        raise ValueError('Private diagnostic result differs')
    _validate_report(value['report'])
    if value['sampled_groups'] != value['report']['overall']['groups']:
        raise ValueError('Private sampled count differs')
    return value


def _validated_outputs(head, binding, guard):
    guard()
    result = _result(_read(PRIVATE/'worker-result.local.json'), head)
    if any(result[key] != binding[key] for key in ('baseline_sha256','params_sha256')):
        raise ValueError('Private model identity differs')
    guard()
    return result


def child():
    head, binding, digest = _runtime_binding()
    claim = previous._document(_read(PRIVATE/'claim.local.json'))
    attempt = previous._document(_read(PRIVATE/'attempt.local.json'))
    if (claim.get('schema') != 'state-blind-diagnostic-claim-v1' or claim.get('commit') != head
            or type(claim.get('attempt')) is not int or claim['attempt'] != 1
            or claim.get('freeze_sha256') != digest or claim.get('retry_allowed') is not False
            or attempt != dict(schema='state-blind-diagnostic-attempt-v1',commit=head,retry_allowed=False)
            or (PRIVATE/'failed.local.json').exists() or (PRIVATE/'closed.local.json').exists()):
        raise ValueError('Child is not uniquely claimed')
    from learning.task_teacher_dev_deviation_v1.membership import ENV_KEY, verify_worker
    from learning.task_pair_fresh_multi_v1.entry import verify_live_owner
    import psutil
    owner = previous._document(_read(PRIVATE/'owner.local.json'))
    verify_worker(os.environ.get(ENV_KEY), claim, digest)
    verify_live_owner(owner, commit=head, binding_sha=digest, current_pid=os.getpid(),
                      process=psutil.Process, executable=sys.executable, samefile=os.path.samefile,
                      now=time.monotonic())
    previous._write_new(PRIVATE/'child-attempt.local.json',dict(
        schema='state-blind-diagnostic-child-attempt-v1',commit=head,retry_allowed=False))
    for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
        os.environ[key] = '1'
    deadline = time.monotonic() + JOB['wall_seconds']
    guard = lambda: previous._guard(deadline)
    completed, phase = 0, 'source'
    try:
        from learning.task_student_anchor_collection_v1 import sources
        from . import diagnostic
        context = sources.open_context(guard=guard)
        identity = _identity(context, guard)
        if identity != {key:value for key,value in binding.items() if key not in ('schema','salt')}:
            raise ValueError('Historical binding changed')
        network, params = train_fit._models(binding, guard)
        records = []
        phase = 'sample'
        for case in CASES:
            guard()
            records.extend(_collect_case(context, case, network.metadata, guard, read_game=sources.read_game))
            completed += 1
        if completed != 8 or len(records) > 192:
            raise ValueError('Source population differs')
        phase = 'score'
        groups = diagnostic.score(records, network, guard=guard)
        report = diagnostic.assess(records, groups, mean=network.mean, scale=network.scale,
                                   params=params, metadata=network.metadata, guard=guard)
        _validate_report(report)
        phase = 'seal'
        if (_identity(context, guard) != identity or _runtime_binding() != (head,binding,digest)):
            raise ValueError('Historical sources changed')
        result = dict(schema='state-blind-diagnostic-result-v1',registration_commit=head,
            source_games=8,trace_turns=8*719,sampled_groups=len(records),report=report,
            baseline_sha256=binding['baseline_sha256'],params_sha256=binding['params_sha256'],
            device='cpu',training=False,model_updates=0,new_games=0,gpu=False,
            independent_evaluation=False,promotion=False)
        _result(previous._encoded(result), head)
        guard()
        previous._write_new(PRIVATE/'worker-result.local.json', result)
    except BaseException as exc:
        previous._write_new(PRIVATE/'worker-failed.local.json',dict(
            phase=phase,category=type(exc).__name__,completed_sources=completed,retry_allowed=False))
        raise


def run():
    if os.name != 'nt' or previous.physical_available() < INITIAL_MEMORY:
        raise MemoryError('Initial memory floor')
    if shutil.disk_usage(PRIVATE).free < MIN_DISK + OUTPUT_RESERVE:
        raise OSError('Private disk floor')
    head, binding, digest = _runtime_binding()
    for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
        os.environ[key] = '1'
    from learning.task_teacher_dev_deviation_v1.membership import token_hash
    from learning.task_pair_fresh_multi_v1.entry import owner_marker
    import psutil
    deadline = time.monotonic() + JOB['wall_seconds'] + 2 * JOB['cleanup_seconds']
    previous._write_new(PRIVATE/'attempt.local.json',dict(
        schema='state-blind-diagnostic-attempt-v1',commit=head,retry_allowed=False))
    try:
        proc = psutil.Process(os.getpid())
        owner = previous._document(owner_marker(commit=head,binding_sha=digest,pid=os.getpid(),
            created_us=int(proc.create_time()*1_000_000),deadline=deadline))
        previous._write_new(PRIVATE/'owner.local.json',owner)
        job_name = 'Local\\TeacherDeviation-' + secrets.token_hex(16)
        previous._write_new(PRIVATE/'claim.local.json',dict(schema='state-blind-diagnostic-claim-v1',
            commit=head,attempt=1,freeze_sha256=digest,job_token_sha256=token_hash(job_name),retry_allowed=False))
        command = [sys.executable,'-B','-m','learning.task_state_blind_diagnostic_v1.experiment','child']
        job = supervisor._supervise_owned(command,str(REPO/'kaggriculture'),JOB,job_name)
        previous._write_new(PRIVATE/'job.local.json',job)
        if not transfer._job_ok(job):
            raise RuntimeError('Owned Job did not close cleanly')
        guard = lambda: previous._guard(deadline)
        _validated_outputs(head,binding,guard)
        if _registered_binding(head)[1] != digest:
            raise ValueError('Registration changed')
        names = ('worker-result','job','attempt','owner','claim','child-attempt')
        receipts = {name:_sha(_read(PRIVATE/(name+'.local.json'))) for name in names}
        previous._write_new(PRIVATE/'closed.local.json',dict(schema='state-blind-diagnostic-closed-v1',
            registration_commit=head,binding_sha256=digest,receipts=receipts,retry_allowed=False))
        print(json.dumps(dict(status='completed')),flush=True)
    except BaseException as exc:
        previous._write_new(PRIVATE/'failed.local.json',dict(
            status='failed',category=type(exc).__name__,retry_allowed=False))
        raise


def load_completed(*, guard=lambda: None):
    """Authenticate a private technical result; never authorize promotion."""
    if (PRIVATE/'failed.local.json').exists() or (PRIVATE/'worker-failed.local.json').exists():
        raise ValueError('Failed attempt cannot be consumed')
    closed = previous._document(_read(PRIVATE/'closed.local.json'))
    head = closed.get('registration_commit')
    binding, digest = _registered_binding(head)
    names = ('worker-result','job','attempt','owner','claim','child-attempt')
    if (set(closed) != {'schema','registration_commit','binding_sha256','receipts','retry_allowed'}
            or closed.get('schema') != 'state-blind-diagnostic-closed-v1'
            or closed.get('binding_sha256') != digest or closed.get('retry_allowed') is not False
            or type(closed.get('receipts')) is not dict or set(closed['receipts']) != set(names)):
        raise ValueError('Completion seal differs')
    for name in names:
        guard()
        if _sha(_read(PRIVATE/(name+'.local.json'))) != closed['receipts'][name]:
            raise ValueError('Completion receipt differs')
    if not transfer._job_ok(previous._document(_read(PRIVATE/'job.local.json'))):
        raise ValueError('Owned Job receipt differs')
    attempt = previous._document(_read(PRIVATE/'attempt.local.json'))
    marker = previous._document(_read(PRIVATE/'child-attempt.local.json'))
    claim = previous._document(_read(PRIVATE/'claim.local.json'))
    owner = previous._document(_read(PRIVATE/'owner.local.json'))
    if (attempt != dict(schema='state-blind-diagnostic-attempt-v1',commit=head,retry_allowed=False)
            or marker != dict(schema='state-blind-diagnostic-child-attempt-v1',commit=head,retry_allowed=False)
            or claim.get('schema') != 'state-blind-diagnostic-claim-v1' or claim.get('commit') != head
            or type(claim.get('attempt')) is not int or claim['attempt'] != 1
            or claim.get('freeze_sha256') != digest or claim.get('retry_allowed') is not False
            or type(claim.get('job_token_sha256')) is not str
            or re.fullmatch(r'[0-9a-f]{64}',claim['job_token_sha256']) is None
            or type(owner) is not dict or set(owner) != {'schema','registration_commit','binding_sha256',
                                                        'pid','created_us','deadline'}
            or owner.get('schema') != 'task-pair-fresh-multi-owner-v1'
            or owner.get('registration_commit') != head or owner.get('binding_sha256') != digest
            or type(owner.get('pid')) is not int or owner['pid'] <= 0
            or type(owner.get('created_us')) is not int or owner['created_us'] <= 0
            or type(owner.get('deadline')) not in (int,float) or not math.isfinite(owner['deadline'])):
        raise ValueError('Completion ownership differs')
    return _validated_outputs(head,binding,guard)


class _PrivateArgumentParser(argparse.ArgumentParser):
    def error(self, _message):
        raise ValueError('Invalid command')


def main():
    parser = _PrivateArgumentParser()
    parser.add_argument('command',choices=('prepare','run','child'))
    args = parser.parse_args()
    {'prepare':prepare,'run':run,'child':child}[args.command]()


if __name__ == '__main__':
    try:
        main()
    except (Exception,KeyboardInterrupt) as exc:
        category = ('memory' if isinstance(exc,MemoryError) else 'timeout' if isinstance(exc,TimeoutError)
                    else 'io' if isinstance(exc,OSError) else 'invalid_or_runtime')
        print(json.dumps(dict(status='failed',category=category)),file=sys.stderr)
        raise SystemExit(1) from None
