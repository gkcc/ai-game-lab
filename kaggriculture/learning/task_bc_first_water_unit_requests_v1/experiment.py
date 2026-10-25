"""Freeze, run once, and preserve bounded single-choice diagnostic evidence."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
PUBLIC = ROOT.parent / 'publication/ai-game-lab'
PREVIOUS = ROOT / 'artifacts/task-bc-saved-student-decisions-v1-20261026'
PACKAGE = Path(__file__).resolve().parent
STAGE = 'task-bc-first-water-unit-requests-v1'
PLANS = ('follow-on-request-comparison-advice.json', 'root-follow-on-request-decision.json')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda: stream.read(1048576), b''):
            digest.update(part)
    return digest.hexdigest()


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def stat(path):
    value = Path(path).stat()
    return dict(size=value.st_size, mtime_ns=value.st_mtime_ns)


def git(*args):
    return subprocess.check_output(['git', *args], cwd=PUBLIC)


def official():
    package = Path(next(iter(importlib.util.find_spec('kaggle_environments').submodule_search_locations)))
    return dict(rules=sha(package / 'envs/kaggriculture/kaggriculture.py'), loader=sha(package / 'agent.py'))


def dependencies():
    return dict(python=sys.version.split()[0], numpy=importlib.metadata.version('numpy'),
                kaggle_environments=importlib.metadata.version('kaggle-environments'))


def terminal_status(key, value):
    """Validate the counter envelope before claiming usable terminal evidence.

    Unknown partial work stays in the original ledger; a missing/malformed
    envelope is not interpreted as zero consumption. The independent checker
    subsequently verifies the successful primary's complete contract.
    """
    require(type(value) is dict, 'Terminal receipt must be an object')

    def integers(group, names):
        require(type(group) is dict and all(type(group.get(name)) is int and group[name] >= 0
                                           for name in names), 'Incomplete terminal counter group')

    if key == 'primary':
        require(value.get('schema') == 'first-water-unit-request-primary-v1' and
                value.get('status') in ('complete', 'failed'), 'Primary terminal schema differs')
        ledger = value.get('ledger')
        require(type(ledger) is dict and set(ledger) ==
                {'source', 'model_io', 'models', 'controller', 'unknown_accounting'},
                'Missing primary consumption group')
        integers(ledger['source'], ('gzip_opens', 'prefix_lines_returned', 'target_json_decode_attempts',
            'target_json_turns_decoded', 'compressed_bytes_read', 'returned_payload_bytes',
            'target_json_bytes', 'target_task_records', 'target_candidate_objects',
            'target_candidate_x_arrays', 'target_shared_x_arrays', 'target_observation_objects',
            'target_configuration_objects', 'target_crop_records', 'target_crop_x_arrays',
            'target_crop_logits_arrays', 'sources_completed'))
        integers(ledger['model_io'], ('checkpoint_hash_passes_started', 'checkpoint_hash_passes_completed',
            'model_object_loads_reserved', 'model_object_loads_started', 'model_object_loads_completed',
            'np_load_calls_started', 'np_load_calls_completed', 'model_read_calls_started',
            'model_read_calls_completed', 'total_model_bytes_returned_by_reads'))
        integers(ledger['models'], ('task_forward_calls', 'task_forward_calls_completed',
            'task_scored_candidates', 'task_scored_candidates_completed', 'crop_forward_calls',
            'crop_forward_calls_completed', 'crop_logits_values', 'crop_logits_values_completed', 'training_calls'))
        integers(ledger['controller'], ('real_calls', 'real_calls_completed', 'baseline_calls',
            'baseline_calls_completed', 'intervention_calls', 'intervention_calls_completed', 'forced_choices',
            'callbacks', 'callbacks_completed', 'task_candidate_encodings', 'task_candidate_encodings_completed',
            'generated_candidate_objects', 'full_teacher_policy_calls', 'official_transitions', 'new_games'))
        unknown = ledger['unknown_accounting']
        require(type(unknown) is dict and set(unknown) == {'planned_prefix', 'failures'} and
                all(type(items) is list for items in unknown.values()), 'Malformed primary unknown accounting')
        passed = value['status'] == 'complete'
        if passed:
            require(value.get('baseline_gate_passed') is True and value.get('completed_baselines') == 6 and
                    value.get('completed_interventions') == 6 and not unknown['failures'],
                    'Primary completion or unknown failure differs')
        return passed
    require(value.get('schema') in ('first-water-unit-request-independent-check-v1',
            'first-water-unit-request-independent-check-failure-v1') and
            type(value.get('passed')) is bool and type(value.get('unknown_accounting')) is list,
            'Checker terminal schema differs')
    integers(value.get('consumption'), ('primary_receipt_open_attempts', 'primary_receipt_files_read',
        'primary_receipt_bytes_read', 'pair_receipt_open_attempts', 'pair_receipt_files_read',
        'pair_receipt_bytes_read', 'json_decode_attempts', 'json_files_decoded', 'json_bytes_submitted',
        'json_bytes_decoded', 'pairs_verified'))
    require(not value['passed'] or not value['unknown_accounting'], 'Successful checker has unknown failure work')
    return value['passed']


def prepare(out):
    require(not out.exists(), 'Preserve any old registration or attempt')
    require(not git('status', '--porcelain').strip(), 'Register from a clean accepted checkout')
    decision = read(PREVIOUS / PLANS[1])
    advice = read(PREVIOUS / PLANS[0])
    previous = read(PREVIOUS / 'registration.json')
    require(decision['status'] == advice['status'] == 'planned_not_started', 'Scope already consumed')
    require(read(PREVIOUS / 'root-evidence-acceptance.json')['passed'] and
            read(PREVIOUS / 'publication-verification.json')['passed'], 'Previous stage not accepted')
    require(sha(PREVIOUS / PLANS[0]) == decision['accepted_advice_sha256'], 'Advice binding changed')
    head = git('rev-parse', 'HEAD').decode().strip()
    require(head == decision['current_published_commit'], 'Unexpected current publication HEAD')
    for rel, digest in advice['evidence_and_source_sha256'].items():
        require(sha(ROOT / rel) == digest, 'Planned source or metadata changed')
    originals = {row['game']: row for row in previous['selected_games']}
    samples = []
    for item in decision['fixed_selection']['samples']:
        old = originals[item['game']]
        require(item['source_path'] == old['source_path'] and
                item['inherited_whole_trace_sha256'] == old['trace_sha256'], 'Source identity differs')
        require(stat(ROOT / item['source_path']) == old['source_stat'], 'Source stat changed')
        samples.append(dict(item, source_stat=old['source_stat']))
    require(len(samples) == 6 and all(row['step'] == 15 for row in samples), 'Exactly six fixed turns')
    models = {key: decision['fixed_models'][key] for key in ('task', 'crop')}
    for item in models.values():
        require(stat(ROOT / item['path']) == previous['model_protection'][item['path']], 'Model stat changed')
    history = read(PREVIOUS / 'protection-before.json')['stat_before'].copy()
    for path in PREVIOUS.rglob('*'):
        if path.is_file() and '__pycache__' not in path.parts:
            history[path.relative_to(ROOT).as_posix()] = stat(path)
    report = ROOT / 'TASK_BC_SAVED_STUDENT_DECISIONS_V1.zh-CN.md'
    history[report.relative_to(ROOT).as_posix()] = stat(report)
    require(all((ROOT / rel).is_file() and stat(ROOT / rel) == value for rel, value in history.items()),
            'Inherited history stat differs before registration')
    out.mkdir(parents=True)
    for name in PLANS:
        (out / name).write_bytes((PREVIOUS / name).read_bytes())
    write_new(out / 'protection-before.json', dict(stat_before=history,
        inherited_protection_sha256=sha(PREVIOUS / 'protection-before.json'),
        method='Inherited hashes and before/after stat; declared code/metadata freshly hashed. Prefix target-line hashes and two model hash/load reads are charged inside the single registered worker.'))
    frozen = read(PREVIOUS / 'freeze.json')
    metadata = advice['evidence_and_source_sha256'].copy()
    for name in ('freeze.json', 'publication-verification.json', 'root-evidence-acceptance.json',
                 'execution-acceptance.json', 'summary.json', 'evidence-review.json'):
        path = PREVIOUS / name
        metadata[path.relative_to(ROOT).as_posix()] = sha(path)
    cfg = dict(schema=STAGE, created_utc=datetime.now(timezone.utc).isoformat(), baseline_head=head,
        samples=samples, models=models, budget=decision['budget'], per_controller_call_seconds_max=10,
        source_metadata_sha256=metadata, inherited_source_sha256=frozen['runtime_files'],
        line_ending_mappings=frozen['line_ending_mappings'],
        planning_sha256={name: sha(out / name) for name in PLANS},
        protection_sha256=sha(out / 'protection-before.json'),
        model_protection={item['path']: stat(ROOT / item['path']) for item in models.values()},
        model_stat_not_fresh_byte_proof_before_execution=True,
        default_sha256=sha(ROOT / 'main.py'), official_sha256=official(), engine_commit=previous['engine_commit'],
        dependencies=dependencies(),
        root_scope_decisions=decision['root_scope_decisions'],
        primary_receipt_bytes_max=65536, checker_terminal_bytes_max=65536,
        acceptance=decision['acceptance'], stop_conditions=decision['stop_conditions'],
        local_commit_not_external_timestamp=True, historical_consumed_seed_not_independent_holdout=True,
        full_historical_market_action_not_reproduced=True,
        clock='One executable clock before all in-memory checks, prefix reads, model hash/load, calls, independent check, final verification and cleanup. All code is reviewed and committed before any data/model read. Human metadata/source review outside the clock adds no data or inference pass.',
        final_acceptance='Require parent zero exit, bound execution-acceptance, no final-deadline-failure, no check/failure, known terminal consumption and natural worker tree closure. summary alone is provisional.',
        previous_diagnostic_stage_acceptance=False, original_policy_admission=False,
        previous_strict_review_acceptance=False, retries_allowed=0)
    require(cfg['official_sha256'] == previous['official_sha256'], 'Official bytes changed')
    write_new(out / 'registration.json', cfg)
    return dict(prepared=True, samples=6, source_data_reads=0, model_reads=0, tests_executed=0)


def freeze(out):
    cfg = read(out / 'registration.json')
    sources = cfg['inherited_source_sha256'].copy()
    for path in PACKAGE.iterdir():
        if path.is_file() and path.suffix in ('.py', '.md'):
            sources[path.relative_to(ROOT).as_posix()] = sha(path)
    published = {rel: sha(PUBLIC / 'kaggriculture' / rel) for rel in sources}
    for rel, digest in sources.items():
        require(sha(ROOT / rel) == digest, 'Runtime source changed')
        if digest != published[rel]:
            require(rel in cfg['line_ending_mappings'] and
                    (ROOT / rel).read_bytes().replace(b'\r\n', b'\n') ==
                    (PUBLIC / 'kaggriculture' / rel).read_bytes().replace(b'\r\n', b'\n'),
                    'Unregistered source byte mapping')
    write_new(out / 'freeze.json', dict(registration_sha256=sha(out / 'registration.json'),
        runtime_files=sources, public_source_sha256=published, line_ending_mappings=cfg['line_ending_mappings']))
    return dict(frozen=True, freeze_sha256=sha(out / 'freeze.json'), sources=len(sources))


def verify(out):
    cfg, frozen = read(out / 'registration.json'), read(out / 'freeze.json')
    require(sha(out / 'registration.json') == frozen['registration_sha256'] and
            sha(out / 'protection-before.json') == cfg['protection_sha256'], 'Registration or history changed')
    for name, digest in cfg['planning_sha256'].items():
        require(sha(out / name) == digest, 'Planning changed')
    for rel, digest in frozen['runtime_files'].items():
        require(sha(ROOT / rel) == digest, 'Frozen source changed')
    for rel, digest in cfg['source_metadata_sha256'].items():
        require(sha(ROOT / rel) == digest, 'Frozen metadata changed')
    for rel, value in cfg['model_protection'].items():
        require(stat(ROOT / rel) == value, 'Model stat changed')
    for sample in cfg['samples']:
        require(stat(ROOT / sample['source_path']) == sample['source_stat'], 'Raw source stat changed')
    require(official() == cfg['official_sha256'], 'Official bytes changed')
    require(dependencies() == cfg['dependencies'], 'Registered Python or package version changed')
    return cfg


def worker(out, key):
    cfg = verify(out)
    request = read(out / 'requests' / (key + '.json'))
    deadline = request['deadline_monotonic']
    require(request['freeze_sha256'] == sha(out / 'freeze.json') and time.monotonic() < deadline,
            'Request binding changed or deadline expired')
    write_new(out / 'worker-started' / (key + '.json'), dict(request_sha256=sha(out / 'requests' / (key + '.json'))))
    if key == 'primary':
        from .probe import run_primary
        result = run_primary(ROOT, out, cfg, deadline)
    elif key == 'independent-check':
        from .check import check
        result = check(out, dict(cfg, primary_receipt_sha256=request['primary_receipt_sha256']), deadline)
        require(not (out / 'check/failure.json').exists(), 'Checker terminal failure exists')
    else:
        raise ValueError('Unknown registered worker')
    result_passed = result.get('status') == 'complete' if key == 'primary' else result.get('passed') is True
    require(result_passed, 'Worker result failed')
    verify(out)
    require(time.monotonic() < deadline, 'Worker expired after final source verification')
    write_new(out / 'worker-complete' / (key + '.json'), dict(passed=True,
        request_sha256=sha(out / 'requests' / (key + '.json')), freeze_sha256=sha(out / 'freeze.json')))


def run(out):
    cfg = verify(out)
    frozen, pre, commit = (read(out / name) for name in ('freeze.json', 'pre-review.json', 'pre-execution-commit.json'))
    require(pre['passed'] is True and pre['freeze_sha256'] == sha(out / 'freeze.json') and
            commit['freeze_sha256'] == sha(out / 'freeze.json') and
            commit['commit'] == git('rev-parse', 'HEAD').decode().strip(), 'Exact reviewed commit required')
    expected = {'kaggriculture/' + rel: digest for rel, digest in frozen['public_source_sha256'].items()}
    for name in ('registration.json', 'freeze.json', 'pre-review.json', *PLANS):
        expected['review/evidence/' + out.name + '/' + name] = sha(out / name)
    require(commit['git_files_sha256'] == expected, 'Incomplete committed source/metadata set')
    for rel, digest in expected.items():
        require(hashlib.sha256(git('show', commit['commit'] + ':' + rel)).hexdigest() == digest,
                'Committed source or metadata bytes differ')
    budget = cfg['budget']
    started = time.monotonic()
    deadline = started + budget['stage_seconds_max']
    primary_deadline = started + budget['primary_phase_seconds_max']
    write_new(out / 'started.json', dict(created_utc=datetime.now(timezone.utc).isoformat(),
        started_monotonic=started, deadline_monotonic=deadline, freeze_sha256=sha(out / 'freeze.json')))
    attempts, terminal = [], {}
    failure = None
    from learning.task_bc_diagnostics_capacity_exit_v1.windows_tree import run_owned
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONUTF8='1', PYTHONIOENCODING='utf-8',
               OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')

    def dispatch(key, seconds, phase):
        work = min(time.monotonic() + seconds, phase - budget['per_close_seconds_max'])
        require(work > time.monotonic(), 'No remaining original work budget')
        request = dict(freeze_sha256=sha(out / 'freeze.json'), deadline_monotonic=work)
        if key == 'independent-check':
            request['primary_receipt_sha256'] = terminal['primary']['sha256']
        write_new(out / 'requests' / (key + '.json'), request)
        attempt = dict(key=key, passed=False, lifecycle={})
        attempts.append(attempt)
        local = {}
        print('first-water:', key, flush=True)
        try:
            (out / 'logs').mkdir(exist_ok=True)
            with (out / 'logs' / (key + '.log')).open('x', encoding='utf-8') as log:
                code = run_owned([sys.executable, '-B', '-m',
                    'learning.task_bc_first_water_unit_requests_v1.experiment', 'worker',
                    '--out', str(out), '--key', key], cwd=ROOT, env=env, stream=log,
                    timeout=work - time.monotonic(), phase_deadline=phase,
                    lifecycle=attempt['lifecycle'], local_lifecycle=local,
                    memory_limit_bytes=budget['process_tree_private_commit_bytes_max'])
            life = attempt['lifecycle']
            require(code == 0 and life['tree_closed'] and life['active_processes_after_close'] == 0 and
                    life['natural_tree_exit'] and not life['termination_requested'] and
                    life['job_handle_closed'] and life['all_process_handles_closed'], 'Worker lifecycle rejected')
            result = read(out / 'worker-complete' / (key + '.json'))
            require(result['passed'] and result['request_sha256'] == sha(out / 'requests' / (key + '.json')) and
                    result['freeze_sha256'] == sha(out / 'freeze.json'), 'Worker completion binding differs')
            attempt['passed'] = True
        finally:
            write_new(out / 'local-lifecycle' / (key + '.json'), local)
            path = (out / 'primary-receipt.json' if key == 'primary' else
                    out / 'check' / ('failure.json' if (out / 'check/failure.json').exists() else 'complete.json'))
            limit = cfg['primary_receipt_bytes_max'] if key == 'primary' else cfg['checker_terminal_bytes_max']
            record = dict(known=False, reason='Missing or incomplete terminal receipt: actual consumption is unknown, not zero.')
            try:
                require(0 < path.stat().st_size <= limit, 'Terminal receipt byte limit')
                raw = path.read_bytes()
                value = json.loads(raw)
                record['partial_receipt'] = value
                passed = terminal_status(key, value)
                record = dict(known=True, passed=passed, sha256=hashlib.sha256(raw).hexdigest(), result=value,
                    scope='Usable terminal counter envelope; registered prefix-buffer uncertainty and failed work remain explicitly represented in the ledger.')
            except Exception as error:
                record['error_type'] = type(error).__name__
            terminal[key] = record
            write_new(out / 'attempts' / (key + '.json'), attempt)
        require(terminal[key]['known'] and terminal[key]['passed'], 'Terminal consumption unverified')
        if key == 'independent-check':
            require(not (out / 'check/failure.json').exists(), 'Checker failure exists')

    try:
        dispatch('primary', budget['primary_phase_seconds_max'], primary_deadline)
        require(time.monotonic() < primary_deadline, 'Primary bookkeeping crossed original deadline')
        dispatch('independent-check', budget['independent_checker_seconds_max'],
                 deadline - budget['finalization_and_close_reserve_seconds'] + budget['per_close_seconds_max'])
        verify(out)
    except BaseException as error:
        failure = dict(kind=type(error).__name__, retry=False)
        raise
    finally:
        history = read(out / 'protection-before.json')['stat_before']
        changed = [rel for rel, value in history.items() if not (ROOT / rel).is_file() or stat(ROOT / rel) != value]
        write_new(out / 'history-stat-check.json', dict(passed=not changed, checked_files=len(history),
            changed_files=changed, method='Inherited hashes and before/after stat, plus separately frozen source and metadata byte checks.'))
        if time.monotonic() >= deadline and failure is None:
            failure = dict(kind='TimeoutError', phase='final-verification', retry=False)
        write_new(out / 'summary.json', dict(passed=failure is None and not changed,
            attempts=len(attempts), accepted_jobs=sum(row['passed'] for row in attempts),
            terminal_consumption=terminal, failure=failure, retries=0,
            elapsed_seconds=time.monotonic() - started, freeze_sha256=sha(out / 'freeze.json'),
            history_stat_check_sha256=sha(out / 'history-stat-check.json'),
            training_runs=0, raw_replay_reads=0, new_games=0, official_transitions=0,
            original_policy_admission=False, previous_diagnostic_stage_acceptance=False,
            previous_strict_review_acceptance=False))
        require(not changed and failure is None, 'Stopped with failure or history change preserved')
    result = read(out / 'summary.json')

    def final_gate():
        if time.monotonic() >= deadline:
            write_new(out / 'final-deadline-failure.json', dict(passed=False,
                elapsed_seconds=time.monotonic() - started, provisional_summary_not_final_acceptance=True))
            raise TimeoutError('Final acceptance exceeded original deadline')

    final_gate()
    write_new(out / 'execution-acceptance.json', dict(passed=result['passed'],
        summary_sha256=sha(out / 'summary.json'), elapsed_seconds=time.monotonic() - started,
        requires_zero_process_exit_and_no_final_deadline_failure=True))
    final_gate()
    return dict(passed=result['passed'], accepted_jobs=result['accepted_jobs'],
                elapsed_seconds=result['elapsed_seconds'])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=('prepare', 'freeze', 'run', 'worker'))
    parser.add_argument('--out', required=True)
    parser.add_argument('--key')
    args = parser.parse_args()
    out = Path(args.out).resolve()
    result = worker(out, args.key) if args.command == 'worker' else globals()[args.command](out)
    if result is not None:
        print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
