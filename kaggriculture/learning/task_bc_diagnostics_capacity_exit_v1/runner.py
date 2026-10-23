"""One-shot exit repair and one historical sample pass; no training or games."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback

from learning.bc_crop.agent import ROOT, sha
from learning.bc_crop.experiment import read, save
from learning.task_bc_diagnostics_capacity_v1.runner import require, check_time, new_json, resources, key_name
from .experiment import OLD, verify, ready


def budget_rows(out, cfg):
    """Read only this stage's chain; retain the immutable predecessor as a carry."""
    path = Path(out) / 'budget-reservations.jsonl'
    if not path.exists():
        return []
    rows = []; previous = cfg['old_artifacts_sha256']['budget-reservations.jsonl']
    for i, line in enumerate(path.read_bytes().splitlines()):
        row = json.loads(line)
        require(row['sequence'] == i and row['previous_line_sha256'] == previous, 'Budget chain mismatch')
        require(row['predecessor_ledger_sha256'] == cfg['old_artifacts_sha256']['budget-reservations.jsonl'],
                'Predecessor ledger mismatch')
        require(row['carry_groups'] == cfg['old_budget_groups'] and row['carry_candidates'] == cfg['old_budget_candidates'],
                'Historical reservation carry mismatch')
        require(all(type(row[k]) is int and row[k] >= 0 for k in ('groups', 'candidates')), 'Invalid reservation')
        require(row['reserved_before_dispatch'] is True and row['refunded'] is False, 'Invalid reservation disposition')
        require(not any(old['key'] == row['key'] for old in rows), 'Repeated job reservation')
        rows.append(row); previous = hashlib.sha256(line).hexdigest()
    totals = {k: sum(row[k] for row in rows) for k in ('groups', 'candidates')}
    require(totals['groups'] <= cfg['new_evaluated_groups_max'] and
            totals['candidates'] <= cfg['new_evaluated_candidates_max'], 'New stage reservation cap')
    require(totals['groups'] + cfg['old_budget_groups'] <= cfg['shared_evaluated_groups_max'] and
            totals['candidates'] + cfg['old_budget_candidates'] <= cfg['shared_evaluated_candidates_max'],
            'Shared reservation cap')
    return rows


def reserve(out, cfg, key, groups, candidates):
    rows = budget_rows(out, cfg); key_name(key)
    require(not any(row['key'] == key for row in rows), 'No dispatch retry is permitted')
    require(type(groups) is int and type(candidates) is int and groups >= 0 and candidates >= 0, 'Invalid reservation')
    new_groups = sum(row['groups'] for row in rows) + groups
    new_candidates = sum(row['candidates'] for row in rows) + candidates
    require(new_groups <= cfg['new_evaluated_groups_max'] and new_candidates <= cfg['new_evaluated_candidates_max'],
            'New stage reservation cap')
    require(new_groups + cfg['old_budget_groups'] <= cfg['shared_evaluated_groups_max'] and
            new_candidates + cfg['old_budget_candidates'] <= cfg['shared_evaluated_candidates_max'], 'Shared reservation cap')
    path = Path(out) / 'budget-reservations.jsonl'
    previous = hashlib.sha256(path.read_bytes().splitlines()[-1]).hexdigest() if rows else cfg['old_artifacts_sha256']['budget-reservations.jsonl']
    row = dict(sequence=len(rows), previous_line_sha256=previous,
        predecessor_ledger_sha256=cfg['old_artifacts_sha256']['budget-reservations.jsonl'],
        carry_groups=cfg['old_budget_groups'], carry_candidates=cfg['old_budget_candidates'],
        key=key, groups=groups, candidates=candidates, reserved_before_dispatch=True, refunded=False,
        created_utc=datetime.now(timezone.utc).isoformat())
    raw = json.dumps(row, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode('utf-8')
    with path.open('ab') as stream:
        stream.write(raw + b'\n'); stream.flush(); os.fsync(stream.fileno())
    return hashlib.sha256(raw).hexdigest()


def worker(out, key):
    out = Path(out).resolve(); key_name(key); cfg = verify(out)
    request_path = out / 'run/job-inputs' / (key + '.json'); request = read(request_path)
    require(request['key'] == key and request['freeze_sha256'] == sha(out / 'freeze.json'), 'Worker identity/freeze mismatch')
    rows = [r for r in budget_rows(out, cfg) if r['key'] == key]
    require(len(rows) == 1, 'Missing unique reservation')
    raw = json.dumps(rows[0], separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode('utf-8')
    require(hashlib.sha256(raw).hexdigest() == request['budget_reservation_sha256'], 'Reservation binding mismatch')
    c = request['pipeline_config']
    require(c['max_evaluated_groups'] == rows[0]['groups'] and c['max_evaluated_candidates'] == rows[0]['candidates'],
            'Worker scoring cap mismatch')
    marker = read(out / 'started.json'); deadline = request['deadline_monotonic']
    limit = cfg['lifecycle_matrix_work_seconds_max'] if key == 'tree' else cfg['numeric_job_work_seconds_max']
    require(deadline <= marker['deadline_monotonic'] - cfg['close_seconds_max_including_grace'] and
            deadline - time.monotonic() <= limit, 'Worker work/cleanup allocation mismatch')
    check_time(deadline); resources(out, cfg)
    new_json(out / 'run/job-started' / (key + '.json'), dict(request_sha256=sha(request_path)))
    if key == 'tree':
        from .verify_tree import run_regressions
        result = run_regressions(out / 'run/tree')
    else:
        from learning.task_bc_diagnostics_capacity_v1 import pipeline
        c = dict(c, sample_root=str(OLD))
        target = out / 'run/candidate'
        if key == 'aggregate': result = pipeline.aggregate(target, c, deadline)
        elif key == 'compare':
            result = pipeline.compare_reference(target, OLD / 'preparation/reference', c, deadline)
            new_json(target / 'comparison.json', result)
        else:
            from learning.task_bc_v1.model import TaskNetwork
            model = TaskNetwork(ROOT / cfg['task_model'], cfg['task_model_sha256'])
            if key == 'build': result = pipeline.build_index(target, c, deadline, model)
            elif key.startswith('score-'): result = pipeline.score_shard(target, c, deadline, key[6:], model)
            else: raise ValueError('Unknown registered job')
    envelope = dict(key=key, result=result, request_sha256=sha(request_path), freeze_sha256=request['freeze_sha256'])
    new_json(out / 'run/job-results' / (key + '.json'), envelope)
    require(isinstance(result, dict) and result.get('passed') is True, 'Worker did not pass')
    if key == 'tree':
        require([row['kind'] for row in result['cases']] == cfg['lifecycle_cases'] and
                len(result['cases']) == cfg['lifecycle_case_count'] and all(row['passed'] for row in result['cases']) and
                result['sentinel_closed'], 'Incomplete registered lifecycle matrix')
    elif key.startswith('score-'):
        count = result['counts']
        require(count['evaluated_groups'] == count['completed_groups'] == rows[0]['groups'] and
                count['evaluated_candidates'] == count['completed_candidates'] == rows[0]['candidates'],
                'Scoring population mismatch')
        require(0 <= count['actual_score_calls'] <= count['evaluated_groups'] and
                0 <= count['actual_score_candidates'] <= count['evaluated_candidates'], 'Invalid actual score counters')
    elif key == 'compare':
        require(result['exact_compact_rows'] == cfg['sample_groups'] and
                result['exact_candidates'] == cfg['sample_candidates'] and result['byte_identical_summary'], 'Comparison mismatch')
    check_time(deadline); verify(out); resources(out, cfg); check_time(deadline)
    new_json(out / 'run/job-complete' / (key + '.json'), dict(passed=True, **envelope))


def run(out):
    from .windows_tree import run_owned
    out = Path(out).resolve(); cfg = ready(out)
    resources(out, cfg)
    started = time.monotonic(); deadline = started + cfg['stage_wall_seconds_max']
    marker = out / 'started.json'
    new_json(marker, dict(started_utc=datetime.now(timezone.utc).isoformat(), deadline_monotonic=deadline,
                         freeze_sha256=sha(out / 'freeze.json')))
    attempts = []; jobs = []; failure = None; previous = sha(marker)
    environment = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
                       PYTHONDONTWRITEBYTECODE='1', PYTHONUTF8='1', PYTHONIOENCODING='utf-8')
    pipeline_config = dict(sample_manifest_sha256=cfg['old_artifacts_sha256']['sample-manifest.json'],
        reference_freeze_sha256=cfg['old_artifacts_sha256']['preparation-freeze.json'],
        reference_receipt_sha256=cfg['old_artifacts_sha256']['preparation/reference/reference-receipt.json'])

    def persist():
        save(out / 'run/attempts.json', attempts); save(out / 'run/jobs-ledger.json', jobs)

    def dispatch(key, groups=0, candidates=0):
        nonlocal previous
        check_time(deadline); verify(out); resources(out, cfg)
        work_limit = cfg['lifecycle_matrix_work_seconds_max'] if key == 'tree' else cfg['numeric_job_work_seconds_max']
        work_deadline = min(time.monotonic() + work_limit, deadline - cfg['close_seconds_max_including_grace'])
        check_time(work_deadline)
        reservation = reserve(out, cfg, key, groups, candidates)
        c = dict(pipeline_config, freeze_sha256=sha(out / 'freeze.json'), previous_receipt_sha256=previous,
            max_evaluated_groups=groups, max_evaluated_candidates=candidates, model_sha256=cfg['task_model_sha256'])
        request = dict(key=key, freeze_sha256=c['freeze_sha256'], deadline_monotonic=work_deadline,
                       budget_reservation_sha256=reservation, pipeline_config=c)
        request_path = out / 'run/job-inputs' / (key + '.json'); new_json(request_path, request)
        attempt = dict(key=key, groups_reserved=groups, candidates_reserved=candidates,
                       budget_reservation_sha256=reservation, status='started', lifecycle={})
        attempts.append(attempt); persist(); print('exit-capacity:', key, flush=True)
        log_path = out / 'run/logs' / (key + '.log'); log_path.parent.mkdir(parents=True, exist_ok=True)
        tick = time.monotonic(); local_lifecycle = {}
        try:
            with log_path.open('x', encoding='utf-8') as log:
                code = run_owned([sys.executable, '-B', '-m', 'learning.task_bc_diagnostics_capacity_exit_v1.runner',
                                  'worker', '--out', str(out), '--key', key], cwd=ROOT, env=environment, stream=log,
                    timeout=work_deadline-time.monotonic(), phase_deadline=deadline, lifecycle=attempt['lifecycle'],
                    local_lifecycle=local_lifecycle, memory_limit_bytes=cfg['resource_limits']['process_tree_private_commit_bytes'])
            life = attempt['lifecycle']
            require(code == 0 and life['tree_closed'] and life['active_processes_after_close'] == 0,
                    'Worker code or whole-tree closure failed')
            require(life['natural_tree_exit'] and not life['termination_requested'] and life['containment_established'] and
                    life['job_handle_closed'] and life['all_process_handles_closed'], 'Forced or unproved worker completion')
            receipt = out / 'run/job-complete' / (key + '.json'); envelope = read(receipt)
            require(envelope['passed'] and envelope['key'] == key and envelope['request_sha256'] == sha(request_path) and
                    envelope['freeze_sha256'] == request['freeze_sha256'], 'Completion receipt binding mismatch')
            jobs.append(dict(key=key, receipt_sha256=sha(receipt), result=envelope['result']))
            previous = sha(receipt); attempt['status'] = 'passed'
            check_time(deadline); return envelope['result']
        except BaseException:
            attempt['status'] = 'failed'; raise
        finally:
            new_json(out / 'run/local-lifecycle' / (key + '.json'), local_lifecycle)
            attempt['local_lifecycle_sha256'] = sha(out / 'run/local-lifecycle' / (key + '.json'))
            attempt['elapsed_seconds'] = time.monotonic()-tick; persist()

    try:
        dispatch('tree')
        sample = read(OLD / 'sample-manifest.json')
        build = dispatch('build')
        pipeline_config.update(support_index_sha256=build['support_index_sha256'],
                               support_index_receipt_sha256=build['support_index_receipt_sha256'])
        hashes = {}
        for group in sample['groups']:
            result = dispatch('score-' + group['game'], group['rows'], group['candidates'])
            hashes[group['game']] = result['receipt_sha256']
        pipeline_config['shard_receipt_sha256'] = hashes
        result = dispatch('aggregate'); pipeline_config['aggregate_receipt_sha256'] = result['receipt_sha256']
        dispatch('compare')
        rows = budget_rows(out, cfg)
        require(sum(row['groups'] for row in rows) == cfg['sample_groups'] and
                sum(row['candidates'] for row in rows) == cfg['sample_candidates'], 'Incomplete sample reservation')
        require(len(jobs) == 14 and len(attempts) == 14, 'Incomplete registered job sequence')
        resources(out, cfg); verify(out); check_time(deadline)
        require(time.monotonic()-started+cfg['old_benchmark_wall_seconds'] <= cfg['prior_plus_new_wall_seconds_max'],
                'Combined capacity wall cap')
    except BaseException as exc:
        failure = dict(kind=type(exc).__name__, passed=False, retry=False)
        log_dir = out / 'run/logs'; log_dir.mkdir(parents=True, exist_ok=True)
        (log_dir / 'failure.log').write_text(traceback.format_exc(), encoding='utf-8')
        new_json(out / 'failure.json', failure)
    finally:
        persist(); rows = budget_rows(out, cfg)
        reserved_groups = sum(row['groups'] for row in rows); reserved_candidates = sum(row['candidates'] for row in rows)
        result = dict(passed=failure is None, sample_only=True, historical_reuse_not_new_holdout=True,
            elapsed_seconds=time.monotonic()-started, completed_jobs=len(jobs), attempts=len(attempts),
            owned_trees_closed=bool(attempts) and all(a['lifecycle'].get('tree_closed', False) for a in attempts),
            successful_workers_forced_termination=sum(a['lifecycle'].get('termination_requested', False) for a in attempts if a['status']=='passed'),
            reserved_real_evaluations=reserved_groups, reserved_real_candidates=reserved_candidates,
            carry_groups=cfg['old_budget_groups'], carry_candidates=cfg['old_budget_candidates'],
            cumulative_reserved_groups=reserved_groups+cfg['old_budget_groups'],
            cumulative_reserved_candidates=reserved_candidates+cfg['old_budget_candidates'],
            predecessor_ledger_sha256=cfg['old_artifacts_sha256']['budget-reservations.jsonl'],
            budget_ledger_sha256=sha(out / 'budget-reservations.jsonl') if rows else None,
            freeze_sha256=sha(out / 'freeze.json'), failure=failure, retries=0,
            training_runs=0, policy_calls=0, new_games=0, official_transitions=0, reference_rescoring=0,
            full_diagnostics_completed=False, old_capacity_stage_acceptance=False,
            original_diagnostic_stage_acceptance=False, original_policy_admission=False)
        new_json(out / 'summary.json', result)
    print('exit-capacity:', 'passed' if failure is None else 'stopped; failure preserved', flush=True)
    return failure is None


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('mode', choices=('run', 'worker'))
    parser.add_argument('--out', type=Path, required=True); parser.add_argument('--key'); args = parser.parse_args()
    if args.mode == 'worker': worker(args.out, args.key)
    else: sys.exit(0 if run(args.out) else 1)
