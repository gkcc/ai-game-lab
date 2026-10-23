"""One owned offline worker at a time; fail closed, never resume or retry.

The two panel passes are an inventory and a diagnostic pass. Replays are only
hashed during inventory; diagnostics read saved observations without stepping
an engine. Full observations, traces, pickle and logs remain local.
"""
import argparse
from collections import Counter
from contextlib import ExitStack
from datetime import datetime, timezone
import gzip
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys
import time
import traceback

from learning.bc_crop.agent import ROOT, sha
from learning.bc_crop.experiment import read, save
from learning.task_bc_v1.deployment import JointAgent
from learning.task_choice_v1.features import SCHEMA_VERSION
from .experiment import ready, verify


COUNTERS = ('turns', 'factual_groups', 'factual_candidates', 'shadow_policy_calls',
            'shadow_groups', 'shadow_candidates')
CAPS = dict(turns='max_source_turns', factual_groups='max_factual_groups',
            factual_candidates='max_factual_candidates', shadow_policy_calls='max_shadow_policy_calls',
            shadow_groups='max_shadow_groups', shadow_candidates='max_shadow_candidates')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def check_deadline(deadline):
    if time.monotonic() >= deadline:
        raise TimeoutError('Diagnostic deadline exceeded; no retry')


def timeout_seconds(deadline, per_job):
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError('Global diagnostic deadline exceeded; no dispatch')
    return min(per_job, remaining)


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def strict_json(line):
    def invalid(_):
        raise ValueError('Nonfinite JSON value')
    return json.loads(line, parse_constant=invalid)


def emit(stream, value):
    stream.write(json.dumps(value, separators=(',', ':'), ensure_ascii=False, allow_nan=False) + '\n')


def token(key):
    value = key.replace('game:', 'game-', 1) if key.startswith('game:') else key
    require(value not in ('', '.', '..') and all(c.isalnum() or c in '-_' for c in value),
            'Invalid job key')
    return value


def source_rows(cfg, out):
    panel = ROOT / cfg['panel_root']
    frozen = read(out / 'freeze.json')['input_files']
    ledger = panel / 'ledger.json'
    require(sha(ledger) == frozen[ledger.relative_to(ROOT).as_posix()], 'Source ledger changed')
    rows = read(ledger)
    expected = {(s, o, p, a) for s in cfg['seeds'] for o in cfg['opponents']
                for p in cfg['seats'] for a in cfg['origins']}
    require(len(rows) == len(expected) == 96 and len({r['game'] for r in rows}) == 96 and
            {(r['seed'], r['opponent'], r['side'], r['policy']) for r in rows} == expected,
            'Source game grid mismatch')
    order = {s: i for i, s in enumerate(cfg['seeds'])}
    return sorted(rows, key=lambda r: (order[r['seed']], cfg['opponents'].index(r['opponent']),
                                      cfg['origins'].index(r['policy']), r['side']))


def source_paths(cfg, row, out):
    panel = ROOT / cfg['panel_root']
    require(token(row['game']) == row['game'], 'Invalid source game identity')
    paths = dict(replay=panel / 'games' / row['game'] / 'replay.json.gz',
                 trace=panel / 'traces' / (row['game'] + '.jsonl.gz'))
    frozen = read(out / 'freeze.json')['input_files']
    for kind, field in (('replay', 'replay_gzip_sha256'), ('trace', 'trace_sha256')):
        require(sha(paths[kind]) == row[field] == frozen[paths[kind].relative_to(ROOT).as_posix()],
                'Frozen source hash mismatch: ' + kind)
    return paths


def validate_trace(trace, step, cfg, origin):
    require(trace['schema_version'] == SCHEMA_VERSION and type(trace['step']) is int and
            trace['step'] == step and isinstance(trace['records'], list),
            'Trace step/records mismatch')
    require(trace['checkpoint_sha256'] == cfg['crop_checkpoint_sha256'] and
            trace['task_checkpoint_sha256'] == (cfg['task_checkpoint_sha256'] if origin == 'trained' else None),
            'Trace checkpoint mismatch')
    require(all(trace[k] == 0 for k in ('fallbacks', 'load_errors', 'inference_errors')),
            'Factual trace reports an inference failure')


def inventory(out, cfg, deadline):
    games = []; totals = Counter()
    for row in source_rows(cfg, out):
        check_deadline(deadline)
        paths = source_paths(cfg, row, out)
        counts = Counter(turns=0, factual_groups=0, factual_candidates=0)
        with gzip.open(paths['trace'], 'rt', encoding='utf-8') as stream:
            for step, line in enumerate(stream):
                check_deadline(deadline)
                require(step < 719, 'Too many source trace turns')
                trace = strict_json(line)
                validate_trace(trace, step, cfg, row['policy'])
                counts['turns'] += 1
                for record in trace['records']:
                    candidates = record['inputs']['candidates']
                    require(isinstance(candidates, list) and len(candidates) > 0, 'Empty factual candidate set')
                    counts['factual_groups'] += 1
                    counts['factual_candidates'] += len(candidates)
                for field in counts:
                    require(totals[field] + counts[field] <= cfg['limits'][CAPS[field]], 'Factual inventory cap exceeded')
        require(counts['turns'] == 719 and row['action_turns'] == 719, 'Incomplete source trace')
        totals.update(counts)
        games.append({k: row[k] for k in ('game', 'seed', 'opponent', 'side', 'policy',
                                          'trace_sha256', 'replay_gzip_sha256')} | dict(counts=dict(counts)))
    require(totals['turns'] == 69024, 'Incomplete source population')
    write_new(out / 'inventory.json', dict(passed=True, games=games, counts=dict(totals),
        game_count=len(games), trace_processing_pass=1, replay_parses=0, official_transitions=0))


def support(out, cfg, deadline):
    from .numeric import build_support
    from learning.task_bc_v1.model import TaskNetwork
    model = TaskNetwork(ROOT / cfg['task_checkpoint'], cfg['task_checkpoint_sha256'])
    index, summary = build_support(ROOT / cfg['dataset_root'], model, deadline)
    check_deadline(deadline)
    with (out / 'support-index.pkl').open('xb') as stream:
        pickle.dump(index, stream, protocol=pickle.HIGHEST_PROTOCOL)
    write_new(out / 'support-summary.json', summary)
    write_new(out / 'support-receipt.json', dict(passed=True,
        index_sha256=sha(out / 'support-index.pkl'), summary_sha256=sha(out / 'support-summary.json'),
        checkpoint_sha256=cfg['task_checkpoint_sha256'], freeze_sha256=sha(out / 'freeze.json')))


class BudgetJoint(JointAgent):
    """Count before the unmodified selector encodes or scores candidates."""
    def __init__(self, *args, remaining, deadline, **kwargs):
        self.remaining, self.deadline = remaining, deadline
        self.used = Counter(shadow_groups=0, shadow_candidates=0)
        super().__init__(*args, **kwargs)

    def select(self, context):
        check_deadline(self.deadline)
        candidates = len(context['choices'])
        require(candidates > 0 and self.used['shadow_groups'] + 1 <= self.remaining['shadow_groups'] and
                self.used['shadow_candidates'] + candidates <= self.remaining['shadow_candidates'],
                'Shadow callback budget exhausted before scoring')
        self.used['shadow_groups'] += 1
        self.used['shadow_candidates'] += candidates
        return super().select(context)


def game(out, cfg, request, deadline):
    from decision.audit_market_teacher import observation
    from learning.task_bc_v1.model import TaskNetwork
    from .numeric import RecordAnalyzer
    from .shadow import compare_turn
    key = request['key'][5:]
    require(sha(out / 'inventory.json') == request['inventory_sha256'], 'Inventory receipt changed')
    row = next(r for r in read(out / 'inventory.json')['games'] if r['game'] == key)
    paths = source_paths(cfg, row, out)
    receipt_path = out / 'support-receipt.json'
    require(sha(receipt_path) == request['support_receipt_sha256'], 'Support receipt changed')
    receipt = read(receipt_path)
    require(receipt['passed'] and receipt['freeze_sha256'] == sha(out / 'freeze.json') and
            receipt['checkpoint_sha256'] == cfg['task_checkpoint_sha256'] and
            sha(out / 'support-summary.json') == receipt['summary_sha256'] and
            sha(out / 'support-index.pkl') == receipt['index_sha256'] == request['support_index_sha256'],
            'Untrusted support pickle; refusing deserialize')
    with (out / 'support-index.pkl').open('rb') as stream:
        index = pickle.load(stream)
    analyzer = RecordAnalyzer(index, TaskNetwork(ROOT / cfg['task_checkpoint'], cfg['task_checkpoint_sha256']))
    origin = row['policy']; opposite = 'teacher_task' if origin == 'trained' else 'trained'
    trained = opposite == 'trained'
    remaining = request['remaining']
    shadow = BudgetJoint(ROOT / cfg['crop_checkpoint'], cfg['crop_checkpoint_sha256'],
        ROOT / cfg['task_checkpoint'] if trained else None, cfg['task_checkpoint_sha256'] if trained else None,
        remaining=remaining, deadline=deadline)
    with gzip.open(paths['replay'], 'rt', encoding='utf-8') as stream:
        replay = strict_json(stream.read())
    require(len(replay['steps']) == 720 and replay['info']['seed'] == row['seed'], 'Replay population mismatch')
    counts = Counter({k: 0 for k in COUNTERS})
    outputs = {name: out / name / (key + '.jsonl.gz') for name in ('compact', 'turns', 'shadows')}
    with ExitStack() as stack:
        streams = {}
        for name, path in outputs.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            streams[name] = stack.enter_context(gzip.open(path, 'xt', encoding='utf-8', compresslevel=1))
        factual = stack.enter_context(gzip.open(paths['trace'], 'rt', encoding='utf-8'))
        for step, (before, after) in enumerate(zip(replay['steps'], replay['steps'][1:])):
            check_deadline(deadline)
            line = next(factual, None)
            require(line is not None, 'Missing factual trace turn')
            trace = strict_json(line); validate_trace(trace, step, cfg, origin)
            meta = dict(game=key, seed=row['seed'], opponent=row['opponent'], side=row['side'],
                        origin=origin, step=step, time_third=step // 240)
            require(counts['turns'] < remaining['turns'] and counts['shadow_policy_calls'] < remaining['shadow_policy_calls'],
                    'Turn/policy-call budget exhausted before policy execution')
            counts['shadow_policy_calls'] += 1
            action = shadow(observation(before, row['side']), replay['configuration'])
            require(shadow.last['step'] == step, 'Shadow observation step mismatch')
            comparison = compare_turn(after[row['side']]['action'], trace, action, shadow.last)
            emit(streams['turns'], comparison | meta)
            emit(streams['shadows'], dict(game=key, step=step, origin=origin,
                 shadow_origin=opposite, action=action, trace=shadow.last))
            for decision, record in enumerate(trace['records']):
                candidates = len(record['inputs']['candidates'])
                require(counts['factual_groups'] + 1 <= remaining['factual_groups'] and
                        counts['factual_candidates'] + candidates <= remaining['factual_candidates'],
                        'Factual scoring budget exhausted')
                counts['factual_groups'] += 1; counts['factual_candidates'] += candidates
                check_deadline(deadline)
                emit(streams['compact'], analyzer.evaluate(record, origin) | meta | dict(decision=decision))
            counts['turns'] += 1
        require(next(factual, None) is None, 'Extra factual trace turn')
    counts.update(shadow.used)
    require(all(counts[k] == v for k, v in row['counts'].items()) and
            counts['turns'] == counts['shadow_policy_calls'] == 719, 'Game inventory accounting mismatch')
    check_deadline(deadline)
    write_new(out / 'receipts' / (key + '.json'), dict(passed=True,
        **{k: row[k] for k in ('game', 'seed', 'opponent', 'side')}, origin=origin, shadow_origin=opposite,
        counts=dict(counts), inputs_sha256=dict(replay=row['replay_gzip_sha256'], trace=row['trace_sha256'],
            support_index=request['support_index_sha256'], task=cfg['task_checkpoint_sha256'],
            crop=cfg['crop_checkpoint_sha256']),
        outputs_sha256={name: sha(path) for name, path in outputs.items()},
        trace_processing_pass=2, official_transitions=0, shadow_executed=False))


def job(out, key):
    out = Path(out).resolve(); label = token(key)
    cfg = verify(out)
    request = read(out / 'job-inputs' / (label + '.json'))
    require(request['key'] == key and request['freeze_sha256'] == sha(out / 'freeze.json'), 'Job request mismatch')
    started = read(out / 'run-started.json')
    require(started['freeze_sha256'] == request['freeze_sha256'] and started['max_workers'] == 1,
            'Missing parent execution marker')
    deadline = request['deadline_monotonic']
    require(deadline <= started['deadline_monotonic'] and
            deadline - time.monotonic() <= cfg['limits']['per_job_seconds'], 'Unbounded job deadline')
    require(all(type(request['remaining'][k]) is int and
                0 <= request['remaining'][k] <= cfg['limits'][CAPS[k]] for k in COUNTERS),
            'Unbounded job counters')
    check_deadline(deadline)
    write_new(out / 'job-started' / (label + '.json'), dict(key=key, request_sha256=sha(out / 'job-inputs' / (label + '.json'))))
    if key == 'inventory':
        inventory(out, cfg, deadline)
    elif key == 'support':
        support(out, cfg, deadline)
    else:
        require(key.startswith('game:'), 'Unknown job kind')
        game(out, cfg, request, deadline)
    check_deadline(deadline)
    verify(out)
    check_deadline(deadline)
    write_new(out / 'job-complete' / (label + '.json'), dict(passed=True, key=key))


def run_owned(command, *, cwd, env, stream, timeout, lifecycle):
    """Always wait for this exact child, including after an interrupted wait."""
    lifecycle['owned_child_closed'] = True  # No child if construction fails.
    process = subprocess.Popen(command, cwd=cwd, env=env, stdout=stream,
        stderr=subprocess.STDOUT,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
    lifecycle['owned_child_closed'] = False
    try:
        return process.wait(timeout=timeout)
    finally:
        # Explicit ownership avoids Popen.__exit__'s short KeyboardInterrupt wait.
        # If cleanup itself fails, leave the receipt false instead of claiming closure.
        try:
            if process.poll() is None:
                process.kill()
        finally:
            process.wait()
        lifecycle['owned_child_closed'] = True


def run(out):
    out = Path(out).resolve()
    started = time.monotonic(); cfg = ready(out); limits = cfg['limits']
    require(limits['wall_seconds'] == 14400 and limits['per_job_seconds'] == 900 and
            limits['max_workers'] == 1 and limits['max_source_games'] == 96 and
            limits['max_source_turns'] == limits['max_shadow_policy_calls'] == 69024 and
            limits['max_factual_groups'] == limits['max_shadow_groups'] == 600000 and
            limits['max_factual_candidates'] == limits['max_shadow_candidates'] == 40000000,
            'Unregistered runtime budgets')
    deadline = started + limits['wall_seconds']
    write_new(out / 'run-started.json', dict(freeze_sha256=sha(out / 'freeze.json'),
        started_utc=datetime.now(timezone.utc).isoformat(), deadline_monotonic=deadline, max_workers=1))
    attempts = []; ledger = []; failure = None; totals = Counter({k: 0 for k in COUNTERS})
    environment = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
                       PYTHONDONTWRITEBYTECODE='1')
    def persist():
        save(out / 'attempts.json', attempts)
        save(out / 'jobs-ledger.json', ledger)
    def dispatch(key):
        check_deadline(deadline); verify(out)
        seconds = timeout_seconds(deadline, limits['per_job_seconds'])
        label = token(key)
        request = dict(key=key, freeze_sha256=sha(out / 'freeze.json'),
            deadline_monotonic=min(deadline, time.monotonic() + seconds),
            remaining={k: limits[CAPS[k]] - totals[k] for k in COUNTERS})
        if key.startswith('game:'):
            request.update(inventory_sha256=sha(out / 'inventory.json'),
                support_receipt_sha256=sha(out / 'support-receipt.json'),
                support_index_sha256=read(out / 'support-receipt.json')['index_sha256'])
        write_new(out / 'job-inputs' / (label + '.json'), request)
        attempt = dict(key=key, timeout_seconds=seconds, attempt=1, status='started',
                       request_sha256=sha(out / 'job-inputs' / (label + '.json')))
        attempts.append(attempt); persist()
        print('diagnostics: starting ' + key, flush=True)
        log = out / 'logs' / (label + '.log'); log.parent.mkdir(parents=True, exist_ok=True)
        job_start = time.monotonic()
        try:
            with log.open('x', encoding='utf-8') as stream:
                # Workers do not create grandchildren or execute engine steps.
                returncode = run_owned([sys.executable, '-B', '-m',
                    'learning.task_bc_error_diagnostics_v1.runner', 'job', '--out', str(out), '--key', key],
                    cwd=ROOT, env=environment, stream=stream,
                    timeout=timeout_seconds(deadline, seconds), lifecycle=attempt)
            attempt.update(returncode=returncode, status='exited')
            require(returncode == 0, 'Offline worker failed; no retry')
            require(read(out / 'job-complete' / (label + '.json')) == dict(passed=True, key=key), 'Missing worker completion')
            check_deadline(deadline)
            if key.startswith('game:'):
                receipt_path = out / 'receipts' / (key[5:] + '.json')
                receipt = read(receipt_path)
                require(receipt['passed'] and receipt['game'] == key[5:], 'Game receipt mismatch')
                for field in COUNTERS:
                    value = receipt['counts'][field]
                    require(type(value) is int and 0 <= value <= request['remaining'][field], 'Budget receipt mismatch')
                totals.update(receipt['counts'])
                ledger.append(dict(key=key, passed=True, receipt_sha256=sha(receipt_path), counts=receipt['counts']))
            else:
                result_path = out / ('inventory.json' if key == 'inventory' else 'support-receipt.json')
                require(read(result_path)['passed'], 'Worker artifact failed')
                ledger.append(dict(key=key, passed=True, receipt_sha256=sha(result_path)))
            attempt['status'] = 'passed'
        except subprocess.TimeoutExpired:
            attempt['status'] = 'timeout'
            attempt['owned_child_killed_and_reaped'] = attempt.get('owned_child_closed', True)
            raise
        except BaseException:
            attempt['status'] = 'failed'
            raise
        finally:
            attempt['elapsed_seconds'] = time.monotonic() - job_start
            attempt.setdefault('owned_child_closed', True)
            persist()
    try:
        verify(out, history=True)
        dispatch('inventory'); dispatch('support')
        for row in read(out / 'inventory.json')['games']:
            dispatch('game:' + row['game'])
        require(len(ledger) == 98 and totals['turns'] == totals['shadow_policy_calls'] == 69024,
                'Incomplete diagnostic population')
        require(all(totals[k] == v for k, v in read(out / 'inventory.json')['counts'].items()),
                'Final factual inventory mismatch')
        require(all(a['owned_child_closed'] for a in attempts), 'Unclosed worker; analysis prohibited')
        save(out / 'workers-closed.json', dict(passed=True, active=0, owned_children_closed=True,
             completed_jobs=len(ledger), max_concurrent_workers=1))
        from .analysis import finish
        finish(out, cfg, deadline)
        check_deadline(deadline); verify(out, history=True); check_deadline(deadline)
    except BaseException as exc:
        failure = dict(passed=False, kind=type(exc).__name__, retry=False,
                       completed_jobs=len(ledger), counts_are_completed_jobs_only=True)
        (out / 'logs').mkdir(exist_ok=True)
        (out / 'logs' / 'failure.log').write_text(traceback.format_exc(), encoding='utf-8')
        save(out / 'failure.json', failure)
    finally:
        persist()
        closed = all(a.get('owned_child_closed', True) for a in attempts)
        save(out / 'workers-closed.json', dict(passed=closed, active=0 if closed else None, owned_children_closed=closed,
             completed_jobs=len(ledger), max_concurrent_workers=1))
        save(out / 'run-summary.json', dict(passed=failure is None, completed_games=sum(r['key'].startswith('game:') for r in ledger),
             completed_jobs=len(ledger), counts=dict(totals), counts_are_completed_jobs_only=True,
             elapsed_seconds=time.monotonic()-started, failures=int(failure is not None), retries=0,
             max_concurrent_workers=1, workers_closed=closed, training_runs=0, new_games=0, official_transitions=0,
             freeze_sha256=sha(out / 'freeze.json')))
    print('diagnostics: ' + ('complete' if failure is None else 'stopped; local failure receipt retained'), flush=True)
    return failure is None


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=('run', 'job'))
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--key')
    args = parser.parse_args()
    if args.mode == 'job':
        if not args.key:
            parser.error('job requires --key')
        job(args.out, args.key)
    else:
        sys.exit(0 if run(args.out) else 1)
