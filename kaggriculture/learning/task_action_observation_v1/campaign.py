"""Fixed 24-game development observation; no training, retries or admission."""
import argparse
from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import sys
import time

import psutil
from learning.task_anchor_once_v1.games import control_difference
from learning.task_anchor_once_v1.processes import CapturedProcess
from learning.task_anchor_once_v1.storage import owned_writes
from learning.task_launcher_contract_repair_v1.observer import reap
from .experiment import ROOT, OUT, ready, verify, read, sha, write_new, deadline_check
from .jobs import make_wrapper, task_identity

ARMS = ('baseline', 'candidate', 'teacher_task')
OPPONENTS = ('round2', 'market')
EXPECTED = ('initial_banks', 'banks', 'bank', 'opponent_bank', 'margin',
            'net_profit', 'opponent_net_profit', 'rewards', 'status', 'score')


def plans(cfg):
    seeds = cfg['validation_seeds']
    assert len(seeds) == len(set(seeds)) == 2 and all(type(s) is int for s in seeds)
    assert cfg['test_seeds'] == [] and cfg['max_games'] == 24 and cfg['max_workers'] == 2
    assert tuple(cfg['opponents']) == OPPONENTS and tuple(cfg['arms']) == ARMS
    assert cfg['seats'] == [0, 1]
    return [dict(split='dev', seed=seed, opponent=opponent, side=side,
                 policy=arm, parity_kind=None,
                 game=f'dev-{seed}-{opponent}-{side}-{arm}')
            for seed in seeds for opponent in OPPONENTS for side in (0, 1) for arm in ARMS]


def owned_size(out):
    return sum(p.stat().st_size for p in Path(out).rglob('*') if p.is_file())


def fixture_receipts(out):
    """Require the model-free entry and its complete cleanup before any game."""
    names = ('fixture-started.json', 'fixture-guard-result.json', 'fixture-result.json',
             'analysis-target-result.json', 'analysis-worker-result.json', 'transport-result.json')
    folder = Path(out)/'entry-fixture'
    records = {name: read(folder/name) for name in names}
    start, guard, result, target, worker, transport = (records[name] for name in names)
    assert start['fixture'] is True and start['freeze_sha256'] == sha(Path(out)/'freeze.json')
    assert guard['fixture'] is True and guard['completed'] is True
    assert guard['within_total_deadline'] is True and guard['cleanup']['proven'] is True
    assert result['passed'] is True and result['complete'] is True and result['marker_verified'] is True
    assert all(result[k] == 0 for k in ('scientific_raw_files_read', 'model_array_decodes',
        'neural_forwards', 'official_transitions', 'new_games', 'training_updates', 'gpu_seconds'))
    assert target['passed'] is True and target['fixture'] is True
    assert worker['passed'] is True and worker['fixture'] is True
    assert worker['cleanup']['target_handle_closed'] is True
    assert transport['passed'] is True and transport['fixture'] is True
    assert transport['linux_cleanup_proven'] is True and transport['windows_cleanup_proven'] is True
    assert transport['within_transport_deadline'] is True
    return {'entry-fixture/'+name: sha(folder/name) for name in names}


class ParentWrites:
    """Cumulative short write contexts; spawning and pipe readers stay outside."""
    def __init__(self, out, limit):
        self.out, self.limit = Path(out), limit
        self.used = self.refused = self.created = 0

    @property
    def remaining_bytes(self):
        return self.limit-self.used

    @contextmanager
    def scope(self, paths):
        budget = owned_writes(self.out, self.remaining_bytes, allowed_paths=paths)
        try:
            with budget:
                yield
        finally:
            value = budget.snapshot()
            self.used += value['used_bytes']
            self.refused += value['refused']
            self.created += value['created_count']

    def write(self, name, value):
        with self.scope([name]):
            write_new(self.out/name, value)

    def snapshot(self):
        return dict(limit_bytes=self.limit, used_bytes=self.used,
                    remaining_bytes=self.remaining_bytes, refused=self.refused,
                    created_count=self.created)


def observe_processes(child, known):
    """Record only this launched process tree, with creation identity per PID."""
    if child.process.poll() is not None:
        return
    try:
        root = psutil.Process(child.process.pid)
        nodes = [root, *root.children(recursive=True)]
        for process in nodes:
            try:
                created = process.create_time()
            except psutil.NoSuchProcess:
                continue
            if process.pid in known and known[process.pid]['created'] != created:
                raise RuntimeError('Owned process identity changed')
            known[process.pid] = dict(created=created)
    except psutil.NoSuchProcess:
        if child.process.poll() is None:
            raise


class Runner:
    def __init__(self, out, cfg, parent, queue):
        self.out, self.cfg, self.parent, self.queue = Path(out), cfg, parent, queue
        self.start = time.monotonic()
        self.attempts, self.rows, self.certificates, self.cleanups = [], [], [], []
        self.cleanup_complete = False
        self.environment = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1',
                                PYTHONDONTWRITEBYTECODE='1')

    def limits(self):
        deadline_check()
        if (self.out/'campaign-cancel.local.json').exists():
            raise InterruptedError('Owned campaign cancelled')
        assert time.monotonic()-self.start < self.cfg['campaign_seconds'], 'Campaign deadline'
        assert len(self.attempts) <= 24
        counts = Counter()
        for certificate in self.certificates:
            counts.update(certificate['total_model_counts'])
        for count, limit in (('task_forwards', 'task_forward_budget'),
                             ('crop_forwards', 'crop_forward_budget')):
            assert counts[count] <= self.cfg[limit]
        assert counts['task_rows']+counts['crop_rows'] <= self.cfg['neural_input_row_budget']
        assert sum(c['policy_turns'] for c in self.certificates) <= self.cfg['policy_turn_budget']
        return dict(counts)

    def storage_ready(self, active):
        storage = self.cfg['storage']
        pending = ((active+1)*storage['game_write_bytes'] + self.parent.remaining_bytes
                   + (24-len(self.rows))*storage['process_log_bytes']
                   + (25-len(self.rows))*storage['small_receipt_bytes'])
        current = owned_size(self.out)
        assert current+pending <= storage['total_bytes'], 'Owned output reservation exceeded'
        assert shutil.disk_usage(self.out).free >= storage['free_floor_bytes']+pending, 'Insufficient reserved free space'

    def collect(self, item, child, capture, cleanup):
        key = item['game']
        paths = {name: self.out/'job-results'/(key+'.'+name+'.json')
                 for name in ('row', 'audit', 'complete', 'storage')}
        values = {name: read(path) for name, path in paths.items()}
        row, certificate, complete, storage = (values[k] for k in ('row', 'audit', 'complete', 'storage'))
        assert complete['game'] == key and complete['passed'] is True
        assert complete['row_sha256'] == sha(paths['row'])
        assert complete['certificate_sha256'] == sha(paths['audit'])
        assert storage['game'] == key and storage['passed'] is True and storage['error_kind'] is None
        assert storage['output']['refused'] == 0
        assert storage['output']['limit_bytes'] == self.cfg['storage']['game_write_bytes']
        assert 0 <= storage['output']['used_bytes'] <= self.cfg['storage']['game_write_bytes']
        assert all(row[k] == value for k, value in item.items()) and row['passed'] is True
        assert row['states'] == 720 and row['action_turns'] == 719 and row['agent_errors'] == 0
        assert row['status'] == ['DONE', 'DONE'] and row['engine_sha256'] == self.cfg['engine_sha256']
        assert certificate['game'] == key and certificate['passed'] is True and certificate['errors'] == 0
        assert certificate['actor_actions_reproduced'] == certificate['rival_actions_reproduced'] == certificate['trace_turns_reproduced'] == 719
        assert certificate['replay_sha256'] == row['replay_gzip_sha256']
        assert certificate['trace_sha256'] == row['trace_sha256']
        assert certificate['official']['engine_sha256'] == self.cfg['engine_sha256']
        assert certificate['official']['final_rewards'] == row['banks']
        certificate = dict(certificate, artifact_sha256={name: sha(path) for name, path in paths.items()},
                           output_write_bytes=storage['output']['used_bytes'], capture=capture, cleanup=cleanup)
        self.rows.append(row)
        self.certificates.append(certificate)
        print(json.dumps(dict(completed=len(self.rows), attempted=len(self.attempts), game=key)), flush=True)

    def execute(self):
        active, cursor = {}, 0
        try:
            while active or cursor < len(self.queue):
                self.limits()
                for key, state in list(active.items()):
                    child = state['child']
                    observe_processes(child, state['known'])
                    child.check()
                    code = child.process.poll()
                    if code is None:
                        if time.monotonic()-state['start'] >= self.cfg['per_job_timeout_seconds']:
                            raise TimeoutError('Owned official job deadline')
                        continue
                    cleanup = reap(child, state['known'])
                    self.cleanups.append(dict(game=key, **cleanup))
                    capture = child.finish()
                    assert code == 0, 'Official job failed; no retry'
                    self.collect(state['item'], child, capture, cleanup)
                    del active[key]
                self.limits()
                while len(active) < self.cfg['max_workers'] and cursor < len(self.queue):
                    assert verify(self.out) == self.cfg, 'Configuration changed'
                    self.storage_ready(len(active))
                    plan = self.queue[cursor]
                    key = plan['game']
                    with self.parent.scope(['agents/'+key+'.py', 'job-inputs/'+key+'.json']):
                        bot = make_wrapper(self.out, plan, self.cfg)
                        _, digest = task_identity(self.out, plan, self.cfg)
                        item = dict(plan, bot_sha256=sha(bot),
                            rival_sha256=sha(self.out/'sources'/(plan['opponent']+'.py')),
                            task_checkpoint_sha256=digest)
                        write_new(self.out/'job-inputs'/(key+'.json'), item)
                    self.attempts.append(item)
                    cursor += 1
                    # Immutable job-input is the consumed attempt marker, before spawn.
                    child = CapturedProcess([sys.executable, '-B', '-X', 'utf8', '-m',
                        'learning.task_action_observation_v1.jobs', '--out',
                        self.out.relative_to(ROOT).as_posix(), '--key', key],
                        cwd=ROOT, env=self.environment, log=self.out/'job-local'/(key+'.log'),
                        limit_bytes=self.cfg['storage']['process_log_bytes'])
                    active[key] = dict(child=child, start=time.monotonic(), known={}, item=item)
                    observe_processes(child, active[key]['known'])
                if active:
                    time.sleep(.2)
        finally:
            errors = []
            for key, state in active.items():
                try:
                    cleanup = reap(state['child'], state['known'])
                    self.cleanups.append(dict(game=key, **cleanup))
                except BaseException as exc:
                    errors.append(dict(game=key, error_type=type(exc).__name__))
            self.cleanup_complete = not errors
            if errors:
                self.parent.write('cleanup-failure.json', dict(errors=errors, passed=False))
                raise RuntimeError('Owned process cleanup not proven')

    def finish(self):
        assert len(self.rows) == len(self.attempts) == len(self.queue) == 24
        index = {row['game']: row for row in self.rows}
        assert len(index) == 24 and set(index) == {p['game'] for p in self.queue}
        self.rows = [index[plan['game']] for plan in self.queue]
        pairs, games = [], []
        for seed in self.cfg['validation_seeds']:
            for opponent in OPPONENTS:
                for side in (0, 1):
                    selected = {r['policy']: r for r in self.rows
                                if (r['seed'], r['opponent'], r['side']) == (seed, opponent, side)}
                    assert set(selected) == set(ARMS)
                    first = control_difference(self.out, selected['candidate'], selected['baseline'])
                    selected['candidate']['first_request_difference_step'] = first
                    pairs.append(dict(seed=seed, opponent=opponent, side=side,
                        game_ids={arm: selected[arm]['game'] for arm in ARMS},
                        first_candidate_baseline_request_difference=first))
        for row in self.rows:
            identity = dict(id=row['game'], key=row['game'], arm=row['policy'], split='development',
                            seed=row['seed'], opponent=row['opponent'], side=row['side'],
                            expected={k: row[k] for k in EXPECTED})
            for name, relative, expected in (
                ('replay', 'games/'+row['game']+'/replay.json.gz', row['replay_gzip_sha256']),
                ('trace', 'traces/'+row['game']+'.jsonl.gz', row['trace_sha256'])):
                path = self.out/relative
                assert sha(path) == expected
                identity[name] = dict(path=relative, size=path.stat().st_size, sha256=expected)
            games.append(identity)
        assert len(pairs) == 8
        assert verify(self.out) == self.cfg
        counts = self.limits()
        assert self.cleanup_complete
        component_counts = {}
        for field in ('live_model_counts', 'primary_model_counts'):
            component = Counter()
            for certificate in self.certificates:
                component.update(certificate[field])
            component_counts[field] = dict(component)
        self.parent.write('ledger.json', self.rows)
        self.parent.write('audit.json', self.certificates)
        self.parent.write('input-manifest.json', dict(schema='task-action-observation-input-manifest-v1',
            games=games, pairs=pairs, raw_files=48, development_only=True,
            ledger_sha256=sha(self.out/'ledger.json'), audit_sha256=sha(self.out/'audit.json')))
        result = dict(schema='task-action-observation-campaign-v1', status='completed_development_observation',
            passed=True, completed_games=24, attempted_games=24, all_attempts_completed=True,
            live_official_transitions=24*719, primary_official_transitions=24*719,
            total_official_transitions=48*719, game_model_counts=counts, **component_counts,
            policy_turns=sum(c['policy_turns'] for c in self.certificates),
            machine_seconds=time.monotonic()-self.start, training_updates=0, fit_attempts=0,
            default_changed=False, model_changed=False, candidate_admission=False,
            holdout_run=False, retry_allowed=False, all_owned_processes_reaped=True,
            input_manifest_sha256=sha(self.out/'input-manifest.json'),
            registration_sha256=sha(self.out/'registration.json'), freeze_sha256=sha(self.out/'freeze.json'))
        self.parent.write('campaign-result.json', result)
        return result


def run(out=OUT):
    out = Path(out).resolve()
    cfg = ready(out)
    deadline_check()
    fixture = fixture_receipts(out)
    queue, storage = plans(cfg), cfg['storage']
    assert out.is_relative_to(ROOT)
    assert not (out/'campaign-started.json').exists(), 'Attempt already consumed'
    initial = owned_size(out)
    reserved = 24*storage['process_log_bytes']+25*storage['small_receipt_bytes']
    allowance = storage['primary_parent_write_bytes']-initial-reserved
    assert allowance > 0
    assert 24*storage['game_write_bytes']+storage['primary_parent_write_bytes'] <= storage['total_bytes']
    caps = cfg['game_limits']
    assert 32*caps['task_forwards'] <= cfg['task_forward_budget']
    assert 48*caps['crop_forwards'] <= cfg['crop_forward_budget']
    assert 48*caps['neural_input_rows'] <= cfg['neural_input_row_budget']
    assert 48*(caps['controller_calls']+719) <= cfg['policy_turn_budget']
    parent, runner, error = ParentWrites(out, allowance), None, None
    try:
        parent.write('campaign-started.json', dict(attempt_consumed=True, retry_allowed=False,
            registration_sha256=sha(out/'registration.json'), freeze_sha256=sha(out/'freeze.json'),
            started_utc=datetime.now(timezone.utc).isoformat(), fixed_plan=queue,
            accepted_entry_fixture_sha256=fixture))
        runner = Runner(out, cfg, parent, queue)
        runner.execute()
        return runner.finish()
    except BaseException as exc:
        error = type(exc).__name__
        raise
    finally:
        with owned_writes(out, storage['small_receipt_bytes'],
                          allowed_paths=['campaign-storage.json', 'failure.json']):
            write_new(out/'campaign-storage.json', dict(passed=error is None,
                output=parent.snapshot(), initial_owned_bytes=initial,
                bounded_logs_and_receipts_reserved_bytes=reserved,
                actual_owned_bytes_at_exit=owned_size(out),
                all_owned_processes_reaped=runner.cleanup_complete if runner else None))
            if error is not None:
                write_new(out/'failure.json', dict(kind=error, retry_allowed=False,
                    attempted_games=len(runner.attempts) if runner else 0,
                    completed_games=len(runner.rows) if runner else 0,
                    partial_work_not_complete_accounting=True))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['run'])
    parser.add_argument('--out', type=Path, default=OUT)
    arguments = parser.parse_args()
    print(json.dumps(run(arguments.out)), flush=True)
