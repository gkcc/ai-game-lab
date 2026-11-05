"""Pure panel, reporting and resource contracts; no private inputs at import."""
from contextlib import contextmanager
import argparse
import math
import re
import sys
import time
from types import SimpleNamespace


COHORTS = {'C01': {'S02': 'F01', 'S03': 'F02'},
           'C02': {'S02': 'F03', 'S03': 'F04'}}
OPPONENTS = ('market', 'round2')
SIDES = (0, 1)
ARMS = ('baseline', 'control')
FIELDS = ('own_net_profit', 'opponent_net_profit', 'margin', 'score')
PLAN_KEYS = frozenset(('game', 'seed_alias', 'seed', 'opponent', 'side', 'policy', 'split', 'parity_kind'))
JOB_KEYS = PLAN_KEYS | frozenset(('bot_sha256', 'rival_sha256', 'task_checkpoint_sha256'))
HEX = re.compile(r'[0-9a-f]{64}\Z')
MIB = 1024**2
GIB = 1024**3
LIMITS = dict(initial_available_bytes=6*GIB, between_available_bytes=6*GIB,
    running_available_bytes=4*GIB, readiness_samples=3, readiness_interval_seconds=10,
    readiness_timeout_seconds=300, max_rss_bytes=2*GIB, max_commit_bytes=3*GIB,
    job_wall_seconds=900, cleanup_seconds=10, poll_seconds=1,
    total_wall_seconds=8*60*60, disk_floor_bytes=4*GIB,
    per_game_reserve_bytes=160*MIB, per_stage_output_bytes=2*GIB,
    total_output_bytes=4*GIB, private_result_bytes=16*MIB,
    max_games=16, max_official_replays=16, max_attempts=1)


def require(condition):
    if not condition:
        raise ValueError('Teacher panel contract differs')


def finite(value):
    require(type(value) in (int, float) and math.isfinite(value))
    return value


def plans(panel):
    """Eight teacher plans for a two-seed manifest, never model-shaped."""
    require(type(panel.seeds) is tuple and len(panel.seeds) == 2
            and all(type(v) is int and 0 <= v < 2**32 for v in panel.seeds)
            and len(set(panel.seeds)) == 2)
    result = []
    for index, alias in enumerate(('S02', 'S03')):
        for opponent in OPPONENTS:
            for side in SIDES:
                result.append(dict(game=f'dev-{alias}-{opponent}-{side}-teacher_task',
                    seed_alias=alias, seed=panel.seeds[index], opponent=opponent,
                    side=side, policy='teacher_task', split='dev', parity_kind=None))
    require(len(result) == 8)
    return result


def validate_plan(plan, expected, *, job=False):
    require(type(job) is bool and type(plan) is dict and type(expected) is dict
            and set(plan) == (JOB_KEYS if job else PLAN_KEYS) and set(expected) == PLAN_KEYS
            and all(type(plan[key]) is type(expected[key]) and plan[key] == expected[key]
                    for key in PLAN_KEYS)
            and plan['seed_alias'] in ('S02', 'S03') and plan['opponent'] in OPPONENTS
            and type(plan['side']) is int and plan['side'] in SIDES
            and type(plan['seed']) is int and 0 <= plan['seed'] < 2**32
            and plan['policy'] == 'teacher_task' and plan['split'] == 'dev'
            and plan['parity_kind'] is None
            and plan['game'] == f"dev-{plan['seed_alias']}-{plan['opponent']}-{plan['side']}-teacher_task")
    if job:
        require(plan['task_checkpoint_sha256'] is None
                and all(type(plan[key]) is str and HEX.fullmatch(plan[key]) is not None
                        for key in ('bot_sha256', 'rival_sha256')))
    return dict(plan)


@contextmanager
def teacher_adapter(module):
    """Temporarily extend the old teacher verifier to both fixed opponents."""
    previous = module.validate_plan
    module.validate_plan = validate_plan
    try:
        yield module
    finally:
        module.validate_plan = previous


@contextmanager
def hide_parsed_argv(run_match):
    """Let the official CLI parse once, then remove argv before Agent import.

    This limits one accidental information path, not same-process adversaries.
    The argv and parser are restored on every exit; the worker owns its cwd.
    """
    original, original_argv = run_match.argparse, sys.argv
    class Parser(argparse.ArgumentParser):
        def parse_args(self, *args, **kwargs):
            parsed = super().parse_args(*args, **kwargs)
            sys.argv = [str(run_match.__file__)]
            return parsed
    run_match.argparse = SimpleNamespace(ArgumentParser=Parser)
    try:
        yield
    finally:
        run_match.argparse = original
        sys.argv = original_argv


def await_memory(reader, *, clock=time.monotonic, sleep=time.sleep,
                 floor=LIMITS['initial_available_bytes'], samples=LIMITS['readiness_samples'],
                 interval=LIMITS['readiness_interval_seconds'], timeout=LIMITS['readiness_timeout_seconds']):
    """Wait for three consecutive sampled values >= floor; do not start a job.

    The caller chooses a deadline <= its campaign deadline. A timeout raises
    with the last sampled bytes attached, rather than inventing a stop reading.
    """
    require(callable(reader) and callable(clock) and callable(sleep)
            and type(floor) is int and floor > 0 and type(samples) is int and samples >= 1
            and type(interval) in (int, float) and math.isfinite(interval) and interval > 0
            and type(timeout) in (int, float) and math.isfinite(timeout)
            and timeout >= interval*(samples-1))
    start = previous = clock()
    require(type(start) in (int, float) and math.isfinite(start))
    last_sample_time, count, last = None, 0, None
    while True:
        now = clock()
        require(type(now) in (int, float) and math.isfinite(now) and now >= previous)
        previous = now
        if now-start > timeout:
            exc = TimeoutError('Stable memory gate did not open')
            exc.last_available_bytes = last
            raise exc
        if last_sample_time is None or now - last_sample_time >= interval:
            last = reader()
            require(type(last) is int and last >= 0)
            sampled = clock()
            require(type(sampled) in (int, float) and math.isfinite(sampled) and sampled >= now)
            now = previous = sampled
            if now-start > timeout:
                exc = TimeoutError('Stable memory gate did not open')
                exc.last_available_bytes = last
                raise exc
            last_sample_time = now
            count = count + 1 if last >= floor else 0
            if count == samples:
                return dict(consecutive_samples=count, last_available_bytes=last,
                            elapsed_seconds=now-start)
        remaining = timeout - (now-start)
        if remaining <= 0:
            exc = TimeoutError('Stable memory gate did not open')
            exc.last_available_bytes = last
            raise exc
        until_sample = max(0, interval-(now-last_sample_time))
        sleep(min(remaining, until_sample))


def _metrics(row):
    require(type(row) is dict)
    out = {key: finite(row[key]) for key in FIELDS}
    require(out['score'] == (1. if out['margin'] > 0 else .5 if out['margin'] == 0 else 0.))
    return out


def teacher_metrics(row):
    require(type(row) is dict)
    return _metrics(dict(own_net_profit=row['net_profit'],
        opponent_net_profit=row['opponent_net_profit'], margin=row['margin'], score=row['score']))


def projection(teacher_rows, prior_public, commit):
    """Produce a private full ledger and an aggregate-only public projection.

    Inputs must already have been authenticated. No source, receipt or game is
    opened here, and no statistics are emitted for an incomplete panel.
    """
    require(type(commit) is str and re.fullmatch(r'[0-9a-f]{40}', commit) is not None
            and type(teacher_rows) is list and len(teacher_rows) == 16
            and type(prior_public) is dict and type(prior_public.get('pairs')) is list
            and len(prior_public['pairs']) == 16)
    old = {}
    for pair in prior_public['pairs']:
        require(type(pair) is dict)
        key = (pair.get('seed_alias'), pair.get('opponent'), pair.get('side'))
        require(key not in old and key[0] in {'F01', 'F02', 'F03', 'F04'}
                and key[1] in OPPONENTS and type(key[2]) is int and key[2] in SIDES)
        old[key] = {arm: _metrics(pair[arm]) for arm in ARMS}
    expected = {(role, opponent, side) for role in ('F01', 'F02', 'F03', 'F04')
                for opponent in OPPONENTS for side in SIDES}
    require(set(old) == expected)
    full, seen = [], set()
    for item in teacher_rows:
        require(type(item) is dict and item.get('cohort') in COHORTS and type(item.get('row')) is dict)
        row, cohort = item['row'], item['cohort']
        alias = row.get('seed_alias')
        require(alias in COHORTS[cohort] and type(row.get('side')) is int and row['side'] in SIDES)
        key = (COHORTS[cohort][alias], row.get('opponent'), row.get('side'))
        require(key in expected and key not in seen)
        seen.add(key)
        teacher = teacher_metrics(row)
        students = old[key]
        full.append(dict(seed_alias=key[0], opponent=key[1], side=key[2], teacher=teacher,
            baseline=students['baseline'], control=students['control'],
            teacher_minus_student={arm: {field: teacher[field]-students[arm][field]
                                         for field in FIELDS} for arm in ARMS}))
    require(seen == expected)

    def summarize(rows):
        require(rows)
        return {arm: dict(pairs=len(rows),
            mean_teacher={field: finite(sum(x['teacher'][field] for x in rows)/len(rows))
                          for field in FIELDS},
            mean_student={field: finite(sum(x[arm][field] for x in rows)/len(rows))
                          for field in FIELDS},
            mean_teacher_minus_student={field: sum(x['teacher_minus_student'][arm][field] for x in rows)/len(rows)
                                        for field in FIELDS},
            teacher_wins=sum(x['teacher']['score'] == 1. for x in rows),
            teacher_draws=sum(x['teacher']['score'] == .5 for x in rows),
            teacher_losses=sum(x['teacher']['score'] == 0. for x in rows),
            student_wins=sum(x[arm]['score'] == 1. for x in rows),
            student_draws=sum(x[arm]['score'] == .5 for x in rows),
            student_losses=sum(x[arm]['score'] == 0. for x in rows)) for arm in ARMS}

    private = dict(schema='teacher-fresh-panel-private-ledger-v1', registration_commit=commit,
                   pairs=full, by_seed={role:summarize([x for x in full if x['seed_alias']==role])
                                       for role in ('F01','F02','F03','F04')})
    public = dict(schema='teacher-fresh-panel-public-v1', registration_commit=commit,
        scope='consumed_development', games=16, official_replays=16, old_student_games=32,
        pairs_per_comparison=16, seed_clusters=4, overall=summarize(full),
        by_opponent={opponent:summarize([x for x in full if x['opponent']==opponent])
                     for opponent in OPPONENTS}, training=False, gpu=False,
        independent_evaluation=False, statistical_significance='not_assessed', promotion=False)
    return private, public
