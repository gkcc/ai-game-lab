"""Serial 24-game development coordinator for an authenticated bootstrap.

The bootstrap owns registration, history and source authentication.  The
driver performs explicit local IO, process supervision and pair loading;
this module validates their results and never selects fresh seeds.
"""
import math
import time

from learning.task_pair_dev_games_v1.panel import ALIASES, OPPONENTS, POLICIES
from learning.task_pair_dev_games_v1 import capture, worker
from learning.task_pair_dev_metrics_v1.panel import paired
from .acceptance import accept, supervision_ok
from .manifest import PrivateManifest


JOB_KEYS = frozenset(('max_rss_bytes', 'max_commit_bytes', 'min_available_bytes',
                      'wall_seconds', 'poll_seconds', 'cleanup_seconds'))
LIMIT_KEYS = frozenset(('campaign_wall_seconds', 'initial_available_bytes',
                        'between_available_bytes', 'disk_free_floor_bytes',
                        'total_output_bytes', 'per_game_reserve_bytes',
                        'replay_decoded_bytes', 'trace_compressed_bytes',
                        'trace_line_bytes', 'trace_decoded_bytes', 'metrics_bytes', 'job'))
REPLAY_DECODED_BYTES = 128 * 1024 * 1024


def limits_snapshot(value):
    """Validate explicit pre-registered limits; no clock or default budget."""
    if type(value) is not dict or set(value) != LIMIT_KEYS or type(value['job']) is not dict:
        raise ValueError('Incomplete campaign limits')
    job = value['job']
    if set(job) != JOB_KEYS:
        raise ValueError('Incomplete Job limits')
    for key in LIMIT_KEYS - {'job'}:
        if type(value[key]) is not int or value[key] <= 0:
            raise ValueError('Invalid campaign limit')
    for key, v in job.items():
        if type(v) not in (int, float) or not math.isfinite(v) or v <= 0:
            raise ValueError('Invalid Job limit')
    if (value['campaign_wall_seconds'] > 86400 or job['wall_seconds'] > 3600
            or job['cleanup_seconds'] > 60 or job['poll_seconds'] > 5
            or value['initial_available_bytes'] < value['between_available_bytes']
            or value['between_available_bytes'] < job['min_available_bytes']
            or value['per_game_reserve_bytes'] < 138 * 1024**2
            or value['total_output_bytes'] < 24 * value['per_game_reserve_bytes'] + 17 * 1024**2
            or value['replay_decoded_bytes'] != REPLAY_DECODED_BYTES
            or value['trace_compressed_bytes'] != capture.MAX_COMPRESSED_BYTES
            or value['trace_line_bytes'] != capture.MAX_LINE_BYTES
            or value['trace_decoded_bytes'] != capture.MAX_DECODED_BYTES
            or value['metrics_bytes'] != capture.MAX_METRICS_BYTES):
        raise ValueError('Inconsistent campaign limits')
    return {**{key: value[key] for key in LIMIT_KEYS - {'job'}}, 'job': dict(job)}


class _Clock:
    def __init__(self, seconds, clock):
        self.clock = clock
        self.start = self.last = self._read()
        self.deadline = self.start + seconds

    def _read(self):
        value = self.clock()
        if type(value) not in (int, float) or not math.isfinite(value):
            raise ValueError('Unreliable campaign clock')
        return value

    def remaining(self):
        current = self._read()
        if current < self.last or current >= self.deadline:
            raise TimeoutError('Campaign deadline')
        self.last = current
        return self.deadline - current


def quiet_api_type(base, create_with_nul):
    """Adapt the existing NUL creator's three handles to the Job sampler API."""
    class QuietAPI(base):
        def create_suspended(self, command, cwd):
            result = create_with_nul(self, command, cwd)
            if type(result) is not tuple or len(result) != 3:
                raise RuntimeError('Unexpected owned process handles')
            process, thread, _pid = result
            return process, thread
    return QuietAPI


def supervise_windows(command, cwd, limits):
    """Use only the existing owned-Job sampler and explicit NUL handles."""
    from learning.task_training_job_supervision_v1.job import WindowsAPI
    from learning.task_job_empty_exit_v1.sampling import supervise
    from learning.panel_eval_v3.launch import WindowsJob
    api = quiet_api_type(WindowsAPI, WindowsJob.create_suspended)()
    return supervise(api, command, cwd, dict(limits))


def run(*, manifest, limits, driver, check, clock=time.monotonic):
    """Run all frozen coordinates serially after external registration.

    The driver must implement claim, preflight, preflight_comparison, launch, record_supervision,
    read_bundle, verify_files, record_acceptance, compare, complete and failure.
    It owns its local paths and
    the child command; neither it nor this function authenticates provenance
    merely by returning True.  All returned data and driver artifacts are
    private.  No repeat, replacement seed or automatic promotion is allowed.
    """
    game = None
    accepted = []
    claimed = False
    try:
        worker._single_python_thread()
        with worker._quiet_output():
            if type(manifest) is not PrivateManifest:
                raise ValueError('Expected authenticated manifest')
            fixed = limits_snapshot(limits)
            budget = _Clock(fixed['campaign_wall_seconds'], clock)
            worker._registered(check)
            plans = manifest.panel.plans()
            if len(plans) != 24 or len({p['game'] for p in plans}) != 24:
                raise ValueError('Development panel differs')
            if driver.preflight(None, fixed, budget.remaining()) is not True:
                raise RuntimeError('Initial resource preflight failed')
            if driver.claim() is not True:
                raise RuntimeError('Campaign start was not claimed')
            claimed = True
            for plan in plans:
                game = plan['game']
                worker._registered(check)
                remaining = budget.remaining()
                if remaining <= fixed['job']['wall_seconds'] + fixed['job']['cleanup_seconds']:
                    raise TimeoutError('Insufficient remaining campaign budget')
                if driver.preflight(plan, fixed, remaining) is not True:
                    raise RuntimeError('Game resource preflight failed')
                worker._registered(check)
                if budget.remaining() <= fixed['job']['wall_seconds'] + fixed['job']['cleanup_seconds']:
                    raise TimeoutError('Insufficient remaining campaign budget')
                supervision = driver.launch(plan, dict(fixed['job']))
                if driver.record_supervision(plan, supervision) is not True:
                    raise RuntimeError('Worker supervision receipt failed')
                if not supervision_ok(supervision):
                    raise RuntimeError('Worker supervision did not pass')
                bundle = driver.read_bundle(plan)
                row = accept(manifest.panel, plan, bundle, supervision, driver.verify_files)
                if driver.record_acceptance(plan, bundle) is not True:
                    raise RuntimeError('Game acceptance receipt failed')
                accepted.append(row)
                worker._registered(check)
                budget.remaining()
            if len(accepted) != 24:
                raise ValueError('Development panel incomplete')
            indexed = {(row['seed_alias'], row['opponent'], row['side'], row['arm']): row
                       for row in accepted}
            if len(indexed) != 24:
                raise ValueError('Duplicate accepted game')
            summaries = {}
            for reference, candidate in (('baseline', 'control'), ('control', 'auxiliary')):
                comparisons = []
                for alias in ALIASES:
                    for opponent in OPPONENTS:
                        for side in (0, 1):
                            worker._registered(check)
                            if driver.preflight_comparison(fixed, budget.remaining()) is not True:
                                raise RuntimeError('Comparison resource preflight failed')
                            left = indexed[(alias, opponent, side, reference)]
                            right = indexed[(alias, opponent, side, candidate)]
                            comparisons.append(driver.compare(left, right))
                selected = [row for row in accepted if row['arm'] in (reference, candidate)]
                summaries[reference + '_to_' + candidate] = paired(
                    selected, seeds=manifest.panel.seeds, opponents=OPPONENTS,
                    reference=reference, candidate=candidate, comparisons=comparisons)
            worker._registered(check)
            budget.remaining()
            if driver.complete(accepted, summaries) is not True:
                raise RuntimeError('Private campaign output failed')
            worker._registered(check)
            budget.remaining()
            return {'passed': True, 'games': 24, 'official_replay_passes': 24,
                    'seed_clusters': 2, 'independent_improvement': False, 'promotion': False}
    except BaseException:
        if claimed:
            try:
                with worker._quiet_output():
                    driver.failure(game, len(accepted))
            except BaseException:
                pass
        raise RuntimeError('Development campaign stopped') from None
