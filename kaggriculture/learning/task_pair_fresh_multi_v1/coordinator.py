"""Single-cohort orchestration over authenticated, private match components.

The outer entry owns registration, fresh-seed provenance, durable attempt and
per-game authorization markers, source identity, stage naming, and the shared
deadline across cohorts.  This module does not authenticate those claims or
publish results.  Its return value contains private rows and comparisons.
"""
import math
from pathlib import Path

from learning.task_pair_dev_campaign_v1 import campaign, driver as driver_module
from learning.task_pair_dev_campaign_v1.acceptance import (
    MAX_RECEIPT_BYTES, _json, accept, supervision_ok,
)
from learning.task_pair_dev_campaign_v1.manifest import PrivateManifest
from learning.task_pair_dev_games_v1 import capture, worker
from learning.task_pair_dev_games_v1.panel import ALIASES, OPPONENTS
from learning.task_control_once_v1.runner import _quiet_control


_ARMS = ('baseline', 'control')
_EXTRA = frozenset(('max_games', 'max_replays', 'max_comparisons', 'max_attempts'))
_BYTE_JOB = frozenset(('max_rss_bytes', 'max_commit_bytes', 'min_available_bytes'))
_GAMES = 16
_COMPARISONS = 8


def _require(value, message):
    if not value:
        raise ValueError(message)


def limits_snapshot(value):
    """Copy and validate the exact preregistered limits for one cohort."""
    _require(type(value) is dict and set(value) == campaign.LIMIT_KEYS | _EXTRA
             and type(value.get('job')) is dict and set(value['job']) == campaign.JOB_KEYS,
             'Invalid cohort limits')
    for key in (campaign.LIMIT_KEYS | _EXTRA) - {'job'}:
        _require(type(value[key]) is int and value[key] > 0, 'Invalid cohort limit')
    job = value['job']
    for key, amount in job.items():
        _require(type(amount) in (int, float) and math.isfinite(amount) and amount > 0
                 and (key not in _BYTE_JOB or type(amount) is int), 'Invalid Job limit')
    _require(value['max_games'] == value['max_replays'] == _GAMES
             and value['max_comparisons'] == _COMPARISONS and value['max_attempts'] == 1,
             'Cohort scope differs')
    _require(value['campaign_wall_seconds'] <= 86400 and job['wall_seconds'] <= 3600
             and job['cleanup_seconds'] <= 60 and job['poll_seconds'] <= 5
             and value['initial_available_bytes'] >= value['between_available_bytes']
             and value['between_available_bytes'] >= job['min_available_bytes']
             and value['per_game_reserve_bytes'] >= 138 * 1024**2
             and value['total_output_bytes'] >= (_GAMES * value['per_game_reserve_bytes']
                                                  + _COMPARISONS * MAX_RECEIPT_BYTES
                                                  + driver_module.MAX_SUMMARY)
             and value['replay_decoded_bytes'] == campaign.REPLAY_DECODED_BYTES
             and value['trace_compressed_bytes'] == capture.MAX_COMPRESSED_BYTES
             and value['trace_line_bytes'] == capture.MAX_LINE_BYTES
             and value['trace_decoded_bytes'] == capture.MAX_DECODED_BYTES
             and value['metrics_bytes'] == capture.MAX_METRICS_BYTES,
             'Inconsistent cohort limits')
    return {**{key: value[key] for key in value if key != 'job'}, 'job': dict(job)}


def _plans(manifest, local_driver, check):
    _require(type(manifest) is PrivateManifest and getattr(local_driver, 'manifest', None) is manifest
             and getattr(local_driver, 'check', None) is check and callable(check)
             and getattr(local_driver, 'claimed', None) is False, 'Invalid cohort context')
    root, stage = Path(local_driver.root), Path(local_driver.stage)
    _require(root.is_absolute() and stage.is_absolute() and stage.parent == root / 'artifacts'
             and stage.name not in ('', '.', '..'), 'Invalid cohort stage')
    panel = manifest.panel
    available = panel.plans()
    _require(type(available) is tuple and len(available) == 24
             and len({p['game'] for p in available}) == 24, 'Panel differs')
    selected = [p for p in available if p['policy'] in _ARMS]
    coordinates = {(a, o, side, arm) for a in ALIASES for o in OPPONENTS
                   for side in (0, 1) for arm in _ARMS}
    _require(len(selected) == _GAMES
             and {(p['seed_alias'], p['opponent'], p['side'], p['policy']) for p in selected}
             == coordinates, 'Cohort coordinates differ')
    return [panel.validate_plan(plan) for plan in selected]


def _stage_size(local_driver):
    size = driver_module._owned_size(local_driver.stage)
    _require(type(size) is int and size >= 0, 'Invalid stage size')
    return size


def _verify_assets(manifest, plan, local_driver):
    """Check selected model, crop model, opponent and Worker bytes."""
    root = Path(local_driver.root)
    task, task_sha = manifest.panel.task_model(plan)
    crop, crop_sha = manifest.panel.crop
    rival, rival_sha = manifest.opponent(plan['opponent'])
    for relative, expected, maximum in (
        (task, task_sha, driver_module.MAX_MODEL),
        (crop, crop_sha, driver_module.MAX_MODEL),
        (rival, rival_sha, driver_module.MAX_SOURCE),
        (worker.WORKER_RELATIVE, manifest.panel.worker_sha, worker.MAX_WORKER_BYTES),
    ):
        _require(driver_module._hash(root, root / relative, maximum) == expected,
                 'Cohort asset identity differs')


def _comparison(value, reference, candidate):
    _require(type(value) is dict and value.get('reference') == reference['arm'] == 'baseline'
             and value.get('candidate') == candidate['arm'] == 'control'
             and value.get('seed_alias') == reference['seed_alias'] == candidate['seed_alias']
             and type(value.get('seed')) is int and value['seed'] == reference['seed'] == candidate['seed']
             and value.get('opponent') == reference['opponent'] == candidate['opponent']
             and value['opponent'] in OPPONENTS and type(value.get('side')) is int
             and value['side'] == reference['side'] == candidate['side'],
             'Comparison identity differs')
    result = value.get('result')
    _require(type(result) is dict and result.get('comparison_complete') is True
             and type(result.get('matched_calls')) is int and result['matched_calls'] >= 0
             and type(result.get('matched_multi_unit_opportunities')) is int
             and 0 <= result['matched_multi_unit_opportunities'] <= result['matched_calls'],
             'Comparison did not complete')


class Coordinator:
    """Single-use guard; durable cross-process no-retry belongs to the caller."""

    def __init__(self):
        self._used = False

    def run(self, *, manifest, limits, driver, check, authorize_game, claim, remaining):
        """Run 16 games and eight comparisons; return only private evidence.

        ``remaining()`` reads one caller-owned total deadline and must remain
        finite, positive, and non-increasing.  Reusing that same callback across
        cohorts preserves the shared deadline; this class does not create or
        reset a clock. ``claim()`` must call the owned driver's claim and bind
        the caller's additional markers. ``authorize_game(plan)`` must create
        the caller's exclusive authorization for that exact plan. Both must
        return exactly True.  The caller owns all source/seed authentication,
        alias mapping, marker durability, and safe private result storage.

        Resource and deadline checks are sampled between operations; parent
        comparison and callbacks have no extra operating-system time quota.
        The generic LocalDriver start receipt says 24 as a capacity label; this
        coordinator permits only the 16 selected games and eight comparisons.
        """
        if self._used:
            raise RuntimeError('Cohort attempt already used')
        self._used = True
        accepted, attempted, current, claimed, claim_attempted = [], 0, None, False, False
        try:
            worker._single_python_thread()
            with _quiet_control():
                fixed = limits_snapshot(limits)
                selected = _plans(manifest, driver, check)
                _require(all(callable(x) for x in (authorize_game, claim, remaining)),
                         'Missing cohort callback')
                last = [float('inf')]

                def time_left():
                    value = remaining()
                    _require(type(value) in (int, float) and math.isfinite(value)
                             and 0 < value <= fixed['campaign_wall_seconds'] and value <= last[0],
                             'Shared deadline differs')
                    last[0] = value
                    return value

                required = fixed['job']['wall_seconds'] + fixed['job']['cleanup_seconds']
                worker._registered(check)
                for plan in selected[:4]:
                    _verify_assets(manifest, plan, driver)
                left = time_left()
                _require(left > required and driver.preflight(None, fixed, left) is True,
                         'Initial cohort resource preflight failed')
                claim_attempted = True
                _require(claim() is True and driver.claimed is True, 'Cohort claim failed')
                claimed = True
                worker._registered(check)
                _require(_stage_size(driver) + _GAMES * fixed['per_game_reserve_bytes']
                         + _COMPARISONS * MAX_RECEIPT_BYTES + driver_module.MAX_SUMMARY
                         <= fixed['total_output_bytes'], 'Insufficient cohort output reserve')
                for plan in selected:
                    current = plan['game']
                    worker._registered(check)
                    _verify_assets(manifest, plan, driver)
                    left = time_left()
                    _require(left > required and driver.preflight(plan, fixed, left) is True,
                             'Game resource preflight failed')
                    _require(_stage_size(driver) + (_GAMES - attempted) * fixed['per_game_reserve_bytes']
                             + _COMPARISONS * MAX_RECEIPT_BYTES + driver_module.MAX_SUMMARY
                             <= fixed['total_output_bytes'], 'Remaining output reserve differs')
                    worker._registered(check)
                    _require(time_left() > required and authorize_game(dict(plan)) is True,
                             'Game authorization failed')
                    worker._registered(check)
                    _require(time_left() > required, 'Insufficient authorized game time')
                    attempted += 1
                    _require(attempted <= _GAMES, 'Game budget exceeded')
                    supervision = driver.launch(plan, dict(fixed['job']))
                    _require(driver.record_supervision(plan, supervision) is True
                             and supervision_ok(supervision), 'Worker supervision failed')
                    bundle = driver.read_bundle(plan)
                    row = accept(manifest.panel, plan, bundle, supervision, driver.verify_files)
                    counts = _json(bundle['audit'], MAX_RECEIPT_BYTES).get('primary_model_counts')
                    _require(type(counts) is dict and all(type(counts.get(k)) is int and counts[k] > 0
                             for k in ('task_forwards', 'crop_forwards')), 'Both heads must execute')
                    _require(driver.record_acceptance(plan, bundle) is True, 'Acceptance receipt failed')
                    accepted.append(row)
                    _require(len(accepted) <= attempted and _stage_size(driver) <= fixed['total_output_bytes'],
                             'Game output or progress differs')
                    worker._registered(check)
                    _verify_assets(manifest, plan, driver)
                    time_left()
                _require(attempted == len(accepted) == _GAMES, 'Games incomplete')
                indexed = {(r['seed_alias'], r['opponent'], r['side'], r['arm']): r for r in accepted}
                _require(len(indexed) == _GAMES, 'Duplicate accepted game')
                comparisons = []
                for alias in ALIASES:
                    for opponent in OPPONENTS:
                        for side in (0, 1):
                            worker._registered(check)
                            _require(driver.preflight_comparison(fixed, time_left()) is True,
                                     'Comparison resource preflight failed')
                            reference = indexed[(alias, opponent, side, 'baseline')]
                            candidate = indexed[(alias, opponent, side, 'control')]
                            before = _stage_size(driver)
                            _require(before + (_COMPARISONS - len(comparisons)) * MAX_RECEIPT_BYTES
                                     + driver_module.MAX_SUMMARY <= fixed['total_output_bytes'],
                                     'Comparison output reserve differs')
                            value = driver.compare(reference, candidate)
                            _comparison(value, reference, candidate)
                            comparisons.append(value)
                            after = _stage_size(driver)
                            _require(before <= after <= before + MAX_RECEIPT_BYTES
                                     and after <= fixed['total_output_bytes'], 'Comparison output exceeded')
                            worker._registered(check)
                            time_left()
                _require(len(comparisons) == _COMPARISONS, 'Comparisons incomplete')
                worker._registered(check)
                for plan in selected[:4]:
                    _verify_assets(manifest, plan, driver)
                time_left()
                return dict(rows=accepted, comparisons=comparisons)
        except BaseException as failure:
            if claimed or (claim_attempted and getattr(driver, 'claimed', False) is True):
                try:
                    with _quiet_control():
                        driver.failure(current, len(accepted))
                except BaseException as receipt_error:
                    if isinstance(failure, Exception) and not isinstance(receipt_error, Exception):
                        raise
            if not isinstance(failure, Exception):
                raise
            raise RuntimeError('Cohort stopped') from None
