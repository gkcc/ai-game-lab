"""Eight previously consumed development games using the existing Worker.

This is a coordinator, not an entry point or identity authority. The caller
must authenticate the registration, runtime, model files, original rules and
the external role mapping S02 -> S01, S03 -> S04 and market -> market. Those
internal labels alone do not establish any of those identities. This stage is
neither a retry of an old attempt nor an independent evaluation.
"""
import math
from pathlib import Path
import time

from learning.task_pair_dev_games_v1 import capture, worker
from learning.task_pair_dev_campaign_v1 import campaign, driver as driver_module
from learning.task_pair_dev_campaign_v1.acceptance import (
    MAX_RECEIPT_BYTES, _json, accept, supervision_ok,
)
from learning.task_pair_dev_campaign_v1.manifest import PrivateManifest
from learning.task_pair_dev_metrics_v1.panel import paired
from learning.task_control_once_v1.runner import _quiet_control


_EXTRA = frozenset(('max_games', 'max_replays', 'max_comparisons', 'max_attempts'))
_BYTE_JOB = frozenset(('max_rss_bytes', 'max_commit_bytes', 'min_available_bytes'))
_STAGE_PREFIX = 'task-pair-historical-compare-v1-'
_ALIASES = ('S02', 'S03')
_ARMS = ('baseline', 'control')


def _require(value, message):
    if not value:
        raise ValueError(message)


def limits_snapshot(value):
    """Validate this stage's explicit eight-game limits, with no defaults."""
    _require(type(value) is dict and set(value) == campaign.LIMIT_KEYS | _EXTRA
             and type(value.get('job')) is dict and set(value['job']) == campaign.JOB_KEYS,
             'Invalid historical comparison limits')
    for key in (campaign.LIMIT_KEYS | _EXTRA) - {'job'}:
        _require(type(value[key]) is int and value[key] > 0, 'Invalid comparison limit')
    job = value['job']
    for key, amount in job.items():
        _require(type(amount) in (int, float) and math.isfinite(amount) and amount > 0
                 and (key not in _BYTE_JOB or type(amount) is int), 'Invalid Job limit')
    _require(value['max_games'] == value['max_replays'] == 8
             and value['max_comparisons'] == 4 and value['max_attempts'] == 1,
             'Historical comparison scope differs')
    _require(value['campaign_wall_seconds'] <= 86400 and job['wall_seconds'] <= 3600
             and job['cleanup_seconds'] <= 60 and job['poll_seconds'] <= 5
             and value['initial_available_bytes'] >= value['between_available_bytes']
             and value['between_available_bytes'] >= job['min_available_bytes']
             and value['per_game_reserve_bytes'] >= 138 * 1024**2
             and value['total_output_bytes'] >= (8 * value['per_game_reserve_bytes']
                                                  + 4 * MAX_RECEIPT_BYTES
                                                  + driver_module.MAX_SUMMARY)
             and value['replay_decoded_bytes'] == campaign.REPLAY_DECODED_BYTES
             and value['trace_compressed_bytes'] == capture.MAX_COMPRESSED_BYTES
             and value['trace_line_bytes'] == capture.MAX_LINE_BYTES
             and value['trace_decoded_bytes'] == capture.MAX_DECODED_BYTES
             and value['metrics_bytes'] == capture.MAX_METRICS_BYTES,
             'Inconsistent comparison limits')
    return {**{key: value[key] for key in value if key != 'job'}, 'job': dict(job)}


def _plans(manifest, plans, local_driver, check):
    _require(type(manifest) is PrivateManifest and getattr(local_driver, 'manifest', None) is manifest
             and getattr(local_driver, 'check', None) is check and callable(check)
             and getattr(local_driver, 'claimed', None) is False
             and type(plans) is list and len(plans) == 8, 'Invalid comparison context')
    expected = [p for alias in _ALIASES for side in (0, 1) for arm in _ARMS
                for p in manifest.panel.plans()
                if (p['seed_alias'], p['opponent'], p['side'], p['policy'])
                == (alias, 'market', side, arm)]
    _require(len(expected) == 8 and plans == expected, 'Comparison plans differ')
    selected = [manifest.panel.validate_plan(plan) for plan in plans]
    root, stage = Path(local_driver.root), Path(local_driver.stage)
    _require(root.is_absolute() and stage.is_absolute() and stage.parent == root / 'artifacts'
             and stage.name.startswith(_STAGE_PREFIX) and len(stage.name) > len(_STAGE_PREFIX),
             'Comparison stage differs')
    return selected


def _stage_size(local_driver):
    value = driver_module._owned_size(local_driver.stage)
    _require(type(value) is int and value >= 0, 'Invalid comparison output size')
    return value


def _verify_assets(manifest, plan, local_driver):
    """Authenticate the selected heads, market opponent and Worker bytes."""
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
                 'Comparison asset identity differs')


def _comparison(value, reference, candidate):
    _require(type(value) is dict and value.get('reference') == 'baseline'
             and value.get('candidate') == 'control'
             and value.get('seed_alias') == reference['seed_alias'] == candidate['seed_alias']
             and type(value.get('seed')) is int and value['seed'] == reference['seed'] == candidate['seed']
             and value.get('opponent') == reference['opponent'] == candidate['opponent'] == 'market'
             and type(value.get('side')) is int and value['side'] == reference['side'] == candidate['side'],
             'Comparison identity differs')
    result = value.get('result')
    _require(type(result) is dict and result.get('comparison_complete') is True
             and type(result.get('matched_calls')) is int and result['matched_calls'] >= 0
             and type(result.get('matched_multi_unit_opportunities')) is int
             and 0 <= result['matched_multi_unit_opportunities'] <= result['matched_calls'],
             'Comparison did not complete')


class Coordinator:
    """An in-process single-use guard; the caller owns durable attempt markers."""

    def __init__(self):
        self._used = False

    def run(self, *, manifest, plans, limits, driver, check, authorize_game, save_result,
            clock=time.monotonic):
        """Supervise eight serial jobs and save four paired private comparisons.

        The outer caller must authenticate all public and private identities,
        reserve a new attempt before private reads, and provide per-game,
        exclusive launch authorization. ``authorize_game(plan)`` must return
        True only after that authorization is established. ``save_result``
        must exclusively persist its private payload within this stage; it must
        not print or publish it. The result callback is not an independent
        acceptance, nor does an inner return prove the outer process exited.

        Deadline and resource checks are sampled between operations. Parent
        comparison and callbacks have no additional OS-enforced quota here.
        The driver's max_games=24 start receipt is a generic capacity label;
        this coordinator permits eight games and four comparisons only.
        """
        if self._used:
            raise RuntimeError('Historical comparison attempt already used')
        self._used = True
        game, attempted, accepted, claimed = None, 0, [], False
        try:
            worker._single_python_thread()
            with _quiet_control():
                fixed = limits_snapshot(limits)
                selected = _plans(manifest, plans, driver, check)
                _require(callable(authorize_game) and callable(save_result),
                         'Missing comparison callbacks')
                budget = campaign._Clock(fixed['campaign_wall_seconds'], clock)
                worker._registered(check)
                for plan in (selected[0], selected[1]):
                    _verify_assets(manifest, plan, driver)
                _require(driver.preflight(None, fixed, budget.remaining()) is True,
                         'Initial comparison resource preflight failed')
                _require(driver.claim() is True, 'Comparison stage claim failed')
                claimed = True
                worker._registered(check)
                _require(_stage_size(driver) + 8 * fixed['per_game_reserve_bytes']
                         + 4 * MAX_RECEIPT_BYTES + driver_module.MAX_SUMMARY
                         <= fixed['total_output_bytes'], 'Insufficient comparison output reserve')
                for plan in selected:
                    game = plan['game']
                    worker._registered(check)
                    _verify_assets(manifest, plan, driver)
                    remaining = budget.remaining()
                    _require(remaining > fixed['job']['wall_seconds'] + fixed['job']['cleanup_seconds']
                             and driver.preflight(plan, fixed, remaining) is True,
                             'Game resource preflight failed')
                    _require(_stage_size(driver) + (8 - attempted) * fixed['per_game_reserve_bytes']
                             + 4 * MAX_RECEIPT_BYTES + driver_module.MAX_SUMMARY
                             <= fixed['total_output_bytes'], 'Remaining game reserve differs')
                    worker._registered(check)
                    _require(budget.remaining() > fixed['job']['wall_seconds'] + fixed['job']['cleanup_seconds'],
                             'Insufficient game time')
                    _require(authorize_game(dict(plan)) is True, 'Game authorization failed')
                    worker._registered(check)
                    _require(budget.remaining() > fixed['job']['wall_seconds'] + fixed['job']['cleanup_seconds'],
                             'Insufficient authorized game time')
                    attempted += 1
                    _require(attempted <= fixed['max_games'], 'Game budget exceeded')
                    supervision = driver.launch(plan, dict(fixed['job']))
                    _require(driver.record_supervision(plan, supervision) is True,
                             'Supervision receipt failed')
                    _require(supervision_ok(supervision), 'Worker supervision failed')
                    bundle = driver.read_bundle(plan)
                    row = accept(manifest.panel, plan, bundle, supervision, driver.verify_files)
                    counts = _json(bundle['audit'], MAX_RECEIPT_BYTES).get('primary_model_counts')
                    _require(type(counts) is dict and all(type(counts.get(key)) is int and counts[key] > 0
                             for key in ('task_forwards', 'crop_forwards')), 'Both heads must execute')
                    _require(driver.record_acceptance(plan, bundle) is True, 'Game acceptance receipt failed')
                    accepted.append(row)
                    _require(len(accepted) <= attempted and _stage_size(driver) <= fixed['total_output_bytes'],
                             'Game output or progress differs')
                    worker._registered(check)
                    _verify_assets(manifest, plan, driver)
                    budget.remaining()
                _require(attempted == len(accepted) == 8, 'Games incomplete')
                indexed = {(row['seed_alias'], row['side'], row['arm']): row for row in accepted}
                _require(len(indexed) == 8, 'Duplicate accepted game')
                comparisons = []
                for alias in _ALIASES:
                    for side in (0, 1):
                        worker._registered(check)
                        _require(driver.preflight_comparison(fixed, budget.remaining()) is True,
                                 'Comparison resource preflight failed')
                        reference = indexed[(alias, side, 'baseline')]
                        candidate = indexed[(alias, side, 'control')]
                        before = _stage_size(driver)
                        value = driver.compare(reference, candidate)
                        _comparison(value, reference, candidate)
                        comparisons.append(value)
                        after = _stage_size(driver)
                        _require(before <= after <= before + MAX_RECEIPT_BYTES
                                 and after <= fixed['total_output_bytes'], 'Comparison output exceeded')
                        worker._registered(check)
                        budget.remaining()
                _require(len(comparisons) == 4, 'Comparisons incomplete')
                statistics = paired(accepted, seeds=manifest.panel.seeds, opponents=('market',),
                                    reference='baseline', candidate='control', comparisons=comparisons)
                _require(statistics.get('expected_games') == statistics.get('observed_games') == 8
                         and statistics.get('seed_clusters') == 2
                         and statistics.get('statistical_significance') == 'unknown'
                         and statistics.get('independently_verified_improvement') is False
                         and statistics.get('promotion') is False, 'Paired result differs')
                worker._registered(check)
                for plan in (selected[0], selected[1]):
                    _verify_assets(manifest, plan, driver)
                payload = dict(schema='task-pair-historical-compare-private-result-v1',
                               scope='consumed_development', attempted=attempted, rows=accepted,
                               comparisons=comparisons, paired=statistics, official_replay_passes=8,
                               registered_game_limit=8, generic_driver_claim_capacity=24,
                               training=False, independent_improvement=False, promotion=False)
                before = _stage_size(driver)
                _require(save_result(payload) is True, 'Comparison result save failed')
                after = _stage_size(driver)
                _require(before <= after <= before + driver_module.MAX_SUMMARY
                         and after <= fixed['total_output_bytes'], 'Result output exceeded')
                worker._registered(check)
                for plan in (selected[0], selected[1]):
                    _verify_assets(manifest, plan, driver)
                budget.remaining()
                return dict(passed=True, scope='consumed_development', games=8, official_replay_passes=8,
                            comparisons=4, seed_clusters=2, registered_game_limit=8,
                            generic_driver_claim_capacity=24, training=False,
                            independent_improvement=False, promotion=False, observed_outer_exit=False)
        except BaseException as failure:
            if claimed:
                try:
                    with _quiet_control():
                        driver.failure(game, len(accepted))
                except BaseException as receipt_error:
                    if isinstance(failure, Exception) and not isinstance(receipt_error, Exception):
                        raise
            if not isinstance(failure, Exception):
                raise
            raise RuntimeError('Historical comparison stopped') from None
