"""Eight authenticated consumed-seed games against the frozen round2 bot.

This is a coordinator, not an entry point or identity authority. Its caller
must authenticate the registration, runtime, fixed rules, model/actor source,
opponent and the actual S02 -> S01 / S03 -> S04 role mapping, and reserve the
attempt and game authorizations. The old market-stage attempt is never resumed.
Nothing here produces a holdout or model promotion claim.
"""
from pathlib import Path
import time

from learning.task_control_once_v1.runner import _quiet_control
from learning.task_pair_dev_campaign_v1 import campaign, driver as driver_module
from learning.task_pair_dev_campaign_v1.acceptance import MAX_RECEIPT_BYTES, _json, accept, supervision_ok
from learning.task_pair_dev_campaign_v1.manifest import PrivateManifest
from learning.task_pair_dev_games_v1 import worker
from learning.task_pair_dev_metrics_v1.panel import paired
from learning.task_pair_historical_compare_v1.runner import limits_snapshot, _verify_assets, _stage_size


_ALIASES = ('S02', 'S03')
_ARMS = ('baseline', 'control')
_OPPONENT = 'round2'
_STAGE_PREFIX = 'task-pair-round2-compare-v1-'
FAILURE_PHASES = frozenset(('setup', 'initial_preflight', 'claim', 'game_preflight',
                            'authorization', 'launch', 'supervision', 'acceptance',
                            'post_game', 'comparison', 'statistics', 'save_result', 'final_check'))
FAILURE_CATEGORIES = frozenset(('TimeoutError', 'MemoryError', 'ValueError', 'OSError',
                                'RuntimeError', 'Exception', 'BaseException'))


def _require(value, message):
    if not value:
        raise ValueError(message)


def _plans(manifest, plans, driver, check):
    _require(type(manifest) is PrivateManifest and getattr(driver, 'manifest', None) is manifest
             and getattr(driver, 'check', None) is check and callable(check)
             and getattr(driver, 'claimed', None) is False
             and type(plans) is list and len(plans) == 8, 'Invalid round2 context')
    expected = [p for alias in _ALIASES for side in (0, 1) for arm in _ARMS
                for p in manifest.panel.plans()
                if (p['seed_alias'], p['opponent'], p['side'], p['policy'])
                == (alias, _OPPONENT, side, arm)]
    _require(len(expected) == 8 and plans == expected, 'Round2 plans differ')
    selected = [manifest.panel.validate_plan(plan) for plan in plans]
    root, stage = Path(driver.root), Path(driver.stage)
    _require(root.is_absolute() and stage.is_absolute() and stage.parent == root / 'artifacts'
             and stage.name.startswith(_STAGE_PREFIX) and len(stage.name) > len(_STAGE_PREFIX),
             'Round2 stage differs')
    return selected


def _comparison(value, reference, candidate):
    _require(type(value) is dict and value.get('reference') == 'baseline'
             and value.get('candidate') == 'control'
             and value.get('seed_alias') == reference['seed_alias'] == candidate['seed_alias']
             and type(value.get('seed')) is int and value['seed'] == reference['seed'] == candidate['seed']
             and value.get('opponent') == reference['opponent'] == candidate['opponent'] == _OPPONENT
             and type(value.get('side')) is int and value['side'] == reference['side'] == candidate['side'],
             'Round2 comparison identity differs')
    result = value.get('result')
    _require(type(result) is dict and result.get('comparison_complete') is True
             and type(result.get('matched_calls')) is int and result['matched_calls'] >= 0
             and type(result.get('matched_multi_unit_opportunities')) is int
             and 0 <= result['matched_multi_unit_opportunities'] <= result['matched_calls'],
             'Round2 comparison did not complete')


def _error_kind(error):
    """Expose only a fixed coarse exception category, never exception text."""
    for kind in (TimeoutError, MemoryError, ValueError, OSError, RuntimeError):
        if isinstance(error, kind):
            return kind.__name__
    return 'Exception' if isinstance(error, Exception) else 'BaseException'


class Coordinator:
    """In-process single-use guard; durable one-shot ownership is external."""

    def __init__(self):
        self._used = False

    def run(self, *, manifest, plans, limits, driver, check, authorize_game, save_result,
            on_failure=None, clock=time.monotonic):
        """Run at most eight serial games, then compare four paired conditions.

        Caller provides an authenticated manifest and strict ordered plans. Each
        ``authorize_game(plan)`` must establish an exclusive durable permission;
        ``save_result(payload)`` exclusively persists the private ledger. A
        successful return does not prove the outer process exited successfully.
        Optional ``on_failure`` receives only fixed phase, coarse error category,
        attempted count and accepted count; it must not change the original
        error or replace the driver's failure marker. Checks and deadlines are
        sampled between calls; comparisons do not have an asynchronous limit.
        """
        if self._used:
            raise RuntimeError('Round2 comparison attempt already used')
        self._used = True
        attempted, accepted, claimed, game, phase = 0, [], False, None, 'setup'
        try:
            worker._single_python_thread()
            with _quiet_control():
                fixed = limits_snapshot(limits)
                selected = _plans(manifest, plans, driver, check)
                _require(callable(authorize_game) and callable(save_result)
                         and (on_failure is None or callable(on_failure)), 'Missing callbacks')
                budget = campaign._Clock(fixed['campaign_wall_seconds'], clock)
                worker._registered(check)
                for plan in selected[:2]:
                    _verify_assets(manifest, plan, driver)
                phase = 'initial_preflight'
                _require(driver.preflight(None, fixed, budget.remaining()) is True,
                         'Initial resource preflight failed')
                phase = 'claim'
                _require(driver.claim() is True, 'Round2 stage claim failed')
                claimed = True
                worker._registered(check)
                _require(_stage_size(driver) + 8 * fixed['per_game_reserve_bytes']
                         + 4 * MAX_RECEIPT_BYTES + driver_module.MAX_SUMMARY
                         <= fixed['total_output_bytes'], 'Insufficient output reserve')
                for plan in selected:
                    game, phase = plan['game'], 'game_preflight'
                    worker._registered(check)
                    _verify_assets(manifest, plan, driver)
                    remaining = budget.remaining()
                    _require(remaining > fixed['job']['wall_seconds'] + fixed['job']['cleanup_seconds']
                             and driver.preflight(plan, fixed, remaining) is True,
                             'Game resource preflight failed')
                    _require(_stage_size(driver) + (8 - attempted) * fixed['per_game_reserve_bytes']
                             + 4 * MAX_RECEIPT_BYTES + driver_module.MAX_SUMMARY
                             <= fixed['total_output_bytes'], 'Remaining output reserve differs')
                    worker._registered(check)
                    _require(budget.remaining() > fixed['job']['wall_seconds'] + fixed['job']['cleanup_seconds'],
                             'Insufficient game time')
                    phase = 'authorization'
                    _require(authorize_game(dict(plan)) is True, 'Game authorization failed')
                    worker._registered(check)
                    _require(budget.remaining() > fixed['job']['wall_seconds'] + fixed['job']['cleanup_seconds'],
                             'Insufficient authorized game time')
                    attempted += 1
                    _require(attempted <= fixed['max_games'], 'Game budget exceeded')
                    phase = 'launch'
                    supervision = driver.launch(plan, dict(fixed['job']))
                    phase = 'supervision'
                    _require(driver.record_supervision(plan, supervision) is True and supervision_ok(supervision),
                             'Worker supervision failed')
                    phase = 'acceptance'
                    bundle = driver.read_bundle(plan)
                    row = accept(manifest.panel, plan, bundle, supervision, driver.verify_files)
                    counts = _json(bundle['audit'], MAX_RECEIPT_BYTES).get('primary_model_counts')
                    _require(type(counts) is dict and all(type(counts.get(key)) is int and counts[key] > 0
                             for key in ('task_forwards', 'crop_forwards')), 'Both model heads must execute')
                    _require(driver.record_acceptance(plan, bundle) is True, 'Acceptance receipt failed')
                    accepted.append(row)
                    phase = 'post_game'
                    _require(len(accepted) <= attempted and _stage_size(driver) <= fixed['total_output_bytes'],
                             'Output or progress differs')
                    worker._registered(check)
                    _verify_assets(manifest, plan, driver)
                    budget.remaining()
                _require(attempted == len(accepted) == 8, 'Games incomplete')
                indexed = {(row['seed_alias'], row['side'], row['arm']): row for row in accepted}
                _require(len(indexed) == 8, 'Duplicate accepted game')
                comparisons = []
                phase = 'comparison'
                for alias in _ALIASES:
                    for side in (0, 1):
                        worker._registered(check)
                        _require(driver.preflight_comparison(fixed, budget.remaining()) is True,
                                 'Comparison resource preflight failed')
                        left, right = indexed[(alias, side, 'baseline')], indexed[(alias, side, 'control')]
                        before = _stage_size(driver)
                        value = driver.compare(left, right)
                        _comparison(value, left, right)
                        comparisons.append(value)
                        after = _stage_size(driver)
                        _require(before <= after <= before + MAX_RECEIPT_BYTES
                                 and after <= fixed['total_output_bytes'], 'Comparison output exceeded')
                        worker._registered(check)
                        budget.remaining()
                _require(len(comparisons) == 4, 'Comparisons incomplete')
                phase = 'statistics'
                statistics = paired(accepted, seeds=manifest.panel.seeds, opponents=(_OPPONENT,),
                                    reference='baseline', candidate='control', comparisons=comparisons)
                _require(statistics.get('expected_games') == statistics.get('observed_games') == 8
                         and statistics.get('seed_clusters') == 2
                         and statistics.get('statistical_significance') == 'unknown'
                         and statistics.get('independently_verified_improvement') is False
                         and statistics.get('promotion') is False, 'Paired result differs')
                worker._registered(check)
                for plan in selected[:2]:
                    _verify_assets(manifest, plan, driver)
                payload = dict(schema='task-pair-round2-compare-private-result-v1',
                               scope='consumed_development', attempted=attempted, rows=accepted,
                               comparisons=comparisons, paired=statistics, official_replay_passes=8,
                               registered_game_limit=8, generic_driver_claim_capacity=24,
                               training=False, independent_improvement=False, promotion=False)
                phase = 'save_result'
                before = _stage_size(driver)
                _require(save_result(payload) is True, 'Round2 result save failed')
                after = _stage_size(driver)
                _require(before <= after <= before + driver_module.MAX_SUMMARY
                         and after <= fixed['total_output_bytes'], 'Result output exceeded')
                phase = 'final_check'
                worker._registered(check)
                for plan in selected[:2]:
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
            if callable(on_failure):
                try:
                    on_failure(dict(phase=phase, category=_error_kind(failure),
                                    attempted=attempted, accepted=len(accepted)))
                except Exception:
                    pass
            if not isinstance(failure, Exception):
                raise
            raise RuntimeError('Round2 comparison stopped') from None
