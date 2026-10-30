"""Coordinate two authenticated toy games without running the 24-game panel.

The caller authenticates the smoke registration, toy-only model bytes, source
imports, official engine, private manifest and child command. This module does
not establish those identities or provide an executable launcher. Its result
is a local technical smoke, never a competitive evaluation or promotion.
"""
import math
from pathlib import Path
import time

from learning.task_pair_dev_games_v1 import capture, worker
from learning.task_pair_dev_campaign_v1 import campaign, driver as driver_module
from learning.task_pair_dev_campaign_v1.acceptance import (MAX_RECEIPT_BYTES, _json,
                                                          accept, supervision_ok)
from learning.task_pair_dev_campaign_v1.manifest import PrivateManifest


_EXTRA = frozenset(('model_reserve_bytes', 'max_games', 'max_replays',
                    'max_comparisons', 'max_attempts'))
_BYTE_JOB = frozenset(('max_rss_bytes', 'max_commit_bytes', 'min_available_bytes'))
_SMOKE_PREFIX = 'task-pair-toy-smoke-v1-'


def _require(value, message):
    if not value:
        raise ValueError(message)


def limits_snapshot(value):
    """Validate the separate, explicitly registered two-game resource limits."""
    _require(type(value) is dict and set(value) == campaign.LIMIT_KEYS | _EXTRA
             and type(value.get('job')) is dict and set(value['job']) == campaign.JOB_KEYS,
             'Invalid toy smoke limits')
    job = value['job']
    for key in (campaign.LIMIT_KEYS | _EXTRA) - {'job'}:
        _require(type(value[key]) is int and value[key] > 0, 'Invalid toy smoke limit')
    for key, amount in job.items():
        _require(type(amount) in (int, float) and math.isfinite(amount) and amount > 0
                 and (key not in _BYTE_JOB or type(amount) is int), 'Invalid Job limit')
    _require(value['max_games'] == value['max_replays'] == 2
             and value['max_comparisons'] == value['max_attempts'] == 1,
             'Toy smoke scope differs')
    _require(value['campaign_wall_seconds'] <= 86400 and job['wall_seconds'] <= 3600
             and job['cleanup_seconds'] <= 60 and job['poll_seconds'] <= 5
             and value['initial_available_bytes'] >= value['between_available_bytes']
             and value['between_available_bytes'] >= job['min_available_bytes']
             and value['per_game_reserve_bytes'] >= 138 * 1024**2
             and value['model_reserve_bytes'] <= value['per_game_reserve_bytes']
             and value['total_output_bytes'] >= (2 * value['per_game_reserve_bytes']
                                                  + value['model_reserve_bytes']
                                                  + driver_module.MAX_SUMMARY)
             and value['replay_decoded_bytes'] == campaign.REPLAY_DECODED_BYTES
             and value['trace_compressed_bytes'] == capture.MAX_COMPRESSED_BYTES
             and value['trace_line_bytes'] == capture.MAX_LINE_BYTES
             and value['trace_decoded_bytes'] == capture.MAX_DECODED_BYTES
             and value['metrics_bytes'] == capture.MAX_METRICS_BYTES,
             'Inconsistent toy smoke limits')
    return {**{key: value[key] for key in value if key != 'job'}, 'job': dict(job)}


def _plans(manifest, plans, local_driver):
    _require(type(manifest) is PrivateManifest and getattr(local_driver, 'manifest', None) is manifest
             and type(plans) is list and len(plans) == 2, 'Invalid toy smoke context')
    expected = [p for role in ('baseline', 'control') for p in manifest.panel.plans()
                if (p['seed_alias'], p['opponent'], p['side'], p['policy']) == ('S02', 'market', 0, role)]
    _require(len(expected) == 2 and plans == expected, 'Toy smoke plans differ')
    for plan in plans:
        manifest.panel.validate_plan(plan)
    _require(plans[0]['seed'] == plans[1]['seed'], 'Toy smoke seed roles differ')
    root, stage = Path(local_driver.root), Path(local_driver.stage)
    _require(root.is_absolute() and stage.is_absolute() and stage.parent == root / 'artifacts'
             and stage.name.startswith(_SMOKE_PREFIX) and len(stage.name) > len(_SMOKE_PREFIX),
             'Toy smoke stage differs')
    for role, (relative, _) in zip(('baseline', 'control', 'auxiliary', 'crop'),
                                   (*manifest.panel.models, manifest.panel.crop)):
        _require(root / relative == stage / 'models' / (role + '.npz'),
                 'Toy model path differs from its smoke stage')
    return [dict(plan) for plan in plans]


def _stage_size(local_driver):
    size = driver_module._owned_size(local_driver.stage)
    _require(type(size) is int and size >= 0, 'Invalid toy output size')
    return size


def _models(manifest, local_driver):
    for relative, expected in (*manifest.panel.models, manifest.panel.crop):
        _require(driver_module._hash(local_driver.root, local_driver.root / relative,
                                    driver_module.MAX_MODEL) == expected,
                 'Toy model bytes differ')


def _comparison(value, plans):
    _require(type(value) is dict and value.get('reference') == 'baseline'
             and value.get('candidate') == 'control' and value.get('seed_alias') == 'S02'
             and value.get('opponent') == 'market' and type(value.get('side')) is int
             and value['side'] == 0 and type(value.get('seed')) is int
             and value['seed'] == plans[0]['seed'], 'Toy comparison identity differs')
    result = value.get('result')
    _require(type(result) is dict and result.get('comparison_complete') is True
             and type(result.get('matched_calls')) is int and result['matched_calls'] >= 0
             and type(result.get('matched_multi_unit_opportunities')) is int
             and 0 <= result['matched_multi_unit_opportunities'] <= result['matched_calls'],
             'Toy comparison did not complete')


def run(*, manifest, plans, limits, driver, check, prepare_inputs, save_result,
        clock=time.monotonic):
    """Run two serial toy jobs and one existing-replay comparison.

    ``prepare_inputs(manifest, driver, model_reserve_bytes)`` must create only
    preregistered toy inputs inside the newly claimed stage, use its own bounded
    exclusive writes, and return True. ``save_result(private_payload)`` must
    persist the private result within the registered output budget and return
    True. Neither callback is trusted to prove source or toy provenance: the
    authenticated outer launcher owns that contract. The private payload must
    not be printed or published as-is. The return value has no private fields.

    Limits are checked between operations; the callbacks and the parent-side
    comparison do not receive an OS-enforced memory or wall-clock quota here.
    ``LocalDriver.claim`` writes its generic max_games=24 receipt; that denotes
    the underlying driver capacity, not the two-game scope enforced here.
    """
    game, attempted, accepted, claimed = None, 0, [], False
    try:
        worker._single_python_thread()
        with worker._quiet_output():
            fixed = limits_snapshot(limits)
            selected = _plans(manifest, plans, driver)
            _require(callable(prepare_inputs) and callable(save_result), 'Missing smoke callbacks')
            budget = campaign._Clock(fixed['campaign_wall_seconds'], clock)
            worker._registered(check)
            _require(driver.preflight(None, fixed, budget.remaining()) is True,
                     'Initial toy resource preflight failed')
            _require(driver.claim() is True, 'Toy stage claim failed')
            claimed = True
            worker._registered(check)
            before = _stage_size(driver)
            _require(before + fixed['model_reserve_bytes'] + 2 * fixed['per_game_reserve_bytes']
                     + driver_module.MAX_SUMMARY <= fixed['total_output_bytes'],
                     'Insufficient toy output reserve')
            _require(driver.preflight(selected[0], fixed, budget.remaining()) is True,
                     'Toy input resource preflight failed')
            worker._registered(check)
            _require(prepare_inputs(manifest, driver, fixed['model_reserve_bytes']) is True,
                     'Toy input preparation failed')
            worker._registered(check)
            after = _stage_size(driver)
            _require(before <= after <= before + fixed['model_reserve_bytes'],
                     'Toy input output limit exceeded')
            _models(manifest, driver)
            budget.remaining()
            for plan in selected:
                game = plan['game']
                worker._registered(check)
                remaining = budget.remaining()
                _require(remaining > fixed['job']['wall_seconds'] + fixed['job']['cleanup_seconds']
                         and driver.preflight(plan, fixed, remaining) is True,
                         'Toy game resource preflight failed')
                worker._registered(check)
                _require(budget.remaining() > fixed['job']['wall_seconds'] + fixed['job']['cleanup_seconds'],
                         'Insufficient toy game time')
                attempted += 1
                _require(attempted <= fixed['max_games'], 'Toy game budget exceeded')
                supervision = driver.launch(plan, dict(fixed['job']))
                _require(driver.record_supervision(plan, supervision) is True,
                         'Toy supervision receipt failed')
                _require(supervision_ok(supervision), 'Toy worker supervision failed')
                bundle = driver.read_bundle(plan)
                row = accept(manifest.panel, plan, bundle, supervision, driver.verify_files)
                # The original acceptance checks task forwards but permits a
                # crop head that was never called; this smoke requires both.
                counts = _json(bundle['audit'], MAX_RECEIPT_BYTES).get('primary_model_counts')
                _require(type(counts) is dict
                         and all(type(counts.get(key)) is int and counts[key] > 0
                                 for key in ('task_forwards', 'crop_forwards')),
                         'Both toy heads must execute')
                _require(driver.record_acceptance(plan, bundle) is True,
                         'Toy acceptance receipt failed')
                accepted.append(row)
                _require(len(accepted) <= attempted and _stage_size(driver) <= fixed['total_output_bytes'],
                         'Toy game output or progress differs')
                worker._registered(check)
                budget.remaining()
            _require(attempted == len(accepted) == fixed['max_games'], 'Toy games incomplete')
            _require(driver.preflight_comparison(fixed, budget.remaining()) is True,
                     'Toy comparison resource preflight failed')
            worker._registered(check)
            comparison = driver.compare(accepted[0], accepted[1])
            _comparison(comparison, selected)
            _require(_stage_size(driver) <= fixed['total_output_bytes'], 'Toy comparison output exceeded')
            worker._registered(check)
            budget.remaining()
            payload = dict(schema='task-pair-toy-smoke-private-result-v1', scope='synthetic_toy',
                           attempted=attempted, games=accepted, comparison=comparison,
                           official_replay_passes=2, comparisons=1, promotion=False,
                           generic_driver_claim_capacity=24, registered_game_limit=2)
            _require(save_result(payload) is True, 'Toy result save failed')
            _require(_stage_size(driver) <= fixed['total_output_bytes'], 'Toy output limit exceeded')
            worker._registered(check)
            budget.remaining()
            return dict(passed=True, scope='synthetic_toy', games=2, official_replay_passes=2,
                        comparisons=1, registered_game_limit=2, generic_driver_claim_capacity=24,
                        independent_improvement=False, promotion=False)
    except BaseException:
        if claimed:
            try:
                with worker._quiet_output():
                    driver.failure(game, len(accepted))
            except BaseException:
                pass
        raise RuntimeError('Toy smoke stopped') from None
