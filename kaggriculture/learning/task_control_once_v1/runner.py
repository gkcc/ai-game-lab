"""One externally authenticated control game using the existing Worker.

The existing Panel requires S02/market/control; those labels are only adapter
coordinates. The outer registration must bind S02 to its intended S01 seed and
market to the intended teacher before this coordinator is ever executed. This
is neither the old 24-game panel nor an independent performance evaluation.

This module is not an entry point or an identity authority. In particular, its
single-use object is not a cross-process attempt marker: the outer entry must
reserve one fixed attempt before resource preflight, prevent reuse across
processes and record pre-stage failures. It must authenticate source, engine,
manifest, roles, stage and child command, and independently inspect any future
diagnostic receipt. No diagnostic result is asserted here.
"""
from contextlib import contextmanager
import math
from pathlib import Path
import time

from learning.task_pair_dev_games_v1 import capture, worker
from learning.task_pair_dev_campaign_v1 import campaign, driver as driver_module
from learning.task_pair_dev_campaign_v1.acceptance import (
    MAX_RECEIPT_BYTES, _json, accept, supervision_ok,
)
from learning.task_pair_dev_campaign_v1.manifest import PrivateManifest


_EXTRA = frozenset(('max_games', 'max_replays', 'max_comparisons', 'max_attempts'))
_BYTE_JOB = frozenset(('max_rss_bytes', 'max_commit_bytes', 'min_available_bytes'))
_STAGE_PREFIX = 'task-control-once-v1-'


def _require(value, message):
    if not value:
        raise ValueError(message)


@contextmanager
def _quiet_control():
    """Keep a cancellation if the existing output-restoration also fails."""
    cancellation = None
    try:
        with worker._quiet_output():
            try:
                yield
            except BaseException as error:
                if not isinstance(error, Exception):
                    cancellation = error
                raise
    except BaseException:
        if cancellation is not None:
            raise cancellation from None
        raise


def limits_snapshot(value):
    """Validate a separately registered one-game budget, not a 24-game one."""
    _require(type(value) is dict and set(value) == campaign.LIMIT_KEYS | _EXTRA
             and type(value.get('job')) is dict and set(value['job']) == campaign.JOB_KEYS,
             'Invalid control limits')
    for key in (campaign.LIMIT_KEYS | _EXTRA) - {'job', 'max_comparisons'}:
        _require(type(value[key]) is int and value[key] > 0, 'Invalid control limit')
    _require(type(value['max_comparisons']) is int and value['max_comparisons'] == 0
             and value['max_games'] == value['max_replays'] == value['max_attempts'] == 1,
             'Control scope differs')
    job = value['job']
    for key, amount in job.items():
        _require(type(amount) in (int, float) and math.isfinite(amount) and amount > 0
                 and (key not in _BYTE_JOB or type(amount) is int), 'Invalid Job limit')
    _require(value['campaign_wall_seconds'] <= 86400 and job['wall_seconds'] <= 3600
             and job['cleanup_seconds'] <= 60 and job['poll_seconds'] <= 5
             and value['initial_available_bytes'] >= value['between_available_bytes']
             and value['between_available_bytes'] >= job['min_available_bytes']
             and value['per_game_reserve_bytes'] >= 138 * 1024**2
             and value['total_output_bytes'] >= value['per_game_reserve_bytes'] + driver_module.MAX_SUMMARY
             and value['replay_decoded_bytes'] == campaign.REPLAY_DECODED_BYTES
             and value['trace_compressed_bytes'] == capture.MAX_COMPRESSED_BYTES
             and value['trace_line_bytes'] == capture.MAX_LINE_BYTES
             and value['trace_decoded_bytes'] == capture.MAX_DECODED_BYTES
             and value['metrics_bytes'] == capture.MAX_METRICS_BYTES,
             'Inconsistent control limits')
    return {**{key: value[key] for key in value if key != 'job'}, 'job': dict(job)}


def _plan(manifest, plan, local_driver, check):
    _require(type(manifest) is PrivateManifest and getattr(local_driver, 'manifest', None) is manifest
             and getattr(local_driver, 'check', None) is check and callable(check)
             and getattr(local_driver, 'claimed', None) is False, 'Invalid control context')
    selected = manifest.panel.validate_plan(plan)
    _require((selected['seed_alias'], selected['opponent'], selected['side'], selected['policy'])
             == ('S02', 'market', 0, 'control'), 'Control plan differs')
    root, stage = Path(local_driver.root), Path(local_driver.stage)
    _require(root.is_absolute() and stage.is_absolute() and stage.parent == root / 'artifacts'
             and stage.name.startswith(_STAGE_PREFIX) and len(stage.name) > len(_STAGE_PREFIX),
             'Control stage differs')
    return selected


def _stage_size(local_driver):
    value = driver_module._owned_size(local_driver.stage)
    _require(type(value) is int and value >= 0, 'Invalid control output size')
    return value


def _verify_assets(manifest, plan, local_driver):
    """Check the selected heads, opponent and old Worker without exposing IDs."""
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
                 'Control asset identity differs')


class Coordinator:
    """Single-use parent coordinator; durable no-retry belongs to the outer entry."""

    def __init__(self):
        self._used = False

    def run(self, *, manifest, plan, limits, driver, check, save_result, clock=time.monotonic):
        """Supervise at most one game and persist a minimal local summary.

        The outer authenticated caller must create an exclusive durable attempt
        marker before calling this method, including before failed preflight. It
        owns the fixed stage and the mapping from adapter labels to actual
        identities. ``save_result`` must write only inside that owned stage,
        with bounded exclusive creation, and return True. No private receipt,
        seed, model identity, path or game result is printed or returned.

        The clock and resource checks run between operations. Parent-side file
        checks, acceptance and ``save_result`` do not have an OS-enforced time
        or memory quota here. ``LocalDriver.claim`` records max_games=24 as a
        generic capacity label; this coordinator permits one game only. A
        returned success is the inner result, not proof of the outer process's
        exit or of any diagnostic receipt.

        After stage claim, failure triggers the existing driver failure writer
        once. Failure to write that receipt is not hidden success; the outer
        caller must verify its presence and preserve the no-retry decision.
        """
        if self._used:
            raise RuntimeError('Single control attempt already used')
        self._used = True
        claimed, attempted, accepted, game = False, 0, 0, None
        try:
            worker._single_python_thread()
            with _quiet_control():
                fixed = limits_snapshot(limits)
                selected = _plan(manifest, plan, driver, check)
                _require(callable(save_result), 'Missing result callback')
                budget = campaign._Clock(fixed['campaign_wall_seconds'], clock)
                worker._registered(check)
                _verify_assets(manifest, selected, driver)
                _require(driver.preflight(None, fixed, budget.remaining()) is True,
                         'Initial resource preflight failed')
                worker._registered(check)
                _require(driver.claim() is True, 'Control stage claim failed')
                claimed = True
                _require(_stage_size(driver) + fixed['per_game_reserve_bytes'] + driver_module.MAX_SUMMARY
                         <= fixed['total_output_bytes'], 'Insufficient control output reserve')
                game = selected['game']
                worker._registered(check)
                remaining = budget.remaining()
                _require(remaining > fixed['job']['wall_seconds'] + fixed['job']['cleanup_seconds']
                         and driver.preflight(selected, fixed, remaining) is True,
                         'Control resource preflight failed')
                worker._registered(check)
                _verify_assets(manifest, selected, driver)
                _require(budget.remaining() > fixed['job']['wall_seconds'] + fixed['job']['cleanup_seconds'],
                         'Insufficient control time')
                attempted += 1
                _require(attempted == fixed['max_games'], 'Control game budget exceeded')
                supervision = driver.launch(selected, dict(fixed['job']))
                _require(driver.record_supervision(selected, supervision) is True,
                         'Control supervision receipt failed')
                _require(supervision_ok(supervision), 'Control worker supervision failed')
                bundle = driver.read_bundle(selected)
                metrics = accept(manifest.panel, selected, bundle, supervision, driver.verify_files)
                _require(type(metrics) is dict and metrics.get('game') == game
                         and metrics.get('arm') == 'control', 'Control acceptance differs')
                counts = _json(bundle['audit'], MAX_RECEIPT_BYTES).get('primary_model_counts')
                _require(type(counts) is dict
                         and all(type(counts.get(key)) is int and counts[key] > 0
                                 for key in ('task_forwards', 'crop_forwards')),
                         'Both control heads must execute')
                _require(driver.record_acceptance(selected, bundle) is True,
                         'Control acceptance receipt failed')
                accepted = 1
                _require(_stage_size(driver) + driver_module.MAX_SUMMARY <= fixed['total_output_bytes'],
                         'Control output reserve exceeded')
                worker._registered(check)
                _verify_assets(manifest, selected, driver)
                budget.remaining()
                summary = dict(schema='task-control-once-v1-local-result', scope='one_control_game',
                               attempted=attempted, accepted=accepted, official_replay_passes=1,
                               comparisons=0, registered_game_limit=1, max_attempts=1,
                               generic_driver_claim_capacity=24, role_binding='external',
                               training=False, independent_improvement=False, promotion=False,
                               diagnostic_verified=False, observed_outer_exit=False)
                before = _stage_size(driver)
                _require(save_result(dict(summary)) is True, 'Control result save failed')
                after = _stage_size(driver)
                _require(before <= after <= before + driver_module.MAX_SUMMARY
                         and after <= fixed['total_output_bytes'], 'Control output exceeded')
                worker._registered(check)
                _verify_assets(manifest, selected, driver)
                budget.remaining()
                return summary
        except BaseException as failure:
            if claimed:
                try:
                    with _quiet_control():
                        driver.failure(game, accepted)
                except BaseException as receipt_error:
                    # A new cancellation while recording an ordinary error is
                    # itself meaningful; never mask an existing cancellation.
                    if isinstance(failure, Exception) and not isinstance(receipt_error, Exception):
                        raise
            if not isinstance(failure, Exception):
                raise
            raise RuntimeError('Single control stopped') from None
