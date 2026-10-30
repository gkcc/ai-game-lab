"""Recheck private development receipts before an allowlisted projection.

The caller authenticates the registration, private manifest and running state.
This checker verifies the local file/receipt chain again; it does not replay the
official engine again or prove a hostile same-user process could not modify it.
It returns public-safe data only, never the private source objects.
"""
import hashlib

from learning.task_pair_dev_games_v1 import worker
from learning.task_pair_dev_games_v1.panel import ALIASES, OPPONENTS
from . import acceptance, driver, projection
from .manifest import PrivateManifest


def _require(value, message):
    if not value:
        raise ValueError(message)


def _read(stage, path, limit):
    return driver._read(stage, path, limit)


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _receipt_index(items, key, count):
    _require(type(items) is list and len(items) == count, 'Incomplete receipt list')
    indexed = {}
    for item in items:
        _require(type(item) is dict, 'Invalid receipt entry')
        identity = key(item)
        _require(identity not in indexed, 'Duplicate receipt entry')
        indexed[identity] = item
    return indexed


def verify_and_project(*, manifest, local_driver, check):
    """Verify all 24 accepted games and 16 comparison links, then project.

    The source comparison is the one made during the registered campaign;
    checking its receipt is not an additional independent comparison pass.
    No external output is written here. The caller decides where to store the
    allowlisted return value after checking its public disclosure policy.
    """
    _require(type(manifest) is PrivateManifest and local_driver.manifest is manifest,
             'Manifest identity differs')
    worker._registered(check)
    stage = local_driver.stage
    result_raw = _read(stage, stage / 'campaign-result.local.json', driver.MAX_SUMMARY)
    result = acceptance._json(result_raw, driver.MAX_SUMMARY)
    _require(result.get('schema') == 'task-pair-dev-campaign-result-v1'
             and result.get('passed') is True and type(result.get('games')) is int
             and result['games'] == 24 and type(result.get('official_replay_passes')) is int
             and result['official_replay_passes'] == 24 and type(result.get('seed_clusters')) is int
             and result['seed_clusters'] == 2 and result.get('independent_improvement') is False
             and result.get('promotion') is False, 'Campaign completion differs')
    game_receipts = _receipt_index(result.get('receipts'), lambda item: item.get('game'), 24)
    plans = manifest.panel.plans()
    _require(set(game_receipts) == {plan['game'] for plan in plans}, 'Game receipt identities differ')
    rows = []
    for plan in plans:
        worker._registered(check)
        game = plan['game']
        receipt = game_receipts[game]
        supervision_raw = _read(stage, stage/'job-results'/(game+'.supervision.local.json'),
                                acceptance.MAX_RECEIPT_BYTES)
        accepted_raw = _read(stage, stage/'job-results'/(game+'.acceptance.local.json'),
                             acceptance.MAX_RECEIPT_BYTES)
        _require(_sha(supervision_raw) == receipt.get('supervision_sha256')
                 and _sha(accepted_raw) == receipt.get('acceptance_sha256'), 'Game receipt hash differs')
        supervision = acceptance._json(supervision_raw, acceptance.MAX_RECEIPT_BYTES)
        accepted = acceptance._json(accepted_raw, acceptance.MAX_RECEIPT_BYTES)
        _require(supervision.get('schema') == 'task-pair-dev-job-supervision-v1'
                 and supervision.get('game') == game and supervision.get('retry_allowed') is False
                 and accepted.get('schema') == 'task-pair-dev-job-acceptance-v1'
                 and accepted.get('game') == game and accepted.get('passed') is True
                 and accepted.get('retry_allowed') is False
                 and accepted.get('supervision_sha256') == _sha(supervision_raw),
                 'Accepted game receipt differs')
        input_raw = _read(stage, stage/'job-inputs'/(game+'.json'), worker.MAX_JOB_BYTES)
        _require(accepted.get('job_input_sha256') == _sha(input_raw), 'Game input hash differs')
        bundle = local_driver.read_bundle(plan)
        _require(type(bundle) is dict and set(bundle) == acceptance.FILES
                 and accepted.get('receipt_sha256') == {name: _sha(raw) for name, raw in bundle.items()},
                 'Game bundle hash differs')
        rows.append(acceptance.accept(manifest.panel, plan, bundle, supervision.get('supervision'),
                                      local_driver.verify_files))
    row_index = {row['game']: row for row in rows}
    _require(len(row_index) == 24, 'Accepted games are not unique')
    comparison_receipts = _receipt_index(result.get('comparison_receipts'),
        lambda item: (item.get('reference_game'), item.get('candidate_game')), 16)
    comparisons = []
    expected = set()
    for reference, candidate in driver.COMPARISON_ARMS:
        for alias in ALIASES:
            for opponent in OPPONENTS:
                for side in (0, 1):
                    worker._registered(check)
                    left = f'dev-{alias}-{opponent}-{side}-{reference}'
                    right = f'dev-{alias}-{opponent}-{side}-{candidate}'
                    key = left, right
                    expected.add(key)
                    _require(key in comparison_receipts, 'Comparison receipt missing')
                    raw = _read(stage, driver._comparison_file(stage, left, candidate),
                                acceptance.MAX_RECEIPT_BYTES)
                    _require(_sha(raw) == comparison_receipts[key].get('sha256'),
                             'Comparison receipt hash differs')
                    doc = acceptance._json(raw, acceptance.MAX_RECEIPT_BYTES)
                    lrow, rrow = row_index[left], row_index[right]
                    _require(doc.get('schema') == 'task-pair-dev-comparison-v1'
                             and doc.get('reference_game') == left and doc.get('candidate_game') == right
                             and doc.get('reference_replay_sha256') == lrow['replay_sha256']
                             and doc.get('candidate_replay_sha256') == rrow['replay_sha256']
                             and doc.get('reference_trace_sha256') == lrow['trace_sha256']
                             and doc.get('candidate_trace_sha256') == rrow['trace_sha256'],
                             'Comparison source identity differs')
                    comparison = doc.get('comparison')
                    _require(type(comparison) is dict and comparison.get('reference') == reference
                             and comparison.get('candidate') == candidate
                             and comparison.get('seed_alias') == alias
                             and comparison.get('opponent') == opponent
                             and type(comparison.get('side')) is int and comparison['side'] == side
                             and type(comparison.get('seed')) is int
                             and comparison['seed'] == lrow['seed'] == rrow['seed'],
                             'Comparison data identity differs from its source')
                    comparisons.append(comparison)
    _require(set(comparison_receipts) == expected, 'Unexpected comparison receipt')
    seeds = {alias: manifest.panel.seeds[index] for index, alias in enumerate(ALIASES)}
    public = projection.project(result, seeds=seeds, rows=rows, comparisons=comparisons)
    worker._registered(check)
    return public
