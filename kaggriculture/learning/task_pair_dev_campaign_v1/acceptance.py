"""Validate one private, already supervised development-game receipt bundle."""
import hashlib
import json
import math

from learning.task_pair_dev_games_v1.capture import MAX_METRICS_BYTES
from learning.task_pair_dev_games_v1.panel import ENGINE_SHA, _JOB, _PLAN, _digest
from learning.task_pair_dev_games_v1.worker import NATIVE_OUTPUT_SCOPE


MAX_RECEIPT_BYTES = 512 * 1024
FILES = frozenset(('row', 'audit', 'complete', 'storage', 'metrics', 'metrics_receipt'))


def _require(value, message):
    if not value:
        raise ValueError(message)


def _json(raw, maximum):
    _require(type(raw) is bytes and 0 < len(raw) <= maximum, 'Invalid private receipt size')
    def pairs(items):
        value = {}
        for key, item in items:
            _require(key not in value, 'Duplicate private receipt key')
            value[key] = item
        return value
    def invalid(_):
        raise ValueError('Nonfinite private receipt')
    value = json.loads(raw.decode('utf-8-sig'), object_pairs_hook=pairs, parse_constant=invalid)
    _require(type(value) is dict, 'Invalid private receipt')
    return value


def _num(value):
    _require(type(value) in (int, float) and math.isfinite(value), 'Invalid economic value')
    return value


def supervision_ok(result):
    return (type(result) is dict and result.get('passed') is True
            and result.get('assigned') is True and result.get('resumed') is True
            and result.get('launcher_reaped') is True and result.get('job_members_zero') is True
            and result.get('error_kind') is None and result.get('cleanup_error_kind') is None
            and result.get('stop_reason') is None and result.get('acceptance_error_kind') is None
            and type(result.get('exit_code')) is int and result['exit_code'] == 0
            and type(result.get('rss_complete_samples')) is int and result['rss_complete_samples'] > 0)


def accept(panel, plan, bundle, supervision, verify_files):
    """Return private metrics only after linked receipts and files check out.

    ``verify_files`` is a separate authenticated caller callback. This function
    does not itself authenticate a registration, play a game or replay a trace.
    """
    plan = panel.validate_plan(plan)
    _require(supervision_ok(supervision), 'Worker supervision did not pass')
    _require(type(bundle) is dict and set(bundle) == FILES, 'Incomplete receipt bundle')
    values = {name: _json(raw, MAX_METRICS_BYTES if name == 'metrics' else MAX_RECEIPT_BYTES)
              for name, raw in bundle.items()}
    row, audit, complete, storage = (values[k] for k in ('row', 'audit', 'complete', 'storage'))
    metrics, receipt = values['metrics'], values['metrics_receipt']
    _require(_JOB.issubset(row), 'Missing job identity')
    job = panel.validate_plan({key: row[key] for key in _JOB}, job=True)
    for name in ('trace_sha256', 'replay_gzip_sha256', 'semantic_replay_sha256', 'semantic_replay_sha256_v2'):
        _digest(row.get(name))
    _require({key: job[key] for key in _PLAN} == plan, 'Accepted job differs')
    _require(row.get('passed') is True and type(row.get('states')) is int and row['states'] == 720
             and type(row.get('action_turns')) is int and row['action_turns'] == 719
             and row.get('status') == ['DONE', 'DONE'] and type(row.get('agent_errors')) is int
             and row['agent_errors'] == 0 and row.get('engine_sha256') == ENGINE_SHA
             and row.get('crop_checkpoint_sha256') == panel.crop[1], 'Official game did not pass')
    _require(type(row.get('banks')) is list and len(row['banks']) == 2
             and type(row.get('initial_banks')) is list and len(row['initial_banks']) == 2
             and row.get('rewards') == row['banks'], 'Invalid official balances')
    banks = [_num(v) for v in row['banks']]
    initial = [_num(v) for v in row['initial_banks']]
    side = plan['side']; other = 1 - side
    margin = banks[side] - banks[other]
    _require(_num(row.get('bank')) == banks[side] and _num(row.get('opponent_bank')) == banks[other]
             and _num(row.get('initial_bank')) == initial[side]
             and _num(row.get('opponent_initial_bank')) == initial[other]
             and _num(row.get('net_profit')) == banks[side] - initial[side]
             and _num(row.get('opponent_net_profit')) == banks[other] - initial[other]
             and _num(row.get('margin')) == margin
             and _num(row.get('score')) == (1. if margin > 0 else .5 if margin == 0 else 0.),
             'Official economic result differs')
    official = audit.get('official')
    _require(audit.get('passed') is True and audit.get('game') == plan['game']
             and type(audit.get('errors')) is int and audit['errors'] == 0
             and all(type(audit.get(k)) is int and audit[k] == 719 for k in
                     ('actor_actions_reproduced', 'rival_actions_reproduced', 'trace_turns_reproduced'))
             and audit.get('trace_sha256') == row.get('trace_sha256')
             and audit.get('replay_sha256') == row.get('replay_gzip_sha256')
             and type(official) is dict and official.get('passed') is True
             and type(official.get('official_transitions_reexecuted')) is int
             and official['official_transitions_reexecuted'] == 719
             and official.get('both_public_and_private_states_match_each_step') is True
             and official.get('inventory_order_checked_each_step') is True
             and official.get('engine_sha256') == ENGINE_SHA
             and official.get('semantic_sha256') == row.get('semantic_replay_sha256')
             and official.get('semantic_sha256_v2') == row.get('semantic_replay_sha256_v2')
             and official.get('final_rewards') == banks, 'Official replay did not pass')
    counts = audit.get('primary_model_counts')
    _require(type(counts) is dict and counts == audit.get('live_model_counts')
             and type(counts.get('task_forwards')) is int and counts['task_forwards'] > 0
             and type(counts.get('controller_calls')) is int and counts['controller_calls'] == 719,
             'Model call reconstruction differs')
    _require(complete.get('passed') is True and complete.get('game') == plan['game']
             and complete.get('row_sha256') == hashlib.sha256(bundle['row']).hexdigest()
             and complete.get('certificate_sha256') == hashlib.sha256(bundle['audit']).hexdigest(),
             'Worker completion identity differs')
    output = storage.get('output')
    _require(storage.get('passed') is True and storage.get('game') == plan['game']
             and storage.get('error_kind') is None
             and storage.get('native_output_accounted_by') == NATIVE_OUTPUT_SCOPE
             and type(output) is dict and type(output.get('limit_bytes')) is int
             and output['limit_bytes'] == panel.worker_config()['storage']['game_write_bytes']
             and type(output.get('used_bytes')) is int and 0 <= output['used_bytes'] <= output['limit_bytes']
             and type(output.get('remaining_bytes')) is int
             and output['remaining_bytes'] == output['limit_bytes'] - output['used_bytes']
             and type(output.get('refused')) is int and output['refused'] == 0,
             'Worker output budget differs')
    _require(receipt.get('schema') == 'task-pair-dev-game-metrics-receipt-v1'
             and receipt.get('game') == plan['game']
             and type(receipt.get('official_replay_passes')) is int and receipt['official_replay_passes'] == 1
             and type(receipt.get('metrics_bytes')) is int and receipt['metrics_bytes'] == len(bundle['metrics'])
             and receipt.get('metrics_sha256') == hashlib.sha256(bundle['metrics']).hexdigest()
             and type(receipt.get('budget_bytes')) is int and receipt['budget_bytes'] == MAX_METRICS_BYTES
             and len(bundle['metrics']) + len(bundle['metrics_receipt']) <= MAX_METRICS_BYTES,
             'Private metrics receipt differs')
    _require(metrics.get('schema') == 'task-pair-dev-game-metrics-v1'
             and metrics.get('game') == plan['game'] and metrics.get('seed_alias') == plan['seed_alias']
             and type(metrics.get('seed')) is int and metrics['seed'] == plan['seed']
             and metrics.get('opponent') == plan['opponent'] and type(metrics.get('side')) is int
             and metrics['side'] == side and metrics.get('arm') == plan['policy']
             and metrics.get('task_checkpoint_sha256') == job['task_checkpoint_sha256']
             and metrics.get('crop_checkpoint_sha256') == panel.crop[1]
             and metrics.get('trace_sha256') == row.get('trace_sha256')
             and metrics.get('replay_sha256') == row.get('replay_gzip_sha256'),
             'Private metrics identity differs')
    result = metrics.get('result')
    _require(type(result) is dict and type(result.get('actor')) is int and result['actor'] == side
             and type(result.get('official_turns')) is int and result['official_turns'] == 719
             and type(result.get('economy')) is dict, 'Consumer result differs')
    own, rival = result['economy'].get('own'), result['economy'].get('opponent')
    _require(type(own) is dict and type(rival) is dict and own.get('cash_reconciled') is True
             and rival.get('cash_reconciled') is True and _num(own.get('initial_cash')) == initial[side]
             and _num(own.get('final_cash')) == banks[side] and _num(own.get('net_profit')) == row['net_profit']
             and _num(rival.get('initial_cash')) == initial[other] and _num(rival.get('final_cash')) == banks[other]
             and _num(rival.get('net_profit')) == row['opponent_net_profit']
             and _num(result.get('margin')) == margin and _num(result.get('score')) == row['score'],
             'Consumer economy differs')
    _require(callable(verify_files) and verify_files(plan, row) is True, 'Game file identity check failed')
    return metrics
