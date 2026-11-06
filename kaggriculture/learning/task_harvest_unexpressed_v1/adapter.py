"""One pure offline request replacement; controller/network intent is untouched.

No agent is called here. The continuation owns exactly-once invocation, normal
model accounting, source authentication and factual-history reproduction. This
is not a legal-action mask or a deployable strategy. The official engine alone
determines execution after earlier unit operations and market effects.
"""
from copy import deepcopy

from learning.task_harvest_coverage_v1 import coverage
from .selection import ordered_hash, require, validate_target, validate_target_trace


CHANGE_SCHEMA = 'task-harvest-unexpressed-change-v1'


def _units(request):
    # Match the frozen controller's request containers, without interpreting,
    # repairing, truncating or reordering any market order or non-target atom.
    require(type(request) is dict and set(request) == {'farmer', 'hands', 'market'}
            and all(type(request[key]) is list for key in request),
            'Incomplete original controller request')
    actions = [request['farmer'], *request['hands']]
    coverage._requests(actions)
    ordered_hash(request)  # Reject non-JSON/nonfinite values, retaining order/types.
    return actions


def replace_harvest(original_request, *, target, trace, first_yield_days):
    """Return independent original/submitted snapshots and one change receipt.

    The trace's unit_request and final_reservations remain original intent.
    Request digests here include market orders; the selected metadata point's
    unit_requests_sha256 explicitly does not. Caller retains the original trace.
    """
    target = validate_target(target)
    validate_target_trace(target, trace, first_yield_days=first_yield_days)
    actions = _units(original_request)
    require(ordered_hash(actions) == target['unit_requests_sha256'],
            'Original request differs from frozen trace unit actions')
    unit = target['unit']
    require(unit < len(actions)
            and ordered_hash(actions[unit]) == ordered_hash(target['original_unit_request']),
            'Original target unit request differs')
    original, submitted = deepcopy(original_request), deepcopy(original_request)
    if unit == 0:
        submitted['farmer'] = ['HARVEST']
    else:
        submitted['hands'][unit - 1] = ['HARVEST']
    updated = _units(submitted)
    require([i for i, (before, after) in enumerate(zip(actions, updated))
             if ordered_hash(before) != ordered_hash(after)] == [unit],
            'Replacement changed another unit or did not change the target')
    require(list(original) == list(submitted)
            and ordered_hash(original['market']) == ordered_hash(submitted['market']),
            'Replacement changed request or market order')
    change = dict(schema_version=CHANGE_SCHEMA, side=target['side'], step=target['step'],
                  unit=unit, target=deepcopy(target['target']), input_sha256=target['input_sha256'],
                  unit_requests_sha256=target['unit_requests_sha256'],
                  original_unit_request=deepcopy(target['original_unit_request']),
                  submitted_unit_request=['HARVEST'],
                  original_request_sha256=ordered_hash(original),
                  submitted_request_sha256=ordered_hash(submitted), changed_units=1,
                  other_units_unchanged=True, market_unchanged=True)
    return dict(original=original, submitted=submitted, change=change)
