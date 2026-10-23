"""Observation-only path comparison. No policy execution or effect attribution.

The caller supplies an already audited factual trace and an opposite-policy
shadow trace from the identical sanitized observation. Callback ordinals are
paired only up to (and including inspection of) the first unequal input.
"""
from copy import deepcopy
import hashlib
import json
import math


RESERVATIONS = ('seeds', 'needed_seeds', 'supply', 'projected_shed', 'unit_actions')


def _typed(value):
    """Lossless JSON-domain encoding, retaining object/list order and f64 bits."""
    if value is None:
        return ['null']
    if isinstance(value, bool):
        return ['bool', value]
    if isinstance(value, int):
        return ['int', str(value)]
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError('Nonfinite comparison input')
        return ['f64', value.hex()]
    if isinstance(value, str):
        return ['str', value]
    if isinstance(value, (list, tuple)):
        return ['array', [_typed(x) for x in value]]
    if isinstance(value, dict) and all(isinstance(k, str) for k in value):
        return ['object', [[k, _typed(v)] for k, v in value.items()]]
    raise TypeError('Comparison inputs must be JSON-domain values')


def ordered_hash(value):
    return hashlib.sha256(json.dumps(_typed(value), ensure_ascii=False,
        separators=(',', ':'), allow_nan=False).encode('utf-8')).hexdigest()


def _equal(a, b):
    return _typed(a) == _typed(b)


def first_difference(a, b, path=()):
    """Return a structural path only, never a raw private value."""
    if _equal(a, b):
        return None
    if isinstance(a, dict) and isinstance(b, dict):
        if list(a) != list(b):
            return list(path) + ['<keys-or-order>']
        for key in a:
            if not _equal(a[key], b[key]):
                return first_difference(a[key], b[key], (*path, key))
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        for i, (left, right) in enumerate(zip(a, b)):
            if not _equal(left, right):
                return first_difference(left, right, (*path, i))
        return list(path) + ['<length>']
    return list(path)


def _validate_trace(trace):
    records = trace['records']
    if not isinstance(records, list):
        raise ValueError('Task records must be an ordered list')
    for record in records:
        inputs = record['inputs']
        for key in ('shared_x', 'candidates', 'planned', 'observation',
                    'configuration', 'inventory_item_order', 'shed_item_order'):
            if key not in inputs:
                raise ValueError('Missing full structured task context: ' + key)
        for key in (*RESERVATIONS, 'available_units'):
            if key not in inputs['planned']:
                raise ValueError('Missing pre-assignment reservation: ' + key)
        candidates = inputs['candidates']
        ids = [c['id'] for c in candidates]
        if not ids or len(set(ids)) != len(ids):
            raise ValueError('Invalid candidate identities')
        if record['chosen'] not in ids or record['teacher_label'] not in ids:
            raise ValueError('Unknown chosen or teacher candidate identity')
        for candidate in candidates:
            for key in ('unit', 'target', 'op', 'x'):
                if key not in candidate:
                    raise ValueError('Incomplete candidate identity or numeric vector')
        record['unit_request']
        ordered_hash(inputs)  # Reject nonfinite values, including actual f64 inputs.
    for key in RESERVATIONS:
        trace['final_reservations'][key]
    return records


def compare_turn(factual_action, factual_trace, shadow_action, shadow_trace):
    """Return compact private-value-free comparison and exact source indexes.

    All differences concern requests/reservations. A missing subsequent callback
    is a path-length difference, never evidence of legal execution. No later
    ordinals are paired after input divergence, even if their inputs reconverge.
    """
    factual, shadow = _validate_trace(factual_trace), _validate_trace(shadow_trace)
    for action in (factual_action, shadow_action):
        for component in ('farmer', 'hands', 'market'):
            action[component]
    result = dict(schema='task-shadow-comparison-v1',
        factual_task_calls=len(factual), shadow_task_calls=len(shadow),
        zero_task_turn=(not factual and not shadow),
        factual_zero_task_turn=not factual, shadow_zero_task_turn=not shadow,
        action_equal=_equal(factual_action, shadow_action),
        action_sha256=dict(factual=ordered_hash(factual_action), shadow=ordered_hash(shadow_action)),
        action_components_differ={key: not _equal(factual_action[key], shadow_action[key])
                                  for key in ('farmer', 'hands', 'market')},
        first_action_difference=first_difference(factual_action, shadow_action),
        matched_context_calls=0, first_choice_difference=None,
        first_context_difference=None, first_candidate_difference=None,
        first_reservation_difference=None, first_unit_request_difference=None,
        aligned_choice_differences=0, aligned_different_choice_same_unit_request=0,
        path_length_different=len(factual) != len(shadow),
        final_reservations_differ={key: not _equal(factual_trace['final_reservations'][key],
                                                  shadow_trace['final_reservations'][key])
                                   for key in RESERVATIONS},
        shadow_executed=False, factual_effect='unknown', shadow_effect='unobserved',
        effect_boundary='Request differences alone establish neither execution nor reward causation')
    stop = min(len(factual), len(shadow))
    for index in range(stop):
        left, right = factual[index], shadow[index]
        a, b = left['inputs'], right['inputs']
        if not _equal(a, b):
            result['first_context_difference'] = dict(factual_index=index, shadow_index=index,
                input_path=first_difference(a, b),
                factual_input_sha256=ordered_hash(a), shadow_input_sha256=ordered_hash(b))
            if not _equal(a['candidates'], b['candidates']):
                result['first_candidate_difference'] = dict(factual_index=index, shadow_index=index,
                    path=first_difference(a['candidates'], b['candidates']))
            if not _equal(a['planned'], b['planned']):
                result['first_reservation_difference'] = dict(factual_index=index, shadow_index=index,
                    path=first_difference(a['planned'], b['planned']))
            stop = index
            break
        result['matched_context_calls'] += 1
        request_equal = _equal(left['unit_request'], right['unit_request'])
        if not request_equal and result['first_unit_request_difference'] is None:
            result['first_unit_request_difference'] = dict(factual_index=index, shadow_index=index,
                same_unit=left['unit'] == right['unit'],
                path=first_difference(left['unit_request'], right['unit_request']))
        if left['chosen'] != right['chosen']:
            result['aligned_choice_differences'] += 1
            result['aligned_different_choice_same_unit_request'] += int(request_equal)
            if result['first_choice_difference'] is None:
                result['first_choice_difference'] = dict(factual_index=index, shadow_index=index,
                    same_unit_request=request_equal, same_unit=left['unit'] == right['unit'],
                    same_operation=left['op'] == right['op'],
                    same_target=_equal(left['target'], right['target']),
                    input_sha256=ordered_hash(a))
    result['alignment_stop_index'] = stop
    result['alignment_stop_reason'] = ('context_divergence' if result['first_context_difference']
        else 'path_length_difference' if result['path_length_different'] else 'both_paths_ended')
    result['unmatched_task_calls'] = dict(factual=len(factual)-stop, shadow=len(shadow)-stop)
    result['unmatched_tail_interpretation'] = 'Unaligned paths; callback ordinal is not a matched decision'
    return result
