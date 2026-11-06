"""Select a real new-H choice from one complete authenticated source trace.

No model, source file or game is executed. Selection is chronological and never
uses labels or future outcomes. The caller supplies the source/model identities.
"""
from copy import deepcopy
import math

from decision.api import Atom, View, assess_unit
from learning.task_choice_v1.features import CANDIDATE_SCOPE, SCHEMA_VERSION, candidate_id
from learning.task_harvest_choice_v1.adapter import SCOPE
from learning.task_harvest_coverage_v1.coverage import _dispatch
from learning.task_harvest_window_v1 import selection as old
from learning.task_pair_dev_games_v1.panel import GAME_LIMITS


TARGET_SCHEMA = 'task-harvest-return-target-v1'
TARGET_KEYS = frozenset(('schema_version', 'side', 'step', 'ordinal', 'unit', 'target',
    'board_size', 'unit_count', 'input_sha256', 'task_checkpoint_sha256',
    'crop_checkpoint_sha256', 'harvest_choice', 'water_choice', 'unit_request'))
COST_KEYS = ('task_forwards', 'task_rows', 'crop_forwards', 'crop_rows',
             'gate_evaluations', 'controller_calls', 'teacher_proposals')
ordered_hash, require = old.ordered_hash, old.require


def _sha(value):
    require(type(value) is str and len(value) == 64
            and all(c in '0123456789abcdef' for c in value), 'Invalid bound SHA256')


def validate_target(target):
    require(type(target) is dict and set(target) == TARGET_KEYS
            and target['schema_version'] == TARGET_SCHEMA, 'Malformed return-pair target')
    # The ordinal cap is the registered controller computation budget, not an
    # official game limit on units or action types.
    require(type(target['side']) is int and target['side'] in (0, 1)
            and type(target['step']) is int and 0 <= target['step'] < old.ACTION_TURNS
            and type(target['ordinal']) is int
            and 0 <= target['ordinal'] < GAME_LIMITS['max_task_forwards_per_turn']
            and type(target['board_size']) is int and target['board_size'] > 0
            and type(target['unit_count']) is int and target['unit_count'] > 0
            and type(target['unit']) is int and 0 <= target['unit'] < target['unit_count']
            and old.position(target['target'], target['board_size']), 'Return-pair target out of bounds')
    for key in ('input_sha256', 'task_checkpoint_sha256', 'crop_checkpoint_sha256'):
        _sha(target[key])
    require(target['harvest_choice'] == candidate_id(target['unit'], target['target'], 'HARVEST')
            and target['water_choice'] == candidate_id(target['unit'], target['target'], 'WATER')
            and type(target['unit_request']) is list and target['unit_request'] == ['HARVEST'],
            'Return-pair target does not bind the original HARVEST')
    return deepcopy(target)


def _validated_inputs(inputs, *, step, side, header=None, turn_view=None):
    require(type(inputs) is dict and inputs.get('candidate_scope') in (CANDIDATE_SCOPE, SCOPE),
            'Unknown harvest candidate scope')
    # Reuse the old full deployment/schema validator. Only its scope marker is
    # normalized after explicit validation; identity always hashes real inputs.
    return old._inputs(dict(inputs, candidate_scope=CANDIDATE_SCOPE), step=step,
                       side=side, header=header, turn_view=turn_view)


def validate_binding(target, inputs):
    """Check the actual callback input before a pair's one selector decision."""
    target = validate_target(target)
    _, state, by_id = _validated_inputs(inputs, step=target['step'], side=target['side'])
    obs, cfg, own, positions, _private = state
    require(ordered_hash(inputs) == target['input_sha256'] and inputs['candidate_scope'] == SCOPE
            and cfg['boardSize'] == target['board_size'] and len(positions) == target['unit_count']
            and positions[target['unit']] == target['target'], 'Return-pair input identity or position differs')
    for key, op in (('harvest_choice', 'HARVEST'), ('water_choice', 'WATER')):
        choice = by_id.get(target[key])
        require(choice is not None and choice['unit'] == target['unit']
                and choice['target'] == target['target'] and choice['op'] == op,
                'Return-pair H/W candidate is missing')
    x, y = target['target']
    tile = own['tiles'][y][x]
    view = View(obs, cfg, {})
    require(type(tile) is dict and tile.get('kind') == 'PLANT' and tile.get('crop') == 'MELON'
            and assess_unit(view, target['unit'], Atom('HARVEST'))[0] == 'available'
            and assess_unit(view, target['unit'], Atom('WATER'))[0] == 'available',
            'Return-pair target is not an available direct MELON H/W choice')
    return by_id


def validate_proposal(inputs, evidence, task_sha, *, raw=False):
    """Validate a recorded score or raw network.last without another forward."""
    keys = {'checkpoint_sha256', 'chosen', 'logits', 'fallbacks'}
    if not raw:
        keys.add('argmax_index')
    require(type(evidence) is dict and set(evidence) == keys
            and evidence['checkpoint_sha256'] == task_sha
            and type(evidence['fallbacks']) is int and evidence['fallbacks'] == 0,
            'Task proposal identity or fallback differs')
    logits, candidates = evidence['logits'], inputs['candidates']
    require(type(logits) is list and len(logits) == len(candidates) and bool(logits)
            and all(type(v) is float and math.isfinite(v) for v in logits), 'Invalid task proposal logits')
    index = max(range(len(logits)), key=logits.__getitem__)
    require(evidence['chosen'] == candidates[index]['id']
            and (raw or type(evidence['argmax_index']) is int and evidence['argmax_index'] == index),
            'Task proposal is not its first argmax')
    result = deepcopy(evidence)
    result['argmax_index'] = index
    return result


def _extension(record, by_id, state):
    extension = record['harvest_choice_extension']
    require(type(extension) is dict and set(extension) == {
        'added_candidate_ids', 'selected_virtual_harvest', 'source_water_id'}, 'Malformed virtual-choice evidence')
    added = extension['added_candidate_ids']
    require(type(added) is list and all(type(key) is str and key in by_id for key in added)
            and len(set(added)) == len(added) and type(extension['selected_virtual_harvest']) is bool,
            'Invalid virtual candidate identities')
    candidates = record['inputs']['candidates']
    old_count = len(candidates) - len(added)
    require(old_count > 0 and [c['id'] for c in candidates[old_count:]] == added
            and [old.semantic(c) for c in candidates[:old_count]] == sorted(old.semantic(c) for c in candidates[:old_count])
            and record['inputs']['candidate_scope'] == (SCOPE if added else CANDIDATE_SCOPE),
            'Original prefix or virtual candidate scope differs')
    obs, cfg, own, positions, _private = state
    view = View(obs, cfg, {})
    water_ids = []
    for key in added:
        candidate = by_id[key]
        unit, point = candidate['unit'], candidate['target']
        water = candidate_id(unit, point, 'WATER')
        x, y = point
        tile = own['tiles'][y][x]
        require(candidate['op'] == 'HARVEST' and positions[unit] == point
                and water in by_id and water not in added
                and type(tile) is dict and tile.get('kind') == 'PLANT' and tile.get('crop') == 'MELON'
                and assess_unit(view, unit, Atom('HARVEST'))[0] == 'available'
                and assess_unit(view, unit, Atom('WATER'))[0] == 'available',
                'Virtual HARVEST has no corresponding available direct WATER')
        water_ids.append(water)
    require([old.semantic(by_id[key]) for key in water_ids] == sorted(old.semantic(by_id[key]) for key in water_ids),
            'Virtual choices are not in the original WATER order')
    selected = record['chosen'] in added
    require(extension['selected_virtual_harvest'] is selected
            and extension['source_water_id'] == (candidate_id(record['unit'], record['target'], 'WATER')
                                                  if selected else None),
            'Actual choice and virtual selection evidence disagree')
    return selected, added


def _costs(value):
    require(type(value) is dict and set(value) == set(COST_KEYS)
            and all(type(v) is int and v >= 0 for v in value.values()), 'Malformed trace model counters')
    return value


def select_target(rows, *, side, task_sha, crop_sha, guard=None):
    """Read all 719 rows to EOF, even after the earliest real new-H choice.

    A missing target is a result, never a reason to pick another source. Counts
    describe requests and model calls only, not official execution or return.
    """
    require(type(side) is int and side in (0, 1), 'Invalid source seat')
    _sha(task_sha)
    _sha(crop_sha)
    require(guard is None or callable(guard), 'Invalid resource guard')
    check = guard if guard is not None else lambda: None
    counts = dict(trace_lines=0, callbacks_checked=0, no_callback_turns=0,
                  callbacks_with_virtual=0, virtual_candidates=0, qualifying_callbacks=0)
    target = header = configuration = None
    previous = {key: 0 for key in COST_KEYS}
    for step, row in enumerate(rows):
        check()
        require(step < old.ACTION_TURNS and type(row) is dict and type(row.get('step')) is int
                and row['step'] == step and type(row.get('records')) is list
                and len(row['records']) <= GAME_LIMITS['max_task_forwards_per_turn'],
                'Malformed, discontinuous or oversized source trace')
        require(row.get('schema_version') == SCHEMA_VERSION
                and row.get('task_checkpoint_sha256') == task_sha and row.get('checkpoint_sha256') == crop_sha
                and all(type(row.get(key)) is int and row[key] == 0
                        for key in ('fallbacks', 'load_errors', 'inference_errors')),
                'Source model identity or error counter differs')
        turn, cumulative = _costs(row['turn_model_counts']), _costs(row['model_counts'])
        require(all(cumulative[key] == previous[key] + turn[key] for key in COST_KEYS)
                and turn['controller_calls'] == 1 and turn['gate_evaluations'] == turn['teacher_proposals'] == 0
                and turn['task_forwards'] == len(row['records'])
                and type(row['crop_records']) is list and turn['crop_forwards'] == len(row['crop_records'])
                and turn['crop_rows'] == turn['crop_forwards'], 'Source per-turn and cumulative costs differ')
        require(turn['crop_forwards'] <= GAME_LIMITS['max_crop_forwards_per_turn']
                and turn['task_rows'] + turn['crop_rows'] <= GAME_LIMITS['max_neural_input_rows_per_turn']
                and cumulative['task_forwards'] <= GAME_LIMITS['task_forwards']
                and cumulative['crop_forwards'] <= GAME_LIMITS['crop_forwards']
                and cumulative['task_rows'] + cumulative['crop_rows'] <= GAME_LIMITS['neural_input_rows'],
                'Source model budget exceeds its registered limits')
        previous = dict(cumulative)
        counts['trace_lines'] += 1
        counts['no_callback_turns'] += not row['records']
        state, input_rows, matches = None, 0, []
        for ordinal, record in enumerate(row['records']):
            check()
            require(type(record) is dict and 'pair_choice' not in record, 'Source contains an intervention record')
            try:
                header, state, by_id = _validated_inputs(record['inputs'], step=step, side=side,
                                                       header=header, turn_view=state)
                if configuration is None:
                    configuration = ordered_hash(state[1])
                require(ordered_hash(state[1]) == configuration, 'Source configuration changed')
                chosen = by_id[record['chosen']]
                require(type(record['unit']) is int and record['unit'] == chosen['unit']
                        and record['target'] == chosen['target'] and record['op'] == chosen['op']
                        and record['source'] == 'callback', 'Source actual choice metadata differs')
                proposal = validate_proposal(record['inputs'], record['task_model'], task_sha)
                require(proposal['chosen'] == record['chosen'], 'Source choice is not the actual model proposal')
                selected, added = _extension(record, by_id, state)
            except (KeyError, IndexError, TypeError, AttributeError):
                raise ValueError('Malformed source task record or deployment input') from None
            input_rows += len(by_id)
            counts['callbacks_checked'] += 1
            counts['callbacks_with_virtual'] += bool(added)
            counts['virtual_candidates'] += len(added)
            if selected:
                require(type(record.get('unit_request')) is list and record['unit_request'] == ['HARVEST'],
                        'Selected new H did not generate a direct HARVEST request')
                obs, cfg, _own, positions, _private = state
                point = dict(schema_version=TARGET_SCHEMA, side=side, step=step, ordinal=ordinal,
                    unit=chosen['unit'], target=deepcopy(chosen['target']), board_size=cfg['boardSize'],
                    unit_count=len(positions), input_sha256=ordered_hash(record['inputs']),
                    task_checkpoint_sha256=task_sha, crop_checkpoint_sha256=crop_sha,
                    harvest_choice=chosen['id'], water_choice=candidate_id(chosen['unit'], chosen['target'], 'WATER'),
                    unit_request=deepcopy(record['unit_request']))
                matches.append(validate_target(point))
        require(input_rows == turn['task_rows'], 'Source task row accounting differs')
        _dispatch(row)  # Existing reservation, whole-tile claim and final-request checks.
        counts['qualifying_callbacks'] += len(matches)
        if target is None and matches:
            target = min(matches, key=lambda p: (p['step'], p['ordinal'], p['unit'], p['target'][1], p['target'][0]))
    require(counts['trace_lines'] == old.ACTION_TURNS, 'Source trace did not reach 719 rows and EOF')
    check()
    return dict(target=target, counts=counts,
                reason=None if target is not None else 'no_actual_new_harvest_with_same_unit_water')
