"""Aggregate frozen teacher-choice geometry from already authenticated arrays.

This module does no I/O, inference or simulation. Its caller owns archive
identity, schema authentication, resource limits and output privacy. The
arrays describe sampled, conditional teacher selections, not completed actions.
All statistics count complete groups; the two competition categories overlap.
"""

import numpy as np

from learning.task_choice_v1.features import OPS


_DTYPES = {
    'shared': np.dtype('float32'),
    'candidates': np.dtype('float32'),
    'offsets': np.dtype('int64'),
    'teacher_choice_label': np.dtype('int64'),
    'ops': np.dtype('uint8'),
}
_COUNTS = (
    'groups',
    'teacher_plant_groups',
    'same_target_unit_competition_groups',
    'teacher_has_farther_alternative_groups',
    'equal_distance_competition_groups',
)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _names(values, length):
    _require(isinstance(values, (list, tuple)) and len(values) == length,
             'Invalid feature names')
    _require(all(isinstance(value, str) and value for value in values)
             and len(set(values)) == length, 'Invalid feature names')
    return {name: index for index, name in enumerate(values)}


def _integer_features(values, maximum=None):
    valid = np.isfinite(values) & (values >= 0) & (values == np.floor(values))
    if maximum is not None:
        valid &= values < maximum
    return bool(np.all(valid))


def _empty_counts():
    return {name: 0 for name in _COUNTS}


def summarize_training_geometry(shared, candidates, offsets, teacher_choice_label,
                                ops, shared_feature_names, feature_names):
    """Count teacher PLANT groups by within-target unit distance competition.

    Inputs must use the frozen training archive dtypes and schema. `encoded_step_2`
    selects the encoded shared feature named `step` with value 2; this module
    does not verify it against original observations or recover callback ordinal.
    The farther-alternative and equal-distance categories are not exclusive:
    a group can contain both a farther unit and an equally distant unit. Units
    include the farmer and farmhands. A
    conditional teacher label is not an executed action, a reward, or an
    optimality claim. Float32 archives cannot distinguish an earlier boolean
    coerced to 1.0 from a number originally encoded as 1.0.

    The caller must verify that these arrays and feature names came from the
    intended authenticated archives/schema; this function cannot establish
    provenance or the full game state from numeric arrays alone.
    """
    values = dict(shared=shared, candidates=candidates, offsets=offsets,
                  teacher_choice_label=teacher_choice_label, ops=ops)
    _require(all(isinstance(values[key], np.ndarray) and values[key].dtype == dtype
                 for key, dtype in _DTYPES.items()), 'Invalid array dtype')
    shared_index = _names(shared_feature_names, 275)
    feature_index = _names(feature_names, 98)
    identity_names = ('unit.index', 'unit.x', 'unit.y', 'target.x', 'target.y')
    required = (*identity_names, 'task.distance', *('task.op.' + op for op in OPS))
    _require('step' in shared_index and 'config.boardSize' in shared_index
             and all(name in feature_index for name in required),
             'Missing required feature')

    _require(shared.ndim == 2 and shared.shape[1] == 275 and len(shared) > 0
             and candidates.ndim == 2 and candidates.shape[1] == 98,
             'Invalid feature shape')
    group_count, row_count = len(shared), len(candidates)
    _require(offsets.shape == (group_count + 1,) and offsets[0] == 0
             and offsets[-1] == row_count and np.all(offsets[1:] > offsets[:-1]),
             'Invalid group offsets')
    _require(teacher_choice_label.shape == (group_count,) and ops.shape == (row_count,),
             'Invalid label or operation shape')
    widths = offsets[1:] - offsets[:-1]
    _require(np.all(teacher_choice_label >= 0) and np.all(teacher_choice_label < widths),
             'Invalid teacher label')
    _require(np.isfinite(shared).all() and np.isfinite(candidates).all(),
             'Nonfinite numeric feature')
    _require(np.all(ops < len(OPS)), 'Invalid candidate operation')
    steps = shared[:, shared_index['step']]
    _require(_integer_features(steps, 719), 'Invalid game step')
    board_sizes = shared[:, shared_index['config.boardSize']]
    _require(_integer_features(board_sizes) and np.all(board_sizes > 0),
             'Invalid board size')

    columns = {name: candidates[:, feature_index[name]] for name in identity_names}
    for value in columns.values():
        _require(_integer_features(value, 32768), 'Invalid candidate identity feature')
    per_row_board_size = np.repeat(board_sizes, widths)
    for name in ('unit.x', 'unit.y', 'target.x', 'target.y'):
        _require(np.all(columns[name] < per_row_board_size), 'Candidate outside board')
    distance = candidates[:, feature_index['task.distance']]
    _require(_integer_features(distance), 'Invalid task distance')
    # Integer coordinate bounds above make this exact in float64.
    expected_distance = (np.abs(columns['unit.x'].astype(np.float64)
                                - columns['target.x'].astype(np.float64))
                         + np.abs(columns['unit.y'].astype(np.float64)
                                  - columns['target.y'].astype(np.float64)))
    _require(np.array_equal(distance, expected_distance), 'Task distance mismatch')
    one_hot = candidates[:, [feature_index['task.op.' + op] for op in OPS]]
    _require(np.array_equal(one_hot, np.eye(len(OPS), dtype=np.float32)[ops]),
             'Candidate operation features mismatch')

    plant = OPS.index('PLANT')
    result = {'all_steps': _empty_counts(), 'encoded_step_2': _empty_counts(),
              'competition_categories_may_overlap': True}
    for group in range(group_count):
        start, end = int(offsets[group]), int(offsets[group + 1])
        previous, positions = None, {}
        for row in range(start, end):
            unit = int(columns['unit.index'][row])
            pos = (int(columns['unit.x'][row]), int(columns['unit.y'][row]))
            _require(unit not in positions or positions[unit] == pos,
                     'Inconsistent candidate unit position')
            positions[unit] = pos
            # Frozen encoder sorts by unit, target y, target x, operation name.
            key = (unit, int(columns['target.y'][row]), int(columns['target.x'][row]),
                   OPS[int(ops[row])])
            _require(previous is None or previous < key,
                     'Duplicate or unordered candidate identity')
            previous = key

        buckets = [result['all_steps']]
        if steps[group] == 2:
            buckets.append(result['encoded_step_2'])
        for bucket in buckets:
            bucket['groups'] += 1
        chosen = start + int(teacher_choice_label[group])
        if int(ops[chosen]) != plant:
            continue
        for bucket in buckets:
            bucket['teacher_plant_groups'] += 1

        others = np.arange(start, end)
        others = others[(ops[start:end] == plant)
                        & (columns['target.x'][start:end] == columns['target.x'][chosen])
                        & (columns['target.y'][start:end] == columns['target.y'][chosen])
                        & (columns['unit.index'][start:end] != columns['unit.index'][chosen])]
        if not len(others):
            continue
        chosen_distance = distance[chosen]
        _require(np.all(distance[others] >= chosen_distance),
                 'Teacher PLANT choice is not nearest for target')
        tied = distance[others] == chosen_distance
        _require(np.all(columns['unit.index'][others[tied]] > columns['unit.index'][chosen]),
                 'Teacher PLANT tie break is inconsistent')
        strict = bool(np.any(distance[others] > chosen_distance))
        equal = bool(np.any(tied))
        for bucket in buckets:
            bucket['same_target_unit_competition_groups'] += 1
            bucket['teacher_has_farther_alternative_groups'] += int(strict)
            bucket['equal_distance_competition_groups'] += int(equal)
    return result
