"""Classify a recorded task choice without emitting private candidate data."""
import math

OPERATIONS = ('DIG', 'FERTILIZE', 'HARVEST', 'PLANT', 'WATER')
MOVES = ('NORTH', 'SOUTH', 'EAST', 'WEST')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def integer(value):
    return type(value) is int


def point(value):
    require(isinstance(value, (list, tuple)) and len(value) == 2 and all(integer(v) for v in value),
            'Invalid recorded point')
    return tuple(value)


def classify_record(record, side, schema):
    require(type(side) is int and side in (0, 1), 'Invalid actor side')
    require(record['source'] == 'callback', 'Expected recorded model callback')
    inputs = record['inputs']
    feature_names, shared_names = schema['feature_names'], schema['shared_feature_names']
    require(inputs['feature_names'] == feature_names and inputs['shared_feature_names'] == shared_names
            and inputs['schema_sha256'] == schema['schema_sha256'], 'Recorded schema differs')
    require(len(feature_names) == len(set(feature_names)) and len(shared_names) == len(set(shared_names)),
            'Duplicate feature names')
    names = ('unit.index', 'unit.x', 'unit.y', 'target.x', 'target.y', 'task.distance')
    operation_names = tuple('task.op.' + op for op in OPERATIONS)
    require(all(name in feature_names for name in names + operation_names) and 'step' in shared_names,
            'Required feature missing')
    indices = {name: feature_names.index(name) for name in names + operation_names}
    observation = inputs['observation']
    require(type(observation['player']) is int and observation['player'] == side
            and type(observation['step']) is int and observation['step'] == 2
            and inputs['shared_x'][shared_names.index('step')] == 2,
            'Recorded turn differs')
    size = inputs['configuration']['boardSize']
    require(integer(size) and size > 0, 'Invalid board size')
    farm = observation['farms'][side]
    units = [point(farm['farmer'])] + [point(pos) for pos in farm['hands']]
    require(all(0 <= x < size and 0 <= y < size for x, y in units), 'Unit outside board')
    available = inputs['planned']['available_units']
    require(type(available) is list and all(integer(i) and 0 <= i < len(units) for i in available)
            and len(set(available)) == len(available), 'Invalid available units')
    candidates = inputs['candidates']
    require(type(candidates) is list and candidates, 'Missing candidates')
    ids = [candidate['id'] for candidate in candidates]
    require(len(set(ids)) == len(ids), 'Duplicate candidate identity')
    chosen = [candidate for candidate in candidates if candidate['id'] == record['chosen']]
    teacher = [candidate for candidate in candidates if candidate['id'] == record['teacher_label']]
    require(len(chosen) == len(teacher) == 1 and record['chosen'] != record['teacher_label'],
            'Choice identities differ from expected contrast')
    chosen, teacher = chosen[0], teacher[0]
    require(chosen['op'] == teacher['op'] == 'PLANT' and point(chosen['target']) == point(teacher['target'])
            and chosen['unit'] != teacher['unit'], 'Expected same-target different-unit choices')
    require(integer(record['unit']) and record['unit'] == chosen['unit']
            and point(record['target']) == point(chosen['target'])
            and record['op'] == 'PLANT' and isinstance(record['unit_request'], list)
            and record['unit_request'] and record['unit_request'][0] in MOVES,
            'Selected record or request differs')

    def measured(candidate):
        unit = candidate['unit']
        target = point(candidate['target'])
        require(integer(unit) and unit in available and 0 <= unit < len(units)
                and all(0 <= coordinate < size for coordinate in target), 'Candidate unit or target invalid')
        values = candidate['x']
        require(isinstance(values, list) and len(values) == len(feature_names)
                and all(type(v) in (int, float) and math.isfinite(v) for v in values), 'Invalid candidate features')
        ux, uy = units[unit]
        tx, ty = target
        distance = abs(ux - tx) + abs(uy - ty)
        expected = {'unit.index': unit, 'unit.x': ux, 'unit.y': uy, 'target.x': tx, 'target.y': ty,
                    'task.distance': distance}
        expected.update({name: float(name == 'task.op.PLANT') for name in operation_names})
        require(all(values[indices[name]] == expected[name] for name in names + operation_names),
                'Candidate geometry differs')
        return unit, distance

    chosen_unit, chosen_distance = measured(chosen)
    teacher_unit, teacher_distance = measured(teacher)
    target = point(teacher['target'])
    competitors = [candidate for candidate in candidates
                   if candidate['op'] == 'PLANT' and point(candidate['target']) == target]
    require(len(competitors) >= 2 and len({candidate['unit'] for candidate in competitors}) == len(competitors),
            'Missing or duplicate same-target competition')
    for candidate in competitors:
        unit, distance = measured(candidate)
        require(distance >= teacher_distance and (distance != teacher_distance or unit >= teacher_unit),
                'Conditional teacher ordering differs')
    if chosen_distance > teacher_distance:
        return 'student_farther'
    require(chosen_distance == teacher_distance and teacher_unit < chosen_unit,
            'Unexpected equal-distance tie')
    return 'same_distance_teacher_lower_unit'
