"""Compare two rule-controller callback streams on one legal observation.

Only counts leave this function. A match means that a teacher's unit/target/op
ID occurs in the terminal controller's opportunities at the same validated
reservation and chosen-action prefix. It does not establish official legality,
task execution, crop choice, complete-request equivalence, or future reward.

Each input is one chronological stream. Its prior-ID prefix strictly grows by
one on every callback, so multiple records with the same full prefix cannot
exist; duplicated or malformed stream structure fails instead of being counted
as an ambiguous match. Market orders are made after these callbacks and are not
present in this interface, so their final order is not compared here.

Reservation quantities and available-unit actions are checked against the
three frozen controllers used by this probe, not a general official legality
rule: those controllers keep non-negative integer counts and leave available
units at PASS until an assignment removes them from the available set.
"""
from decision.api import Atom, CROPS, ITEMS, observe
from learning.task_choice_v1.features import OPS, SCHEMA_VERSION
from learning.task_multi_teacher_probe_v1.teacher import ProbeError, _snapshot
from learning.task_teacher_choice_bc_v1.extract import extract


MAX_CALLBACKS = 32
MAX_CANDIDATES = 256
RECORD_KEYS = frozenset(('inputs', 'teacher_label', 'chosen', 'unit', 'target', 'op',
                         'source', 'unit_request', 'intended_action'))
PLANNED_KEYS = frozenset(('seeds', 'needed_seeds', 'supply', 'projected_shed',
                          'unit_actions', 'available_units', 'interpretation'))
SCHEMA_KEYS = ('schema_version', 'schema_sha256', 'shared_feature_names', 'feature_names')


class AlignmentError(ValueError):
    """A private observation, callback or comparison contract is invalid."""


def _require(value, message):
    if not value:
        raise AlignmentError(message)


def _frozen(value):
    try:
        return _snapshot(value)
    except ProbeError:
        raise AlignmentError('invalid structured comparison value') from None


def _schema(value):
    _require(type(value) is dict and all(key in value for key in SCHEMA_KEYS), 'missing task schema')
    _require(value.get('schema_version') == SCHEMA_VERSION
             and ('ops' not in value or value['ops'] == list(OPS)), 'task schema differs')
    return {**{key: value[key] for key in SCHEMA_KEYS}, 'ops': list(OPS)}


def _request(value):
    """Validate canonical syntax only, not isolated or joint official legality."""
    _require(type(value) is list, 'invalid unit request')
    try:
        Atom.parse(value)
        return _frozen(value)
    except (ValueError, TypeError, OverflowError):
        raise AlignmentError('invalid unit request') from None


def _intent(value, op):
    _require(type(value) is list and value and value[0] == op and op in OPS,
             'invalid intended task')
    if op == 'PLANT':
        _require(len(value) in (1, 2) and (len(value) == 1 or value[1] in CROPS),
                 'invalid planting intent')
    else:
        _require(len(value) == 1, 'invalid intended task')
    return value


def _prior_token(record):
    """A moving PLANT has no proven crop identity and breaks later alignment."""
    op, intended, request = record['op'], record['intended_action'], record['unit_request']
    if op == 'PLANT':
        if not (len(request) == 2 and request[0] == 'PLANT' and request[1] in CROPS):
            return None
        if len(intended) == 2:
            _require(intended[1] == request[1], 'plant intent and unit request differ')
        intended = ['PLANT', request[1]]
    return (record['chosen'], _frozen(intended), _frozen(request))


def _prepare(records, *, view, schema, guard):
    _require(type(records) is list and len(records) <= MAX_CALLBACKS, 'callback limit exceeded')
    prepared, history, previous_available, previous_unit, seen_units = [], (), None, None, set()
    prior_requests = {}
    expected_view = (_frozen(view.observation), _frozen(view.configuration))
    units = [view.own['farmer'], *view.own['hands']]
    for record in records:
        guard()
        _require(type(record) is dict and set(record) == RECORD_KEYS and record.get('source') == 'teacher'
                 and record.get('chosen') == record.get('teacher_label'), 'invalid teacher callback')
        inputs = record.get('inputs')
        _require(type(inputs) is dict and (_frozen(inputs.get('observation')),
                 _frozen(inputs.get('configuration'))) == expected_view, 'single-player view differs')
        try:
            extract(record, schema, side=view.observation['player'], step=view.observation['step'],
                    max_candidates=MAX_CANDIDATES)
        except ValueError:
            raise AlignmentError('invalid candidate encoding or visibility') from None
        planned = inputs.get('planned')
        _require(type(planned) is dict and set(planned) == PLANNED_KEYS,
                 'invalid reservation context')
        for key in ('seeds', 'needed_seeds', 'supply', 'projected_shed'):
            value = planned.get(key)
            allowed = ITEMS if key == 'projected_shed' else CROPS
            _require(type(value) is dict and all(type(item) is str and item in allowed
                     and type(amount) is int and amount >= 0
                     for item, amount in value.items()), 'invalid reservation resources')
        available, actions = planned.get('available_units'), planned.get('unit_actions')
        _require(type(available) is list and all(type(i) is int and 0 <= i < len(units) for i in available)
                 and available == sorted(set(available)) and type(actions) is list and len(actions) == len(units),
                 'invalid reserved units')
        for action in actions:
            _request(action)
        _require(all(actions[index] == ['PASS'] for index in available),
                 'available unit already has an action')
        _require(all(_frozen(actions[index]) == request for index, request in prior_requests.items()),
                 'prior unit request differs from reservation')
        unit = record.get('unit')
        _require(type(unit) is int and unit in available and unit not in seen_units,
                 'duplicate or unavailable selected unit')
        if previous_available is not None:
            _require(set(available) == previous_available - {previous_unit},
                     'selected unit history differs')
        _intent(record.get('intended_action'), record.get('op'))
        _request(record.get('unit_request'))
        prefix = history
        # The complete planned record includes unit-action order and all
        # resources. Explicit item-order arrays also retain private inventory
        # and shed order that can affect the official engine.
        reservation = (_frozen(planned), _frozen(inputs.get('inventory_item_order')),
                       _frozen(inputs.get('shed_item_order')))
        prepared.append(dict(prefix=prefix, reservation=reservation, record=record))
        token = _prior_token(record)
        if history is not None:
            # A later apparent reconvergence does not repair an earlier
            # reservation mismatch; retain every prior ordered context.
            history = None if token is None else (*history, (token, reservation))
        previous_available, previous_unit = set(available), unit
        prior_requests[unit] = _frozen(record['unit_request'])
        seen_units.add(unit)
    return prepared


def align_turn(observation, configuration, teacher_records, terminal_records, *, schema, guard=lambda: None):
    """Return anonymous counts from two complete same-turn callback streams.

    The caller supplies only one already-legal player view and two complete,
    chronological rule-mode streams generated on that same view. Histories
    are matched by prior ``(chosen ID, intended action, unit request)`` tuples,
    never by assuming two record ordinals describe the same decision. An
    unresolved earlier PLANT crop or any reservation mismatch is unknown.

    Feature equality is reported only for a supported target: it compares
    shared features and the selected candidate's features, not whole candidate
    sets. No teacher label or source identity is passed to a decision callback.
    Exceptions, including resource-guard exceptions, must stop the caller.
    """
    _require(callable(guard) and type(observation) is dict and type(configuration) is dict
             and type(observation.get('player')) is int and observation['player'] in (0, 1)
             and type(observation.get('step')) is int and 0 <= observation['step'] < 719,
             'invalid single-player view')
    guard()
    try:
        view = observe(observation, configuration)
    except (KeyError, TypeError, ValueError, IndexError):
        raise AlignmentError('invalid single-player view') from None
    _require((_frozen(view.observation), _frozen(view.configuration)) ==
             (_frozen(observation), _frozen(configuration)), 'single-player view contains excluded data')
    expected_schema = _schema(schema)
    teachers = _prepare(teacher_records, view=view, schema=expected_schema, guard=guard)
    terminals = _prepare(terminal_records, view=view, schema=expected_schema, guard=guard)
    by_prefix = {}
    for item in terminals:
        prefix = item['prefix']
        if prefix is not None:
            _require(prefix not in by_prefix, 'duplicate terminal history')
            by_prefix[prefix] = item
    result = dict(teacher_callbacks=len(teachers), terminal_callbacks=len(terminals),
                  teacher_empty_turn=int(not teachers), terminal_empty_turn=int(not terminals),
                  supported=0, unsupported=0, unknown=0, ambiguous=0,
                  supported_target_features_equal=0, supported_target_features_differ=0)
    for teacher in teachers:
        guard()
        prefix = teacher['prefix']
        other = by_prefix.get(prefix) if prefix is not None else None
        if other is None or teacher['reservation'] != other['reservation']:
            result['unknown'] += 1
            continue
        left, right = teacher['record'], other['record']
        wanted = left['chosen']
        terminal_candidates = {candidate['id']: candidate for candidate in right['inputs']['candidates']}
        if wanted not in terminal_candidates:
            result['unsupported'] += 1
            continue
        result['supported'] += 1
        own_candidate = next(candidate for candidate in left['inputs']['candidates'] if candidate['id'] == wanted)
        equal = (_frozen(left['inputs']['shared_x']) == _frozen(right['inputs']['shared_x'])
                 and _frozen(own_candidate['x']) == _frozen(terminal_candidates[wanted]['x']))
        result['supported_target_features_equal' if equal else 'supported_target_features_differ'] += 1
    _require(result['supported'] + result['unsupported'] + result['unknown'] == len(teachers)
             and result['supported_target_features_equal'] + result['supported_target_features_differ']
                 == result['supported'], 'alignment counts do not balance')
    return result
