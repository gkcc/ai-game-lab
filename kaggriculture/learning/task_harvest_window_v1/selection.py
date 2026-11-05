"""First observed direct-harvest alternative in a fixed consumed-trace window.

This is metadata selection, not model execution or an economic/legal action mask.
The caller authenticates source files and budgets. Every row is validated through
EOF, including rows after the first hit. Hidden labels and later outcomes are
never consulted. The selected input retains the existing order-aware digest.
"""
import ast
import hashlib
import math

from decision.api import observe
from learning.task_bc_error_diagnostics_v1.shadow import ordered_hash
from learning.task_choice_v1.features import (
    CANDIDATE_SCOPE, OPS, SCHEMA_VERSION, candidate_id, schema_hash,
)

ENGINE_SHA = 'bc8a54879ef02c7ea64b8b333d6a976f0ea65c4949149d01f463f23bccee653e'
WINDOW = (240, 648)
ACTION_TURNS = 719
INPUT_KEYS = frozenset((
    'schema_version', 'schema_sha256', 'feature_names', 'shared_feature_names',
    'shared_x', 'candidate_scope', 'observation', 'configuration', 'planned',
    'candidates', 'inventory_item_order', 'shed_item_order',
))
PLANNED_KEYS = frozenset((
    'seeds', 'needed_seeds', 'supply', 'projected_shed', 'unit_actions',
    'available_units', 'interpretation',
))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def first_yield_days_from_source(raw):
    """Extract only the hash-pinned official CROPS literal; never execute source."""
    require(type(raw) is bytes and hashlib.sha256(raw).hexdigest() == ENGINE_SHA,
            'Official crop source SHA differs')
    try:
        tree = ast.parse(raw.decode('utf-8'))
        definitions = [node.value for node in tree.body
                       if isinstance(node, ast.Assign)
                       and any(isinstance(target, ast.Name) and target.id == 'CROPS'
                               for target in node.targets)]
        require(len(definitions) == 1, 'Official CROPS literal is ambiguous')
        crops = ast.literal_eval(definitions[0])
        require(type(crops) is dict and crops, 'Official CROPS literal is empty')
        days = {}
        for crop, definition in crops.items():
            require(type(crop) is str and bool(crop) and type(definition) is dict,
                    'Malformed official crop definition')
            value = definition['first_yield_day']
            require(type(value) is int and value >= 0, 'Invalid official first yield day')
            days[crop] = value
        return days
    except (SyntaxError, UnicodeError, TypeError, KeyError) as exc:
        raise ValueError('Official CROPS is not a valid literal table') from None


def finite(value):
    if type(value) not in (int, float):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def vector(value, length):
    require(type(value) is list and len(value) == length and all(map(finite, value)),
            'Malformed finite task vector')


def position(value, board):
    return (type(value) is list and len(value) == 2
            and all(type(v) is int and 0 <= v < board for v in value))


def inventory(value):
    require(type(value) is dict and all(type(k) is str and type(n) is int and n >= 0
                                       for k, n in value.items()), 'Malformed own inventory')


def semantic(candidate):
    x, y = candidate['target']
    return candidate['unit'], y, x, candidate['op'], candidate['id']


def _inputs(inputs, *, step, side, header, turn_view):
    require(type(inputs) is dict and set(inputs) == INPUT_KEYS,
            'Malformed deployment task input fields')
    names, shared = inputs['feature_names'], inputs['shared_feature_names']
    require(type(names) is list and type(shared) is list
            and all(type(n) is str and n for n in names + shared),
            'Malformed feature names')
    current = (inputs['schema_version'], inputs['schema_sha256'],
               tuple(names), tuple(shared), inputs['candidate_scope'])
    if header is None:
        require(current[0] == SCHEMA_VERSION and names and shared
                and len(set(names)) == len(names) and len(set(shared)) == len(shared)
                and current[1] == schema_hash(names, shared) and current[4] == CANDIDATE_SCOPE,
                'Task input schema differs')
        header = current
    require(current == header, 'Task input schema changed within trace')
    vector(inputs['shared_x'], len(shared))
    obs, cfg = inputs['observation'], inputs['configuration']
    require(type(obs) is dict and type(cfg) is dict
            and type(obs.get('player')) is int and obs['player'] == side
            and type(obs.get('step')) is int and obs['step'] == step,
            'Observation step or seat differs')
    if turn_view is None:
        normalized = observe(obs, cfg)
        require(normalized.observation == obs and normalized.configuration == cfg,
                'Task input contains non-deployment observation fields')
    else:
        require(obs == turn_view[0] and cfg == turn_view[1],
                'Callbacks do not share the same turn observation')
    # Python equality admits True == 1. Recheck cheap decision-relevant types
    # on every callback and use its actual observed values, never a stale view.
    board, tpd = cfg['boardSize'], cfg['turnsPerDay']
    require(type(board) is int and board > 0 and type(tpd) is int and tpd > 0
            and type(cfg['episodeSteps']) is int and cfg['episodeSteps'] == 720
            and type(cfg['shedCapacity']) is int and cfg['shedCapacity'] >= 0
            and type(obs['day']) is int and type(obs['hour']) is int
            and (obs['day'], obs['hour']) == divmod(step, tpd),
            'Observation configuration or clock differs')
    farms = obs['farms']
    require(type(farms) is list and len(farms) == 2, 'Malformed public farms')
    for farm in farms:
        require(type(farm) is dict and type(farm['tiles']) is list
                and len(farm['tiles']) == board
                and all(type(row) is list and len(row) == board for row in farm['tiles'])
                and position(farm['farmer'], board) and type(farm['hands']) is list
                and all(position(p, board) for p in farm['hands']),
                'Malformed public map or units')
    own = farms[side]
    positions = [own['farmer'], *own['hands']]
    private = obs['private']
    require(type(private['inventories']) is list
            and len(private['inventories']) == len(positions), 'Own units/inventories differ')
    for inv in private['inventories']:
        inventory(inv)
    inventory(private['shed'])
    inventory(private['seeds'])
    previous_private = private if turn_view is None else turn_view[4]
    require(inputs['inventory_item_order'] == [list(inv) for inv in obs['private']['inventories']]
            and inputs['shed_item_order'] == list(obs['private']['shed'])
            and inputs['inventory_item_order'] == [list(inv) for inv in previous_private['inventories']]
            and inputs['shed_item_order'] == list(previous_private['shed']),
            'Observable inventory order differs')
    turn_view = (obs, cfg, own, positions, private)
    planned = inputs['planned']
    require(type(planned) is dict and set(planned) == PLANNED_KEYS
            and all(type(planned[k]) is dict for k in
                    ('seeds', 'needed_seeds', 'supply', 'projected_shed'))
            and type(planned['unit_actions']) is list
            and len(planned['unit_actions']) == len(positions)
            and type(planned['interpretation']) is str,
            'Malformed pre-assignment reservations')
    available = planned['available_units']
    require(type(available) is list
            and all(type(u) is int and 0 <= u < len(positions) for u in available)
            and len(set(available)) == len(available), 'Malformed available units')
    candidates = inputs['candidates']
    require(type(candidates) is list and candidates, 'Empty task callback candidates')
    by_id = {}
    for candidate in candidates:
        require(type(candidate) is dict and set(candidate) == {'id', 'unit', 'target', 'op', 'x'}
                and type(candidate['unit']) is int and candidate['unit'] in available
                and position(candidate['target'], cfg['boardSize']) and candidate['op'] in OPS
                and type(candidate['id']) is str
                and candidate['id'] == candidate_id(candidate['unit'], candidate['target'], candidate['op'])
                and candidate['id'] not in by_id, 'Malformed candidate identity')
        vector(candidate['x'], len(names))
        by_id[candidate['id']] = candidate
    return header, turn_view, by_id


def _eligible(candidate, original, view, first_yield_days):
    if (candidate['op'] != 'HARVEST' or candidate['id'] == original['id']
            or candidate['unit'] != original['unit']):
        return None
    obs, cfg, own, positions, private = view
    unit = candidate['unit']
    if positions[unit] != candidate['target']:
        return None
    x, y = candidate['target']
    tile = own['tiles'][y][x]
    if not isinstance(tile, dict) or tile.get('kind') != 'PLANT':
        return None
    crop, planted, amount = tile.get('crop'), tile.get('planted_day'), tile.get('yield_units')
    require(type(crop) is str and crop in first_yield_days
            and type(planted) is int and type(amount) is int and amount >= 0,
            'Malformed directly harvestable crop state')
    age = obs['day'] - planted
    if amount <= 0 or age < first_yield_days[crop]:
        return None
    # Official HARVEST appends to the unit inventory without a carrying cap.
    # Shed pressure is an annotation, never a qualification or economic mask.
    return dict(kind='PLANT', crop=crop, age=age, first_yield_day=first_yield_days[crop],
                yield_units=amount, original_op=original['op'], harvest_op='HARVEST',
                same_unit=True, at_current_position=True,
                carried_units=sum(private['inventories'][unit].values()),
                shed_units=sum(private['shed'].values()), shed_capacity=cfg['shedCapacity'],
                scope='static_observed_preconditions_not_execution_or_profit')


def select_target(rows, *, side, first_yield_days, guard=None):
    """Consume exactly 719 parsed rows; preserve the earliest qualifying point."""
    require(type(side) is int and side in (0, 1), 'Invalid trace seat')
    require(type(first_yield_days) is dict and first_yield_days
            and all(type(k) is str and k and type(v) is int and v >= 0
                    for k, v in first_yield_days.items()), 'Invalid authenticated crop maturity table')
    require(guard is None or callable(guard), 'Invalid selection resource guard')
    check = guard if guard is not None else lambda: None
    counts = dict(trace_lines=0, callbacks_checked=0, window_callbacks=0,
                  qualifying_callbacks=0, qualifying_candidates=0)
    target = eligibility = header = configuration = None
    for step, row in enumerate(rows):
        check()
        require(step < ACTION_TURNS and type(row) is dict
                and type(row.get('step')) is int and row['step'] == step
                and type(row.get('records')) is list, 'Malformed or discontinuous trace row')
        counts['trace_lines'] += 1
        turn_view = None
        for ordinal, record in enumerate(row['records']):
            check()
            require(type(record) is dict and type(record.get('inputs')) is dict,
                    'Malformed task record')
            try:
                header, turn_view, by_id = _inputs(
                    record['inputs'], step=step, side=side, header=header, turn_view=turn_view)
                if configuration is None:
                    configuration = turn_view[1]
                require(turn_view[1] == configuration, 'Configuration changed within trace')
                original = by_id[record['chosen']]
                require(type(record['unit']) is int and record['unit'] == original['unit']
                        and record['target'] == original['target'] and record['op'] == original['op']
                        and type(record['unit_request']) is list and bool(record['unit_request'])
                        and type(record['unit_request'][0]) is str,
                        'Chosen task metadata is inconsistent')
            except (KeyError, IndexError, TypeError, AttributeError):
                raise ValueError('Malformed task record or deployment input') from None
            counts['callbacks_checked'] += 1
            if not WINDOW[0] <= step <= WINDOW[1]:
                continue
            counts['window_callbacks'] += 1
            eligible = [(candidate, _eligible(candidate, original, turn_view, first_yield_days))
                        for candidate in sorted(by_id.values(), key=semantic)]
            eligible = [(candidate, info) for candidate, info in eligible if info is not None]
            counts['qualifying_candidates'] += len(eligible)
            counts['qualifying_callbacks'] += bool(eligible)
            if target is None and eligible:
                chosen, eligibility = eligible[0]
                target = dict(side=side, step=step, ordinal=ordinal,
                              input_sha256=ordered_hash(record['inputs']),
                              original_choice=original['id'], harvest_choice=chosen['id'])
    require(counts['trace_lines'] == ACTION_TURNS, 'Trace ended before all 719 rows')
    check()
    reason = (None if target is not None else 'no_task_callbacks_in_window'
              if counts['window_callbacks'] == 0 else
              'no_same_unit_direct_mature_harvest_alternative_in_window')
    return dict(target=target, counts=counts, reason=reason, eligibility=eligibility)
