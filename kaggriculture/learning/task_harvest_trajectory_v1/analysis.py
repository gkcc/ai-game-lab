"""One-pass, lightweight explanation of already certified 48-step event streams.

No environment, model, source-file IO or rule simulation is performed here.
Only the target step has atomic execution evidence. Later requests, state
changes and cash changes remain distinct observations, not fill/cause labels.
"""
from collections import Counter
from copy import deepcopy
from fractions import Fraction
import math

from learning.task_single_intervention_v1.agent_adapter import _same


HORIZON = 48
PRODUCTS = ('WHEAT', 'CARROT', 'TOMATO', 'STRAWBERRY', 'MELON', 'EGG', 'MILK', 'WOOL', 'FERTILIZER')
CROPS = PRODUCTS[:5]
UNIT_OPS = ('NORTH', 'SOUTH', 'EAST', 'WEST', 'PASS', 'WATER', 'HARVEST', 'PLANT',
            'FERTILIZE', 'DROP', 'PICKUP', 'PLACE', 'DIG', 'BUILD_COOP', 'BUILD_PASTURE',
            'FEED', 'CARE', 'COLLECT_FERTILIZER')
MARKET_OPS = ('SELL', 'BUY_PRODUCT', 'BUY_SEED', 'BUY_ANIMAL', 'HIRE', 'BUY_LAND')
TILE_FIELDS = ('crop', 'planted_day', 'yield_units', 'watered_today', 'consecutive_unwatered',
               'max_lifespan_step', 'fertilized_until_day')
FIRST_KEYS = ('own_request', 'opponent_request', 'original_own_request', 'followup_request',
              'own_position', 'opponent_position', 'own_inventory', 'opponent_inventory',
              'target_tile', 'own_cash', 'opponent_cash', 'margin_cash')
LIFECYCLE_FLAGS = ('plant_removed_or_replaced_observed', 'new_plant_observed',
                   'same_plant_yield_increased_observed', 'same_plant_yield_decreased_observed',
                   'target_tile_changed_observed')


def require(value, message='Trajectory input or accounting differs'):
    if not value:
        raise ValueError(message)


def specification():
    return dict(schema='task-harvest-trajectory-analysis-v1', horizon=48,
        event_order=['control:arm_start,prefix,suffix,arm_end', 'intervene:arm_start,prefix,suffix,arm_end'],
        prefix_scope='validate phase, count and consecutive step only; not analysis samples',
        frame_step_source='frame[0].observation.step; other observation.step is not assumed shared',
        request_equality='exact JSON scalar types, dictionary key order ignored, all array order retained',
        atomic_harvest='target execution inventory crop increase; effective HARVEST only; tile amount cross-check',
        later_execution='unknown; request and before/after observation are separate',
        cash='exact rational sum of recorded finite JSON numeric balances; no inventory valuation',
        top_three='absolute incremental(intervention own-minus-opponent minus control own-minus-opponent)',
        tie_break='earlier offset', top_three_includes_zero_increments=True,
        request_step_offset='0..47', state_offset='0..48; after request i is state i+1',
        first_followup_request='first offset greater than zero with either player request different',
        scheduling_counts='all unit requested operations per arm/player; cross-arm op-pairs split offset0 vs1..47',
        no_market_fills_inferred=True, no_causal_percentages=True,
        public_excludes=['unit ids', 'plot coordinates', 'raw private inventories', 'input hashes',
                         'private file digests', 'raw traces', 'raw frames', 'seed values'],
        training=False, new_official_transitions=0, formal_labels=False)


def _numeric(value):
    require(type(value) in (int, float) and math.isfinite(value), 'Nonfinite or nonnumeric amount')
    return value


def _integer(value, minimum=None):
    require(type(value) is int and (minimum is None or value >= minimum), 'Expected an exact integer')
    return value


def _json(value):
    kind = type(value)
    if kind is dict:
        require(all(type(k) is str for k in value), 'Non-string JSON key')
        for child in value.values():
            _json(child)
    elif kind is list:
        for child in value:
            _json(child)
    elif kind is float:
        _numeric(value)
    else:
        require(kind in (str, int, bool, type(None)), 'Unsupported JSON type')
    return value


def same_request(left, right):
    _json(left)
    _json(right)
    return _same(left, right)


def _number(value):
    return value.numerator if value.denominator == 1 else float(value)


def _cash_delta(before, after):
    return [_number(Fraction(b)-Fraction(a)) for a, b in zip(before, after)]


def _mapping(value):
    require(type(value) is dict and all(type(k) is str for k in value))
    return {key: _numeric(amount) for key, amount in value.items()}


def _pairs(value):
    require(type(value) is list and all(type(pair) is list and len(pair) == 2
        and type(pair[0]) is str for pair in value))
    require(len({pair[0] for pair in value}) == len(value), 'Duplicate inventory item')
    return {key: _numeric(amount) for key, amount in value}


def _position(value):
    require(type(value) is list and len(value) == 2)
    return [_integer(v, 0) for v in value]


def _tile(value):
    if value is None:
        return {'kind': 'EMPTY'}
    if type(value) is str:
        require(value == 'LOCKED')
        return {'kind': 'LOCKED'}
    require(type(value) is dict and value.get('kind') in ('PLANT', 'WEED', 'COOP', 'PASTURE'))
    result = {'kind': value['kind']}
    if value['kind'] == 'PLANT':
        require(value.get('crop') in CROPS and type(value.get('watered_today')) is bool)
        result.update({name: value[name] for name in TILE_FIELDS})
        for name in TILE_FIELDS:
            if name not in ('crop', 'watered_today'):
                _integer(result[name])
    elif 'animal' in value:
        require(value['animal'] in ('GOOSE', 'COW', 'SHEEP'))
        result.update(animal=value['animal'], yield_units=_integer(value['yield_units'], 0))
    return result


def _market(value):
    require(type(value) is dict and all(type(value.get(k)) is dict and set(value[k]) == set(PRODUCTS)
                                      for k in ('inventory', 'prices')))
    return {key: {item: _numeric(value[key][item]) for item in PRODUCTS}
            for key in ('inventory', 'prices')}


def _snapshot(frame, target, step):
    require(type(frame) is list and len(frame) == 2 and all(type(p) is dict
        and p.get('status') in ('ACTIVE', 'DONE') and type(p.get('observation')) is dict for p in frame))
    observation = frame[0]['observation']
    require(type(observation.get('step')) is int and observation['step'] == step)
    farms = observation['farms']
    require(type(farms) is list and len(farms) == 2)
    players = []
    for player, farm in zip(frame, farms):
        require(type(farm) is dict and type(farm['hands']) is list)
        positions = [_position(farm['farmer']), *[_position(p) for p in farm['hands']]]
        private = player['observation']['private']
        require(type(private) is dict and type(private['inventories']) is list
            and len(private['inventories']) == len(positions))
        players.append(dict(positions=positions,
            inventories=[list(_mapping(inv).items()) for inv in private['inventories']],
            shed=list(_mapping(private['shed']).items())))
    x, y = target['target']
    tile = farms[target['side']]['tiles'][y][x]
    return dict(step=step, cash=[_numeric(farm['money']) for farm in farms], players=players,
                tile=_tile(tile), market=_market(observation['market']))


def _request(value):
    require(type(value) is dict and set(value) == {'farmer', 'hands', 'market'}
        and all(type(value[k]) is list for k in value)
        and all(type(action) is list for action in value['hands']))
    _json(value)
    return deepcopy(value)


def _units(request):
    return [request['farmer'], *request['hands']]


def _op(action, *, market=False):
    allowed = MARKET_OPS if market else UNIT_OPS
    return action[0] if type(action) is list and action and type(action[0]) is str and action[0] in allowed else 'OTHER'


def _market_requests(orders):
    result = []
    for order in orders:
        op = _op(order, market=True)
        item = order[1] if type(order) is list and len(order) > 1 and order[1] in PRODUCTS else None
        quantity = order[2] if type(order) is list and len(order) > 2 else None
        numeric = type(quantity) in (int, float) and math.isfinite(quantity)
        result.append(dict(op=op, item=item, requested_quantity=quantity if numeric else None,
            quantity_kind='number' if numeric else 'missing_or_non_numeric'))
    return result


def _difference(before, after):
    return {key: _number(Fraction(after.get(key, 0))-Fraction(before.get(key, 0)))
            for key in sorted(set(before)|set(after)) if before.get(key, 0) != after.get(key, 0)}


def _unit_state_difference(before, after):
    result = []
    for unit in range(max(len(before['positions']), len(after['positions']))):
        old_pos = before['positions'][unit] if unit < len(before['positions']) else None
        new_pos = after['positions'][unit] if unit < len(after['positions']) else None
        old = before['inventories'][unit] if unit < len(before['inventories']) else []
        new = after['inventories'][unit] if unit < len(after['inventories']) else []
        if old_pos != new_pos or not _same(old, new):
            result.append(dict(unit=unit, position_before=old_pos, position_after=new_pos,
                position_changed=old_pos != new_pos, inventory_changed=not _same(old, new),
                inventory_delta=_difference(dict(old), dict(new)),
                inventory_order_changed=[k for k, _ in old] != [k for k, _ in new]))
    return result


def _request_difference(control, intervention):
    old, new = _units(control), _units(intervention)
    units = []
    for unit in range(max(len(old), len(new))):
        left, right = old[unit] if unit < len(old) else None, new[unit] if unit < len(new) else None
        if not same_request(left, right):
            units.append(dict(unit=unit, control=deepcopy(left), intervention=deepcopy(right),
                              control_op=_op(left), intervention_op=_op(right)))
    return dict(changed=not same_request(control, intervention), units=units,
        market_changed=not same_request(control['market'], intervention['market']),
        control_market_requests=_market_requests(control['market']),
        intervention_market_requests=_market_requests(intervention['market']))


def _atomic(record, event, target):
    require(type(record) is dict and record.get('phase') == 'unit'
        and type(record.get('step')) is int and record['step'] == target['step']
        and type(record.get('player')) is int and record['player'] == target['side']
        and type(record.get('unit')) is int and record['unit'] == target['unit']
        and record.get('position') == target['target']
        and record.get('status') in ('executed', 'blocked', 'no_effect', 'partial', 'pass'))
    unit = target['unit']
    submitted, original = _units(event['request'])[unit], _units(event['original_request'])[unit]
    require(same_request(record['requested'], submitted) and type(record.get('effective')) is list)
    before, after = _tile(record['tile_before']), _tile(record['tile_after'])
    crop = before.get('crop')
    require(crop in CROPS)
    inv_before = _pairs(record['before']['inventories'][unit])
    inv_after = _pairs(record['after']['inventories'][unit])
    gained = _number(Fraction(inv_after.get(crop, 0))-Fraction(inv_before.get(crop, 0)))
    harvested = gained if _op(record['effective']) == 'HARVEST' else 0
    if _op(record['effective']) == 'HARVEST':
        require(harvested >= 0)
        if harvested > 0:
            require(record['status'] == 'executed' and harvested == before['yield_units']
                and (after['kind'] == 'EMPTY' or after.get('crop') == crop and after.get('yield_units') == 0))
    return dict(original_op=_op(original), submitted_op=_op(submitted), effective_op=_op(record['effective']),
        status=record['status'], crop=crop, actual_harvest_units=harvested,
        atomic_crop_inventory_delta=gained, tile_before=before, tile_after=after,
        evidence='target atomic official execution record; not whole-turn inventory difference')


def _lifecycle(before, after):
    result = []
    old_plant, new_plant = before.get('kind') == 'PLANT', after.get('kind') == 'PLANT'
    same_plant = old_plant and new_plant and (before['crop'], before['planted_day']) == (after['crop'], after['planted_day'])
    if old_plant and not same_plant:
        result.append('plant_removed_or_replaced_observed')
    if new_plant and not same_plant:
        result.append('new_plant_observed')
    if same_plant and after['yield_units'] > before['yield_units']:
        result.append('same_plant_yield_increased_observed')
    if same_plant and after['yield_units'] < before['yield_units']:
        result.append('same_plant_yield_decreased_observed')
    if not _same(before, after):
        result.append('target_tile_changed_observed')
    return result


def _end_inventory(snapshot):
    result = []
    for player in snapshot['players']:
        total = Counter()
        for inventory in player['inventories']:
            total.update(dict(inventory))
        result.append(dict(carried_totals=dict(total), shed=dict(player['shed'])))
    return result


def analyze_case(events, case, guard=lambda: None):
    require(callable(guard) and type(case) is dict and case.get('passed') is True
        and case.get('schema') == 'task-harvest-unexpressed-case-v1'
        and case['row'].get('status') == 'paired' and case['row'].get('technical_passed') is True)
    source, target, row = case['source'], case['target'], case['row']
    start, side = _integer(target['step'], 0), _integer(target['side'], 0)
    require(side in (0, 1) and start+HORIZON <= 719 and source['side'] == side
        and source['role'] in ('F01', 'F02', 'F03', 'F04')
        and source['opponent'] in ('market', 'round2') and source['purpose'] in ('exploration', 'diagnostic'))
    _integer(target['unit'], 0)
    _position(target['target'])
    names = ('control', 'intervene')
    arm_index, phase, prefix, suffix, lines = 0, 'arm_start', 0, 0, 0
    data = {}
    for event in events:
        guard()
        lines += 1
        require(type(event) is dict and arm_index < 2 and event.get('arm') == names[arm_index])
        arm = names[arm_index]
        branch = case['core']['control' if arm == 'control' else 'intervention']
        if phase == 'arm_start':
            require(event.get('kind') == 'arm_start' and type(event.get('step')) is int and event['step'] == start)
            snap = _snapshot(event['frame'], target, start)
            require(snap['cash'] == branch['endpoints']['start']['cash'])
            data[arm] = dict(states=[snap], requests=[], turns=[], tile_changes=[], atomic=None)
            phase, prefix, suffix = 'prefix' if start else 'suffix', 0, 0
            continue
        current = data[arm]
        if phase == 'prefix':
            require(event.get('kind') == 'turn' and event.get('scope') == 'prefix'
                and type(event.get('step')) is int and event['step'] == prefix)
            prefix += 1
            if prefix == start:
                phase = 'suffix'
            continue
        if phase == 'suffix':
            require(event.get('kind') == 'turn' and event.get('scope') == 'suffix'
                and type(event.get('step')) is int and event['step'] == start+suffix)
            before, after = current['states'][-1], _snapshot(event['frame'], target, start+suffix+1)
            requests = {key: _request(event[key]) for key in ('request', 'original_request', 'rival_request')}
            if arm == 'control' or suffix > 0:
                require(same_request(requests['request'], requests['original_request']) and event.get('patch') is None)
            if suffix == 0:
                require(_same(event['target_execution'], branch['target_execution']))
                current['atomic'] = _atomic(event['target_execution'], requests, target)
            else:
                require(event.get('target_execution') is None)
            changes = _lifecycle(before['tile'], after['tile'])
            on_target = [dict(unit=i, op=_op(action)) for i, action in enumerate(_units(requests['request']))
                if i < len(before['players'][side]['positions'])
                and before['players'][side]['positions'][i] == target['target']]
            if changes or any(action['op'] == 'HARVEST' for action in on_target):
                current['tile_changes'].append(dict(offset=suffix, flags=changes,
                    before=before['tile'], after=after['tile'], requests_on_target=on_target,
                    target_unit_atomic_record_available=suffix == 0))
            delta = _cash_delta(before['cash'], after['cash'])
            current['turns'].append(dict(offset=suffix, action_step=start+suffix, after_state_step=start+suffix+1,
                cash_after=dict(own=after['cash'][side], opponent=after['cash'][1-side]),
                cash_increment=dict(own=delta[side], opponent=delta[1-side], margin=delta[side]-delta[1-side]),
                observed_unit_changes={label: _unit_state_difference(before['players'][p], after['players'][p])
                    for label, p in (('own', side), ('opponent', 1-side))},
                observed_shed_delta={label: _difference(dict(before['players'][p]['shed']), dict(after['players'][p]['shed']))
                    for label, p in (('own', side), ('opponent', 1-side))},
                requested_unit_operations=dict(own=[_op(a) for a in _units(requests['request'])],
                                               opponent=[_op(a) for a in _units(requests['rival_request'])]),
                requests_on_target=on_target, target_tile_changes=changes))
            current['states'].append(after)
            current['requests'].append(requests)
            suffix += 1
            if suffix == HORIZON:
                phase = 'arm_end'
            continue
        require(phase == 'arm_end' and event.get('kind') == 'arm_end'
            and _same(event.get('result'), branch))
        endpoint = _snapshot(event['frame'], target, start+HORIZON)
        require(_same(endpoint, current['states'][-1])
            and endpoint['cash'] == branch['endpoints']['end']['cash'])
        label = 'baseline' if arm == 'control' else 'intervention'
        for name, state in (('start', current['states'][0]), ('end', endpoint)):
            require(row[label][name] == dict(own=state['cash'][side], opponent=state['cash'][1-side]))
        for label, p in (('own', side), ('opponent', 1-side)):
            total = sum((Fraction(t['cash_increment'][label]) for t in current['turns']), Fraction())
            require(total == Fraction(endpoint['cash'][p])-Fraction(current['states'][0]['cash'][p]),
                    'Per-step cash does not reconcile to endpoint')
        arm_index, phase = arm_index+1, 'arm_start'
    require(arm_index == 2 and phase == 'arm_start' and lines == 2*(start+HORIZON+2)
        and case['events']['lines'] == lines and case['emitted_events'] == lines, 'Missing, repeated or extra events')
    require(_same(data['control']['states'][0], data['intervene']['states'][0]), 'Arms start from different projected states')
    paired, first = [], {}
    for offset in range(HORIZON):
        guard()
        control, intervention = data['control'], data['intervene']
        old, new = control['requests'][offset], intervention['requests'][offset]
        request_changes = dict(own=_request_difference(old['request'], new['request']),
            opponent=_request_difference(old['rival_request'], new['rival_request']),
            original_own=_request_difference(old['original_request'], new['original_request']))
        c_state, i_state = control['states'][offset+1], intervention['states'][offset+1]
        actual = {label: _unit_state_difference(c_state['players'][p], i_state['players'][p])
                  for label, p in (('own', side), ('opponent', 1-side))}
        cash = _cash_delta(c_state['cash'], i_state['cash'])
        cash_difference = dict(own=cash[side], opponent=cash[1-side], margin=cash[side]-cash[1-side])
        increments = {key: intervention['turns'][offset]['cash_increment'][key]
                          - control['turns'][offset]['cash_increment'][key] for key in ('own', 'opponent', 'margin')}
        entry = dict(offset=offset, cash_difference=cash_difference, cash_difference_increment=increments,
            requests=request_changes, observed_unit_differences=actual,
            target_tile_different=not _same(c_state['tile'], i_state['tile']))
        paired.append(entry)
        checks = {**{key+'_request': value['changed'] for key, value in request_changes.items()},
            'followup_request': offset > 0 and any(request_changes[k]['changed'] for k in ('own', 'opponent')),
            'own_position': any(x['position_changed'] for x in actual['own']),
            'opponent_position': any(x['position_changed'] for x in actual['opponent']),
            'own_inventory': any(x['inventory_changed'] for x in actual['own']),
            'opponent_inventory': any(x['inventory_changed'] for x in actual['opponent']),
            'target_tile': entry['target_tile_different'],
            **{key+'_cash': value != 0 for key, value in cash_difference.items()}}
        for key, condition in checks.items():
            if condition and key not in first:
                first[key] = offset
    cash_windows = {}
    for arm in names:
        states = data[arm]['states']
        delta = _cash_delta(states[0]['cash'], states[-1]['cash'])
        cash_windows[arm] = dict(own=delta[side], opponent=delta[1-side])
    delta = {key: cash_windows['intervene'][key]-cash_windows['control'][key] for key in ('own', 'opponent')}
    delta['margin'] = delta['own']-delta['opponent']
    require(delta == paired[-1]['cash_difference'])
    for key in delta:
        require(sum((Fraction(p['cash_difference_increment'][key]) for p in paired), Fraction()) == Fraction(delta[key]))
    top = []
    for point in sorted(paired, key=lambda p: (-abs(p['cash_difference_increment']['margin']), p['offset']))[:3]:
        offset = point['offset']
        top.append(dict(offset=offset, cash_difference_increment=point['cash_difference_increment'],
            cash_difference=point['cash_difference'], arms={arm: dict(
                cash_increment=data[arm]['turns'][offset]['cash_increment'],
                market_before=data[arm]['states'][offset]['market'],
                market_after=data[arm]['states'][offset+1]['market'],
                own_market_requests=_market_requests(data[arm]['requests'][offset]['request']['market']),
                opponent_market_requests=_market_requests(data[arm]['requests'][offset]['rival_request']['market']))
                for arm in names}))
    report = dict(schema='task-harvest-trajectory-case-v1',
        source={key: source[key] for key in ('role', 'opponent', 'side', 'purpose')},
        target=deepcopy(target), counts=dict(events=lines, prefix_skipped=2*start, suffix_transitions=96,
                                           arms=2, states_per_arm=49),
        baseline_window_cash=cash_windows['control'], intervention_window_cash=cash_windows['intervene'],
        delta=delta, combined_cash_delta=delta['own']+delta['opponent'],
        first_divergence={key: first.get(key) for key in FIRST_KEYS},
        arms={arm: dict(atomic=data[arm]['atomic'], turns=data[arm]['turns'],
            target_start=data[arm]['states'][0]['tile'], target_end=data[arm]['states'][-1]['tile'],
            tile_changes=data[arm]['tile_changes'], end_inventory=_end_inventory(data[arm]['states'][-1]))
            for arm in names}, paired_turns=paired, top_margin_increments=top,
        limitations=dict(later_atomic_execution_unknown=True, actual_market_fills_unknown=True,
            changes_can_occur_between_saved_frames=True, cash_not_causal_decomposition=True,
            inventory_not_valued_as_cash=True, not_formal_labels=True, training=False))
    return dict(private=report, public=public_projection(report))


def _public_cash(value, *, margin=False):
    keys = ('own', 'opponent', 'margin') if margin else ('own', 'opponent')
    return {key: _numeric(value[key]) for key in keys}


def _offset(value):
    if value is None:
        return None
    require(type(value) is int and 0 <= value < HORIZON)
    return value


def _flag(value):
    require(type(value) is bool)
    return value


def _operation_counts(operations):
    result = Counter()
    for op in operations:
        require(op in (*UNIT_OPS, 'OTHER'))
        result[op] += 1
    return dict(result)


def _public_tile(value):
    kind = value['kind']
    require(kind in ('EMPTY', 'LOCKED', 'PLANT', 'WEED', 'COOP', 'PASTURE'))
    if kind in ('EMPTY', 'LOCKED'):
        return {'kind': kind}
    return _tile(value)


def _public_order(order):
    require(order['op'] in (*MARKET_OPS, 'OTHER') and (order['item'] is None or order['item'] in PRODUCTS)
        and order['quantity_kind'] in ('number', 'missing_or_non_numeric'))
    quantity = None if order['requested_quantity'] is None else _numeric(order['requested_quantity'])
    return dict(op=order['op'], item=order['item'], requested_quantity=quantity, quantity_kind=order['quantity_kind'])


def _operation_pairs(changes):
    counts = Counter()
    for unit in changes['units']:
        left, right = unit['control_op'], unit['intervention_op']
        require(left in (*UNIT_OPS, 'OTHER') and right in (*UNIT_OPS, 'OTHER'))
        counts[left+'->'+right] += 1
    return dict(counts)


def public_projection(report):
    """Construct a new whitelist object; never recursively copy private records."""
    source = report['source']
    require(source['role'] in ('F01', 'F02', 'F03', 'F04') and source['opponent'] in ('market', 'round2')
        and type(source['side']) is int and source['side'] in (0, 1)
        and source['purpose'] in ('exploration', 'diagnostic'))
    arms = {}
    for arm in ('control', 'intervene'):
        data, atomic = report['arms'][arm], report['arms'][arm]['atomic']
        milestones = {}
        for event in data['tile_changes']:
            for flag in event['flags']:
                require(flag in LIFECYCLE_FLAGS)
                milestones.setdefault(flag, _offset(event['offset']))
            if any(x['op'] == 'HARVEST' for x in event['requests_on_target']):
                milestones.setdefault('harvest_requested_at_target', _offset(event['offset']))
        require(all(atomic[key] in (*UNIT_OPS, 'OTHER') for key in ('original_op', 'submitted_op', 'effective_op'))
            and atomic['status'] in ('executed', 'blocked', 'no_effect', 'partial', 'pass') and atomic['crop'] in CROPS)
        _numeric(atomic['actual_harvest_units'])
        arms[arm] = dict(atomic_target_step={key: atomic[key] for key in (
            'original_op', 'submitted_op', 'effective_op', 'status', 'crop', 'actual_harvest_units')},
            target_start=_public_tile(data['target_start']), target_end=_public_tile(data['target_end']),
            first_observed_milestones=milestones,
            own_target_crop_end_aggregate=dict(crop=atomic['crop'],
                carried=_numeric(data['end_inventory'][source['side']]['carried_totals'].get(atomic['crop'], 0)),
                shed=_numeric(data['end_inventory'][source['side']]['shed'].get(atomic['crop'], 0))),
            target_requested_units_by_operation=_operation_counts(action['op'] for turn in data['turns']
                for action in turn['requests_on_target']),
            all_unit_request_operation_counts={label: _operation_counts(op for turn in data['turns']
                for op in turn['requested_unit_operations'][label]) for label in ('own', 'opponent')},
            scope='unit requests and observed target changes; only first target action has atomic execution evidence')
    timeline = []
    for point in report['paired_turns']:
        timeline.append(dict(offset=_offset(point['offset']), cash_difference=_public_cash(point['cash_difference'], margin=True),
            cash_difference_increment=_public_cash(point['cash_difference_increment'], margin=True),
            request_changed={key: _flag(point['requests'][key]['changed']) for key in ('own', 'opponent', 'original_own')},
            changed_requested_unit_count={key: len(point['requests'][key]['units']) for key in ('own', 'opponent')},
            observed_position_difference_unit_count={key: sum(v['position_changed'] for v in point['observed_unit_differences'][key])
                for key in ('own', 'opponent')},
            observed_inventory_difference_unit_count={key: sum(v['inventory_changed'] for v in point['observed_unit_differences'][key])
                for key in ('own', 'opponent')},
            changed_requested_operations={label: _operation_pairs(point['requests'][label]) for label in ('own', 'opponent')},
            market_request_changed={label: _flag(point['requests'][label]['market_changed']) for label in ('own', 'opponent')},
            arms={arm: dict(cash_after=_public_cash(report['arms'][arm]['turns'][point['offset']]['cash_after']),
                cash_increment=_public_cash(report['arms'][arm]['turns'][point['offset']]['cash_increment'], margin=True))
                for arm in ('control', 'intervene')}))
    # top_margin_increments already contains only cash, public market values and
    # sanitized request summaries; explicitly reconstruct every nested level.
    top = []
    for point in report['top_margin_increments']:
        top.append(dict(offset=_offset(point['offset']), cash_difference=_public_cash(point['cash_difference'], margin=True),
            cash_difference_increment=_public_cash(point['cash_difference_increment'], margin=True), arms={arm: dict(
                cash_increment=_public_cash(point['arms'][arm]['cash_increment'], margin=True),
                market_before=_market(point['arms'][arm]['market_before']),
                market_after=_market(point['arms'][arm]['market_after']),
                own_market_requests=[_public_order(order) for order in point['arms'][arm]['own_market_requests']],
                opponent_market_requests=[_public_order(order) for order in point['arms'][arm]['opponent_market_requests']])
                for arm in ('control', 'intervene')}))
    op_pairs = {}
    for phase in ('target_step', 'followup'):
        op_pairs[phase] = {}
        for player in ('own', 'opponent'):
            total = Counter()
            for point in report['paired_turns']:
                if (point['offset'] == 0) is (phase == 'target_step'):
                    total.update(_operation_pairs(point['requests'][player]))
            op_pairs[phase][player] = dict(total)
    first_followup = _offset(report['first_divergence']['followup_request'])
    followup = (None if first_followup is None else dict(offset=first_followup,
        **{player: dict(operation_pairs=_operation_pairs(report['paired_turns'][first_followup]['requests'][player]),
            market_changed=_flag(report['paired_turns'][first_followup]['requests'][player]['market_changed']))
           for player in ('own', 'opponent')}))
    return dict(schema='task-harvest-trajectory-public-case-v1',
        **{key: source[key] for key in ('role', 'opponent', 'side', 'purpose')},
        counts={key: _integer(report['counts'][key], 0) for key in ('events', 'prefix_skipped', 'suffix_transitions', 'arms', 'states_per_arm')},
        baseline_window_cash=_public_cash(report['baseline_window_cash']),
        intervention_window_cash=_public_cash(report['intervention_window_cash']), delta=_public_cash(report['delta'], margin=True),
        combined_cash_delta=_numeric(report['combined_cash_delta']),
        first_divergence={key: _offset(report['first_divergence'][key]) for key in FIRST_KEYS},
        changed_request_operation_pairs=op_pairs, first_followup_request=followup,
        arms=arms, cash_and_difference_timeline=timeline, top_margin_increments=top,
        limitations=dict(later_atomic_execution_unknown=True, actual_market_fills_unknown=True,
            cash_not_causal_decomposition=True, inventory_not_valued_as_cash=True,
            not_formal_labels=True, training=False))
