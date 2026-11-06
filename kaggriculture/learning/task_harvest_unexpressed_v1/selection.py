"""Select the first unexpressed direct harvest from authenticated metadata.

No source file, model, engine, teacher label or future outcome is read here.
The caller authenticates a complete source and the official maturity table.
The first callback is a binding observation, not a proposed network callback.
Final requests exclude sites already covered by any unit in this same turn.
Availability means available at the first callback, not idle for the whole turn.
"""
from copy import deepcopy

from learning.task_harvest_coverage_v1 import coverage
from learning.task_harvest_window_v1 import selection as old


WINDOW = old.WINDOW
ACTION_TURNS = old.ACTION_TURNS
SCHEMA_VERSION = 'task-harvest-unexpressed-selection-v1'
TARGET_SCHEMA = 'task-harvest-unexpressed-target-v1'
TARGET_KEYS = frozenset(('schema_version', 'side', 'step', 'ordinal', 'unit',
                         'target', 'input_sha256', 'original_unit_request',
                         'unit_requests_sha256'))
ordered_hash = old.ordered_hash
require = old.require


def _days(first_yield_days):
    require(type(first_yield_days) is dict and bool(first_yield_days)
            and all(type(k) is str and bool(k) and type(v) is int and v >= 0
                    for k, v in first_yield_days.items()),
            'Invalid authenticated crop maturity table')


def validate_target(target):
    """Check a frozen point without reading data or constructing an actor."""
    require(type(target) is dict and set(target) == TARGET_KEYS
            and target['schema_version'] == TARGET_SCHEMA, 'Malformed target fields')
    require(type(target['side']) is int and target['side'] in (0, 1)
            and type(target['step']) is int and WINDOW[0] <= target['step'] <= WINDOW[1]
            and type(target['ordinal']) is int and target['ordinal'] == 0
            and type(target['unit']) is int and target['unit'] >= 0,
            'Invalid target seat, turn, first callback or unit')
    point = target['target']
    require(type(point) is list and len(point) == 2
            and all(type(v) is int and v >= 0 for v in point), 'Invalid target position')
    for key in ('input_sha256', 'unit_requests_sha256'):
        value = target[key]
        require(type(value) is str and len(value) == 64
                and all(c in '0123456789abcdef' for c in value), 'Invalid target digest')
    action = target['original_unit_request']
    coverage._requests([action])
    require(action[0] != 'HARVEST', 'Target already requests HARVEST')
    return deepcopy(target)


def _opportunities(row, context, first_yield_days):
    """Yield unit/site pairs in (step, unit, y, x) order, after row validation."""
    if context is None or not WINDOW[0] <= row['step'] <= WINDOW[1]:
        return
    obs, positions, _assigned, requests = context
    first = row['records'][0]['inputs']
    available = set(first['planned']['available_units'])
    # Use the whole turn, including candidates appearing after its first callback.
    expressed = {(candidate['unit'], tuple(candidate['target']))
                 for record in row['records'] for candidate in record['inputs']['candidates']
                 if candidate['op'] == 'HARVEST'
                 and positions[candidate['unit']] == candidate['target']}
    actions = row['final_reservations']['unit_actions']
    binding = None
    for unit, position in enumerate(positions):
        point = tuple(position)
        if (unit not in available or (unit, point) in expressed or requests.get(point)
                or not coverage._mature(obs, position, first_yield_days)):
            continue
        if binding is None:
            binding = ordered_hash(first), ordered_hash(actions)
        target = dict(schema_version=TARGET_SCHEMA, side=obs['player'], step=row['step'],
                      ordinal=0, unit=unit, target=list(position), input_sha256=binding[0],
                      original_unit_request=deepcopy(actions[unit]),
                      unit_requests_sha256=binding[1])
        x, y = position
        tile = obs['farms'][obs['player']]['tiles'][y][x]
        private = obs['private']
        eligibility = dict(kind='PLANT', crop=tile['crop'],
            age=obs['day'] - tile['planted_day'], first_yield_day=first_yield_days[tile['crop']],
            yield_units=tile['yield_units'], initial_available=True,
            own_direct_harvest_expressed=False, site_final_harvest_requested=False,
            carried_units=sum(private['inventories'][unit].values()),
            shed_units=sum(private['shed'].values()), shed_capacity=first['configuration']['shedCapacity'],
            scope='static_observed_preconditions_not_execution_or_profit')
        yield target, eligibility


def select_target(rows, *, side, first_yield_days, guard=None):
    """Validate all 719 rows, keeping at most the first eligible unit/site.

    Request binding covers final unit actions, not market orders: this trace
    does not store the latter. The factual continuation separately authenticates
    the full original request against the source replay before any intervention.
    """
    target = eligibility = None
    counts = dict(observed_window_turns=0, unknown_no_callback_window_turns=0,
                  qualifying_turns=0, qualifying_unit_sites=0)

    def validated_rows():
        nonlocal target, eligibility
        for row in rows:
            # The frozen selector validates the yielded row before asking for
            # the next one. Our checks therefore also run on the final row at EOF.
            yield row
            if guard is not None:
                guard()
            context = coverage._dispatch(row)
            if not WINDOW[0] <= row['step'] <= WINDOW[1]:
                continue
            counts['observed_window_turns' if context is not None
                   else 'unknown_no_callback_window_turns'] += 1
            matches = list(_opportunities(row, context, first_yield_days))
            counts['qualifying_turns'] += bool(matches)
            counts['qualifying_unit_sites'] += len(matches)
            if target is None and matches:
                target, eligibility = matches[0]

    validated = old.select_target(validated_rows(), side=side,
                                  first_yield_days=first_yield_days, guard=guard)
    checked = {key: validated['counts'][key]
               for key in ('trace_lines', 'callbacks_checked', 'window_callbacks')}
    counts = dict(**checked, **counts)
    require(counts['observed_window_turns'] + counts['unknown_no_callback_window_turns']
            == WINDOW[1] - WINDOW[0] + 1, 'Selection window accounting differs')
    reason = (None if target is not None else 'no_task_callbacks_in_window'
              if counts['window_callbacks'] == 0 else
              'no_initial_available_unexpressed_uncovered_direct_harvest_in_window')
    return dict(schema_version=SCHEMA_VERSION, target=target, eligibility=eligibility,
                counts=counts, reason=reason)


def validate_target_trace(target, trace, *, first_yield_days):
    """Recheck one complete current intent trace, without changing that trace.

    This does not establish source identity or equality of the complete stable
    trace and replay request. Those are the continuation caller's obligations.
    """
    target = validate_target(target)
    _days(first_yield_days)
    require(type(trace) is dict and type(trace.get('step')) is int
            and trace['step'] == target['step'] and type(trace.get('records')) is list
            and bool(trace['records']), 'Target trace is missing or has another turn')
    header = turn_view = None
    for record in trace['records']:
        require(type(record) is dict and type(record.get('inputs')) is dict,
                'Malformed target task record')
        try:
            header, turn_view, candidates = old._inputs(record['inputs'], step=target['step'],
                side=target['side'], header=header, turn_view=turn_view)
            chosen = candidates[record['chosen']]
            require(type(record['unit']) is int and record['unit'] == chosen['unit']
                    and ordered_hash(record['target']) == ordered_hash(chosen['target'])
                    and record['op'] == chosen['op'], 'Chosen target task metadata differs')
        except (KeyError, IndexError, TypeError, AttributeError):
            raise ValueError('Malformed target task record or deployment input') from None
    context = coverage._dispatch(trace)
    for current, eligibility in _opportunities(trace, context, first_yield_days):
        if current['unit'] == target['unit']:
            # Metadata may be serialized with sorted keys. Field values retain
            # their type/order identity; the target envelope's key order is not
            # deployment input order and must not make a valid freeze unusable.
            require(all(ordered_hash(current[key]) == ordered_hash(target[key])
                        for key in TARGET_KEYS), 'Frozen target binding differs')
            return deepcopy(eligibility)
    raise ValueError('Frozen target no longer meets whole-turn selection conditions')
