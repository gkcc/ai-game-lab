"""Describe exposed opportunities and submitted requests, never execution/profit.

One invocation consumes one authenticated trace. The caller owns its source ID,
source authentication, crop-source authentication, and publication projection.
No teacher label, future outcome, model, engine, or file is read here.
"""
from learning.task_harvest_window_v1 import selection as old


SCHEMA_VERSION = 'task-harvest-coverage-v1'
MOVES = frozenset(('NORTH', 'SOUTH', 'EAST', 'WEST'))

# Fixed output keys; descriptive denominators and partitions are in specification.
FIELDS = {
    'trace': ('rows', 'callbacks', 'window_turns', 'observed_window_turns',
              'unknown_no_callback_window_turns'),
    'callbacks': ('total', 'has_harvest', 'has_mature_positive_harvest',
                  'has_direct_mature_harvest', 'has_same_unit_direct_mature_harvest',
                  'has_same_unit_direct_alternative', 'chosen_is_direct_mature_harvest',
                  'only_other_unit_direct_mature_harvest'),
    'candidates': ('total', 'selected_at_first_seen', 'selected_later_same_unit',
                   'covered_by_other_unit', 'no_final_harvest_request'),
    'sites': ('total', 'request_at_first_seen', 'request_later',
              'no_final_harvest_request'),
    'expression': ('total', 'expressed', 'preassigned', 'available_unexpressed',
                   'only_other_unit_candidate'),
    'facts': ('direct_candidate_occurrences', 'harvest_task_move_assignments',
              'sites_with_harvest_task_move', 'sites_with_other_task'),
}


def empty_counts():
    """A new six-group zero dictionary, suitable for strict caller aggregation."""
    return {group: dict.fromkeys(names, 0) for group, names in FIELDS.items()}


def specification():
    """Exact JSON-compatible counting contract to bind before any source read."""
    return {
        'schema_version': SCHEMA_VERSION,
        'window': list(old.WINDOW), 'required_rows': old.ACTION_TURNS,
        'source_scope': 'one source per call; caller namespaces all keys by source',
        'fields': {group: list(names) for group, names in FIELDS.items()},
        'validation': [
            'old select_target validates all 719 rows and deployment inputs through EOF',
            'count each row only after old selection has validated that row',
            'all rows: final requests present; observed turns: one assignment per unit and tile',
            'available units remove exactly the prior selected unit at each callback',
            'claimed tiles cannot recur in later candidate pools',
            'available-unit planned requests are PASS; other planned requests match final requests',
            'record unit_request equals final unit request including JSON value types',
            'unassigned available units end with PASS; HARVEST requests must match direct HARVEST tasks',
            'no-callback rows lack a usable observation and remain unknown',
        ],
        'static_harvest_condition': {
            'crop': 'PLANT, positive integer yield_units, age >= authenticated official first_yield_day',
            'direct': 'observed unit position equals candidate target',
            'limits': 'static observed prerequisites only; no inventory/shed capacity or profitability mask',
        },
        'trace': {
            'rows': 'all source rows, including outside the fixed window',
            'callbacks': 'all records, including outside the fixed window',
            'window_turns': 'rows in the inclusive fixed window',
            'observed_window_turns': 'window rows with at least one validated callback observation',
            'unknown_no_callback_window_turns': 'window rows without callbacks; not zero physical opportunity',
            'partition': ['observed_window_turns', 'unknown_no_callback_window_turns'],
        },
        'callbacks': {
            'denominator': 'total = callbacks in the fixed window; every stage is a callback count',
            'nested_stages': ['has_harvest', 'has_mature_positive_harvest',
                              'has_direct_mature_harvest', 'has_same_unit_direct_mature_harvest',
                              'has_same_unit_direct_alternative'],
            'stage_meanings': [
                'any candidate op HARVEST',
                'any HARVEST candidate meets static crop condition',
                'any such candidate is direct',
                'any direct mature HARVEST candidate has the original chosen unit',
                'any such same-unit candidate differs from the original chosen ID',
            ],
            'chosen_is_direct_mature_harvest': 'original chosen ID belongs to direct mature HARVEST candidates',
            'only_other_unit_direct_mature_harvest': 'direct mature candidates exist but none uses the chosen unit',
            'nonpartition': 'stages and extra columns overlap and must not be summed',
        },
        'candidates': {
            'denominator': 'total = distinct direct mature HARVEST candidates seen in window callbacks',
            'dedup_key': ['source', 'step', 'unit', 'target', 'op'],
            'first_seen': 'earliest callback ordinal; semantic (unit,y,x,op,id) order within callback',
            'partition_priority': ['selected_at_first_seen', 'selected_later_same_unit',
                                   'covered_by_other_unit', 'no_final_harvest_request'],
            'selected_at_first_seen': 'same candidate chosen when first seen, final request HARVEST',
            'selected_later_same_unit': 'same candidate chosen at a later callback, final request HARVEST',
            'covered_by_other_unit': 'another unit starts on this site and finally requests HARVEST',
            'no_final_harvest_request': 'no starting-on-site unit finally requests HARVEST at this site',
        },
        'sites': {
            'denominator': 'total = distinct sites of direct mature HARVEST candidates, not all observed mature sites',
            'dedup_key': ['source', 'step', 'target'],
            'partition_priority': ['request_at_first_seen', 'request_later', 'no_final_harvest_request'],
            'request_at_first_seen': 'a unit at this site requests HARVEST at first candidate appearance callback',
            'request_later': 'a unit at this site requests HARVEST after first candidate appearance callback',
            'no_final_harvest_request': 'no final HARVEST request at this site',
        },
        'expression': {
            'denominator': 'total = unit-current-site pairs meeting static crop condition in first callback observation',
            'dedup_key': ['source', 'step', 'unit', 'current_target'],
            'partition_priority': ['expressed', 'preassigned', 'available_unexpressed'],
            'expressed': 'own direct mature HARVEST candidate appeared in any callback this turn',
            'preassigned': 'unexpressed and unit absent from first callback available_units',
            'available_unexpressed': 'unit available at first callback, own direct HARVEST never expressed; not continuously idle',
            'only_other_unit_candidate': 'overlapping attribute of unexpressed pairs only: mature HARVEST at same site appeared for another unit, possibly remote',
        },
        'facts': {
            'nonpartition': 'parallel facts have different explicit denominators and may overlap',
            'direct_candidate_occurrences': 'direct mature HARVEST candidate appearances across window callbacks, before dedup',
            'harvest_task_move_assignments': 'chosen HARVEST tasks in window whose final request is a cardinal MOVE; targets may lack direct candidates',
            'sites_with_harvest_task_move': 'among sites.total, a chosen HARVEST task targets site but final request is MOVE',
            'sites_with_other_task': 'among sites.total, a chosen non-HARVEST task targets site, even if request is MOVE/PASS',
        },
        'examples': 'first local reference for each nonzero counter; step then ordinal then semantic key; never publish or use as model input',
        'evidence_boundary': 'planning order and final requests only; no actual harvest, legality guarantee, cash, yield, or economic benefit',
        'ignored': ['teacher_label', 'teacher_score', 'future_cash', 'future_market', 'future_rewards'],
    }


def _requests(value, length=None):
    old.require(type(value) is list and bool(value)
                and (length is None or len(value) == length)
                and all(type(action) is list and action and type(action[0]) is str
                        for action in value), 'Malformed final or planned unit requests')
    # Type-sensitive identity also rejects non-JSON/nonfinite request contents.
    return [old.ordered_hash(action) for action in value]


def _dispatch(row):
    """Controller trace invariants only, not a replacement action interpreter."""
    final = row.get('final_reservations')
    old.require(type(final) is dict, 'Missing final reservations')
    actions = final.get('unit_actions')
    final_keys = _requests(actions)
    records = row['records']
    if not records:
        return None
    first = records[0]['inputs']
    obs = first['observation']
    own = obs['farms'][obs['player']]
    positions = [own['farmer'], *own['hands']]
    old.require(len(actions) == len(positions), 'Final unit count differs from observation')
    remaining = set(first['planned']['available_units'])
    assigned, claimed = {}, set()
    pass_key = old.ordered_hash(['PASS'])
    for ordinal, record in enumerate(records):
        inputs = record['inputs']
        planned = inputs['planned']
        old.require(set(planned['available_units']) == remaining,
                    'Available units do not follow one assignment per callback')
        planned_keys = _requests(planned['unit_actions'], len(positions))
        for unit in range(len(positions)):
            old.require(planned_keys[unit] == (pass_key if unit in remaining else final_keys[unit]),
                        'Pre-assignment requests disagree with reservations or final requests')
        old.require(all(tuple(candidate['target']) not in claimed for candidate in inputs['candidates']),
                    'A claimed tile reappeared in a later candidate pool')
        unit, target = record['unit'], tuple(record['target'])
        old.require(unit in remaining and unit not in assigned and target not in claimed,
                    'Unit or tile assigned more than once in a turn')
        old.require(old.ordered_hash(record['unit_request']) == final_keys[unit],
                    'Record unit_request differs from final unit request')
        if record['op'] == 'HARVEST':
            expected = ['HARVEST'] if positions[unit] == record['target'] else None
            old.require(actions[unit] == expected if expected is not None
                        else len(actions[unit]) == 1 and actions[unit][0] in MOVES,
                        'HARVEST task disagrees with direct or travel request')
        elif actions[unit][0] == 'HARVEST':
            raise ValueError('A non-HARVEST task produced a HARVEST request')
        assigned[unit] = (ordinal, record)
        claimed.add(target)
        remaining.remove(unit)
    for unit in remaining:
        old.require(final_keys[unit] == pass_key, 'Unassigned available unit changed its final request')
    requests = {}
    for unit, action in enumerate(actions):
        if action[0] != 'HARVEST':
            continue
        old.require(unit in assigned, 'HARVEST request has no task assignment')
        ordinal, record = assigned[unit]
        target = tuple(positions[unit])
        requests.setdefault(target, []).append((unit, ordinal))
    return obs, positions, assigned, requests


def _mature(obs, target, first_yield_days):
    x, y = target
    tile = obs['farms'][obs['player']]['tiles'][y][x]
    if type(tile) is not dict or tile.get('kind') != 'PLANT':
        return False
    crop, planted, amount = tile.get('crop'), tile.get('planted_day'), tile.get('yield_units')
    old.require(type(crop) is str and crop in first_yield_days
                and type(planted) is int and type(amount) is int and amount >= 0,
                'Malformed observed harvest crop state')
    return amount > 0 and obs['day'] - planted >= first_yield_days[crop]


def analyze(rows, *, side, first_yield_days, guard=None):
    """Consume one complete trace using old validation, then classify requests."""
    counts, examples = empty_counts(), {}

    def consume(row):
        context = _dispatch(row)  # Check final/assignment evidence outside the window too.
        records, step = row['records'], row['step']
        ref_cache = {}

        def reference(ordinal=None, candidate=None, unit=None, target=None):
            if ordinal is not None and ordinal not in ref_cache:
                ref_cache[ordinal] = old.ordered_hash(records[ordinal]['inputs'])
            result = dict(step=step, ordinal=ordinal, side=side,
                          input_sha256=ref_cache.get(ordinal))
            if candidate is not None:
                unit, target = candidate['unit'], candidate['target']
                result['candidate_id'] = candidate['id']
            if unit is not None:
                result['unit'] = unit
            if target is not None:
                result['target'] = list(target)
            return result

        def add(group, name, *, ordinal=None, candidate=None, unit=None, target=None, amount=1):
            if not amount:
                return
            counts[group][name] += amount
            key = group + '.' + name
            if key not in examples:
                examples[key] = reference(ordinal, candidate, unit, target)

        add('trace', 'rows')
        add('trace', 'callbacks', ordinal=0 if records else None, amount=len(records))
        if not old.WINDOW[0] <= step <= old.WINDOW[1]:
            return
        add('trace', 'window_turns')
        if context is None:
            add('trace', 'unknown_no_callback_window_turns')
            return
        add('trace', 'observed_window_turns', ordinal=0)
        obs, positions, assigned, requests = context
        direct_seen, sites, mature_units_by_site = {}, {}, {}
        for ordinal, record in enumerate(records):
            candidates = sorted(record['inputs']['candidates'], key=old.semantic)
            harvest = [c for c in candidates if c['op'] == 'HARVEST']
            mature = [c for c in harvest if _mature(record['inputs']['observation'],
                                                   c['target'], first_yield_days)]
            direct = [c for c in mature if positions[c['unit']] == c['target']]
            same = [c for c in direct if c['unit'] == record['unit']]
            different = [c for c in same if c['id'] != record['chosen']]
            chosen = [c for c in direct if c['id'] == record['chosen']]
            add('callbacks', 'total', ordinal=ordinal)
            for name, matches in (
                ('has_harvest', harvest), ('has_mature_positive_harvest', mature),
                ('has_direct_mature_harvest', direct), ('has_same_unit_direct_mature_harvest', same),
                ('has_same_unit_direct_alternative', different), ('chosen_is_direct_mature_harvest', chosen),
                ('only_other_unit_direct_mature_harvest', direct if not same else []),
            ):
                if matches:
                    add('callbacks', name, ordinal=ordinal, candidate=matches[0])
            for candidate in mature:
                target = tuple(candidate['target'])
                mature_units_by_site.setdefault(target, set()).add(candidate['unit'])
            for candidate in direct:
                target = tuple(candidate['target'])
                key = (candidate['unit'], target, candidate['op'])
                add('facts', 'direct_candidate_occurrences', ordinal=ordinal, candidate=candidate)
                direct_seen.setdefault(key, (ordinal, candidate))
                sites.setdefault(target, (ordinal, candidate))
            if record['op'] == 'HARVEST' and record['unit_request'][0] in MOVES:
                add('facts', 'harvest_task_move_assignments', ordinal=ordinal,
                    unit=record['unit'], target=record['target'])

        for (unit, target, _op), (ordinal, candidate) in direct_seen.items():
            add('candidates', 'total', ordinal=ordinal, candidate=candidate)
            chosen = assigned.get(unit)
            if (chosen is not None and chosen[1]['chosen'] == candidate['id']
                    and chosen[1]['unit_request'] == ['HARVEST']):
                old.require(chosen[0] >= ordinal, 'Candidate first appears after its own assignment')
                category = ('selected_at_first_seen' if chosen[0] == ordinal
                            else 'selected_later_same_unit')
            elif requests.get(target):
                old.require(all(request_unit != unit for request_unit, _ in requests[target]),
                            'Own harvest request does not match the expressed candidate')
                category = 'covered_by_other_unit'
            else:
                category = 'no_final_harvest_request'
            add('candidates', category, ordinal=ordinal, candidate=candidate)

        for target, (ordinal, candidate) in sites.items():
            add('sites', 'total', ordinal=ordinal, candidate=candidate)
            if requests.get(target):
                request_ordinal = min(i for _unit, i in requests[target])
                old.require(request_ordinal >= ordinal, 'Site first appears after its harvest assignment')
                category = 'request_at_first_seen' if request_ordinal == ordinal else 'request_later'
            else:
                category = 'no_final_harvest_request'
            add('sites', category, ordinal=ordinal, candidate=candidate)
            at_site = [record for record in records if tuple(record['target']) == target]
            if any(r['op'] == 'HARVEST' and r['unit_request'][0] in MOVES for r in at_site):
                add('facts', 'sites_with_harvest_task_move', ordinal=ordinal, candidate=candidate)
            if any(r['op'] != 'HARVEST' for r in at_site):
                add('facts', 'sites_with_other_task', ordinal=ordinal, candidate=candidate)

        initial_available = set(records[0]['inputs']['planned']['available_units'])
        for unit, position in enumerate(positions):
            if not _mature(obs, position, first_yield_days):
                continue
            target = tuple(position)
            add('expression', 'total', ordinal=0, unit=unit, target=target)
            if (unit, target, 'HARVEST') in direct_seen:
                category = 'expressed'
            else:
                category = 'preassigned' if unit not in initial_available else 'available_unexpressed'
                if mature_units_by_site.get(target, set()) - {unit}:
                    add('expression', 'only_other_unit_candidate', ordinal=0, unit=unit, target=target)
            add('expression', category, ordinal=0, unit=unit, target=target)

    def validated_tap():
        for row in rows:
            # The old consumer finishes validating this row before requesting
            # the next one. Even the final yielded row is consumed at EOF.
            yield row
            if guard is not None:
                guard()
            consume(row)

    original = old.select_target(validated_tap(), side=side,
                                 first_yield_days=first_yield_days, guard=guard)
    old.require(counts['trace']['rows'] == original['counts']['trace_lines']
                and counts['trace']['callbacks'] == original['counts']['callbacks_checked']
                and counts['callbacks']['total'] == original['counts']['window_callbacks']
                and counts['callbacks']['has_same_unit_direct_alternative']
                == original['counts']['qualifying_callbacks'], 'Coverage and old selection counts differ')
    for group in ('candidates', 'sites'):
        old.require(counts[group]['total'] == sum(v for k, v in counts[group].items() if k != 'total'),
                    'Request coverage partition does not reconcile')
    old.require(counts['expression']['total'] == sum(counts['expression'][key]
                for key in ('expressed', 'preassigned', 'available_unexpressed')),
                'Expression partition does not reconcile')
    return dict(schema_version=SCHEMA_VERSION, side=side, window=list(old.WINDOW),
                counts=counts, examples=examples, original_selection=original)
