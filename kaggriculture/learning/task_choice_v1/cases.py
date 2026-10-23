"""Six predeclared synthetic boundary pairs, exactly twelve official transitions.

These are constructed legal-effect fixtures, not naturally reached trajectories,
trained policy demonstrations or economic comparisons. No fixture retry search.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

from decision.verify_mechanisms import env, obs, rules, PASS
from decision.execution import ExecutionTrace
from decision.semantics import canonical, digest
from bots.teacher_fertilizer_terminal_v2 import CROPS as TEACHER_CROPS
from learning.bc_crop.agent import sha
from .agent import TaskAgent

CASES = ('PLANT', 'DIG', 'WATER', 'HARVEST', 'FERTILIZE', 'joint_PLANT')


def fixture(kind):
    e = env()
    day = 28 if kind == 'HARVEST' else 7
    e.state[0].observation.step = day * 24
    for s in e.state:
        s.observation.day, s.observation.hour = day, 0
    f = e.state[0].observation.farms[0]
    private = e.state[0].observation.private
    f['farmer'], f['hands'] = [1, 1], []
    private['inventories'], private['shed'] = [{}], {}
    private['seeds'] = {c: 1 for c in rules.CROPS}
    # No irrelevant tasks: young, watered plants on previously unlocked tiles.
    for y, row in enumerate(f['tiles']):
        for x, tile in enumerate(row):
            if tile != 'LOCKED':
                t = rules._new_plant('WHEAT', day, 24)
                t['watered_today'] = True
                row[x] = t
    if kind in ('PLANT', 'DIG'):
        f['tiles'][1][1] = None if kind == 'PLANT' else {'kind': 'WEED'}
        t = rules._new_plant('CARROT', 4, 24)
        t.update(yield_units=3, watered_today=True)
        f['tiles'][1][2] = t
    elif kind in ('WATER', 'HARVEST', 'FERTILIZE'):
        planted = 0 if kind != 'HARVEST' else 20
        t = rules._new_plant('TOMATO', planted, 24)
        t['yield_units'] = 2 if kind == 'HARVEST' else 0
        f['tiles'][1][1] = t
        private['inventories'] = [{'FERTILIZER': 1}]
        # Public low stock ensures a productive fertilizer window is valuable.
        e.state[0].observation.market['inventory']['TOMATO'] = 9600
        if kind == 'FERTILIZE':
            f['hands'] = [[1, 1]]
            private['inventories'].append({'FERTILIZER': 1})
    elif kind == 'joint_PLANT':
        f['hands'] = [[1, 1]]
        private['inventories'].append({})
        f['tiles'][1][1], f['tiles'][1][2] = None, None
    else:
        raise ValueError(kind)
    # The official framework assigns the next observation.step from len(steps).
    # None entries align that counter only; they are NOT simulated history and
    # must never be exported or interpreted as a complete replay.
    e.steps = [None] * (day * 24) + [deepcopy(e.state)]
    return e


def choose(kind):
    first = True
    def callback(inputs):
        nonlocal first
        if first:
            first = False
            unit = 1 if kind in ('FERTILIZE', 'joint_PLANT') else 0
            op = 'PLANT' if kind == 'joint_PLANT' else kind
            matches = [c for c in inputs['candidates']
                       if c['unit'] == unit and c['target'] == [1, 1] and c['op'] == op]
            if len(matches) != 1:
                raise AssertionError('Predeclared alternative missing')
            return matches[0]['id']
        return inputs['candidates'][0]['id']
    return callback


def prepare_pair(kind):
    e = fixture(kind)
    observation, cfg = obs(e), e.configuration
    baseline, alternative = TaskAgent(), TaskAgent(choose(kind))
    a = baseline(deepcopy(observation), cfg)
    b = alternative(deepcopy(observation), cfg)
    assert a != b, 'Alternative must change complete requested action'
    expected = 'EAST' if kind in ('PLANT', 'DIG') else 'PLANT' if kind == 'joint_PLANT' else 'FERTILIZE'
    assert a['farmer'][0] == expected, ('Unexpected predeclared baseline', kind, a)
    target_unit = 1 if kind in ('FERTILIZE', 'joint_PLANT') else 0
    requested = [b['farmer'], *b['hands']][target_unit]
    assert requested[0] == ('PLANT' if kind == 'joint_PLANT' else kind)
    if kind == 'joint_PLANT':
        first, second = alternative.last['records'][:2]
        assert first['unit'] == 1 and second['unit'] == 0
        x = second['inputs']
        assert x['planned']['available_units'] == [0]
        assert all(c['unit'] == 0 and c['target'] != [1, 1] for c in x['candidates'])
        crop = b['hands'][0][1]
        assert x['planned']['seeds'][crop] == 0
        assert x['planned']['supply'][crop] == first['inputs']['planned']['supply'][crop] + TEACHER_CROPS[crop][3]
        # Seed consumption is unit-phase only; a later BUY_SEED may replenish it.
        assert sum(q[0] == 'PLANT' for q in [b['farmer'], *b['hands']]) == 1
        assert alternative.last['final_reservations']['needed_seeds'].get(crop, 0) == 1
    return e, a, b, baseline.last, alternative.last, target_unit


def run(out):
    from .experiment import ready, save
    out = Path(out)
    ready(out)
    if (out/'cases-started.json').exists():
        raise FileExistsError('Cases already started; no retry/fixture search')
    save(out/'cases-started.json', {'pairs': list(CASES), 'max_transitions': 12})
    rows, actual = [], 0
    try:
        for kind in CASES:
            initial, a, b, baseline, alternative, target_unit = prepare_pair(kind)
            before_hash = digest(initial.state)
            states, outcomes = [], []
            for branch, action in [('baseline', a), ('alternative', b)]:
                e = fixture(kind)
                assert digest(e.state) == before_hash
                trace = ExecutionTrace(rules)
                with trace.installed(e):
                    actual += 1
                    e.step([action, deepcopy(PASS)])
                # Local-only raw evidence for an independent reviewer. Public
                # certificates carry hashes and effects, never private states.
                raw_path = out/'cases-local'/(kind+'-'+branch+'.json')
                if raw_path.exists():raise FileExistsError('Preserve single-step evidence')
                save(raw_path, {'case':kind,'branch':branch,'configuration':dict(e.configuration),
                     'before':deepcopy(initial.state),'after':deepcopy(e.state),
                     'action':action,'rival_action':deepcopy(PASS),'trace':trace.records,
                     'counter_placeholders_are_not_simulated_history':True})
                assert not any(s.status in ('ERROR', 'INVALID', 'TIMEOUT') for s in e.state)
                after_step = initial.state[0].observation.step + 1
                assert e.state[0].observation.step == after_step
                for state in e.state:
                    assert (state.observation.day,state.observation.hour) == divmod(after_step,24)
                events = [x for x in trace.records if x['phase'] == 'unit' and x['player'] == 0]
                outcome = {'branch': branch, 'action': action,
                           'local_raw_sha256': sha(raw_path),
                           'state_sha256': digest(e.state),
                           'effect_sha256': trace.effect_hash.hexdigest(),
                           'units': [{'unit': x['unit'], 'requested': x['requested'],
                                      'effective': x['effective'], 'status': x['status'],
                                      'successful_plant': x.get('successful_plant', False)} for x in events]}
                if branch == 'alternative':
                    event = next(x for x in events if x['unit'] == target_unit)
                    assert event['status'] == 'executed'
                    if kind in ('PLANT', 'joint_PLANT'):
                        assert event.get('successful_plant') and sum(x.get('successful_plant', False) for x in events) == 1
                    elif kind == 'DIG':
                        assert event['tile_before']['kind'] == 'WEED' and event['tile_after'] is None
                    elif kind == 'WATER':
                        assert not event['tile_before']['watered_today'] and event['tile_after']['watered_today']
                    elif kind == 'HARVEST':
                        assert event['tile_before']['yield_units'] == 2 and event['tile_after']['yield_units'] == 0
                        assert dict(event['after']['inventories'][target_unit]).get('TOMATO', 0) == 2
                    elif kind == 'FERTILIZE':
                        assert dict(event['before']['inventories'][1])['FERTILIZER'] == 1
                        assert dict(event['after']['inventories'][1]).get('FERTILIZER', 0) == 0
                        assert dict(event['after']['inventories'][0])['FERTILIZER'] == 1
                        assert event['tile_after']['fertilized_until_day'] == 9
                    outcome['target_direct_official_effect_verified'] = True
                states.append(canonical(e.state))
                outcomes.append(outcome)
            assert states[0] != states[1]
            rows.append({'case': kind, 'synthetic_not_natural_trajectory': True,
                         'input_state_sha256': before_hash, 'branches': outcomes,
                         'first_teacher_label': baseline['records'][0]['teacher_label'],
                         'alternative_first_choice': alternative['records'][0]['chosen'],
                         'planned_second_unit_verified': kind == 'joint_PLANT'})
            save(out/'official-cases.json', {'passed': len(rows) == 6, 'pairs': len(rows),
                 'new_official_transitions': actual, 'new_complete_games': 0, 'rows': rows})
        assert actual == 12
        ready(out)
    except Exception as exc:
        save(out/'cases-failed.json', {'type': type(exc).__name__, 'completed_pairs': len(rows),
             'transitions_attempted': actual, 'automatic_retry_allowed': False})
        raise
