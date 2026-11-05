"""One constructed assignment counterexample: four official action transitions.

The fixture scores are not neural outputs. This tests action expression through
the existing TaskAgent and fixed official engine, not model quality or profit.
Initialization and placeholder step counters are not simulated match history.
"""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time


REPO = Path(__file__).resolve().parents[3]
ENGINE_COMMIT = 'f994f2050d58d84f649329c1e35ce09ada7007c5'
ENGINE_SHA = 'bc8a54879ef02c7ea64b8b333d6a976f0ea65c4949149d01f463f23bccee653e'
CORE_SHA = '0922c4599a1b6e0d8c3dadf06ae5297f98138d859d686f00daee6f36e6d45d0e'
SPEC_SHA = 'a82c89c1a2315b93f39775d8e025471a01b738647c9772658368ee6b1b6f4867'
START_STEP = 70
A, B = (3, 0), (4, 0)
SCORES = {'u0:x3:y0:WATER': 8.0, 'u1:x3:y0:WATER': 10.0,
          'u1:x4:y0:WATER': 9.0}
SOURCE_NAMES = (
    'learning/task_joint_decoder_v1/official_probe.py',
    'learning/task_joint_decoder_v1/decoder.py',
    'learning/task_choice_v1/agent.py', 'learning/task_choice_v1/features.py',
    'learning/bc_crop/agent.py', 'learning/bc_crop/features.py',
    'bots/teacher_fertilizer_terminal_v2.py', 'decision/api.py',
    'decision/verify_mechanisms.py', 'decision/execution.py',
)
SPECIFICATION = dict(
    constructed_not_natural_trajectory=True, seed_source='existing_public_mechanism_fixture',
    step=START_STEP, day=2, hour=22, episode_steps=720,
    units=[[2, 0], [3, 0]], targets=[list(A), list(B)], target_crop='WHEAT',
    target_planted_day=0, target_yield_units=1, target_consecutive_unwatered=1,
    target_watered_today=False, own_money=100, own_hires_today=1,
    other_unlocked_tiles='young_wheat_watered_today',
    own_seeds_shed_and_inventories_empty=True, scores=SCORES,
    arms=['greedy', 'joint'], action_steps_per_arm=2,
    max_action_transitions=4, max_initializations=2,
    no_model_forward=True, complete_game=False, future_return_evaluated=False,
    initial_available_bytes=6*1024**3, running_available_bytes=4*1024**3,
    wall_seconds=300, resource_checks='cooperative_at_major_operations',
    retries_allowed=False,
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def encoded(value):
    return (json.dumps(value, ensure_ascii=True, separators=(',', ':'),
                       allow_nan=False) + '\n').encode()


def plain(value):
    return json.loads(encoded(value))


def digest(value):
    return hashlib.sha256(encoded(value)).hexdigest()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    raw = encoded(value)
    with Path(path).open('xb') as stream:
        require(stream.write(raw) == len(raw), 'Incomplete probe evidence write')
    require(Path(path).read_bytes() == raw, 'Probe evidence bytes differ')
    return hashlib.sha256(raw).hexdigest()


def sources():
    return {name: sha(REPO/'kaggriculture'/name) for name in SOURCE_NAMES}


def fixture(make_fixture, rules, counts):
    require(counts['initializations_attempted'] < 2, 'Initialization budget exhausted')
    counts['initializations_attempted'] += 1
    environment = make_fixture()
    counts['initializations_returned'] += 1
    require(environment.interpreter is rules.interpreter
            and environment.configuration['episodeSteps'] == 720
            and environment.configuration['turnsPerDay'] == 24
            and environment.configuration['boardSize'] == 10,
            'Official fixture configuration differs')
    environment.state[0].observation.step = START_STEP
    for state in environment.state:
        state.observation.day, state.observation.hour = 2, 22
    farm = environment.state[0].observation.farms[0]
    private = environment.state[0].observation.private
    farm['farmer'], farm['hands'] = [2, 0], [[3, 0]]
    farm['money'], farm['hires_today'] = 100.0, 1
    private['inventories'], private['shed'] = [{}, {}], {}
    private['seeds'] = {crop: 0 for crop in rules.CROPS}
    for y, row in enumerate(farm['tiles']):
        for x, tile in enumerate(row):
            if tile != 'LOCKED':
                plant = rules._new_plant('WHEAT', 2, 24)
                plant.update(watered_today=True, consecutive_unwatered=0)
                row[x] = plant
    for x, y in (A, B):
        require(farm['tiles'][y][x] != 'LOCKED', 'Fixture target is not owned')
        plant = rules._new_plant('WHEAT', 0, 24)
        require(plant['yield_units'] == 1 and plant['consecutive_unwatered'] == 1,
                'Official plant defaults differ')
        farm['tiles'][y][x] = plant
    # Only framework bookkeeping: this is NOT a 71-frame replay.
    environment.steps = [None] * START_STEP + [deepcopy(environment.state)]
    return environment


def selector(mode, choose, captured, guard):
    require(mode in ('greedy', 'joint'), 'Unknown probe arm')

    def select(inputs):
        guard()
        require(not any(k in inputs for k in ('teacher_label', 'teacher_priority', 'chosen')),
                'Label exposed to probe selector')
        require('seed' not in inputs['configuration']
                and 'seed' not in inputs['observation']
                and not any(k in inputs['observation']['farms'][1]
                            for k in ('private', 'inventories', 'shed', 'seeds')),
                'Privileged fields exposed to probe selector')
        candidates = inputs['candidates']
        ids = [row['id'] for row in candidates]
        require(ids and len(set(ids)) == len(ids) and all(i in SCORES for i in ids),
                'Unexpected fixture candidate')
        scores = [SCORES[i] for i in ids]
        greedy = ids[max(range(len(ids)), key=lambda index: scores[index])]
        selection = choose(candidates, scores) if mode == 'joint' else None
        returned = selection.choice_id if selection is not None else greedy
        captured.append(dict(
            step=inputs['observation']['step'], candidate_ids=ids, fixture_logits=scores,
            original_greedy_choice=greedy, returned_choice=returned,
            matching_ids=None if selection is None else list(selection.matching_ids),
            input_sha256=digest(inputs), model_forwards=0,
        ))
        return returned

    return select


def check_turn(mode, turn, request, captured):
    require(request == (
        {'farmer': ['PASS'], 'hands': [['WATER']], 'market': []}
        if mode == 'greedy' and turn == 0 else
        {'farmer': ['PASS'], 'hands': [['PASS']], 'market': []}
        if mode == 'greedy' else
        {'farmer': ['EAST'], 'hands': [['EAST']], 'market': []}
        if turn == 0 else
        {'farmer': ['WATER'], 'hands': [['WATER']], 'market': []}),
        'Predeclared official request differs')
    if turn == 0:
        require(set(captured[0]['candidate_ids']) == set(SCORES),
                'Initial graph is not the predeclared three edges')
        require(len(captured) == (1 if mode == 'greedy' else 2),
                'Unexpected initial selector count')
    else:
        require(len(captured) == (0 if mode == 'greedy' else 2),
                'Unexpected final-hour selector count')


def check_state(environment, mode, turn):
    step = START_STEP + turn + 1
    require(environment.state[0].observation.step == step
            and all(s.status == 'ACTIVE' for s in environment.state),
            'Official transition counter or status differs')
    require(all((s.observation.day, s.observation.hour) == divmod(step, 24)
                for s in environment.state), 'Official day counter differs')
    farm = environment.state[0].observation.farms[0]
    require(farm['money'] == 100, 'Fixture cash changed unexpectedly')
    if turn == 0:
        require(farm['farmer'] == ([2, 0] if mode == 'greedy' else [3, 0])
                and farm['hands'] == ([[3, 0]] if mode == 'greedy' else [[4, 0]]),
                'Official movement or occupancy differs')
        require(farm['tiles'][0][3]['watered_today'] == (mode == 'greedy')
                and not farm['tiles'][0][4]['watered_today'],
                'Unexpected first-step watering')
    else:
        require(farm['farmer'] == [4, 4] and farm['hands'] == [],
                'Official day-end unit reset differs')
        require(farm['tiles'][0][3]['kind'] == 'PLANT'
                and farm['tiles'][0][3]['consecutive_unwatered'] == 0,
                'Target A should survive in both arms')
        target = farm['tiles'][0][4]
        require(target == {'kind': 'WEED'} if mode == 'greedy' else
                target['kind'] == 'PLANT' and target['consecutive_unwatered'] == 0,
                'Predeclared target B survival differs')


def run(output):
    """Single attempt; keep four raw transition records even on assertion failure."""
    import psutil

    started = time.monotonic()
    checks = 0

    def guard(initial=False):
        nonlocal checks
        checks += 1
        require(time.monotonic() - started < SPECIFICATION['wall_seconds'],
                'Probe wall limit exceeded')
        threshold = SPECIFICATION['initial_available_bytes' if initial
                                  else 'running_available_bytes']
        require(psutil.virtual_memory().available >= threshold,
                'Probe memory floor not met')

    guard(initial=True)
    output = Path(output).absolute()
    require(output.parent.resolve(strict=True) == output.parent,
            'Probe parent must be an existing real directory')
    output.mkdir(exist_ok=False)
    frozen = sources()
    registration_sha = write(output/'registration.local.json', dict(
        schema='joint-decoder-official-micro-registration-v1',
        recorded_utc=datetime.now(timezone.utc).isoformat(),
        specification=SPECIFICATION, sources=frozen, engine_commit=ENGINE_COMMIT,
        engine_sha256=ENGINE_SHA, core_sha256=CORE_SHA, spec_sha256=SPEC_SHA,
        local_registration_not_third_party_timestamp=True,
    ))
    counts = dict(initializations_attempted=0, initializations_returned=0,
                  action_transitions_attempted=0, action_transitions_returned=0)
    write(output/'started.local.json', dict(
        registration_sha256=registration_sha, retry_allowed=False,
        max_action_transitions=4, max_initializations=2,
    ))
    error = None
    rows = []
    result_sha = None
    evidence_write_errors = []
    try:
        guard()
        from decision.verify_mechanisms import env as make_fixture, obs, rules, PASS
        from decision.execution import ExecutionTrace
        from kaggle_environments import core
        from learning.task_choice_v1.agent import TaskAgent
        from learning.task_joint_decoder_v1.decoder import choose
        require(sha(rules.__file__) == ENGINE_SHA and sha(core.__file__) == CORE_SHA
                and sha(Path(rules.__file__).with_suffix('.json')) == SPEC_SHA,
                'Official engine, core or specification identity differs')
        guard()
        initial_state_sha = None
        for mode in SPECIFICATION['arms']:
            guard(initial=True)
            environment = fixture(make_fixture, rules, counts)
            before = plain(environment.state)
            if initial_state_sha is None:
                initial_state_sha = digest(before)
            require(digest(before) == initial_state_sha, 'Arms do not share an initial state')
            captured = []
            actor = TaskAgent(selector(mode, choose, captured, guard))
            write(output/(mode+'-initial.local.json'), dict(
                state=before, configuration=plain(environment.configuration),
                info=plain(environment.info), fixture_scores_not_model=True,
                placeholder_prefix_not_replay=True,
            ))
            for turn in range(2):
                guard()
                captured.clear()
                request = actor(obs(environment), environment.configuration)
                write(output/(mode+'-plan-'+str(turn)+'.local.json'), dict(
                    arm=mode, turn=turn, action=plain(request),
                    selector_records=deepcopy(captured), fixture_scores_not_model=True,
                ))
                check_turn(mode, turn, request, captured)
                record = dict(
                    arm=mode, turn=turn, before=plain(environment.state),
                    action=plain(request), rival_action=deepcopy(PASS),
                    selector_records=deepcopy(captured), fixture_scores_not_model=True,
                    model_forwards=0, transition_returned=False,
                )
                trace = ExecutionTrace(rules)
                step_error = None
                require(counts['action_transitions_attempted'] < 4,
                        'Official action-transition budget exhausted')
                counts['action_transitions_attempted'] += 1
                try:
                    with trace.installed(environment):
                        environment.step([request, deepcopy(PASS)])
                        counts['action_transitions_returned'] += 1
                        record['transition_returned'] = True
                except BaseException as exc:
                    step_error = exc
                record.update(after=plain(environment.state), execution=plain(trace.records),
                              error_kind=None if step_error is None else type(step_error).__name__)
                try:
                    write(output/(mode+'-step-'+str(turn)+'.local.json'), record)
                except BaseException as exc:
                    evidence_write_errors.append(type(exc).__name__)
                    if step_error is None:
                        raise
                if step_error is not None:
                    raise step_error
                guard()
                check_state(environment, mode, turn)
                rows.append(dict(
                    arm=mode, turn=turn, action_transitions=1,
                    selector_calls=len(captured), model_forwards=0,
                    target_kinds=[environment.state[0].observation.farms[0]['tiles'][y][x]['kind']
                                  for x, y in (A, B)],
                ))
            del actor, environment
        require(counts['action_transitions_attempted'] == 4
                and counts['action_transitions_returned'] == 4
                and counts['initializations_returned'] == 2 and sources() == frozen,
                'Probe scope or frozen source differs')
        guard()
        result = dict(
            schema='joint-decoder-official-micro-result-v1', passed=True,
            registration_sha256=registration_sha, counts=counts, rows=rows,
            greedy_surviving_targets=1, joint_surviving_targets=2,
            initial_graph_edges=3, fixture_scores_not_model=True,
            complete_games=0, model_forwards=0, training=False, gpu=False,
            future_return_evaluated=False, promotion=False,
            conclusion='official_assignment_expression_counterexample_only',
        )
        result_sha = write(output/'result.local.json', result)
        return result
    except BaseException as exc:
        error = exc
        raise
    finally:
        closed = dict(
            schema='joint-decoder-official-micro-closed-v1',
            outcome='completed' if error is None else 'failed',
            error_kind=None if error is None else type(error).__name__,
            registration_sha256=registration_sha, result_sha256=result_sha,
            counts=counts, checked_transitions=len(rows), evidence_write_errors=evidence_write_errors,
            resource_check_calls=checks, elapsed_seconds=time.monotonic()-started,
            retry_allowed=False,
        )
        try:
            write(output/'closed.local.json', closed)
        except BaseException:
            if error is None:
                raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    # Local project modules, never another game checkout or new package install.
    package_root = str(REPO/'kaggriculture')
    if package_root not in sys.path:
        sys.path.insert(0, package_root)
    result = run(args.output)
    print(json.dumps({key: result[key] for key in (
        'passed', 'counts', 'greedy_surviving_targets', 'joint_surviving_targets',
        'complete_games', 'model_forwards', 'training', 'future_return_evaluated',
    )}, sort_keys=True))


if __name__ == '__main__':
    main()
