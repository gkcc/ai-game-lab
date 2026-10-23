"""Bounded old-development regression: 24 policy calls, no engine or matches.

Run as a module with the project on PYTHONPATH; all paths derive from this file
or explicit arguments. No new holdout trace/replay is read by this check.
"""
import argparse
from copy import deepcopy
import gzip
import hashlib
import json
from pathlib import Path

from decision.api import observe
from learning.task_bc_v1.deployment import JointAgent
from .shadow import compare_turn, ordered_hash, RESERVATIONS


TASK_SHA = '9253bc05e632eec2d5a813557272655ab97e023c09843c60a80781b6cc2ab143'
CROP_SHA = 'd497e8284dde1f7b836b1cf88915822f82a96be1f99e7b126d06ea263e281ffd'
OLD_ROOT = 'artifacts/task-bc-v1-20261023-run1'


def stable(value):
    if isinstance(value, dict):
        return {k: stable(v) for k, v in value.items() if k != 'inference_seconds'}
    if isinstance(value, list):
        return [stable(v) for v in value]
    return value


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def synthetic_checks():
    planned = {key: {} for key in RESERVATIONS}
    planned.update(unit_actions=[['PASS']], available_units=[0])
    candidates = [dict(id='a', unit=0, target=[0, 0], op='DIG', x=[1.0]),
                  dict(id='b', unit=0, target=[1, 0], op='DIG', x=[2.0])]
    inputs = dict(shared_x=[1.0], candidates=candidates, planned=planned,
                  observation={'inventory': {'WHEAT': 1, 'CARROT': 2}}, configuration={},
                  inventory_item_order=[['WHEAT', 'CARROT']], shed_item_order=[])
    record = dict(inputs=inputs, chosen='a', teacher_label='a', unit=0,
                  target=[0, 0], op='DIG', unit_request=['EAST'])
    trace = dict(records=[record, deepcopy(record), deepcopy(record)],
                 final_reservations={key: {} for key in RESERVATIONS})
    action = dict(farmer=['EAST'], hands=[], market=[])
    other = deepcopy(trace)
    other['records'][0].update(chosen='b', target=[1, 0])
    other['records'][1]['inputs']['planned']['seeds']['WHEAT'] = 1
    other['records'][1]['inputs']['candidates'].reverse()
    other['records'][2]['chosen'] = 'b'  # Re-convergence must not be paired.
    result = compare_turn(action, trace, action, other)
    assert result['matched_context_calls'] == 1
    assert result['aligned_choice_differences'] == 1
    assert result['aligned_different_choice_same_unit_request'] == 1
    assert result['first_choice_difference']['factual_index'] == 0
    assert result['first_candidate_difference']['factual_index'] == 1
    assert result['first_reservation_difference']['factual_index'] == 1
    assert result['unmatched_task_calls'] == dict(factual=2, shadow=2)
    shorter = deepcopy(trace); shorter['records'] = shorter['records'][:1]
    assert compare_turn(action, trace, action, shorter)['alignment_stop_reason'] == 'path_length_difference'
    zero = deepcopy(trace); zero['records'] = []
    assert compare_turn(action, zero, action, zero)['zero_task_turn']
    assert compare_turn(action, zero, action, trace)['factual_zero_task_turn']
    altered_action = deepcopy(action); altered_action['market'] = [['BUY_SEED', 'WHEAT', 1]]
    different = compare_turn(action, zero, altered_action, zero)
    assert different['action_components_differ'] == dict(farmer=False, hands=False, market=True)
    assert different['first_action_difference'][0] == 'market'
    for field in RESERVATIONS:
        final = deepcopy(zero); final['final_reservations'][field] = {'changed': 1}
        assert compare_turn(action, zero, action, final)['final_reservations_differ'][field]
    assert ordered_hash({'a': 1, 'b': 2}) != ordered_hash({'b': 2, 'a': 1})
    assert ordered_hash(1.0) != ordered_hash(1.0 + 2 ** -52)
    for change in ('numeric_f64', 'candidate_identity', 'visible_item_order', 'planned_item_order'):
        final = deepcopy(trace)
        target = final['records'][0]['inputs']
        if change == 'numeric_f64': target['shared_x'][0] += 2 ** -52
        elif change == 'candidate_identity': target['candidates'][0]['target'] = [5, 5]
        elif change == 'visible_item_order': target['inventory_item_order'][0].reverse()
        else: target['planned'] = dict(reversed(list(target['planned'].items())))
        assert compare_turn(action, trace, action, final)['matched_context_calls'] == 0
    try:
        ordered_hash(float('nan'))
    except ValueError:
        pass
    else:
        raise AssertionError('Nonfinite hash input accepted')
    return dict(passed=True, context_precision='ordered typed f64 and semantic identity',
                post_divergence_ordinal_matching=False, path_length_is_execution=False)


def _old_observation(frame, side, configuration):
    # Same observation extraction as the established replay boundary; no audit
    # imports, engine imports, next-state values, or opponent private inputs.
    raw = deepcopy(frame[side]['observation'])
    for key in ('step', 'farms', 'market', 'town', 'day', 'hour'):
        raw[key] = deepcopy(frame[0]['observation'][key])
    view = observe(raw, configuration)
    return view.observation, view.configuration


def run(project_root=None):
    root = Path(project_root).resolve() if project_root else Path(__file__).resolve().parents[2]
    synthetic = synthetic_checks()
    old = root / OLD_ROOT
    registration = json.loads((old / 'registration.json').read_text(encoding='utf-8'))
    crop = root / registration['crop_checkpoint']
    task = old / 'model/trained.npz'
    assert sha(crop) == CROP_SHA and sha(task) == TASK_SHA
    ledger = json.loads((old / 'development-ledger.json').read_text(encoding='utf-8'))
    seeds = registration['validation_seeds']
    rows = [next(row for row in ledger if row['policy'] == 'trained'
                 and row['opponent'] == 'round2' and row['seed'] == seed and row['side'] == side)
            for seed in seeds for side in (0, 1)]
    assert len(rows) == 4
    observations, inputs_manifest = {}, {}
    # Bounded fixed positions on already consumed development observations.
    # Four game/seat contexts, a same-game next turn, then return to the first.
    sequence = [(0, 100), (0, 101), (1, 100), (2, 100), (3, 100), (0, 100)]
    for index, row in enumerate(rows):
        path = old / 'games' / row['game'] / 'replay.json.gz'
        assert sha(path) == row['replay_gzip_sha256']
        inputs_manifest[path.relative_to(root).as_posix()] = row['replay_gzip_sha256']
        with gzip.open(path, 'rt', encoding='utf-8') as stream:
            replay = json.load(stream)
        for step in (100, 101) if index == 0 else (100,):
            obs, cfg = _old_observation(replay['steps'][step], row['side'], replay['configuration'])
            observations[index, step] = (obs, cfg, deepcopy(replay['steps'][step+1][row['side']]['action']))
    calls, certificates = 0, []
    def make(head):
        return JointAgent(crop, CROP_SHA, task if head == 'trained' else None,
                          TASK_SHA if head == 'trained' else None)
    for head in ('teacher_task', 'trained'):
        reused = make(head)
        first_hash = None
        for index, step in sequence:
            obs, cfg, recorded_action = observations[index, step]
            before = ordered_hash([obs, cfg])
            fresh = make(head)
            assert calls + 2 <= 24
            actual = reused(deepcopy(obs), deepcopy(cfg)); calls += 1
            expected = fresh(deepcopy(obs), deepcopy(cfg)); calls += 1
            assert ordered_hash([obs, cfg]) == before
            assert ordered_hash(actual) == ordered_hash(expected), 'Fresh/reuse action differs'
            trace = stable(reused.last)
            assert ordered_hash(trace) == ordered_hash(stable(fresh.last)), 'Fresh/reuse trace differs'
            if head == 'trained':
                assert actual == recorded_action, 'Old recorded action does not reproduce'
            comparison = compare_turn(actual, trace, expected, stable(fresh.last))
            assert comparison['action_equal'] and comparison['first_context_difference'] is None
            digest = ordered_hash([actual, trace])
            if first_hash is None: first_hash = digest
            if (index, step) == sequence[0]: assert digest == first_hash
            certificates.append(dict(head=head, game=rows[index]['game'], side=rows[index]['side'],
                step=step, sanitized_input_sha256=before, action_trace_sha256=digest,
                task_calls=len(trace['records']), fresh_reuse_equal=True))
    return dict(schema='task-shadow-pre-freeze-verification-v1', passed=True,
        synthetic=synthetic, policy_calls=calls, official_transitions=0, matches=0,
        old_observation_only=True, new_holdout_private_reads=0,
        task_checkpoint_sha256=TASK_SHA, crop_checkpoint_sha256=CROP_SHA,
        input_files=inputs_manifest, checks=certificates,
        scope='Existing stateless TaskAgent only; no general stateful-policy equivalence claim')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = run(args.project_root)
    encoded = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
    if args.output:
        with args.output.open('x', encoding='utf-8') as stream:
            stream.write(encoded)
    else:
        print(encoded, end='')


if __name__ == '__main__':
    main()
