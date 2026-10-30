"""Run one control and one one-shot intervention through the official engine."""
import argparse
from contextlib import redirect_stderr, redirect_stdout
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import traceback
import types

FREEZE_SHA = '249d72b0770bb97cb4a1cc66fc14586dd674c778185f6aebf8c50c73fad9cb4d'
REGISTRATION_SHA = '293478899d30f47830add190e699b3077d0466806d2276cee0a4828b822bb245'
TRACE_VALIDATOR_SHA = '771eca5e1d0093fa05b9aee5c26ccb2414c87aed75903c21843bbf71e0e98553'
NUL_LAUNCHER_SHA = '1bb87548635ca57552608ca8c69257717e541c497ab307e8c7f9ad178a941cd5'
TRUSTED_IMPORTS_SHA = '329d3b6415e3fac1cabea5d27571db3510281cbabd9c5118c55cec9aa43b115a'
SOURCE_FILES = frozenset(('prepare.py', 'worker.py', 'runner.py', 'agent_adapter.py',
                          'intervention.py', 'trusted_imports.py'))


class QuietParser(argparse.ArgumentParser):
    def error(self, _message):
        raise ValueError('Invalid command arguments')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def same_typed(left, right):
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(same_typed(left[key], right[key]) for key in left)
    if isinstance(left, list):
        return len(left) == len(right) and all(same_typed(a, b) for a, b in zip(left, right))
    return left == right


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def read(path, expected=None, normalize=False):
    raw = path.read_bytes()
    if expected is not None:
        candidate = raw.replace(b'\r\n', b'\n') if normalize else raw
        require(sha(candidate) == expected, 'Input identity changed')
    return json.loads(raw.decode('utf-8-sig'))


def write_new(path, value):
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def write_success(stage, value):
    raw = (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n').encode('utf-8')
    pending, result = stage/'result.pending.json', stage/'result.json'
    with pending.open('xb') as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    require(pending.read_bytes() == raw and not result.exists(), 'Result write failed')
    os.replace(pending, result)


def check_source_manifest(private, gate):
    sources, wrappers = private.get('source_sha256'), private.get('wrapper_sha256')
    require(isinstance(sources, dict) and set(sources) == SOURCE_FILES
            and all(isinstance(value, str) and len(value) == 64
                    and all(c in '0123456789abcdef' for c in value) for value in sources.values())
            and isinstance(wrappers, dict) and set(wrappers) == {'control', 'intervene'}
            and all(isinstance(value, str) and len(value) == 64
                    and all(c in '0123456789abcdef' for c in value) for value in wrappers.values())
            and sources['trusted_imports.py'] == TRUSTED_IMPORTS_SHA
            and gate.get('adapter_source_sha256') == sources['agent_adapter.py']
            and gate.get('intervention_source_sha256') == sources['intervention.py'],
            'Experiment source manifest incomplete or inconsistent')


def verify_runtime(root, panel, stage, private, gate):
    cfg = read(panel/'registration.json', REGISTRATION_SHA, normalize=True)
    freeze = read(panel/'freeze.json', FREEZE_SHA, normalize=True)
    for relative, digest in freeze['runtime_files'].items():
        target = (root/relative).resolve(strict=True)
        require(target.is_relative_to(root) and target.is_file() and sha(target.read_bytes()) == digest,
                'Frozen runtime changed')
    for name in ('merged', 'crop'):
        target = (root/cfg[name+'_checkpoint']).resolve(strict=True)
        require(target.is_relative_to(root) and sha(target.read_bytes()) == cfg[name+'_checkpoint_sha256'],
                'Frozen model changed')
    require(sha((root/'learning/task_smoke_weight_probe_v1/core.py').read_bytes()) == TRACE_VALIDATOR_SHA,
            'Trace validation source changed')
    require(sha((root/'learning/panel_eval_v3/launch.py').read_bytes()) == NUL_LAUNCHER_SHA,
            'NUL process launcher source changed')
    installed = root/'.venv/Lib/site-packages/kaggle_environments'
    require(sha((installed/'envs/kaggriculture/kaggriculture.py').read_bytes()) == cfg['engine_sha256']
            and sha((installed/'core.py').read_bytes()) == cfg['official_core_sha256']
            and private['engine_sha256'] == cfg['engine_sha256'] and private['engine_commit'] == cfg['engine_commit'],
            'Official engine changed')
    check_source_manifest(private, gate)
    for name, digest in private['source_sha256'].items():
        require(sha((stage/'sources'/name).read_bytes()) == digest, 'Experiment source changed')
    require(private['source_sha256'].get('trusted_imports.py') == TRUSTED_IMPORTS_SHA,
            'Experiment import guard identity changed')
    for mode in ('control', 'intervene'):
        require(sha((stage/'agents'/(mode+'.py')).read_bytes()) == private['wrapper_sha256'][mode],
                'Experimental wrapper changed')
    require(sha((stage/'opponents/market.py').read_bytes()) == private['opponent_sha256']
            and private['opponent_sha256'] == cfg['source_sha256']['market.py']
            and gate['crop_sha256'] == cfg['crop_checkpoint_sha256']
            and gate['task_sha256'] == cfg['merged_checkpoint_sha256']
            and gate['limits'] == cfg['game_limits'], 'Frozen policy configuration differs')
    return cfg


def nul_api():
    """Reuse the existing explicit inherited-handle-list NUL launcher."""
    from learning.task_training_job_supervision_v1.job import WindowsAPI
    from learning.panel_eval_v3.launch import WindowsJob
    class NulWindowsAPI(WindowsAPI):
        def create_suspended(self, command, cwd):
            process, thread, _pid = WindowsJob().create_suspended(command, cwd)
            return process, thread
    return NulWindowsAPI()


def launch(root, stage, mode, limits):
    from learning.task_job_empty_exit_v1.sampling import supervise
    command = [str(root/'.venv/Scripts/python.exe'), '-B', '-I', str(stage/'sources/worker.py'),
               '--stage', str(stage), '--game-root', str(root), '--mode', mode]
    outcome = supervise(nul_api(), command, root.parent, limits)
    write_new(stage/('job-'+mode+'.local.json'), outcome)
    require(outcome.get('passed') is True and outcome.get('launcher_reaped') is True
            and outcome.get('job_members_zero') is True and outcome.get('exit_code') == 0,
            'Game supervision failed')
    require(read(stage/('worker-'+mode+'.done.json')) ==
            dict(passed=True, mode=mode, states=720, action_turns=719, agent_errors=0), 'Worker did not finish')


def replay_file(path, *, compressed=False, expected=None):
    with path.open('rb') as stream:
        raw = stream.read(128*2**20+1)
    require(len(raw) <= 128*2**20, 'Replay exceeds input budget')
    if expected is not None:
        require(sha(raw) == expected, 'Historical replay identity changed')
    if compressed:
        with gzip.GzipFile(fileobj=io.BytesIO(raw), mode='rb') as stream:
            data = stream.read(128*2**20+1)
    else:
        data = raw
    require(len(data) <= 128*2**20, 'Replay exceeds decoded budget')
    value = json.loads(data.decode('utf-8'))
    require(isinstance(value.get('steps'), list) and len(value['steps']) == 720, 'Incomplete replay')
    return value


def _orders(frame, player):
    private = frame[player]['observation']['private']
    return ([list(inv.items()) for inv in private['inventories']], list(private['shed'].items()))


def same_frame(left, right, canonical):
    return (canonical(left) == canonical(right)
            and all(_orders(left, p) == _orders(right, p) for p in (0, 1)))


def trace_control(history, current, expected_sha, ordered_hash):
    require(sha(history.read_bytes()) == expected_sha, 'Historical trace identity changed')
    target = None
    with gzip.open(history, 'rt', encoding='utf-8') as a, gzip.open(current, 'rt', encoding='utf-8') as b:
        for step in range(719):
            left, right = a.readline(), b.readline()
            require(left and right, 'Trace is incomplete')
            old, new = json.loads(left), json.loads(right)
            require(old.get('step') == new.get('step') == step and ordered_hash(old) == ordered_hash(new),
                    'Control trace differs from history')
            if step == 2:
                target = old
        require(not a.readline() and not b.readline() and target is not None, 'Trace has extra turns')
    return target


def trace_prefix(control, intervene, gate, ordered_hash):
    target_a = target_b = None
    with gzip.open(control, 'rt', encoding='utf-8') as a, gzip.open(intervene, 'rt', encoding='utf-8') as b:
        for step in range(3):
            left, right = a.readline(), b.readline()
            require(left and right, 'Intervention trace prefix missing')
            old, new = json.loads(left), json.loads(right)
            require(old.get('step') == new.get('step') == step, 'Intervention trace step differs')
            if step < 2:
                require(ordered_hash(old) == ordered_hash(new), 'Trace diverged before target')
            else:
                target_a, target_b = old, new
    a, b = target_a['records'], target_b['records']
    require(len(a) > 5 and len(b) > 5, 'Intervention target absent')
    for index in range(6):
        require(ordered_hash(a[index]['inputs']) == ordered_hash(b[index]['inputs']), 'Target inputs differ')
        require(a[index]['teacher_label'] == b[index]['teacher_label'], 'Conditional teacher differs')
        if index < 5:
            require(a[index]['chosen'] == b[index]['chosen'], 'Earlier selection differs')
    expected = gate['expected']
    require(ordered_hash(a[5]['inputs']) == expected['input_sha256']
            and a[5]['chosen'] == expected['original_choice']
            and b[5]['chosen'] == b[5]['teacher_label'] == expected['teacher_choice'], 'Target choice mismatch')
    candidates = {x['id']: x for x in a[5]['inputs']['candidates']}
    return {'model_worker': candidates[expected['original_choice']]['unit'],
            'teacher_worker': candidates[expected['teacher_choice']]['unit']}


def check_proofs(stage, gate):
    a = read(stage/'events/control.target.local.json')
    b = read(stage/'events/intervene.target.local.json')
    expected = gate['expected']
    for record, mode in ((a, 'control'), (b, 'intervene')):
        require(record['mode'] == mode and record['step'] == 2 and record['selection_record_index'] == 5
                and record['input_sha256'] == expected['input_sha256']
                and record['original_choice'] == expected['original_choice']
                and record['teacher_choice'] == expected['teacher_choice']
                and record['teacher_label_verified'] is True, 'Target proof differs')
    require(a['returned_choice'] == expected['original_choice'] and b['returned_choice'] == expected['teacher_choice']
            and same_typed(a['budget_before'], b['budget_before'])
            and same_typed(a['budget_after'], b['budget_after']),
            'Target return or budget differs')


def check_control_proof(stage, gate, candidate_count):
    record = read(stage/'events/control.target.local.json')
    expected = gate['expected']
    require(record.get('mode') == 'control' and record.get('step') == 2
            and record.get('selection_record_index') == 5 and record.get('teacher_label_verified') is True
            and record.get('input_sha256') == expected['input_sha256']
            and record.get('original_choice') == record.get('returned_choice') == expected['original_choice']
            and record.get('teacher_choice') == expected['teacher_choice'], 'Control target proof differs')
    fields = {'task_forwards', 'task_rows', 'crop_forwards', 'crop_rows',
              'gate_evaluations', 'controller_calls', 'teacher_proposals'}
    before, after = record.get('budget_before'), record.get('budget_after')
    require(type(candidate_count) is int and candidate_count > 0 and isinstance(before, dict)
            and isinstance(after, dict) and set(before) == set(after) == {'total', 'turn'},
            'Control budget proof missing')
    for scope in ('total', 'turn'):
        b, a = before[scope], after[scope]
        require(isinstance(b, dict) and isinstance(a, dict) and set(b) == set(a) == fields
                and all(type(value) is int and value >= 0 for value in (*b.values(), *a.values())),
                'Control budget proof invalid')
        require(all(a[key]-b[key] == (1 if key == 'task_forwards' else candidate_count if key == 'task_rows' else 0)
                    for key in fields), 'Control target budget increment differs')


def validate_intervention_trace(path, cfg, root):
    """Check all 719 saved rows and budget totals without another model call."""
    raw = (root/'learning/task_smoke_weight_probe_v1/core.py').read_bytes()
    require(sha(raw) == TRACE_VALIDATOR_SHA, 'Trace validation source changed')
    module = types.ModuleType('frozen_trace_validator')
    exec(compile(raw, 'frozen-trace-validator', 'exec', dont_inherit=True), module.__dict__)
    def lines():
        with gzip.open(path, 'rt', encoding='utf-8') as stream:
            for line in stream:
                yield json.loads(line)
    retained = module.validate_trace(lines(), cfg['game_limits'], cfg['merged_checkpoint_sha256'], cfg['crop_checkpoint_sha256'])
    require(isinstance(retained, dict), 'Intervention trace validation failed')


def official_audit(stage, mode, replay, target_units, reexecute):
    effects = []
    def collect(event):
        if (event.get('phase') == 'unit' and event.get('player') == 0
                and event.get('step') == 2 and event.get('unit') in target_units.values()):
            effects.append(dict(unit=event['unit'], status=event['status'],
                                requested_op=(event['requested'][0] if isinstance(event.get('requested'), list)
                                              and event['requested'] else None),
                                requested_equals_effective=event['requested'] == event['effective']))
    with (stage/('audit-'+mode+'.local.log')).open('x', encoding='utf-8') as log:
        with redirect_stdout(log), redirect_stderr(log):
            receipt = reexecute(replay, trace=True, label_sink=collect)
    require(receipt.get('passed') is True and receipt.get('official_transitions_reexecuted') == 719
            and receipt.get('both_public_and_private_states_match_each_step') is True
            and receipt.get('inventory_order_checked_each_step') is True
            and {e['unit'] for e in effects} == set(target_units.values()) and len(effects) == len(target_units),
            'Official replay or target execution check failed')
    by_unit = {event['unit']: event for event in effects}
    return dict(transitions=719, roles={role: {key: by_unit[unit][key] for key in
                ('status', 'requested_op', 'requested_equals_effective')}
                for role, unit in target_units.items()})


def banks(replay, state):
    return [farm['money'] for farm in replay['steps'][state][0]['observation']['farms']]


def execute(root, panel, stage):
    if not __debug__:
        raise RuntimeError('Optimized Python is not allowed')
    private, gate = read(stage/'input-private.json'), read(stage/'gate.local.json')
    require(private.get('schema') == 'single-task-input-v1' and gate.get('schema') == 'single-task-gate-v1'
            and read(stage/'prepared.json').get('passed') is True, 'Experiment not prepared')
    cfg = verify_runtime(root, panel, stage, private, gate)
    trusted_raw = (stage/'sources/trusted_imports.py').read_bytes()
    require(sha(trusted_raw) == TRUSTED_IMPORTS_SHA, 'Experiment import guard changed')
    trusted = types.ModuleType('frozen_intervention_imports')
    exec(compile(trusted_raw, 'frozen-intervention-imports', 'exec', dont_inherit=True), trusted.__dict__)
    trusted.bind(root)
    from decision.semantics import canonical
    from decision.full_audit import reexecute
    from learning.task_bc_error_diagnostics_v1.shadow import ordered_hash
    from learning.task_request_gate_v1.policy import same_request

    write_new(stage/'run.started.json', dict(attempt=1, retry_allowed=False))
    old_replay_path = panel/'games'/private['historical_game']/'replay.json.gz'
    old_trace_path = panel/'traces'/(private['historical_game']+'.jsonl.gz')
    print(json.dumps({'stage': 'control_started'}), flush=True)
    launch(root, stage, 'control', cfg['official_job_supervision'])
    historical = replay_file(old_replay_path, compressed=True, expected=private['historical_replay_sha256'])
    control = replay_file(stage/'games/control/replay.json')
    require(historical.get('info', {}).get('seed') == control.get('info', {}).get('seed') == private['seed']
            and canonical(historical['configuration']) == canonical(control['configuration']), 'Control configuration differs')
    for step in range(720):
        require(same_frame(historical['steps'][step], control['steps'][step], canonical), 'Control state differs')
        if step:
            for side in (0, 1):
                require(same_request(historical['steps'][step][side]['action'],
                                     control['steps'][step][side]['action']), 'Control request differs')
    old_target = trace_control(old_trace_path, stage/'traces/control.jsonl.gz',
                               private['historical_trace_sha256'], ordered_hash)
    rec = old_target['records'][5]
    options = {c['id']: c for c in rec['inputs']['candidates']}
    target_units = {'model_worker': options[gate['expected']['original_choice']]['unit'],
                    'teacher_worker': options[gate['expected']['teacher_choice']]['unit']}
    require(len(set(target_units.values())) == 2, 'Expected different target workers')
    check_control_proof(stage, gate, len(rec['inputs']['candidates']))
    control_effect = official_audit(stage, 'control', control, target_units, reexecute)
    verify_runtime(root, panel, stage, private, gate)
    print(json.dumps({'stage': 'control_verified', 'replay_transitions': 719}), flush=True)

    print(json.dumps({'stage': 'intervention_started'}), flush=True)
    launch(root, stage, 'intervene', cfg['official_job_supervision'])
    intervene = replay_file(stage/'games/intervene/replay.json')
    require(intervene.get('info', {}).get('seed') == private['seed']
            and canonical(intervene['configuration']) == canonical(control['configuration']), 'Intervention configuration differs')
    for step in range(3):
        require(same_frame(control['steps'][step], intervene['steps'][step], canonical), 'Pre-intervention state differs')
        if step:
            for side in (0, 1):
                require(same_request(control['steps'][step][side]['action'],
                                     intervene['steps'][step][side]['action']), 'Pre-intervention request differs')
    require(same_request(control['steps'][3][1]['action'], intervene['steps'][3][1]['action']),
            'Opponent request differed at intervention')
    units = trace_prefix(stage/'traces/control.jsonl.gz', stage/'traces/intervene.jsonl.gz', gate, ordered_hash)
    require(units == target_units, 'Target worker identities differ')
    check_proofs(stage, gate)
    validate_intervention_trace(stage/'traces/intervene.jsonl.gz', cfg, root)
    components = ('farmer', 'hands', 'market')
    control_action, intervention_action = control['steps'][3][0]['action'], intervene['steps'][3][0]['action']
    action_difference = {key: not same_request(control_action[key], intervention_action[key]) for key in components}
    intervention_effect = official_audit(stage, 'intervene', intervene, target_units, reexecute)
    verify_runtime(root, panel, stage, private, gate)
    final_a, final_b = banks(control, 719), banks(intervene, 719)
    at24_a, at24_b = banks(control, 26), banks(intervene, 26)
    def summary(a, b):
        return dict(control_own=a[0], control_opponent=a[1], intervention_own=b[0], intervention_opponent=b[1],
                    own_delta=b[0]-a[0], opponent_delta=b[1]-a[1], margin_delta=(b[0]-b[1])-(a[0]-a[1]))
    return dict(schema='single-task-intervention-result-v1', passed=True, sample='historical consumed S01, market, seat 0',
        control_matches_historical_states=720, control_matches_historical_actions=719,
        control_matches_historical_traces=719, target_step=2, target_selection_index=5,
        target_input_equal=True, single_replacement=True, pre_intervention_budget_equal=True,
        action_components_differ=action_difference, control_target_execution=control_effect,
        intervention_target_execution=intervention_effect, cash_after_24_actions=summary(at24_a, at24_b),
        final_cash=summary(final_a, final_b),
        final_score=dict(control=1.0 if final_a[0]>final_a[1] else 0.0 if final_a[0]<final_a[1] else 0.5,
                         intervention=1.0 if final_b[0]>final_b[1] else 0.0 if final_b[0]<final_b[1] else 0.5),
        new_games=2, live_official_transitions=2*719, same_engine_replay_transitions=2*719,
        training_updates=0, gpu_operations=0, independent_seed_test=False,
        causal_scope='One fixed initial condition, frozen subsequent strategies, endogenous official random stream.')


def main():
    parser = QuietParser(description=__doc__)
    parser.add_argument('--game-root', required=True, type=Path)
    parser.add_argument('--panel', required=True, type=Path)
    parser.add_argument('--stage', required=True, type=Path)
    args = parser.parse_args()
    root, panel, stage = (p.resolve(strict=True) for p in (args.game_root, args.panel, args.stage))
    require(stage.parent == (root/'artifacts').resolve(strict=True) and not stage.is_relative_to(panel),
            'Invalid experiment stage')
    require(Path(__file__).resolve() == (stage/'sources/runner.py').resolve(strict=True),
            'Run the frozen experiment source')
    require(not any((stage/name).exists() for name in ('run.started.json', 'result.json', 'run.failure.json')),
            'Attempt already consumed')
    try:
        result = execute(root, panel, stage)
        write_success(stage, result)
    except BaseException as exc:
        try:
            write_new(stage/'run.failure.json', dict(passed=False, error_type=type(exc).__name__, retry_allowed=False))
            with (stage/'run.failure.local.log').open('x', encoding='utf-8') as log:
                log.write(traceback.format_exc())
        finally:
            print(json.dumps({'passed': False, 'error_type': type(exc).__name__}), flush=True)
        raise SystemExit(1) from None
    print(json.dumps({'passed': True, 'new_games': 2, 'official_transitions': 4*719}), flush=True)


if __name__ == '__main__':
    try:
        main()
    except SystemExit:
        raise
    except BaseException as exc:
        print(json.dumps({'passed': False, 'error_type': type(exc).__name__}), flush=True)
        raise SystemExit(1) from None
