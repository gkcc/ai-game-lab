"""Per-episode adapter; reads only the private one-shot gate and own observation."""
import hashlib
import json
from pathlib import Path
import types


def _relative(root, value):
    if not isinstance(value, str) or not value or '\\' in value or ':' in value:
        raise ValueError('Invalid project-relative path')
    path = Path(value)
    if path.is_absolute() or any(part in ('.', '..') for part in path.parts):
        raise ValueError('Invalid project-relative path')
    resolved = (root/path).resolve(strict=True)
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError('Checkpoint outside project')
    return resolved


def _write_new(path, value):
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def _same(left, right):
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(_same(left[key], right[key]) for key in left)
    if isinstance(left, list):
        return len(left) == len(right) and all(_same(a, b) for a, b in zip(left, right))
    return left == right


def _verify_control(control, intervention):
    """Compare only the earlier actor's own target proof before submitting B."""
    keys = ('step', 'selection_record_index', 'input_sha256', 'original_choice', 'teacher_choice',
            'budget_before', 'budget_after', 'teacher_label_verified')
    if (not isinstance(control, dict) or not isinstance(intervention, dict)
        or control.get('mode') != 'control' or intervention.get('mode') != 'intervene'
        or not all(key in control and key in intervention and _same(control[key], intervention[key]) for key in keys)
        or control.get('teacher_label_verified') is not True
        or control.get('returned_choice') != control.get('original_choice')
        or intervention.get('returned_choice') != intervention.get('teacher_choice')):
        raise ValueError('Control and intervention target proofs differ')


def build(stage, game_root, mode):
    """Create one independent actor; gate config contains no game seed."""
    from learning.task_dagger_once_v1.policy import BoundedAgent

    stage, game_root = Path(stage).resolve(strict=True), Path(game_root).resolve(strict=True)
    if mode not in ('control', 'intervene'):
        raise ValueError('Invalid arm')
    gate_data = json.loads((stage/'gate.local.json').read_text(encoding='utf-8'))
    keys = {'schema', 'expected', 'crop_checkpoint', 'crop_sha256', 'task_checkpoint',
            'task_sha256', 'limits', 'intervention_source_sha256', 'adapter_source_sha256'}
    if not isinstance(gate_data, dict) or set(gate_data) != keys or gate_data['schema'] != 'single-task-gate-v1':
        raise ValueError('Invalid gate configuration')
    raw = (stage/'sources/intervention.py').read_bytes()
    if hashlib.sha256(raw).hexdigest() != gate_data['intervention_source_sha256']:
        raise ValueError('Intervention source differs')
    module = types.ModuleType('frozen_single_intervention')
    exec(compile(raw, 'frozen-single-intervention', 'exec', dont_inherit=True), module.__dict__)
    actor = BoundedAgent(_relative(game_root, gate_data['crop_checkpoint']), gate_data['crop_sha256'],
        _relative(game_root, gate_data['task_checkpoint']), gate_data['task_sha256'],
        trace=stage/'traces'/(mode+'.jsonl.gz'), limits=gate_data['limits'])
    gate = module.OneShotSelector(original=actor.controller.task_selector,
        index_reader=lambda: len(actor.controller.records),
        budget_reader=lambda: {'total': dict(actor.budget.counts), 'turn': dict(actor.budget.turn)},
        expected=gate_data['expected'], mode=mode)
    actor.controller.task_selector = gate

    def agent(observation, configuration):
        request = actor(observation, configuration)
        gate.verify_turn(actor.last)
        step = actor.last['step']
        if step == module.TARGET_STEP:
            proof = gate.proof()
            if mode == 'intervene':
                control = json.loads((stage/'events/control.target.local.json').read_text(encoding='utf-8'))
                _verify_control(control, proof)
            _write_new(stage/'events'/(mode+'.target.local.json'), proof)
        if step == module.ACTION_TURNS-1:
            gate.verify_complete()
            _write_new(stage/'events'/(mode+'.complete.json'),
                       dict(passed=True, action_turns=module.ACTION_TURNS, target_verified=True))
        return request

    return agent
