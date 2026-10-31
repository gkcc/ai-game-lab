"""Private filename-entry adapter. It never receives a seed or opponent data."""
import hashlib
import json
from pathlib import Path
import types


def _relative(root,value):
    if (type(value) is not str or not value or '\\' in value or ':' in value):
        raise ValueError('Invalid project-relative model')
    path=Path(value)
    if path.is_absolute() or any(part in ('.','..') for part in path.parts):
        raise ValueError('Invalid project-relative model')
    resolved=(root/path).resolve(strict=True)
    if not resolved.is_relative_to(root.resolve()): raise ValueError('Model outside project')
    return resolved


def _write(path,value):
    with path.open('x',encoding='utf-8',newline='\n') as stream:
        json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False)
        stream.write('\n')


def _verify_proof(control,intervention):
    from learning.task_bc_error_diagnostics_v1.shadow import first_difference
    keys=('step','selection_record_index','input_sha256','original_choice','teacher_choice',
          'budget_before','budget_after','teacher_label_verified')
    if (type(control) is not dict or type(intervention) is not dict
            or control.get('mode')!='control' or intervention.get('mode')!='intervene'
            or any(key not in control or key not in intervention
                   or first_difference(control[key],intervention[key]) is not None for key in keys)
            or control.get('teacher_label_verified') is not True
            or control.get('returned_choice')!=control.get('original_choice')
            or intervention.get('returned_choice')!=intervention.get('teacher_choice')):
        raise ValueError('Target proof differs')


def build(stage,game_root,mode):
    """Load only a seed-free private gate and construct one independent actor."""
    from learning.task_dagger_once_v1.policy import BoundedAgent
    stage,root=Path(stage).resolve(strict=True),Path(game_root).resolve(strict=True)
    if mode not in ('control','intervene'): raise ValueError('Invalid actor arm')
    gate_data=json.loads((stage/'gate.local.json').read_text(encoding='utf-8'))
    keys={'schema','expected','step','ordinal','crop_checkpoint','crop_sha256','task_checkpoint',
          'task_sha256','limits','gate_source_sha256','actor_source_sha256'}
    if type(gate_data) is not dict or set(gate_data)!=keys or gate_data['schema']!='teacher-pair-gate-v1':
        raise ValueError('Invalid private gate')
    raw=(stage/'sources/gate.py').read_bytes()
    if hashlib.sha256(raw).hexdigest()!=gate_data['gate_source_sha256']:
        raise ValueError('Frozen gate changed')
    module=types.ModuleType('frozen_teacher_pair_gate')
    exec(compile(raw,'<frozen-teacher-pair-gate>','exec',dont_inherit=True),module.__dict__)
    actor=BoundedAgent(_relative(root,gate_data['crop_checkpoint']),gate_data['crop_sha256'],
        _relative(root,gate_data['task_checkpoint']),gate_data['task_sha256'],
        trace=stage/'traces'/(mode+'.jsonl.gz'),limits=gate_data['limits'])
    selector=module.OneShotSelector(actor.controller.task_selector,
        lambda:len(actor.controller.records),
        lambda:dict(total=dict(actor.budget.counts),turn=dict(actor.budget.turn)),
        gate_data['expected'],mode,step=gate_data['step'],ordinal=gate_data['ordinal'])
    actor.controller.task_selector=selector

    def agent(observation,configuration):
        request=actor(observation,configuration)
        selector.verify_turn(actor.last)
        step=actor.last['step']
        if step==gate_data['step']:
            proof=selector.proof()
            if mode=='intervene':
                control=json.loads((stage/'events/control.target.local.json').read_text(encoding='utf-8'))
                _verify_proof(control,proof)
            _write(stage/'events'/(mode+'.target.local.json'),proof)
        if step==module.ACTION_TURNS-1:
            selector.verify_complete()
            _write(stage/'events'/(mode+'.complete.local.json'),
                   dict(passed=True,action_turns=module.ACTION_TURNS,target_verified=True))
        return request
    return agent
