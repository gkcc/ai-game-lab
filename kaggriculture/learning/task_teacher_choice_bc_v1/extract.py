"""Pure per-callback conditional-teacher extraction; no files or training.

The caller must establish trace provenance and bound streaming resources. Mixed
behavior states are not teacher trajectories or evidence of task completion.
"""
import numpy as np
from decision.api import observe
from learning.task_choice_v1.features import CANDIDATE_SCOPE, OPS, SCHEMA_VERSION, candidate_id, schema_hash


def extract(record, expected_schema, *, side, step, max_candidates):
    if type(side) is not int or side not in (0,1) or type(step) is not int or not 0 <= step < 719:
        raise ValueError('Invalid callback identity')
    if type(max_candidates) is not int or max_candidates < 1: raise ValueError('Invalid candidate budget')
    if not isinstance(record,dict) or not isinstance(expected_schema,dict): raise ValueError('Invalid input')
    try:
        inputs = record['inputs']
        fields = {'schema_version','schema_sha256','feature_names','shared_feature_names','shared_x','candidate_scope',
                  'observation','configuration','planned','candidates','inventory_item_order','shed_item_order'}
        if not isinstance(inputs,dict) or set(inputs) != fields: raise ValueError('Unexpected selector inputs')
        names, shared_names = inputs['feature_names'],inputs['shared_feature_names']
        schema = dict(schema_version=SCHEMA_VERSION,schema_sha256=schema_hash(names,shared_names),
                      shared_feature_names=shared_names,feature_names=names,ops=list(OPS))
        if (schema != expected_schema or inputs['schema_version'] != SCHEMA_VERSION
            or inputs['schema_sha256'] != schema['schema_sha256'] or len(names) != 98 or len(set(names)) != 98
            or len(shared_names) != 275 or len(set(shared_names)) != 275 or inputs['candidate_scope'] != CANDIDATE_SCOPE):
            raise ValueError('Untrusted or incompatible schema')
        if any(not isinstance(n,str) or any(term in n.lower() for term in ('teacher','priority','opponent.private')) for n in names+shared_names):
            raise ValueError('Forbidden model feature')
        obs,cfg = inputs['observation'],inputs['configuration']
        view = observe(obs,cfg)
        if view.observation != obs or view.configuration != cfg or obs['player'] != side or obs['step'] != step:
            raise ValueError('Invalid observation projection')
        planned = inputs['planned']
        if (set(planned) != {'seeds','needed_seeds','supply','projected_shed','unit_actions','available_units','interpretation'}
            or planned['interpretation'] != 'before assignment: reservations and requests, not official executed state'
            or inputs['inventory_item_order'] != [list(i) for i in obs['private']['inventories']]
            or inputs['shed_item_order'] != list(obs['private']['shed'])):
            raise ValueError('Invalid planning projection')
        candidates = inputs['candidates']
        if not isinstance(candidates,list) or not 1 <= len(candidates) <= max_candidates:
            raise ValueError('Candidate budget exceeded')
        required = ['unit.index','unit.x','unit.y','target.x','target.y','task.distance']+['task.op.'+op for op in OPS]
        if any(names.count(n) != 1 for n in required): raise ValueError('Missing identity features')
        shared = np.asarray(inputs['shared_x'],dtype=np.float32)
        local = np.asarray([c['x'] for c in candidates],dtype=np.float32)
        if shared.shape != (275,) or local.shape != (len(candidates),98) or not np.isfinite(shared).all() or not np.isfinite(local).all():
            raise ValueError('Invalid numeric features')
        own = obs['farms'][side]
        positions = [own['farmer'],*own['hands']]
        ids, ordering, ops = [],[],[]
        for i,c in enumerate(candidates):
            if not isinstance(c,dict) or set(c) != {'id','unit','target','op','x'}: raise ValueError('Unexpected candidate fields')
            unit,target,op = c['unit'],c['target'],c['op']
            if (type(unit) is not int or not 0 <= unit < min(len(positions),32768) or unit not in planned['available_units']
                or type(target) is not list or len(target) != 2 or any(type(v) is not int or not 0 <= v < 32768 for v in target)
                or op not in OPS or c['id'] != candidate_id(unit,target,op)):
                raise ValueError('Invalid candidate identity')
            ux,uy = positions[unit]
            checks = {'unit.index':unit,'unit.x':ux,'unit.y':uy,'target.x':target[0],'target.y':target[1],
                      'task.distance':abs(ux-target[0])+abs(uy-target[1])}
            checks.update({'task.op.'+kind:int(kind==op) for kind in OPS})
            if any(float(local[i,names.index(key)]) != value for key,value in checks.items()):
                raise ValueError('Candidate feature mismatch')
            ids.append(c['id']); ordering.append((unit,target[1],target[0],op)); ops.append(op)
        label = record['teacher_label']
        if len(ids) != len(set(ids)) or ordering != sorted(ordering) or not isinstance(label,str) or ids.count(label) != 1:
            raise ValueError('Invalid teacher label or ordering')
        actual = record['chosen']
        if not isinstance(actual,str) or ids.count(actual) != 1:
            raise ValueError('Invalid actual selection')
        actual_candidate = candidates[ids.index(actual)]
        if any(record[key] != actual_candidate[key] for key in ('unit','target','op')):
            raise ValueError('Actual selection identity mismatch')
        return dict(shared=shared.copy(),candidates=local.copy(),teacher_choice_label=ids.index(label),candidate_ops=tuple(ops))
    except (KeyError,TypeError,IndexError,AttributeError) as exc:
        raise ValueError('Malformed callback') from exc
