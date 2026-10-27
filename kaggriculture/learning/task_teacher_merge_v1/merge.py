"""Pure ordered teacher-choice merge; caller must bind input source hashes."""
import numpy as np
from learning.task_choice_v1.features import OPS
from learning.task_teacher_choice_bc_v1 import numeric

MAX_GROUP_WIDTH=65536


def _validate(arrays,max_groups,max_candidates):
    numeric.validate(arrays,max_groups,max_candidates)
    if np.any(np.diff(arrays['offsets'])>MAX_GROUP_WIDTH):
        raise ValueError('Candidate group exceeds trainer width limit')


def _seeds(values):
    if not isinstance(values,(list,tuple,set,frozenset)) or not values:
        raise ValueError('Nonempty seed set required')
    if any(type(v) is not int or not 0<=v<=0xffffffff for v in values) or len(set(values))!=len(values):
        raise ValueError('Invalid seed set')
    return set(values)


def merge(old,new,*,old_seeds,new_seeds,holdout_seeds,max_groups,max_candidates):
    """Merge unchanged order, rejecting source/holdout role or resource drift.

    This is not an archive loader and does not verify source hashes. The caller
    must first authenticate immutable source snapshots and shared feature schema.
    """
    old_set,new_set,holdout=_seeds(old_seeds),_seeds(new_seeds),_seeds(holdout_seeds)
    if old_set & new_set or (old_set|new_set)&holdout: raise ValueError('Seed role overlap')
    _validate(old,max_groups,max_candidates)
    _validate(new,max_groups,max_candidates)
    if set(map(int,old['seed']))!=old_set or set(map(int,new['seed']))!=new_set:
        raise ValueError('Source seed identities differ')
    groups=len(old['shared'])+len(new['shared']); candidates=len(old['candidates'])+len(new['candidates'])
    if groups>max_groups or candidates>max_candidates: raise ValueError('Combined budget exceeded')
    result={key:np.concatenate((old[key],new[key])) for key in ('shared','candidates','teacher_choice_label','ops','seed')}
    result['offsets']=np.concatenate((old['offsets'],new['offsets'][1:]+len(old['candidates'])))
    return numeric.validate(result,max_groups,max_candidates)


def to_groups(arrays,*,train_seeds,holdout_seeds,max_groups,max_candidates):
    """Produce independent numeric groups accepted by teacher-choice training.fit."""
    train,holdout=_seeds(train_seeds),_seeds(holdout_seeds)
    if train & holdout: raise ValueError('Seed role overlap')
    _validate(arrays,max_groups,max_candidates)
    if set(map(int,arrays['seed']))!=train: raise ValueError('Training seed identities differ')
    groups=[]
    for i in range(len(arrays['shared'])):
        lo,hi=int(arrays['offsets'][i]),int(arrays['offsets'][i+1])
        groups.append(dict(shared=arrays['shared'][i].copy(),candidates=arrays['candidates'][lo:hi].copy(),
            teacher_choice_label=int(arrays['teacher_choice_label'][i]),
            candidate_ops=tuple(OPS[int(op)] for op in arrays['ops'][lo:hi]),seed=int(arrays['seed'][i])))
    return groups
