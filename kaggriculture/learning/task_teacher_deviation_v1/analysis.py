"""Pure fixed training-set teacher-versus-student aggregate comparison.

Input loading, provenance, model identity and process supervision belong to a
future independently frozen entrypoint; this module performs no I/O.
"""
import hashlib
import numpy as np
from learning.task_choice_v1.features import OPS

POPULATION=2253
SAMPLE=200
MAX_ROWS=120000
DOMAIN='task-teacher-deviation-v1:training-group:'


def select_indices(population=POPULATION):
    if type(population) is not int or population != POPULATION: raise ValueError('Unexpected population')
    ranked=sorted((hashlib.sha256((DOMAIN+str(i)).encode('ascii')).digest(),i) for i in range(population))
    return [index for _,index in ranked[:SAMPLE]]


def analyze(arrays,new_model,old_model,progress=None):
    """Score both models in fixed new-then-old order and return aggregates."""
    keys={'shared','candidates','offsets','teacher_choice_label','ops','seed'}
    if (not isinstance(arrays,dict) or set(arrays)!=keys or not callable(getattr(new_model,'score',None))
        or not callable(getattr(old_model,'score',None)) or (progress is not None and not callable(progress))):
        raise ValueError('Invalid inputs')
    shared,candidates,offsets=arrays['shared'],arrays['candidates'],arrays['offsets']
    labels,ops,seeds=arrays['teacher_choice_label'],arrays['ops'],arrays['seed']
    if any(not isinstance(x,np.ndarray) for x in (shared,candidates,offsets,labels,ops,seeds)):
        raise ValueError('Invalid arrays')
    if (shared.dtype!=np.float32 or shared.shape!=(POPULATION,275) or candidates.dtype!=np.float32
        or candidates.ndim!=2 or candidates.shape[1]!=98 or offsets.dtype!=np.int64
        or offsets.shape!=(POPULATION+1,) or labels.dtype!=np.int64 or labels.shape!=(POPULATION,)
        or ops.dtype!=np.uint8 or ops.shape!=(len(candidates),) or seeds.dtype!=np.uint32 or seeds.shape!=(POPULATION,)):
        raise ValueError('Invalid array schema')
    if (offsets[0]!=0 or offsets[-1]!=len(candidates) or np.any(offsets[1:]<=offsets[:-1])
        or np.any(labels<0) or np.any(labels>=np.diff(offsets)) or np.any(ops>=len(OPS))
        or not np.isfinite(shared).all() or not np.isfinite(candidates).all()):
        raise ValueError('Invalid array values')
    selected=select_indices()
    rows=sum(int(offsets[i+1]-offsets[i]) for i in selected)
    if 2*rows>MAX_ROWS: raise ValueError('Candidate budget exceeded')
    groups={model:{op:dict(count=0,top1_matches=0,rank_sum=0) for op in OPS} for model in ('new','old')}
    top1={'new':0,'old':0}; paired=dict(both_correct=0,new_only=0,old_only=0,both_wrong=0)
    completed=0
    for i in selected:
        start,end=int(offsets[i]),int(offsets[i+1]); teacher=int(labels[i])
        matches={}
        for name,model in (('new',new_model),('old',old_model)):
            if progress is not None: progress(dict(forwards_started=completed+1,forwards_completed=completed))
            logits=np.asarray(model.score(shared[i],candidates[start:end]),dtype=np.float64)
            completed+=1
            if progress is not None: progress(dict(forwards_started=completed,forwards_completed=completed))
            if logits.shape!=(end-start,) or not np.isfinite(logits).all(): raise ValueError('Invalid model scores')
            matches[name]=int(np.argmax(logits))==teacher
            rank=1+int(np.count_nonzero(logits>logits[teacher]))+int(np.count_nonzero(logits[:teacher]==logits[teacher]))
            group=groups[name][OPS[int(ops[start+teacher])]]
            group['count']+=1; group['top1_matches']+=int(matches[name]); group['rank_sum']+=rank
            top1[name]+=int(matches[name])
        key=('both_correct' if matches['new'] and matches['old'] else 'new_only' if matches['new'] else 'old_only' if matches['old'] else 'both_wrong')
        paired[key]+=1
    for model_groups in groups.values():
        for group in model_groups.values():
            count=group['count']; group['top1_rate']=group['top1_matches']/count if count else None
            group['mean_teacher_rank']=group['rank_sum']/count if count else None
    return dict(groups=SAMPLE,candidate_rows_per_model=rows,candidate_rows_total=2*rows,cpu_forwards=completed,
        models={name:dict(top1_matches=top1[name],top1_rate=top1[name]/SAMPLE,by_teacher_operation=groups[name]) for name in ('new','old')},
        paired=paired,rank_rule='descending score, original candidate order breaks exact ties',
        scope='New model trained on these groups; old model may not have seen them. Not independent holdout, generalization or economic causality')
