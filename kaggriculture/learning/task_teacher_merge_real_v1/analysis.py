"""In-memory merge compatibility check; caller authenticates source and schema."""
import numpy as np
from learning.task_teacher_merge_v1.merge import merge,to_groups
from learning.task_teacher_choice_bc_v1 import training


def check(old,*,old_seeds,holdout_seeds,synthetic_seed,max_groups,max_candidates):
    """Return aggregate checks only; never persist or return input arrays."""
    if type(synthetic_seed) is not int or not 0<=synthetic_seed<=0xffffffff:
        raise ValueError('Invalid synthetic seed')
    synthetic=dict(shared=np.zeros((2,275),np.float32),candidates=np.zeros((3,98),np.float32),
        offsets=np.array([0,1,3],np.int64),teacher_choice_label=np.array([0,1],np.int64),
        ops=np.array([0,1,2],np.uint8),seed=np.array([synthetic_seed,synthetic_seed],np.uint32))
    old_groups=len(old['shared']); old_rows=len(old['candidates'])
    joined=merge(old,synthetic,old_seeds=old_seeds,new_seeds=[synthetic_seed],holdout_seeds=holdout_seeds,
                 max_groups=max_groups,max_candidates=max_candidates)
    aligned=(all(np.array_equal(joined[k][:len(old[k])],old[k]) for k in ('shared','candidates','teacher_choice_label','ops','seed'))
             and np.array_equal(joined['offsets'][:old_groups+1],old['offsets'])
             and np.array_equal(joined['offsets'][old_groups+1:],synthetic['offsets'][1:]+old_rows))
    if not aligned: raise ValueError('Ordered merge alignment failed')
    groups=to_groups(joined,train_seeds=list(old_seeds)+[synthetic_seed],holdout_seeds=holdout_seeds,
                     max_groups=max_groups,max_candidates=max_candidates)
    groups=training.as_groups(groups); training.validate(groups)
    indices=(0,old_groups,old_groups+1)
    shapes=[list(training.inputs(groups[i]).shape) for i in indices]
    old_shape_valid=shapes[0]==[int(old['offsets'][1]-old['offsets'][0]),373]
    if not old_shape_valid or shapes[1:] != [[1,373],[2,373]]: raise ValueError('Group input shape changed')
    return dict(passed=True,old_groups=old_groups,old_candidate_rows=old_rows,synthetic_groups=2,
                total_groups=len(groups),alignment_passed=True,old_input_shape_valid=old_shape_valid,synthetic_input_shapes=shapes[1:],
                source_identity='Caller must verify source SHA and common trusted schema before invocation',
                interpretation='Synthetic interface check only; no GPU training or game outcome evaluated')
