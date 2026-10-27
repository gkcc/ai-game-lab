"""Fixed training-only inverse-frequency teacher-operation weights."""
import math
import numpy as np
from learning.task_choice_v1.features import OPS


def operation_weights(groups):
    if not isinstance(groups,(list,tuple)) or not groups: raise ValueError('Training groups required')
    counts={op:0 for op in OPS}
    for group in groups:
        label=group.teacher_choice_label; operations=group.candidate_ops
        if (isinstance(label,(bool,np.bool_)) or not isinstance(label,(int,np.integer)) or not 0<=label<len(operations)
            or any(op not in OPS for op in operations)):
            raise ValueError('Invalid teacher label or operation')
        counts[operations[label]]+=1
    if any(value==0 for value in counts.values()): raise ValueError('Every operation must be represented')
    total=len(groups)
    return {op:min(2.,math.sqrt(total/(len(OPS)*count))) for op,count in counts.items()}
