"""Fixed quantiles of the old selector's fully validated eligible targets.

The source is consumed once through the unchanged frozen selector code. A
private function namespace captures its already validated point constructions;
the original module and helpers are never patched. The caller authenticates
the source bytes and prior target, and owns stream closure and resource limits.
No source file, checkpoint, model forward or official environment is opened.
"""
from copy import deepcopy
from types import FunctionType

from learning.task_harvest_return_pairs_v1 import selection as frozen


def _same_target(left, right):
    """Use the existing pair core's fieldwise, order-aware target identity."""
    if left is None or right is None:
        return left is right
    return (type(left) is dict and type(right) is dict and set(left) == set(right)
            and all(frozen.ordered_hash(left[key]) == frozen.ordered_hash(right[key])
                    for key in left))


def _coordinate(target):
    return (target['step'], target['ordinal'], target['unit'],
            target['target'][1], target['target'][0])


def select_quantiles(rows, *, side, task_sha, crop_sha, old_target, guard=None):
    """Validate all 719 rows and EOF, then retain three fixed quantile slots.

    Index zero is the authenticated prior reference, not a new experiment.
    Repeated indices remain empty slots; no replacement target is searched.
    An empty population accepts only an absent prior reference and has no
    defined quantile indices. A present but mismatched prior target is an error.
    """
    expected = None if old_target is None else frozen.validate_target(old_target)
    captured = []
    original = frozen.select_target
    namespace = dict(original.__globals__)
    validate = namespace['validate_target']

    def collect(target):
        validated = validate(target)
        captured.append(deepcopy(validated))
        return validated

    namespace['validate_target'] = collect
    scan = FunctionType(original.__code__, namespace, original.__name__,
                        original.__defaults__, original.__closure__)
    scan.__kwdefaults__ = (None if original.__kwdefaults__ is None
                           else dict(original.__kwdefaults__))
    result = scan(rows, side=side, task_sha=task_sha, crop_sha=crop_sha, guard=guard)
    frozen.require(len(captured) == result['counts']['qualifying_callbacks'],
                   'Captured quantile population differs from validated callbacks')
    captured.sort(key=_coordinate)
    coordinates = [_coordinate(target) for target in captured]
    frozen.require(len(set(coordinates)) == len(coordinates),
                   'Repeated eligible target coordinate')
    reference = captured[0] if captured else None
    frozen.require(_same_target(reference, result['target']),
                   'Quantile reference differs from frozen first-target selection')
    frozen.require(_same_target(reference, expected),
                   'Quantile reference differs from authenticated prior target')

    count = len(captured)
    indices = ([0, (count - 1) // 3, 2 * (count - 1) // 3, count - 1]
               if count else [0, None, None, None])
    used, points = {0}, []
    for label, index in zip(('Q1', 'Q2', 'Q3'), indices[1:]):
        target = None
        if index is None:
            reason = 'no_eligible_target'
        elif index in used:
            reason = 'duplicate_index'
        else:
            reason = None
            target = deepcopy(captured[index])
            used.add(index)
        points.append(dict(label=label, index=index, target=target, reason=reason))
    return dict(reference=deepcopy(reference), points=points, indices=indices,
                eligible_count=count, counts=deepcopy(result['counts']))
