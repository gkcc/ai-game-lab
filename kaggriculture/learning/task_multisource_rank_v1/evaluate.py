"""Offline aggregate-only comparison of a cross-operation task ranker.

This module does not read sources, authenticate a model, train, or play. The
caller supplies authenticated groups and non-identifying registered role
labels. Metrics compare the teacher's target index within each existing task
opportunity set; they do not establish game reward, no forgetting, independent
generalization, or official action execution.
"""
from importlib import import_module
import math
import re

import numpy as np

from learning.task_choice_v1.features import OPS
from learning.task_water_residual_v1 import policy


MAX_GROUPS = 8192
MAX_CANDIDATES = 256
_ROLE = re.compile(r'[A-Z][0-9]{2}\Z')


class EvaluationError(ValueError):
    """The supplied offline groups or scoring contract are inconsistent."""


def _require(value):
    if not value:
        raise EvaluationError('Invalid private ranking evaluation input')


def _snapshot(array, dtype, shape):
    _require(type(array) is np.ndarray and array.dtype == np.dtype(dtype)
             and array.shape == shape and np.isfinite(array).all())
    return np.frombuffer(array.tobytes(order='C'), dtype=np.dtype(dtype)).reshape(shape)


def _nll(scores, target):
    """Finite float64 logsumexp; reject loss too large to represent."""
    maximum = np.max(scores)
    with np.errstate(over='ignore', under='ignore', invalid='ignore'):
        shifted = scores - maximum
        normalizer = np.exp(shifted).sum()
        value = -shifted[target] + np.log(normalizer)
    # Other logits may safely underflow to -inf in the shifted representation;
    # the selected loss itself must be finite and publishable as JSON.
    _require(not np.isnan(shifted).any() and np.isfinite(normalizer)
             and normalizer >= 1 and np.isfinite(value))
    return float(value)


def _summary(rows):
    count = len(rows)
    result = dict(groups=count, coverage='observed' if count else 'unknown',
                  baseline_correct=0, residual_correct=0, changed=0, op_flips=0,
                  improved=0, regressed=0, baseline_nll=None, residual_nll=None)
    for row in rows:
        for name in ('baseline_correct', 'residual_correct', 'changed', 'op_flips', 'improved', 'regressed'):
            result[name] += int(row[name])
    if count:
        try:
            result['baseline_nll'] = math.fsum(row['baseline_nll'] / count for row in rows)
            result['residual_nll'] = math.fsum(row['residual_nll'] / count for row in rows)
        except OverflowError:
            raise EvaluationError('Unrepresentable aggregate loss') from None
        _require(math.isfinite(result['baseline_nll']) and math.isfinite(result['residual_nll']))
    return result


def _role_selection(groups, roles, role_labels):
    _require(type(role_labels) in (list, tuple) and len(role_labels) <= 100
             and all(type(role) is str and _ROLE.fullmatch(role) is not None for role in role_labels)
             and len(set(role_labels)) == len(role_labels))
    if roles is not None:
        _require(type(roles) in (list, tuple) and len(roles) == len(groups) and bool(role_labels))
        selected = list(roles)
    elif role_labels:
        selected = [getattr(group, 'role', None) for group in groups]
    else:
        return None
    _require(all(type(role) is str and role in role_labels for role in selected))
    return selected


def assess(groups, mean, scale, weights, metadata, *, roles=None, role_labels=(), guard=lambda: None):
    """Return only aggregate accuracy, operation-flip, and mean conditional NLL.

    Each item is a core Group or an outer object with ``group`` and ``role``.
    An optional same-length ``roles`` list overrides wrapper roles; callers
    register the entire anonymous ``role_labels`` vocabulary, including any
    currently absent role. Baseline logits stay float64. Residuals follow the
    core contract: raw32 -> float64 normalization -> float32 dot -> add to the
    baseline in float64. Any disagreement with the actual core selector fails.
    No per-group indices, features, logits, source IDs or seeds are returned.
    """
    _require(type(groups) in (list, tuple) and len(groups) <= MAX_GROUPS and callable(guard))
    guard()
    schema = policy._schema(metadata)
    policy._normalization(mean, scale, schema)
    mean = _snapshot(mean, 'float64', (policy.DIM,))
    scale = _snapshot(scale, 'float64', (policy.DIM,))
    weights = _snapshot(weights, 'float32', (policy.DIM,))
    selected_roles = _role_selection(groups, roles, role_labels)
    # Import only at call time, so this module remains importable while the
    # independently developed core is not yet present.
    core = import_module('learning.task_multisource_rank_v1.core')
    _require(callable(getattr(core, 'select', None)) and callable(getattr(core, 'Group', None)))
    rows = []
    for index, outer in enumerate(groups):
        guard()
        item = getattr(outer, 'group', outer)
        try:
            shared, candidates, baseline, target, source = (
                item.shared, item.candidates, item.baseline, item.target, item.source)
        except AttributeError:
            raise EvaluationError('Invalid private ranking evaluation input') from None
        _require(type(candidates) is np.ndarray and candidates.dtype == np.dtype('float32')
                 and candidates.ndim == 2 and candidates.shape[1] == policy.LOCAL_DIM
                 and 1 <= len(candidates) <= MAX_CANDIDATES
                 and type(target) is int and 0 <= target < len(candidates)
                 and type(source) is str and source in ('old', 'new'))
        count = len(candidates)
        shared = _snapshot(shared, 'float32', (policy.SHARED_DIM,))
        candidates = _snapshot(candidates, 'float32', (count, policy.LOCAL_DIM))
        baseline = _snapshot(baseline, 'float64', (count,))
        raw = np.concatenate((np.broadcast_to(shared, (count, policy.SHARED_DIM)), candidates), axis=1)
        operations = policy._operations(raw, schema)
        base_index = int(np.argmax(baseline))
        if np.any(weights):
            with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
                normalized = ((raw.astype(np.float64) - mean) / scale).astype(np.float32)
                residual = normalized @ weights
                adjusted = baseline + residual.astype(np.float64)
            _require(normalized.dtype == np.dtype('float32') and np.isfinite(normalized).all()
                     and residual.dtype == np.dtype('float32') and np.isfinite(residual).all()
                     and adjusted.dtype == np.dtype('float64') and np.isfinite(adjusted).all())
        else:
            adjusted = baseline
        private_group = core.Group(shared, candidates, baseline, target, source)
        chosen = core.select(private_group, mean, scale, weights)
        _require(type(chosen) is int and 0 <= chosen < count and chosen == int(np.argmax(adjusted)))
        baseline_correct, residual_correct = base_index == target, chosen == target
        rows.append(dict(source=source, role=None if selected_roles is None else selected_roles[index],
            target_op=OPS[int(np.argmax(operations[target]))], baseline_correct=baseline_correct,
            residual_correct=residual_correct, changed=base_index != chosen,
            op_flips=not np.array_equal(operations[base_index], operations[chosen]),
            improved=not baseline_correct and residual_correct,
            regressed=baseline_correct and not residual_correct,
            baseline_nll=_nll(baseline, target), residual_nll=_nll(adjusted, target)))
    guard()
    by_operation = {op: _summary([row for row in rows if row['target_op'] == op]) for op in OPS}
    by_source = {source: _summary([row for row in rows if row['source'] == source])
                 for source in ('old', 'new')}
    by_role = {}
    if selected_roles is not None:
        for role in role_labels:
            local = [row for row in rows if row['role'] == role]
            by_role[role] = dict(overall=_summary(local),
                by_target_operation={op: _summary([row for row in local if row['target_op'] == op])
                                     for op in OPS})
    result = dict(overall=_summary(rows), by_target_operation=by_operation,
                  by_source=by_source, by_role=by_role, independent_evaluation=False)
    guard()
    return result
