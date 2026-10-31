"""Aggregate-only diagnostics on authenticated training disagreements.

The caller authenticates model, sample and role identities. No trace, model
file, game, or diagnostic split is opened here. The input baseline scores are
the original float64 deployment scores; targets and roles never enter model
features. These descriptive training statistics do not establish causal
effects, generalization, or game reward. The fixed margin tolerance is a
numerical reporting threshold, not a statistical-significance test.
"""
import math

import numpy as np

from learning.task_choice_v1.features import OPS
from learning.task_multisource_rank_v1 import core as rank
from learning.task_multisource_rank_v1 import evaluate as prior
from learning.task_state_residual_v1 import core as state
from learning.task_state_residual_v1 import evaluate as state_evaluate
from learning.task_water_residual_v1 import policy


ROLES = ('A03', 'A04')
GROUPS = 42
MARGIN_TOLERANCE = 1e-6
COUNT_FIELDS = ('count', 'baseline_correct', 'candidate_correct', 'changed', 'op_flips',
                'margin_increased', 'margin_decreased', 'margin_within_tolerance')
MEAN_FIELDS = ('mean_base_nll', 'mean_candidate_nll', 'mean_base_margin', 'mean_candidate_margin')
BUCKET_FIELDS = frozenset(COUNT_FIELDS + MEAN_FIELDS + ('coverage',))


class AnalysisError(ValueError):
    """A private training sample or aggregate contract differs."""


def _require(condition):
    if not condition:
        raise AnalysisError('Invalid private training diagnostic')


def _margin(scores, target):
    """Teacher score minus highest nonteacher score, without tie shortcuts."""
    _require(type(scores) is np.ndarray and scores.dtype == np.dtype('float64')
             and scores.ndim == 1 and len(scores) >= 2 and np.isfinite(scores).all()
             and type(target) is int and 0 <= target < len(scores))
    best_other = max(float(np.max(scores[:target])) if target else -math.inf,
                     float(np.max(scores[target + 1:])) if target + 1 < len(scores) else -math.inf)
    with np.errstate(over='ignore', invalid='ignore'):
        margin = float(np.float64(scores[target]) - np.float64(best_other))
    _require(math.isfinite(margin))
    return margin


def _margin_category(baseline, candidate):
    """Use an inclusive numerical tolerance; never infer significance."""
    delta = candidate - baseline
    _require(math.isfinite(delta))
    if delta > MARGIN_TOLERANCE:
        return 'margin_increased'
    if delta < -MARGIN_TOLERANCE:
        return 'margin_decreased'
    return 'margin_within_tolerance'


def _bucket(rows):
    n = len(rows)
    result = dict(count=n, coverage='observed' if n else 'unknown',
        baseline_correct=0, candidate_correct=0, changed=0, op_flips=0,
        mean_base_nll=None, mean_candidate_nll=None,
        mean_base_margin=None, mean_candidate_margin=None,
        margin_increased=0, margin_decreased=0, margin_within_tolerance=0)
    for row in rows:
        for key in COUNT_FIELDS[1:]:
            result[key] += int(row[key])
    if n:
        for out, field in (('mean_base_nll','base_nll'), ('mean_candidate_nll','candidate_nll'),
                           ('mean_base_margin','base_margin'), ('mean_candidate_margin','candidate_margin')):
            try:
                result[out] = math.fsum(row[field] / n for row in rows)
            except OverflowError:
                raise AnalysisError('Unrepresentable private aggregate') from None
            _require(math.isfinite(result[out]))
    return result


def validate(value):
    """Validate aggregate shape and accounting, not underlying source truth."""
    _require(type(value) is dict and set(value) == {'overall', 'by_role', 'by_target_operation'}
             and type(value['by_role']) is dict and set(value['by_role']) == set(ROLES)
             and type(value['by_target_operation']) is dict and set(value['by_target_operation']) == set(OPS))

    def checked(row):
        _require(type(row) is dict and set(row) == BUCKET_FIELDS
                 and all(type(row[key]) is int and 0 <= row[key] <= GROUPS for key in COUNT_FIELDS))
        n = row['count']
        _require(all(row[key] <= n for key in COUNT_FIELDS[1:])
                 and row['coverage'] == ('observed' if n else 'unknown')
                 and row['baseline_correct'] == 0 and row['candidate_correct'] <= row['changed']
                 and row['op_flips'] <= row['changed']
                 and row['margin_increased'] + row['margin_decreased'] + row['margin_within_tolerance'] == n)
        if not n:
            _require(all(row[key] is None for key in MEAN_FIELDS))
        else:
            _require(all(type(row[key]) in (int, float) and math.isfinite(row[key]) for key in MEAN_FIELDS)
                     and row['mean_base_nll'] >= 0 and row['mean_candidate_nll'] >= 0
                     and row['mean_base_margin'] <= 0)

    def partition(total, parts):
        checked(total)
        for part in parts:
            checked(part)
        _require(all(total[key] == sum(part[key] for part in parts) for key in COUNT_FIELDS))
        if total['count']:
            for key in MEAN_FIELDS:
                weighted = math.fsum(part[key] * (part['count'] / total['count'])
                                     for part in parts if part['count'])
                _require(math.isfinite(weighted)
                         and math.isclose(total[key], weighted, rel_tol=1e-10, abs_tol=1e-10))

    _require(type(value['overall']) is dict and value['overall'].get('count') == GROUPS)
    partition(value['overall'], [value['by_role'][role] for role in ROLES])
    partition(value['overall'], [value['by_target_operation'][op] for op in OPS])
    return True


def assess(groups, student_indices, roles, *, mean, scale, params, metadata, guard):
    """Analyze exactly 42 authenticated training groups and return aggregates.

    Baseline argmax must equal the recorded student, and teacher must differ.
    Both choice and target margin use the state residual's actual NumPy
    deployment arithmetic, including common-offset removal and first-argmax.
    Inputs must remain unchanged while called; snapshots prevent later edits.
    """
    def sequence(values):
        return (type(values) in (list, tuple, np.ndarray)
                and (type(values) is not np.ndarray or values.ndim == 1)
                and len(values) == GROUPS)
    _require(callable(guard) and type(groups) in (list, tuple) and len(groups) == GROUPS
             and sequence(student_indices) and sequence(roles))
    guard()
    schema = policy._schema(metadata)
    policy._normalization(mean, scale, schema)
    _require(schema[3] == tuple(rank.SHARED_DIM + rank.OP_OFFSET + i for i in range(len(OPS))))
    mean, scale = rank._normalization(mean, scale)
    params = state._checked_params(params)
    rows = []
    for group, student, role in zip(groups, student_indices, roles):
        guard()
        _require(isinstance(role, str) and role in ROLES
                 and isinstance(student, (int, np.integer)) and not isinstance(student, (bool, np.bool_)))
        student = int(student)
        rank._group(group)
        _require(group.source == 'new' and 0 <= student < len(group.candidates)
                 and student != group.target)
        n = len(group.candidates)
        shared = prior._snapshot(group.shared, 'float32', (rank.SHARED_DIM,))
        candidates = prior._snapshot(group.candidates, 'float32', (n, rank.LOCAL_DIM))
        baseline = prior._snapshot(group.baseline, 'float64', (n,))
        frozen = rank.Group(shared, candidates, baseline, group.target, group.source)
        _require(int(np.argmax(baseline)) == student)
        residual = state_evaluate._numpy_residual(frozen, mean, scale, params)
        adjusted = state._deployment_scores(baseline, residual) if np.any(params.v) else baseline
        candidate = state.select_values(shared, candidates, baseline, mean, scale, params)
        _require(type(candidate) is int and candidate == int(np.argmax(adjusted)))
        raw = np.concatenate((np.broadcast_to(shared, (n, rank.SHARED_DIM)), candidates), axis=1)
        operations = policy._operations(raw, schema)
        base_margin, candidate_margin = _margin(baseline, group.target), _margin(adjusted, group.target)
        category = _margin_category(base_margin, candidate_margin)
        rows.append(dict(role=str(role), target_op=OPS[int(np.argmax(operations[group.target]))],
            baseline_correct=False, candidate_correct=candidate == group.target,
            changed=candidate != student, op_flips=not np.array_equal(operations[student], operations[candidate]),
            base_nll=prior._nll(baseline, group.target), candidate_nll=prior._nll(adjusted, group.target),
            base_margin=base_margin, candidate_margin=candidate_margin,
            margin_increased=category == 'margin_increased',
            margin_decreased=category == 'margin_decreased',
            margin_within_tolerance=category == 'margin_within_tolerance'))
    guard()
    output = dict(overall=_bucket(rows),
                  by_role={role: _bucket([row for row in rows if row['role'] == role]) for role in ROLES},
                  by_target_operation={op: _bucket([row for row in rows if row['target_op'] == op]) for op in OPS})
    validate(output)
    guard()
    return output
