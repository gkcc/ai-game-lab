"""Pure development evaluation of two TaskNetwork-compatible scorers.

The caller authenticates inputs and models, bounds memory and inference, and
keeps exact in-memory statistics private.  This module neither reads nor writes
files, imports a model, runs a game, or claims an independent test.  Candidate
sets are frozen teacher opportunities, not the official complete action space.
"""

from fractions import Fraction

import numpy as np

from learning.task_choice_v1.features import OPS, SCHEMA_VERSION, schema_hash
from learning.task_teacher_choice_bc_v1 import numeric
from learning.task_worker_geometry_diagnostic_v1.training_summary import summarize_training_geometry


METRICS = (
    'conditional_teacher_top1',
    'overall_teacher_top1',
    'overall_same_target_plant',
)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _prepare(shared, candidates, offsets, teacher_choice_label, ops, cluster_ids,
             shared_feature_names, feature_names, max_candidates_per_group):
    _require(type(max_candidates_per_group) is int and 1 <= max_candidates_per_group <= 65536,
             'Invalid candidate width limit')
    _require(isinstance(shared, np.ndarray) and isinstance(candidates, np.ndarray),
             'Invalid arrays')
    arrays = dict(shared=shared, candidates=candidates, offsets=offsets,
                  teacher_choice_label=teacher_choice_label, ops=ops, seed=cluster_ids)
    try:
        numeric.validate(arrays, len(shared), len(candidates))
        summarize_training_geometry(shared, candidates, offsets, teacher_choice_label,
                                    ops, shared_feature_names, feature_names)
    except (ValueError, TypeError, IndexError, AttributeError):
        raise ValueError('Invalid authenticated training arrays or schema') from None
    _require(np.all(offsets[1:] - offsets[:-1] <= max_candidates_per_group),
             'Candidate width exceeds limit')

    unit_col = feature_names.index('unit.index')
    x_col, y_col = feature_names.index('target.x'), feature_names.index('target.y')
    plant = OPS.index('PLANT')
    eligible, coverage = [], {}
    for group in range(len(shared)):
        start, end = int(offsets[group]), int(offsets[group + 1])
        teacher = int(teacher_choice_label[group])
        rows, row_ops = candidates[start:end], ops[start:end]
        if int(row_ops[teacher]) != plant:
            continue
        same_target = np.flatnonzero((row_ops == plant)
            & (rows[:, x_col] == rows[teacher, x_col])
            & (rows[:, y_col] == rows[teacher, y_col]))
        # The frozen geometry validator already rejects duplicate candidate IDs.
        if len(same_target) < 2 or len(np.unique(rows[same_target, unit_col])) < 2:
            continue
        cluster = int(cluster_ids[group])
        coverage[cluster] = coverage.get(cluster, 0) + 1
        eligible.append((group, same_target))

    # Source and qualifying groups must both cover the same four clusters.
    source_clusters = set(int(value) for value in cluster_ids)
    sufficient = (len(eligible) >= 20 and len(source_clusters) == 4
                  and set(coverage) == source_clusters
                  and all(number >= 5 for number in coverage.values()))
    return eligible if sufficient else None


def assess_coverage(shared, candidates, offsets, teacher_choice_label, ops, cluster_ids,
                    shared_feature_names, feature_names, *, max_candidates_per_group=256):
    """Return only whether the fixed development eligibility gate is met.

    Cluster IDs remain internal and are never returned. This check lets a
    caller defer loading either scorer until the gate passes. Five groups in a
    cluster are not five independent games. It does not prove archive provenance
    or that any observation is an independent test.
    """
    return _prepare(shared, candidates, offsets, teacher_choice_label, ops, cluster_ids,
                    shared_feature_names, feature_names, max_candidates_per_group) is not None


def _validate_scorer(scorer, shared_names, names):
    try:
        metadata, score = scorer.metadata, scorer.score
        expected = schema_hash(names, shared_names)
        ok = (isinstance(metadata, dict) and callable(score)
              and metadata.get('schema_version') == SCHEMA_VERSION
              and metadata.get('schema_sha256') == expected
              and metadata.get('shared_feature_names') == list(shared_names)
              and metadata.get('feature_names') == list(names)
              and metadata.get('ops') == list(OPS))
    except (AttributeError, TypeError, ValueError):
        ok = False
    _require(ok, 'Scorer schema mismatch')


def _score(scorer, shared, candidates):
    # Each scorer receives an independent full-group copy: one scorer cannot
    # accidentally change the other scorer's input or the caller's arrays.
    own_shared, own_candidates = shared.copy(), candidates.copy()
    own_shared.flags.writeable = own_candidates.flags.writeable = False
    try:
        scores = np.asarray(scorer.score(own_shared, own_candidates))
        if (scores.shape != (len(candidates),)
                or scores.dtype not in (np.dtype('float32'), np.dtype('float64'))
                or not np.isfinite(scores).all()):
            raise ValueError('invalid scores')
        scores = scores.astype(np.float64, copy=False)
        if not np.isfinite(scores).all():
            raise ValueError('invalid scores')
        return scores
    except Exception:
        raise ValueError('Scorer returned invalid scores or failed') from None


def _group_metrics(scores, same_target, teacher, rows, operations, x_col, y_col):
    # Canonical candidate order breaks exact score ties just as np.argmax does.
    reference = scores[teacher]
    rank = 1 + sum(scores[index] > reference or (scores[index] == reference and index < teacher)
                   for index in same_target)
    best = int(np.argmax(scores))
    same = bool(rows[best, x_col] == rows[teacher, x_col]
                and rows[best, y_col] == rows[teacher, y_col])
    return (
        Fraction(int(rank == 1)),
        Fraction(int(best == teacher)),
        Fraction(int(same and int(operations[best]) == OPS.index('PLANT'))),
    )


def evaluate_private(shared, candidates, offsets, teacher_choice_label, ops, cluster_ids,
                     shared_feature_names, feature_names, scorer_a, scorer_b, *,
                     max_candidates_per_group=256):
    """Return private exact group-weighted, then cluster-weighted statistics.

    The eligibility gate is rechecked before any scorer is inspected or called.
    Each scorer receives all candidates in a qualifying group exactly once.
    Exact results are sensitive: the caller must not print or persist them.
    Use `project_public` only after the registered disclosure review.
    """
    eligible = _prepare(shared, candidates, offsets, teacher_choice_label, ops, cluster_ids,
                        shared_feature_names, feature_names, max_candidates_per_group)
    if eligible is None:
        return {'coverage': 'insufficient'}
    _validate_scorer(scorer_a, shared_feature_names, feature_names)
    _validate_scorer(scorer_b, shared_feature_names, feature_names)
    x_col, y_col = feature_names.index('target.x'), feature_names.index('target.y')
    aggregates = {}
    for group, subset in eligible:
        start, end = int(offsets[group]), int(offsets[group + 1])
        label = int(teacher_choice_label[group])
        rows, operations = candidates[start:end], ops[start:end]
        scores_a = _score(scorer_a, shared[group], rows)
        scores_b = _score(scorer_b, shared[group], rows)
        a = _group_metrics(scores_a, subset, label, rows, operations, x_col, y_col)
        b = _group_metrics(scores_b, subset, label, rows, operations, x_col, y_col)
        cluster = int(cluster_ids[group])
        if cluster not in aggregates:
            aggregates[cluster] = [0, [Fraction(0)] * len(METRICS), [Fraction(0)] * len(METRICS)]
        row = aggregates[cluster]
        row[0] += 1
        row[1] = [old + new for old, new in zip(row[1], a)]
        row[2] = [old + new for old, new in zip(row[2], b)]

    metrics = {}
    for index, name in enumerate(METRICS):
        mean_a = sum((row[1][index] / row[0] for row in aggregates.values()), Fraction(0)) / 4
        mean_b = sum((row[2][index] / row[0] for row in aggregates.values()), Fraction(0)) / 4
        metrics[name] = {'model_a': mean_a, 'model_b': mean_b}
    return {'coverage': 'sufficient', 'metrics': metrics}


def _validate_rate(value):
    _require(isinstance(value, Fraction) and 0 <= value <= 1, 'Invalid private rate')


def _difference_band(value):
    _require(isinstance(value, Fraction) and -1 <= value <= 1, 'Invalid private difference')
    if value < Fraction(-1, 5):
        return '<-20pp'
    if value < Fraction(-1, 20):
        return '[-20,-5)pp'
    if value <= Fraction(1, 20):
        return '[-5,+5]pp'
    if value <= Fraction(1, 5):
        return '(+5,+20]pp'
    return '>+20pp'


def project_public(private_result):
    """Project one paired-difference band and an exploratory global guard.

    No group/cluster counts, identifiers, logits, coordinate or observations
    are returned. The guard passes if both global metric differences are at
    least -5 percentage points; it is exploratory, not evidence of improvement.
    Five groups in a cluster are not five independent games. This projection
    does not guarantee formal anonymity. The caller must independently review
    privacy before publication and avoid small private cross-tables.
    """
    _require(isinstance(private_result, dict), 'Invalid private result')
    if private_result == {'coverage': 'insufficient'}:
        return {'coverage': 'insufficient'}
    _require(set(private_result) == {'coverage', 'metrics'}
             and private_result['coverage'] == 'sufficient'
             and isinstance(private_result['metrics'], dict)
             and set(private_result['metrics']) == set(METRICS), 'Invalid private result')
    differences = {}
    for name in METRICS:
        item = private_result['metrics'][name]
        _require(isinstance(item, dict) and set(item) == {'model_a', 'model_b'},
                 'Invalid private metric')
        a, b = item['model_a'], item['model_b']
        _validate_rate(a)
        _validate_rate(b)
        differences[name] = a - b
    guard_passed = all(differences[name] >= Fraction(-1, 20)
                       for name in ('overall_teacher_top1', 'overall_same_target_plant'))
    return {'coverage': 'sufficient',
            'conditional_teacher_top1': {
                'model_a_minus_model_b_band': _difference_band(differences['conditional_teacher_top1'])},
            'global_guard': 'pass' if guard_passed else 'fail'}
