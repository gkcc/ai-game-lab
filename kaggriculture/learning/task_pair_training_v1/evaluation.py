"""Private developmental metrics for a ragged task-head candidate set.

This module only computes metrics from supplied arrays. It does not load a model,
source data or observations, and does not execute a game. Labels, cluster IDs
and masks are evaluation metadata, never deployment-model inputs. The returned
numeric means are private and must not be published without a separately
reviewed privacy projection. No per-cluster identity is returned.

The caller must authenticate the view, the mask-to-source mapping, and expected
cluster membership before use. Matching a numeric input or passing these checks
does not establish independent data, official legality or performance in games.
"""

import numpy as np

from learning.task_choice_v1.features import OPS


def _need(condition, message):
    if not condition:
        raise ValueError(message)


def _integer_vector(value, shape):
    return (isinstance(value, np.ndarray) and value.shape == shape
            and value.dtype.kind in 'iu')


def _validate(view, logits, selected, qualifying, expected_clusters):
    arrays = (view.shared, view.candidates, view.offsets, view.labels,
              view.ops, view.private_clusters)
    _need(all(isinstance(value, np.ndarray) for value in arrays), 'Expected ragged arrays')
    _need(view.shared.ndim == 2 and view.candidates.ndim == 2,
          'Invalid ragged shape')
    count, rows = len(view.shared), len(view.candidates)
    _need(count > 0 and view.candidates.dtype.kind == 'f', 'Invalid candidate input')
    _need(_integer_vector(view.offsets, (count + 1,))
          and _integer_vector(view.labels, (count,))
          and _integer_vector(view.ops, (rows,))
          and _integer_vector(view.private_clusters, (count,)), 'Invalid ragged metadata')
    _need(int(view.offsets[0]) == 0 and int(view.offsets[-1]) == rows
          and np.all(view.offsets[1:] > view.offsets[:-1])
          and np.all(view.offsets <= rows), 'Invalid group offsets')
    lengths = np.diff(view.offsets)
    _need(np.all(view.labels < lengths), 'Invalid teacher label')
    if view.labels.dtype.kind == 'i':
        _need(np.all(view.labels >= 0), 'Invalid teacher label')
    _need(np.all(view.ops < len(OPS)), 'Invalid operation')
    if view.ops.dtype.kind == 'i':
        _need(np.all(view.ops >= 0), 'Invalid operation')
    _need(isinstance(logits, np.ndarray) and logits.shape == (rows,)
          and logits.dtype.kind == 'f' and bool(np.isfinite(logits).all()),
          'Invalid logits')
    for mask in (selected, qualifying):
        _need(isinstance(mask, np.ndarray) and mask.shape == (count,)
              and mask.dtype == np.dtype(bool), 'Invalid group mask')
    _need(isinstance(view.feature_names, (tuple, list))
          and all(isinstance(name, str) and name for name in view.feature_names)
          and len(view.feature_names) == view.candidates.shape[1]
          and len(set(view.feature_names)) == len(view.feature_names),
          'Invalid feature names')
    names = {name: i for i, name in enumerate(view.feature_names)}
    needed = ('unit.index', 'target.x', 'target.y')
    _need(all(name in names for name in needed), 'Missing candidate identity feature')
    for name in needed:
        values = view.candidates[:, names[name]]
        _need(bool(np.isfinite(values).all()) and np.all(values >= 0)
              and np.all(values < 32768) and np.all(values == np.floor(values)),
              'Invalid candidate identity')
    expected = np.asarray(expected_clusters)
    _need(expected.ndim == 1 and len(expected) > 0 and expected.dtype.kind in 'iu'
          and len(np.unique(expected)) == len(expected), 'Invalid expected clusters')
    if expected.dtype.kind == 'i':
        _need(np.all(expected >= 0), 'Invalid expected clusters')
    _need(np.all(expected <= 0xffffffff)
          and np.all(np.isin(view.private_clusters[selected], expected)),
          'Selected group outside expected clusters')
    return names, expected


def _new_metric(cluster_count):
    return np.zeros(cluster_count, dtype=np.int64), np.zeros(cluster_count, dtype=np.int64)


def _observe(metric, cluster_index, success):
    counts, successes = metric
    counts[cluster_index] += 1
    successes[cluster_index] += int(success)


def _summarize(metric):
    counts, successes = metric
    supported = (len(counts) >= 2 and int(counts.sum()) >= 20
                 and int(np.count_nonzero(counts)) >= 2 and bool(np.all(counts >= 5)))
    if not supported:
        return {'support': 'unknown', 'value': None}
    return {'support': 'supported', 'value': float(np.mean(successes / counts))}


def evaluate(view, logits, selected_groups, qualifying_plant, expected_clusters):
    """Return private equal-cluster means and support flags for selected groups.

    ``qualifying_plant`` is the source-bound per-group mask from the existing
    data preflight. For every selected group its PLANT/same-target/distinct-unit
    meaning is recomputed and must agree. ``expected_clusters`` names the full
    expected population, including clusters absent from a selected subset; a
    missing or undersized cluster yields ``unknown``, never a zero or an omitted
    mean. The caller must separately apply source identity checks.

    "Cross-operation" means at least two different operation codes are present
    among the group's actual candidates; recall is conditioned on the teacher's
    selected operation within that subset. Ties use the earliest candidate in
    original order, including for the same-target conditional ranking.
    """
    names, expected = _validate(view, logits, selected_groups, qualifying_plant,
                                expected_clusters)
    cluster_positions = {int(cluster): i for i, cluster in enumerate(expected)}
    metrics = {key: _new_metric(len(expected)) for key in
               ('full_top1', 'full_target_operation', 'plant_same_target_unit')}
    recalls = {name: _new_metric(len(expected)) for name in OPS}
    plant = OPS.index('PLANT')

    for group in np.flatnonzero(selected_groups):
        lo, hi = int(view.offsets[group]), int(view.offsets[group + 1])
        label = int(view.labels[group])
        candidates = view.candidates[lo:hi]
        ops = view.ops[lo:hi]
        units = candidates[:, names['unit.index']]
        xs, ys = candidates[:, names['target.x']], candidates[:, names['target.y']]
        identities = np.column_stack((units, xs, ys, ops))
        _need(len(np.unique(identities, axis=0)) == len(identities),
              'Duplicate candidate identity')
        same_target = ((ops == plant) & (xs == xs[label]) & (ys == ys[label]))
        same_units = units[same_target]
        eligible = bool(ops[label] == plant and len(np.unique(same_units)) >= 2)
        _need(bool(qualifying_plant[group]) == eligible, 'PLANT qualification differs')

        prediction = int(np.argmax(logits[lo:hi]))
        cluster = cluster_positions[int(view.private_clusters[group])]
        _observe(metrics['full_top1'], cluster, prediction == label)
        _observe(metrics['full_target_operation'], cluster,
                 ops[prediction] == ops[label] and xs[prediction] == xs[label]
                 and ys[prediction] == ys[label])
        if eligible:
            members = np.flatnonzero(same_target)
            predicted_member = int(members[np.argmax(logits[lo:hi][members])])
            _observe(metrics['plant_same_target_unit'], cluster,
                     units[predicted_member] == units[label])
        if bool(np.any(ops != ops[0])):
            _observe(recalls[OPS[int(ops[label])]], cluster,
                     ops[prediction] == ops[label])

    result = {name: _summarize(metric) for name, metric in metrics.items()}
    result['cross_operation_recall'] = {name: _summarize(recalls[name]) for name in OPS}
    return result
