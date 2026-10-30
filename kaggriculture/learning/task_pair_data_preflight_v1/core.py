"""Pure ragged-data preflight; no filesystem, model, simulation or output.

Callers authenticate complete sources, schema and private role assignments.
Exact results and per-group masks stay private and must not become model input.
The training matrix is neither copied nor converted into per-group objects.
"""
from dataclasses import dataclass
import hashlib

import numpy as np

from learning.task_choice_v1.features import OPS, SCHEMA_VERSION, schema_hash


ROLES = ('train', 'teacher_dev', 'conditional_dev')
ROLE_COUNTS = {'train': 8, 'teacher_dev': 2, 'conditional_dev': 4}
MAX_GROUP_WIDTH = 65536
BLOCK_ROWS = 8192


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _schema(schema):
    fields = {'schema_version', 'schema_sha256', 'shared_feature_names', 'feature_names', 'ops'}
    _require(type(schema) is dict and set(schema) == fields, 'Invalid feature schema')
    shared, local = schema['shared_feature_names'], schema['feature_names']
    _require(type(shared) is list and len(shared) == 275 and type(local) is list and len(local) == 98,
             'Invalid feature schema')
    _require(all(type(name) is str and name for name in shared + local)
             and len(set(shared)) == len(shared) and len(set(local)) == len(local),
             'Invalid feature schema')
    _require(schema['schema_version'] == SCHEMA_VERSION and schema['ops'] == list(OPS)
             and schema['schema_sha256'] == schema_hash(local, shared), 'Invalid feature schema')
    required = ('unit.index', 'unit.x', 'unit.y', 'target.x', 'target.y', 'task.distance')
    _require('step' in shared and 'config.boardSize' in shared
             and all(name in local for name in required)
             and all('task.op.' + op in local for op in OPS), 'Missing identity feature')
    return tuple(shared), tuple(local), schema['schema_sha256']


@dataclass(frozen=True, repr=False)
class RaggedView:
    shared: np.ndarray
    candidates: np.ndarray
    offsets: np.ndarray
    labels: np.ndarray
    ops: np.ndarray
    private_clusters: np.ndarray
    shared_names: tuple
    feature_names: tuple
    schema_digest: str
    selected_ops: object = None
    candidate_identity: object = None

    @classmethod
    def from_cache(cls, arrays, schema):
        keys = {'shared', 'candidates', 'offsets', 'labels', 'selected_ops',
                'candidate_ops', 'candidate_identity', 'seeds'}
        _require(type(arrays) is dict and set(arrays) == keys, 'Invalid cache fields')
        names, features, digest = _schema(schema)
        return cls(arrays['shared'], arrays['candidates'], arrays['offsets'], arrays['labels'],
                   arrays['candidate_ops'], arrays['seeds'], names, features, digest,
                   arrays['selected_ops'], arrays['candidate_identity'])

    @classmethod
    def from_conditional(cls, arrays, schema):
        keys = {'shared', 'candidates', 'offsets', 'teacher_choice_label', 'ops', 'seed'}
        _require(type(arrays) is dict and set(arrays) == keys, 'Invalid conditional fields')
        names, features, digest = _schema(schema)
        return cls(arrays['shared'], arrays['candidates'], arrays['offsets'],
                   arrays['teacher_choice_label'], arrays['ops'], arrays['seed'],
                   names, features, digest)


@dataclass(frozen=True, repr=False)
class PrivatePreflightResult:
    coverage: dict
    overlap: dict
    masks: dict

    def __repr__(self):
        return '<PrivatePreflightResult>'


def _structure(view):
    _require(type(view) is RaggedView, 'Invalid ragged view')
    _require(type(view.shared_names) is tuple and type(view.feature_names) is tuple,
             'Invalid feature schema')
    _schema({'schema_version': SCHEMA_VERSION, 'schema_sha256': view.schema_digest,
             'shared_feature_names': list(view.shared_names),
             'feature_names': list(view.feature_names), 'ops': list(OPS)})
    arrays = (view.shared, view.candidates, view.offsets, view.labels, view.ops, view.private_clusters)
    _require(all(isinstance(value, np.ndarray) for value in arrays), 'Invalid array')
    _require(view.shared.dtype == np.dtype('float32') and view.candidates.dtype == np.dtype('float32')
             and view.offsets.dtype == np.dtype('int64') and view.labels.dtype == np.dtype('int64')
             and view.ops.dtype in (np.dtype('int8'), np.dtype('uint8'))
             and view.private_clusters.dtype in (np.dtype('int64'), np.dtype('uint32')),
             'Invalid array dtype')
    _require(view.shared.ndim == 2 and view.shared.shape[1] == 275 and len(view.shared) > 0
             and view.candidates.ndim == 2 and view.candidates.shape[1] == 98,
             'Invalid ragged shape')
    n, m = len(view.shared), len(view.candidates)
    _require(view.offsets.shape == (n + 1,) and view.labels.shape == (n,)
             and view.private_clusters.shape == (n,) and view.ops.shape == (m,),
             'Invalid ragged shape')
    if view.selected_ops is not None or view.candidate_identity is not None:
        _require(isinstance(view.selected_ops, np.ndarray) and view.selected_ops.dtype == np.dtype('int8')
                 and view.selected_ops.shape == (n,) and isinstance(view.candidate_identity, np.ndarray)
                 and view.candidate_identity.dtype == np.dtype('int16')
                 and view.candidate_identity.shape == (m, 3), 'Invalid cache identity')
    _require(int(view.offsets[0]) == 0 and int(view.offsets[-1]) == m, 'Invalid group offsets')


def _integer(values, upper):
    return bool(np.isfinite(values).all() and np.all(values >= 0)
                and np.all(values < upper) and np.all(values == np.floor(values)))


def _group(view, group):
    start, end = int(view.offsets[group]), int(view.offsets[group + 1])
    _require(0 <= start < end <= len(view.candidates) and end - start <= MAX_GROUP_WIDTH,
             'Invalid group bounds')
    label = int(view.labels[group])
    _require(0 <= label < end - start, 'Invalid teacher label')
    shared, rows, operations = view.shared[group], view.candidates[start:end], view.ops[start:end]
    _require(np.isfinite(shared).all() and np.isfinite(rows).all(), 'Nonfinite numeric input')
    _require(np.all(operations >= 0) and np.all(operations < len(OPS)), 'Invalid operation')
    shared_index = {name: i for i, name in enumerate(view.shared_names)}
    index = {name: i for i, name in enumerate(view.feature_names)}
    step, size = shared[shared_index['step']], shared[shared_index['config.boardSize']]
    _require(_integer(np.asarray([step]), 719) and _integer(np.asarray([size]), float('inf')) and size > 0,
             'Invalid encoded time or board')
    unit, ux, uy = (rows[:, index[name]] for name in ('unit.index', 'unit.x', 'unit.y'))
    x, y, distance = (rows[:, index[name]] for name in ('target.x', 'target.y', 'task.distance'))
    for value in (unit, ux, uy, x, y):
        _require(_integer(value, 32768), 'Invalid candidate identity')
    _require(all(np.all(value < size) for value in (ux, uy, x, y)), 'Candidate outside board')
    _require(_integer(distance, 65535)
             and np.array_equal(distance, np.abs(ux.astype(np.float64) - x.astype(np.float64))
                                + np.abs(uy.astype(np.float64) - y.astype(np.float64))),
             'Invalid candidate distance')
    for code, op in enumerate(OPS):
        _require(np.array_equal(rows[:, index['task.op.' + op]], (operations == code).astype(np.float32)),
                 'Invalid operation feature')
    if view.selected_ops is not None:
        _require(int(view.selected_ops[group]) == int(operations[label]), 'Selected operation mismatch')
        identities = view.candidate_identity[start:end]
        _require(np.array_equal(identities[:, 0], unit) and np.array_equal(identities[:, 1], x)
                 and np.array_equal(identities[:, 2], y), 'Candidate identity mismatch')
    if len(rows) > 1:
        # The frozen source sorts by (unit, target-y, target-x, operation-name).
        rank = np.asarray([sorted(OPS).index(op) for op in OPS], dtype=np.int8)[operations]
        u0, u1, y0, y1, x0, x1 = unit[:-1], unit[1:], y[:-1], y[1:], x[:-1], x[1:]
        increasing = ((u1 > u0) | ((u1 == u0) & ((y1 > y0) | ((y1 == y0)
                      & ((x1 > x0) | ((x1 == x0) & (rank[1:] > rank[:-1])))))))
        _require(np.all(increasing), 'Unordered or duplicate candidate identity')
        same_unit = u1 == u0
        _require(np.all((~same_unit) | ((ux[1:] == ux[:-1]) & (uy[1:] == uy[:-1]))),
                 'Inconsistent unit position')
    plant = OPS.index('PLANT')
    same = ((operations == plant) & (x == x[label]) & (y == y[label]))
    qualifying = bool(operations[label] == plant and np.count_nonzero(same) >= 2
                      and len(np.unique(unit[same])) >= 2)
    return start, end, label, int(operations[label]), qualifying


def _fingerprint(view, group):
    """Match NumericEquivalence: width + ordered float32 data; normalize signed zero."""
    start, end = int(view.offsets[group]), int(view.offsets[group + 1])
    digest = hashlib.sha256()
    digest.update(np.asarray([end - start], dtype='<i8').tobytes())
    shared = np.array(view.shared[group], dtype='<f4', order='C', copy=True)
    shared[shared == 0] = 0.0
    digest.update(shared.tobytes())
    for row in range(start, end, BLOCK_ROWS):
        block = np.array(view.candidates[row:min(row + BLOCK_ROWS, end)], dtype='<f4', order='C', copy=True)
        block[block == 0] = 0.0
        digest.update(block.tobytes())
    return digest.digest()


def _equal_group(a, ai, b, bi, check):
    alo, ahi = int(a.offsets[ai]), int(a.offsets[ai + 1])
    blo, bhi = int(b.offsets[bi]), int(b.offsets[bi + 1])
    if ahi - alo != bhi - blo:
        return False
    check()
    if not np.array_equal(a.shared[ai], b.shared[bi]):
        return False
    for offset in range(0, ahi - alo, BLOCK_ROWS):
        length = min(BLOCK_ROWS, ahi - alo - offset)
        check()
        if not np.array_equal(a.candidates[alo + offset:alo + offset + length],
                              b.candidates[blo + offset:blo + offset + length]):
            return False
    return True


def _coverage():
    return {'groups': 0, 'qualifying_plant_groups': 0, 'operation_groups': {op: 0 for op in OPS},
            '_operation_clusters': {op: set() for op in OPS}, '_qualifying_by_cluster': {}}


def _observe_coverage(stats, cluster, operation, qualifying):
    stats['groups'] += 1
    stats['operation_groups'][OPS[operation]] += 1
    stats['_operation_clusters'][OPS[operation]].add(cluster)
    stats['_qualifying_by_cluster'].setdefault(cluster, 0)
    if qualifying:
        stats['qualifying_plant_groups'] += 1
        stats['_qualifying_by_cluster'][cluster] += 1


def _finish_coverage(stats):
    return {'groups': stats['groups'], 'qualifying_plant_groups': stats['qualifying_plant_groups'],
            'operations': {op: {'groups': stats['operation_groups'][op],
                                'clusters_with_groups': len(stats['_operation_clusters'][op])} for op in OPS},
            'qualifying_per_cluster_sorted': sorted(stats['_qualifying_by_cluster'].values())}


def _roles(role_views, expected_roles, check):
    _require(type(role_views) is dict and set(role_views) == set(ROLES)
             and type(expected_roles) is dict and set(expected_roles) == set(ROLES), 'Invalid roles')
    expected = {}
    for role in ROLES:
        values = expected_roles[role]
        _require(type(values) in (list, tuple, set, frozenset)
                 and len(values) == ROLE_COUNTS[role]
                 and all(type(v) is int and 0 <= v <= 0xffffffff for v in values)
                 and len(set(values)) == len(values), 'Invalid role identities')
        expected[role] = set(values)
        _require(type(role_views[role]) in (list, tuple) and len(role_views[role]) > 0, 'Missing source')
    _require(all(expected[a].isdisjoint(expected[b]) for i, a in enumerate(ROLES)
                 for b in ROLES[i + 1:]), 'Overlapping source roles')
    reference_schema = None
    for role in ROLES:
        observed = set()
        for view in role_views[role]:
            check()
            _structure(view)
            identity = (view.shared_names, view.feature_names, view.schema_digest)
            if reference_schema is None:
                reference_schema = identity
            _require(identity == reference_schema, 'Source schema differs')
            for start in range(0, len(view.private_clusters), BLOCK_ROWS):
                check()
                block = view.private_clusters[start:start + BLOCK_ROWS]
                _require(np.all(block >= 0) and np.all(block <= 0xffffffff), 'Invalid source role')
                observed.update(int(value) for value in np.unique(block))
        _require(observed == expected[role], 'Source roles differ')


def analyze(train_views, teacher_dev_views, conditional_dev_views, expected_roles, *, check):
    """Return private group coverage, duplicate categories and evaluation masks.

    `check()` is called for every group and before every numeric equality
    comparison. It may stop the caller's run by raising. Numeric SHA matches
    are verified against bounded slices, so collisions never become matches.
    Identical input with any different training label is `label_conflict=True`,
    including when a same-label training record also exists. No sample is removed.
    Role membership is verified on the supplied arrays; completeness and source
    authenticity remain the caller's responsibility.
    """
    _require(callable(check), 'Missing resource callback')
    roles = dict(train=train_views, teacher_dev=teacher_dev_views, conditional_dev=conditional_dev_views)
    _roles(roles, expected_roles, check)
    coverage = {role: {'aggregate': _coverage(), 'sources': [_coverage() for _ in roles[role]]}
                for role in ROLES}
    masks = {role: [dict(seen_in_train=np.zeros(len(v.labels), dtype=bool),
                         label_conflict=np.zeros(len(v.labels), dtype=bool),
                         qualifying_plant=np.zeros(len(v.labels), dtype=bool)) for v in roles[role]]
             for role in ('teacher_dev', 'conditional_dev')}
    fingerprints = {}

    for role in ('teacher_dev', 'conditional_dev', 'train'):
        for source, view in enumerate(roles[role]):
            for group in range(len(view.labels)):
                check()
                _, _, label, operation, qualifying = _group(view, group)
                cluster = int(view.private_clusters[group])
                _observe_coverage(coverage[role]['aggregate'], cluster, operation, qualifying)
                _observe_coverage(coverage[role]['sources'][source], cluster, operation, qualifying)
                fingerprint = _fingerprint(view, group)
                buckets = fingerprints.get(fingerprint, [])
                match = None
                for bucket in buckets:
                    check()
                    if _equal_group(view, group, bucket['view'], bucket['group'], check):
                        match = bucket
                        break
                if role == 'train':
                    if match is not None:
                        match['training_labels'].add(label)
                    continue
                masks[role][source]['qualifying_plant'][group] = qualifying
                if match is None:
                    match = dict(view=view, group=group, training_labels=set(), references=[])
                    fingerprints.setdefault(fingerprint, []).append(match)
                match['references'].append((role, source, group, label))

    for buckets in fingerprints.values():
        for bucket in buckets:
            labels = bucket['training_labels']
            for role, source, group, label in bucket['references']:
                masks[role][source]['seen_in_train'][group] = bool(labels)
                masks[role][source]['label_conflict'][group] = any(value != label for value in labels)

    overlap = {}
    for role in ('teacher_dev', 'conditional_dev'):
        sources = []
        for source, view in enumerate(roles[role]):
            seen = masks[role][source]['seen_in_train']
            conflict = masks[role][source]['label_conflict']
            eligible = masks[role][source]['qualifying_plant']
            def categories(selected):
                return {'unmatched': int(np.count_nonzero(selected & ~seen)),
                        'same_label_only': int(np.count_nonzero(selected & seen & ~conflict)),
                        'label_conflict': int(np.count_nonzero(selected & conflict))}
            sources.append({'all': categories(np.ones(len(view.labels), dtype=bool)),
                            'qualifying_plant': categories(eligible)})
        overlap[role] = sources
    finished = {role: {'aggregate': _finish_coverage(data['aggregate']),
                       'sources': [_finish_coverage(value) for value in data['sources']]}
                for role, data in coverage.items()}
    return PrivatePreflightResult(finished, overlap, masks)
