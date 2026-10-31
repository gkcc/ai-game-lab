"""NumPy inference for a WATER-only residual on a frozen task model.

The caller authenticates the checkpoint and metadata before constructing the
selector and must bound resources. No files, models, games, or training are
opened or started here. Opportunities come from the existing controller and
are not an official legal-action mask. WATER means only that task operation;
this router does not decide crop choice or add other action constraints.
"""
import math

import numpy as np

from learning.task_bc_v1.model import DIMENSIONS, continuous_mask
from learning.task_choice_v1.features import OPS, SCHEMA_VERSION, candidate_id, schema_hash


SHARED_DIM, LOCAL_DIM = 275, 98
DIM = SHARED_DIM + LOCAL_DIM


class ResidualError(ValueError):
    """The supplied task schema, features, or residual is inconsistent."""


def _require(condition):
    if not condition:
        raise ResidualError('Invalid WATER residual input')


def _schema(metadata):
    _require(type(metadata) is dict and metadata.get('schema_version') == SCHEMA_VERSION
             and metadata.get('ops') == list(OPS) and metadata.get('dimensions') == list(DIMENSIONS))
    shared, local = metadata.get('shared_feature_names'), metadata.get('feature_names')
    _require(type(shared) is list and len(shared) == SHARED_DIM
             and type(local) is list and len(local) == LOCAL_DIM
             and all(type(name) is str and name for name in shared + local)
             and len(set(shared + local)) == DIM)
    digest = metadata.get('schema_sha256')
    _require(type(digest) is str and digest == schema_hash(local, shared))
    ops = tuple('task.op.' + op for op in OPS)
    _require(all(local.count(name) == 1 and name not in shared for name in ops))
    _require(all(local.count(name) == 1 for name in ('unit.index', 'target.x', 'target.y')))
    return (tuple(shared), tuple(local), digest, tuple(SHARED_DIM + local.index(name) for name in ops))


def _vector(value, dtype, shape):
    _require(type(value) is np.ndarray and value.dtype == np.dtype(dtype)
             and value.shape == shape and np.isfinite(value).all())
    return value


def _normalization(mean, scale, schema):
    mean, scale = _vector(mean, 'float64', (DIM,)), _vector(scale, 'float64', (DIM,))
    _require((scale >= 1).all())
    mask = continuous_mask(list(schema[0] + schema[1]))
    _require(np.all(mean[~mask] == 0) and np.all(scale[~mask] == 1))
    return mean, scale


def _operations(raw, schema):
    columns = raw[:, schema[3]]
    _require(columns.shape == (len(raw), len(OPS)) and np.isin(columns, (0.0, 1.0)).all()
             and np.all(columns.sum(axis=1) == 1.0))
    return columns


def select(raw, baseline_logits, mean, scale, weights, metadata):
    """Return the selected Python integer index for one candidate set.

    ``raw`` is the exact float32 training representation; baseline logits are
    the original float64 TaskNetwork scores, not float32-cast scores. Routing
    first chooses the baseline global argmax. Only when it is WATER can the
    residual change the selection, and then only among WATER candidates. A
    zero vector is an exact baseline identity even at extreme finite inputs.
    """
    schema = _schema(metadata)
    _require(type(raw) is np.ndarray and raw.dtype == np.dtype('float32') and raw.ndim == 2
             and raw.shape[1] == DIM and len(raw) > 0 and np.isfinite(raw).all())
    shared = raw[:, :SHARED_DIM]
    _require(np.array_equal(shared, np.broadcast_to(shared[:1], shared.shape)))
    logits = _vector(baseline_logits, 'float64', (len(raw),))
    mean, scale = _normalization(mean, scale, schema)
    weights = _vector(weights, 'float32', (DIM,))
    operations = _operations(raw, schema)
    baseline = int(np.argmax(logits))
    water = operations[:, OPS.index('WATER')] == 1.0
    routed = bool(water[baseline])
    if not routed or not np.any(weights):
        return baseline
    indices = np.flatnonzero(water)
    # Match the planned training representation exactly; the base model's
    # logits remain float64 throughout the residual comparison.
    with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
        normalized = ((raw[indices].astype(np.float64) - mean) / scale).astype(np.float32)
        residual = normalized @ weights
        adjusted = logits[indices] + residual.astype(np.float64)
    _require(normalized.dtype == np.dtype('float32') and np.isfinite(normalized).all()
             and residual.dtype == np.dtype('float32') and np.isfinite(residual).all()
             and adjusted.dtype == np.dtype('float64') and np.isfinite(adjusted).all())
    chosen = int(indices[int(np.argmax(adjusted))])
    return chosen


def _numeric_list(values, width):
    _require(type(values) is list and len(values) == width)
    try:
        valid = all(type(value) in (int, float) and math.isfinite(value) for value in values)
    except OverflowError:
        valid = False
    _require(valid)


class WaterResidualSelector:
    """Callable ``TaskAgent`` selector using a separately authenticated model.

    The caller supplies the expected authenticated metadata. Its schema must
    match the network and each runtime input. The checkpoint's file identity
    and parameter immutability outside this selector are caller obligations.
    ``last`` contains only anonymous indexes and is cleared on every call.
    """

    def __init__(self, network, weights, metadata):
        expected = _schema(metadata)
        _require(callable(getattr(network, 'score', None)) and _schema(network.metadata) == expected)
        mean, scale = _normalization(network.mean, network.scale, expected)
        weights = _vector(weights, 'float32', (DIM,))
        self._network = network
        self._schema = expected
        # Immutable copies prevent a caller editing the learned residual or
        # normalization through references retained at construction.
        self._weights = np.frombuffer(weights.tobytes(), dtype=np.float32)
        self._mean = np.frombuffer(mean.tobytes(), dtype=np.float64)
        self._scale = np.frombuffer(scale.tobytes(), dtype=np.float64)
        self._metadata = dict(schema_version=SCHEMA_VERSION, ops=list(OPS), dimensions=list(DIMENSIONS),
                              shared_feature_names=list(expected[0]), feature_names=list(expected[1]),
                              schema_sha256=expected[2])
        self.last = None

    def _check_model(self):
        _require(_schema(self._network.metadata) == self._schema)
        mean, scale = _normalization(self._network.mean, self._network.scale, self._schema)
        _require(np.array_equal(mean, self._mean) and np.array_equal(scale, self._scale))

    def __call__(self, inputs):
        self.last = None
        self._check_model()
        _require(type(inputs) is dict and inputs.get('schema_version') == SCHEMA_VERSION
                 and inputs.get('schema_sha256') == self._schema[2]
                 and inputs.get('shared_feature_names') == list(self._schema[0])
                 and inputs.get('feature_names') == list(self._schema[1]))
        shared, candidates = inputs.get('shared_x'), inputs.get('candidates')
        _numeric_list(shared, SHARED_DIM)
        _require(type(candidates) is list and len(candidates) > 0)
        local_names = self._schema[1]
        ids, rows = [], []
        for candidate in candidates:
            _require(type(candidate) is dict and set(candidate) == {'id', 'unit', 'target', 'op', 'x'})
            unit, target, op = candidate['unit'], candidate['target'], candidate['op']
            _require(type(unit) is int and 0 <= unit < 32768 and type(target) is list and len(target) == 2
                     and all(type(value) is int and 0 <= value < 32768 for value in target)
                     and type(op) is str and op in OPS and type(candidate['id']) is str
                     and candidate['id'] == candidate_id(unit, target, op))
            row = candidate['x']
            _numeric_list(row, LOCAL_DIM)
            _require(row[local_names.index('unit.index')] == unit
                     and row[local_names.index('target.x')] == target[0]
                     and row[local_names.index('target.y')] == target[1]
                     and all(row[local_names.index('task.op.' + kind)] == int(kind == op) for kind in OPS))
            ids.append(candidate['id'])
            rows.append(row)
        _require(len(ids) == len(set(ids)))
        shared64, local64 = np.asarray(shared, dtype=np.float64), np.asarray(rows, dtype=np.float64)
        _require(shared64.shape == (SHARED_DIM,) and local64.shape == (len(ids), LOCAL_DIM)
                 and np.isfinite(shared64).all() and np.isfinite(local64).all())
        # Call score with the same float64 values as TaskNetwork.__call__,
        # before constructing the raw32 residual input. No logits are returned.
        logits = self._network.score(shared64.copy(), local64.copy())
        self._check_model()
        with np.errstate(over='ignore', invalid='ignore'):
            raw = np.concatenate((np.broadcast_to(shared64, (len(ids), SHARED_DIM)), local64), axis=1).astype(np.float32)
        _require(np.isfinite(raw).all())
        selected = select(raw, logits, self._mean, self._scale, self._weights, self._metadata)
        self.last = selected
        return ids[selected]
