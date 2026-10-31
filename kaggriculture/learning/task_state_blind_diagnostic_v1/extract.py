"""Private, ordered task features from a single authenticated callback."""
from dataclasses import dataclass

import numpy as np

from learning.task_choice_v1.features import CANDIDATE_SCOPE, OPS, SCHEMA_VERSION, candidate_id
from learning.task_disagreement_trace_v1 import extract as prior
from learning.task_multisource_rank_v1 import dedup
from learning.task_water_residual_v1 import policy
from .sampling import ROLES


@dataclass(frozen=True, slots=True, repr=False)
class PrivateExample:
    item: dedup.Item
    step: int
    ordinal: int
    student_index: int

    def __repr__(self):
        return '<PrivateBlindExample>'


def extract_example(*, role, step, ordinal, inputs, teacher_label, chosen, metadata, guard):
    """Return numeric copied features; preserve the actual A05/A06 role."""
    prior._require(callable(guard))
    guard()
    prior._require(type(role) is str and role in ROLES
        and type(step) is int and 0 <= step < 719
        and type(ordinal) is int and 0 <= ordinal < 32
        and type(inputs) is dict and set(inputs) == prior._INPUT_KEYS
        and type(teacher_label) is str and type(chosen) is str)
    schema = prior._safe_schema(metadata)
    prior._require(inputs['schema_version'] == SCHEMA_VERSION and inputs['schema_sha256'] == schema[2]
        and inputs['shared_feature_names'] == list(schema[0])
        and inputs['feature_names'] == list(schema[1]) and inputs['candidate_scope'] == CANDIDATE_SCOPE)
    positions, board_size = prior._single_player(inputs, step)
    planned = inputs['planned']
    prior._require(type(planned) is dict and set(planned) == prior._PLANNED_KEYS
        and planned['interpretation'] == prior._PLANNED_INTERPRETATION)
    available = planned['available_units']
    prior._require(type(available) is list and len(available) == len(set(available))
        and all(type(i) is int and 0 <= i < len(positions) for i in available))
    names = schema[1]
    required = ('unit.index', 'unit.x', 'unit.y', 'target.x', 'target.y', 'task.distance')
    prior._require(all(names.count(name) == 1 for name in required))
    prior._numeric(inputs['shared_x'], policy.SHARED_DIM)
    candidates = inputs['candidates']
    prior._require(type(candidates) is list and 1 <= len(candidates) <= prior.MAX_CANDIDATES)
    identifiers, rows = [], []
    for candidate in candidates:
        guard()
        prior._require(type(candidate) is dict and set(candidate) == prior._CANDIDATE_KEYS)
        unit, target, op, identifier = candidate['unit'], candidate['target'], candidate['op'], candidate['id']
        prior._require(type(unit) is int and 0 <= unit < min(len(positions), 32768) and unit in available
            and type(target) is list and len(target) == 2
            and all(type(x) is int and 0 <= x < min(board_size, 32768) for x in target)
            and type(op) is str and op in OPS and type(identifier) is str
            and identifier == candidate_id(unit, target, op))
        row = candidate['x']
        prior._numeric(row, policy.LOCAL_DIM)
        ux, uy = positions[unit]
        expected = {'unit.index': unit, 'unit.x': ux, 'unit.y': uy,
            'target.x': target[0], 'target.y': target[1],
            'task.distance': abs(ux - target[0]) + abs(uy - target[1])}
        expected.update({'task.op.' + kind: int(kind == op) for kind in OPS})
        prior._require(all(row[names.index(name)] == value for name, value in expected.items()))
        identifiers.append(identifier)
        rows.append(row)
    prior._require(len(identifiers) == len(set(identifiers))
        and identifiers.count(teacher_label) == identifiers.count(chosen) == 1)
    try:
        with np.errstate(over='ignore', invalid='ignore'):
            shared64 = np.asarray(inputs['shared_x'], dtype=np.float64)
            local64 = np.asarray(rows, dtype=np.float64)
            shared, local = shared64.astype(np.float32), local64.astype(np.float32)
        prior._require(shared.shape == (policy.SHARED_DIM,)
            and local.shape == (len(candidates), policy.LOCAL_DIM)
            and np.isfinite(shared64).all() and np.isfinite(local64).all()
            and np.isfinite(shared).all() and np.isfinite(local).all()
            and np.array_equal(shared.astype(np.float64), shared64)
            and np.array_equal(local.astype(np.float64), local64))
        raw = np.empty((len(candidates), policy.DIM), dtype=np.float32)
        raw[:, :policy.SHARED_DIM], raw[:, policy.SHARED_DIM:] = shared, local
        policy._operations(raw, schema)
        shared = np.frombuffer(shared.tobytes(order='C'), dtype=np.float32)
        local = np.frombuffer(local.tobytes(order='C'), dtype=np.float32).reshape(local.shape)
    except MemoryError:
        raise
    except prior.ExtractionError:
        raise
    except Exception:
        raise prior.ExtractionError('Invalid private blind example') from None
    guard()
    item = dedup.Item(shared, local, identifiers.index(teacher_label), 'new_dev', role)
    return PrivateExample(item, step, ordinal, identifiers.index(chosen))
