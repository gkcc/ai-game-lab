"""Private conversion of authenticated student-callback demonstrations.

This module never opens a trace, checkpoint, replay, or game. The caller must
authenticate the single-player callback, role, and model schema before use.
Only numeric features already present in the callback are copied; a teacher
label is a candidate index, never a model feature. These conditional labels
do not establish official execution, economic value, or generalization.
"""
from dataclasses import dataclass
import math

import numpy as np

from decision.api import observe
from learning.task_choice_v1.features import CANDIDATE_SCOPE, OPS, SCHEMA_VERSION, candidate_id
from learning.task_multisource_rank_v1 import dedup
from learning.task_water_residual_v1 import policy


ROLE_POOLS = {'A03': 'new_train', 'A04': 'new_train',
              'A09': 'new_dev', 'A10': 'new_dev'}
MAX_CANDIDATES = 256
MAX_ORDINAL = 32
_INPUT_KEYS = frozenset(('schema_version', 'schema_sha256', 'feature_names',
                         'shared_feature_names', 'shared_x', 'candidate_scope',
                         'observation', 'configuration', 'planned', 'candidates',
                         'inventory_item_order', 'shed_item_order'))
_PLANNED_KEYS = frozenset(('seeds', 'needed_seeds', 'supply', 'projected_shed',
                           'unit_actions', 'available_units', 'interpretation'))
_CANDIDATE_KEYS = frozenset(('id', 'unit', 'target', 'op', 'x'))
_PLANNED_INTERPRETATION = 'before assignment: reservations and requests, not official executed state'


class ExtractionError(ValueError):
    """A private demonstration failed its conversion contract."""


def _require(condition):
    if not condition:
        raise ExtractionError('Invalid private disagreement example')


@dataclass(frozen=True, slots=True, repr=False)
class PrivateExample:
    """One immutable private item and non-feature callback coordinates."""

    item: dedup.Item
    step: int
    ordinal: int
    student_index: int

    @property
    def role(self):
        return self.item.role

    def __repr__(self):
        return '<PrivateDisagreementExample>'


def _numeric(values, width):
    _require(type(values) is list and len(values) == width)
    try:
        valid = all(type(value) in (int, float) and math.isfinite(value) for value in values)
    except (OverflowError, TypeError, ValueError):
        valid = False
    _require(valid)


def _safe_schema(metadata):
    try:
        schema = policy._schema(metadata)
    except MemoryError:
        raise
    except Exception:
        raise ExtractionError('Invalid private disagreement example') from None
    # Crop-seed features such as seed.initial.WHEAT are ordinary observed
    # resources; replay RNG seeds and teacher/opponent-private fields are not.
    for name in schema[0] + schema[1]:
        lower = name.lower()
        _require(not any(token in lower for token in ('teacher', 'priority', 'opponent.private',
                                                       'opponent_private', 'random_seed', 'rng_seed',
                                                       'random.seed', 'rng.seed', 'info.seed',
                                                       'game.seed', 'episode.seed'))
                 and lower != 'seed')
    return schema


def _single_player(inputs, step):
    obs, cfg = inputs['observation'], inputs['configuration']
    _require(type(obs) is dict and type(cfg) is dict
             and type(obs.get('player')) is int and obs['player'] in (0, 1)
             and type(obs.get('step')) is int and obs['step'] == step
             and type(cfg.get('boardSize')) is int and cfg['boardSize'] > 0)
    try:
        view = observe(obs, cfg)
        _require(view.observation == obs and view.configuration == cfg)
        farms = obs['farms']
        _require(type(farms) is list and len(farms) == 2)
        own = farms[obs['player']]
        _require(type(own) is dict and type(own['hands']) is list)
        positions = [own['farmer'], *own['hands']]
        _require(all(type(pos) is list and len(pos) == 2
                     and all(type(value) is int and 0 <= value < cfg['boardSize'] for value in pos)
                     for pos in positions))
        private = obs['private']
        _require(inputs['inventory_item_order'] == [list(inv) for inv in private['inventories']]
                 and inputs['shed_item_order'] == list(private['shed']))
    except MemoryError:
        raise
    except ExtractionError:
        raise
    except Exception:
        raise ExtractionError('Invalid private disagreement example') from None
    return positions, cfg['boardSize']


def extract_example(*, role, step, ordinal, inputs, teacher_label, chosen, metadata, guard):
    """Convert one bound callback into ``PrivateExample``.

    ``role`` determines the fixed development split. ``step`` and ``ordinal``
    are private source coordinates, not model inputs. The chosen student ID
    is used only to validate disagreement and record its candidate index.
    ``guard`` is mandatory; its exceptions propagate unchanged. Callers must
    not mutate inputs while this function runs, and must not publish results.
    """
    _require(callable(guard))
    guard()
    _require(type(role) is str and role in ROLE_POOLS
             and type(step) is int and 0 <= step < 719
             and type(ordinal) is int and 0 <= ordinal < MAX_ORDINAL
             and type(inputs) is dict and set(inputs) == _INPUT_KEYS
             and type(teacher_label) is str and type(chosen) is str)
    schema = _safe_schema(metadata)
    _require(inputs['schema_version'] == SCHEMA_VERSION and inputs['schema_sha256'] == schema[2]
             and inputs['shared_feature_names'] == list(schema[0])
             and inputs['feature_names'] == list(schema[1])
             and inputs['candidate_scope'] == CANDIDATE_SCOPE)
    positions, board_size = _single_player(inputs, step)
    planned = inputs['planned']
    _require(type(planned) is dict and set(planned) == _PLANNED_KEYS
             and planned['interpretation'] == _PLANNED_INTERPRETATION)
    available = planned['available_units']
    _require(type(available) is list and all(type(i) is int and 0 <= i < len(positions) for i in available)
             and len(available) == len(set(available)))
    local_names = schema[1]
    needed = ('unit.index', 'unit.x', 'unit.y', 'target.x', 'target.y', 'task.distance')
    _require(all(local_names.count(name) == 1 for name in needed))
    _numeric(inputs['shared_x'], policy.SHARED_DIM)
    candidates = inputs['candidates']
    _require(type(candidates) is list and 1 <= len(candidates) <= MAX_CANDIDATES)
    identifiers, rows = [], []
    for candidate in candidates:
        guard()
        _require(type(candidate) is dict and set(candidate) == _CANDIDATE_KEYS)
        unit, target, op, identifier = candidate['unit'], candidate['target'], candidate['op'], candidate['id']
        _require(type(unit) is int and 0 <= unit < min(len(positions), 32768) and unit in available
                 and type(target) is list and len(target) == 2
                 and all(type(value) is int and 0 <= value < min(board_size, 32768) for value in target)
                 and type(op) is str and op in OPS and type(identifier) is str
                 and identifier == candidate_id(unit, target, op))
        row = candidate['x']
        _numeric(row, policy.LOCAL_DIM)
        ux, uy = positions[unit]
        expected = {'unit.index': unit, 'unit.x': ux, 'unit.y': uy,
                    'target.x': target[0], 'target.y': target[1],
                    'task.distance': abs(ux - target[0]) + abs(uy - target[1])}
        expected.update({'task.op.' + kind: int(kind == op) for kind in OPS})
        _require(all(row[local_names.index(name)] == value for name, value in expected.items()))
        identifiers.append(identifier)
        rows.append(row)
    _require(len(identifiers) == len(set(identifiers))
             and identifiers.count(teacher_label) == 1 and identifiers.count(chosen) == 1)
    target, student_index = identifiers.index(teacher_label), identifiers.index(chosen)
    _require(ROLE_POOLS[role] != 'new_train' or target != student_index)
    try:
        with np.errstate(over='ignore', invalid='ignore'):
            # The pinned encoder emits float32 features before serializing
            # them as JSON numbers. The recorded task model reads those
            # numbers as float64, so reject rather than silently round any
            # input which cannot make that exact round trip.
            shared64 = np.asarray(inputs['shared_x'], dtype=np.float64)
            local64 = np.asarray(rows, dtype=np.float64)
            shared = shared64.astype(np.float32)
            local = local64.astype(np.float32)
        _require(shared.shape == (policy.SHARED_DIM,)
                 and local.shape == (len(candidates), policy.LOCAL_DIM)
                 and np.isfinite(shared64).all() and np.isfinite(local64).all()
                 and np.isfinite(shared).all() and np.isfinite(local).all()
                 and np.array_equal(shared.astype(np.float64), shared64)
                 and np.array_equal(local.astype(np.float64), local64))
        raw = np.empty((len(candidates), policy.DIM), dtype=np.float32)
        raw[:, :policy.SHARED_DIM] = shared
        raw[:, policy.SHARED_DIM:] = local
        policy._operations(raw, schema)
        shared = np.frombuffer(shared.tobytes(order='C'), dtype=np.float32)
        local = np.frombuffer(local.tobytes(order='C'), dtype=np.float32).reshape(local.shape)
    except MemoryError:
        raise
    except ExtractionError:
        raise
    except Exception:
        raise ExtractionError('Invalid private disagreement example') from None
    guard()
    item = dedup.Item(shared, local, target, ROLE_POOLS[role], role)
    return PrivateExample(item, step, ordinal, student_index)
