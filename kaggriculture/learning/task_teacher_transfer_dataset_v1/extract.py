"""Extract conditional transfer examples without reading games or training.

The caller authenticates both complete single-player streams and owns their
resource limits. A group is an opportunity-level demonstration only: support
does not establish official legality, execution, reward, or full-game success.
Feature payloads are private and must not be logged or published.

The frozen alignment is the authority for each teacher callback. Cumulative
alignment on each teacher prefix supplies its exact supported/unsupported/
unknown classification; its own preparation routine locates the corresponding
terminal context. This deliberately trades at most 32 bounded rechecks for a
per-callback invariant without changing the frozen alignment interface.
"""
from dataclasses import dataclass
import math

import numpy as np

from decision.api import observe
from learning.task_teacher_choice_bc_v1.extract import extract as extract_callback
from learning.task_teacher_support_probe_v1 import alignment


SHARED_WIDTH = 275
CANDIDATE_WIDTH = 98
FEATURE_WIDTH = SHARED_WIDTH + CANDIDATE_WIDTH
_CLASSES = ('supported', 'unsupported', 'unknown')


class TransferError(ValueError):
    """The conditional transfer contract is inconsistent."""


@dataclass(frozen=True, slots=True, repr=False)
class PrivateGroup:
    """Read-only numeric copy, candidate-index label, and anonymous metadata."""

    _payload: bytes
    candidate_count: int
    label: int
    operation: str
    teacher_ordinal: int
    terminal_ordinal: int

    @property
    def features(self):
        # A bytes-backed array cannot be made writeable; later caller mutations
        # of inputs or of copies of this view cannot change the stored group.
        return np.frombuffer(self._payload, dtype=np.float32).reshape(self.candidate_count, FEATURE_WIDTH)

    def __repr__(self):
        return '<PrivateTransferGroup>'


def _require(condition):
    if not condition:
        raise TransferError('Inconsistent private transfer example')


def _check_numeric_vector(values, width):
    _require(type(values) is list and len(values) == width)
    try:
        valid = all(type(value) in (int, float) and math.isfinite(value) for value in values)
    except OverflowError:
        valid = False
    _require(valid)


def _check_raw_features(records, guard):
    """Reject coercible strings/bools before accepting float32 conversion."""
    for record in records:
        guard()
        inputs = record['inputs']
        _check_numeric_vector(inputs['shared_x'], SHARED_WIDTH)
        for candidate in inputs['candidates']:
            _check_numeric_vector(candidate['x'], CANDIDATE_WIDTH)


def _group(teacher, terminal, *, teacher_ordinal, terminal_ordinal, schema, view, guard):
    guard()
    left, right = teacher['record'], terminal['record']
    _require(teacher['prefix'] is not None and teacher['prefix'] == terminal['prefix']
             and teacher['reservation'] == terminal['reservation'])
    candidates = right['inputs']['candidates']
    matches = [i for i, candidate in enumerate(candidates) if candidate['id'] == left['chosen']]
    _require(len(matches) == 1)
    # This is the TERMINAL controller's encoded view. The opposing teacher's
    # features, observation, ID, scores, and private state are never returned.
    data = extract_callback(right, schema, side=view.observation['player'], step=view.observation['step'],
                            max_candidates=alignment.MAX_CANDIDATES)
    shared, local = data['shared'], data['candidates']
    n, label = len(candidates), matches[0]
    _require(1 <= n <= alignment.MAX_CANDIDATES and type(label) is int and 0 <= label < n
             and shared.shape == (SHARED_WIDTH,) and local.shape == (n, CANDIDATE_WIDTH)
             and shared.dtype == np.dtype(np.float32) and local.dtype == np.dtype(np.float32)
             and np.isfinite(shared).all() and np.isfinite(local).all())
    matrix = np.empty((n, FEATURE_WIDTH), dtype=np.float32)
    matrix[:, :SHARED_WIDTH] = shared
    matrix[:, SHARED_WIDTH:] = local
    _require(matrix.shape == (n, FEATURE_WIDTH) and matrix.dtype == np.dtype(np.float32)
             and np.isfinite(matrix).all())
    guard()
    return PrivateGroup(matrix.tobytes(order='C'), n, label, left['op'], teacher_ordinal, terminal_ordinal)


def extract_turn(observation, configuration, teacher_records, terminal_records, *, schema, guard):
    """Return ``(groups, anonymous_counts)`` for one authenticated player turn.

    A supported group's row ordering is exactly the terminal candidate order;
    its label is the index matching the teacher's semantic unit/target/op ID.
    The ID itself and both raw callback streams stay private. ``guard`` is
    mandatory and its failures propagate. Callers must not mutate any input
    while this function is running. Groups contain no references to inputs.
    """
    _require(callable(guard))
    guard()
    counts = alignment.align_turn(observation, configuration, teacher_records, terminal_records,
                                  schema=schema, guard=guard)
    _check_raw_features(teacher_records, guard)
    _check_raw_features(terminal_records, guard)
    # align_turn has already checked the complete projection and both streams.
    view = observe(observation, configuration)
    expected_schema = alignment._schema(schema)
    teachers = alignment._prepare(teacher_records, view=view, schema=expected_schema, guard=guard)
    terminals = alignment._prepare(terminal_records, view=view, schema=expected_schema, guard=guard)
    terminal_by_prefix = {}
    for ordinal, terminal in enumerate(terminals):
        guard()
        prefix = terminal['prefix']
        if prefix is not None:
            _require(prefix not in terminal_by_prefix)
            terminal_by_prefix[prefix] = (ordinal, terminal)
    groups, previous = [], {key: 0 for key in _CLASSES}
    for ordinal, teacher in enumerate(teachers):
        guard()
        # Classify THIS callback by exact change in frozen alignment counts;
        # a merely equal total count would not detect accidental mispairing.
        current = alignment.align_turn(observation, configuration, teacher_records[:ordinal + 1],
                                       terminal_records, schema=schema, guard=guard)
        delta = {key: current[key] - previous[key] for key in _CLASSES}
        _require(current['teacher_callbacks'] == ordinal + 1
                 and all(type(delta[key]) is int and delta[key] in (0, 1) for key in _CLASSES)
                 and sum(delta.values()) == 1)
        previous = {key: current[key] for key in _CLASSES}
        if delta['supported']:
            match = terminal_by_prefix.get(teacher['prefix']) if teacher['prefix'] is not None else None
            _require(match is not None)
            terminal_ordinal, terminal = match
            groups.append(_group(teacher, terminal, teacher_ordinal=ordinal,
                                 terminal_ordinal=terminal_ordinal, schema=expected_schema,
                                 view=view, guard=guard))
    _require(all(previous[key] == counts[key] for key in _CLASSES)
             and len(groups) == counts['supported'])
    guard()
    return tuple(groups), counts
