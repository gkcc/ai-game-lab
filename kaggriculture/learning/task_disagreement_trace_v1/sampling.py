"""Select conditionally labelled callbacks from an authenticated trace stream.

No seeds, model execution, outcomes or files are read here. Coordinates are
public anonymous role labels. Inputs and labels returned by this adapter are
private. The caller must authenticate the compressed source and wait until
its surrounding same-file final rehash has passed before using any samples.
The existing scanner authenticates step/model identity, callback identities,
runtime errors and model-budget continuity; it does not prove a game result or
validate every numerical feature's semantics.
"""
import hashlib
import json

from learning.task_student_continuation_v1 import samples as previous


TRAIN_ROLES = ('A03', 'A04')
DIAGNOSTIC_ROLES = ('A09', 'A10')
OPPONENTS = ('market', 'round2')
WINDOWS = ((0, 119), (120, 239), (240, 359),
           (360, 479), (480, 599), (600, 718))
PER_WINDOW = 4
MAX_CALLBACKS_PER_TURN = 32
DOMAIN = b'kaggriculture.task-disagreement-trace-v1.rank\x00'
FIELDS = frozenset(('role', 'opponent', 'side', 'step', 'ordinal', 'inputs', 'teacher', 'chosen'))


def coordinate_rank(role, opponent, side, step, ordinal):
    """Hash only a fixed anonymous coordinate, with explicit tie breakers."""
    if (type(role) is not str or role not in TRAIN_ROLES + DIAGNOSTIC_ROLES
            or type(opponent) is not str or opponent not in OPPONENTS
            or type(side) is not int or side not in (0, 1)
            or type(step) is not int or not 0 <= step < 719
            or type(ordinal) is not int or not 0 <= ordinal < MAX_CALLBACKS_PER_TURN):
        raise ValueError('Invalid public sampling coordinate')
    payload = json.dumps([role, opponent, side, step, ordinal], ensure_ascii=True,
                         separators=(',', ':')).encode('ascii')
    return hashlib.sha256(DOMAIN + payload).digest()


class RankedStream:
    """Single-use wrapper around the original 719-step scanner.

    Initial selection stores at most 24 raw lines. ``scan`` must complete and
    return after the original scanner has validated all steps and EOF. Only
    then can ``samples`` materialize the six private windows, once. For train,
    only teacher/student disagreements compete. For diagnostic, *all* task
    callbacks compete: labels, operation, result and payoff do not affect rank.
    The underlying stream remains owned by the caller, including on failure.
    """

    def __init__(self, stream, *, mode, role, opponent, side, guard):
        if (type(mode) is not str or mode not in ('train', 'diagnostic')
                or (mode == 'train' and role not in TRAIN_ROLES)
                or (mode == 'diagnostic' and role not in DIAGNOSTIC_ROLES)
                or not callable(guard) or not callable(getattr(stream, 'readline', None))
                or not callable(getattr(stream, 'read', None))):
            raise ValueError('Invalid trace sampling contract')
        coordinate_rank(role, opponent, side, 0, 0)
        self._stream, self._guard = stream, guard
        self._mode, self._role, self._opponent, self._side = mode, role, opponent, side
        self._state, self._lines, self._decoded, self._at_eof = 'ready', 0, 0, False
        self._ranked = [[] for _ in WINDOWS]

    def __repr__(self):
        return '<DisagreementTrace private>'

    def _consider(self, raw):
        try:
            doc = json.loads(raw, parse_constant=previous._invalid_constant)
        except (UnicodeError, json.JSONDecodeError):
            raise ValueError('Invalid private trace line') from None
        if (not isinstance(doc, dict) or type(doc.get('step')) is not int
                or doc['step'] != self._lines or not isinstance(doc.get('records'), list)
                or len(doc['records']) > MAX_CALLBACKS_PER_TURN):
            raise ValueError('Invalid private trace step or callbacks')
        step = doc['step']
        if not 0 <= step < 719:
            raise ValueError('Trace step exceeds registered range')
        window = next(i for i, (start, end) in enumerate(WINDOWS) if start <= step <= end)
        best = None
        for ordinal, record in enumerate(doc['records']):
            if ordinal % 64 == 0:
                self._guard()
            # Malformed records are still rejected by the unchanged scanner.
            if not isinstance(record, dict):
                continue
            if self._mode == 'train' and record.get('teacher_label') == record.get('chosen'):
                continue
            rank = (coordinate_rank(self._role, self._opponent, self._side, step, ordinal),
                    step, ordinal)
            if best is None or rank < best:
                best = rank
        if best is not None:
            ranked = self._ranked[window]
            ranked.append((best, raw))
            ranked.sort(key=lambda item: item[0])
            del ranked[PER_WINDOW:]
        self._lines += 1
        self._decoded += len(raw)
        if self._decoded > previous.MAX_DECODED:
            raise ValueError('Trace decoded-byte budget exceeded')

    def readline(self, size=-1):
        if self._state != 'scanning' or self._lines >= 719 or self._at_eof:
            raise RuntimeError('Trace reads require the active validated scan')
        self._guard()
        raw = self._stream.readline(size)
        if type(raw) is not bytes or not raw or len(raw) > previous.MAX_LINE:
            raise ValueError('Missing or oversized private trace line')
        self._consider(raw)
        return raw

    def read(self, size=-1):
        if self._state != 'scanning' or self._lines != 719 or self._at_eof or size != 1:
            raise RuntimeError('Only the final EOF check is available')
        self._guard()
        raw = self._stream.read(size)
        if type(raw) is not bytes:
            raise ValueError('Invalid private trace EOF check')
        if raw == b'':
            self._at_eof = True
        return raw

    def scan(self, *, task_sha256, crop_sha256):
        """Complete exactly one underlying scan without disclosing samples."""
        if self._state != 'ready':
            raise RuntimeError('A trace adapter permits only one scan')
        self._state = 'scanning'
        try:
            result = previous.scan(self, task_sha256=task_sha256,
                                   crop_sha256=crop_sha256, guard=self._guard)
            if (self._lines != 719 or not self._at_eof or type(result) is not dict
                    or type(result.get('lines')) is not int or result['lines'] != 719
                    or type(result.get('decoded_bytes')) is not int
                    or result['decoded_bytes'] != self._decoded):
                raise ValueError('Complete trace verification did not finish')
            summary = dict(lines=self._lines, decoded_bytes=self._decoded)
        except BaseException:
            self._state, self._ranked = 'failed', None
            raise
        self._state = 'complete'
        return summary

    def samples(self):
        """Consume six private ranked windows after the public scan returned.

        The caller must still wait for the surrounding source rehash. Output
        fields are exactly the registered anonymous coordinate, the agent's
        legal observation/candidate inputs and the two recorded choices.
        """
        if self._state != 'complete':
            raise RuntimeError('Complete trace verification is required')
        try:
            windows = []
            for ranked in self._ranked:
                window = []
                for rank, raw in ranked:
                    self._guard()
                    doc = json.loads(raw, parse_constant=previous._invalid_constant)
                    record = doc['records'][rank[2]]
                    inputs, teacher, chosen = record['inputs'], record['teacher_label'], record['chosen']
                    candidates = inputs.get('candidates') if isinstance(inputs, dict) else None
                    ids = [candidate.get('id') if isinstance(candidate, dict) else None
                           for candidate in candidates] if isinstance(candidates, list) else []
                    if (doc.get('step') != rank[1] or not ids
                            or any(type(identifier) is not str or not identifier for identifier in ids)
                            or len(ids) != len(set(ids)) or teacher not in ids or chosen not in ids
                            or (self._mode == 'train' and teacher == chosen)):
                        raise ValueError('Selected private callback differs')
                    value = dict(role=self._role, opponent=self._opponent, side=self._side,
                                 step=rank[1], ordinal=rank[2], inputs=inputs,
                                 teacher=teacher, chosen=chosen)
                    if set(value) != FIELDS:
                        raise ValueError('Selected callback fields differ')
                    window.append(value)
                windows.append(window)
        except BaseException:
            self._state, self._ranked = 'failed', None
            raise
        self._state, self._ranked = 'consumed', None
        return windows
