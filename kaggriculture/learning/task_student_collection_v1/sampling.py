"""Select fixed ranked disagreements while the existing trace audit runs.

All selected documents and candidate identifiers are private. The caller must
authenticate the source game, trace, model identities and compressed bytes;
this module neither reads replay data nor infers action values. It runs no
model, training, official transition or complete game.
"""
import hashlib
import json

from learning.task_student_continuation_v1 import samples as previous


WINDOWS = ((0, 119), (120, 239), (240, 359),
           (360, 479), (480, 599), (600, 718))
DOMAIN = b'kaggriculture.task-student-collection-v1.rank\x00'
PREVIOUS_FIELDS = ('step', 'task_checkpoint_sha256', 'checkpoint_sha256',
                   'load_errors', 'inference_errors', 'fallbacks', 'model_counts')


def coordinate_rank(role, opponent, side, step, ordinal):
    """Digest only public coordinates; lower digest wins within a window."""
    if (role not in previous.ROLES or opponent not in previous.OPPONENTS
            or type(side) is not int or side not in (0, 1)
            or type(step) is not int or not 0 <= step < 719
            or type(ordinal) is not int or ordinal < 0):
        raise ValueError('Invalid public sampling coordinate')
    payload = json.dumps([role, opponent, side, step, ordinal], ensure_ascii=True,
                         separators=(',', ':')).encode('ascii')
    return hashlib.sha256(DOMAIN + payload).digest()


class RankedStream:
    """A single-use BinaryIO adapter for the old complete trace scanner.

    Call ``scan(task_sha256=..., crop_sha256=...)`` first. It invokes the
    unmodified previous scanner on this adapter and returns only its ordinary
    byte/line/work accounting. ``selected()`` then returns a six-element list
    of private selected documents or ``None`` for empty windows. A selected
    value has the same step, ordinal, third_candidate, document and compact
    previous_document fields consumed by the existing comparison interface.

    Direct reads outside ``scan`` are refused: observing EOF alone cannot prove
    the previous scanner returned successfully. Failures are non-retryable on
    this adapter; exceptions including resource and I/O errors propagate.
    """

    def __init__(self, stream, *, role, opponent, side, guard):
        coordinate_rank(role, opponent, side, 0, 0)
        if not callable(guard) or not callable(getattr(stream, 'readline', None)) \
                or not callable(getattr(stream, 'read', None)):
            raise ValueError('Invalid trace stream or guard')
        self._stream, self._guard = stream, guard
        self._role, self._opponent, self._side = role, opponent, side
        self._state, self._lines, self._at_eof = 'ready', 0, False
        self._previous = None
        self._best = [None] * len(WINDOWS)
        self._selected = [None] * len(WINDOWS)

    def __repr__(self):
        return '<RankedStream private>'

    def _consider(self, raw):
        # The outer scanner validates the same line before it advances. This
        # observation can never be returned if any of those checks fail.
        try:
            doc = json.loads(raw, parse_constant=previous._invalid_constant)
        except (UnicodeError, json.JSONDecodeError):
            raise ValueError('Invalid trace line') from None
        if not isinstance(doc, dict) or type(doc.get('step')) is not int \
                or doc['step'] != self._lines or not isinstance(doc.get('records'), list):
            raise ValueError('Invalid trace step or callbacks')
        step = doc['step']
        if not 0 <= step < 719:
            raise ValueError('Trace step exceeds registered range')
        window = next(i for i, (start, end) in enumerate(WINDOWS) if start <= step <= end)
        for ordinal, record in enumerate(doc['records']):
            # The old scanner checks source, counts, candidate identity and
            # membership. A malformed record must never become a result.
            if not isinstance(record, dict):
                continue
            teacher, chosen = record.get('teacher_label'), record.get('chosen')
            if not isinstance(teacher, str) or not isinstance(chosen, str) or teacher == chosen:
                continue
            inputs = record.get('inputs')
            candidates = inputs.get('candidates') if isinstance(inputs, dict) else None
            if not isinstance(candidates, list):
                continue
            ids = [candidate.get('id') if isinstance(candidate, dict) else None
                   for candidate in candidates]
            rank = (coordinate_rank(self._role, self._opponent, self._side, step, ordinal),
                    step, ordinal)
            if self._best[window] is None or rank < self._best[window]:
                third = next((item for item in ids if item not in (teacher, chosen)), None)
                self._best[window] = rank
                self._selected[window] = dict(step=step, ordinal=ordinal,
                    third_candidate=third, document=doc, previous_document=self._previous)
        try:
            self._previous = {key: doc[key] for key in PREVIOUS_FIELDS}
        except KeyError:
            raise ValueError('Missing previous-turn authentication fields') from None
        self._lines += 1

    def readline(self, size=-1):
        if self._state != 'scanning':
            raise RuntimeError('Trace reads require the validated scan')
        raw = self._stream.readline(size)
        if isinstance(raw, bytes) and 0 < len(raw) <= previous.MAX_LINE:
            self._guard()
            self._consider(raw)
        return raw

    def read(self, size=-1):
        if self._state != 'scanning':
            raise RuntimeError('Trace reads require the validated scan')
        raw = self._stream.read(size)
        if raw == b'':
            self._at_eof = True
        return raw

    def scan(self, *, task_sha256, crop_sha256):
        if self._state != 'ready':
            raise RuntimeError('A trace adapter permits only one scan')
        self._state = 'scanning'
        try:
            result = previous.scan(self, task_sha256=task_sha256,
                                   crop_sha256=crop_sha256, guard=self._guard)
            if self._lines != 719 or not self._at_eof:
                raise ValueError('Complete trace verification did not finish')
        except BaseException:
            self._state = 'failed'
            self._best = self._selected = self._previous = None
            raise
        self._state = 'complete'
        self._previous = self._best = None
        return {key: result[key] for key in ('decoded_bytes', 'lines', 'selection_work')}

    def selected(self):
        if self._state != 'complete':
            raise RuntimeError('Complete trace verification is required')
        return list(self._selected)
