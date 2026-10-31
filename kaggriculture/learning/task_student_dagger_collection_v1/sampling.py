"""Rank private disagreements from one already-authenticated student trace.

No seed, future payoff, replay, model execution or result comparison enters the
ranking. The caller authenticates the source game and compressed trace. The
returned observations, candidate IDs and previous model identities are
private and must not be published.
"""
import hashlib
import json

from learning.task_student_continuation_v1 import samples as previous


ROLES = ('T01', 'T02', 'T03', 'T04', 'V01', 'V02')
OPPONENTS = ('market', 'round2')
WINDOWS = ((0, 119), (120, 239), (240, 359),
           (360, 479), (480, 599), (600, 718))
PER_WINDOW = 4
DOMAIN = b'kaggriculture.task-student-dagger-collection-v1.rank\x00'
PREVIOUS_FIELDS = ('step', 'task_checkpoint_sha256', 'checkpoint_sha256',
                   'load_errors', 'inference_errors', 'fallbacks', 'model_counts')


def coordinate_rank(role, opponent, side, step, ordinal):
    """Rank a public coordinate without using a source seed or game outcome."""
    if (role not in ROLES or opponent not in OPPONENTS
            or type(side) is not int or side not in (0, 1)
            or type(step) is not int or not 0 <= step < 719
            or type(ordinal) is not int or ordinal < 0):
        raise ValueError('Invalid public sampling coordinate')
    value = json.dumps([role, opponent, side, step, ordinal], ensure_ascii=True,
                       separators=(',', ':')).encode('ascii')
    return hashlib.sha256(DOMAIN + value).digest()


class RankedStream:
    """Single-use adapter around the existing full-episode trace scanner.

    ``scan(task_sha256=..., crop_sha256=...)`` calls the unchanged previous
    scanner and returns its byte/line/work accounting. Only after the full scan
    succeeds can ``candidates()`` return six ranked lists, each with at most
    four distinct steps. Each item supplies the existing compare interface's
    step, ordinal, third_candidate, document and previous_document fields.

    During streaming, retained whole turns are raw bytes rather than decoded
    objects. Result materialization samples the caller's guard per item; an
    error ends the adapter without returning partial candidates. Direct reads
    outside ``scan`` are refused so reaching EOF alone cannot unlock results.
    """

    def __init__(self, stream, *, role, opponent, side, guard):
        coordinate_rank(role, opponent, side, 0, 0)
        if (not callable(guard) or not callable(getattr(stream, 'readline', None))
                or not callable(getattr(stream, 'read', None))):
            raise ValueError('Invalid trace stream or guard')
        self._stream, self._guard = stream, guard
        self._role, self._opponent, self._side = role, opponent, side
        self._state, self._lines, self._at_eof = 'ready', 0, False
        self._previous = None
        self._ranked = [[] for _ in WINDOWS]

    def __repr__(self):
        return '<DaggerRankedStream private>'

    def _consider(self, raw):
        # This preliminary decode is never exposed unless the outer scanner has
        # accepted the complete source, including model and budget continuity.
        try:
            doc = json.loads(raw, parse_constant=previous._invalid_constant)
        except (UnicodeError, json.JSONDecodeError):
            raise ValueError('Invalid trace line') from None
        if (not isinstance(doc, dict) or type(doc.get('step')) is not int
                or doc['step'] != self._lines or not isinstance(doc.get('records'), list)):
            raise ValueError('Invalid trace step or callbacks')
        step = doc['step']
        if not 0 <= step < 719:
            raise ValueError('Trace step exceeds registered range')
        window = next(i for i, (start, end) in enumerate(WINDOWS) if start <= step <= end)
        best = None
        for ordinal, record in enumerate(doc['records']):
            # The previous scanner verifies every record's source, candidate
            # identities, model counts and errors before results are unlocked.
            if not isinstance(record, dict):
                continue
            teacher, chosen = record.get('teacher_label'), record.get('chosen')
            if not isinstance(teacher, str) or not isinstance(chosen, str) or teacher == chosen:
                continue
            inputs = record.get('inputs')
            candidates = inputs.get('candidates') if isinstance(inputs, dict) else None
            if not isinstance(candidates, list):
                continue
            rank = (coordinate_rank(self._role, self._opponent, self._side, step, ordinal),
                    step, ordinal)
            if best is None or rank < best[0]:
                ids = [candidate.get('id') if isinstance(candidate, dict) else None
                       for candidate in candidates]
                third = next((identifier for identifier in ids if identifier not in (teacher, chosen)), None)
                best = (rank, third)
        if best is not None:
            ranked = self._ranked[window]
            ranked.append((best[0], best[1], raw, self._previous))
            ranked.sort(key=lambda item: item[0])
            del ranked[PER_WINDOW:]
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
            self._ranked = self._previous = None
            raise
        self._state, self._previous = 'complete', None
        return {key: result[key] for key in ('decoded_bytes', 'lines', 'selection_work')}

    def candidates(self):
        """Materialize private records after, and only after, a complete scan."""
        if self._state != 'complete':
            raise RuntimeError('Complete trace verification is required')
        try:
            result = []
            for ranked in self._ranked:
                window = []
                for rank, third, raw, before in ranked:
                    self._guard()
                    doc = json.loads(raw, parse_constant=previous._invalid_constant)
                    prior = None if before is None else dict(before, model_counts=dict(before['model_counts']))
                    window.append(dict(step=rank[1], ordinal=rank[2], third_candidate=third,
                                       document=doc, previous_document=prior))
                result.append(window)
            return result
        except BaseException:
            self._state = 'failed'
            self._ranked = None
            raise
