"""Blind coordinate sampling of completed, authenticated student traces.

The owner must authenticate the source and finish the surrounding same-file
rehash before consuming the selected private callbacks. No game is executed.
"""
import hashlib
import json

from learning.task_student_continuation_v1 import samples as historical

ROLES = ('A05', 'A06')
OPPONENTS = ('market', 'round2')
WINDOWS = ((0, 119), (120, 239), (240, 359),
           (360, 479), (480, 599), (600, 718))
PER_WINDOW = 4
MAX_CALLBACKS = 32
_STEP_DOMAIN = b'kaggriculture.state-blind-diagnostic-v1.step\x00'
_ORDINAL_DOMAIN = b'kaggriculture.state-blind-diagnostic-v1.ordinal\x00'


def _rank(domain, coordinates):
    return hashlib.sha256(domain + json.dumps(coordinates, ensure_ascii=True,
        separators=(',', ':')).encode('ascii')).digest()


def step_rank(role, opponent, side, step):
    if (role not in ROLES or opponent not in OPPONENTS or type(side) is not int
            or side not in (0, 1) or type(step) is not int or not 0 <= step < 719):
        raise ValueError('Invalid anonymous sampling coordinate')
    return _rank(_STEP_DOMAIN, [role, opponent, side, step])


def ordinal_rank(role, opponent, side, step, ordinal):
    step_rank(role, opponent, side, step)
    if type(ordinal) is not int or not 0 <= ordinal < MAX_CALLBACKS:
        raise ValueError('Invalid anonymous sampling coordinate')
    return _rank(_ORDINAL_DOMAIN, [role, opponent, side, step, ordinal])


class RankedStream:
    """First rank eligible steps, then independently rank a callback in each.

    Neither ranking stage uses teacher/chosen labels, operation, model output
    or outcomes. The reused full scanner reads labels for semantic validation;
    its own first-disagreement selection is discarded. One step contributes at
    most one callback; no exclusions refill.
    """
    def __init__(self, stream, *, role, opponent, side, guard):
        step_rank(role, opponent, side, 0)
        if (not callable(guard) or not callable(getattr(stream, 'readline', None))
                or not callable(getattr(stream, 'read', None))):
            raise ValueError('Invalid private stream')
        self._stream, self._guard = stream, guard
        self._role, self._opponent, self._side = role, opponent, side
        self._state, self._lines, self._decoded, self._eof = 'ready', 0, 0, False
        self._ranked = [[] for _ in WINDOWS]

    def __repr__(self):
        return '<PrivateBlindDiagnosticStream>'

    def readline(self, size=-1):
        if self._state != 'scanning' or self._lines >= 719 or self._eof:
            raise RuntimeError('No active full trace scan')
        self._guard()
        raw = self._stream.readline(size)
        if type(raw) is not bytes or not raw or len(raw) > historical.MAX_LINE:
            raise ValueError('Invalid private trace line')
        try:
            doc = json.loads(raw, parse_constant=historical._invalid_constant)
        except (UnicodeError, json.JSONDecodeError):
            raise ValueError('Invalid private trace line') from None
        if (type(doc) is not dict or type(doc.get('step')) is not int
                or doc['step'] != self._lines or type(doc.get('records')) is not list
                or len(doc['records']) > MAX_CALLBACKS):
            raise ValueError('Invalid private trace coordinate')
        step, count = self._lines, len(doc['records'])
        if count:
            window = step // 120
            rank = (step_rank(self._role, self._opponent, self._side, step), step)
            ranked = self._ranked[window]
            ranked.append((rank, raw, count))
            ranked.sort(key=lambda value: value[0])
            del ranked[PER_WINDOW:]
        self._lines += 1
        self._decoded += len(raw)
        if self._decoded > historical.MAX_DECODED:
            raise ValueError('Private trace exceeds resource budget')
        return raw

    def read(self, size=-1):
        if self._state != 'scanning' or self._lines != 719 or self._eof or size != 1:
            raise RuntimeError('Only the final EOF check is available')
        self._guard()
        raw = self._stream.read(size)
        if type(raw) is not bytes:
            raise ValueError('Invalid trace EOF')
        self._eof = raw == b''
        return raw

    def scan(self, *, task_sha256, crop_sha256):
        if self._state != 'ready':
            raise RuntimeError('Trace scan is single use')
        self._state = 'scanning'
        try:
            result = historical.scan(self, task_sha256=task_sha256,
                                     crop_sha256=crop_sha256, guard=self._guard)
            if (self._lines != 719 or not self._eof or type(result) is not dict
                    or result.get('lines') != 719 or result.get('decoded_bytes') != self._decoded):
                raise ValueError('Incomplete private trace scan')
        except BaseException:
            self._state, self._ranked = 'failed', None
            raise
        self._state = 'complete'
        return {'lines': self._lines, 'decoded_bytes': self._decoded}

    def samples(self):
        if self._state != 'complete':
            raise RuntimeError('Complete authenticated scan required')
        try:
            windows = []
            for ranked in self._ranked:
                selected = []
                for (_digest, step), raw, count in ranked:
                    self._guard()
                    ordinal = min(range(count), key=lambda i: (
                        ordinal_rank(self._role, self._opponent, self._side, step, i), i))
                    doc = json.loads(raw, parse_constant=historical._invalid_constant)
                    record = doc['records'][ordinal]
                    if type(record) is not dict:
                        raise ValueError('Invalid selected callback')
                    selected.append(dict(role=self._role, opponent=self._opponent,
                        side=self._side, step=step, ordinal=ordinal,
                        inputs=record.get('inputs'), teacher=record.get('teacher_label'),
                        chosen=record.get('chosen')))
                windows.append(selected)
        except BaseException:
            self._state, self._ranked = 'failed', None
            raise
        self._state, self._ranked = 'consumed', None
        return windows
