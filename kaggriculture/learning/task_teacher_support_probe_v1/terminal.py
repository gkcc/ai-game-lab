"""One-turn access to the pinned terminal rule teacher's task candidates.

This is a bounded wrapper around the existing TaskAgent with both selectors in
teacher mode. It does not run a neural model, game engine, or a complete match.
The caller supplies one already-sanitized player observation and owns the
process, wall-clock, and physical-memory limits. Requests and ``last`` records
are private; a caller retaining references must discard those independently.
"""
from copy import deepcopy

from learning.task_choice_v1.agent import TaskAgent


MAX_CANDIDATES = 256
MAX_CALLBACKS_PER_TURN = 32


class TerminalProbeError(ValueError):
    """Invalid lifecycle or a registered task-candidate limit was exceeded."""


class TerminalTeacherProbe(TaskAgent):
    """A single-use, teacher-only controller on one sampled decision turn.

    Frozen source verification and actual action/record construction remain
    entirely in TaskAgent and CropAgent. After a successful call the inherited
    ``last`` holds the original private records; call ``clear`` and release
    caller-held references when finished. Failed calls discard owned records and
    still consume the instance. A fresh instance is required for another turn.
    """

    def __init__(self, *, guard):
        if not callable(guard):
            raise TerminalProbeError('A resource guard is required')
        self._guard, self._running, self._used = guard, False, False
        self._callbacks = 0
        guard()
        super().__init__(task_selector=None, crop_mode='teacher')
        guard()

    def __repr__(self):
        return '<PrivateTerminalTeacherProbe>'

    def _release(self):
        self.last, self.records = None, []
        self.crop.last, self.crop.records = None, []

    def clear(self):
        """Release probe-owned task/crop records while no call is in progress."""
        if self._running:
            raise TerminalProbeError('Cannot clear an active terminal probe')
        self._release()

    def select(self, context):
        """Check bounds before the original task feature encoder is called."""
        if not self._running:
            raise TerminalProbeError('Task callback outside a terminal probe call')
        self._guard()
        if (type(context) is not dict or type(context.get('choices')) is not list
                or not 0 < len(context['choices']) <= MAX_CANDIDATES):
            raise TerminalProbeError('Terminal task candidate limit exceeded')
        if self._callbacks >= MAX_CALLBACKS_PER_TURN:
            raise TerminalProbeError('Terminal task callback limit exceeded')
        self._callbacks += 1
        previous_records = len(self.records)
        result = super().select(context)
        # Capture the task's own pre-execution request, before the frozen
        # controller pops the task or resolves a PLANT crop. The concrete crop
        # is reported separately in the original CropAgent record.
        if (type(result) is not tuple or len(result) < 2 or type(result[-1]) is not int
                or type(context.get('tasks')) is not list or not 0 <= result[-1] < len(context['tasks'])
                or type(context['tasks'][result[-1]]) not in (list, tuple)
                or len(context['tasks'][result[-1]]) < 2 or len(self.records) != previous_records + 1
                or type(self.records[-1]) is not dict):
            raise TerminalProbeError('Terminal task record differs')
        self.records[-1]['intended_action'] = deepcopy(context['tasks'][result[-1]][1])
        self._guard()
        return result

    def __call__(self, observation, configuration):
        if self._running:
            raise TerminalProbeError('Terminal probe is already active')
        if self._used:
            self._release()
            raise TerminalProbeError('A terminal probe permits only one turn')
        self._used, self._running = True, True
        try:
            self._guard()
            request = super().__call__(observation, configuration)
            self._guard()
            return request
        except BaseException:
            self._release()
            raise
        finally:
            self._running = False
