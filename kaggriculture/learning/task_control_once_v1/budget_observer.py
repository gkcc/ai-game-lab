"""In-memory observation of direct budget rejections, not an identity authority.

The future caller must authenticate the classes, exact original method objects
and their code objects, target game and live phase before asserting either trust flag. These booleans
are caller attestations, not proof of provenance or process supervision. This
module does not perform filesystem IO, run a game or write any receipt.

Ordinary diagnostic exceptions do not replace the budget result or exception.
Cancellation (including KeyboardInterrupt and SystemExit) takes priority and
propagates, even when it arrives while classifying an existing budget error.
Restoration after cancellation is best effort if interruptions repeat.

Function-code and direct-raise checks detect ordinary code drift at the
observation boundary. They are not protection against malicious, transient
same-process modification or forged interpreter-level execution state.

Use only in a dedicated, single-threaded child. The scoped class patches are
process-global, and the lock does not guard code outside this module. A briefly
held exception reference may retain its traceback in private memory until the
caller reports the job boundary or closes the observer; it is never returned,
rendered or written by this module.
"""
from contextlib import contextmanager
import dis
from functools import wraps
import threading
from types import FunctionType, TracebackType


_LOCK = threading.Lock()
_FORWARD_ERROR = ('Shared neural computation budget exhausted',)
_CONTROLLER_ERROR = ('Controller/proposal computation budget exhausted',)
_LIMIT_KEYS = frozenset(('task_forwards', 'crop_forwards', 'neural_input_rows',
                         'gate_evaluations', 'controller_calls', 'teacher_proposals',
                         'max_task_forwards_per_turn', 'max_crop_forwards_per_turn',
                         'max_neural_input_rows_per_turn', 'max_controller_calls_per_turn'))


def _valid_limits(limits):
    return (type(limits) is dict and limits.keys() == _LIMIT_KEYS
            and all(type(value) is int and value > 0 for value in limits.values()))


def _matching_limits(actual, expected):
    return expected is not None and _valid_limits(actual) and actual == expected


def _direct_raise(error, wrapper_code, expected_code):
    tb = error.__traceback__
    if (type(tb) is not TracebackType or tb.tb_frame.f_code is not wrapper_code
            or type(tb.tb_next) is not TracebackType):
        return False
    origin = tb.tb_next
    if origin.tb_frame.f_code is not expected_code or origin.tb_next is not None:
        return False
    return any(instruction.offset == origin.tb_lasti and instruction.opname == 'RAISE_VARARGS'
               and instruction.arg == 1 for instruction in dis.get_instructions(expected_code))


def _counter(mapping, key):
    if type(mapping) is not dict or type(mapping.get(key)) is not int or mapping[key] < 0:
        raise ValueError('Invalid synthetic budget counters')
    return mapping[key]


def _forward_reasons(budget, kind, rows):
    if type(kind) is not str or kind not in ('task', 'crop') or type(rows) is not int or rows < 1:
        raise ValueError('Invalid budget call')
    counts, turn, limits = budget.counts, budget.turn, budget.limits
    reasons = []
    if _counter(counts, kind + '_forwards') >= _counter(limits, kind + '_forwards'):
        reasons.append(kind + '_forwards_total')
    if _counter(turn, kind + '_forwards') >= _counter(limits, 'max_' + kind + '_forwards_per_turn'):
        reasons.append(kind + '_forwards_per_turn')
    if (_counter(counts, 'task_rows') + _counter(counts, 'crop_rows') + rows
            > _counter(limits, 'neural_input_rows')):
        reasons.append('neural_input_rows_total')
    if (_counter(turn, 'task_rows') + _counter(turn, 'crop_rows') + rows
            > _counter(limits, 'max_neural_input_rows_per_turn')):
        reasons.append('neural_input_rows_per_turn')
    return reasons


def _controller_reasons(budget, proposal):
    if type(proposal) is not bool:
        raise ValueError('Invalid controller call')
    counts, turn, limits = budget.counts, budget.turn, budget.limits
    reasons = []
    if _counter(counts, 'controller_calls') >= _counter(limits, 'controller_calls'):
        reasons.append('controller_calls_total')
    if _counter(turn, 'controller_calls') >= _counter(limits, 'max_controller_calls_per_turn'):
        reasons.append('controller_calls_per_turn')
    if proposal and _counter(counts, 'teacher_proposals') >= _counter(limits, 'teacher_proposals'):
        reasons.append('teacher_proposals_total')
    return reasons


def _wrappers(observer, old_init, old_forward, old_controller):
    @wraps(old_init)
    def init(agent, *args, **kwargs):
        value = old_init(agent, *args, **kwargs)
        try:
            observer._register(agent)
        except Exception:
            observer._broken = True
        except BaseException:
            observer._broken = True
            raise
        return value

    @wraps(old_forward)
    def forward(budget, kind, rows):
        try:
            observer._called(budget, 'forward')
        except Exception:
            observer._broken = True
        except BaseException:
            observer._broken = True
            raise
        try:
            return old_forward(budget, kind, rows)
        except BaseException as error:
            try:
                observer._classify(budget, 'forward', error, wrapper_code=forward.__code__,
                                   kind=kind, rows=rows)
            except Exception:
                observer._broken = True
            except BaseException:
                observer._broken = True
                raise
            raise

    @wraps(old_controller)
    def controller(budget, proposal=False):
        try:
            observer._called(budget, 'controller')
        except Exception:
            observer._broken = True
        except BaseException:
            observer._broken = True
            raise
        try:
            return old_controller(budget, proposal)
        except BaseException as error:
            try:
                observer._classify(budget, 'controller', error, wrapper_code=controller.__code__,
                                   proposal=proposal)
            except Exception:
                observer._broken = True
            except BaseException:
                observer._broken = True
                raise
            raise

    return init, forward, controller


class BudgetObserver:
    """Observe one caller-attested live scope around exact supplied methods.

    Construct with the real budget and agent classes and three method objects
    and their current code objects independently authenticated by the caller, together with the exact
    authenticated positive-integer limits. Code objects are snapshotted at
    construction. The limits are checked when binding
    the target, when classifying an exception, and at the job boundary; an
    intermediate change that has been reverted is not detectable. Use ``live_scope`` only around
    the live official game; call ``job_boundary`` when the old Worker job has
    actually returned or raised. Results are unknown when not attested, when the
    target is missing/ambiguous, when source identities drift, or when observation
    fails. Calling ``job_boundary`` does not prove a process exit or cleanup.
    """
    def __init__(self, budget_type, agent_type, *, expected_forward, expected_controller,
                 expected_agent_init, expected_limits):
        self.budget_type = budget_type
        self.agent_type = agent_type
        self.expected_forward = expected_forward
        self.expected_controller = expected_controller
        self.expected_agent_init = expected_agent_init
        self._forward_code = expected_forward.__code__ if type(expected_forward) is FunctionType else None
        self._controller_code = expected_controller.__code__ if type(expected_controller) is FunctionType else None
        self._agent_code = expected_agent_init.__code__ if type(expected_agent_init) is FunctionType else None
        self._expected_limits = dict(expected_limits) if _valid_limits(expected_limits) else None
        self._identity_attested = False
        self._phase_attested = False
        self._closed = False
        self._active = False
        self._broken = False
        self._target_count = 0
        self._target_budget = None
        self._other_budget_seen = False
        self._forward_seen = False
        self._controller_seen = False
        self._unclassified = False
        self._events = 0
        self._event_method = None
        self._reasons = ()
        self._exception = None
        self._job_relation = 'unknown'
        self._job_boundary_seen = False

    def _authenticatable(self):
        return (type(self.budget_type) is type and type(self.agent_type) is type
                and type(self.expected_forward) is FunctionType
                and type(self.expected_controller) is FunctionType
                and type(self.expected_agent_init) is FunctionType
                and self.budget_type.forward is self.expected_forward
                and self.budget_type.controller is self.expected_controller
                and self.agent_type.__init__ is self.expected_agent_init
                and self.expected_forward.__code__ is self._forward_code
                and self.expected_controller.__code__ is self._controller_code
                and self.expected_agent_init.__code__ is self._agent_code
                and self.expected_agent_init.__globals__.get('_Budget') is self.budget_type
                and self._expected_limits is not None)

    def _register(self, agent):
        if (type(agent) is not self.agent_type
                or type(getattr(agent, 'budget', None)) is not self.budget_type
                or self.expected_agent_init.__code__ is not self._agent_code
                or not _matching_limits(agent.budget.limits, self._expected_limits)):
            self._broken = True
            return
        self._target_count += 1
        if self._target_count == 1:
            self._target_budget = agent.budget
        else:
            self._broken = True

    def _called(self, budget, method):
        if budget is not self._target_budget or self._target_count != 1:
            self._other_budget_seen = True
            return
        if method == 'forward':
            self._forward_seen = True
        else:
            self._controller_seen = True

    def _classify(self, budget, method, error, *, wrapper_code, kind=None, rows=None, proposal=None):
        """Called only immediately after the supplied original method raised."""
        if budget is not self._target_budget or self._target_count != 1:
            return
        if not _matching_limits(budget.limits, self._expected_limits):
            self._broken = True
            return
        expected_method, expected_code = ((self.expected_forward, self._forward_code) if method == 'forward'
                                          else (self.expected_controller, self._controller_code))
        if expected_method.__code__ is not expected_code or not _direct_raise(error, wrapper_code, expected_code):
            self._unclassified = True
            return
        if type(error) is not RuntimeError:
            self._unclassified = True
            return
        if method == 'forward' and error.args == _FORWARD_ERROR:
            reasons = _forward_reasons(budget, kind, rows)
        elif method == 'controller' and error.args == _CONTROLLER_ERROR:
            reasons = _controller_reasons(budget, proposal)
        else:
            self._unclassified = True
            return
        if not reasons or self._events:
            self._unclassified = True
            return
        self._events += 1
        self._event_method = method
        self._reasons = tuple(reasons)
        self._exception = error

    def _restore(self, installed):
        cancellation = None
        for cls, name, wrapped, previous in reversed(installed):
            try:
                current = getattr(cls, name)
                if current is wrapped:
                    setattr(cls, name, previous)
                elif current is not previous:
                    self._broken = True
            except BaseException as error:
                self._broken = True
                if not isinstance(error, Exception) and cancellation is None:
                    cancellation = error
                # A single interruption can occur before or after assignment.
                try:
                    if getattr(cls, name) is wrapped:
                        setattr(cls, name, previous)
                except BaseException as retry_error:
                    self._broken = True
                    if not isinstance(retry_error, Exception) and cancellation is None:
                        cancellation = retry_error
        return cancellation

    @contextmanager
    def live_scope(self, *, identity_attested=False, phase_attested=False):
        """Install reversible wrappers; unverified/failed setup remains unknown."""
        if self._active or self._closed:
            self._broken = True
            yield self
            return
        self._identity_attested = identity_attested is True
        self._phase_attested = phase_attested is True
        if not (self._identity_attested and self._phase_attested):
            self._closed = True
            yield self
            return
        try:
            valid = self._authenticatable()
        except Exception:
            valid = False
        except BaseException:
            self._broken = True
            self._closed = True
            raise
        if not valid or not _LOCK.acquire(blocking=False):
            self._broken = True
            self._closed = True
            yield self
            return
        installed = []
        try:
            old_forward, old_controller, old_init = (self.expected_forward, self.expected_controller,
                                                      self.expected_agent_init)
            try:
                init, forward, controller = _wrappers(self, old_init, old_forward, old_controller)
            except Exception:
                self._broken = True
                yield self
                return
            # Prepare every restoration before the first class assignment, so a
            # cancellation immediately after assignment cannot strand a hook.
            installed = [(self.budget_type, 'forward', forward, old_forward),
                         (self.budget_type, 'controller', controller, old_controller),
                         (self.agent_type, '__init__', init, old_init)]
            try:
                for cls, name, wrapped, _previous in installed:
                    setattr(cls, name, wrapped)
            except Exception:
                self._broken = True
                cancellation = self._restore(installed)
                installed = []
                if cancellation is not None:
                    raise cancellation
                yield self
                return
            self._active = True
            yield self
        except BaseException:
            if not self._active:
                self._broken = True
            raise
        finally:
            try:
                cancellation = self._restore(installed)
            finally:
                self._active = False
                self._closed = True
                _LOCK.release()
            if cancellation is not None:
                raise cancellation

    def job_boundary(self, error=None):
        """Record only whether this exact exception reached the caller's job boundary."""
        if self._job_boundary_seen:
            self._broken = True
            self._job_relation = 'unknown'
            self._exception = None
            return
        self._job_boundary_seen = True
        if not self._closed or self._active:
            self._broken = True
        if self._target_budget is not None:
            try:
                if not _matching_limits(self._target_budget.limits, self._expected_limits):
                    self._broken = True
            except Exception:
                self._broken = True
        if self._exception is not None:
            self._job_relation = ('same_exception' if error is self._exception else
                                  'job_returned' if error is None else 'different_exception')
        self._exception = None

    def close(self):
        """Drop any private exception reference even if no job boundary was observed."""
        self._exception = None

    def result(self):
        """Construct a new allowlisted summary; no exception or traceback is returned."""
        trusted = (self._closed and not self._active and not self._broken
                   and self._identity_attested and self._phase_attested
                   and self._target_count == 1 and not self._unclassified)
        target = ('one' if self._target_count == 1 else 'multiple' if self._target_count > 1 else 'none')
        status = ('observed' if self._events == 1 else 'not_observed') if trusted else 'unknown'
        target_known = (self._closed and self._identity_attested and self._phase_attested and not self._broken)
        return dict(schema='task-budget-observer-v1',
                    source='caller_attested' if self._identity_attested and not self._broken else 'unknown',
                    phase='caller_attested' if self._phase_attested and not self._broken else 'unknown',
                    target=target if target_known else 'unknown',
                    status=status, method=self._event_method if status == 'observed' else None,
                    reasons=list(self._reasons) if status == 'observed' else [],
                    job_relation=self._job_relation if status == 'observed' else 'unknown',
                    saw_target_forward=self._forward_seen if trusted else None,
                    saw_target_controller=self._controller_seen if trusted else None,
                    other_budget_seen=self._other_budget_seen if trusted else None)
