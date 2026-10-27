"""Reuse the established label collector under this experiment's freeze."""
from contextlib import contextmanager
import json
from pathlib import Path
from threading import RLock

from learning.task_completion_bc_v1 import collect as core
from . import experiment

_LOCK = RLock()


class EntryReceiptFailure(RuntimeError):
    def __init__(self, primary_error_kind, receipt_error_kind):
        super().__init__('Could not persist extraction failure receipt')
        self.primary_error_kind = primary_error_kind
        self.receipt_error_kind = receipt_error_kind


@contextmanager
def bindings():
    names = ('ready', 'write_new', 'deadline_check')
    old = {name: getattr(core, name) for name in names}
    try:
        for name in names:
            setattr(core, name, getattr(experiment, name))
        yield
    finally:
        for name, value in old.items():
            setattr(core, name, value)


def test(out):
    from .campaign import verify_model_seal
    from decision import full_audit
    out = Path(out).resolve()
    with _LOCK:
        verify_model_seal(out)
        state = dict(marker_created=False, marker_complete=False, replay_started=0,
                     replay_completed=0, confirmed_transitions=0, replay_in_progress=False)
        original = full_audit.reexecute
        def tracked(*args, **kwargs):
            state['replay_started'] += 1
            state['replay_in_progress'] = True
            result = original(*args, **kwargs)
            state['replay_completed'] += 1
            state['replay_in_progress'] = False
            state['confirmed_transitions'] += result['official_transitions_reexecuted']
            return result
        def write(path, value):
            path = Path(path).resolve()
            if path != out/'test-data-started.json':
                return experiment.write_new(path, value)
            if not path.is_relative_to(experiment.OUT): raise ValueError('Marker outside experiment')
            with path.open('x', encoding='utf-8', newline='\n') as stream:
                state['marker_created'] = True
                json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
                stream.write('\n')
            state['marker_complete'] = True
        try:
            with bindings():
                core.write_new = write
                full_audit.reexecute = tracked
                result = core.test(out)
            verify_model_seal(out)
            return result
        except BaseException as exc:
            if state['marker_created']:
                try:
                    experiment.write_new(out/'test-data-entry-failure.local.json', dict(
                        error_kind=type(exc).__name__, **state,
                        current_replay_partial_transitions='unknown' if state['replay_in_progress'] else 0,
                        retry_allowed=False, partial_work_not_refunded=True))
                except BaseException as receipt_exc:
                    raise EntryReceiptFailure(type(exc).__name__, type(receipt_exc).__name__) from None
            raise
        finally:
            full_audit.reexecute = original
