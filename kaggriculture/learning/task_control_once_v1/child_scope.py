"""Attach a private diagnostic only around the authenticated live game call.

The outer entry must authenticate all inputs, enforce process supervision and
provide an exclusively owned bounded receipt writer. This adapter does not
authorize a game, replace an official worker or interpret a past exception.
"""
from contextlib import contextmanager
from functools import wraps


def _unknown():
    return dict(schema='task-budget-observer-v1', source='unknown', phase='unknown', target='unknown',
                status='unknown', method=None, reasons=[], job_relation='unknown',
                saw_target_forward=None, saw_target_controller=None, other_budget_seen=None)


def _receipt(value):
    fields = set(_unknown())
    allowed_reasons = {'task_forwards_total', 'task_forwards_per_turn', 'crop_forwards_total',
                       'crop_forwards_per_turn', 'neural_input_rows_total', 'neural_input_rows_per_turn',
                       'controller_calls_total', 'controller_calls_per_turn', 'teacher_proposals_total'}
    if (type(value) is not dict or set(value) != fields or value.get('schema') != 'task-budget-observer-v1'
            or value.get('source') not in ('unknown', 'caller_attested')
            or value.get('phase') not in ('unknown', 'caller_attested')
            or value.get('target') not in ('unknown', 'none', 'one', 'multiple')
            or value.get('status') not in ('unknown', 'observed', 'not_observed')
            or value.get('method') not in (None, 'forward', 'controller')
            or value.get('job_relation') not in ('unknown', 'same_exception', 'different_exception', 'job_returned')
            or type(value.get('reasons')) is not list
            or any(type(reason) is not str or reason not in allowed_reasons for reason in value['reasons'])
            or any(type(value.get(key)) not in (bool, type(None))
                   for key in ('saw_target_forward', 'saw_target_controller', 'other_budget_seen'))):
        return _unknown()
    if ((value['status'] == 'observed' and (value['method'] is None or not value['reasons']))
            or (value['status'] != 'observed' and (value['method'] is not None or value['reasons']
                                                  or value['job_relation'] != 'unknown'))
            or (value['status'] != 'unknown' and (value['source'] != 'caller_attested'
                                                   or value['phase'] != 'caller_attested'
                                                   or value['target'] != 'one'))
            or len(value['reasons']) != len(set(value['reasons']))
            or (value['method'] == 'forward' and any(reason.startswith(('controller_', 'teacher_'))
                                                      for reason in value['reasons']))
            or (value['method'] == 'controller' and any(not reason.startswith(('controller_', 'teacher_'))
                                                         for reason in value['reasons']))):
        return _unknown()
    return dict(value, reasons=list(value['reasons']))


def run(*, worker, observer, child_call, save_observation, identity_attested, phase_attested):
    """Run exactly the caller-supplied child operation; diagnostic IO is separate.

    The writer receives only a fresh allowlisted diagnostic dictionary after the
    old child has left its write guard. Writer/diagnostic Exception failures do
    not change the child's return or exception; process cancellation propagates.
    The caller must separately verify whether a receipt was actually saved.
    """
    if (identity_attested is not True or phase_attested is not True or not callable(child_call)
            or not callable(save_observation)):
        raise ValueError('Unattested diagnostic context')
    worker._single_python_thread()
    original = worker._load_snapshot
    if not callable(original):
        raise ValueError('Missing original worker loader')
    broken = [False]
    live_interrupt = [None]

    def boundary(error):
        try:
            observer.job_boundary(error)
        except Exception:
            broken[0] = True

    @wraps(original)
    def load(*args, **kwargs):
        module = original(*args, **kwargs)
        official, job = module.official_entry, module.job
        if not callable(official) or not callable(job):
            raise ValueError('Worker scope unavailable')

        @wraps(official)
        def live(*values, **options):
            try:
                with observer.live_scope(identity_attested=True, phase_attested=True):
                    return official(*values, **options)
            except KeyboardInterrupt as error:
                live_interrupt[0] = error
                raise

        @wraps(job)
        def observed_job(*values, **options):
            try:
                result = job(*values, **options)
            except BaseException as error:
                boundary(error)
                raise
            boundary(None)
            return result

        module.official_entry, module.job = live, observed_job
        return module

    def save():
        try:
            result = _unknown() if broken[0] else _receipt(observer.result())
            save_observation(result)
        except Exception:
            pass

    try:
        worker._load_snapshot = load
        try:
            outcome = child_call()
        finally:
            if worker._load_snapshot is load:
                worker._load_snapshot = original
            else:
                broken[0] = True
    except BaseException as error:
        save()
        if live_interrupt[0] is not None and isinstance(error, Exception):
            raise live_interrupt[0] from None
        raise
    else:
        save()
        if live_interrupt[0] is not None:
            raise live_interrupt[0] from None
        return outcome
    finally:
        try:
            observer.close()
        except Exception:
            pass
