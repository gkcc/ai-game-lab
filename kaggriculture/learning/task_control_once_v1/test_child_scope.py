"""Synthetic checks for scoped diagnostics without changing child behavior."""
from contextlib import contextmanager
from types import SimpleNamespace
import unittest

from . import child_scope


class Observer:
    def __init__(self):
        self.active = False
        self.boundary = 'unset'
        self.closed = False

    @contextmanager
    def live_scope(self, **_kwargs):
        self.active = True
        try:
            yield self
        finally:
            self.active = False

    def job_boundary(self, error):
        self.boundary = error

    def result(self):
        return child_scope._unknown()

    def close(self):
        self.closed = True


class ChildScopeTests(unittest.TestCase):
    def test_live_scope_ends_before_postgame_work_and_restores_loader(self):
        observer, saved = Observer(), []
        def official():
            self.assertTrue(observer.active)
            return 'game-result'
        module = SimpleNamespace(official_entry=official)
        def job():
            value = module.official_entry()
            self.assertFalse(observer.active)
            return value
        module.job = job
        original = lambda: module
        worker = SimpleNamespace(_load_snapshot=original, _single_python_thread=lambda: None)
        result = child_scope.run(worker=worker, observer=observer,
            child_call=lambda: worker._load_snapshot().job(), save_observation=saved.append,
            identity_attested=True, phase_attested=True)
        self.assertEqual(result, 'game-result')
        self.assertIsNone(observer.boundary)
        self.assertTrue(observer.closed)
        self.assertIs(worker._load_snapshot, original)
        self.assertEqual(saved, [child_scope._unknown()])

    def test_sink_failure_does_not_replace_original_child_exception(self):
        observer = Observer()
        original_error = RuntimeError('synthetic private marker')
        def official():
            raise original_error
        module = SimpleNamespace(official_entry=official)
        module.job = lambda: module.official_entry()
        original = lambda: module
        worker = SimpleNamespace(_load_snapshot=original, _single_python_thread=lambda: None)
        def bad_sink(_value):
            raise OSError('synthetic write failure')
        with self.assertRaises(RuntimeError) as caught:
            child_scope.run(worker=worker, observer=observer, child_call=lambda: worker._load_snapshot().job(),
                            save_observation=bad_sink, identity_attested=True, phase_attested=True)
        self.assertIs(caught.exception, original_error)
        self.assertIs(observer.boundary, original_error)
        self.assertIs(worker._load_snapshot, original)
        self.assertTrue(observer.closed)

    def test_bad_diagnostic_is_reduced_to_unknown(self):
        self.assertEqual(child_scope._receipt({'private': 'must-not-escape'}), child_scope._unknown())
        bad = child_scope._unknown()
        bad.update(status='observed', method='forward', reasons=['neural_input_rows_total'])
        self.assertEqual(child_scope._receipt(bad), child_scope._unknown())

    def test_live_interrupt_survives_legacy_child_exception_wrapping(self):
        observer = Observer()
        interrupt = KeyboardInterrupt('synthetic cancellation')
        def official():
            raise interrupt
        module = SimpleNamespace(official_entry=official)
        module.job = lambda: module.official_entry()
        original = lambda: module
        worker = SimpleNamespace(_load_snapshot=original, _single_python_thread=lambda: None)
        def legacy_child():
            try:
                worker._load_snapshot().job()
            except BaseException:
                raise RuntimeError('legacy wrapper') from None
        with self.assertRaises(KeyboardInterrupt) as stopped:
            child_scope.run(worker=worker, observer=observer, child_call=legacy_child,
                            save_observation=lambda _value: None,
                            identity_attested=True, phase_attested=True)
        self.assertIs(stopped.exception, interrupt)
        self.assertIs(worker._load_snapshot, original)


if __name__ == '__main__':
    unittest.main()
