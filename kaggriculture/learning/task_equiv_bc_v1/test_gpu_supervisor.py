"""Signal-handler contract tests; no device, model, data or filesystem writes."""
import signal
import unittest

from learning.task_equiv_bc_v1.gpu_supervisor import Cancellation


class CancellationTests(unittest.TestCase):
    def test_external_cancellation_and_signal_restore(self):
        before = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGINT)}
        state = {'cancel': False}
        with Cancellation(lambda: state['cancel']) as scope:
            self.assertFalse(scope.requested())
            state['cancel'] = True
            self.assertTrue(scope.requested())
            state['cancel'] = False
            signal.raise_signal(signal.SIGTERM)
            self.assertTrue(scope.requested())
        self.assertEqual(before, {sig: signal.getsignal(sig) for sig in before})


if __name__ == '__main__':
    unittest.main()
