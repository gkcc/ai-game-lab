import unittest

from . import memory_gate as gate


class FakeTime:
    def __init__(self, values=(), start=0.0):
        self.now = start
        self.values = iter(values)
        self.sample_times = []
        self.sleeps = []

    def clock(self):
        return self.now

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += seconds

    def available(self):
        self.sample_times.append(self.now)
        value = next(self.values)
        if isinstance(value, Exception):
            raise value
        return value


class MemoryGateTests(unittest.TestCase):
    def check(self, fake, budget=None, deadline=10000.0, job_not_started=True):
        budget = budget if budget is not None else gate.WaitBudget()
        result = gate.wait_for_preflight(budget=budget, deadline=deadline,
            available=fake.available, clock=fake.clock, sleep=fake.sleep,
            job_not_started=job_not_started)
        return result, budget

    def test_initial_success_uses_three_spaced_samples_and_no_wait_budget(self):
        fake = FakeTime([gate.MIN_AVAILABLE_BYTES] * 3)
        result, budget = self.check(fake, gate.WaitBudget(blocked_seconds=7.0))
        self.assertTrue(result.passed)
        self.assertEqual(result.reason, 'ready')
        self.assertEqual(fake.sample_times, [0.0, 10.0, 20.0])
        self.assertEqual(result.samples, 3)
        self.assertEqual(result.blocked_seconds, 0.0)
        self.assertEqual(budget.blocked_seconds, 7.0)
        self.assertFalse(budget.terminal)

    def test_recovery_restarts_full_preflight_after_thirty_seconds(self):
        fake = FakeTime([gate.MIN_AVAILABLE_BYTES, 0] + [gate.MIN_AVAILABLE_BYTES] * 3)
        result, budget = self.check(fake)
        self.assertTrue(result.passed)
        self.assertEqual(fake.sample_times, [0.0, 10.0, 30.0, 40.0, 50.0])
        self.assertEqual((result.attempts, result.samples, result.shortages), (2, 5, 1))
        self.assertEqual(result.last_shortage_sample, 2)
        self.assertEqual(result.blocked_seconds, 40.0)
        self.assertEqual(budget.blocked_seconds, 40.0)

    def test_blocked_time_accumulates_across_games(self):
        budget = gate.WaitBudget()
        first = FakeTime([0] + [gate.MIN_AVAILABLE_BYTES] * 3)
        one, _ = self.check(first, budget)
        second = FakeTime([0] + [gate.MIN_AVAILABLE_BYTES] * 3, start=100.0)
        two, _ = self.check(second, budget)
        self.assertTrue(one.passed and two.passed)
        self.assertEqual((one.blocked_seconds, two.blocked_seconds), (50.0, 50.0))
        self.assertEqual(budget.blocked_seconds, 100.0)

    def test_continuous_shortage_ends_at_single_wait_limit(self):
        fake = FakeTime([0] * 100)
        result, budget = self.check(fake)
        self.assertEqual(result.reason, 'single_wait_limit')
        self.assertFalse(result.passed)
        self.assertEqual(fake.now, gate.SINGLE_WAIT_SECONDS)
        self.assertEqual(result.blocked_seconds, gate.SINGLE_WAIT_SECONDS)
        self.assertTrue(budget.terminal)
        calls = len(fake.sample_times)
        again, _ = self.check(fake, budget)
        self.assertEqual(again.reason, 'budget_terminal')
        self.assertEqual(len(fake.sample_times), calls)

    def test_total_limit_stops_before_single_limit(self):
        fake = FakeTime([0] * 5)
        budget = gate.WaitBudget(blocked_seconds=gate.TOTAL_WAIT_SECONDS - 10.0)
        result, _ = self.check(fake, budget)
        self.assertEqual(result.reason, 'total_wait_limit')
        self.assertEqual(fake.now, 10.0)
        self.assertEqual(budget.blocked_seconds, gate.TOTAL_WAIT_SECONDS)
        self.assertTrue(budget.terminal)

    def test_external_deadline_can_stop_successful_sampling(self):
        fake = FakeTime([gate.MIN_AVAILABLE_BYTES] * 3)
        result, budget = self.check(fake, deadline=15.0)
        self.assertEqual(result.reason, 'deadline')
        self.assertEqual(fake.sample_times, [0.0, 10.0])
        self.assertEqual(result.blocked_seconds, 0.0)
        self.assertTrue(budget.terminal)

    def test_invalid_samples_are_not_treated_as_shortage(self):
        for value in (True, False, 1.0, None, -1):
            with self.subTest(value=value):
                fake = FakeTime([value])
                result, budget = self.check(fake)
                self.assertEqual(result.reason, 'invalid_sample')
                self.assertEqual((result.samples, result.shortages), (1, 0))
                self.assertEqual(fake.sleeps, [])
                self.assertTrue(budget.terminal)

    def test_sampling_exception_is_classified_without_exposing_text(self):
        fake = FakeTime([RuntimeError('sensitive callback detail')])
        result, budget = self.check(fake)
        self.assertEqual(result.reason, 'sample_error')
        self.assertNotIn('sensitive', repr(result))
        self.assertTrue(budget.terminal)

    def test_sampling_time_after_a_shortage_counts_toward_wait(self):
        fake = FakeTime([0, gate.MIN_AVAILABLE_BYTES, gate.MIN_AVAILABLE_BYTES, gate.MIN_AVAILABLE_BYTES])
        def slow_sample():
            value = fake.available()
            fake.now += 3.0
            return value
        budget = gate.WaitBudget()
        result = gate.wait_for_preflight(budget=budget, deadline=10000.0,
            available=slow_sample, clock=fake.clock, sleep=fake.sleep, job_not_started=True)
        self.assertTrue(result.passed)
        self.assertEqual(fake.sample_times, [0.0, 30.0, 43.0, 56.0])
        self.assertEqual(result.blocked_seconds, 59.0)

    def test_clock_errors_and_stalled_sleep_stop(self):
        good = gate.MIN_AVAILABLE_BYTES
        for faulty_clock in (lambda: float('nan'), lambda: True):
            with self.subTest(clock=faulty_clock):
                budget = gate.WaitBudget()
                result = gate.wait_for_preflight(budget=budget, deadline=10000.0,
                    available=lambda: good, clock=faulty_clock, sleep=lambda _: None, job_not_started=True)
                self.assertEqual(result.reason, 'clock_error')
                self.assertTrue(budget.terminal)
        times = iter([1.0, 0.0])
        budget = gate.WaitBudget()
        result = gate.wait_for_preflight(budget=budget, deadline=10000.0,
            available=lambda: good, clock=lambda: next(times), sleep=lambda _: None, job_not_started=True)
        self.assertEqual(result.reason, 'clock_error')
        fake = FakeTime([0])
        budget = gate.WaitBudget()
        result = gate.wait_for_preflight(budget=budget, deadline=10000.0,
            available=fake.available, clock=fake.clock, sleep=lambda _: None, job_not_started=True)
        self.assertEqual(result.reason, 'clock_error')
        self.assertIsNone(result.blocked_seconds)
        self.assertIsNone(result.total_blocked_seconds)
        self.assertFalse(budget.duration_known)
        repeated, _ = self.check(fake, budget)
        self.assertIsNone(repeated.total_blocked_seconds)

    def test_sleep_error_stops_without_sampling_again(self):
        fake = FakeTime([0, gate.MIN_AVAILABLE_BYTES])
        def failed_sleep(_):
            fake.now += 3.0
            raise RuntimeError('sleep failed')
        result = gate.wait_for_preflight(budget=gate.WaitBudget(), deadline=10000.0,
            available=fake.available, clock=fake.clock, sleep=failed_sleep, job_not_started=True)
        self.assertEqual(result.reason, 'sleep_error')
        self.assertEqual(fake.sample_times, [0.0])
        self.assertEqual(result.blocked_seconds, 3.0)

    def test_clock_precision_loss_cannot_skip_sample_or_retry_intervals(self):
        for sample in (gate.MIN_AVAILABLE_BYTES, 0):
            with self.subTest(sample=sample):
                calls = []
                budget = gate.WaitBudget()
                result = gate.wait_for_preflight(budget=budget, deadline=2e20,
                    available=lambda: calls.append(1) or sample, clock=lambda: 1e20,
                    sleep=lambda _: self.fail('Must stop before sleep'), job_not_started=True)
                self.assertFalse(result.passed)
                self.assertEqual(result.reason, 'clock_error')
                self.assertEqual(calls, [1])
                self.assertTrue(budget.terminal)

    def test_clock_failure_during_first_sample_has_unknown_duration(self):
        times = iter([0.0, 0.0, float('nan')])
        budget = gate.WaitBudget()
        result = gate.wait_for_preflight(budget=budget, deadline=10000.0,
            available=lambda: 0, clock=lambda: next(times), sleep=lambda _: None,
            job_not_started=True)
        self.assertEqual(result.reason, 'clock_error')
        self.assertIsNone(result.blocked_seconds)
        self.assertIsNone(result.total_blocked_seconds)
        self.assertTrue(budget.terminal)
        self.assertFalse(budget.duration_known)

    def test_interrupt_after_shortage_makes_budget_terminal_without_swallowing(self):
        for interrupt in (KeyboardInterrupt, SystemExit):
            with self.subTest(interrupt=interrupt):
                fake = FakeTime([0, gate.MIN_AVAILABLE_BYTES])
                budget = gate.WaitBudget()
                def interrupted_sleep(_):
                    fake.now += 25.0
                    raise interrupt()
                with self.assertRaises(interrupt):
                    gate.wait_for_preflight(budget=budget, deadline=10000.0,
                        available=fake.available, clock=fake.clock, sleep=interrupted_sleep,
                        job_not_started=True)
                self.assertTrue(budget.terminal)
                self.assertFalse(budget.duration_known)
                result, _ = self.check(fake, budget)
                self.assertEqual(result.reason, 'budget_terminal')
                self.assertIsNone(result.total_blocked_seconds)
                self.assertEqual(fake.sample_times, [0.0])

    def test_rounded_clock_never_shortens_sampling_or_retry_intervals(self):
        start = float(2**55)
        for values in ([gate.MIN_AVAILABLE_BYTES] * 3,
                       [0] + [gate.MIN_AVAILABLE_BYTES] * 3):
            with self.subTest(shortage=values[0] == 0):
                fake = FakeTime(values, start=start)
                result, _ = self.check(fake, deadline=start + 10000.0)
                self.assertTrue(result.passed)
                gaps = [right - left for left, right in zip(fake.sample_times, fake.sample_times[1:])]
                if values[0] == 0:
                    self.assertGreaterEqual(gaps[0], gate.RETRY_INTERVAL_SECONDS)
                    gaps = gaps[1:]
                self.assertTrue(all(gap >= gate.SAMPLE_INTERVAL_SECONDS for gap in gaps))

    def test_active_or_unknown_job_fails_before_any_callback(self):
        for state in (False, None, 1):
            with self.subTest(state=state):
                fake = FakeTime([gate.MIN_AVAILABLE_BYTES] * 3)
                result, budget = self.check(fake, job_not_started=state)
                self.assertEqual(result.reason, 'job_active_or_unknown')
                self.assertEqual(fake.sample_times, [])
                self.assertTrue(budget.terminal)


if __name__ == '__main__':
    unittest.main()
