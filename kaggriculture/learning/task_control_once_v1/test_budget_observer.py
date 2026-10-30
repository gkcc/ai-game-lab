"""Synthetic in-memory checks of the observer, with the existing budget API."""
import builtins
import json
import unittest
from types import SimpleNamespace
from unittest import mock

from learning.task_dagger_once_v1 import policy as agent_policy
from learning.task_request_gate_v1.policy import _Budget
from . import budget_observer
from .budget_observer import BudgetObserver


LIMITS = dict(task_forwards=10, crop_forwards=10, neural_input_rows=100,
              gate_evaluations=1, controller_calls=10, teacher_proposals=10,
              max_task_forwards_per_turn=10, max_crop_forwards_per_turn=10,
              max_neural_input_rows_per_turn=100, max_controller_calls_per_turn=10)


class SyntheticAgent:
    def __init__(self, limits=None):
        self.budget = _Budget(dict(LIMITS, **(limits or {})))


def observer(limits=None):
    return BudgetObserver(_Budget, SyntheticAgent, expected_forward=_Budget.forward,
                          expected_controller=_Budget.controller,
                          expected_agent_init=SyntheticAgent.__init__,
                          expected_limits=dict(LIMITS, **(limits or {})))


class BudgetObserverTests(unittest.TestCase):
    def setUp(self):
        self.originals = _Budget.forward, _Budget.controller, SyntheticAgent.__init__

    def tearDown(self):
        self.assertIs(_Budget.forward, self.originals[0])
        self.assertIs(_Budget.controller, self.originals[1])
        self.assertIs(SyntheticAgent.__init__, self.originals[2])

    def test_success_values_counters_and_method_restoration_match_original(self):
        direct = SyntheticAgent().budget
        expected = (direct.controller(), direct.forward('task', 3), direct.forward('crop', 1))
        seen = observer()
        with seen.live_scope(identity_attested=True, phase_attested=True):
            tracked = SyntheticAgent().budget
            actual = (tracked.controller(), tracked.forward('task', 3), tracked.forward('crop', 1))
        seen.job_boundary()
        self.assertEqual(actual, expected)
        self.assertEqual(tracked.counts, direct.counts)
        self.assertEqual(tracked.turn, direct.turn)
        result = seen.result()
        self.assertEqual((result['source'], result['phase'], result['target'], result['status']),
                         ('caller_attested', 'caller_attested', 'one', 'not_observed'))
        self.assertTrue(result['saw_target_forward'])
        self.assertTrue(result['saw_target_controller'])
        seen.close()

    def test_real_bounded_agent_registers_and_wraps_both_heads_without_model_io(self):
        self.assertIs(agent_policy._Budget, _Budget)
        original_init = agent_policy.BoundedAgent.__init__
        seen = BudgetObserver(_Budget, agent_policy.BoundedAgent,
                              expected_forward=_Budget.forward,
                              expected_controller=_Budget.controller,
                              expected_agent_init=original_init, expected_limits=LIMITS)
        crop_network = SimpleNamespace(logits=lambda x, names: ('crop-output',))
        controller = SimpleNamespace(crop=SimpleNamespace(network=crop_network))
        with mock.patch.object(agent_policy, 'TaskNetwork', return_value=lambda inputs: ('task-output',)), \
             mock.patch.object(agent_policy, 'TaskAgent', return_value=controller) as task_agent:
            with seen.live_scope(identity_attested=True, phase_attested=True):
                agent = agent_policy.BoundedAgent('synthetic-crop', 'synthetic-crop-id',
                                                  'synthetic-task', 'synthetic-task-id',
                                                  limits=LIMITS)
                selector = task_agent.call_args.args[0]
                self.assertEqual(selector({'candidates': [0, 1]}), ('task-output',))
                self.assertEqual(crop_network.logits([], []), ('crop-output',))
                self.assertIsNone(agent.budget.controller())
                self.assertEqual((agent.budget.counts['task_rows'], agent.budget.counts['crop_rows']), (2, 1))
        seen.job_boundary()
        self.assertEqual((seen.result()['target'], seen.result()['status']), ('one', 'not_observed'))
        self.assertIs(agent_policy.BoundedAgent.__init__, original_init)
        seen.close()

    def test_all_forward_rejection_reasons_and_exact_exception_propagation(self):
        cases = (
            ('task', {'task_forwards': 1}, 'task_forwards_total'),
            ('crop', {'crop_forwards': 1}, 'crop_forwards_total'),
            ('task', {'max_task_forwards_per_turn': 1}, 'task_forwards_per_turn'),
            ('crop', {'max_crop_forwards_per_turn': 1}, 'crop_forwards_per_turn'),
            ('task', {'neural_input_rows': 1}, 'neural_input_rows_total'),
            ('task', {'max_neural_input_rows_per_turn': 1}, 'neural_input_rows_per_turn'),
        )
        for kind, limits, expected in cases:
            with self.subTest(reason=expected):
                seen = observer(limits)
                caught = None
                with seen.live_scope(identity_attested=True, phase_attested=True):
                    budget = SyntheticAgent(limits).budget
                    budget.forward(kind, 1)
                    before = dict(budget.counts), dict(budget.turn)
                    try:
                        budget.forward(kind, 1)
                    except RuntimeError as error:
                        caught = error
                    self.assertIsNotNone(caught)
                    self.assertIs(seen._exception, caught)
                    self.assertEqual((budget.counts, budget.turn), before)
                seen.job_boundary(caught)
                result = seen.result()
                self.assertEqual((result['status'], result['method'], result['reasons'], result['job_relation']),
                                 ('observed', 'forward', [expected], 'same_exception'))
                self.assertIsNone(seen._exception)
                seen.close()

    def test_all_simultaneously_triggered_limits_are_recorded(self):
        limits = dict(task_forwards=1, max_task_forwards_per_turn=1,
                      neural_input_rows=1, max_neural_input_rows_per_turn=1)
        seen = observer(limits)
        caught = None
        with seen.live_scope(identity_attested=True, phase_attested=True):
            budget = SyntheticAgent(limits).budget
            budget.forward('task', 1)
            try: budget.forward('task', 1)
            except RuntimeError as error: caught = error
        seen.job_boundary(caught)
        self.assertEqual(seen.result()['reasons'], ['task_forwards_total', 'task_forwards_per_turn',
                                                   'neural_input_rows_total', 'neural_input_rows_per_turn'])

    def test_controller_rejection_types_and_proposal_flag(self):
        cases = (({'controller_calls': 1}, False, 'controller_calls_total'),
                 ({'max_controller_calls_per_turn': 1}, False, 'controller_calls_per_turn'),
                 ({'teacher_proposals': 1}, True, 'teacher_proposals_total'))
        for limits, proposal, reason in cases:
            with self.subTest(reason=reason):
                seen = observer(limits); caught = None
                with seen.live_scope(identity_attested=True, phase_attested=True):
                    budget = SyntheticAgent(limits).budget
                    budget.controller(proposal)
                    before = dict(budget.counts), dict(budget.turn)
                    try: budget.controller(proposal)
                    except RuntimeError as error: caught = error
                    self.assertEqual((budget.counts, budget.turn), before)
                seen.job_boundary(caught)
                self.assertEqual((seen.result()['status'], seen.result()['method'], seen.result()['reasons']),
                                 ('observed', 'controller', [reason]))

    def test_replaced_exception_and_attached_budget_trace_are_not_same(self):
        seen = observer({'neural_input_rows': 1}); original = None
        with seen.live_scope(identity_attested=True, phase_attested=True):
            budget = SyntheticAgent({'neural_input_rows': 1}).budget
            budget.forward('task', 1)
            try: budget.forward('task', 1)
            except RuntimeError as error: original = error
        other = RuntimeError('synthetic-other').with_traceback(original.__traceback__)
        seen.job_boundary(other)
        result = seen.result()
        self.assertEqual((result['status'], result['job_relation']), ('observed', 'different_exception'))
        self.assertNotIn('synthetic-other', json.dumps(result))
        self.assertNotIn('traceback', json.dumps(result))

    def test_other_and_non_budget_errors_never_invent_a_rejection(self):
        seen = observer()
        with seen.live_scope(identity_attested=True, phase_attested=True):
            budget = SyntheticAgent().budget
            try: budget.forward('invalid', 1)
            except KeyError as error: caught = error
        seen.job_boundary(caught)
        self.assertEqual(seen.result()['status'], 'unknown')
        self.assertEqual(seen.result()['reasons'], [])

        # A forged traceback supplied only at the job boundary is not an event.
        seen = observer()
        with seen.live_scope(identity_attested=True, phase_attested=True):
            SyntheticAgent().budget.forward('task', 1)
        seen.job_boundary(caught)
        self.assertEqual(seen.result()['status'], 'not_observed')
        self.assertEqual(seen.result()['job_relation'], 'unknown')

    def test_multiple_instances_are_unknown_and_foreign_budget_is_not_target(self):
        seen = observer()
        with seen.live_scope(identity_attested=True, phase_attested=True):
            first, second = SyntheticAgent(), SyntheticAgent()
            first.budget.forward('task', 1)
            second.budget.forward('task', 1)
        seen.job_boundary()
        self.assertEqual((seen.result()['target'], seen.result()['status']), ('unknown', 'unknown'))

        seen = observer()
        with seen.live_scope(identity_attested=True, phase_attested=True):
            target = SyntheticAgent().budget
            foreign = _Budget(dict(LIMITS, neural_input_rows=1))
            foreign.forward('task', 1)
            with self.assertRaises(RuntimeError): foreign.forward('task', 1)
            target.forward('task', 1)
        seen.job_boundary()
        self.assertEqual((seen.result()['target'], seen.result()['status'], seen.result()['other_budget_seen']),
                         ('one', 'not_observed', True))

    def test_unattested_or_changed_sources_never_claim_coverage(self):
        for identity, phase in ((False, True), (True, False), (1, True)):
            seen = observer()
            with seen.live_scope(identity_attested=identity, phase_attested=phase):
                SyntheticAgent().budget.forward('task', 1)
            seen.job_boundary()
            self.assertEqual((seen.result()['target'], seen.result()['status']), ('unknown', 'unknown'))
            self.assertIsNone(seen.result()['other_budget_seen'])
        seen = BudgetObserver(_Budget, SyntheticAgent, expected_forward=lambda *_: None,
            expected_controller=_Budget.controller, expected_agent_init=SyntheticAgent.__init__,
            expected_limits=LIMITS)
        with seen.live_scope(identity_attested=True, phase_attested=True):
            SyntheticAgent().budget.forward('task', 1)
        seen.job_boundary()
        self.assertEqual(seen.result()['status'], 'unknown')

    def test_observer_classification_failure_preserves_original_exception(self):
        seen = observer({'neural_input_rows': 1}); caught = None
        with mock.patch.object(seen, '_classify', side_effect=RuntimeError('synthetic diagnostic failure')):
            with seen.live_scope(identity_attested=True, phase_attested=True):
                budget = SyntheticAgent({'neural_input_rows': 1}).budget
                budget.forward('task', 1)
                before = dict(budget.counts), dict(budget.turn)
                try: budget.forward('task', 1)
                except RuntimeError as error: caught = error
                self.assertEqual(caught.args, ('Shared neural computation budget exhausted',))
                self.assertEqual((budget.counts, budget.turn), before)
        seen.job_boundary(caught)
        self.assertEqual(seen.result()['status'], 'unknown')
        self.assertNotIn('synthetic diagnostic failure', json.dumps(seen.result()))

    def test_diagnostic_hook_failure_does_not_change_successful_budget_calls(self):
        for method in ('_called', '_register'):
            seen = observer()
            with self.subTest(method=method), \
                 mock.patch.object(seen, method, side_effect=RuntimeError('synthetic diagnostic failure')):
                with seen.live_scope(identity_attested=True, phase_attested=True):
                    budget = SyntheticAgent().budget
                    self.assertIsNone(budget.forward('task', 3))
                    self.assertEqual((budget.counts['task_forwards'], budget.counts['task_rows']), (1, 3))
            seen.job_boundary()
            self.assertEqual(seen.result()['status'], 'unknown')
            self.assertNotIn('synthetic diagnostic failure', json.dumps(seen.result()))

    def test_setup_failure_allows_original_calls(self):
        seen = observer()
        with mock.patch.object(budget_observer, '_wrappers', side_effect=RuntimeError('diagnostic setup')):
            with seen.live_scope(identity_attested=True, phase_attested=True):
                budget = SyntheticAgent().budget
                self.assertIsNone(budget.forward('task', 3))
                self.assertEqual(budget.counts['task_rows'], 3)
        seen.job_boundary()
        self.assertEqual(seen.result()['status'], 'unknown')
        self.assertIsNone(seen.result()['other_budget_seen'])

        seen = observer()
        fired = [False]
        def fail_one_installation(obj, name, value):
            builtins.setattr(obj, name, value)
            if obj is _Budget and name == 'forward' and not fired[0]:
                fired[0] = True
                raise RuntimeError('synthetic setup failure')
        with mock.patch.object(budget_observer, 'setattr', side_effect=fail_one_installation, create=True):
            with seen.live_scope(identity_attested=True, phase_attested=True):
                self.assertIs(_Budget.forward, self.originals[0])
                budget = SyntheticAgent().budget
                self.assertIsNone(budget.forward('task', 3))
                self.assertEqual(budget.counts['task_rows'], 3)
        seen.job_boundary()
        self.assertEqual(seen.result()['status'], 'unknown')

    def test_cancellation_propagates_and_hooks_and_lock_are_restored(self):
        for signal_type in (KeyboardInterrupt, SystemExit):
            for stage in ('authenticate', 'build', 'install', 'register', 'called', 'classify', 'restore'):
                with self.subTest(signal=signal_type.__name__, stage=stage):
                    signal = signal_type('synthetic cancellation')
                    limits = {'neural_input_rows': 1} if stage == 'classify' else {}
                    seen = observer(limits)
                    if stage == 'authenticate':
                        patcher = mock.patch.object(seen, '_authenticatable', side_effect=signal)
                    elif stage == 'build':
                        patcher = mock.patch.object(budget_observer, '_wrappers', side_effect=signal)
                    elif stage == 'install':
                        fired = [False]
                        def cancel_one_installation(obj, name, value):
                            builtins.setattr(obj, name, value)
                            if obj is _Budget and name == 'forward' and not fired[0]:
                                fired[0] = True
                                raise signal
                        patcher = mock.patch.object(budget_observer, 'setattr',
                                                  side_effect=cancel_one_installation, create=True)
                    elif stage in ('register', 'called', 'classify'):
                        patcher = mock.patch.object(seen, '_' + stage, side_effect=signal)
                    else:
                        fired = [False]
                        def cancel_one_restoration(obj, name, *default):
                            if obj is SyntheticAgent and name == '__init__' and not fired[0]:
                                fired[0] = True
                                raise signal
                            return builtins.getattr(obj, name, *default)
                        patcher = mock.patch.object(budget_observer, 'getattr',
                                                  side_effect=cancel_one_restoration, create=True)
                    with patcher, self.assertRaises(signal_type) as raised:
                        with seen.live_scope(identity_attested=True, phase_attested=True):
                            budget = SyntheticAgent(limits).budget
                            budget.forward('task', 1)
                            if stage == 'classify':
                                budget.forward('task', 1)
                    self.assertIs(raised.exception, signal)
                    if stage == 'classify':
                        self.assertIsInstance(signal.__context__, RuntimeError)
                    self.assertEqual(seen.result()['status'], 'unknown')
                    self.assertIsNone(seen.result()['other_budget_seen'])
                    self.assertIs(_Budget.forward, self.originals[0])
                    self.assertIs(_Budget.controller, self.originals[1])
                    self.assertIs(SyntheticAgent.__init__, self.originals[2])
                    fresh = observer()
                    with fresh.live_scope(identity_attested=True, phase_attested=True):
                        SyntheticAgent().budget.forward('task', 1)
                    fresh.job_boundary()
                    self.assertEqual(fresh.result()['status'], 'not_observed')

    def test_invalid_or_mismatched_expected_limits_leave_calls_unchanged(self):
        invalid = (None, {}, dict(LIMITS, neural_input_rows=0),
                   dict(LIMITS, neural_input_rows=True), dict(LIMITS, extra=1),
                   dict(LIMITS, neural_input_rows=1))
        for expected in invalid:
            with self.subTest(expected=expected):
                seen = BudgetObserver(_Budget, SyntheticAgent, expected_forward=_Budget.forward,
                    expected_controller=_Budget.controller, expected_agent_init=SyntheticAgent.__init__,
                    expected_limits=expected)
                with seen.live_scope(identity_attested=True, phase_attested=True):
                    budget = SyntheticAgent().budget
                    self.assertIsNone(budget.forward('task', 3))
                    self.assertEqual(budget.counts['task_rows'], 3)
                seen.job_boundary()
                self.assertEqual(seen.result()['status'], 'unknown')
                self.assertIsNone(seen.result()['other_budget_seen'])

    def test_changed_or_invalid_limits_cannot_be_classified_as_rejection(self):
        for change in ('zero', 'different', 'bool', 'extra', 'missing'):
            with self.subTest(change=change):
                seen = observer({'neural_input_rows': 1})
                with seen.live_scope(identity_attested=True, phase_attested=True):
                    budget = SyntheticAgent({'neural_input_rows': 1}).budget
                    budget.forward('task', 1)
                    before = dict(budget.counts), dict(budget.turn)
                    if change == 'zero': budget.limits['neural_input_rows'] = 0
                    if change == 'different': budget.limits['neural_input_rows'] = 2
                    if change == 'bool': budget.limits['neural_input_rows'] = True
                    if change == 'extra': budget.limits['extra'] = 1
                    if change == 'missing': del budget.limits['neural_input_rows']
                    with self.assertRaises((RuntimeError, KeyError)) as raised:
                        budget.forward('task', 2)
                    self.assertEqual((budget.counts, budget.turn), before)
                seen.job_boundary(raised.exception)
                self.assertEqual(seen.result()['status'], 'unknown')
                self.assertEqual(seen.result()['reasons'], [])

    def test_mismatch_at_binding_or_boundary_is_unknown(self):
        class ChangedAgent:
            def __init__(self):
                self.budget = _Budget(dict(LIMITS))
                self.budget.limits['neural_input_rows'] = 0
        seen = BudgetObserver(_Budget, ChangedAgent, expected_forward=_Budget.forward,
            expected_controller=_Budget.controller, expected_agent_init=ChangedAgent.__init__,
            expected_limits=LIMITS)
        with seen.live_scope(identity_attested=True, phase_attested=True):
            with self.assertRaises(RuntimeError) as raised:
                ChangedAgent().budget.forward('task', 1)
        seen.job_boundary(raised.exception)
        self.assertEqual(seen.result()['status'], 'unknown')
        seen = observer()
        with seen.live_scope(identity_attested=True, phase_attested=True):
            budget = SyntheticAgent().budget
            budget.forward('task', 1)
        budget.limits['neural_input_rows'] = 0
        seen.job_boundary()
        self.assertEqual(seen.result()['status'], 'unknown')

    def test_replaced_code_object_during_rejection_is_unknown(self):
        def replacement(self, kind, rows):
            raise RuntimeError('Shared neural computation budget exhausted')
        seen = observer({'neural_input_rows': 1})
        original_code = seen.expected_forward.__code__
        with seen.live_scope(identity_attested=True, phase_attested=True):
            budget = SyntheticAgent({'neural_input_rows': 1}).budget
            budget.forward('task', 1)
            try:
                seen.expected_forward.__code__ = replacement.__code__
                with self.assertRaises(RuntimeError) as raised:
                    budget.forward('task', 1)
            finally:
                seen.expected_forward.__code__ = original_code
        seen.job_boundary(raised.exception)
        self.assertEqual(seen.result()['status'], 'unknown')
        self.assertEqual(seen.result()['reasons'], [])

    def test_restored_code_object_still_checks_actual_raising_frame(self):
        seen = observer({'neural_input_rows': 1})
        original = seen.expected_forward
        original_code = original.__code__
        namespace = original.__globals__
        absent = object()
        previous = namespace.get('_synthetic_observer_restore', absent)
        try:
            namespace['_synthetic_observer_restore'] = (original, original_code)
            definition = {}
            exec("def replacement(self, kind, rows):\n"
                 "    _synthetic_observer_restore[0].__code__ = _synthetic_observer_restore[1]\n"
                 "    raise RuntimeError('Shared neural computation budget exhausted')\n",
                 namespace, definition)
            with seen.live_scope(identity_attested=True, phase_attested=True):
                budget = SyntheticAgent({'neural_input_rows': 1}).budget
                budget.forward('task', 1)
                original.__code__ = definition['replacement'].__code__
                with self.assertRaises(RuntimeError) as raised:
                    budget.forward('task', 1)
        finally:
            original.__code__ = original_code
            if previous is absent:
                namespace.pop('_synthetic_observer_restore', None)
            else:
                namespace['_synthetic_observer_restore'] = previous
        seen.job_boundary(raised.exception)
        self.assertEqual(seen.result()['status'], 'unknown')
        self.assertEqual(seen.result()['reasons'], [])

    def test_error_from_nested_call_is_not_direct_budget_raise(self):
        def other_origin(message):
            raise RuntimeError(message)
        seen = observer({'neural_input_rows': 1})
        budget_globals = seen.expected_forward.__globals__
        absent = object()
        previous = budget_globals.get('RuntimeError', absent)
        try:
            with seen.live_scope(identity_attested=True, phase_attested=True):
                budget = SyntheticAgent({'neural_input_rows': 1}).budget
                budget.forward('task', 1)
                budget_globals['RuntimeError'] = other_origin
                with self.assertRaises(RuntimeError) as raised:
                    budget.forward('task', 1)
        finally:
            if previous is absent:
                budget_globals.pop('RuntimeError', None)
            else:
                budget_globals['RuntimeError'] = previous
        seen.job_boundary(raised.exception)
        self.assertEqual(seen.result()['status'], 'unknown')
        self.assertEqual(seen.result()['reasons'], [])

    def test_unrelated_runtime_error_does_not_become_budget_rejection(self):
        unrelated = RuntimeError('synthetic unrelated private text')
        class OtherBudget:
            def __init__(self):
                self.limits = dict(LIMITS)
            def forward(self, kind, rows):
                raise unrelated
            def controller(self, proposal=False):
                return None
        namespace = {'_Budget': OtherBudget}
        exec('class OtherAgent:\n def __init__(self): self.budget = _Budget()\n', namespace)
        other_agent = namespace['OtherAgent']
        seen = BudgetObserver(OtherBudget, other_agent, expected_forward=OtherBudget.forward,
            expected_controller=OtherBudget.controller, expected_agent_init=other_agent.__init__,
            expected_limits=LIMITS)
        caught = None
        with seen.live_scope(identity_attested=True, phase_attested=True):
            budget = other_agent().budget
            try: budget.forward('task', 1)
            except RuntimeError as error: caught = error
        seen.job_boundary(caught)
        self.assertIs(caught, unrelated)
        self.assertEqual(seen.result()['status'], 'unknown')
        self.assertEqual(seen.result()['reasons'], [])
        self.assertNotIn('synthetic unrelated private text', json.dumps(seen.result()))

    def test_duplicate_job_boundary_invalidates_prior_propagation_claim(self):
        seen = observer({'neural_input_rows': 1}); caught = None
        with seen.live_scope(identity_attested=True, phase_attested=True):
            budget = SyntheticAgent({'neural_input_rows': 1}).budget
            budget.forward('task', 1)
            try: budget.forward('task', 1)
            except RuntimeError as error: caught = error
        seen.job_boundary(caught)
        self.assertEqual(seen.result()['job_relation'], 'same_exception')
        seen.job_boundary(RuntimeError('later replacement'))
        self.assertEqual((seen.result()['status'], seen.result()['job_relation']), ('unknown', 'unknown'))

    def test_context_exception_restores_hooks_and_unobserved_boundary_is_unknown(self):
        seen = observer()
        with self.assertRaisesRegex(RuntimeError, 'synthetic unrelated'):
            with seen.live_scope(identity_attested=True, phase_attested=True):
                SyntheticAgent().budget.forward('task', 1)
                raise RuntimeError('synthetic unrelated')
        self.assertEqual(seen.result()['status'], 'not_observed')
        self.assertEqual(seen.result()['job_relation'], 'unknown')
        seen.close()

    def test_missing_and_unobserved_job_boundaries_never_claim_propagation(self):
        seen = observer({'neural_input_rows': 1})
        with seen.live_scope(identity_attested=True, phase_attested=True):
            budget = SyntheticAgent({'neural_input_rows': 1}).budget
            budget.forward('task', 1)
            with self.assertRaises(RuntimeError): budget.forward('task', 1)
        self.assertEqual(seen.result()['job_relation'], 'unknown')
        seen.job_boundary()
        self.assertEqual(seen.result()['job_relation'], 'job_returned')


if __name__ == '__main__':
    unittest.main()
