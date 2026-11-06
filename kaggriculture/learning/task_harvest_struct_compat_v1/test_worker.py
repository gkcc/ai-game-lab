"""Real official Struct tests with zero actors, environments or transitions."""
from collections import Counter
from copy import deepcopy
import io
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest

from . import compat, worker


def request():
    return dict(farmer=['PASS'], hands=[['MOVE', 'E']],
                market=[['SELL', 'WHEAT', 2], ['BUY', 'SEED_WHEAT', 3]])


def frame(action):
    return [dict(observation=dict(step=241, private=dict(shed={'WHEAT': 1, 'MELON': 2},
        inventories=[{'WHEAT': 2, 'MELON': 1}])), action=deepcopy(action), status='ACTIVE', reward=0)
        for _ in range(2)]


class Sink:
    def __init__(self, fail=None):
        self.events, self.fail = 241, fail
    def __call__(self, event):
        if self.fail is not None:
            raise self.fail
        self.events += 1


class OfficialStructTests(unittest.TestCase):
    forbidden_calls = []

    @classmethod
    def setUpClass(cls):
        cls.previous_profile = sys.getprofile()
        def reject_execution(frame, event, argument):
            if event != 'call':
                return
            filename, name = frame.f_code.co_filename.replace('\\', '/'), frame.f_code.co_name
            blocked = (filename.endswith('/kaggle_environments/core.py') and name in ('make', 'step', '__init__')
                or filename.endswith('/envs/kaggriculture/kaggriculture.py') and name == 'interpreter'
                or name == '__init__' and any(filename.endswith(path) for path in (
                    '/learning/task_dagger_once_v1/policy.py', '/learning/task_bc_v1/model.py',
                    '/learning/bc_crop/model.py')))
            if blocked:
                cls.forbidden_calls.append(name)
                raise AssertionError('Test attempted a model or official environment invocation')
        sys.setprofile(reject_execution)
        try:
            from learning.task_student_continuation_v1 import official_step
            from learning.task_request_gate_v1.policy import same_request
            from learning.task_harvest_unexpressed_v1 import core
            cls.core, cls.original = core, staticmethod(same_request)
            cls.Struct, cls.binding = compat.authenticate(official_step.official_core, same_request)
            cls.utils = sys.modules[compat.UTILS_MODULE]
        except BaseException:
            sys.setprofile(cls.previous_profile)
            raise

    @classmethod
    def tearDownClass(cls):
        sys.setprofile(cls.previous_profile)
        if cls.forbidden_calls:
            raise AssertionError('Forbidden execution occurred')

    def compare(self):
        return compat.Comparator(self.Struct, self.original)

    def test_actual_root_and_nested_struct_pass_without_changing_input(self):
        original = request()
        actual = self.utils.structify(original)
        nested = {'outer': [self.Struct(value=2, nested=self.Struct(enabled=True))]}
        nested_plain = {'outer': [{'value': 2, 'nested': {'enabled': True}}]}
        self.assertIs(type(actual), self.Struct)
        with self.assertRaises(ValueError):
            self.original(actual, original)
        compare = self.compare()
        self.assertTrue(compare(actual, original))
        self.assertTrue(compare(nested, nested_plain))
        copied, count = compat.comparison_copy(actual, self.Struct)
        self.assertGreater(count, 0)
        self.assertIs(type(copied), dict)
        copied['market'][0][2] = 999
        self.assertEqual(actual['market'][0][2], 2)
        self.assertIs(type(actual), self.Struct)
        self.assertEqual(actual, original)
        self.assertIs(type(nested['outer'][0]), self.Struct)

    def test_plain_equivalence_scalar_types_and_order_are_preserved(self):
        pairs = [(None, None), (True, 1), (1, 1.0), (-0.0, 0.0),
                 ({'a': 1, 'b': 2}, {'b': 2, 'a': 1}), ([1, 2], [2, 1]),
                 (request(), request())]
        changed = request()
        changed['market'].reverse()
        pairs.append((request(), changed))
        for left, right in pairs:
            self.assertEqual(self.compare()(left, right), self.original(left, right))
        self.assertFalse(self.compare()(self.Struct(value=True), {'value': 1}))
        self.assertFalse(self.compare()(self.Struct(value=1), {'value': 1.0}))

    def test_invalid_keys_nonfinite_and_unknown_container_subclasses_still_reject(self):
        class ForeignDict(dict):
            pass
        class ForeignList(list):
            pass
        class StructChild(self.Struct):
            pass
        class IntChild(int):
            pass
        values = [float('nan'), float('inf'), float('-inf'), {1: 'bad'}, {True: 'bad'},
                  ForeignDict(ok=1), ForeignList([1]), StructChild(ok=1), IntChild(1), (1,), object(),
                  {'inside': ForeignDict(ok=1)}, self.Struct(value=ForeignList([1]))]
        for value in values:
            with self.assertRaises((ValueError, TypeError)):
                self.compare()(value, value)
        for value in values[:-1]:
            with self.assertRaises((ValueError, TypeError)):
                self.original(value, value)

    def test_two_compiled_cores_bind_real_function_globals_and_keep_old_checks(self):
        first, second = worker._frozen_core(), worker._frozen_core()
        old = self.core.same_request
        first.same_request, second.same_request = self.compare(), self.compare()
        actual = self.utils.structify(request())
        self.assertTrue(first._run_arm.__globals__['same_request'](actual, request()))
        self.assertEqual(first.same_request.snapshot()['struct_calls'], 1)
        self.assertEqual(second.same_request.snapshot()['struct_calls'], 0)
        self.assertIs(first.run_pair.__globals__, vars(first))
        self.assertIs(second.run_pair.__globals__, vars(second))
        self.assertIsNot(vars(first), vars(second))
        self.assertIs(self.core.same_request, old)
        for name in ('_request', 'request_digest', 'frame_digest', '_restore', '_observation'):
            self.assertIs(getattr(first, name), getattr(self.core, name))
            self.assertIs(getattr(second, name), getattr(self.core, name))
        for operation in (first._request, first.request_digest):
            with self.assertRaises(ValueError):
                operation(actual)  # Deployment output validation is deliberately still strict.
        with self.assertRaises(ValueError):
            old(actual, request())

    def test_original_frame_shed_and_backpack_order_checks_are_unchanged(self):
        core = worker._frozen_core()
        plain = frame(request())
        actual = self.utils.structify(plain)
        self.assertEqual(core.frame_digest(actual), core.frame_digest(plain))
        for field in ('shed', 'inventories'):
            changed = deepcopy(plain)
            private = changed[0]['observation']['private']
            if field == 'shed':
                private[field] = dict(reversed(list(private[field].items())))
            else:
                private[field][0] = dict(reversed(list(private[field][0].items())))
            self.assertNotEqual(core.frame_digest(plain), core.frame_digest(changed))

    def test_binding_requires_core_structify_class_identity_not_its_name(self):
        from learning.task_student_continuation_v1 import official_step
        counterfeit = ModuleType('counterfeit_official_core')
        counterfeit.__file__ = official_step.official_core.__file__
        counterfeit.structify = lambda value: self.Struct(**value)
        with self.assertRaises(ValueError):
            compat.authenticate(counterfeit, self.original)
        self.assertEqual(self.binding, compat.identity())

    def _gate(self, sink=None):
        receipt, compare = worker._new_receipt(), self.compare()
        meter = SimpleNamespace(calls=Counter(official_step_attempts=1, official_step_returns=1,
                                              suffix_actor_attempts=1))
        gate = worker.FirstSuffixGate(receipt, compare, self.core, meter,
                                     {'step': 240}, sink or Sink(), lambda: None)
        for _ in range(2):
            compare(request(), request())
        for _ in range(2):
            compare(self.utils.structify(request()), request())
        return gate, receipt

    def test_first_actual_struct_suffix_must_be_saved_before_next_loop_guard(self):
        gate, receipt = self._gate()
        gate.guard()  # The frozen post-step check, before first emit.
        event = dict(kind='turn', arm='control', scope='suffix', step=240,
                     frame=self.utils.structify(frame(request())))
        gate.emit(event)
        self.assertTrue(receipt['first_factual_suffix']['original_sink_returned'])
        self.assertEqual(receipt['first_factual_suffix']['saved_event_number'], 242)
        gate.guard()  # Original second suffix loop guard, before its actor call.
        self.assertTrue(receipt['second_suffix_gate_before_actor2'])

    def test_failed_sink_or_missing_first_suffix_cannot_open_second_loop(self):
        primary = OSError('synthetic output failure')
        gate, receipt = self._gate(Sink(primary))
        gate.guard()
        try:
            gate.emit(dict(kind='turn', arm='control', scope='suffix', step=240,
                           frame=self.utils.structify(frame(request()))))
        except OSError as caught:
            self.assertIs(caught, primary)
        else:
            self.fail('Original sink exception was swallowed')
        self.assertIsNone(receipt['first_factual_suffix'])
        with self.assertRaises(ValueError):
            gate.guard()
        self.assertFalse(receipt['second_suffix_gate_before_actor2'])

    def test_plain_only_suffix_cannot_claim_actual_struct_execution(self):
        gate, receipt = self._gate()
        with self.assertRaises(ValueError):
            gate.emit(dict(kind='turn', arm='control', scope='suffix', step=240,
                           frame=frame(request())))
        self.assertIsNone(receipt['first_factual_suffix'])

    def test_real_frozen_event_sink_completes_its_write_and_both_guards_before_gate(self):
        module = worker.prior._frozen_worker()
        for fail_after_write in (False, True):
            stream, receipt, compare = io.BytesIO(), worker._new_receipt(), self.compare()
            primary, armed = OSError('synthetic final sink guard failure'), False
            sink = None
            def original_guard():
                if armed:
                    self.assertIsNone(receipt['first_factual_suffix'])
                    if fail_after_write and sink.events == 242:
                        raise primary
            sink = module.EventSink(stream, original_guard)
            body = request()
            def turn(step, scope):
                total = {key: 0 for key in self.core.BUDGET_KEYS}
                total['controller_calls'] = step+1
                one = dict(total, controller_calls=1)
                trace = dict(step=step, records=[], crop_records=[], fallbacks=0,
                    load_errors=0, inference_errors=0,
                    final_reservations={'unit_actions': [body['farmer'], *body['hands']]},
                    model_counts=total, turn_model_counts=one)
                return dict(kind='turn', arm='control', scope=scope, step=step,
                    request=deepcopy(body), original_request=deepcopy(body), rival_request=deepcopy(body),
                    trace=trace, stable_trace_sha256=self.core.ordered_hash(module._stable(trace)),
                    patch=None, target_execution=None, frame=self.utils.structify(frame(body)))
            sink(dict(kind='arm_start', arm='control', step=240, frame=frame(body)))
            for step in range(240):
                sink(turn(step, 'prefix'))
            meter = SimpleNamespace(calls=Counter(official_step_attempts=1,
                official_step_returns=1, suffix_actor_attempts=1))
            gate = worker.FirstSuffixGate(receipt, compare, self.core, meter,
                {'step': 240}, sink, lambda: None)
            for value in (body, body, self.utils.structify(body), self.utils.structify(body)):
                compare(value, body)
            gate.guard()
            armed = True
            if fail_after_write:
                try:
                    gate.emit(turn(240, 'suffix'))
                except OSError as caught:
                    self.assertIs(caught, primary)
                else:
                    self.fail('Frozen EventSink final guard failure was hidden')
                self.assertIsNone(receipt['first_factual_suffix'])
                with self.assertRaises(ValueError):
                    gate.guard()
            else:
                gate.emit(turn(240, 'suffix'))
                self.assertTrue(receipt['first_factual_suffix']['original_sink_returned'])
                gate.guard()
                self.assertTrue(receipt['second_suffix_gate_before_actor2'])
            saved = [json.loads(line) for line in stream.getvalue().splitlines()]
            self.assertEqual(len(saved), 242)
            self.assertEqual((saved[-1]['scope'], saved[-1]['step']), ('suffix', 240))

    def test_bad_source_bytes_reject_before_execution_and_failure_capture_is_secondary(self):
        with self.assertRaises(ValueError):
            worker._compile_core(b'raise AssertionError("not executed")', 'synthetic.py')
        primary, observed = ValueError('primary'), []
        class Registry:
            def identify(self, error):
                observed.append(error)
                raise MemoryError('capture failed')
        receipt = worker._new_receipt()
        worker._capture_failure(receipt, Registry(), primary)
        self.assertEqual(observed, [primary])
        self.assertTrue(receipt['failure_location_capture_failed'])
        self.assertIsNone(receipt['failure_location'])

    def test_receipt_identity_booleans_and_saved_event_counts_keep_exact_json_types(self):
        receipt = worker._new_receipt()
        counts = dict(compat.empty_stats(), calls=1251, returns=1251, true_returns=1251,
            struct_calls=192, struct_objects=192, struct_true_returns=192, struct_root_left_calls=192)
        first_counts = dict(compat.empty_stats(), calls=4, returns=4, true_returns=4,
            struct_calls=2, struct_objects=2, struct_true_returns=2, struct_root_left_calls=2)
        receipt.update(status='completed', binding=compat.identity(), comparison_counts=counts,
            second_suffix_gate_before_actor2=True, suffix_events_saved=dict(control=48, intervene=48),
            first_factual_suffix=dict(step=240, arm='control', original_sink_returned=True,
                official_step_attempts=1, official_step_returns=1, saved_event_number=242,
                comparison_delta=first_counts,
                accepted_action_types=['exact_official_Struct', 'exact_official_Struct'],
                frame_physical_sha256='a'*64))
        document = {worker.FIELD: receipt, 'target': {'step': 240},
                    'core': {'control': {'physical_state_sha256': ['b'*64, 'a'*64]}}}
        worker._verify_receipt(document)
        altered = deepcopy(document)
        altered[worker.FIELD]['identity']['original_output_validation_unchanged'] = 1
        with self.assertRaises(ValueError):
            worker._verify_receipt(altered)
        for arm in ('control', 'intervene'):
            altered = deepcopy(document)
            altered[worker.FIELD]['suffix_events_saved'][arm] = 48.0
            with self.assertRaises(ValueError):
                worker._verify_receipt(altered)


if __name__ == '__main__':
    unittest.main()
