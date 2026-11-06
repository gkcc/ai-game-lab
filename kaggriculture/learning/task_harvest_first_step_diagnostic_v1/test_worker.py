"""Small synthetic checks; real files only inside root-owned test scratch."""
from collections import Counter
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

from . import capture, worker


def _compiled(raw, path='synthetic_registered.py'):
    namespace = {}
    exec(compile(raw, path, 'exec'), namespace)
    return namespace, capture.RegisteredSources([(path, raw, [path])])


def _frame(action=None):
    return [dict(observation=dict(step=241, private=dict(shed={'WHEAT': 3, 'MELON': 4}, inventories=[{}])),
                 action=deepcopy(action), status='ACTIVE', reward=0) for _ in (0, 1)]


ARM = b'''def _run_arm(error, frame, request, runtime, records, emit=None):
    mode, step, offset = 'control', 240, 0
    original = request
    tracer = type('Tracer', (), {'records': records})()
    environment = type('Environment', (), {'state': frame, 'interpreter': runtime.interpreter})()
    trace = {'model_counts': {'controller_calls': 241}, 'turn_model_counts': {'controller_calls': 1}}
    if emit is not None:
        emit({'kind': 'turn', 'arm': 'control', 'scope': 'suffix', 'step': 240})
    raise error
'''


def _collector_fixture(comparator=None):
    namespace, registry = _compiled(ARM)
    original = dict(farmer=['PASS'], hands=[], market=[])
    frame = _frame(original)
    runtime = SimpleNamespace(interpreter=object(), stable=lambda value: value)
    from learning.task_harvest_window_v1.core import frame_digest
    core = SimpleNamespace(_run_arm=namespace['_run_arm'], frame_digest=frame_digest,
                           same_request=comparator or (lambda a, b: a == b))
    replay = dict(steps=[None]*241+[deepcopy(frame)])
    return core, registry, original, frame, runtime, replay


def _collect(error, *, comparator=None, records=None):
    core, registry, original, frame, runtime, replay = _collector_fixture(comparator)
    collector = capture.Collector(registry)
    try:
        core._run_arm(error, frame, original, runtime, [] if records is None else records)
    except BaseException as caught:
        collector.record(caught, core=core, target={'side': 0}, replay=replay, runtime=runtime)
        same = caught is error
    return collector, same


class Sink:
    def __init__(self, failure=None):
        self.events, self.decoded_bytes, self.failure = 0, 0, failure

    def __call__(self, event):
        if self.failure is not None:
            raise self.failure
        self.events += 1
        self.decoded_bytes += len(json.dumps(event))


def _module(run):
    from learning.task_harvest_window_v1.core import Meter
    module = ModuleType('synthetic_worker_namespace')
    module.core = ModuleType('synthetic_core_namespace')
    module.core.Meter, module.core.run_pair = Meter, run
    module._write_case = lambda dm, out, document: deepcopy(document)
    return module


class PureTests(unittest.TestCase):
    def test_static_raise_vs_unregistered_callee_and_line_tampering(self):
        raw = b'''import json
def fail():
    raise ValueError('fixed assertion')
def via_library():
    json.loads('bad')
'''
        ns, registry = _compiled(raw)
        for name, wanted in (('fail', 'direct_raise'), ('via_library', 'registered_callsite_only')):
            try:
                ns[name]()
            except ValueError as error:
                value = registry.identify(error)
            self.assertEqual(value['location_kind'], wanted)
            registry.validate_location(value)
            self.assertNotIn('fixed assertion', json.dumps(value))
            if name == 'fail':
                tampered = deepcopy(value)
                tampered['location']['line'] = 2
                tampered['locations'][-1]['line'] = 2
                with self.assertRaisesRegex(ValueError, 'static line'):
                    registry.validate_location(tampered)

    def test_registered_type_rejection_records_only_safe_container_label(self):
        raw = b"def _typed(value):\n    raise ValueError('Request contains an unsupported non-JSON value')\n"
        path = 'kaggriculture/learning/task_request_gate_v1/policy.py'
        ns, registry = _compiled(raw, path)
        Struct = type('Struct', (dict,), {'__module__': 'kaggle_environments.utils'})
        try:
            ns['_typed'](Struct(secret='must never be copied'))
        except ValueError as error:
            value = registry.identify(error)
        self.assertEqual(value['rejected_value_kind'], 'official_Struct')
        self.assertNotIn('secret', json.dumps(value))
        registry.validate_location(value)

    def test_capture_preserves_original_identity_and_whitelist(self):
        original = ValueError('private content cannot appear in metadata')
        collector, same = _collect(original)
        self.assertTrue(same)
        self.assertEqual(collector.capture['status'], 'complete')
        self.assertEqual(collector.context['original_type_request_comparison']['outcome'], 'equal')
        self.assertNotIn('private content', json.dumps(collector.snapshot()))
        self.assertNotIn('info', collector.context)
        self.assertNotIn('trace', collector.context)
        self.assertLessEqual(collector.capture['bytes'], capture.MAX_CONTEXT_BYTES)

    def test_subtype_comparison_failure_is_not_laundered_by_json_equality(self):
        def comparator(left, right):
            raise ValueError('strict type compatibility fixture')
        collector, same = _collect(ValueError('primary'), comparator=comparator)
        self.assertTrue(same)
        self.assertEqual(collector.capture['status'], 'complete')
        self.assertEqual(collector.context['original_type_request_comparison'],
                         dict(outcome='raises', error_kind='ValueError'))
        comparison = collector.context['comparison']
        self.assertEqual(comparison['actual_ordered_sha256'], comparison['expected_ordered_sha256'])

    def test_capture_limit_and_comparison_resource_error_fail_closed(self):
        oversized = [{'text': 'x'*capture.MAX_CONTEXT_BYTES}]
        collector, same = _collect(ValueError('primary'), records=oversized)
        self.assertTrue(same)
        self.assertEqual(collector.capture['status'], 'limit_exceeded')
        self.assertIsNone(collector.context)
        for cls in (MemoryError, TimeoutError, OSError, KeyboardInterrupt, SystemExit):
            def comparator(left, right, error_type=cls):
                raise error_type()
            collector, same = _collect(ValueError('primary'), comparator=comparator)
            self.assertTrue(same)
            self.assertEqual(collector.capture['status'], 'capture_failed')
            self.assertIsNone(collector.context)

    def test_all_single_step_caps_refuse_before_next_call(self):
        from learning.task_harvest_window_v1.core import Meter
        cls = worker._capped_meter(Meter)
        limits = dict(arm=1, actor_construct=1, rival_construct=1, restore=1,
                      official_step=1, request_patch=0)
        for op, maximum in limits.items():
            meter, actual_calls = cls(), 0
            scope = 'suffix' if op in ('official_step', 'request_patch') else None
            for _ in range(maximum):
                meter.start(op, scope)
                actual_calls += 1
                meter.returned(op, scope)
            with self.assertRaises(capture.DiagnosticCapExceeded):
                meter.start(op, scope)
                actual_calls += 1
            self.assertEqual(actual_calls, maximum)
        for op in ('actor', 'rival'):
            meter = cls()
            for _ in range(240):
                meter.start(op, 'prefix')
                meter.returned(op, 'prefix')
            meter.start(op, 'suffix')
            meter.returned(op, 'suffix')
            with self.assertRaises(capture.DiagnosticCapExceeded):
                meter.start(op, 'suffix')
            self.assertEqual(meter.calls[op+'_attempts'], 241)

    def test_first_suffix_stops_after_original_sink_and_sink_failure_stays_original(self):
        after = []
        def run(replay, traces, **kwargs):
            kwargs['emit'](dict(kind='turn', arm='control', scope='suffix', step=240))
            after.append('second turn would begin')
        module = _module(run)
        shared_run = module.core.run_pair
        collector = capture.Collector(capture.RegisteredSources([]))
        worker._adapt(module, collector)
        sink = Sink()
        with self.assertRaises(capture.FirstStepComplete):
            module.core.run_pair({}, None, target=dict(step=240, side=0), horizon=48,
                                 emit=sink, runtime=None)
        self.assertTrue(collector.suffix_sink_completed)
        self.assertEqual(sink.events, 1)
        self.assertEqual(after, [])
        self.assertIsNot(module.core.run_pair, shared_run)
        primary = OSError('sink failure')
        module, collector = _module(run), capture.Collector(capture.RegisteredSources([]))
        worker._adapt(module, collector)
        try:
            module.core.run_pair({}, None, target=dict(step=240, side=0), horizon=48,
                                 emit=Sink(primary), runtime=None)
        except OSError as caught:
            self.assertIs(caught, primary)
        else:
            self.fail('Original sink failure was swallowed')
        self.assertFalse(collector.suffix_sink_completed)

    def test_capture_failure_never_replaces_original_exception(self):
        primary = ValueError('original')
        def run(*args, **kwargs):
            raise primary
        module, collector = _module(run), capture.Collector(capture.RegisteredSources([]))
        worker._adapt(module, collector)
        with patch.object(collector, 'record', side_effect=RuntimeError('capture defect')):
            try:
                module.core.run_pair({}, None, target=dict(step=240, side=0), horizon=48,
                                     emit=Sink(), runtime=None)
            except ValueError as caught:
                self.assertIs(caught, primary)
            else:
                self.fail('Original exception was swallowed')
        self.assertEqual(collector.capture['status'], 'capture_failed')

    def test_whole_case_reserve_not_only_context_is_bounded(self):
        module = _module(lambda *args, **kwargs: None)
        collector = capture.Collector(capture.RegisteredSources([]))
        collector.context = {'allowed': 'x'*200}
        collector.capture = dict(status='complete', bytes=200, sha256='0'*64)
        worker._adapt(module, collector)
        with patch.object(worker, 'CASE_RESERVE', 800):
            value = module._write_case(None, Path('nonexistent-synthetic-output'),
                                      {'legacy_header': 'x'*300})
        self.assertEqual(value[worker.FIELD]['capture']['status'], 'case_limit_exceeded')
        self.assertIsNone(value[worker.FIELD]['context'])

    def test_post_return_meter_required_and_forbidden_work_rejected(self):
        calls = dict(arm_attempts=1, actor_construct_attempts=1, actor_construct_returns=1,
            rival_construct_attempts=1, rival_construct_returns=1, actor_attempts=241,
            actor_returns=241, rival_attempts=241, rival_returns=241,
            prefix_actor_attempts=240, prefix_actor_returns=240,
            prefix_rival_attempts=240, prefix_rival_returns=240,
            suffix_actor_attempts=1, suffix_actor_returns=1, suffix_rival_attempts=1,
            suffix_rival_returns=1, restore_attempts=1, restore_returns=1,
            official_step_attempts=1, official_step_returns=1,
            suffix_official_step_attempts=1, suffix_official_step_returns=1)
        meter = dict(calls=calls, diagnostic_cap_refusals=0, unknown_budget_observations=0)
        worker._meter_check(meter)
        for key, value in (('official_step_returns', 0), ('request_patch_attempts', 1), ('arm_returns', 1)):
            bad = deepcopy(meter)
            bad['calls'][key] = value
            with self.assertRaises(ValueError):
                worker._meter_check(bad)

    def test_context_consumer_recomputes_order_and_rejects_resealed_tampering(self):
        collector, _ = _collect(ValueError('primary'))
        core, registry, *_ = _collector_fixture()
        module = SimpleNamespace(core=core)
        value = dict(schema=worker.SCHEMA, **collector.snapshot(), diagnostic_only=True,
                     original_core_completed=False)
        worker._context_check(value, {'side': 0}, module, registry)
        tampered = deepcopy(value)
        shed = tampered['context']['actual_after_frame'][0]['observation']['private']['shed']
        tampered['context']['actual_after_frame'][0]['observation']['private']['shed'] = dict(reversed(list(shed.items())))
        raw = capture.bounded_json(tampered['context'], capture.MAX_CONTEXT_BYTES)
        tampered['capture'] = dict(status='complete', bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
        with self.assertRaises(ValueError):
            worker._context_check(tampered, {'side': 0}, module, registry)

    def test_context_consumer_refuses_extra_private_fields_resources_and_fake_stop(self):
        core, registry, *_ = _collector_fixture()
        module = SimpleNamespace(core=core)
        for error in (MemoryError(), ValueError('original')):
            collector, _ = _collect(error)
            value = dict(schema=worker.SCHEMA, **collector.snapshot(), diagnostic_only=True,
                         original_core_completed=False)
            if isinstance(error, ValueError):
                altered = deepcopy(value)
                altered['context']['extra_private_replay'] = []
                with self.assertRaises(ValueError):
                    worker._context_check(altered, {'side': 0}, module, registry)
                value['suffix_sink_completed'] = True
            with self.assertRaises(ValueError):
                worker._context_check(value, {'side': 0}, module, registry)


class ScratchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        value = os.environ.get('AI_GAME_TEST_SCRATCH')
        if not value:
            raise unittest.SkipTest('Root owns all file tests')
        cls.root = Path(value)
        if not cls.root.is_absolute() or not cls.root.is_dir():
            raise ValueError('Invalid externally owned scratch')

    def test_actual_frozen_worker_closes_stream_and_preserves_primary_when_receipt_write_fails(self):
        out = self.root/'first-step-original-exception'
        out.mkdir(exist_ok=False)
        module = worker.previous._frozen_worker()
        source = dict(role='F01', opponent='market', side=0, purpose='exploration', game='synthetic')
        selection = {'target': {'step': 240, 'side': 0}}
        manifest = SimpleNamespace(panel=SimpleNamespace(models=[('task', '1'*64)], crop=('crop', '2'*64)))
        row = dict(replay_gzip_sha256='3'*64, trace_sha256='4'*64, seed=17, initial_banks=[1, 2])
        module._source = lambda *args: (manifest, row, out, out)
        io = ModuleType('synthetic_io_facade')
        io.__dict__.update(vars(module.io))
        io._assets = lambda *args: out/'not-read-rival.py'
        module.io = io
        module._factories = lambda *args: (None, None)
        primary = ValueError('primary retained by actual frozen finally')
        local_core = ModuleType('synthetic_core_facade')
        local_core.__dict__.update(vars(module.core))
        def run(*args, **kwargs):
            raise primary
        local_core.run_pair = run
        module.core = local_core
        collector = capture.Collector(capture.RegisteredSources([]))
        worker._adapt(module, collector)
        module._write_case = lambda *args: (_ for _ in ()).throw(OSError('reserved write failure'))
        class Driver:
            def _no_links(self, *args):
                pass
            def _replay(self, *args):
                return dict(info={'seed': 17}, steps=[[dict(observation=dict(farms=[{'money': 1}, {'money': 2}]))]])
        try:
            module.execute_case({'driver': Driver(), 'first_yield_days': {'WHEAT': 2}},
                                None, source, selection, out, lambda: None)
        except ValueError as caught:
            self.assertIs(caught, primary)
        else:
            self.fail('Actual frozen Worker masked or swallowed primary')
        import gzip
        with gzip.open(out/worker.EVENT_FILE, 'rb') as stream:
            self.assertEqual(stream.read(), b'')
        self.assertFalse((out/worker.CASE_FILE).exists())


if __name__ == '__main__':
    unittest.main()
