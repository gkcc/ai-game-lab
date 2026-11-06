"""Pure seam tests plus real cold-import checks in caller-owned scratch only.

The cold subprocess uses production mixed import binding and real preparation
and tracer installation. No checkpoint, private Replay/trace, model invocation,
environment construction, or interpreter call is part of these regressions.
"""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

from . import runtime, worker


def fake_module(events, fail_core=False):
    """Only for seam contract tests, never used as cold-runtime evidence."""
    core = ModuleType('synthetic_frozen_core')
    def run_pair(*args, **kwargs):
        events.append(('run_pair', kwargs))
        if fail_core:
            raise OSError('synthetic core failure')
        return {'passed': True}
    core.run_pair = run_pair
    module = ModuleType('synthetic_case_worker')
    module.core = core
    def factories(*args, **kwargs):
        events.append(('factories', None))
        return object(), object()
    module._factories = factories
    module._write_case = lambda dm, out, document: deepcopy(document)
    return module


class PureTests(unittest.TestCase):
    def setUp(self):
        self.events = []
        self.prepared = runtime.Prepared(object(), lambda env: None, runtime.expected_receipt())

    def prepare(self):
        self.events.append(('prepare', None))
        return self.prepared

    def test_local_facade_prepares_first_injects_explicit_dependencies_and_preserves_shared_core(self):
        module = fake_module(self.events)
        original_core, original_run = module.core, module.core.run_pair
        receipt = worker._adapt(module, has_target=True, guard=lambda: None, prepare=self.prepare)
        with self.assertRaises(ValueError):
            module.core.run_pair('not_prepared')
        module._factories(None, None, None)
        self.assertTrue(module.core.run_pair('fixture')['passed'])
        self.assertEqual([event[0] for event in self.events], ['prepare', 'factories', 'run_pair'])
        self.assertIs(self.events[-1][1]['runtime'], self.prepared.runtime)
        self.assertIs(self.events[-1][1]['target_trace_factory'], self.prepared.target_trace_factory)
        self.assertIs(original_core.run_pair, original_run)
        self.assertIsNot(module.core, original_core)
        document = module._write_case(None, None, {'passed': True})
        self.assertEqual(document[worker.RECEIPT_FIELD], receipt)
        worker._verify_preparation(document, True)

    def test_per_case_state_is_independent_and_cannot_prepare_or_inject_twice(self):
        first, second = fake_module(self.events), fake_module(self.events)
        worker._adapt(first, has_target=True, guard=lambda: None, prepare=self.prepare)
        second_receipt = worker._adapt(second, has_target=True, guard=lambda: None, prepare=self.prepare)
        first._factories(None, None, None)
        first.core.run_pair()
        self.assertEqual(second_receipt['prepare_attempts'], 0)
        for operation in (lambda: first._factories(None, None, None), lambda: first.core.run_pair(),
                          lambda: second.core.run_pair(runtime=object())):
            with self.assertRaises(ValueError):
                operation()

    def test_failed_preparation_or_core_retains_distinct_attempt_return_counts(self):
        module = fake_module(self.events)
        def failed():
            raise OSError('synthetic import failure')
        receipt = worker._adapt(module, has_target=True, guard=lambda: None, prepare=failed)
        with self.assertRaises(OSError):
            module._factories(None, None, None)
        self.assertEqual((receipt['prepare_attempts'], receipt['prepare_returns']), (1, 0))
        self.assertEqual(receipt['status'], 'failed')
        self.assertEqual(self.events, [])
        module = fake_module(self.events, fail_core=True)
        receipt = worker._adapt(module, has_target=True, guard=lambda: None, prepare=self.prepare)
        module._factories(None, None, None)
        with self.assertRaises(OSError):
            module.core.run_pair()
        self.assertEqual((receipt['run_pair_attempts'], receipt['run_pair_returns']), (1, 0))
        self.assertEqual(receipt['status'], 'prepared')

    def test_missing_case_needs_no_preparation_and_receipt_tampering_is_rejected(self):
        module = fake_module(self.events)
        worker._adapt(module, has_target=False, guard=lambda: None, prepare=self.prepare)
        document = module._write_case(None, None, {'passed': True})
        worker._verify_preparation(document, False)
        self.assertEqual(self.events, [])
        document[worker.RECEIPT_FIELD]['prepare_returns'] = True
        with self.assertRaises(ValueError):
            worker._verify_preparation(document, False)
        module = fake_module(self.events)
        worker._adapt(module, has_target=True, guard=lambda: None, prepare=self.prepare)
        module._factories(None, None, None)
        module.core.run_pair()
        document = module._write_case(None, None, {'passed': True})
        document[worker.RECEIPT_FIELD]['prepared']['preparation_official_transitions'] = False
        with self.assertRaises(ValueError):
            worker._verify_preparation(document, True)

    def test_frozen_source_digest_rejects_before_any_import_or_compile(self):
        with self.assertRaisesRegex(ValueError, 'worker SHA'):
            worker._compile_worker(b'raise AssertionError("must not execute")', 'synthetic.py')

    def test_verification_runs_frozen_wire_consumer_before_preparation_receipt_checks(self):
        calls = []
        legacy = ModuleType('synthetic_verifier')
        def verify(*args):
            calls.append('frozen_wire_verified')
            return dict(document={'passed': True})  # Deliberately missing new receipt.
        legacy.verify_case = verify
        with patch.object(worker, '_frozen_worker', return_value=legacy):
            with self.assertRaisesRegex(ValueError, 'runtime receipt'):
                worker.verify_case({}, None, {}, {'target': None}, None, lambda: None)
        self.assertEqual(calls, ['frozen_wire_verified'])

    def test_raw_identity_is_not_replaced_by_newline_normalization(self):
        public = b'first\nsecond\n'
        game = b'first\r\nsecond\r\n'
        identities = {'game': hashlib.sha256(game).hexdigest(), 'public': hashlib.sha256(public).hexdigest()}
        with patch.object(runtime, 'EXECUTION_RAW_SHA256', identities), \
             patch.object(runtime, 'EXECUTION_PUBLIC_SHA256', identities['public']):
            runtime._execution_bytes(game, public, 'game')
            runtime._execution_bytes(public, public, 'public')
            for raw, origin in ((public, 'game'), (game, 'public'), (b'first\r\nsecond\n', 'game')):
                with self.assertRaisesRegex(ValueError, 'raw source identity'):
                    runtime._execution_bytes(raw, public, origin)

    def test_public_source_is_explicitly_test_only_and_cannot_validate_production_receipt(self):
        value = runtime.expected_receipt('public')
        runtime.validate_receipt(value)
        with self.assertRaisesRegex(ValueError, 'Public-only binding'):
            runtime.validate_receipt(value, production=True)
        document = {worker.RECEIPT_FIELD: dict(worker._new_receipt(True), status='prepared',
            prepare_attempts=1, prepare_returns=1, run_pair_attempts=1, run_pair_returns=1, prepared=value)}
        with self.assertRaisesRegex(ValueError, 'Public-only binding'):
            worker._verify_preparation(document, True)


COLD_SCRIPT = r'''
import importlib.util, json, pathlib, sys
from types import SimpleNamespace
repo, out = map(pathlib.Path, sys.argv[1:])
assert sys.flags.isolated and sys.flags.dont_write_bytecode
runner_path = repo/'kaggriculture/learning/task_harvest_unexpressed_runtime_v2/runner.py'
spec = importlib.util.spec_from_file_location('_cold_runtime_v2_public_runner', runner_path)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)
shared = runner._shared()
infra = shared.infrastructure()
reg = infra.decode(infra.read(repo/runner.PRIOR_REG))
bootstrap = infra.snapshot(infra.BOOTSTRAP, reg['public_sources'][infra.BOOTSTRAP], runner.PRIOR_COMMIT)
modules = infra._runtime(bootstrap, reg)
worker = runner._import('task_harvest_unexpressed_runtime_v2.worker')
from learning.task_anchor_once_v1 import storage
from decision import execution
assert pathlib.Path(execution.__file__).resolve().is_relative_to(runner.GAME.resolve())
assert pathlib.Path(worker.__file__).resolve().is_relative_to(repo.resolve())
forbidden = []
def profile(frame, event, arg):
    if event != 'call':
        return
    path, name = frame.f_code.co_filename.replace('\\','/'), frame.f_code.co_name
    blocked = (name == 'make' and path.endswith('/kaggle_environments/core.py')
        or name == 'interpreter' and path.endswith('/envs/kaggriculture/kaggriculture.py')
        or name == '__init__' and any(path.endswith(p) for p in (
            '/learning/task_dagger_once_v1/policy.py', '/learning/task_bc_v1/model.py', '/learning/bc_crop/model.py')))
    if blocked:
        forbidden.append(name)
        raise AssertionError('Unexpected model/environment invocation during preparation')
sys.setprofile(profile)
try:
    prepared = worker._prepare_production_runtime()
finally:
    sys.setprofile(None)
assert not forbidden and storage._ACTIVE is None
assert prepared.receipt['execution_trace_source'] == 'game'
carrier = SimpleNamespace(interpreter=prepared.runtime.interpreter)
import builtins, os
original_import = builtins.__import__
def no_delayed_import(*args, **kwargs):
    raise AssertionError('Prepared target factory still imports inside owned_writes')
outside = out.parent/'must-not-create-outside.bin'
assert not outside.exists()
with storage.owned_writes(out, 256, allowed_paths=['allowed.bin']) as budget:
    builtins.__import__ = no_delayed_import
    try:
        with prepared.target_trace_factory(carrier) as tracer:
            assert tracer.rules.interpreter is prepared.runtime.interpreter
            assert carrier.interpreter is not prepared.runtime.interpreter
    finally:
        builtins.__import__ = original_import
    assert carrier.interpreter is prepared.runtime.interpreter
    (out/'allowed.bin').write_bytes(b'x')
    rejected = []
    for path in (os.devnull, outside):
        try:
            with open(path, 'wb') as stream:
                stream.write(b'forbidden')
        except storage.UnsafeWriteError:
            rejected.append(True)
        else:
            raise AssertionError('Owned writes allowed an unowned path')
assert rejected == [True, True] and not outside.exists() and storage._ACTIVE is None
print('RUNTIME_V2_COLD_RESULT=' + json.dumps(dict(
    isolated=True, bytecode_disabled=True, production_mixed_binding=True,
    real_runtime=True, real_tracer_install_restore=True, no_delayed_import=True,
    devnull_rejected=True, outside_rejected=True, forbidden_calls=forbidden,
    receipt=prepared.receipt), sort_keys=True))
'''


class ScratchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        value = os.environ.get('AI_GAME_TEST_SCRATCH')
        if not value:
            raise unittest.SkipTest('Root must run file/subprocess checks in its owned scratch context')
        cls.scratch = Path(value)
        if not cls.scratch.is_absolute() or not cls.scratch.is_dir():
            raise ValueError('Invalid externally owned test directory')

    def setUp(self):
        self.out = self.scratch/('runtime-v2-'+self._testMethodName)
        self.out.mkdir(exist_ok=False)

    def test_real_isolated_cold_production_initializer_and_tracer_with_write_guard(self):
        result = subprocess.run([sys.executable, '-I', '-B', '-c', COLD_SCRIPT,
            str(worker.REPO), str(self.out)], stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=self.out, timeout=180, check=False)
        self.assertEqual(result.returncode, 0, result.stderr.decode('utf-8', 'replace')[-6000:])
        lines = result.stdout.decode('utf-8', 'replace').splitlines()
        result_lines = [line for line in lines if line.startswith('RUNTIME_V2_COLD_RESULT=')]
        self.assertEqual(len(result_lines), 1)
        value = json.loads(result_lines[0].split('=', 1)[1])
        self.assertEqual(value['receipt'], runtime.expected_receipt())
        self.assertEqual(value['forbidden_calls'], [])
        self.assertTrue(all(value[key] is True for key in ('isolated', 'bytecode_disabled',
            'production_mixed_binding', 'real_runtime', 'real_tracer_install_restore',
            'no_delayed_import', 'devnull_rejected', 'outside_rejected')))

    def test_real_frozen_worker_injects_only_after_assets_replay_and_before_budget_and_closes_failure(self):
        from learning.task_anchor_once_v1 import storage
        module = worker._frozen_worker()
        shared_core, shared_io = module.core, module.io
        original_run, original_assets = shared_core.run_pair, shared_io._assets
        stages = []
        source = dict(role='F01', cohort='C01', game='synthetic', opponent='market', side=0, purpose='exploration')
        selected = dict(target={'side': 0, 'step': 240})
        manifest = SimpleNamespace(panel=SimpleNamespace(models=[('task', '1'*64)], crop=('crop', '2'*64)))
        row = dict(replay_gzip_sha256='3'*64, trace_sha256='4'*64, seed=17, initial_banks=[10, 20])
        root = self.out/'synthetic-source-not-created'
        def source_check(*args):
            stages.append('source')
            return manifest, row, root, root/'stage'
        module._source = source_check
        io = ModuleType('synthetic_case_io_facade')
        io.__dict__.update(vars(shared_io))
        def assets(*args):
            stages.append('assets')
            return root/'rival-not-read'
        io._assets = assets
        module.io = io
        core = ModuleType('synthetic_case_core_facade')
        core.__dict__.update(vars(shared_core))
        def stop_before_policy(*args, **kwargs):
            self.assertIsNotNone(storage._ACTIVE)
            self.assertIs(kwargs['runtime'], prepared[0].runtime)
            self.assertIs(kwargs['target_trace_factory'], prepared[0].target_trace_factory)
            stages.append('protected_core')
            raise OSError('synthetic stop before any policy/environment call')
        core.run_pair = stop_before_policy
        module.core = core
        class Driver:
            def _no_links(self, *args):
                pass
            def _replay(self, *args):
                stages.append('replay')
                return dict(info={'seed': 17}, steps=[[dict(observation=dict(farms=[{'money': 10}, {'money': 20}]))]])
            def _write_new(self, out, path, value, limit):
                raw = json.dumps(value, allow_nan=False).encode()
                if len(raw) > limit:
                    raise ValueError('Synthetic receipt exceeds reserve')
                Path(path).write_bytes(raw)
            def _hash(self, out, path, limit):
                raw = Path(path).read_bytes()
                if len(raw) > limit:
                    raise ValueError('Synthetic receipt exceeds reserve')
                return hashlib.sha256(raw).hexdigest()
        prepared = []
        def prepare():
            self.assertIsNone(storage._ACTIVE)
            stages.append('prepare')
            value = runtime.prepare_runtime()  # Real initializer; no mocks of it or the tracer.
            prepared.append(value)
            return value
        worker._adapt(module, has_target=True, guard=lambda: None, prepare=prepare)
        with self.assertRaisesRegex(OSError, 'synthetic stop'):
            module.execute_case(dict(driver=Driver(), first_yield_days={'WHEAT': 2}),
                None, source, selected, self.out, lambda: None)
        self.assertEqual(stages, ['source', 'assets', 'replay', 'prepare', 'protected_core'])
        self.assertIs(shared_core.run_pair, original_run)
        self.assertIs(shared_io._assets, original_assets)
        self.assertIsNone(storage._ACTIVE)
        document = json.loads((self.out/worker.CASE_FILE).read_bytes())
        self.assertFalse(document['passed'])
        receipt = document[worker.RECEIPT_FIELD]
        self.assertEqual(receipt['prepared']['execution_trace_source'], 'public')
        self.assertEqual((receipt['prepare_attempts'], receipt['prepare_returns'],
                          receipt['run_pair_attempts'], receipt['run_pair_returns']), (1, 1, 1, 0))
        self.assertEqual(document['emitted_events'], 0)
        self.assertEqual(document['meter']['calls'], {})
        self.assertEqual(document['output_budget']['created_count'], 1)
        with gzip_open(self.out/worker.EVENT_FILE) as stream:
            self.assertEqual(stream.read(), b'')  # Gzip was closed despite the core exception.


def gzip_open(path):
    import gzip
    return gzip.open(path, 'rb')


if __name__ == '__main__':
    unittest.main()
