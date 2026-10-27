"""Actual fixture control-flow checks with synthetic process boundaries only.

The ownership utility creates and removes real owned temporary directories.
No subprocess, thread, process snapshot, signal, or real process reap is used.
"""
from contextlib import ExitStack, contextmanager
import importlib.util
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from . import experiment as exp
from . import runner
from .evidence import Recorder, PHASES


def _helper():
    path = os.environ.get('TASK_LAUNCHER_ARTIFACT_HELPER')
    if not path:
        raise RuntimeError('Tests require the declared ownership helper environment variable')
    spec = importlib.util.spec_from_file_location('launcher_fixture_test_artifacts', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _Thread:
    def join(self, timeout=None):
        return None

    def is_alive(self):
        return False


class _Images:
    bytes_read = 0

    def identity(self, path):
        return {'basename': 'python.exe', 'sha256': 'a' * 64}


class FixtureRetentionTests(unittest.TestCase):
    @contextmanager
    def scenario(self, *, launch_failure=False, nonce_failure=False,
                 final_detail_failure=False, reap_failure=False):
        helper = _helper()
        # The utility validates real ownership. Its OS identity functions are
        # never passed a synthetic child: only the fixture helper proxy is fake.
        parent_root = Path(tempfile.gettempdir()) / 'codex-launcher-fixture-tests'
        with helper.scratch_directory('launcher-fixture-test', root=parent_root) as workspace:
            out = workspace / 'out'
            scratch_paths = []
            synthetic_sys = SimpleNamespace(executable='synthetic-python', _base_executable='synthetic-python',
                prefix='synthetic-prefix', base_prefix='synthetic-prefix', path=['synthetic-module-path'])

            @contextmanager
            def scratch_directory(purpose, root=None):
                with helper.scratch_directory(purpose, root=root) as path:
                    scratch_paths.append(path)
                    yield path

            fixture_helper = SimpleNamespace(scratch_directory=scratch_directory,
                protect_children=lambda *args, **kwargs: None,
                process_identity=lambda pid: ('active', 'windows:123'))

            class Process:
                pid = 303
                returncode = None

                def __init__(self, scratch):
                    self.scratch = scratch

                def poll(self):
                    if (self.scratch / 'go.json').exists():
                        self.returncode = 0
                    return self.returncode

                def wait(self, timeout=None):
                    self.returncode = 0
                    return 0

            class Capture:
                def __init__(self, log, limit_bytes):
                    self.log = Path(log)
                    self.limit_bytes = limit_bytes
                    self.process = None
                    self.thread = _Thread()
                    self.error = None
                    self.seen = 0
                    self.written = 0

                def start(self, command, cwd):
                    if launch_failure:
                        raise RuntimeError('synthetic launch failure')
                    self.command = list(command)
                    self.process = Process(self.log.parent)
                    self.log.write_bytes(runner.NORMAL)
                    self.seen = self.written = len(runner.NORMAL)
                    nonce = command[command.index('--nonce') + 1]
                    mode = command[command.index('--mode') + 1]
                    handshake = dict(pid=303, parent_pid=101, nonce='wrong-synthetic-nonce' if nonce_failure else nonce,
                        mode=mode, process_guard_probe_blocked=True, **runner.expected_runtime())
                    (self.log.parent / 'handshake.json').write_text(json.dumps(handshake), encoding='utf-8')
                    return self

                def check(self):
                    if self.error is not None:
                        raise self.error

                def finish(self):
                    self.check()
                    return dict(seen_bytes=self.seen, stored_bytes=self.written, reader_joined=True)

            def snapshot(process, images, known):
                node = dict(pid=303, parent_pid=101, created=123.0, cmdline=list(active_capture[0].command),
                            image=images.identity('synthetic-python'), working_set=1024, cpu=0.001)
                known[(303, 123.0)] = node
                return [node]

            active_capture = []

            def create_capture(*args, **kwargs):
                instance = Capture(*args, **kwargs)
                active_capture.append(instance)
                return instance

            def reap(captured, known):
                if reap_failure:
                    raise RuntimeError('synthetic reap failure')
                if captured.process is not None:
                    captured.process.returncode = 0
                return dict(owned_instances_remaining=0, reader_joined=True, owned_instances_seen=len(known))

            original_private = Recorder.private

            def private(recorder, name, value):
                if final_detail_failure and name == 'final_detail':
                    raise OSError('synthetic private final-detail write failure')
                return original_private(recorder, name, value)

            synthetic_psutil = SimpleNamespace(Process=lambda *args, **kwargs:
                SimpleNamespace(memory_info=lambda: SimpleNamespace(rss=2048)),
                NoSuchProcess=type('SyntheticNoSuchProcess', (Exception,), {}))
            with ExitStack() as stack:
                for target, name, value in (
                    (exp, 'OUT', out), (runner, 'OUT', out), (runner, 'deadline_check', lambda: None),
                    (runner, 'Capture', create_capture), (runner, 'snapshot', snapshot), (runner, 'reap', reap),
                    (runner, 'ownership_helper', lambda: fixture_helper), (runner, 'psutil', synthetic_psutil),
                    (runner, 'os', SimpleNamespace(getpid=lambda: 101, environ=os.environ)),
                    (runner, 'sys', synthetic_sys),
                    (runner, 'tempfile', SimpleNamespace(gettempdir=lambda: str(workspace))),
                    (Recorder, 'private', private)):
                    stack.enter_context(patch.object(target, name, value))
                stack.enter_context(patch.dict(os.environ, {'TASK_LAUNCHER_TEMP_BYTES': '0'}))
                yield SimpleNamespace(out=out, scratch_paths=scratch_paths, images=_Images())
        self.assertFalse(workspace.exists(), 'Owned test workspace must actually be removed')

    def receipt(self, scenario, name):
        public = scenario.out / (name + '-evidence.json')
        private = scenario.out / 'private' / (name + '-events.jsonl')
        self.assertTrue(public.is_file(), 'Every fixture must preserve its public receipt')
        self.assertTrue(private.is_file(), 'Every fixture must preserve private diagnostic evidence')
        self.assertGreater(private.stat().st_size, 0)
        value = json.loads(public.read_text(encoding='utf-8'))
        self.assertNotIn('synthetic launch failure', public.read_text(encoding='utf-8'))
        return value

    def test_launch_failure_retains_receipts_and_removes_scratch(self):
        with self.scenario(launch_failure=True) as case:
            result = runner.fixture('unit-launch', 'normal', case.images)
            evidence = self.receipt(case, 'unit-launch')
            self.assertTrue(result['primary_failed'])
            self.assertEqual(evidence['primary_failure']['phase'], 'launch')
            self.assertTrue(result['scratch_cleaned'])
            self.assertTrue(case.scratch_paths)
            self.assertTrue(all(not p.exists() for p in case.scratch_paths))

    def test_private_final_detail_failure_cannot_skip_scratch_cleanup(self):
        with self.scenario(launch_failure=True, final_detail_failure=True) as case:
            result = runner.fixture('unit-private', 'normal', case.images)
            evidence = self.receipt(case, 'unit-private')
            self.assertEqual(evidence['primary_failure']['phase'], 'launch')
            self.assertIsNotNone(evidence['cleanup_failure'])
            self.assertTrue(result['cleanup_failed'])
            self.assertTrue(result['scratch_cleaned'])
            self.assertTrue(all(not p.exists() for p in case.scratch_paths))

    def test_reap_failure_cannot_replace_primary_failure(self):
        with self.scenario(nonce_failure=True, reap_failure=True) as case:
            result = runner.fixture('unit-reap', 'normal', case.images)
            evidence = self.receipt(case, 'unit-reap')
            self.assertTrue(result['primary_failed'])
            self.assertTrue(result['cleanup_failed'])
            self.assertEqual(evidence['primary_failure']['phase'], 'schema_nonce')
            self.assertEqual(evidence['cleanup_failure']['phase'], 'cleanup_reap')

    def test_normal_requires_complete_evidence_and_no_remaining_processes(self):
        with self.scenario() as case:
            result = runner.fixture('unit-normal', 'normal', case.images)
            evidence = self.receipt(case, 'unit-normal')
            self.assertEqual(result['outcome'], 'normal')
            self.assertFalse(result['primary_failed'])
            self.assertFalse(result['cleanup_failed'])
            self.assertFalse(evidence['evidence_failure'])
            self.assertIsNone(evidence['primary_failure'])
            self.assertIsNone(evidence['cleanup_failure'])
            self.assertEqual(result['cleanup']['owned_instances_remaining'], 0)
            self.assertTrue(any(event.get('safe', {}).get('checks', {}).get('reader_joined') is True
                                for event in evidence['events']))
            self.assertTrue(result['scratch_cleaned'])
            self.assertTrue(all(not p.exists() for p in case.scratch_paths))

    def test_every_phase_durably_preserves_partial_and_private_failure(self):
        with self.scenario() as case:
            for index, phase in enumerate(PHASES):
                with self.subTest(phase=phase):
                    name = 'unit-phase-' + str(index)
                    recorder = Recorder(case.out, name)
                    recorder.checkpoint('launch', {'checks': {'launched': True}, 'roles': ['unresolved']})
                    recorder.checkpoint(phase)
                    before_failure = (case.out / (name + '-events.jsonl')).read_text(encoding='utf-8')
                    self.assertEqual(json.loads(before_failure.splitlines()[-1])['phase'], phase)
                    try:
                        raise RuntimeError('synthetic-secret-phase-' + str(index))
                    except RuntimeError as exc:
                        recorder.fail(exc)
                    recorder.persist()
                    public_text = (case.out / (name + '-evidence.json')).read_text(encoding='utf-8')
                    public = json.loads(public_text)
                    private_text = (case.out / 'private' / (name + '-events.jsonl')).read_text(encoding='utf-8')
                    private = [json.loads(line) for line in private_text.splitlines()]
                    self.assertEqual(public['primary_failure']['phase'], phase)
                    self.assertEqual(public['primary_failure']['reason'], phase + '_failed')
                    self.assertTrue(public['events'][0]['safe']['checks']['launched'])
                    self.assertIn('synthetic-secret-phase-' + str(index), private_text)
                    self.assertIn('Traceback (most recent call last)', private[-1]['value']['traceback'])
                    self.assertNotIn('synthetic-secret', public_text)
                    self.assertNotIn('synthetic-secret', (case.out / (name + '-events.jsonl')).read_text(encoding='utf-8'))
                    self.assertFalse(public['evidence_failure'])


if __name__ == '__main__':
    unittest.main()
