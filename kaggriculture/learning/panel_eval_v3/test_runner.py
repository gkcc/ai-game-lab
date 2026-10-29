import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from . import runner, experiment


class FakeContext:
    ROOT = runner.ROOT

    def __init__(self, stage, commit):
        self.OUT = Path(stage)
        self.prereg_commit = commit

    def ready(self, *_): return {'deadline_utc':'2099-01-01T00:00:00+00:00',
                                  'official_job_supervision':{'limit':1},**experiment.run_contract(experiment.FORMAL)}
    def refresh(self): return self.ready()
    def verify(self, *_): return self.ready()
    def verify_commit(self, *_): return self.ready()
    def deadline_check(self): pass
    def load_seeds(self, *_): return []
    def trusted_out(self, *_): return self.OUT
    def write_new(self, path, value):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('x', encoding='utf-8') as stream: json.dump(value, stream)


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.stage = Path(self.temp.name)/'stage'
        self.stage.mkdir()
        self.scratch = Path(self.temp.name)/'scratch'
        self.scratch.mkdir()

    def test_failed_initial_preflight_does_not_mark_or_start_games(self):
        with patch.object(runner.isolation, 'validate_runtime'), \
             patch.object(runner, 'Context', FakeContext), \
             patch.object(runner, 'EpochContext', side_effect=lambda value:value), \
             patch.object(runner.memory_gate, 'wait_for_preflight', return_value=SimpleNamespace(passed=False)), \
             patch.object(runner.campaign, 'run') as games:
            with self.assertRaises(RuntimeError): runner.run(self.stage, 'a'*40, self.scratch)
            games.assert_not_called()
        self.assertFalse((self.stage/'results').exists())

    def test_existing_attempt_refuses_restart_before_preflight(self):
        (self.stage/'results').mkdir()
        with patch.object(runner.isolation, 'validate_runtime'), \
             patch.object(runner, 'Context', FakeContext), \
             patch.object(runner, 'EpochContext', side_effect=lambda value:value), \
             patch.object(runner.memory_gate, 'wait_for_preflight') as preflight:
            with self.assertRaises(ValueError): runner.run(self.stage, 'a'*40, self.scratch)
            preflight.assert_not_called()

    def test_campaign_failure_consumes_attempt_and_never_evaluates(self):
        with patch.object(runner.isolation, 'validate_runtime'), \
             patch.object(runner, 'Context', FakeContext), \
             patch.object(runner, 'EpochContext', side_effect=lambda value:value), \
             patch.object(runner.memory_gate, 'wait_for_preflight', return_value=SimpleNamespace(passed=True)), \
             patch.object(runner.campaign, 'run', side_effect=RuntimeError('synthetic')), \
             patch.object(runner.independent_runner, 'run') as replay, \
             patch.object(runner.independent_runner, 'finalize') as evaluate:
            with self.assertRaises(RuntimeError): runner.run(self.stage, 'a'*40, self.scratch)
            replay.assert_not_called()
            evaluate.assert_not_called()
            with self.assertRaises(ValueError): runner.run(self.stage, 'a'*40, self.scratch)
        self.assertTrue((self.stage/'results'/'coordinator-started.json').is_file())
        failure = json.loads((self.stage/'results'/'coordinator-failure.json').read_text())
        self.assertFalse(failure['retry_allowed'])
        self.assertEqual(failure['cleanup_state'], 'not_started')

    def test_supervisor_exception_keeps_cleanup_unknown(self):
        controller = runner.Coordinator(FakeContext(self.stage, 'a'*40), self.scratch,
                                        runner.memory_gate.WaitBudget())
        with patch.object(runner, 'supervise_job', side_effect=RuntimeError('synthetic')):
            with self.assertRaises(RuntimeError):
                controller._supervise(None, [], runner.ROOT, {}, self.scratch/'cache-test')
        self.assertFalse(controller.active)
        self.assertEqual(controller.cleanup_state, 'unknown_after_spawn_attempt')

    def test_failed_canonical_child_never_starts_game(self):
        context = FakeContext(self.stage, 'a'*40)
        controller = runner.Coordinator(context, self.scratch, runner.memory_gate.WaitBudget())
        key = 'holdout-S01-round2-0-merged'
        wrapper = self.stage/'agents'/(key+'.py')
        wrapper.parent.mkdir()
        wrapper.write_bytes(b'synthetic wrapper')
        item = {'game':key, 'bot_sha256':runner._sha(wrapper)}
        input_path = self.stage/'job-inputs'/(key+'.json')
        input_path.parent.mkdir()
        input_path.write_text(json.dumps(item), encoding='utf-8')
        command = [runner.sys.executable, '-B', '-m',
                   'learning.task_teacher_merged_eval_v1.games', '--out', str(self.stage), '--key', key]
        with patch.object(controller, '_command', return_value=(self.scratch/'cache', [])), \
             patch.object(controller, '_supervise', return_value={'passed':False}) as supervise:
            with self.assertRaises(RuntimeError):
                controller.supervise_game(None, command, runner.ROOT, {'limit':1})
            self.assertEqual(supervise.call_count, 1)
        self.assertFalse((self.stage/'results'/'job-start').exists())

    def canonical_fixture(self):
        key = 'holdout-S01-round2-0-merged'
        wrapper = self.stage/'agents'/(key+'.py')
        wrapper.parent.mkdir()
        wrapper.write_bytes(b'synthetic wrapper')
        sha = runner._sha(wrapper)
        item = {'game':key, 'bot_sha256':sha}
        source = self.stage/'job-inputs'/(key+'.json')
        source.parent.mkdir()
        source.write_text(json.dumps(item), encoding='utf-8')
        documents = {
            'canonical': dict(game=key, wrapper_sha256=sha, passed=True, launcher_reaped=True,
                              job_members_zero=True, isolated_cache_prefix_absent=True, retry_allowed=False),
            'job-start': dict(game=key, wrapper_sha256=sha, retry_allowed=False),
            'canonical-post': dict(game=key, wrapper_sha256=sha, passed=True, retry_allowed=False),
        }
        for name, document in documents.items():
            path = self.stage/'results'/name/(key+'.json')
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps(document), encoding='utf-8')
        return key, item

    def test_acceptance_binds_all_three_persistent_receipts(self):
        key, item = self.canonical_fixture()
        with patch.object(runner, '_original_accept_job', return_value=({'game':key}, {})):
            row, audit = runner.accept_persistent_job(self.stage, item, {})
            self.assertEqual(row['game'], key)
            self.assertEqual(set(audit['canonical_receipt_sha256']), {'canonical','job-start','canonical-post'})
            for name in ('canonical','job-start','canonical-post'):
                path = self.stage/'results'/name/(key+'.json')
                old = path.read_bytes()
                path.unlink()
                with self.assertRaises((ValueError, OSError)):
                    runner.accept_persistent_job(self.stage, item, {})
                path.write_bytes(old)
                document = json.loads(old)
                document['wrapper_sha256'] = 'a'*64
                path.write_text(json.dumps(document), encoding='utf-8')
                with self.assertRaises(ValueError):
                    runner.accept_persistent_job(self.stage, item, {})
                path.write_bytes(old)

            path = self.stage/'results'/'canonical'/(key+'.json')
            document = json.loads(path.read_bytes())
            document['passed'] = 1
            document['retry_allowed'] = 0
            path.write_text(json.dumps(document), encoding='utf-8')
            with self.assertRaises(ValueError):
                runner.accept_persistent_job(self.stage, item, {})

    def test_receipt_change_during_acceptance_fails_closed(self):
        key, item = self.canonical_fixture()
        def tamper(*_):
            (self.stage/'results'/'canonical-post'/(key+'.json')).unlink()
            return {}, {}
        with patch.object(runner, '_original_accept_job', side_effect=tamper):
            with self.assertRaises((ValueError, OSError)):
                runner.accept_persistent_job(self.stage, item, {})

    def test_replay_worker_uses_persistent_receipt_checks(self):
        from . import independent_worker
        key, item = self.canonical_fixture()
        args = ['worker', '--out', str(self.stage), '--prereg', 'a'*40, '--scratch', str(self.scratch)]
        def replay(_):
            return runner.campaign.accept_job(self.stage, item, {})
        with patch.object(runner.isolation, 'validate_runtime'), \
             patch('learning.panel_eval_v3.experiment.Context', FakeContext), \
             patch('learning.panel_eval_v3.epoch.EpochContext', side_effect=lambda value:value), \
             patch.object(runner, '_original_accept_job', return_value=({}, {})), \
             patch.object(runner.independent_runner, 'run'), \
             patch('learning.task_teacher_merged_eval_v1.independent_method.run', side_effect=replay) as method, \
             patch.object(runner.sys, 'argv', args):
            independent_worker.main()
            self.assertEqual(method.call_count, 1)
            (self.stage/'results'/'canonical'/(key+'.json')).unlink()
            with self.assertRaises((ValueError, OSError)):
                independent_worker.main()

    def test_memory_drop_after_canonical_never_starts_game_if_wait_fails(self):
        context = FakeContext(self.stage, 'a'*40)
        controller = runner.Coordinator(context, self.scratch, runner.memory_gate.WaitBudget())
        key = 'holdout-S01-round2-0-merged'
        wrapper = self.stage/'agents'/(key+'.py')
        wrapper.parent.mkdir()
        wrapper.write_bytes(b'synthetic wrapper')
        item = {'game':key, 'bot_sha256':runner._sha(wrapper)}
        input_path = self.stage/'job-inputs'/(key+'.json')
        input_path.parent.mkdir()
        input_path.write_text(json.dumps(item), encoding='utf-8')
        command = [runner.sys.executable, '-B', '-m',
                   'learning.task_teacher_merged_eval_v1.games', '--out', str(self.stage), '--key', key]
        ok = dict(passed=True, assigned=True, resumed=True, launcher_reaped=True, job_members_zero=True,
                  error_kind=None, cleanup_error_kind=None, stop_reason=None, exit_code=0)
        low = runner.memory_gate.MIN_AVAILABLE_BYTES - 1
        with patch.object(controller, '_command', return_value=(self.scratch/'cache', [])), \
             patch.object(controller, '_supervise', return_value=ok) as supervise, \
             patch.object(controller, '_available', return_value=low), \
             patch.object(controller, 'memory_ready', return_value=False) as wait:
            with self.assertRaises(RuntimeError):
                controller.supervise_game(None, command, runner.ROOT, {'limit':1})
            self.assertEqual(supervise.call_count, 1)
            self.assertEqual(wait.call_args.kwargs['first_sample'], low)
        self.assertFalse((self.stage/'results'/'job-start').exists())

    def test_game_boundary_rescan_failure_prevents_launch_or_acceptance(self):
        for failed_refresh in (2, 3):
            with self.subTest(failed_refresh=failed_refresh), tempfile.TemporaryDirectory() as temp:
                stage = Path(temp)/'stage'
                stage.mkdir()
                class ChangingContext(FakeContext):
                    calls = 0
                    def refresh(self):
                        self.calls += 1
                        if self.calls == failed_refresh: raise ValueError('synthetic changed history')
                        return self.ready()
                context = ChangingContext(stage, 'a'*40)
                controller = runner.Coordinator(context, self.scratch, runner.memory_gate.WaitBudget())
                key = 'holdout-S01-round2-0-merged'
                wrapper = stage/'agents'/(key+'.py')
                wrapper.parent.mkdir()
                wrapper.write_bytes(b'synthetic wrapper')
                item = {'game':key, 'bot_sha256':runner._sha(wrapper)}
                input_path = stage/'job-inputs'/(key+'.json')
                input_path.parent.mkdir()
                input_path.write_text(json.dumps(item), encoding='utf-8')
                command = [runner.sys.executable, '-B', '-m',
                           'learning.task_teacher_merged_eval_v1.games', '--out', str(stage), '--key', key]
                ok = dict(passed=True, assigned=True, resumed=True, launcher_reaped=True, job_members_zero=True,
                          error_kind=None, cleanup_error_kind=None, stop_reason=None, exit_code=0)
                with patch.object(controller, '_command', return_value=(self.scratch/'cache', [])), \
                     patch.object(controller, '_supervise', return_value=ok) as supervise, \
                     patch.object(controller, '_available', return_value=runner.memory_gate.MIN_AVAILABLE_BYTES):
                    with self.assertRaises(ValueError):
                        controller.supervise_game(None, command, runner.ROOT, {'limit':1})
                    self.assertEqual(supervise.call_count, failed_refresh-1)
                self.assertFalse((stage/'results'/'canonical-post'/(key+'.json')).exists())


if __name__ == '__main__':
    unittest.main()
