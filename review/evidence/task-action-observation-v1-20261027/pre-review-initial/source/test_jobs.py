"""Synthetic adapter/attempt regressions; no models, OS children or games."""
from contextlib import contextmanager
from pathlib import Path
import types
import unittest
from unittest.mock import Mock, patch

from . import jobs, campaign


class JobTests(unittest.TestCase):
    def test_existing_model_identity_and_scoped_binding_restore(self):
        cfg = dict(task_checkpoint='existing/student.npz', task_checkpoint_sha256='student',
                   candidate_checkpoint='existing/candidate.npz', candidate_checkpoint_sha256='candidate')
        for arm, expected in (('baseline', 'task'), ('candidate', 'candidate')):
            path, value = jobs.task_identity(Path('unused'), {'policy': arm}, cfg)
            self.assertEqual(path, jobs.experiment.ROOT/cfg[expected+'_checkpoint'])
            self.assertEqual(value, cfg[expected+'_checkpoint_sha256'])
        self.assertEqual(jobs.task_identity(None, {'policy': 'teacher_task'}, cfg), (None, None))
        previous = jobs.core.verify, jobs.core.task_identity
        def failing_wrapper(*args):
            self.assertIs(jobs.core.verify, jobs.experiment.verify)
            self.assertIs(jobs.core.task_identity, jobs.task_identity)
            raise RuntimeError('synthetic wrapper refusal')
        with patch.object(jobs.core, 'wrapper', failing_wrapper):
            with self.assertRaisesRegex(RuntimeError, 'synthetic wrapper refusal'):
                jobs.make_wrapper(None, None, None)
        self.assertEqual((jobs.core.verify, jobs.core.task_identity), previous)

    def test_readiness_refusal_prevents_official_entry(self):
        with patch.object(jobs.experiment, 'ready', side_effect=ValueError('not frozen')), \
                patch.object(jobs.core, 'job') as launch:
            with self.assertRaisesRegex(ValueError, 'not frozen'):
                jobs.job(Path('.'), 'unused')
            launch.assert_not_called()
        records = {'fixture-started.json': {'fixture': True, 'freeze_sha256': 'frozen'},
                   'fixture-guard-result.json': {'fixture': True, 'completed': False},
                   'fixture-result.json': {}, 'analysis-target-result.json': {},
                   'analysis-worker-result.json': {}}
        with patch.object(campaign, 'read', side_effect=lambda path: records[path.name]), \
                patch.object(campaign, 'sha', return_value='frozen'):
            with self.assertRaises(AssertionError):
                campaign.fixture_receipts(Path('synthetic-unwritten'))

    def test_fixed_24_attempts_and_failure_reaps_all_without_retry(self):
        cfg = dict(validation_seeds=[10001, 10002], test_seeds=[], max_games=24, max_workers=2,
                   opponents=['round2', 'market'], arms=list(campaign.ARMS), seats=[0, 1],
                   per_job_timeout_seconds=900, storage={'process_log_bytes': 1024})
        plan = campaign.plans(cfg)
        self.assertEqual(len(plan), 24)
        self.assertEqual(len({j['game'] for j in plan}), 24)
        self.assertEqual([j['policy'] for j in plan[:3]], list(campaign.ARMS))
        for seed in cfg['validation_seeds']:
            for opponent in campaign.OPPONENTS:
                for side in (0, 1):
                    self.assertEqual({j['policy'] for j in plan if
                        (j['seed'], j['opponent'], j['side']) == (seed, opponent, side)}, set(campaign.ARMS))
        @contextmanager
        def scope(paths):
            yield
        parent = types.SimpleNamespace(scope=scope, write=Mock())
        first = types.SimpleNamespace(process=types.SimpleNamespace(poll=lambda: 1),
                                      check=Mock(), finish=Mock(return_value={}))
        second = types.SimpleNamespace(process=types.SimpleNamespace(poll=lambda: None),
                                       check=Mock())
        spawned, reaped, records = [], [], []
        def capture(*args, **kwargs):
            child = (first, second)[len(spawned)]
            spawned.append(child)
            return child
        def cleanup(child, known):
            reaped.append(child)
            return dict(owned_instances_remaining=0, reader_joined=True, launched_process_reaped=True)
        runner = campaign.Runner(campaign.ROOT/'synthetic-unwritten', cfg, parent, plan)
        with patch.object(runner, 'limits'), patch.object(runner, 'storage_ready'), \
                patch.object(campaign, 'verify', return_value=cfg), \
                patch.object(campaign, 'make_wrapper', return_value=Path('fake.py')), \
                patch.object(campaign, 'task_identity', return_value=(Path('existing.npz'), 'digest')), \
                patch.object(campaign, 'sha', return_value='digest'), \
                patch.object(campaign, 'write_new', side_effect=lambda path, value: records.append(value)), \
                patch.object(campaign, 'CapturedProcess', side_effect=capture), \
                patch.object(campaign, 'observe_processes'), \
                patch.object(campaign, 'reap', side_effect=cleanup), \
                patch.object(campaign.time, 'sleep'):
            with self.assertRaisesRegex(AssertionError, 'no retry'):
                runner.execute()
        self.assertEqual(len(spawned), 2)
        self.assertEqual(len(runner.attempts), 2)
        self.assertEqual(records, runner.attempts)
        self.assertTrue(all(any(item is child for item in reaped) for child in spawned))
        self.assertEqual(runner.rows, [])


if __name__ == '__main__':
    unittest.main()
