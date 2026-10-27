import json
from unittest import TestCase, main
from unittest.mock import mock_open, patch
import sys, types
from unittest.mock import Mock
fake = types.ModuleType('learning.task_anchor_once_v1.games')
fake.verify = Mock()
fake.task_identity = Mock()
fake.wrapper = Mock()
fake.job = Mock()
fake.read = Mock()
fake_experiment = types.ModuleType('learning.task_completion_bc_reentry_v1.experiment')
fake_experiment.verify = Mock()
fake_experiment.ready = Mock()
fake_experiment.deadline_check = Mock()
import importlib.util
from pathlib import Path
spec = importlib.util.spec_from_file_location('learning.task_completion_bc_reentry_v1._test_games_adapter', Path(__file__).with_name('games.py'))
games = importlib.util.module_from_spec(spec)
with patch.dict(sys.modules, {'learning.task_anchor_once_v1.games': fake,
                            'learning.task_completion_bc_reentry_v1.experiment': fake_experiment}):
    spec.loader.exec_module(games)

CFG = {'test_seeds': [101], 'opponents': ['round2'], 'seats': [0, 1]}
PLAN = {'game': 'holdout-101-round2-0-teacher_task', 'policy': 'teacher_task', 'split': 'holdout', 'seed': 101, 'opponent': 'round2', 'side': 0, 'parity_kind': None}
JOB = dict(PLAN, bot_sha256='a'*64, rival_sha256='b'*64, task_checkpoint_sha256=None)

class AdapterTests(TestCase):
    def test_scope(self):
        games.validate_plan(PLAN, CFG)
        for field, value in [('policy', 'candidate'), ('split', 'dev'), ('seed', 102), ('opponent', 'other'), ('side', 1), ('game', 'other'), ('parity_kind', 'student'), ('unknown', 1)]:
            with self.subTest(field=field), self.assertRaises(ValueError):
                games.validate_plan(dict(PLAN, **{field: value}), CFG)
        self.assertEqual(games.task_identity(None, PLAN, CFG), (None, None))
        with self.assertRaises(ValueError):
            json.loads('{"seed": 1, "seed": 2}', object_pairs_hook=games._unique_pairs)

    def test_bindings_restore_on_failure(self):
        previous = games.core.verify, games.core.task_identity
        def fail(*args):
            self.assertIs(games.core.verify, games.experiment.verify)
            self.assertIs(games.core.task_identity, games.task_identity)
            raise RuntimeError('synthetic')
        with patch.object(games.core, 'wrapper', side_effect=fail):
            with self.assertRaises(RuntimeError): games.make_wrapper(None, PLAN, CFG)
        self.assertEqual((games.core.verify, games.core.task_identity), previous)
        with patch.object(games.experiment, 'ready', return_value=CFG), patch.object(games.experiment, 'deadline_check'), patch('pathlib.Path.open', mock_open(read_data=json.dumps(JOB))), patch.object(games.core, 'job', side_effect=fail):
            with self.assertRaises(RuntimeError): games.job('.', PLAN['game'])
        self.assertEqual((games.core.verify, games.core.task_identity), previous)

    def test_invalid_job_never_calls_core(self):
        with patch.object(games.experiment, 'ready', return_value=CFG), patch.object(games.experiment, 'deadline_check'), patch('pathlib.Path.open', mock_open(read_data=json.dumps(dict(JOB, policy='baseline')))), patch.object(games.core, 'job') as call:
            with self.assertRaises(ValueError): games.job('.', PLAN['game'])
            call.assert_not_called()

    def test_changed_second_read_is_rejected(self):
        changed = dict(JOB, bot_sha256='c'*64)
        previous = games.core.read
        def core_job(out, key):
            return games.core.read(out/'job-inputs'/(key+'.json'))
        with patch.object(games.experiment, 'ready', return_value=CFG), patch.object(games.experiment, 'deadline_check'), patch('pathlib.Path.open', mock_open(read_data=json.dumps(JOB))) as opened, patch.object(games.core, 'job', side_effect=core_job):
            opened.side_effect = [mock_open(read_data=json.dumps(JOB)).return_value, mock_open(read_data=json.dumps(changed)).return_value]
            with self.assertRaisesRegex(ValueError, 'changed'):
                games.job('.', PLAN['game'])
        self.assertIs(games.core.read, previous)

if __name__ == '__main__': main()
