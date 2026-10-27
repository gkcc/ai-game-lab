import importlib.util
import io
import json
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

core = types.ModuleType('learning.task_anchor_once_v1.games')
core.verify, core.task_identity, core.read = object(), object(), lambda _: None
core.reexecute = lambda *args, **kwargs: {}
experiment = types.ModuleType('learning.task_completion_smoke_v1.experiment')
experiment.ROOT = Path(__file__).resolve().parents[2]
experiment.verify = lambda *a: None
experiment.deadline_check = lambda: None
spec = importlib.util.spec_from_file_location('learning.task_completion_smoke_v1._games_test', Path(__file__).with_name('games.py'))
games = importlib.util.module_from_spec(spec)
with patch.dict(sys.modules, {'learning.task_anchor_once_v1.games':core,
                            'learning.task_completion_smoke_v1.experiment':experiment}):
    spec.loader.exec_module(games)


class GamesTest(unittest.TestCase):
    def setUp(self):
        self.cfg = dict(test_seeds=[7], opponents=['round2'], seats=[0,1],
                        candidate_checkpoint='artifacts/new/private/model-final.npz', candidate_checkpoint_sha256='a'*64)
    def plan(self, side=0, policy='candidate'):
        return dict(game=f'holdout-7-round2-{side}-{policy}', policy=policy,
                    split='holdout', seed=7, opponent='round2', side=side, parity_kind=None)
    def test_panel_and_identity(self):
        for side in (0,1):
            for policy in ('candidate','teacher_task'):
                plan = self.plan(side,policy)
                games.validate_plan(plan,self.cfg)
                path, digest = games.task_identity(None,plan,self.cfg)
                self.assertEqual(digest, 'a'*64 if policy == 'candidate' else None)
                if path is not None: self.assertTrue(path.is_relative_to(experiment.ROOT))
                job = dict(plan, bot_sha256='b'*64, rival_sha256='c'*64, task_checkpoint_sha256=digest)
                games.validate_plan(job,self.cfg,job=True)
                job['task_checkpoint_sha256'] = 'd'*64
                with self.assertRaises(ValueError): games.validate_plan(job,self.cfg,job=True)
        with self.assertRaises(ValueError): games.validate_plan(self.plan(policy='baseline'),self.cfg)
        for rel in ('../outside', '/absolute', 'a/../b', 'a\\b', 'C:' + '/model'):
            with self.subTest(rel=rel), self.assertRaises(ValueError):
                games.task_identity(None,self.plan(),{**self.cfg,'candidate_checkpoint':rel})
        with self.assertRaises(ValueError): games.validate_plan(self.plan(),{**self.cfg,'test_seeds':[7,8]})

    def test_changed_second_read_and_restoration(self):
        plan = dict(self.plan(),bot_sha256='b'*64,rival_sha256='c'*64,task_checkpoint_sha256='a'*64)
        changed = dict(plan, rival_sha256='d'*64)
        out = experiment.ROOT/'synthetic'
        path = (out/'job-inputs'/(plan['game']+'.json')).resolve()
        experiment.ready = lambda _: self.cfg
        original = core.verify, core.task_identity, core.read, core.reexecute
        core.job = lambda out,key: core.read(path)
        with patch.object(Path,'open',side_effect=[io.StringIO(json.dumps(plan)),io.StringIO(json.dumps(changed))]):
            with self.assertRaises(ValueError): games.job(out,plan['game'])
        self.assertEqual((core.verify,core.task_identity,core.read,core.reexecute),original)

    def test_official_events_are_counted_without_changing_the_sink(self):
        forwarded = []
        def replay(*args, label_sink=None, **kwargs):
            for player, status in ((0,'executed'),(0,'blocked'),(1,'executed')):
                label_sink(dict(player=player,phase='unit',requested=['FERTILIZE'],status=status))
            return dict(passed=True)
        result = games.counted_reexecute(replay,0,{},trace=True,label_sink=forwarded.append)
        self.assertEqual(len(forwarded),3)
        self.assertEqual(result['fertilizer_actor_request_status'],dict(requested=2,
            executed_with_observed_change=1,blocked=1,no_effect=0,error=0,other=0))


if __name__ == '__main__': unittest.main()
