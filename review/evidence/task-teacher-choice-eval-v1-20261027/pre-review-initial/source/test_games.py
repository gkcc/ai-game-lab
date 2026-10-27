import importlib.util
import io
import json
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

core=types.ModuleType('learning.task_anchor_once_v1.games')
core.verify,core.task_identity,core.read=object(),object(),lambda _:None
exp=types.ModuleType('learning.task_teacher_choice_eval_v1.experiment')
exp.ROOT=Path(__file__).resolve().parents[2]; exp.verify=lambda *a:None
exp.ready=lambda _:None; exp.verify_commit=lambda _:None; exp.deadline_check=lambda:None
spec=importlib.util.spec_from_file_location('learning.task_teacher_choice_eval_v1._games_test',Path(__file__).with_name('games.py'))
g=importlib.util.module_from_spec(spec)
with patch.dict(sys.modules,{'learning.task_anchor_once_v1.games':core,'learning.task_teacher_choice_eval_v1.experiment':exp}): spec.loader.exec_module(g)


class GamesTest(unittest.TestCase):
    def setUp(self):
        self.cfg=dict(test_seeds=[1,2],opponents=['round2'],seats=[0,1],candidate_checkpoint='new/model.npz',
            candidate_checkpoint_sha256='a'*64,baseline_checkpoint='old/model.npz',baseline_checkpoint_sha256='b'*64,
            engine_sha256='e'*64,crop_checkpoint_sha256='f'*64)
    def test_all_plans_and_identity(self):
        for p in g.plans([1,2]):
            g.validate_plan(p,self.cfg)
            _,sha=g.task_identity(None,p,self.cfg)
            self.assertEqual(sha,('a' if p['policy']=='candidate' else 'b')*64)
        p=g.plans([1,2])[0]
        with self.assertRaises(ValueError): g.validate_plan({**p,'seed':3},self.cfg)
        with self.assertRaises(ValueError): g.task_identity(None,p,{**self.cfg,'candidate_checkpoint':'../bad'})
        with self.assertRaises(ValueError): g.task_identity(None,p,{**self.cfg,'baseline_checkpoint_sha256':'a'*64})

    def test_second_read_changed_and_binding_restored(self):
        plan=g.plans([1,2])[0]
        plan.update(bot_sha256='c'*64,rival_sha256='d'*64,task_checkpoint_sha256='a'*64)
        changed={**plan,'rival_sha256':'e'*64}
        out=exp.ROOT/'synthetic'; path=(out/'job-inputs'/(plan['game']+'.json')).resolve()
        exp.ready=lambda _:self.cfg; original=(core.verify,core.task_identity,core.read)
        core.job=lambda out,key:core.read(path)
        with patch.object(Path,'open',side_effect=[io.StringIO(json.dumps(plan)),io.StringIO(json.dumps(changed))]):
            with self.assertRaises(ValueError): g.job(out,plan['game'])
        self.assertEqual((core.verify,core.task_identity,core.read),original)

    def test_official_receipt_bindings(self):
        plan=g.plans([1,2])[0]; plan.update(bot_sha256='c'*64,rival_sha256='d'*64,task_checkpoint_sha256='a'*64)
        row=dict(plan,passed=True,states=720,action_turns=719,status=['DONE','DONE'],agent_errors=0,
            engine_sha256='e'*64,crop_checkpoint_sha256='f'*64,replay_gzip_sha256='1'*64,trace_sha256='2'*64,
            semantic_replay_sha256='3'*64,semantic_replay_sha256_v2='4'*64,rewards=[1,2])
        cert=dict(game=plan['game'],passed=True,errors=0,actor_actions_reproduced=719,rival_actions_reproduced=719,
            trace_turns_reproduced=719,replay_sha256='1'*64,trace_sha256='2'*64,
            official=dict(official_transitions_reexecuted=719,engine_sha256='e'*64,semantic_sha256='3'*64,
                          semantic_sha256_v2='4'*64,final_rewards=[1,2]))
        complete=dict(game=plan['game'],passed=True,row_sha256='r',certificate_sha256='c')
        g.validate_receipts(plan,row,cert,complete,'r','c',self.cfg)
        with self.assertRaises(ValueError): g.validate_receipts(plan,row,{**cert,'trace_sha256':'bad'},complete,'r','c',self.cfg)


if __name__ == '__main__': unittest.main()
