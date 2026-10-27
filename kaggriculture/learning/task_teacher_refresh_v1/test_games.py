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
exp=types.ModuleType('learning.task_teacher_refresh_v1.experiment'); exp.ROOT=Path(__file__).resolve().parents[2]
exp.verify=lambda _:{}; exp.ready=lambda _:None; exp.verify_commit=lambda _:None; exp.deadline_check=lambda:None
spec=importlib.util.spec_from_file_location('learning.task_teacher_refresh_v1._games_test',Path(__file__).with_name('games.py'))
g=importlib.util.module_from_spec(spec)
with patch.dict(sys.modules,{'learning.task_anchor_once_v1.games':core,'learning.task_teacher_refresh_v1.experiment':exp}): spec.loader.exec_module(g)


class GamesTest(unittest.TestCase):
    def setUp(self):
        self.cfg=dict(train_seeds=[1,2],reserved_test_seeds=[3,4],opponents=['round2'],seats=[0,1],
            source_sha256={'round2.py':'d'*64},engine_sha256='e'*64,crop_checkpoint_sha256='f'*64,
            game_limits=dict(controller_calls=719,crop_forwards=8000,neural_input_rows=160000,gate_evaluations=1,teacher_proposals=1))
    def test_panel_and_invalid_roles(self):
        panel=g.plans(self.cfg); self.assertEqual(len(panel),4)
        for plan in panel: g.validate_plan(plan,self.cfg); self.assertEqual(g.task_identity(None,plan,self.cfg),(None,None))
        for change in (dict(seed=3),dict(side=True),dict(opponent='market'),dict(policy='candidate'),dict(split='holdout')):
            with self.subTest(change=change),self.assertRaises(ValueError): g.validate_plan({**panel[0],**change},self.cfg)
        with self.assertRaises(ValueError): g.plans({**self.cfg,'train_seeds':[1,3]})
        for reserved in ([3,3],[3,True],[3], [2,4]):
            with self.assertRaises(ValueError): g.plans({**self.cfg,'reserved_test_seeds':reserved})

    def test_job_reread_and_binding_restoration(self):
        plan=g.plans(self.cfg)[0]; plan.update(bot_sha256='c'*64,rival_sha256='d'*64,task_checkpoint_sha256=None)
        exp.ready=lambda _:self.cfg; exp.verify=lambda _:self.cfg
        out=exp.ROOT/'synthetic'; path=(out/'job-inputs'/(plan['game']+'.json')).resolve()
        original=core.verify,core.task_identity,core.read
        def fake_job(out,key):
            self.assertEqual(core.verify(out)['validation_seeds'],[1,2])
            return core.read(path)
        core.job=fake_job
        with patch.object(Path,'open',side_effect=[io.StringIO(json.dumps(plan)),io.StringIO(json.dumps({**plan,'bot_sha256':'a'*64}))]):
            with self.assertRaises(ValueError): g.job(out,plan['game'])
        self.assertEqual((core.verify,core.task_identity,core.read),original)

    def test_receipt_checks(self):
        plan=g.plans(self.cfg)[0]; plan.update(bot_sha256='c'*64,rival_sha256='d'*64,task_checkpoint_sha256=None)
        row=dict(plan,passed=True,states=720,action_turns=719,status=['DONE','DONE'],agent_errors=0,
            engine_sha256='e'*64,crop_checkpoint_sha256='f'*64,replay_gzip_sha256='1'*64,trace_sha256='2'*64,
            semantic_replay_sha256='3'*64,semantic_replay_sha256_v2='4'*64,rewards=[1,2])
        counts=dict(task_forwards=0,task_rows=0,crop_forwards=3,crop_rows=3,gate_evaluations=0,controller_calls=719,teacher_proposals=0)
        audit=dict(game=plan['game'],passed=True,errors=0,actor_actions_reproduced=719,rival_actions_reproduced=719,
            trace_turns_reproduced=719,replay_sha256='1'*64,trace_sha256='2'*64,live_model_counts=counts,primary_model_counts=dict(counts),
            official=dict(official_transitions_reexecuted=719,engine_sha256='e'*64,semantic_sha256='3'*64,
                semantic_sha256_v2='4'*64,final_rewards=[1,2],both_public_and_private_states_match_each_step=True,inventory_order_checked_each_step=True))
        complete=dict(game=plan['game'],passed=True,row_sha256='a'*64,certificate_sha256='b'*64)
        storage=dict(passed=True,error_kind=None,output=dict(refused=0))
        g.validate_receipts(plan,row,audit,complete,storage,'a'*64,'b'*64,self.cfg)
        for change in (dict(actor_actions_reproduced=718),dict(trace_sha256='bad'),dict(live_model_counts={**counts,'task_forwards':1})):
            with self.assertRaises(ValueError): g.validate_receipts(plan,row,{**audit,**change},complete,storage,'a'*64,'b'*64,self.cfg)
        with self.assertRaises(ValueError): g.validate_receipts(plan,row,audit,complete,{**storage,'passed':False},'a'*64,'b'*64,self.cfg)


if __name__=='__main__': unittest.main()
