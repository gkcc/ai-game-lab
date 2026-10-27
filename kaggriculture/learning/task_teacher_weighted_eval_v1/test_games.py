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
exp=types.ModuleType('learning.task_teacher_weighted_eval_v1.experiment')
exp.ROOT=Path(__file__).resolve().parents[2]
exp.verify=lambda *a:None
exp.ready=lambda _:None
exp.verify_commit=lambda _:None
exp.deadline_check=lambda:None
spec=importlib.util.spec_from_file_location('learning.task_teacher_weighted_eval_v1._games_test',Path(__file__).with_name('games.py'))
g=importlib.util.module_from_spec(spec)
with patch.dict(sys.modules,{'learning.task_anchor_once_v1.games':core,'learning.task_teacher_weighted_eval_v1.experiment':exp}): spec.loader.exec_module(g)


class GamesTest(unittest.TestCase):
    def setUp(self):
        self.cfg=dict(test_seeds=[1,2,3],weighted_checkpoint='new/model.npz',weighted_checkpoint_sha256='a'*64,
            unweighted_checkpoint='old/model.npz',unweighted_checkpoint_sha256='b'*64,
            source_sha256={'round2.py':'d'*64,'market.py':'c'*64},engine_sha256='e'*64,crop_checkpoint_sha256='f'*64,
            game_limits=dict(controller_calls=719,task_forwards=4000,crop_forwards=8000,neural_input_rows=160000,
                             gate_evaluations=1,teacher_proposals=1))

    def job_plan(self):
        p=g.plans([1,2,3])[0]
        return dict(p,bot_sha256='c'*64,rival_sha256='d'*64,task_checkpoint_sha256='a'*64)

    def test_plans_and_checkpoint_guard(self):
        for p in g.plans([1,2,3]):
            g.validate_plan(p,self.cfg)
            self.assertEqual(g.task_identity(None,p,self.cfg)[1],('a' if p['policy']=='weighted' else 'b')*64)
        p=g.plans([1,2,3])[0]
        for changes in ({'seed':4},{'side':True},{'opponent':'unknown'},{'policy':'other'}):
            with self.assertRaises(ValueError): g.validate_plan({**p,**changes},self.cfg)
        for path in ('../outside','/absolute','x/../other','C:' + '/bad','x\\bad'):
            with self.assertRaises(ValueError): g.task_identity(None,p,{**self.cfg,'weighted_checkpoint':path})
        with self.assertRaises(ValueError): g.validate_plan({**self.job_plan(),'task_checkpoint_sha256':'b'*64},self.cfg,job=True)

    def test_changed_second_read_and_restoration(self):
        plan=self.job_plan(); changed={**plan,'bot_sha256':'1'*64}
        out=exp.ROOT/'synthetic'; path=(out/'job-inputs'/(plan['game']+'.json')).resolve()
        exp.ready=lambda _:self.cfg; original=(core.verify,core.task_identity,core.read)
        core.job=lambda out,key:core.read(path)
        with patch.object(Path,'open',side_effect=[io.StringIO(json.dumps(plan)),io.StringIO(json.dumps(changed))]):
            with self.assertRaises(ValueError): g.job(out,plan['game'])
        self.assertEqual((core.verify,core.task_identity,core.read),original)

    def test_receipt_identity_and_counts(self):
        plan=self.job_plan()
        row=dict(plan,passed=True,states=720,action_turns=719,status=['DONE','DONE'],agent_errors=0,
            engine_sha256='e'*64,crop_checkpoint_sha256='f'*64,replay_gzip_sha256='1'*64,trace_sha256='2'*64,
            semantic_replay_sha256='3'*64,semantic_replay_sha256_v2='4'*64,rewards=[1,2])
        counts=dict(task_forwards=10,task_rows=20,crop_forwards=3,crop_rows=3,gate_evaluations=0,controller_calls=719,teacher_proposals=0)
        audit=dict(game=plan['game'],passed=True,errors=0,actor_actions_reproduced=719,rival_actions_reproduced=719,
            trace_turns_reproduced=719,replay_sha256='1'*64,trace_sha256='2'*64,
            live_model_counts=counts,primary_model_counts=dict(counts),official=dict(official_transitions_reexecuted=719,
            passed=True,engine_sha256='e'*64,semantic_sha256='3'*64,semantic_sha256_v2='4'*64,final_rewards=[1,2],
            both_public_and_private_states_match_each_step=True,inventory_order_checked_each_step=True))
        complete=dict(game=plan['game'],passed=True,row_sha256='5'*64,certificate_sha256='6'*64)
        storage=dict(passed=True,error_kind=None,output={'refused':0})
        def validate(r=row,a=audit): g.validate_receipts(plan,r,a,complete,storage,'5'*64,'6'*64,self.cfg)
        validate()
        for bad in ({**row,'agent_errors':1},{**row,'trace_sha256':'invalid'}):
            with self.assertRaises(ValueError): validate(r=bad)
        for bad in ({**audit,'errors':1},{**audit,'trace_sha256':'7'*64},
                    {**audit,'official':{**audit['official'],'passed':False}},
                    {**audit,'live_model_counts':None},{**audit,'live_model_counts':{**counts,'task_forwards':0}},
                    {**audit,'primary_model_counts':{**counts,'task_rows':21}}):
            with self.assertRaises(ValueError): validate(a=bad)


if __name__=='__main__': unittest.main()
