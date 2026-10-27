import importlib.util
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch
from . import evaluation
from learning.task_job_sampling_race_v1 import sampling

exp=types.ModuleType('learning.task_teacher_choice_eval_v2.experiment')
exp.ROOT=Path(__file__).resolve().parents[2]; exp.OUT=exp.ROOT/'synthetic'
for name in ('ready','verify_commit','deadline_check','read','sha','write_new'): setattr(exp,name,lambda *a:None)
obs=types.ModuleType('learning.task_action_observation_v1.campaign'); obs.observe_processes=lambda *a:None
games=types.ModuleType('learning.task_teacher_choice_eval_v2.games')
for name in ('make_wrapper','task_identity','validate_receipts'): setattr(games,name,lambda *a:None)
spec=importlib.util.spec_from_file_location('learning.task_teacher_choice_eval_v2._campaign_test',Path(__file__).with_name('campaign.py'))
c=importlib.util.module_from_spec(spec)
with patch.dict(sys.modules,{'learning.task_teacher_choice_eval_v2.experiment':exp,
    'learning.task_action_observation_v1.campaign':obs,'learning.task_teacher_choice_eval_v2.games':games}): spec.loader.exec_module(c)


class CampaignTest(unittest.TestCase):
    def test_failure_receipt_blocks_retry_without_started(self):
        with patch.object(c,'ready',return_value={}),patch.object(c,'verify_commit'), \
             patch.object(Path,'exists',autospec=True,side_effect=lambda p:Path(p).name=='games-failure.local.json'), \
             patch.object(c,'supervise') as supervised:
            with self.assertRaises(ValueError): c.run(exp.OUT)
            supervised.assert_not_called()
    def test_sampling_supervisor_failure_stops_first_game(self):
        self.assertIs(c.supervise,sampling.supervise)
        cfg=dict(test_seeds=[1,2],official_job_supervision={}); records={}
        def marker(out,value,state): state.update(created=True,complete=True)
        with patch.object(c,'ready',return_value=cfg),patch.object(c,'verify_commit'),patch.object(Path,'exists',return_value=False), \
             patch.object(c,'marker',side_effect=marker),patch.object(c,'sha',return_value='h'),patch.object(c,'deadline_check'), \
             patch.object(c,'make_wrapper',return_value=exp.OUT/'agent'),patch.object(c,'task_identity',return_value=(None,'h')), \
             patch.object(c,'write_new',side_effect=lambda p,v:records.update({Path(p).name:v})), \
             patch.object(c,'WindowsAPI',return_value=None),patch.object(c,'supervise',return_value={'passed':False}) as supervised, \
             patch.object(c,'accept_job') as accept:
            with self.assertRaises(c.CampaignFailure): c.run(exp.OUT)
            self.assertEqual(supervised.call_count,1); accept.assert_not_called()
    def test_bank_and_profit_consistency(self):
        row=dict(side=0,banks=[100,70],initial_banks=[20,10],bank=100,opponent_bank=70,
                 initial_bank=20,opponent_initial_bank=10,net_profit=80,opponent_net_profit=60,margin=30,rewards=[100,70])
        c.verify_economy(row)
        for field in ('net_profit','opponent_net_profit','initial_bank','margin'):
            with self.subTest(field=field),self.assertRaises(ValueError): c.verify_economy({**row,field:999})

    def test_partial_and_spawn_failure_stop(self):
        for case in ('partial','spawn'):
            records={}
            def marker(out,value,state):
                state['created']=True; records['games-started.json']=value
                if case=='partial': raise OSError('partial')
                state['complete']=True
            cfg=dict(test_seeds=[1,2],official_job_supervision={})
            with patch.object(c,'ready',return_value=cfg),patch.object(c,'verify_commit'), \
                 patch.object(Path,'exists',autospec=True,side_effect=lambda p:Path(p).name in records), \
                 patch.object(c,'marker',side_effect=marker),patch.object(c,'sha',return_value='h'), \
                 patch.object(c,'deadline_check'),patch.object(c,'make_wrapper',return_value=exp.OUT/'agent'), \
                 patch.object(c,'task_identity',return_value=(None,'h')), \
                 patch.object(c,'write_new',side_effect=lambda p,v:records.update({Path(p).name:v})), \
                 patch.object(c,'WindowsAPI',return_value=None),patch.object(c,'supervise',side_effect=OSError('spawn')):
                with self.assertRaises(c.CampaignFailure): c.run(exp.OUT)
                self.assertIn('games-failure.local.json',records)
                self.assertNotIn('games-result.json',records)
                if case=='spawn': self.assertEqual(records['games-failure.local.json']['child_lifecycle'],'unknown_after_spawn_attempt')
                self.assertEqual(records['games-failure.local.json']['confirmed_completed_live_transitions'],0)
                with self.assertRaises(ValueError): c.run(exp.OUT)


if __name__ == '__main__': unittest.main()
