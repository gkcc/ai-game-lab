"""No post-stop dispatch and real-request control evidence, without real games."""
from copy import deepcopy
from pathlib import Path
import unittest
from unittest.mock import patch

from . import campaign
from .games import first_request_difference
from .policy import LIMITS
from .statistics import ARMS,EARLY,DEVELOPMENT,HOLDOUT,BOOTSTRAP


def config():
    return dict(train_seeds=[1,2,3,4],validation_seeds=[5,6],test_seeds=list(range(7,15)),
        opponents=['round2','market'],seats=[0,1],arms=list(ARMS),early_screen=EARLY,
        development=DEVELOPMENT,holdout=HOLDOUT,bootstrap=BOOTSTRAP,game_limits=LIMITS,
        task_forward_budget=776000,crop_forward_budget=2192000,
        neural_input_row_budget=43840000,policy_turn_budget=392574)


def frames():
    unit = dict(action={'farmer':['WAIT'],'hands':[],'market':[]},
                observation={'private':{'inventories':[{'CARROT':1,'MELON':2}]}})
    return [[deepcopy(unit),deepcopy(unit)] for _ in range(720)]


class CampaignTests(unittest.TestCase):
    def test_first_seed_failed_screen_dispatches_only_four_registered_cells(self):
        cfg = config(); seen=[]; saved={}
        runner = campaign.Runner(Path('unused'),cfg)
        def execute(jobs):
            seen.extend(jobs)
            return {j['game']:dict(j,passed=True,states=720,action_turns=719,
                status=['DONE','DONE'],agent_errors=0,label_binding_passed=True,
                groups=10,candidate_rows=20,teacher_ID_disagreements=0) for j in jobs}
        runner.execute=execute
        with patch.object(campaign,'summarize_game',side_effect=lambda out,row,cfg:row), \
             patch.object(campaign,'write_new',side_effect=lambda path,value:saved.__setitem__(path.name,value)):
            self.assertFalse(runner.collect())
        self.assertEqual({j['seed'] for j in seen},{1})
        self.assertEqual(len(seen),4)
        self.assertFalse(saved['early-gate.json']['passed'])

    def test_complete_budget_reserves_every_dispatch_including_inflight(self):
        cfg=config();runner=campaign.Runner(Path('unused'),cfg)
        jobs=[]
        for seed in cfg['train_seeds']:jobs.extend(campaign.plans(cfg,'train',seed,['baseline']))
        for seed in cfg['validation_seeds']:jobs.extend(campaign.plans(cfg,'dev',seed,cfg['arms']))
        for seed in cfg['test_seeds']:jobs.extend(campaign.plans(cfg,'holdout',seed,cfg['arms']))
        self.assertEqual(len(jobs),136)
        for job in jobs:
            runner.reserve(job);runner.attempts.append(job)
        with self.assertRaises(AssertionError):runner.reserve(jobs[-1])

    def test_full_request_order_change_and_absent_change(self):
        a=frames();b=deepcopy(a)
        self.assertIsNone(first_request_difference(a,b,0))
        a[101][0]['action']['market']=[['SELL','MELON',1],['SELL','CARROT',1]]
        b[101][0]['action']['market']=[['SELL','CARROT',1],['SELL','MELON',1]]
        self.assertEqual(first_request_difference(a,b,0),100)

    def test_opponent_or_unexplained_state_difference_is_not_candidate_control(self):
        a=frames();b=deepcopy(a)
        a[11][1]['action']['farmer']=['MOVE','N']
        with self.assertRaises(AssertionError):first_request_difference(a,b,0)
        a=frames()
        a[10][0]['observation']['private']['inventories'][0]={'MELON':2,'CARROT':1}
        with self.assertRaises(AssertionError):first_request_difference(a,b,0)


if __name__=='__main__':unittest.main()
