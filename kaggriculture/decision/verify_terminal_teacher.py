"""Regression contracts for terminal economic windows, using official transitions."""
import unittest
from copy import deepcopy
from decision.verify_mechanisms import env, obs, step, rules, PASS
from bots.teacher_fertilizer_terminal_v2 import agent, saleable_bonus_nights
from bots.teacher_fertilizer_v1 import agent as old_agent


class TerminalTeacher(unittest.TestCase):
    def fixture(self, crop='STRAWBERRY', planted=16, at=672, fertilizer=1):
        e=env()
        e.state[0].observation.step=at
        for s in e.state:
            s.observation.day=at//24; s.observation.hour=at%24
        f=e.state[0].observation.farms[0]
        f['farmer']=[4,4]; f['hands']=[]
        f['tiles'][4][4]=rules._new_plant(crop,planted,24)
        p=e.state[0].observation.private
        p['inventories']=[{'FERTILIZER':fertilizer}]; p['shed']={}
        return e

    def test_no_production_no_purchase_or_fertilization(self):
        for held in (0,1):
            e=self.fixture(fertilizer=held)
            e.state[0].observation.market['inventory']['STRAWBERRY']=9700
            a=agent(obs(e),e.configuration)
            self.assertNotEqual(a['farmer'],['FERTILIZE'])
            self.assertFalse(any(x[:2]==['BUY_PRODUCT','FERTILIZER'] for x in a['market']))
            old=old_agent(obs(e),e.configuration)
            self.assertTrue(old['farmer']==['FERTILIZE'] if held else any(x[:2]==['BUY_PRODUCT','FERTILIZER'] for x in old['market']))

    def test_tomato_only_one_real_night(self):
        e=self.fixture('TOMATO',20)
        t=e.state[0].observation.farms[0]['tiles'][4][4]
        self.assertEqual(saleable_bonus_nights(t,(4,4),28,e.configuration),1)
        step(e,{'farmer':['FERTILIZE'],'hands':[],'market':[]})
        nights=[]
        while not e.done:
            n=e.state[0].observation.step
            step(e,{'farmer':['WATER'],'hands':[],'market':[]})
            if (n+1)%24==0:nights.append(n)
        self.assertEqual(nights,[695]);self.assertEqual(e.state[0].status,'DONE')

    def test_strawberry_alternate_night_and_nonterminal(self):
        e=self.fixture()
        tile=rules._new_plant('STRAWBERRY',16,24)
        self.assertEqual(saleable_bonus_nights(tile,(4,4),27,e.configuration),1)
        self.assertEqual(saleable_bonus_nights(tile,(4,4),28,e.configuration),0)
        tile=rules._new_plant('TOMATO',0,24)
        self.assertEqual(saleable_bonus_nights(tile,(4,4),7,e.configuration),3)

    def test_exact_cash_deadline_and_distance(self):
        e=self.fixture('TOMATO',20)
        tile=e.state[0].observation.farms[0]['tiles'][4][4]
        cfg=dict(e.configuration)
        # Night at 695, HARVEST at 696, DROP+SELL at 697 requires last_action >=697.
        cfg['episodeSteps']=699
        self.assertEqual(saleable_bonus_nights(tile,(4,4),28,cfg),1)
        self.assertEqual(saleable_bonus_nights(tile,(0,0),28,cfg),0)
        cfg['episodeSteps']=698
        self.assertEqual(saleable_bonus_nights(tile,(4,4),28,cfg),0)

    def test_official_last_turn_drop_and_sell(self):
        e=self.fixture('TOMATO',20,at=695)
        e.configuration.episodeSteps=699
        tile=e.state[0].observation.farms[0]['tiles'][4][4]
        tile['fertilized_until_day']=30
        step(e,{'farmer':['WATER'],'hands':[],'market':[]})
        self.assertEqual(e.state[0].observation.farms[0]['tiles'][4][4]['yield_units'],2)
        step(e,{'farmer':['HARVEST'],'hands':[],'market':[]})
        before=e.state[0].observation.farms[0]['money']
        step(e,{'farmer':['DROP'],'hands':[],'market':[['SELL','TOMATO',2]]})
        self.assertGreater(e.state[0].observation.farms[0]['money'],before)
        self.assertEqual(e.state[0].observation.private['shed'].get('TOMATO',0),0)
        self.assertEqual(e.state[0].status,'DONE')


if __name__=='__main__': unittest.main()
