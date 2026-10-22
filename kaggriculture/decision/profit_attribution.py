"""OFFLINE ONLY: observation hooks around unchanged official production and actions.

Production opportunity is rule-derived, admitted units are observed state deltas.
Clipping is a quantity diagnostic, never a counterfactual cash reward.
"""
from collections import Counter
from contextlib import contextmanager
from copy import deepcopy
from threading import Lock
from decision.full_audit import reexecute, rules
from decision.action_views import opcode_view as _opcode_view

_LOCK=Lock()

class Attribution:
    def __init__(self, side):
        self.side=side;self.counts=Counter();self.sales={};self.production={};self.market_cash=0

    def sink(self,r):
        if r['player']!=self.side:return
        if r['phase']=='unit':
            op=_opcode_view(r['effective']);self.counts['unit:'+op+':'+r['status']]+=1
            tile=r.get('tile_before')
            if op=='HARVEST' and r['status']=='executed' and isinstance(tile,dict):
                crop=tile.get('crop','unknown');self.counts['harvest_units:'+crop]+=tile.get('yield_units',0)
            if op=='DROP':self.counts['manual_drop_discard_units']+=r.get('discarded_units',0)
        elif r['phase']=='market':
            a=r['requested'];cash=r.get('cash_delta',0);n=r.get('executed_units',0)
            self.market_cash+=cash
            op=_opcode_view(a)
            item=str(a[1]) if isinstance(a,list) and len(a)>1 else ''
            self.counts['cash:'+op+':'+item]+=cash
            self.counts['market_units:'+op+':'+item]+=n
            if op=='SELL':
                d=self.sales.setdefault(item,dict(units=0,revenue=0));d['units']+=n;d['revenue']+=cash

    @contextmanager
    def installed(self):
        assert _LOCK.acquire(blocking=False),'overlapping production hooks'
        end=rules._end_of_day;daily=rules._daily_refresh_plants;drop=rules._drop_inventories_to_shed
        selected=[None,None]
        def wrapped_end(state,env,day):
            selected[:]=[state[0].observation.farms[self.side],state[self.side].observation.private]
            try:return end(state,env,day)
            finally:selected[:]=[None,None]
        def wrapped_daily(farm,day,tpd):
            before=deepcopy(farm['tiles']) if farm is selected[0] else None
            result=daily(farm,day,tpd)
            if before is not None:
                for y,row in enumerate(before):
                    for x,tile in enumerate(row):
                        if not isinstance(tile,dict) or tile.get('kind')!='PLANT':continue
                        cd=rules.CROPS[tile['crop']];after=farm['tiles'][y][x]
                        if not cd['ongoing'] or after.get('kind')!='PLANT':continue
                        age=day+1-tile['planted_day']-cd['first_yield_day']
                        if age<0 or age%cd['interval'] or age//cd['interval']+1>cd['max_yield']:continue
                        opportunity=2 if tile['watered_today'] and tile['fertilized_until_day']>=day else 1
                        admitted=after['yield_units']-tile['yield_units']
                        assert 0<=admitted<=opportunity
                        d=self.production.setdefault(tile['crop'],Counter())
                        d.update(opportunities=1,potential_units=opportunity,admitted_units=admitted,clipped_units=opportunity-admitted)
            return result
        def wrapped_drop(private,capacity):
            before=sum(sum(v.values()) for v in private['inventories'])+sum(private['shed'].values()) if private is selected[1] else None
            result=drop(private,capacity)
            if before is not None:
                self.counts['automatic_drop_calls']+=1
                after=sum(sum(v.values()) for v in private['inventories'])+sum(private['shed'].values())
                assert before>=after
                self.counts['automatic_drop_discard_units']+=before-after
            return result
        rules._end_of_day=wrapped_end;rules._daily_refresh_plants=wrapped_daily;rules._drop_inventories_to_shed=wrapped_drop
        try:yield self
        finally:
            rules._end_of_day=end;rules._daily_refresh_plants=daily;rules._drop_inventories_to_shed=drop
            _LOCK.release()

    def result(self,replay):
        initial=replay['steps'][0][0]['observation']['farms'][self.side]['money']
        final=replay['steps'][-1][0]['observation']['farms'][self.side]['money']
        assert final-initial==self.market_cash, 'cash reconciliation'
        private=replay['steps'][-1][self.side]['observation']['private']
        stock=Counter(private['shed'])
        for inv in private['inventories']:stock.update(inv)
        return dict(counts=dict(self.counts),sales=self.sales,production={k:dict(v) for k,v in self.production.items()},market_cash=self.market_cash,cash_residual=final-initial-self.market_cash,terminal_stock=dict(stock))
