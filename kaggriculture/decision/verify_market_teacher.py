"""Official market boundary tests and the SELL-only observational contract."""
import argparse, hashlib, json
from copy import deepcopy
from pathlib import Path
from decision.verify_mechanisms import env, obs, step, PASS, rules
from decision.execution import ExecutionTrace
from decision.profit_attribution import Attribution
from decision.semantics import canonical

ROOT=Path(__file__).resolve().parents[1]
def load(path):
    ns={'__name__':'market_test_bot','__file__':str(path)}
    exec(compile(path.read_bytes(),str(path),'exec'),ns)
    return ns

def compare(base,candidate,o,cfg):
    before=deepcopy(o)
    a,b=base(o,cfg),candidate(o,cfg)
    assert o==before,'observation mutated'
    assert a['farmer']==b['farmer'] and a['hands']==b['hands']
    non=lambda x:[(i,v) for i,v in enumerate(x['market']) if v[0]!='SELL']
    assert non(a)==non(b),'non-SELL slot/quantity mismatch'
    assert len(b['market'])<=cfg['maxMarketOrdersPerTurn']
    return a,b

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    assert not a.output.exists();a.output.parent.mkdir(parents=True,exist_ok=True)
    baseline=load(ROOT/'bots/teacher_fertilizer_terminal_v2.py')
    candidate=load(ROOT/'bots/teacher_market_v1.py')
    rows=[]
    def record(name,**kw):rows.append(dict(case=name,passed=True,**kw))
    # Actual quote parity with official function; no speculative future tested as fact.
    quote_cases=0
    for c in baseline['CROPS']:
        for inv in range(9300,10801,17):
            assert candidate['quote'](c,inv,{})==rules.market_price(c,inv)
            quote_cases+=1
    record('current_price_curve',quotes=quote_cases)
    for side in (0,1):
        other=1-side
        # Both unit quotes use the same inventory before either commit.
        e=env()
        for s in e.state:s.observation.private['shed']['STRAWBERRY']=3
        market=e.state[0].observation.market
        i=market['inventory']['STRAWBERRY']
        expected=sum(rules.market_price('STRAWBERRY',i+2*j) for j in range(3))
        joint=[dict(farmer=['PASS'],hands=[],market=[['SELL','STRAWBERRY',3]]) for _ in (0,1)]
        events=[]
        with ExecutionTrace(rules,events.append).installed(e):e.step(joint)
        assert [f['money'] for f in e.state[0].observation.farms]==[3000+expected]*2
        record('same_slot_lockstep',side=side,cash_each=expected)
        # Separate slots lose simultaneity; duplicate orders compete for stock.
        e=env()
        e.state[side].observation.private['shed']['STRAWBERRY']=3
        e.state[other].observation.private['shed']['STRAWBERRY']=3
        joint=[deepcopy(PASS),deepcopy(PASS)]
        joint[side]['market']=[['SELL','STRAWBERRY',2],['SELL','STRAWBERRY',3]]
        joint[other]['market']=[['BUY_SEED','WHEAT',1],['SELL','STRAWBERRY',3]]
        m=Attribution(side);events=[]
        def sink(r):m.sink(r);events.append(r)
        with ExecutionTrace(rules,sink).installed(e):e.step(joint)
        sales=[r for r in events if r['phase']=='market' and r['player']==side]
        assert [r['executed_units'] for r in sales]==[2,1]
        assert sales[-1]['status']=='partial' and m.sales['STRAWBERRY']['units']==3
        assert sales[1]['first_quote']==rules.market_price('STRAWBERRY',10002)
        record('multiple_orders_partial_and_slot_order',side=side,executed=[2,1],cash=m.market_cash)
        # Cash arrives at SELL, not in advance of an earlier BUY.
        for sell_first in (False,True):
            e=env();e.state[0].observation.farms[side]['money']=0
            priv=e.state[side].observation.private;priv['shed']['CARROT']=1
            initial=priv['seeds']['WHEAT'];joint=[deepcopy(PASS),deepcopy(PASS)]
            orders=[['SELL','CARROT',1],['BUY_SEED','WHEAT',1]]
            joint[side]['market']=orders if sell_first else orders[::-1]
            e.step(joint)
            assert e.state[side].observation.private['seeds']['WHEAT']==initial+sell_first
            record('cash_order',side=side,sell_first=sell_first)
        # Units deposit before SELL; selling cannot rescue pre-market overflow.
        e=env();priv=e.state[side].observation.private
        priv['shed']['WHEAT']=99;priv['inventories'][0]={'TOMATO':2,'CARROT':2}
        joint=[deepcopy(PASS),deepcopy(PASS)]
        joint[side]=dict(farmer=['DROP'],hands=[],market=[['SELL','TOMATO',2]])
        m=Attribution(side)
        projected,_=candidate['shed_after_transfers'](obs(e,side),e.configuration,joint[side])
        assert projected['TOMATO']==1 and projected['CARROT']==0
        with m.installed(),ExecutionTrace(rules,m.sink).installed(e):e.step(joint)
        assert m.counts['manual_drop_discard_units']==3 and m.sales['TOMATO']['units']==1
        record('drop_before_sell_capacity_and_insertion_order',side=side,discarded=3,sold=1)
        # Extra orders are ignored by official max-order truncation.
        e=env();e.configuration.maxMarketOrdersPerTurn=1
        e.state[side].observation.private['shed']['WHEAT']=2
        joint=[deepcopy(PASS),deepcopy(PASS)]
        joint[side]['market']=[['BUY_SEED','WHEAT',1],['SELL','WHEAT',2]]
        e.step(joint)
        assert e.state[side].observation.private['shed']['WHEAT']==2
        record('order_limit',side=side)
        # At price floor, executed sales do not increase supply.
        e=env();market=e.state[0].observation.market
        market['inventory']['MELON']=12000;rules._refresh_prices(market)
        e.state[side].observation.private['shed']['MELON']=3
        joint=[deepcopy(PASS),deepcopy(PASS)];joint[side]['market']=[['SELL','MELON',3]]
        e.step(joint)
        assert e.state[0].observation.farms[side]['money']==3003
        assert e.state[0].observation.market['inventory']['MELON']==11999 # center consumes at 0
        record('floor_sale_supply',side=side)
    # Real Environment nightly automatic deposit; SELL frees space before night.
    for sell in (False,True):
        e=env()
        for _ in range(23):e.step([deepcopy(PASS),deepcopy(PASS)])
        priv=e.state[0].observation.private
        priv['shed']['WHEAT']=99;priv['inventories'][0]={'CARROT':3}
        action=deepcopy(PASS)
        if sell:action['market']=[['SELL','WHEAT',2]]
        m=Attribution(0)
        with m.installed(),ExecutionTrace(rules,m.sink).installed(e):e.step([action,deepcopy(PASS)])
        assert m.counts['automatic_drop_discard_units']==(0 if sell else 2)
        record('night_storage',sell_first=sell,discarded=m.counts['automatic_drop_discard_units'])
    # Official interpreter boundary at last action: no invented final auto-deposit.
    for drop in (False,True):
        e=env();priv=e.state[0].observation.private;priv['inventories'][0]={'CARROT':2}
        action=dict(farmer=['DROP'] if drop else ['PASS'],hands=[],market=[['SELL','CARROT',2]])
        step(e,action,at=718)
        assert e.state[0].status=='DONE'
        assert e.state[0].reward==(3069 if drop else 3000)
        assert sum(e.state[0].observation.private['inventories'][0].values())==(0 if drop else 2)
        record('last_action_drop_sell_no_auto_settlement',drop=drop,reward=e.state[0].reward)
    # Same-input contract, price floor, full shed, last day, order-limit pressure.
    cases=0
    for side in (0,1):
        for at in (0,240,262,696,718):
            for limit in (1,2,10):
                e=env();e.configuration.maxMarketOrdersPerTurn=limit
                o=deepcopy(obs(e,side));o.update(step=at,day=at//24,hour=at%24)
                o['town']['unlocked_shops']=['PIZZA_SHOP','BRUNCH_SPOT']*3
                o['private']['shed'].update(STRAWBERRY=40,TOMATO=40)
                o['farms'][side]['money']=10000
                compare(baseline['agent'],candidate['agent'],o,e.configuration);cases+=1
    record('same_input_contract_boundary_states',observations=cases)
    # Explicit partial hold, bounded flush and competitor-visible supply guard.
    e=env();o=deepcopy(obs(e));o.update(step=240,day=10,hour=0)
    o['town']['unlocked_shops']=['BRUNCH_SPOT']*3
    o['private']['shed']['STRAWBERRY']=40
    action=dict(farmer=['PASS'],hands=[],market=[['SELL','STRAWBERRY',40],['BUY_PRODUCT','FERTILIZER',2],['HIRE']])
    planned=candidate['sale_plan'](o,e.configuration,action)
    assert 0<planned[0][2]<40 and planned[1:]==action['market'][1:]
    held=40-planned[0][2]
    for at in (262,696,718):
        x=deepcopy(o);x.update(step=at,day=at//24,hour=at%24)
        assert candidate['sale_plan'](x,e.configuration,action)[0][2]==40
    for tile_x in range(5):
        t=rules._new_plant('STRAWBERRY',0,24);t['yield_units']=4
        o['farms'][1]['tiles'][0][tile_x]=t
    assert candidate['sale_plan'](o,e.configuration,action)[0][2]==40
    record('bounded_hold_flush_and_visible_rival',held_units=held)
    # Privileged opponent inventory and info.seed do not enter actual Agent obs.
    e=env();before=deepcopy(obs(e));x=candidate['agent'](before,e.configuration)
    e.state[1].observation.private['shed']['STRAWBERRY']=99999;e.info['seed']=987654
    after=deepcopy(obs(e));assert before==after
    assert candidate['agent'](after,e.configuration)==x
    assert e.configuration['seed'] is None and 'info' not in before
    record('opponent_private_and_seed_isolation')
    # Stateless candidate remains identical after unrelated and reversed calls.
    e=env();o=deepcopy(obs(e));expected=candidate['agent'](o,e.configuration)
    for at in (718,24,0,200):
        x=deepcopy(o);x.update(step=at,day=at//24,hour=at%24)
        candidate['agent'](x,e.configuration)
    assert candidate['agent'](o,e.configuration)==expected
    record('cross_game_statelessness')
    result=dict(passed=True,cases=rows,candidate_sha256=hashlib.sha256((ROOT/'bots/teacher_market_v1.py').read_bytes()).hexdigest(),scope='Official full Environment entry except explicitly labelled terminal interpreter boundary. Constructed states; economic forecast is not certified future truth.')
    a.output.write_bytes((json.dumps(result,indent=2)+'\n').encode())
    print(json.dumps(dict(passed=True,cases=len(rows),quote_comparisons=quote_cases,same_input_cases=cases)))

if __name__=='__main__':main()
