"""Official Environment -> ExecutionTrace -> Attribution compatibility regressions."""
import argparse,gzip,io,json
from contextlib import redirect_stdout,redirect_stderr
from copy import deepcopy
from pathlib import Path
from decision.verify_mechanisms import env,PASS,rules
from decision.execution import ExecutionTrace
from decision.profit_attribution import Attribution
from decision.full_audit import reexecute
from decision.semantics import canonical

def consume(side,action,setup=None,hire=False):
    plain,traced=env(),env()
    if hire:
        actions=[deepcopy(PASS),deepcopy(PASS)];actions[side]['market']=[['HIRE']]
        for e in (plain,traced):e.step(deepcopy(actions))
        assert len(traced.state[0].observation.farms[side]['hands'])==1
    if setup:
        for e in (plain,traced):setup(e,side)
    actions=[deepcopy(PASS),deepcopy(PASS)];actions[side]=deepcopy(action)
    measure=Attribution(side);events=[]
    def sink(record):
        before=deepcopy(record);measure.sink(record)
        assert record==before,'consumer mutated raw event'
        events.append(record)
    with redirect_stdout(io.StringIO()),redirect_stderr(io.StringIO()):
        plain.step(deepcopy(actions))
        with measure.installed(),ExecutionTrace(rules,sink).installed(traced):traced.step(deepcopy(actions))
    assert actions[side]==action
    assert canonical(plain.state)==canonical(traced.state)
    assert canonical(traced.state[side].action)==canonical(action)
    return measure,[r for r in events if r['player']==side]

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True);out=parser.parse_args().output
    assert not out.exists();out.parent.mkdir(parents=True,exist_ok=True)
    rows=[]
    malformed=[[],None,{},7,'PASS',[None],[7],['']]
    for side in (0,1):
        for unit in ('farmer','hand'):
            for payload in malformed:
                action=deepcopy(PASS)
                if unit=='farmer':action['farmer']=deepcopy(payload)
                else:action['hands']=[deepcopy(payload)]
                m,events=consume(side,action,hire=unit=='hand')
                event=next(r for r in events if r['phase']=='unit' and r['unit']==(0 if unit=='farmer' else 1))
                assert event['requested']==event['effective']==payload and event['status']=='no_effect'
                assert m.counts['unit:invalid:no_effect']==1
                assert not any(v for k,v in m.counts.items() if k.startswith('unit:invalid:') and k!='unit:invalid:no_effect')
                rows.append(dict(side=side,branch=unit,payload=payload,status=event['status'],raw_event_unchanged=True,official_state_equal=True))
        for payload in [[],None,{},7,'SELL',[None],[7],[''],[{}]]:
            action=deepcopy(PASS);action['market']=[payload]
            m,events=consume(side,action)
            event=next(r for r in events if r['phase']=='market')
            assert event['requested']==payload and event['invalid'] and event['status']=='no_effect'
            assert event['executed_units']==0 and event['cash_delta']==0
            assert m.market_cash==0 and not m.sales
            assert m.counts['market_units:invalid:']==0
            rows.append(dict(side=side,branch='market',payload=payload,status=event['status'],executed_units=0,raw_event_unchanged=True,official_state_equal=True))

    valid=[]
    for side in (0,1):
        m,events=consume(side,PASS)
        assert m.counts['unit:PASS:pass']==1 and m.counts['unit:invalid:no_effect']==0
        action=deepcopy(PASS);action['farmer']=['UNRECOGNIZED']
        m,_=consume(side,action);assert m.counts['unit:UNRECOGNIZED:no_effect']==1
        action=deepcopy(PASS);action['farmer']=['PLANT','CARROT']
        def no_seeds(e,p):e.state[p].observation.private['seeds']['CARROT']=0
        m,events=consume(side,action,setup=no_seeds)
        event=next(r for r in events if r['phase']=='unit')
        assert event['requested']==['PLANT','CARROT'] and event['effective']==['PASS'] and event['status']=='blocked'
        assert m.counts['unit:PASS:blocked']==1 and m.counts['unit:PLANT:executed']==0
        action=deepcopy(PASS);action['market']=[['SELL','CARROT',5]]
        def two_carrots(e,p):e.state[p].observation.private['shed']['CARROT']=2
        m,events=consume(side,action,setup=two_carrots)
        event=next(r for r in events if r['phase']=='market')
        assert event['status']=='partial' and event['executed_units']==m.sales['CARROT']['units']==2
        assert m.sales['CARROT']['revenue']==m.market_cash>0
        sale_cash=m.market_cash
        action=deepcopy(PASS);action['market']=[['BUY_SEED','CARROT',1]]
        m,events=consume(side,action)
        assert m.counts['market_units:BUY_SEED:CARROT']==1 and m.market_cash==-20

        # Three real official transitions: harvest, drop+sell, then no effect.
        plain,traced=env(),env();m=Attribution(side)
        for e in (plain,traced):
            t=rules._new_plant('CARROT',-8,24);t['yield_units']=3
            e.state[0].observation.farms[side]['tiles'][4][4]=t
        actions=[dict(farmer=['HARVEST'],hands=[],market=[]),dict(farmer=['DROP'],hands=[],market=[['SELL','CARROT',3]]),deepcopy(PASS)]
        expected=[]
        for action in actions:
            joint=[deepcopy(PASS),deepcopy(PASS)];joint[side]=action
            plain.step(deepcopy(joint));expected.append(canonical(plain.state))
        with m.installed(),ExecutionTrace(rules,m.sink).installed(traced):
            for action,wanted in zip(actions,expected):
                joint=[deepcopy(PASS),deepcopy(PASS)];joint[side]=action
                traced.step(deepcopy(joint))
                assert canonical(traced.state)==wanted
        assert m.counts['harvest_units:CARROT']==m.sales['CARROT']['units']==3
        assert m.counts['unit:HARVEST:executed']==m.counts['unit:DROP:executed']==1
        assert m.result(traced.toJSON())['cash_residual']==0

        # Do not swallow a genuine official error or replace it with a consumer error.
        errors=[]
        action=deepcopy(PASS);action['farmer']=[{}]
        for tracing in (False,True):
            e=env();joint=[deepcopy(PASS),deepcopy(PASS)];joint[side]=deepcopy(action)
            try:
                with redirect_stdout(io.StringIO()),redirect_stderr(io.StringIO()):
                    if tracing:
                        with Attribution(side).installed(),ExecutionTrace(rules,Attribution(side).sink).installed(e):e.step(joint)
                    else:e.step(joint)
            except Exception as error:errors.append((type(error).__name__,str(error)))
            else:raise AssertionError('official error was swallowed')
        assert errors[0]==errors[1] and errors[0][0]=='TypeError'
        valid.append(dict(side=side,pass_preserved=True,unknown_string_preserved=True,atomic_block_preserved=True,partial_sale_units=2,partial_sale_cash=sale_cash,buy_seed_cost=20,harvest_drop_sale_units=3,cash_residual=0,original_error_preserved=errors[0]))

    complete=[]
    for side in (0,1):
        e=env();action=dict(farmer=[],hands=[],market=[[None]])
        while not e.done:
            joint=[deepcopy(PASS),deepcopy(PASS)];joint[side]=deepcopy(action);e.step(joint)
        replay=e.toJSON();assert len(replay['steps'])==720
        for turn in replay['steps'][1:]:assert turn[side]['action']==action
        raw=json.dumps(replay,separators=(',',':')).encode()
        path=out.parent/f'boundary-{side}.json.gz';assert not path.exists();path.write_bytes(gzip.compress(raw,mtime=0))
        m=Attribution(side)
        with m.installed():audit=reexecute(replay,trace=True,label_sink=m.sink)
        assert audit['final_rewards']==[3000.,3000.]
        assert m.counts['unit:invalid:no_effect']==719 and m.counts['unit:PASS:pass']==0
        assert m.market_cash==0 and not m.sales
        complete.append(dict(side=side,audit=audit,metrics=m.result(replay)))
    report=dict(passed=True,malformed_official_entry_cases=rows,valid_and_error_cases=valid,complete_boundary_replays=complete,scope='Installed official engine -> actual tracer -> repaired consumer; raw events unmodified, malformed no-effects distinct from PASS; privileged telemetry remains offline')
    out.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(passed=True,malformed_cases=len(rows),valid_sides=len(valid),complete_boundary_games=len(complete))))

if __name__=='__main__':main()
