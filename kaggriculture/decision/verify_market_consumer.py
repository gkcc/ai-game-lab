"""Official entry -> real tracer -> market counter/comparators regressions."""
import argparse,gzip,hashlib,io,json
from collections import Counter
from contextlib import redirect_stdout,redirect_stderr
from copy import deepcopy
from pathlib import Path
from decision.verify_mechanisms import env,obs,PASS,rules
from decision.execution import ExecutionTrace
from decision.profit_attribution import Attribution
from decision.audit_market_teacher import check,count_request
from decision.verify_market_teacher import compare,load
from decision.compare_market_trajectories import differences
from decision.action_views import opcode_view
from decision.semantics import canonical,digest

PAYLOADS=[[],None,{},7,'SELL',[None],[7],[''],[{}],['UNRECOGNIZED'],['SELL']]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()

def wire(orders):return dict(farmer=['PASS'],hands=[],market=deepcopy(orders))

def consume(side,orders,setup=None):
    plain,traced=env(),env()
    if setup:
        for e in (plain,traced):setup(e,side)
    action=wire(orders);joint=[deepcopy(PASS),deepcopy(PASS)];joint[side]=action
    counts=Counter();measure=Attribution(side);rival=Attribution(1-side);records=[]
    def sink(r):
        before=deepcopy(r);measure.sink(r);rival.sink(r);count_request(counts,side,r)
        assert r==before,'consumer rewrote raw event'
        records.append(r)
    plain.step(deepcopy(joint))
    with measure.installed(),ExecutionTrace(rules,sink).installed(traced):traced.step(deepcopy(joint))
    assert canonical(plain.state)==canonical(traced.state)
    assert traced.state[side].action==action and action==wire(orders)
    return counts,measure,[r for r in records if r['player']==side and r['phase']=='market']

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True,type=Path);out=p.parse_args().output
    assert not out.exists();out.parent.mkdir(parents=True,exist_ok=True)
    fixtures=out.parent/'fixtures';fixtures.mkdir(exist_ok=True)
    malformed=[];valid=[];complete=[]
    for side in (0,1):
        for payload in PAYLOADS:
            counts,m,events=consume(side,[payload]);r=events[0];op=opcode_view(payload)
            assert len(events)==1 and r['requested']==payload and r['invalid'] and r['status']=='no_effect'
            assert r['executed_units']==r['cash_delta']==m.market_cash==0
            assert all(v['units']==v['revenue']==0 for v in m.sales.values())
            assert counts['market:'+op+':no_effect']==counts['market_invalid']==1
            assert counts['non_sell_incomplete_orders']==counts['non_sell_requested_orders']==int(op!='SELL')
            assert counts['sell_requested_units']==0
            e=env();policy=lambda o,c:wire([payload]);compare(policy,policy,obs(e,side),e.configuration)
            malformed.append(dict(side=side,payload=payload,label=op,status=r['status'],executed_units=0,cash_delta=0,raw_unchanged=True,official_states_equal=True,same_input_comparison_passed=True))
        # Real partial sale plus a later purchase; quantities are not order counts.
        def stock(e,s):e.state[s].observation.private['shed']['CARROT']=2
        counts,m,events=consume(side,[[],['SELL','CARROT',5],['BUY_SEED','CARROT',1]],stock)
        assert [r['status'] for r in events]==['no_effect','partial','executed']
        assert counts['sell_requested_units']==5 and m.sales['CARROT']['units']==2
        assert counts['non_sell_requested_orders']==2 and counts['non_sell_incomplete_orders']==1
        assert counts['market:SELL:partial']==counts['market:BUY_SEED:executed']==1
        assert m.counts['cash:BUY_SEED:CARROT']==-20
        partial_cash=m.sales['CARROT']['revenue']
        # Beyond-cap event is ignored, not a parsed invalid or an execution.
        cap=env().configuration.maxMarketOrdersPerTurn
        counts,_,events=consume(side,[[]]*cap+[[None]])
        assert counts['market:invalid:no_effect']==counts['market_invalid']==cap
        assert counts['market:invalid:ignored']==1
        assert counts['non_sell_incomplete_orders']==counts['non_sell_requested_orders']==cap+1
        ignored=next(r for r in events if r['status']=='ignored')
        assert ignored['requested']==[None] and 'invalid' not in ignored
        # Existing native market error must remain an error, not a successful no-op.
        errors=[]
        for tracing in (False,True):
            e=env();joint=[deepcopy(PASS),deepcopy(PASS)];joint[side]=wire([['SELL','WHEAT',float('inf')]])
            counts=Counter();m=Attribution(side)
            def sink(r):m.sink(r);count_request(counts,side,r)
            try:
                with redirect_stdout(io.StringIO()),redirect_stderr(io.StringIO()):
                    if tracing:
                        with m.installed(),ExecutionTrace(rules,sink).installed(e):e.step(joint)
                    else:e.step(joint)
            except Exception as ex:errors.append((type(ex).__name__,str(ex)))
            else:raise AssertionError('official error swallowed')
        assert errors[0]==errors[1] and errors[0][0]=='OverflowError'
        valid.append(dict(side=side,partial_requested_units=5,partial_filled_units=2,partial_sale_cash=partial_cash,purchase_cost=20,invalid_non_sell_order_count=1,ignored_order_count=1,parser_rejected_orders_at_cap=cap,official_error_preserved=errors[0]))

        # Actual complete official game -> unchanged check() public entry. No
        # synthetic tracer records, hidden sink monkeypatch, or comparison bypass.
        folder=fixtures/f'complete-{side}';folder.mkdir();sources=folder/'sources';sources.mkdir()
        code=(f'PAYLOADS={PAYLOADS!r}\n'
              "def agent(observation,configuration):\n"
              "    return {'farmer':['PASS'],'hands':[],'market':[PAYLOADS[observation['step']%len(PAYLOADS)]]}\n")
        for n in ('baseline.py','candidate.py'):(sources/n).write_text(code,encoding='utf-8')
        (sources/'opponent.py').write_text("def agent(observation,configuration):\n    return {'farmer':['PASS'],'hands':[],'market':[]}\n",encoding='utf-8')
        e=env();policy=load(sources/'candidate.py')['agent'];opponent=load(sources/'opponent.py')['agent']
        bots=[opponent,opponent];bots[side]=policy;e.run(bots)
        replay=e.toJSON();assert e.done and len(e.steps)==720 and [s.status for s in e.state]==['DONE','DONE']
        replay_path=folder/'replay.json.gz';replay_path.write_bytes(gzip.compress(json.dumps(replay,separators=(',',':')).encode(),mtime=0))
        row=dict(seed=replay['info']['seed'],side=side,policy='candidate',opponent='opponent',banks=[s.reward for s in e.state],
            replay_gzip_sha256=sha(replay_path),semantic_replay_sha256=digest(e.steps,1),semantic_replay_sha256_v2=digest(e.steps),bot_sha256=sha(sources/'candidate.py'),rival_sha256=sha(sources/'opponent.py'))
        cert=check((row,str(folder),str(sources),'boundary_regression'))
        expected_non_sell=sum(opcode_view(PAYLOADS[t%len(PAYLOADS)])!='SELL' for t in range(719))
        assert cert['request_outcomes']['market_invalid']==719
        assert cert['request_outcomes']['non_sell_requested_orders']==cert['request_outcomes']['non_sell_incomplete_orders']==expected_non_sell
        assert cert['request_outcomes']['sell_requested_units']==0
        assert cert['audit']['final_rewards']==[3000.,3000.] and cert['metrics']['market_cash']==0
        assert cert['same_input_contract']['observations']==719 and cert['metrics']['cash_residual']==0
        assert differences([replay,replay],side)==dict(unit_action_turns_different=0,non_sell_slot_turns_different=0,non_sell_sequence_turns_different=0,turns=719)
        complete.append(cert)

    # Both comparators must retain malformed raw values/positions, not drop them.
    comparison=[]
    for left,right,slot,seq in [([[]],[[None]],1,1),([[],['SELL','WHEAT',1]],[['SELL','WHEAT',1],[]],1,0),([[],['SELL','WHEAT',1]],[[],['SELL','WHEAT',2]],0,0)]:
        games=[]
        for orders in (left,right):
            e=env();e.step([wire(orders),deepcopy(PASS)]);games.append(e.toJSON())
        delta=differences(games,0)
        assert delta['non_sell_slot_turns_different']==slot and delta['non_sell_sequence_turns_different']==seq
        e=env();a=lambda o,c:wire(left);b=lambda o,c:wire(right)
        try:compare(a,b,obs(e),e.configuration)
        except AssertionError as ex:
            assert slot and str(ex)=='non-SELL slot/quantity mismatch'
        else:assert not slot
        comparison.append(dict(left=left,right=right,**delta))
    result=dict(passed=True,malformed_cases=malformed,valid_and_error_cases=valid,comparator_cases=comparison,complete_boundary_games=complete,
        full_official_entry_to_auditor_games=2,official_reexecution_transitions=1438,
        scope='Actual official transitions and full replay check(); raw requests preserved. Comparator expects a market list; this is not arbitrary-policy legality certification. Boundary games excluded from historical economics.')
    out.write_bytes((json.dumps(result,indent=2)+'\n').encode())
    print(json.dumps(dict(passed=True,malformed_cases=len(malformed),valid_sides=len(valid),comparator_cases=len(comparison),complete_boundary_games=len(complete))))

if __name__=='__main__':main()
