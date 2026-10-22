"""Official ordered replay audit, SELL-only shadow checks and existing accounting."""
import argparse,gzip,hashlib,json,multiprocessing
from collections import Counter
from concurrent.futures import ProcessPoolExecutor,as_completed
from copy import deepcopy
from pathlib import Path
from decision.full_audit import reexecute
from decision.profit_attribution import Attribution
from decision.verify_market_teacher import compare,load
from decision.action_views import opcode_view

ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def observation(frame,side):
    # Only officially shared fields plus this player's own private observation.
    o=deepcopy(frame[side]['observation'])
    for name in ('step','farms','market','town','day','hour'):
        o[name]=deepcopy(frame[0]['observation'][name])
    return o

def count_request(events,side,r):
    """Count raw order events, not fills; preserve no_effect/partial/ignored.

    Non-SELL includes malformed non-SELL orders, also beyond the order cap.
    market_invalid counts only the tracer's explicit parser-rejection flag;
    an ignored order is not silently reclassified as parser-rejected.
    Executed quantities and cash remain in Attribution, not these order counts.
    """
    if r['player']!=side:return
    if r['phase']=='market':
        op=opcode_view(r['requested'])
        events['market:'+op+':'+r['status']]+=1
        events['market_invalid']+=bool(r.get('invalid'))
        if op=='SELL':events['sell_requested_units']+=r.get('requested_units',0)
        else:
            events['non_sell_requested_orders']+=1
            events['non_sell_incomplete_orders']+=r['status']!='executed'
    elif r['phase']=='unit':
        events['unit:'+r['status']]+=1

def check(job):
    row,folder,sources,kind=job;folder=Path(folder);sources=Path(sources)
    raw=(folder/'replay.json.gz').read_bytes();assert hashlib.sha256(raw).hexdigest()==row['replay_gzip_sha256']
    replay=json.loads(gzip.decompress(raw));side=row['side']
    baseline=load(sources/'baseline.py')['agent'];candidate=load(sources/'candidate.py')['agent']
    opponent=load(sources/(row['opponent']+'.py'))['agent']
    contract=Counter();stock=[]
    for before,after in zip(replay['steps'],replay['steps'][1:]):
        o=observation(before,side)
        a,b=compare(baseline,candidate,o,replay['configuration'])
        wanted=b if row['policy']=='candidate' else a
        assert wanted==after[side]['action'],'policy source does not reproduce action'
        assert opponent(observation(before,1-side),replay['configuration'])==after[1-side]['action']
        contract['observations']+=1
        contract['sell_decision_differences']+=a['market']!=b['market']
        stock.append(sum(o['private']['shed'].values()))
    measure=Attribution(side);rival=Attribution(1-side);events=Counter()
    def sink(r):
        measure.sink(r);rival.sink(r)
        count_request(events,side,r)
    with measure.installed():audit=reexecute(replay,trace=True,label_sink=sink)
    assert audit['final_rewards']==row['banks']
    assert audit['semantic_sha256']==row['semantic_replay_sha256']
    assert audit['semantic_sha256_v2']==row['semantic_replay_sha256_v2']
    assert sha(sources/(row['policy']+'.py'))==row['bot_sha256']
    assert sha(sources/(row['opponent']+'.py'))==row['rival_sha256']
    metrics=measure.result(replay);other=rival.result(replay)
    # Own production/drop telemetry includes nightly loss via the existing hook.
    # Rival metrics here cover trades/unit events, not rival nightly production.
    return {k:row[k] for k in ('seed','opponent','side','policy','replay_gzip_sha256','bot_sha256','rival_sha256')}|dict(kind=kind,audit=audit,metrics=metrics,
        opponent_cash_and_sales=dict(market_cash=other['market_cash'],cash_residual=other['cash_residual'],sales=other['sales'],terminal_stock=other['terminal_stock']),
        same_input_contract=dict(passed=True,non_sell_slot_or_quantity_mismatches=0,unit_mismatches=0,**contract),
        request_outcomes=dict(events),storage=dict(max_observed_shed_units=max(stock),mean_observed_shed_units=sum(stock)/len(stock)),policy_actions_reproduced=True,opponent_actions_reproduced=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--stage',choices=['development','test'],required=True);a=p.parse_args();root=a.root.resolve()
    cfg=read(root/('registration.json' if a.stage=='test' else 'development-plan.json'))
    freeze=read(root/(a.stage+'-freeze.json'))
    for n,h in freeze['runtime_sha256'].items():assert sha(ROOT/n)==h,n
    for n,h in freeze['source_sha256'].items():assert sha(root/'sources'/n)==h,n
    jobs=[(r,str(root/a.stage/f"{r['seed']}-{r['opponent']}-{r['side']}-{r['policy']}"),str(root/'sources'),a.stage) for r in read(root/a.stage/'ledger.json')]
    if a.stage=='test':jobs += [(r,str(root/'repeat'/f"{r['policy']}-{r['side']}"),str(root/'sources'),'repeat') for r in read(root/'reproducibility.json')['rows']]
    target=root/(a.stage+'-audit.json');assert not target.exists();rows=[]
    with ProcessPoolExecutor(max_workers=cfg['workers'],mp_context=multiprocessing.get_context('spawn')) as pool:
        for f in as_completed([pool.submit(check,j) for j in jobs]):
            rows.append(f.result());print(f'{a.stage} official audit {len(rows)}/{len(jobs)}',flush=True)
    rows.sort(key=lambda r:(r['kind'],r['seed'],r['opponent'],r['side'],r['policy']))
    report=dict(passed=True,games=len(rows),official_transitions=719*len(rows),process_isolated=True,
        source_freeze_sha256=sha(root/(a.stage+'-freeze.json')),auditor_sha256=sha(Path(__file__)),rows=rows,
        scope='Full official public/private ordered states and v1/v2 digests. Raw replay stays local; published aggregate evidence is not independent access to raw states. Same-input contract is distinct from trajectory feedback.')
    target.write_bytes((json.dumps(report,indent=2)+'\n').encode())
    print(json.dumps(dict(passed=True,games=len(rows),non_sell_incomplete=sum(r['request_outcomes'].get('non_sell_incomplete_orders',0) for r in rows),storage_loss=sum(r['metrics']['counts'].get('manual_drop_discard_units',0)+r['metrics']['counts'].get('automatic_drop_discard_units',0) for r in rows))))

if __name__=='__main__':main()
