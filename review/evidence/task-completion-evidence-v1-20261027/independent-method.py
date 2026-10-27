"""One-shot independent sampled outcome review; outputs only safe aggregates."""
import gzip, hashlib, json
from pathlib import Path
from learning.task_action_impact_diagnostic_v1.analysis import Meter,read_json,trace_lines,compact_replay,validate_trace,descriptor
from learning.task_action_observation_v1.analysis import raw_path
from decision.full_audit import reexecute
from learning.task_completion_evidence_v1.experiment import deadline_check

ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
SOURCE=ROOT/'artifacts/task-action-observation-v1-20261027'
def load(p): return json.loads(p.read_text(encoding='utf-8-sig'))
def effect(op,e,day):
    try:
        b,a=e['tile_before'],e['tile_after']; x,y=e['before'],e['after']; u=e['unit']
        inv=lambda state,item:dict(state['inventories'][u]).get(item,0)
        if op=='DIG': return b is not None and b!='LOCKED' and a is None
        if op=='WATER': return isinstance(b,dict) and b['kind']=='PLANT' and b['watered_today'] is False and isinstance(a,dict) and a['kind']=='PLANT' and a['crop']==b['crop'] and a['watered_today'] is True
        if op=='PLANT':
            crop=e['effective'][1]
            return b is None and isinstance(a,dict) and a['kind']=='PLANT' and a['crop']==crop and x['seeds'].get(crop,0)-y['seeds'].get(crop,0)==1
        if op=='HARVEST':
            if not isinstance(b,dict) or b['kind']!='PLANT': return False
            crop,n=b['crop'],b['yield_units']
            return isinstance(n,(int,float)) and n>0 and (a is None or isinstance(a,dict) and a['kind']=='PLANT' and a['crop']==crop and a['yield_units']==0) and inv(y,crop)-inv(x,crop)==n
        if op=='FERTILIZE': return isinstance(day,int) and not isinstance(day,bool) and isinstance(b,dict) and b['kind']=='PLANT' and isinstance(a,dict) and a['kind']=='PLANT' and a['crop']==b['crop'] and inv(x,'FERTILIZER')-inv(y,'FERTILIZER')==1 and a['fertilized_until_day']==max(b.get('fertilized_until_day',-1),day+2)
    except (KeyError,IndexError,TypeError,ValueError): return False
    return False
def main():
    if not load(OUT/'run-result.json')['passed'] or not load(OUT/'analysis-result.json')['complete']: raise ValueError('Main run not accepted')
    deadline_check()
    with (OUT/'independent-attempt.local.json').open('x',encoding='utf-8') as f:
        json.dump({'attempt_consumed':True,'retry_allowed':False},f)
    games=load(OUT/'input-manifest.json')['games']; game=next(g for g in games if g['seed']==3911611436 and g['opponent']=='round2' and g['side']==0 and g['arm']=='candidate')
    manifest=load(OUT/'dataset-manifest.json'); part=next(x for x in manifest['games'] if x['game']==game['key'])
    data=(OUT/part['path']).read_bytes()
    if hashlib.sha256(data).hexdigest()!=part['sha256']: raise ValueError('Label identity mismatch')
    rows=[json.loads(s) for s in gzip.decompress(data).splitlines()]
    def rank(r):return hashlib.sha256(json.dumps([r['game'],r['first_step'],r['first_ordinal']],ensure_ascii=True,separators=(',',':')).encode('ascii')).digest()
    sample=sorted(rows,key=rank)[:20]
    if len(sample)!=20 or len({(r['first_step'],r['first_ordinal']) for r in sample})!=20:raise ValueError('Sample invalid')
    meter=Meter(compressed_limit=128*2**20,decoded_limit=2**30)
    replay=read_json(raw_path(SOURCE,game['replay'],'replay'),game['replay'],meter)
    compact=compact_replay(replay,game); desc=descriptor(); selections=[]
    stream=trace_lines(raw_path(SOURCE,game['trace'],'trace'),game['trace'],meter)
    for step,line in enumerate(stream):
        if step>=719:raise ValueError('Extra trace')
        records=validate_trace(line,compact,step,desc)
        selections.append(records)
    if len(selections)!=719:raise ValueError('Incomplete trace')
    events={}
    def sink(e):
        deadline_check()
        if e.get('player')==game['side'] and e.get('phase')=='unit': events.setdefault((e['step'],e['unit']),[]).append(e)
    audit=reexecute(replay,trace=True,label_sink=sink)
    ledger=next(x for x in load(SOURCE/'ledger.json') if x['game']==game['key'])
    if not audit['passed'] or audit['official_transitions_reexecuted']!=719 or audit['semantic_sha256_v2']!=ledger['semantic_replay_sha256_v2'] or audit['engine_sha256']!=ledger['engine_sha256']:raise ValueError('Replay audit mismatch')
    comparisons=[]
    for row in sample:
        start,ordinal=row['first_step'],row['first_ordinal']; selected=selections[start][ordinal]
        unit,target,op=selected['unit'],selected['target'],selected['op']
        identity=(row['game']==game['key'] and row['side']==game['side'] and row['unit']==unit and row['target']==target and row['op']==op and row['chosen']==selected['chosen'])
        status='unknown'; end=start
        for step in range(start,719):
            matches=[s for s in selections[step] if s['unit']==unit]
            if len(matches)!=1 or matches[0]['target']!=target or matches[0]['op']!=op:break
            s=matches[0];end=step; ev=events.get((step,unit),[]); req=s['unit_request']
            if len(ev)==1:
                e=ev[0]
                if e.get('status')=='executed' and e.get('position')==target and req and req[0]==op and e.get('requested')==req and e.get('effective')==req and effect(op,e,compact['days'][step]):status='completed_observed';break
        comparisons.append(identity and row['status']==status and row['last_step']==end)
    result=dict(passed=all(comparisons),sample_count=len(sample),matched=sum(comparisons),official_transitions=719,raw_reads=meter.snapshot(),selection_rule='registered SHA256 identity, independent of label',scope='historical development data; no causal or performance claim',private_details_exported=False)
    with (OUT/'independent-replay-verification.json').open('x',encoding='utf-8') as f:json.dump(result,f,indent=2);f.write('\n')
    if not result['passed']:raise ValueError('Sample comparison failed')
if __name__=='__main__': main()
