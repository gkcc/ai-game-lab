"""Read saved official boundary evidence; no Environment.step or new games."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
sys.path.insert(0, str(ROOT))
from decision.semantics import digest
from learning.task_choice_v1.cases import prepare_pair

def read(p):
    return json.loads(p.read_text(encoding='utf-8-sig'))

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    report = read(OUT/'official-cases.json')
    assert report['passed'] and report['pairs'] == 6
    assert report['new_official_transitions'] == 12
    public = ROOT.parent/'publication/ai-game-lab'
    commit = read(OUT/'pre-game-commit.json')['commit']
    assert commit == '1d43efcfaf1b9c96e6f3c107ecb9b11a12bdca4c'
    for name in ('registration.json', 'freeze.json', 'pre-review.json'):
        blob = subprocess.check_output(['git','-C',str(public),'show',
            commit+':review/evidence/'+OUT.name+'/'+name])
        assert blob == (OUT/name).read_bytes()
    checked = []
    for row in report['rows']:
        kind = row['case']
        initial,a,b,base,alt,target = prepare_pair(kind) # observations only
        assert digest(initial.state) == row['input_state_sha256']
        assert base['records'][0]['teacher_label'] == row['first_teacher_label']
        assert alt['records'][0]['chosen'] == row['alternative_first_choice']
        raw_pair = []
        for summary in row['branches']:
            branch = summary['branch']
            path = OUT/'cases-local'/f'{kind}-{branch}.json'
            raw = read(path)
            assert sha(path) == summary['local_raw_sha256']
            assert raw['counter_placeholders_are_not_simulated_history'] is True
            assert digest(raw['before']) == row['input_state_sha256']
            assert digest(raw['after']) == summary['state_sha256']
            assert raw['action'] == summary['action'] == (a if branch == 'baseline' else b)
            prior = raw['before'][0]['observation']['step']
            assert raw['after'][0]['observation']['step'] == prior + 1
            for state in raw['after']:
                o = state['observation']
                assert (o['day'],o['hour']) == divmod(prior+1,24)
                assert state['status'] not in ('ERROR','INVALID','TIMEOUT')
            events=[]; units=[]
            for t in raw['trace']:
                assert t['step']==prior
                p=t['player']
                if t['phase']=='unit':
                    events.append([p,'unit',t['unit'],t['requested'],t['effective'],t['status']])
                    if p==0:
                        units.append({k:t[k] for k in ('unit','requested','effective','status')} |
                                     {'successful_plant':t.get('successful_plant',False)})
                else:
                    # These six cases have at most one one-unit market order.
                    assert t['phase']=='market' and p==0 and t['order']==0
                    assert t['executed_units']==1 and t['failed_commits']==0
                    op=t['requested'][0]
                    if op=='HIRE':
                        events.append([p,'market_atomic',0,t['requested'],1])
                    else:
                        assert op=='BUY_SEED' and t['requested'][2]==1
                        assert t['first_quote']==t['last_quote']
                        events.append([p,'market_fill',0,op,t['requested'][1],t['first_quote'],True])
            h=hashlib.sha256()
            for i,event in enumerate(events,1):
                h.update(json.dumps([i,event],sort_keys=True,separators=(',',':')).encode()+b'\n')
            assert h.hexdigest()==summary['effect_sha256']
            assert units==summary['units']
            if branch=='alternative':
                event=next(t for t in raw['trace'] if t['phase']=='unit' and t['player']==0 and t['unit']==target)
                assert event['status']=='executed'
                before,after=event['tile_before'],event['tile_after']
                if kind in ('PLANT','joint_PLANT'):
                    crop=event['effective'][1]
                    assert before is None and after['crop']==crop
                    assert event['before']['seeds'][crop]-event['after']['seeds'][crop]==1
                    assert sum(u['successful_plant'] for u in units)==1
                elif kind=='DIG': assert before['kind']=='WEED' and after is None
                elif kind=='WATER': assert not before['watered_today'] and after['watered_today']
                elif kind=='HARVEST':
                    assert before['yield_units']==2 and after['yield_units']==0
                    inv_before=dict(event['before']['inventories'][target])
                    inv_after=dict(event['after']['inventories'][target])
                    assert inv_after.get('TOMATO',0)-inv_before.get('TOMATO',0)==2
                else:
                    assert target==1 and after['fertilized_until_day']==9
                    assert dict(event['before']['inventories'][1])['FERTILIZER']==1
                    assert dict(event['after']['inventories'][1]).get('FERTILIZER',0)==0
                    assert dict(event['after']['inventories'][0])['FERTILIZER']==1
                assert summary['target_direct_official_effect_verified']
            raw_pair.append(raw)
        assert raw_pair[0]['before']==raw_pair[1]['before']
        assert raw_pair[0]['after']!=raw_pair[1]['after']
        if kind=='FERTILIZE':
            end=raw_pair[0]['after'][0]['observation']['private']['inventories']
            assert end[0].get('FERTILIZER',0)==0 and end[1]['FERTILIZER']==1
        if kind=='joint_PLANT':
            first,second=alt['records'][:2]
            assert (first['unit'],second['unit'])==(1,0)
            x=second['inputs'];crop=b['hands'][0][1]
            assert x['planned']['available_units']==[0] and x['planned']['seeds'][crop]==0
            assert all(c['unit']==0 and c['target']!=[1,1] for c in x['candidates'])
            assert alt['final_reservations']['needed_seeds'][crop]==1
            # The official market replenishes the consumed seed; zero final stock is not required.
            assert raw_pair[1]['after'][0]['observation']['private']['seeds'][crop]==1
        checked.append(kind)
    receipt=dict(passed=True,case_count=len(checked),saved_raw_states_checked=12,
        new_official_transitions_executed_by_reviewer=0,new_complete_games=0,training_runs=0,
        official_case_transitions_recorded_by_root=12,preexecution_commit=commit,
        freeze_sha256=sha(OUT/'freeze.json'),official_cases_sha256=sha(OUT/'official-cases.json'),
        script_sha256=sha(Path(__file__)),checked_cases=checked,
        limitations=['Reads original official execution evidence and reconstructs requests without executing transitions.',
                    'Synthetic fixtures are not natural trajectories or economic evidence.',
                    'Full60 dataset reconstruction and planned two old replay checks remain separate.'])
    path=OUT/'independent-case-review.json'
    assert not path.exists()
    path.write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(receipt))

if __name__=='__main__':main()
