"""Independent task dataset audit: saved rows plus two preselected old replays."""
from collections import Counter
from contextlib import ExitStack
import gzip
import hashlib
from itertools import zip_longest
import json
from pathlib import Path
import sys

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[1]
sys.path.insert(0,str(ROOT))
OPS=('PLANT','DIG','WATER','HARVEST','FERTILIZE')

def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def canon(v):return json.dumps(v,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False)
def digest(v):return hashlib.sha256(canon(v).encode()).hexdigest()
def actor(p):
    ns={'__name__':'independent_frozen_actor','__file__':str(p)}
    exec(p.read_bytes(),ns)
    return ns['agent']

def main():
    cfg=read(OUT/'registration.json');ledger=read(OUT/'reconstruction-ledger.json')
    coverage=read(OUT/'coverage.json');manifest=read(OUT/'dataset-manifest.json')
    source=ROOT/cfg['demonstration_root']
    originals={r['game']:r for name in cfg['ledger_names'] for r in read(source/name)}
    assert len(originals)==60 and len(ledger['rows'])==60
    assert ledger['passed'] and ledger['games']==ledger['games_dispatched']==60 and not ledger['failures']
    assert ledger['new_games']==ledger['training_runs']==0
    assert ledger['official_existing_replay_transitions']==43140
    assert set(originals)=={r['game'] for r in ledger['rows']}
    expected={(s,o,p) for s in cfg['train_seeds']+cfg['validation_seeds'] for o in ('v1','market','round2') for p in (0,1)}
    assert {(r['seed'],r['opponent'],r['side']) for r in ledger['rows']}==expected
    assert manifest['reconstruction_ledger_sha256']==sha(OUT/'reconstruction-ledger.json')
    assert manifest['coverage_sha256']==sha(OUT/'coverage.json')
    assert manifest['files']==[dict(game=r['game'],split=r['split'],files=r['files']) for r in ledger['rows']]
    totals={s:{'decisions':0,'competition':0,'counts':{op:Counter() for op in OPS},
        'contexts':{},'selected':{op:set() for op in OPS},'candidate_seeds':{op:set() for op in OPS},
        'selected_seeds':{op:set() for op in OPS}} for s in ('train','validation')}
    files=0
    for ri,row in enumerate(ledger['rows'],1):
        original=originals[row['game']];split='train' if row['seed'] in cfg['train_seeds'] else 'validation'
        assert split==row['split'];acc=totals[split]
        assert all(row[k]==719 for k in ('reconstructed_actor_actions','original_teacher_actions_checked','reproduced_rival_actions'))
        audit=row['audit']
        assert audit['passed'] and audit['official_transitions_reexecuted']==719
        for a,b in [('semantic_sha256','semantic_replay_sha256'),('semantic_sha256_v2','semantic_replay_sha256_v2'),('final_rewards','rewards')]:
            assert audit[a]==original[b]
        replaypath=source/'games'/row['game']/'replay.json.gz'
        assert sha(replaypath)==original['replay_gzip_sha256']==row['replay_gzip_sha256']
        assert sha(source/'sources'/(row['opponent']+'.py'))==row['rival_sha256']
        count={op:Counter() for op in OPS};contexts={};selected={op:set() for op in OPS}
        n=competition=0;last=(-1,-1)
        with ExitStack() as stack:
            streams=[]
            for kind in ('inputs','labels','index'):
                f=row['files'][kind];path=OUT/f['path']
                assert sha(path)==f['sha256'] and f['rows']==row['decisions']
                streams.append(stack.enter_context(gzip.open(path,'rt',encoding='utf-8')));files+=1
            for line,label_line,index_line in zip_longest(*streams):
                assert line is not None and label_line is not None and index_line is not None
                x,y,m=map(json.loads,(line,label_line,index_line));n+=1
                assert set(y)=={'index','teacher_label'}
                assert all(m[k]==row[k] for k in ('game','seed','opponent','side','split'))
                assert 0<=m['step']<719
                key=(m['step'],m['decision']);assert key>last
                assert m['decision']==(last[1]+1 if m['step']==last[0] else 0);last=key
                h=hashlib.sha256(line.rstrip('\n').encode()).hexdigest()
                assert h==m['input_sha256']
                assert x['schema_sha256']==row['schema_sha256']==manifest['schema_sha256']
                for forbidden in ('teacher_label','priority','rank','teacher_score','selected_flag','game','seed','split'):
                    assert ('"'+forbidden+'":') not in line
                candidates=x['candidates'];ids=[c['id'] for c in candidates]
                assert len(ids)==len(set(ids)) and ids[y['index']]==y['teacher_label']
                assert candidates==sorted(candidates,key=lambda c:(c['unit'],c['target'][1],c['target'][0],c['op'],c['id']))
                byop=Counter(c['op'] for c in candidates);assert set(byop)<=set(OPS)
                comp=len(byop)>1;competition+=comp
                op=candidates[y['index']]['op']
                selected[op].add((h,y['teacher_label']));acc['selected'][op].add((h,y['teacher_label']))
                contexts[h]=byop;acc['contexts'][h]=byop
                for name,num in byop.items():
                    count[name]['candidate_count']+=num
                    count[name]['candidate_competitive_count']+=num*comp
                    acc['candidate_seeds'][name].add(row['seed'])
                count[op]['selected_count']+=1;count[op]['selected_competitive_count']+=comp
                acc['selected_seeds'][op].add(row['seed'])
        assert n==row['decisions'] and competition==row['cross_task_competition_decisions']
        unique=Counter()
        for byop in contexts.values():unique.update(byop)
        for op in OPS:
            actual=row['per_task'][op]
            for key in ('candidate_count','candidate_competitive_count','selected_count','selected_competitive_count'):
                assert count[op][key]==actual.get(key,0),(row['game'],op,key)
            assert unique[op]==actual['unique_candidate_encodings']
            assert len(selected[op])==actual['unique_selected_encodings']
            acc['counts'][op].update(count[op])
        acc['decisions']+=n;acc['competition']+=competition
        if ri%10==0:print('saved dataset checked',ri,flush=True)
    for split,acc in totals.items():
        expected=coverage['splits'][split]
        assert acc['decisions']==expected['decisions'] and acc['competition']==expected['cross_task_competition_decisions']
        unique=Counter()
        for byop in acc['contexts'].values():unique.update(byop)
        for op in OPS:
            target=expected['tasks'][op]
            for key in ('candidate_count','candidate_competitive_count','selected_count','selected_competitive_count'):
                assert acc['counts'][op][key]==target[key]
            assert unique[op]==target['unique_candidate_encodings']
            assert len(acc['selected'][op])==target['unique_selected_encodings']
            assert sorted(acc['candidate_seeds'][op])==target['candidate_seeds']
            assert sorted(acc['selected_seeds'][op])==target['selected_seeds']
            assert target['selected_count']>0 and target['selected_competitive_count']>0
    assert coverage['passed'] and all(coverage['gates'].values())
    assert manifest['examples']==sum(a['decisions'] for a in totals.values())==135406
    # Independent loop rather than dataset.check; two preselected historical trajectories only.
    from learning.task_choice_v1.agent import TaskAgent
    from decision.audit_market_teacher import observation
    from decision.full_audit import reexecute
    sample_results=[]
    for sample in read(OUT/'independent-sample-plan.json')['existing_replay_samples']:
        row=next(r for r in ledger['rows'] if r['game']==sample['game']);side=row['side']
        with gzip.open(source/'games'/row['game']/'replay.json.gz','rt',encoding='utf-8') as f:replay=json.load(f)
        teacher=actor(source/'sources/teacher.py');rival=actor(source/'sources'/(row['opponent']+'.py'));agent=TaskAgent()
        decisions=0
        with gzip.open(OUT/row['files']['inputs']['path'],'rt',encoding='utf-8') as xs, gzip.open(OUT/row['files']['labels']['path'],'rt',encoding='utf-8') as ys:
            for before,after in zip(replay['steps'],replay['steps'][1:]):
                o=observation(before,side);a=agent(o,replay['configuration'])
                assert a==teacher(o,replay['configuration'])==after[side]['action']
                assert rival(observation(before,1-side),replay['configuration'])==after[1-side]['action']
                for r in agent.last['records']:
                    x=json.loads(next(xs));y=json.loads(next(ys));decisions+=1
                    assert r['inputs']==x and r['teacher_label']==y['teacher_label']
                    # Direct observable joint identity; no task-rank target.
                    chosen=x['candidates'][y['index']]
                    assert chosen['id']==f"u{chosen['unit']}:x{chosen['target'][0]}:y{chosen['target'][1]}:{chosen['op']}"
                    assert chosen['unit']==r['unit'] and chosen['target']==r['target'] and chosen['op']==r['op']
            assert next(xs,None) is None and next(ys,None) is None
        audit=reexecute(replay)
        assert audit==row['audit']
        sample_results.append({'game':row['game'],'actor_actions':719,'rival_actions':719,'input_and_label_rows_reconstructed':decisions,'official_transitions':719,'audit':audit})
        print('old replay sample passed',row['game'],flush=True)
    protected=read(OUT/'protection-before.json');public=ROOT.parent/'publication/ai-game-lab'
    for rel,h in protected['runtime'].items():assert sha(ROOT/rel)==h
    for rel,h in protected['public'].items():assert sha(public/rel)==h
    result=dict(passed=True,script_sha256=sha(Path(__file__)),freeze_sha256=sha(OUT/'freeze.json'),
        ledger_sha256=sha(OUT/'reconstruction-ledger.json'),coverage_sha256=sha(OUT/'coverage.json'),
        all_dataset_files_hashed=files,all_dataset_rows_aligned=manifest['examples'],
        reconstructed_sample_input_rows=sum(s['input_and_label_rows_reconstructed'] for s in sample_results),
        unique_count_method='Sum candidate operation counts once per distinct complete context hash; selected uniqueness uses context hash plus candidate identity.',
        root_existing_official_replay_transitions=43140,reviewer_existing_official_replay_transitions=1438,
        reviewer_new_official_case_transitions=0,new_complete_games=0,training_runs=0,
        protected_runtime_files=len(protected['runtime']),protected_public_files=len(protected['public']),
        samples=sample_results,selected_counts={s:{op:a['counts'][op]['selected_count'] for op in OPS} for s,a in totals.items()},
        limitations=['All saved dataset rows checked; only two preselected old trajectories independently reexecuted and reconstructed.',
                    'No new official case transitions; six cases separately reviewed from original saved evidence.',
                    'Structural coverage and uniqueness do not establish independent examples, class balance, learned policy strength or full-game legality coverage.'])
    path=OUT/'independent-review.json';assert not path.exists()
    path.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='samples'}))

if __name__=='__main__':main()
