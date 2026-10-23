"""Descriptive aggregation and deterministic local witnesses, no policy calls."""
from collections import Counter
import gzip
import json
from pathlib import Path
import time

from learning.bc_crop.agent import ROOT, sha
from learning.bc_crop.experiment import read, save
from decision.audit_market_teacher import observation
from .numeric import SummaryAccumulator, summarize


def check_time(deadline):
    if time.monotonic() >= deadline: raise TimeoutError('Diagnostic global deadline exceeded')


def stream(out, kind, rows, deadline):
    for row in rows:
        with gzip.open(Path(out)/kind/(row['game']+'.jsonl.gz'),'rt',encoding='utf-8') as f:
            for line in f:
                check_time(deadline)
                value=json.loads(line)
                assert value['game']==row['game']
                yield value


def rank(row,cfg):
    return (cfg['seeds'].index(row['seed']),row['side'],row['step'],row.get('decision',0))


def stratum(row): return row['opponent'],row['origin']


def meta(row,kind,decision=0):
    return {**{k:row[k] for k in ('game','seed','opponent','side','origin','step')},
            'decision':decision,'kind':kind}


def read_turn(path,step):
    with gzip.open(path,'rt',encoding='utf-8') as f:
        for i,line in enumerate(f):
            if i==step:return json.loads(line)
    raise ValueError('Missing witness turn')


def identity(record,row):
    assert record['game']==row['game'] and record['seed']==row['seed']
    assert record['side']==row['side'] and record['opponent']==row['opponent']
    assert record['origin']==row['policy']


def validate_compact(records,row,expected,deadline):
    groups=candidates=0;previous=None
    for r in records:
        check_time(deadline);identity(r,row)
        step,decision=r['step'],r['decision']
        assert type(step) is int and 0<=step<719 and type(decision) is int and decision>=0
        assert r['time_third']==step//240
        assert previous is None or (step,decision)>previous
        assert decision==(0 if previous is None or step!=previous[0] else previous[1]+1)
        previous=(step,decision)
        assert type(r['candidate_count']) is int and r['candidate_count']>0
        groups+=1;candidates+=r['candidate_count']
    assert groups==expected['factual_groups'] and candidates==expected['factual_candidates'], 'Incomplete compact population'
    return dict(factual_groups=groups,factual_candidates=candidates)


def validate_turns(records,row,expected,deadline):
    turns=factual=shadow=0
    for r in records:
        check_time(deadline);identity(r,row)
        assert type(r['step']) is int and r['step']==turns and r['time_third']==turns//240
        for name in ('factual_task_calls','shadow_task_calls'):
            assert type(r[name]) is int and r[name]>=0
        factual+=r['factual_task_calls'];shadow+=r['shadow_task_calls'];turns+=1
    assert turns==expected['turns']==719 and factual==expected['factual_groups'] and shadow==expected['shadow_groups'], 'Incomplete or duplicated turn population'
    return dict(turns=turns,factual_groups=factual,shadow_groups=shadow)


def validate_outputs(out,source,deadline):
    ledger=read(out/'jobs-ledger.json');assert len(ledger)==98 and all(r['passed'] for r in ledger)
    jobs={r['key']:r for r in ledger};assert len(jobs)==98
    assert set(jobs)=={'inventory','support'}|{'game:'+r['game'] for r in source}
    assert sha(out/'inventory.json')==jobs['inventory']['receipt_sha256']
    assert sha(out/'support-receipt.json')==jobs['support']['receipt_sha256']
    inventory=read(out/'inventory.json');inv={r['game']:r for r in inventory['games']}
    assert len(inv)==96 and set(inv)=={r['game'] for r in source}
    totals=Counter();receipts={}
    for row in source:
        check_time(deadline);game=row['game'];p=out/'receipts'/(game+'.json')
        assert sha(p)==jobs['game:'+game]['receipt_sha256']
        receipt=read(p);assert receipt['passed'] and receipt['game']==game
        assert receipt['counts']==jobs['game:'+game]['counts']
        assert receipt['inputs_sha256']['replay']==row['replay_gzip_sha256']
        assert receipt['inputs_sha256']['trace']==row['trace_sha256']
        for kind in ('compact','turns','shadows'):
            assert sha(out/kind/(game+'.jsonl.gz'))==receipt['outputs_sha256'][kind], 'Diagnostic output changed'
        count=validate_compact(stream(out,'compact',[row],deadline),row,receipt['counts'],deadline)
        validate_turns(stream(out,'turns',[row],deadline),row,receipt['counts'],deadline)
        assert all(receipt['counts'][k]==v for k,v in inv[game]['counts'].items())
        totals.update(dict(**count,turns=receipt['counts']['turns']))
        receipts[game]=sha(p)
    assert dict(totals)==inventory['counts'] and totals['turns']==69024
    return dict(passed=True,games=96,counts=dict(totals),receipts_sha256=receipts,
                verified_outputs=288,inventory_sha256=sha(out/'inventory.json'))


def turn_summary(records):
    counts=Counter();components=Counter();reservations=Counter()
    for r in records:
        counts['turns']+=1
        for name in ('zero_task_turn','factual_zero_task_turn','shadow_zero_task_turn','path_length_different'):
            counts[name]+=bool(r[name])
        counts['action_mismatch']+=not r['action_equal']
        for name in ('first_choice_difference','first_context_difference','first_candidate_difference','first_reservation_difference'):
            counts[name]+=r[name] is not None
        for name in ('matched_context_calls','aligned_choice_differences','aligned_different_choice_same_unit_request'):
            counts[name]+=r[name]
        components.update({k:int(v) for k,v in r['action_components_differ'].items()})
        reservations.update({k:int(v) for k,v in r['final_reservations_differ'].items()})
    n=counts['turns']
    return dict(counts=dict(counts),action_component_different_turns=dict(components),
        reservation_different_turns=dict(reservations),turn_denominator=n,
        action_mismatch_rate=counts['action_mismatch']/n if n else None,
        conditional_choice_difference_rate=(counts['aligned_choice_differences']/counts['matched_context_calls']
            if counts['matched_context_calls'] else None),
        boundary='Submitted requests and planned reservations only; no shadow execution or reward-causal attribution')


def finish(out,cfg,deadline):
    out=Path(out);panel=ROOT/cfg['panel_root'];source=read(panel/'ledger.json')
    assert len(source)==96 and len({r['game'] for r in source})==96
    coverage=validate_outputs(out,source,deadline)
    save(out/'aggregation-coverage.json',coverage)
    # Stream compact records; only numeric-key counts are retained in memory.
    overall_acc=SummaryAccumulator();key_seats={};key_seeds={};precision={}
    for r in stream(out,'compact',source,deadline):
        overall_acc.add(r)
        key_seats[r['key64']]=key_seats.get(r['key64'],0)|(1<<r['side'])
        key_seeds.setdefault(r['key64'],set()).add(r['seed'])
        precision.setdefault(r['key32'],set()).add(r['key64'])
    overall=overall_acc.result()
    conflicts={k for k,v in overall_acc.unique['key64'].items() if len(v['labels'])>1}
    del overall_acc
    tables={}
    filters=[('origin_opponent/'+a+'/'+o,lambda r,a=a,o=o:r['origin']==a and r['opponent']==o)
             for a in cfg['origins'] for o in cfg['opponents']]
    filters += [('seed/'+str(s),lambda r,s=s:r['seed']==s) for s in cfg['seeds']]
    filters += [('seat/'+str(s),lambda r,s=s:r['side']==s) for s in cfg['seats']]
    filters += [('time_third/'+str(t),lambda r,t=t:r['time_third']==t) for t in range(3)]
    filters += [('teacher_op/'+op,lambda r,op=op:r['teacher_op']==op)
                for op in ('DIG','FERTILIZE','HARVEST','PLANT','WATER')]
    for name,predicate in filters:
        tables[name]=summarize(r for r in stream(out,'compact',source,deadline) if predicate(r))
        print('Aggregated',name,flush=True)
    all_turns=list(stream(out,'turns',source,deadline));assert len(all_turns)==69024
    turns=turn_summary(all_turns)
    turns_by_stratum={a+'/'+o:turn_summary(r for r in all_turns if r['origin']==a and r['opponent']==o)
                     for a in cfg['origins'] for o in cfg['opponents']}
    turns_by_seed={str(s):turn_summary(r for r in all_turns if r['seed']==s) for s in cfg['seeds']}
    # Eligibility and order fixed before outcomes; never rank by profit loss.
    candidates={}
    def offer(row,kind,decision=0):
        key=(*stratum(row),kind);m=meta(row,kind,decision)
        bucket=candidates.setdefault(key,[])
        bucket.append(m);bucket.sort(key=lambda x:rank(x,cfg))
        # First two distinct turns suffice for the second distinct witness rule.
        seen=set();kept=[]
        for item in bucket:
            identity=item['game'],item['step']
            if identity not in seen:kept.append(item);seen.add(identity)
            if len(kept)==2:break
        candidates[key]=kept
    for r in stream(out,'compact',source,deadline):
        if r['alias64'] or r['label_disagrees_with_training64'] or r['key64'] in conflicts:
            offer(r,'deployment64_alias_or_label_conflict',r['decision'])
    for r in all_turns:
        if not r['action_equal'] and r['first_choice_difference'] is not None:
            offer(r,'conditional_difference_with_changed_action',r['first_choice_difference']['factual_index'])
        if r['first_context_difference'] is not None and (
                r['first_reservation_difference'] is not None or r['first_candidate_difference'] is not None):
            offer(r,'subsequent_context_divergence',r['first_context_difference']['factual_index'])
    selected=[]
    for o in cfg['opponents']:
        for a in cfg['origins']:
            first=(candidates.get((o,a,'deployment64_alias_or_label_conflict')) or
                   candidates.get((o,a,'conditional_difference_with_changed_action')) or [])
            if first:selected.append(first[0])
            for second in candidates.get((o,a,'subsequent_context_divergence'),[]):
                if not first or (second['game'],second['step'])!=(first[0]['game'],first[0]['step']):
                    selected.append(second);break
    assert len(selected)<=12
    witness_dir=out/'witnesses';witness_dir.mkdir()
    lookup={r['game']:r for r in source};manifest=[]
    for number,item in enumerate(selected,1):
        check_time(deadline);row=lookup[item['game']];step=item['step'];side=item['side']
        rp=panel/'games'/row['game']/'replay.json.gz';tp=panel/'traces'/(row['game']+'.jsonl.gz')
        sp=out/'shadows'/(row['game']+'.jsonl.gz')
        assert sha(rp)==row['replay_gzip_sha256'] and sha(tp)==row['trace_sha256']
        with gzip.open(rp,'rt',encoding='utf-8') as f:replay=json.load(f)
        fact=read_turn(tp,step);shadow=read_turn(sp,step)
        case=dict(**item,observation=observation(replay['steps'][step],side),configuration=replay['configuration'],
                  factual_action=replay['steps'][step+1][side]['action'],factual_trace=fact,
                  shadow_action=shadow['action'],shadow_trace=shadow['trace'],
                  source_replay_sha256=sha(rp),source_trace_sha256=sha(tp),shadow_file_sha256=sha(sp),
                  shadow_executed=False,causal_profit_effect='unknown')
        path=witness_dir/('case-'+str(number).zfill(2)+'.json');save(path,case)
        manifest.append(dict(**item,case=path.relative_to(out).as_posix(),sha256=sha(path),
            source_replay_sha256=sha(rp),source_trace_sha256=sha(tp),shadow_file_sha256=sha(sp),
            effect='Local diagnostic witness only; shadow never executed and no causal profit estimate'))
    save(out/'witness-manifest.json',dict(cases=manifest,count=len(manifest),policy_calls=0,official_transitions=0))
    seed_rates=[tables['seed/'+str(s)]['raw_weighted']['rates']['candidate_mismatch'] for s in cfg['seeds']]
    result=dict(passed=True,exploratory=True,original_statistical_admission_passed=False,
        overall=overall,by_slice=tables,turns=turns,turns_by_origin_opponent=turns_by_stratum,turns_by_seed=turns_by_seed,
        equal_seed_candidate_mismatch_mean=sum(seed_rates)/8 if all(v is not None for v in seed_rates) else None,
        observed_seed_clusters=8,independent_new_samples=0,
        numeric_groups_shared_across_seats=sum(v==3 for v in key_seats.values()),
        numeric_groups_shared_across_seeds=sum(len(v)>1 for v in key_seeds.values()),
        float32_groups_with_multiple_float64_inputs=sum(len(v)>1 for v in precision.values()),
        deployment64_conflicting_label_groups=len(conflicts),witnesses=len(manifest),
        aggregation_coverage_sha256=sha(out/'aggregation-coverage.json'),
        intervention_decision='Pending evidence review; these descriptive results alone do not justify fitting or policy edits',
        training_runs=0,new_games=0,official_transitions=0,
        limitations=['Consumed panel only; no new confirmatory inference.',
                    'Teacher preference and logit gaps are not reward regret.',
                    'Numeric novelty and first action divergence do not identify causes of terminal profit loss.',
                    'Structured context hashes canonicalize mapping keys; explicit item-order arrays and numeric candidate order remain represented.'])
    save(out/'results.json',result)
    return dict(passed=True,results_sha256=sha(out/'results.json'),witness_manifest_sha256=sha(out/'witness-manifest.json'),
                witnesses=len(manifest),aggregate_policy_calls=0,official_transitions=0)
