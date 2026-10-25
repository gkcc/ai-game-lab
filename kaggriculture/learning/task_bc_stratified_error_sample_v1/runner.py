"""Execute the single registered stratified diagnostic, with no resume or retry."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback

from learning.bc_crop.agent import ROOT, sha
from learning.bc_crop.experiment import read, save
from learning.task_bc_diagnostics_capacity_v1.runner import require, check_time, new_json, resources, key_name
from learning.task_bc_diagnostics_capacity_exit_v1.runner import budget_rows as evaluation_rows
from .experiment import OLD, verify, ready, stat


def budget_rows(out,cfg):
    rows=evaluation_rows(out,cfg)
    for row in rows:
        require(all(type(row[k]) is int and row[k]>=0 for k in ('decoded_groups','decoded_candidates')),
                'Invalid derived decode reservation')
    require(sum(r['decoded_groups'] for r in rows)<=cfg['max_derived_decoded_groups'] and
            sum(r['decoded_candidates'] for r in rows)<=cfg['max_derived_decoded_candidates'],
            'Derived input decode reservation exceeded')
    return rows


def reserve(out,cfg,key,groups,candidates,decoded_groups,decoded_candidates):
    rows=budget_rows(out,cfg); key_name(key)
    require(not any(r['key']==key for r in rows),'No job retry is permitted')
    amounts=dict(groups=groups,candidates=candidates,decoded_groups=decoded_groups,decoded_candidates=decoded_candidates)
    caps=dict(groups=cfg['new_evaluated_groups_max'],candidates=cfg['new_evaluated_candidates_max'],
              decoded_groups=cfg['max_derived_decoded_groups'],decoded_candidates=cfg['max_derived_decoded_candidates'])
    for field,value in amounts.items():
        require(type(value) is int and value>=0 and sum(r[field] for r in rows)+value<=caps[field],
                'Allocated stage budget exceeded: '+field)
    require(sum(r['groups'] for r in rows)+groups+cfg['old_budget_groups']<=cfg['shared_evaluated_groups_max'] and
            sum(r['candidates'] for r in rows)+candidates+cfg['old_budget_candidates']<=cfg['shared_evaluated_candidates_max'],
            'Shared evaluation budget exceeded')
    path=Path(out)/'budget-reservations.jsonl'
    predecessor=cfg['old_artifacts_sha256']['budget-reservations.jsonl']
    previous=hashlib.sha256(path.read_bytes().splitlines()[-1]).hexdigest() if rows else predecessor
    row=dict(sequence=len(rows),previous_line_sha256=previous,predecessor_ledger_sha256=predecessor,
        carry_groups=cfg['old_budget_groups'],carry_candidates=cfg['old_budget_candidates'],key=key,
        **amounts,reserved_before_dispatch=True,refunded=False,created_utc=datetime.now(timezone.utc).isoformat())
    raw=json.dumps(row,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode('utf-8')
    with path.open('ab') as stream: stream.write(raw+b'\n'); stream.flush(); os.fsync(stream.fileno())
    return hashlib.sha256(raw).hexdigest()


def merge_manifest(out,cfg,c,deadline):
    """Commit only already extracted records, without reading input arrays again."""
    from .pipeline import SAMPLE_SCHEMA
    groups=[]; counters=dict.fromkeys(cfg['source_scan_limits'],0)
    expected=[g['game'] for g in cfg['selected_groups']]
    require(c['collection_order']==expected and set(c['collection_receipts_sha256'])==set(expected),
            'Incomplete or changed collection order')
    for registered in cfg['selected_groups']:
        check_time(deadline); game=registered['game']; path=out/'collection'/game/'receipt.json'
        require(sha(path)==c['collection_receipts_sha256'][game],'Collection receipt changed')
        row=read(path); require(row['passed'] and row['game']==game,'Collection did not complete')
        group=row['group']
        require({k:group[k] for k in ('game','split','seed','side','opponent','source_sha256')}==
                {k:registered[k] for k in ('game','split','seed','side','opponent','source_sha256')},'Collected source mismatch')
        require(row['source_files_hashed']==row['source_streams_completed']==3 and row['original_stream_passes']==1,
                'Original streams were not each hashed and scanned once')
        require(group['source_population_rows']==registered['source_rows'],'Source population mismatch')
        require(sum(cell['population'] for cell in group['strata'])==registered['source_rows'] and
                sum(cell['selected_count'] for cell in group['strata'])==group['rows'],'Stratum accounting mismatch')
        for field in counters: counters[field]+=row['counters'][field]
        groups.append(group)
    rows=sum(g['rows'] for g in groups); candidates=sum(g['candidates'] for g in groups)
    train=sum(g['rows'] for g in groups if g['split']=='train')
    validation=rows-train
    require(0<rows<=cfg['sample_groups_max'] and candidates<=cfg['sample_candidates_max'] and
            0<train<=cfg['sample_train_groups_max'] and 0<validation<=cfg['sample_validation_groups_max'],
            'Empty or oversized fixed selected population')
    for field,value in counters.items(): require(value<=cfg['source_scan_limits'][field],'Source scan cap: '+field)
    for field in ('compressed_source_bytes_hashed','index_metadata_rows','label_rows','input_physical_lines_read',
                  'all_original_gzip_physical_lines'):
        require(counters[field]==cfg['source_scan_limits'][field],'Incomplete original scan: '+field)
    require(counters['input_json_rows_decoded']==rows and counters['input_json_candidates_decoded']==candidates,
            'Selected extraction counts mismatch')
    manifest=dict(schema=SAMPLE_SCHEMA,sample_only=True,historical_reuse_not_new_holdout=True,groups=groups,
        source_manifest_sha256=cfg['source_metadata_sha256']['dataset-manifest.json'],
        registration_sha256=sha(out/'registration.json'),rows=rows,candidates=candidates,
        train_rows=train,validation_rows=validation,
        train_candidates=sum(g['candidates'] for g in groups if g['split']=='train'),
        source_scan_counters=counters,collection_receipts_sha256=c['collection_receipts_sha256'])
    new_json(out/'sample-manifest.json',manifest)
    new_json(out/'sample-freeze.json',dict(sample_manifest_sha256=sha(out/'sample-manifest.json'),
        experiment_freeze_sha256=sha(out/'freeze.json'),collection_receipts_sha256=c['collection_receipts_sha256'],
        source_membership_fixed_before_first_model_score=True,selection='Registered deterministic per-cell hashes; no replacement or deduplication.'))
    return dict(passed=True,sample_manifest_sha256=sha(out/'sample-manifest.json'),
        sample_freeze_sha256=sha(out/'sample-freeze.json'),rows=rows,candidates=candidates,
        train_rows=train,validation_rows=validation,train_candidates=manifest['train_candidates'],
        source_scan_counters=counters,model_calls=0)


def worker(out,key):
    out=Path(out).resolve(); cfg=verify(out); key_name(key)
    request_path=out/'run/job-inputs'/(key+'.json'); request=read(request_path); c=request['pipeline_config']
    require(request['key']==key and request['freeze_sha256']==sha(out/'freeze.json'),'Worker request/freeze mismatch')
    rows=[r for r in budget_rows(out,cfg) if r['key']==key]; require(len(rows)==1,'Unique reservation absent')
    raw=json.dumps(rows[0],separators=(',',':'),ensure_ascii=False,allow_nan=False).encode('utf-8')
    require(hashlib.sha256(raw).hexdigest()==request['budget_reservation_sha256'],'Reservation changed')
    for stored,field in (('groups','max_evaluated_groups'),('candidates','max_evaluated_candidates'),
                         ('decoded_groups','max_decoded_groups'),('decoded_candidates','max_decoded_candidates')):
        require(rows[0][stored]==c[field],'Worker reservation limit mismatch')
    marker=read(out/'started.json'); deadline=request['deadline_monotonic']
    require(deadline<=marker['deadline_monotonic']-cfg['close_seconds_max_including_grace'] and
            deadline-time.monotonic()<=cfg['numeric_job_work_seconds_max'],'Worker deadline allocation mismatch')
    check_time(deadline); resources(out,cfg)
    new_json(out/'run/job-started'/(key+'.json'),dict(request_sha256=sha(request_path)))
    if key=='synthetic':
        from .verify_sampling import run_regressions as sampling_checks
        from .verify_pipeline import run_regressions as pipeline_checks
        a=sampling_checks(out/'run/synthetic/sampling',deadline=deadline)
        require(a['passed'],'Sampler checks failed')
        b=pipeline_checks(out/'run/synthetic/pipeline',deadline=deadline)
        require(b['passed'],'Sparse pipeline checks failed')
        cases=len(a['cases'])+len(b['cases'])
        groups=a['synthetic_numeric_processing_groups']+b['synthetic_numeric_processing_groups']
        candidates=a['synthetic_numeric_processing_candidates']+b['synthetic_numeric_processing_candidates']
        require(cases==cfg['synthetic_limits']['required_cases']<=cfg['synthetic_limits']['cases'] and groups<=cfg['synthetic_limits']['rows'] and
                candidates<=cfg['synthetic_limits']['candidates'],'Shared synthetic ceiling exceeded')
        result=dict(passed=True,cases=cases,numeric_processing_groups=groups,numeric_processing_candidates=candidates,
            sampling=a,pipeline=b,model_calls=0,original_streams_read=0)
    elif key.startswith('collect-'):
        from .sampling import collect_group
        game=key[len('collect-'):]; selected=[g for g in cfg['selected_groups'] if g['game']==game]
        require(len(selected)==1,'Unregistered original game')
        options=dict(cfg,scan_limits=c['scan_limits'],sample_rows_remaining=c['sample_rows_remaining'],
                     sample_candidates_remaining=c['sample_candidates_remaining'])
        result=collect_group(out,options,deadline,selected[0])
    elif key=='manifest': result=merge_manifest(out,cfg,c,deadline)
    else:
        from . import pipeline
        require(sha(out/'sample-manifest.json')==c['sample_manifest_sha256'] and
                sha(out/'sample-freeze.json')==c['sample_freeze_sha256'],'Sample membership not frozen before scoring')
        sample_freeze=read(out/'sample-freeze.json')
        require(sample_freeze['sample_manifest_sha256']==c['sample_manifest_sha256'] and
                sample_freeze['experiment_freeze_sha256']==request['freeze_sha256'],'Sample/source freeze mismatch')
        c=dict(c,sample_root=str(out),registered_groups=cfg['registered_groups'])
        target=out/'run/candidate'
        if key=='aggregate': result=pipeline.aggregate(target,c,deadline)
        elif key=='compare':
            result=pipeline.compare_reference(target,out/'run/reference',c,deadline)
            new_json(target/'comparison.json',result)
        elif key=='report':
            from .reporting import report
            result=report(out,dict(cfg,**c),deadline)
        else:
            from learning.task_bc_v1.model import TaskNetwork
            model=TaskNetwork(ROOT/cfg['task_model'],cfg['task_model_sha256'])
            if key=='reference': result=pipeline.reference(out/'run/reference',c,deadline,model)
            elif key=='build': result=pipeline.build_index(target,c,deadline,model)
            elif key.startswith('score-'): result=pipeline.score_shard(target,c,deadline,key[6:],model)
            else: raise ValueError('Unknown registered job')
    envelope=dict(key=key,result=result,request_sha256=sha(request_path),freeze_sha256=request['freeze_sha256'])
    new_json(out/'run/job-results'/(key+'.json'),envelope)
    require(result.get('passed') is True,'Job result rejected')
    if key=='reference' or key.startswith('score-'):
        count=result['counts']
        require(count['evaluated_groups']==count['completed_groups']==rows[0]['groups'] and
                count['evaluated_candidates']==count['completed_candidates']==rows[0]['candidates'],'Evaluation population mismatch')
    if key in ('reference','build') or key.startswith('score-'):
        require(result['counts']['decoded_groups']==rows[0]['decoded_groups'] and
                result['counts']['decoded_candidates']==rows[0]['decoded_candidates'],'Derived decode population mismatch')
    check_time(deadline); verify(out); resources(out,cfg); check_time(deadline)
    new_json(out/'run/job-complete'/(key+'.json'),dict(passed=True,**envelope))


def run(out):
    from learning.task_bc_diagnostics_capacity_exit_v1.windows_tree import run_owned
    out=Path(out).resolve(); cfg=ready(out); resources(out,cfg)
    started=time.monotonic(); deadline=started+cfg['stage_wall_seconds_max']
    marker=out/'started.json'
    new_json(marker,dict(started_utc=datetime.now(timezone.utc).isoformat(),deadline_monotonic=deadline,
        freeze_sha256=sha(out/'freeze.json')))
    attempts=[]; jobs=[]; failure=None; previous=sha(marker)
    scan=dict.fromkeys(cfg['source_scan_limits'],0); selected_rows=selected_candidates=0
    c=dict(reference_freeze_sha256=sha(out/'freeze.json')); collections={}
    environment=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',
        PYTHONDONTWRITEBYTECODE='1',PYTHONUTF8='1',PYTHONIOENCODING='utf-8')

    def persist():
        save(out/'run/attempts.json',attempts); save(out/'run/jobs-ledger.json',jobs)

    def dispatch(key,groups=0,candidates=0,decoded_groups=0,decoded_candidates=0):
        nonlocal previous
        check_time(deadline); verify(out); resources(out,cfg)
        work_deadline=min(time.monotonic()+cfg['numeric_job_work_seconds_max'],deadline-cfg['close_seconds_max_including_grace'])
        check_time(work_deadline)
        reservation=reserve(out,cfg,key,groups,candidates,decoded_groups,decoded_candidates)
        pc=dict(c,freeze_sha256=sha(out/'freeze.json'),model_sha256=cfg['task_model_sha256'],
            previous_receipt_sha256=previous,max_evaluated_groups=groups,max_evaluated_candidates=candidates,
            max_decoded_groups=decoded_groups,max_decoded_candidates=decoded_candidates)
        if key.startswith('collect-'):
            pc.update(scan_limits={k:v-scan[k] for k,v in cfg['source_scan_limits'].items()},
                sample_rows_remaining=cfg['sample_groups_max']-selected_rows,
                sample_candidates_remaining=cfg['sample_candidates_max']-selected_candidates)
        request=dict(key=key,freeze_sha256=pc['freeze_sha256'],deadline_monotonic=work_deadline,
            budget_reservation_sha256=reservation,pipeline_config=pc)
        request_path=out/'run/job-inputs'/(key+'.json'); new_json(request_path,request)
        attempt=dict(key=key,status='started',groups_reserved=groups,candidates_reserved=candidates,
            derived_groups_reserved=decoded_groups,derived_candidates_reserved=decoded_candidates,
            budget_reservation_sha256=reservation,lifecycle={})
        attempts.append(attempt); persist(); print('stratified:',key,flush=True)
        log_path=out/'run/logs'/(key+'.log'); log_path.parent.mkdir(parents=True,exist_ok=True)
        tick=time.monotonic(); local={}
        try:
            with log_path.open('x',encoding='utf-8') as log:
                code=run_owned([sys.executable,'-B','-m','learning.task_bc_stratified_error_sample_v1.runner',
                    'worker','--out',str(out),'--key',key],cwd=ROOT,env=environment,stream=log,
                    timeout=work_deadline-time.monotonic(),phase_deadline=deadline,lifecycle=attempt['lifecycle'],
                    local_lifecycle=local,memory_limit_bytes=cfg['resource_limits']['process_tree_private_commit_bytes'])
            life=attempt['lifecycle']
            require(code==0 and life['tree_closed'] and life['active_processes_after_close']==0 and
                life['natural_tree_exit'] and not life['termination_requested'] and life['containment_established'] and
                life['job_handle_closed'] and life['all_process_handles_closed'],'Worker completion/closure rejected')
            path=out/'run/job-complete'/(key+'.json'); envelope=read(path)
            require(envelope['passed'] and envelope['key']==key and envelope['request_sha256']==sha(request_path) and
                    envelope['freeze_sha256']==request['freeze_sha256'],'Worker completion binding mismatch')
            jobs.append(dict(key=key,receipt_sha256=sha(path),result=envelope['result']))
            previous=sha(path); attempt['status']='passed'; check_time(deadline)
            return envelope['result']
        except BaseException:
            attempt['status']='failed'; raise
        finally:
            path=out/'run/local-lifecycle'/(key+'.json'); new_json(path,local)
            attempt['local_lifecycle_sha256']=sha(path); attempt['elapsed_seconds']=time.monotonic()-tick; persist()

    try:
        dispatch('synthetic')
        for group in cfg['selected_groups']:
            result=dispatch('collect-'+group['game'])
            for field,value in result['counters'].items():
                scan[field]+=value; require(scan[field]<=cfg['source_scan_limits'][field],'Global source scan cap')
            selected_rows+=result['group']['rows']; selected_candidates+=result['group']['candidates']
            collections[group['game']]=sha(out/'collection'/group['game']/'receipt.json')
        c.update(collection_receipts_sha256=collections,collection_order=[g['game'] for g in cfg['selected_groups']])
        sample=dispatch('manifest')
        c.update(sample_manifest_sha256=sample['sample_manifest_sha256'],sample_freeze_sha256=sample['sample_freeze_sha256'])
        ref=dispatch('reference',sample['rows'],sample['candidates'],sample['rows']+sample['train_rows'],
                     sample['candidates']+sample['train_candidates'])
        c['reference_receipt_sha256']=ref['receipt_sha256']
        built=dispatch('build',decoded_groups=sample['train_rows'],decoded_candidates=sample['train_candidates'])
        c.update(support_index_sha256=built['support_index_sha256'],support_index_receipt_sha256=built['support_index_receipt_sha256'])
        shards={}; manifest=read(out/'sample-manifest.json')
        for group in manifest['groups']:
            result=dispatch('score-'+group['game'],group['rows'],group['candidates'],group['rows'],group['candidates'])
            shards[group['game']]=result['receipt_sha256']
        c.update(shard_receipt_sha256=shards,shard_order=[g['game'] for g in manifest['groups']])
        aggregated=dispatch('aggregate'); c['aggregate_receipt_sha256']=aggregated['receipt_sha256']
        compared=dispatch('compare')
        require(compared['exact_compact_rows']==sample['rows'] and compared['exact_candidates']==sample['candidates'] and
                compared['byte_identical_summary'],'Incomplete exact comparison')
        c['comparison_sha256']=sha(out/'run/candidate/comparison.json')
        dispatch('report')
        require(len(jobs)==47 and len(attempts)==47,'Incomplete registered job sequence')
        rows=budget_rows(out,cfg)
        require(sum(r['groups'] for r in rows)==2*sample['rows'] and
                sum(r['candidates'] for r in rows)==2*sample['candidates'],'Two scoring paths did not reconcile')
        resources(out,cfg); verify(out); check_time(deadline)
    except BaseException as error:
        failure=dict(kind=type(error).__name__,retry=False)
        folder=out/'run/logs'; folder.mkdir(parents=True,exist_ok=True)
        (folder/'failure.log').write_text(traceback.format_exc(),encoding='utf-8')
        new_json(out/'failure.json',failure)
    finally:
        persist(); rows=budget_rows(out,cfg)
        # All known collection receipts, including a failed one, contribute
        # actual consumption. A killed job without a receipt remains unknown.
        actual_scan=dict.fromkeys(cfg['source_scan_limits'],0); incomplete_scan_receipts=[]
        for attempt in attempts:
            if not attempt['key'].startswith('collect-'): continue
            game=attempt['key'][len('collect-'):]; path=out/'collection'/game/'receipt.json'
            if path.exists():
                for field,value in read(path)['counters'].items(): actual_scan[field]+=value
            else: incomplete_scan_receipts.append(game)
        protection=read(out/'protection-before.json')
        changed=[rel for rel,info in protection['stat_before'].items() if not (ROOT/rel).exists() or stat(ROOT/rel)!=info]
        new_json(out/'history-stat-check.json',dict(passed=not changed,checked=len(protection['stat_before']),changed=changed,
            inherited_hashes_retained=True,new_full_byte_verification=False,
            method='Before/after stat comparison of inherited historical files; selected original hashes are separately checked during collection.'))
        if changed and failure is None: failure=dict(kind='HistoricalMetadataChanged',retry=False)
        elapsed=time.monotonic()-started
        if elapsed>cfg['stage_wall_seconds_max'] and failure is None: failure=dict(kind='StageDeadlineExceeded',retry=False)
        result=dict(passed=failure is None,sample_only=True,historical_reuse_not_new_holdout=True,
            elapsed_seconds=elapsed,attempts=len(attempts),completed_jobs=len(jobs),failure=failure,retries=0,
            owned_trees_closed=bool(attempts) and all(a['lifecycle'].get('tree_closed',False) for a in attempts),
            successful_workers_forced_termination=sum(a['lifecycle'].get('termination_requested',False) for a in attempts if a['status']=='passed'),
            reserved_real_evaluations=sum(r['groups'] for r in rows),reserved_real_candidates=sum(r['candidates'] for r in rows),
            reserved_derived_decoded_groups=sum(r['decoded_groups'] for r in rows),
            reserved_derived_decoded_candidates=sum(r['decoded_candidates'] for r in rows),
            carry_groups=cfg['old_budget_groups'],carry_candidates=cfg['old_budget_candidates'],
            cumulative_reserved_groups=cfg['old_budget_groups']+sum(r['groups'] for r in rows),
            cumulative_reserved_candidates=cfg['old_budget_candidates']+sum(r['candidates'] for r in rows),
            source_scan_known_counters=actual_scan,source_scan_unfinished_without_receipt=incomplete_scan_receipts,
            budget_ledger_sha256=sha(out/'budget-reservations.jsonl') if rows else None,
            predecessor_ledger_sha256=cfg['old_artifacts_sha256']['budget-reservations.jsonl'],freeze_sha256=sha(out/'freeze.json'),
            history_stat_check_sha256=sha(out/'history-stat-check.json'),training_runs=0,policy_calls=0,new_games=0,
            official_transitions=0,full_diagnostics_completed=False,original_diagnostic_stage_acceptance=False,original_policy_admission=False)
        new_json(out/'summary.json',result)
    print('stratified:','passed' if failure is None else 'stopped; failure preserved',flush=True)
    return failure is None


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('mode',choices=('run','worker'))
    parser.add_argument('--out',type=Path,required=True); parser.add_argument('--key'); args=parser.parse_args()
    if args.mode=='worker': worker(args.out,args.key)
    else: sys.exit(0 if run(args.out) else 1)
