"""Register and execute six saved student traces once under one bounded clock."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
PUBLIC = ROOT.parent / 'publication/ai-game-lab'
PREVIOUS = ROOT / 'artifacts/task-bc-stratified-report-repair-v1-20261025'
SOURCE = ROOT / 'artifacts/task-bc-holdout-v1-20261023'
INVENTORY = ROOT / 'artifacts/task-bc-error-diagnostics-v1-20261023/inventory.json'
PACKAGE = Path(__file__).resolve().parent
OPS = ('DIG', 'FERTILIZE', 'HARVEST', 'PLANT', 'WATER')
STAGE = 'task-bc-saved-student-decisions-v1'


def require(ok, message):
    if not ok: raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1048576), b''): h.update(chunk)
    return h.hexdigest()


def write_new(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False); stream.write('\n')


def stat(path):
    value = Path(path).stat(); return dict(size=value.st_size, mtime_ns=value.st_mtime_ns)


def git(*args):
    return subprocess.check_output(['git', *args], cwd=PUBLIC)


def official():
    package = Path(next(iter(importlib.util.find_spec('kaggle_environments').submodule_search_locations)))
    return dict(rules=sha(package / 'envs/kaggriculture/kaggriculture.py'), loader=sha(package / 'agent.py'))


def scratch_helper():
    # The user explicitly names this standalone ownership utility. No workflow
    # registration, router, hooks, monitoring or updates are imported or enabled.
    return Path(os.environ['CODEX_HOME']) / 'skills/agent-workflow/scripts/artifacts.py'


def consumed_receipt(raw, game, keys):
    """Return known counts or an explicit unknown; never infer zero actual work."""
    try:
        row=json.loads(raw)
        require(row['schema']=='saved-student-extraction-v1' and row['game']==game,'Receipt identity differs')
        values=row['counts']; unknown=row['unknown_accounting']
        require(type(values) is dict and set(values)==set(keys) and
                all(type(v) is int and v>=0 for v in values.values()),'Invalid consumption counts')
        require(type(unknown) is list and all(type(v) is dict for v in unknown),'Invalid unknown accounting')
        return values,unknown
    except Exception as error:
        return dict.fromkeys(keys,0),[dict(game=game,reason='terminal_receipt_unreadable_or_incomplete_consumption_unknown',error_type=type(error).__name__)]


def control_regressions(keys):
    examples=[b'{"counts":',json.dumps(dict(schema='saved-student-extraction-v1',game='fixture',
        counts={k:(False if i==0 else 0) for i,k in enumerate(keys)},unknown_accounting=[])).encode()]
    for raw in examples:
        counts,unknown=consumed_receipt(raw,'fixture',keys)
        require(unknown and not any(counts.values()),'Unverified receipt consumption was treated as complete')
    return dict(passed=True,cases=2,kind='in-memory terminal receipt failures',source_reads=0,model_calls=0)


def prepare(out):
    require(not out.exists(), 'Preserve any previous registration or attempt')
    require(not git('status', '--porcelain').strip(), 'Start from the clean accepted checkout')
    decision = read(PREVIOUS / 'root-continuation-decision.json')
    require(decision['status'] == 'planned_not_started', 'Only the unexecuted approved scope is accepted')
    require(read(PREVIOUS / 'root-evidence-acceptance.json')['concrete_report_defect_closed'], 'Previous defect not closed')
    old_cfg = read(SOURCE / 'registration.json')
    require(old_cfg['test_seeds'][0] == decision['fixed_selection']['seed'] == 1714414373, 'Fixed seed changed')
    inventory = {row['game']:row for row in read(INVENTORY)['games']}
    metadata = [SOURCE / 'registration.json', INVENTORY]
    groups = []
    for item in decision['fixed_selection']['selected_games']:
        game = item['game']; row_path = SOURCE / 'job-results' / (game + '.row.json')
        audit_path = SOURCE / 'job-results' / (game + '.audit.json')
        row, audit, inv = read(row_path), read(audit_path), inventory[game]
        require(row['policy'] == 'trained' and row['seed'] == 1714414373 and row['passed'] and audit['passed'], 'Wrong historical arm')
        require(row['trace_sha256'] == audit['trace_sha256'] == inv['trace_sha256'] == item['trace_sha256'], 'Source hash chain differs')
        require(inv['counts'] == dict(turns=item['turns'], factual_groups=item['task_callbacks'],
            factual_candidates=item['task_candidates']), 'Inventory count differs')
        require(audit['counts']['task_calls'] == item['task_callbacks'] and
                  audit['counts']['crop_calls'] == item['crop_records'], 'Original audit count differs')
        path = SOURCE / 'traces' / (game + '.jsonl.gz')
        require(path.stat().st_size == item['compressed_bytes'], 'Source size changed')
        groups.append(dict(item, seed=row['seed'], side=row['side'], opponent=row['opponent'],
            source_path=path.relative_to(ROOT).as_posix(), source_stat=stat(path),
            task_checkpoint_sha256=old_cfg['task_checkpoint_sha256'],
            crop_checkpoint_sha256=old_cfg['crop_checkpoint_sha256'],
            expected_conditional_mismatches=audit['counts']['different_from_conditional_teacher'],
            expected_selected_ops={op:audit['counts'].get('selected_' + op, 0) for op in OPS}))
        metadata.extend((row_path, audit_path))
    require([(g['opponent'],g['side']) for g in groups] == [(o,s) for o in ('v1','market','round2') for s in (0,1)], 'Selection/order differs')
    out.mkdir(parents=True)
    for name in ('next-stage-advice.json','root-continuation-decision.json'):
        (out / name).write_bytes((PREVIOUS / name).read_bytes())
    history = read(PREVIOUS / 'protection-before.json')['stat_before'].copy()
    for path in PREVIOUS.rglob('*'):
        if path.is_file() and '__pycache__' not in path.parts: history[path.relative_to(ROOT).as_posix()] = stat(path)
    for path in (ROOT / 'TASK_BC_STRATIFIED_REPORT_REPAIR_V1.zh-CN.md',):
        history[path.relative_to(ROOT).as_posix()] = stat(path)
    require(all((ROOT/rel).is_file() and stat(ROOT/rel)==value for rel,value in history.items()),
            'Inherited historical stat changed before registration')
    write_new(out / 'protection-before.json', dict(stat_before=history,
        inherited_protection_sha256=sha(PREVIOUS / 'protection-before.json'),
        method='Inherited historical byte hashes, before/after stat. Only source-code/metadata allowlists freshly hashed; six raw traces hash once inside charged extraction. Models stat only, never read.'))
    source_caps = dict(source_trace_files=6, compressed_sha_passes=6, gzip_json_passes=6,
        compressed_bytes_hashed=40660889, compressed_bytes_for_decode=40660889,
        compressed_source_bytes_read=81321778, uncompressed_bytes=1073741824,
        physical_lines=4314, task_callback_objects=12232, candidate_objects=575674,
        candidate_x_arrays=575674, shared_x_arrays=12232, observation_objects=12232,
        configuration_objects=12232, crop_records=920, compact_rows=12232)
    frozen = read(PREVIOUS / 'freeze.json')
    model_paths = [SOURCE / 'model/trained.npz', ROOT / old_cfg['crop_checkpoint']]
    metadata.extend(PREVIOUS / name for name in ('summary.json', 'root-evidence-acceptance.json',
        'evidence-review.json', 'publication-verification.json', 'freeze.json'))
    cfg = dict(schema=STAGE, created_utc=datetime.now(timezone.utc).isoformat(),
        baseline_head=git('rev-parse','HEAD').decode().strip(), selected_games=groups,
        selected_seed_is_consumed_historical=True, independent_holdout=False,
        source_metadata_sha256={p.relative_to(ROOT).as_posix():sha(p) for p in metadata},
        inherited_source_sha256=frozen['runtime_files'], line_ending_mappings=frozen['line_ending_mappings'],
        planning_sha256={name:sha(out/name) for name in ('next-stage-advice.json','root-continuation-decision.json')},
        protection_sha256=sha(out/'protection-before.json'),
        model_protection={p.relative_to(ROOT).as_posix():stat(p) for p in model_paths},
        model_protection_is_stat_not_fresh_byte_proof=True, default_sha256=sha(ROOT/'main.py'),
        scratch_helper_sha256=sha(scratch_helper()),
        official_sha256=official(), engine_commit=old_cfg['engine_commit'], source_limits=source_caps,
        stage_seconds_max=600, primary_seconds_max=450, per_game_seconds_max=60,
        synthetic_seconds_max=30, check_seconds_max=120, cleanup_reserve_seconds=30, per_close_seconds_max=10,
        process_tree_private_commit_bytes=1073741824, max_workers=1, blas_threads=1,
        synthetic_limits=dict(cases=6, physical_lines=24, task_callbacks=48, candidates=96, crop_records=24),
        control_regression_cases=2,
        source_byte_measurement='uncompressed_bytes means payload bytes returned by gzip.readline, not all internal zlib work. A failed decode before EOF marks unreturned buffered bytes unknown. Successful EOF covers the complete returned stream.',
        final_acceptance='summary.json alone is provisional; require execution-acceptance.json bound to its SHA, zero process exit and absence of final-deadline-failure.json. Independent checker consumption must be known.',
        checker_limits=dict(compact_rows=12232, receipt_bytes=2097152, witness_bytes=131072,
            compact_line_bytes=65536, all_decoded_artifact_bytes=1073741824),
        cumulative_scoring_groups=3844, cumulative_scoring_candidates=226982,
        no_model_reads_loads_scores_reconstruction_policy_replay_training_or_games=True,
        local_commit_not_external_timestamp=True,
        clock='One continuous executable clock before synthetic through extraction, independently authored compact checker and cleanup. All executable checker code frozen before data. Human review and prose outside clock cannot add data passes.',
        acceptance=['All focused synthetic checks pass within their separate tiny allowance.',
            'Six exact sources: one hash plus one gzip JSON pass each, full ordered 719 turns including empty-task turns, all old counts/identities exact.',
            'Consume-before-check counters, bounded lines/bytes/time/memory, one independent new compact pass and exact descriptive aggregate comparison.',
            'Earliest conditional mismatch or null for every game; at most six witnesses, overlapping differences retained.',
            'Natural owned tree closure, preserved history/default/models, independent review and privacy checks.'],
        stop='Any failure stops dispatch; no retry/refill/inference fallback or budget extension.',
        limitations=['Teacher label is conditional on factual student reservations, not a complete teacher shadow.',
            'Factual unit_request is the end-of-turn request, not official execution or economic regret.',
            'One consumed seed, correlated games; no CI, significance, general deployment error or causal reward claim.',
            'Independent compact verification is not independent raw trace reproduction.'],
        previous_strict_review_acceptance=False, original_diagnostic_stage_acceptance=False, original_policy_admission=False)
    require(cfg['official_sha256']['rules'] == old_cfg['engine_sha256'], 'Official rule changed')
    write_new(out/'registration.json', cfg)
    return dict(prepared=True, selected_games=6, original_trace_reads=0, model_reads=0)


def freeze(out):
    cfg=read(out/'registration.json'); sources=cfg['inherited_source_sha256'].copy()
    sources['main.py']=cfg['default_sha256']
    for path in PACKAGE.iterdir():
        if path.is_file() and path.suffix in ('.py','.md'): sources[path.relative_to(ROOT).as_posix()]=sha(path)
    published={rel:sha(PUBLIC/'kaggriculture'/rel) for rel in sources}
    for rel,digest in sources.items():
        require(sha(ROOT/rel)==digest,'Source changed: '+rel)
        if published[rel]!=digest:
            require(rel in cfg['line_ending_mappings'] and (ROOT/rel).read_bytes().replace(b'\r\n',b'\n') ==
                    (PUBLIC/'kaggriculture'/rel).read_bytes().replace(b'\r\n',b'\n'),'Unregistered byte mapping')
    write_new(out/'freeze.json',dict(registration_sha256=sha(out/'registration.json'),runtime_files=sources,
        public_source_sha256=published,line_ending_mappings=cfg['line_ending_mappings']))
    return dict(frozen=True,freeze_sha256=sha(out/'freeze.json'),sources=len(sources))


def verify(out):
    cfg=read(out/'registration.json'); frozen=read(out/'freeze.json')
    require(sha(out/'registration.json')==frozen['registration_sha256'] and
            sha(out/'protection-before.json')==cfg['protection_sha256'],'Registration/protection changed')
    for rel,digest in cfg['planning_sha256'].items(): require(sha(out/rel)==digest,'Planning changed')
    for rel,digest in frozen['runtime_files'].items(): require(sha(ROOT/rel)==digest,'Frozen source changed')
    for rel,digest in cfg['source_metadata_sha256'].items(): require(sha(ROOT/rel)==digest,'Source metadata changed')
    for rel,value in cfg['model_protection'].items(): require(stat(ROOT/rel)==value,'Protected model stat changed')
    require(sha(scratch_helper())==cfg['scratch_helper_sha256'],'Scratch ownership helper changed')
    for group in cfg['selected_games']: require(stat(ROOT/group['source_path'])==group['source_stat'],'Raw source stat changed')
    require(official()==cfg['official_sha256'],'Official bytes changed')
    return cfg


def worker(out,key):
    cfg=verify(out); request=read(out/'requests'/(key+'.json')); deadline=request['deadline_monotonic']
    require(request['freeze_sha256']==sha(out/'freeze.json') and time.monotonic()<deadline,'Request changed or expired')
    write_new(out/'worker-started'/(key+'.json'),dict(request_sha256=sha(out/'requests'/(key+'.json'))))
    if key=='synthetic':
        from .verify_extract import run_regressions
        controls=control_regressions(cfg['source_limits'])
        require(controls['cases']==cfg['control_regression_cases'],'Control cases changed')
        write_new(out/'control-regressions.json',controls)
        result=run_regressions(out/'synthetic',deadline,helper_sha256=cfg['scratch_helper_sha256'])
        limits=cfg['synthetic_limits']; count=result['counts']
        require(len(result['cases'])==limits['cases'] and count['physical_lines']<=limits['physical_lines'] and
                count['task_callback_objects']<=limits['task_callbacks'] and
                count['candidate_objects']<=limits['candidates'] and count['crop_records']<=limits['crop_records'],
                'Focused synthetic allowance exceeded')
    elif key=='independent-check':
        from .check import check
        result=check(out,dict(cfg,primary_receipts_sha256=request['primary_receipts_sha256']),deadline)
    else:
        from .extract import extract_game
        group=next(g for g in cfg['selected_games'] if g['game']==key)
        result=extract_game(ROOT/group['source_path'],out/'extracted'/key,group,request['remaining_source_budget'],deadline)
    require(result['passed'] is True,'Worker result failed')
    verify(out); require(time.monotonic()<deadline,'Worker deadline exceeded')
    write_new(out/'worker-complete'/(key+'.json'),dict(passed=True,result=result,
        request_sha256=sha(out/'requests'/(key+'.json')),freeze_sha256=sha(out/'freeze.json')))


def run(out):
    cfg=verify(out); frozen=read(out/'freeze.json'); pre=read(out/'pre-review.json'); commit=read(out/'pre-execution-commit.json')
    require(pre['passed'] is True and pre['freeze_sha256']==sha(out/'freeze.json') and
            commit['freeze_sha256']==sha(out/'freeze.json') and commit['commit']==git('rev-parse','HEAD').decode().strip(),'Exact reviewed commit required')
    expected={'kaggriculture/'+rel:digest for rel,digest in frozen['public_source_sha256'].items()}
    for name in ('registration.json','freeze.json','pre-review.json','next-stage-advice.json','root-continuation-decision.json'):
        expected['review/evidence/'+out.name+'/'+name]=sha(out/name)
    require(commit['git_files_sha256']==expected,'Incomplete committed source/metadata set')
    for rel,digest in expected.items(): require(hashlib.sha256(git('show',commit['commit']+':'+rel)).hexdigest()==digest,'Committed bytes differ')
    started=time.monotonic(); deadline=started+cfg['stage_seconds_max']; primary_deadline=started+cfg['primary_seconds_max']
    write_new(out/'started.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),deadline_monotonic=deadline,freeze_sha256=sha(out/'freeze.json')))
    counts=dict.fromkeys(cfg['source_limits'],0); attempts=[]; receipts={}; unknown=[]; failure=None
    check_consumption=dict(started=False,known=False)
    from learning.task_bc_diagnostics_capacity_exit_v1.windows_tree import run_owned
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONUTF8='1',PYTHONIOENCODING='utf-8',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')

    def dispatch(key,seconds,phase):
        work=min(time.monotonic()+seconds,phase-cfg['per_close_seconds_max'])
        require(work>time.monotonic(),'No remaining work budget')
        request=dict(freeze_sha256=sha(out/'freeze.json'),deadline_monotonic=work,
            remaining_source_budget={k:v-counts[k] for k,v in cfg['source_limits'].items()},primary_receipts_sha256=receipts.copy())
        write_new(out/'requests'/(key+'.json'),request)
        attempt=dict(key=key,passed=False,lifecycle={}); attempts.append(attempt); local={}
        print('saved-student:',key,flush=True)
        try:
            (out/'logs').mkdir(exist_ok=True)
            with (out/'logs'/(key+'.log')).open('x',encoding='utf-8') as log:
                code=run_owned([sys.executable,'-B','-m','learning.task_bc_saved_student_decisions_v1.experiment','worker','--out',str(out),'--key',key],cwd=ROOT,env=env,stream=log,timeout=work-time.monotonic(),phase_deadline=phase,lifecycle=attempt['lifecycle'],local_lifecycle=local,memory_limit_bytes=cfg['process_tree_private_commit_bytes'])
            life=attempt['lifecycle']
            require(code==0 and life['tree_closed'] and life['active_processes_after_close']==0 and life['natural_tree_exit'] and not life['termination_requested'] and life['job_handle_closed'] and life['all_process_handles_closed'],'Worker lifecycle rejected')
            result=read(out/'worker-complete'/(key+'.json'))
            require(result['passed'] and result['request_sha256']==sha(out/'requests'/(key+'.json')) and result['freeze_sha256']==sha(out/'freeze.json'),'Completion binding differs')
            attempt['passed']=True
            return result['result']
        finally:
            write_new(out/'local-lifecycle'/(key+'.json'),local)
            if key not in ('synthetic','independent-check'):
                path=out/'extracted'/key/'receipt.json'
                try:
                    require(path.stat().st_size<=2097152,'Terminal receipt oversized')
                    raw=path.read_bytes()
                except Exception: raw=None
                consumed,unmeasured=consumed_receipt(raw,key,counts)
                for field in counts: counts[field]+=consumed[field]
                unknown.extend(unmeasured)
                if attempt['passed']:
                    require(not unmeasured,'Accepted worker lacks complete consumption evidence')
                    receipts[key]=hashlib.sha256(raw).hexdigest()
            elif key=='independent-check':
                check_consumption.update(started=True,known=False)
                terminal=out/'check'/('failure.json' if (out/'check/failure.json').exists() else 'complete.json')
                try:
                    value=read(terminal)
                    fields=('compact_rows_reserved','compact_rows_read','compact_rows_decoded','compact_bytes_read')
                    require(all(type(value[k]) is int and value[k]>=0 for k in fields),'Incomplete checker counters')
                    check_consumption.update(known=True,terminal_sha256=sha(terminal),**{k:value[k] for k in fields})
                except Exception:
                    check_consumption['reason']='checker_terminal_receipt_missing_or_incomplete_consumption_unknown'
            write_new(out/'attempts'/(key+'.json'),attempt)
    try:
        synth=dispatch('synthetic',cfg['synthetic_seconds_max'],primary_deadline)
        require(synth['passed'],'Synthetic checks failed')
        for group in cfg['selected_games']:
            dispatch(group['game'],cfg['per_game_seconds_max'],primary_deadline)
            require(all(counts[k]<=v for k,v in cfg['source_limits'].items()),'Cumulative source limit exceeded')
        require(not unknown and all(counts[k]==v for k,v in cfg['source_limits'].items() if k!='uncompressed_bytes'),'Incomplete fixed source consumption')
        require(time.monotonic()<primary_deadline,'Primary bookkeeping exceeded its original deadline')
        # Its phase end is 580: work <=570 and the same original close <=10;
        # the outer 600-second deadline still includes final verification.
        dispatch('independent-check',cfg['check_seconds_max'],deadline-cfg['cleanup_reserve_seconds']+cfg['per_close_seconds_max'])
        require(check_consumption['known'],'Independent checker consumption is unknown')
        verify(out)
    except BaseException as error:
        failure=dict(kind=type(error).__name__,retry=False)
        raise
    finally:
        history=read(out/'protection-before.json')['stat_before']
        changed=[rel for rel,value in history.items() if not (ROOT/rel).is_file() or stat(ROOT/rel)!=value]
        write_new(out/'history-stat-check.json',dict(passed=not changed,checked_files=len(history),changed_files=changed,method='stat plus separately fixed metadata/source byte checks; models stat only'))
        if time.monotonic()>deadline and failure is None: failure=dict(kind='TimeoutError',phase='final-verification',retry=False)
        write_new(out/'summary.json',dict(passed=failure is None and not changed,attempts=len(attempts),accepted_jobs=sum(a['passed'] for a in attempts),failure=failure,retries=0,
            elapsed_seconds=time.monotonic()-started,source_counts=counts,unknown_accounting=unknown,source_receipts_sha256=receipts,
            source_counts_are_complete=not unknown,independent_check_consumption=check_consumption,
            freeze_sha256=sha(out/'freeze.json'),history_stat_check_sha256=sha(out/'history-stat-check.json'),
            cumulative_scoring_groups=3844,cumulative_scoring_candidates=226982,model_reads=0,model_loads=0,model_calls=0,candidate_reconstructions=0,
            policy_calls=0,raw_replay_reads=0,training_runs=0,new_games=0,official_transitions=0,
            original_diagnostic_stage_acceptance=False,original_policy_admission=False,previous_strict_review_acceptance=False))
        require(not changed and failure is None,'Stopped with preserved failure or history change')
    result=read(out/'summary.json')
    def final_gate():
        if time.monotonic()>=deadline:
            write_new(out/'final-deadline-failure.json',dict(passed=False,summary_is_not_final_acceptance=True,
                elapsed_seconds=time.monotonic()-started,reason='Final persistence/read crossed original deadline'))
            raise TimeoutError('Final acceptance exceeded original phase deadline')
    final_gate()
    write_new(out/'execution-acceptance.json',dict(passed=result['passed'],summary_sha256=sha(out/'summary.json'),
        elapsed_seconds=time.monotonic()-started,requires_zero_process_exit_and_no_final_deadline_failure=True))
    final_gate()
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['prepare','freeze','run','worker']);parser.add_argument('--out',required=True);parser.add_argument('--key')
    args=parser.parse_args();out=Path(args.out).resolve()
    result=worker(out,args.key) if args.command=='worker' else globals()[args.command](out)
    if result is not None: print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__': main()
