"""One bounded panel of official workers and their primary replay audits."""
import json
from pathlib import Path
import sys
from learning.task_training_job_supervision_v1.job import WindowsAPI
from learning.task_job_empty_exit_v1.sampling import supervise
from learning.task_teacher_choice_eval_v2.campaign import verify_economy
from .experiment import OUT, ROOT, ready, verify_commit, deadline_check, read, sha, write_new
from .games import plans, validate_plan, make_wrapper, validate_receipts, task_identity
from .resources import memory_ready


class CampaignFailure(RuntimeError):
    def __init__(self, primary, receipt=None):
        super().__init__('Panel failed; attempt consumed')
        self.primary_error_kind, self.receipt_error_kind = primary, receipt


def job_ok(result):
    return (isinstance(result,dict) and result.get('passed') is True and result.get('assigned') is True
        and result.get('resumed') is True and result.get('launcher_reaped') is True
        and result.get('job_members_zero') is True and result.get('error_kind') is None
        and result.get('cleanup_error_kind') is None and result.get('stop_reason') is None
        and type(result.get('exit_code')) is int and result['exit_code'] == 0)


def identity(out):
    names=('freeze.json','registration.json','pre-execution-commit.json','games-started.json')
    return {name:sha(out/name) for name in names}


def check_identity(out, identities):
    if not isinstance(identities,dict) or set(identities)!={'freeze.json','registration.json','pre-execution-commit.json','games-started.json'}:
        raise ValueError('Incomplete panel identity')
    if any(sha(out/name)!=value for name,value in identities.items()): raise ValueError('Panel identity changed')


def sanitized(row):
    fields=('game','seed','opponent','side','policy','split','parity_kind','passed','states','action_turns','status',
            'agent_errors','bank','opponent_bank','initial_bank','opponent_initial_bank','net_profit',
            'opponent_net_profit','margin','score')
    return {key:row[key] for key in fields}


def accept_job(out, plan, cfg):
    validate_plan(plan,cfg,job=True)
    key=plan['game']; folder=out/'job-results'
    paths={name:folder/(key+'.'+name+'.json') for name in ('row','audit','complete','storage')}
    row,audit,complete,storage=(read(paths[k]) for k in ('row','audit','complete','storage'))
    hashes={name:sha(path) for name,path in paths.items()}
    validate_receipts(plan,row,audit,complete,storage,hashes['row'],hashes['audit'],cfg)
    verify_economy(row)
    task,digest=task_identity(out,plan,cfg)
    rival=out/'sources'/(plan['opponent']+'.py')
    if (sha(out/'agents'/(key+'.py'))!=plan['bot_sha256']
        or sha(task)!=digest or sha(ROOT/cfg['crop_checkpoint'])!=cfg['crop_checkpoint_sha256']
        or sha(rival)!=cfg['source_sha256'][plan['opponent']+'.py'] or plan['rival_sha256']!=sha(rival)):
        raise ValueError('Model or opponent identity changed')
    replay=out/'games'/key/'replay.json.gz'; trace=out/'traces'/(key+'.jsonl.gz')
    if sha(replay)!=row['replay_gzip_sha256'] or sha(trace)!=row['trace_sha256']:
        raise ValueError('Replay or trace identity changed')
    return sanitized(row),dict(game=key,receipt_sha256=hashes,replay_sha256=row['replay_gzip_sha256'],
        trace_sha256=row['trace_sha256'],task_checkpoint_sha256=digest,official_transitions=719)


def marker(out, state):
    path=(out/'games-started.json').resolve()
    if not path.is_relative_to(OUT.resolve()): raise ValueError('Marker outside experiment')
    with path.open('x',encoding='utf-8',newline='\n') as stream:
        state['created']=True
        json.dump(dict(attempt=1,freeze_sha256=sha(out/'freeze.json'),retry_allowed=False),stream)
        stream.write('\n')
    state['complete']=True


def run(out=OUT):
    out=Path(out).resolve(); cfg=ready(out); verify_commit(out)
    if (cfg['host_preflight']['samples'] != 3 or cfg['host_preflight']['min_available_bytes'] < 4*1024**3
        or cfg['official_job_supervision']['min_available_bytes'] < 2*1024**3):
        raise ValueError('Insufficient registered memory limits')
    if any((out/name).exists() for name in ('games-started.json','games-failure.local.json','games-result.json',
        'job-inputs','job-results','games','traces','agents','job-supervision','job-acceptance','game-ledger.json','game-audit-ledger.json')):
        raise ValueError('Panel attempt already consumed')
    if not memory_ready(cfg): raise RuntimeError('Insufficient preflight memory before panel')
    state=dict(created=False,complete=False); rows=[]; audits=[]; attempted=0; active=None
    supervision=None; entered=False; error=None; stop_reason=None
    try:
        marker(out,state); identities=identity(out)
        for index,plan in enumerate(plans(cfg['test_seeds'])):
            deadline_check(); ready(out); verify_commit(out); check_identity(out,identities)
            if index and not memory_ready(cfg):
                stop_reason='available_memory_preflight'; raise RuntimeError('Insufficient preflight memory')
            wrapper=make_wrapper(out,plan,cfg); rival=out/'sources'/(plan['opponent']+'.py')
            _,digest=task_identity(out,plan,cfg)
            item=dict(plan,bot_sha256=sha(wrapper),rival_sha256=sha(rival),task_checkpoint_sha256=digest)
            validate_plan(item,cfg,job=True)
            write_new(out/'job-inputs'/(plan['game']+'.json'),item)
            active=plan['game']; attempted+=1; entered=True; supervision=None
            supervisor_error=None
            try:
                supervision=supervise(WindowsAPI(),[sys.executable,'-B','-m','learning.task_teacher_weighted_eval_v2.games',
                    '--out',str(out),'--key',plan['game']],ROOT,cfg['official_job_supervision'])
            except BaseException as exc:
                supervisor_error=type(exc).__name__
            supervision_path=out/'job-supervision'/(plan['game']+'.json')
            write_new(supervision_path,dict(game=plan['game'],job_input_sha256=sha(out/'job-inputs'/(plan['game']+'.json')),
                freeze_sha256=identities['freeze.json'],supervision=supervision,error_kind=supervisor_error,
                cleanup_state='recorded' if supervision is not None else 'unknown',retry_allowed=False))
            if supervisor_error: raise RuntimeError('Job supervision failed')
            if not job_ok(supervision): stop_reason='job_failure_or_cleanup'; raise RuntimeError('Job failed or cleanup unconfirmed')
            check_identity(out,identities)
            if sha(wrapper)!=item['bot_sha256']: raise ValueError('Wrapper identity changed')
            row,audit=accept_job(out,item,cfg); audit['job_supervision']=supervision
            acceptance=out/'job-acceptance'/(plan['game']+'.json')
            write_new(acceptance,dict(passed=True,game=plan['game'],freeze_sha256=identities['freeze.json'],
                job_input_sha256=sha(out/'job-inputs'/(plan['game']+'.json')),supervision_sha256=sha(supervision_path),
                evidence=audit,retry_allowed=False))
            audit['supervision_receipt_sha256']=sha(supervision_path)
            audit['acceptance_receipt_sha256']=sha(acceptance)
            rows.append(row); audits.append(audit); active=None; entered=False
        check_identity(out,identities); ready(out); verify_commit(out); deadline_check()
        write_new(out/'game-ledger.json',rows); write_new(out/'game-audit-ledger.json',audits)
        verify_ledger(out,cfg,rows,audits,identities)
        check_identity(out,identities); ready(out); verify_commit(out); deadline_check()
        result=dict(passed=True,complete_games=24,live_official_transitions=24*719,primary_replay_transitions=24*719,
            game_ledger_sha256=sha(out/'game-ledger.json'),audit_ledger_sha256=sha(out/'game-audit-ledger.json'),
            identities=identities,independent_replay_not_yet_completed=True,final_economic_screening_completed=False,
            worker_stdout_stderr_discarded=True,retry_allowed=False)
        write_new(out/'games-result.json',result)
        return result
    except BaseException as exc:
        error=type(exc).__name__
    finally:
        if error and state['created']:
            failure=dict(error_kind=error,stop_reason=stop_reason,marker=state,attempted_games=attempted,completed_games=len(rows),
                confirmed_completed_live_transitions=len(rows)*719,confirmed_completed_primary_replay_transitions=len(rows)*719,
                uncertain_current_game=active,
                current_game_additional_transitions='unknown' if active else 0,
                child_lifecycle='supervision_recorded' if supervision is not None else 'unknown_after_spawn_attempt' if entered else 'not_started',
                job_supervision=supervision,retry_allowed=False,partial_work_not_refunded=True)
            try: write_new(out/'games-failure.local.json',failure)
            except BaseException as exc: raise CampaignFailure(error,type(exc).__name__) from None
    if error: raise CampaignFailure(error)


def verify_ledger(out,cfg,rows,audits,identities):
    """Rebind accepted receipts and compressed-file hashes without decoding."""
    check_identity(out,identities)
    if read(out/'game-ledger.json')!=rows or read(out/'game-audit-ledger.json')!=audits:
        raise ValueError('Panel ledger changed')
    queue=plans(cfg['test_seeds'])
    if len(rows)!=24 or len(audits)!=24: raise ValueError('Incomplete ledger')
    for plan,expected_row,expected_audit in zip(queue,rows,audits):
        item=read(out/'job-inputs'/(plan['game']+'.json'))
        if any(item.get(k)!=v for k,v in plan.items()): raise ValueError('Panel plan changed')
        row,audit=accept_job(out,item,cfg)
        supervision_path=out/'job-supervision'/(plan['game']+'.json')
        acceptance_path=out/'job-acceptance'/(plan['game']+'.json')
        supervision=read(supervision_path); acceptance=read(acceptance_path)
        input_sha=sha(out/'job-inputs'/(plan['game']+'.json'))
        if (supervision.get('game')!=plan['game'] or supervision.get('job_input_sha256')!=input_sha
            or supervision.get('freeze_sha256')!=identities['freeze.json'] or supervision.get('error_kind') is not None
            or supervision.get('cleanup_state')!='recorded' or supervision.get('retry_allowed') is not False
            or not job_ok(supervision.get('supervision'))): raise ValueError('Job evidence invalid')
        audit['job_supervision']=supervision['supervision']
        if (acceptance!=dict(passed=True,game=plan['game'],freeze_sha256=identities['freeze.json'],
            job_input_sha256=input_sha,supervision_sha256=sha(supervision_path),evidence=audit,retry_allowed=False)):
            raise ValueError('Acceptance evidence changed')
        audit['supervision_receipt_sha256']=sha(supervision_path)
        audit['acceptance_receipt_sha256']=sha(acceptance_path)
        if row!=expected_row or audit!=expected_audit: raise ValueError('Panel evidence changed')
    check_identity(out,identities)
    return rows


def validate_panel(out,cfg):
    """Rebind all accepted receipts and raw-file hashes, without decoding them."""
    out=Path(out).resolve(); ready(out); verify_commit(out); result=read(out/'games-result.json')
    if (result.get('passed') is not True or result.get('complete_games')!=24
        or result.get('live_official_transitions')!=24*719 or result.get('primary_replay_transitions')!=24*719
        or result.get('game_ledger_sha256')!=sha(out/'game-ledger.json')
        or result.get('audit_ledger_sha256')!=sha(out/'game-audit-ledger.json')): raise ValueError('Panel result changed')
    rows=read(out/'game-ledger.json'); audits=read(out/'game-audit-ledger.json')
    return verify_ledger(out,cfg,rows,audits,result['identities'])


if __name__=='__main__':
    result=run()
    print(json.dumps(dict(passed=result['passed'])))
