"""Bounded paired panel: official worker execution and primary replay seals."""
import json
import math
from pathlib import Path
import sys
from learning.task_training_job_supervision_v1.job import WindowsAPI
from learning.task_job_sampling_race_v1.sampling import supervise
from .experiment import OUT,ROOT,ready,verify_commit,deadline_check,read,sha,write_new
from .evaluation import plans
from .games import make_wrapper,task_identity,validate_receipts


class CampaignFailure(RuntimeError):
    def __init__(self,primary,receipt=None):
        super().__init__('Panel failed; attempt consumed')
        self.primary_error_kind,self.receipt_error_kind=primary,receipt


def verify_economy(row):
    side=row['side']; banks=row['banks']; initial=row['initial_banks']
    if (type(side) is not int or side not in (0,1) or not isinstance(banks,list) or len(banks)!=2
        or not isinstance(initial,list) or len(initial)!=2): raise ValueError('Invalid bank identity')
    for x in banks+initial:
        if type(x) not in (int,float) or not math.isfinite(x): raise ValueError('Invalid bank value')
    expected=dict(bank=banks[side],opponent_bank=banks[1-side],initial_bank=initial[side],
        opponent_initial_bank=initial[1-side],net_profit=banks[side]-initial[side],
        opponent_net_profit=banks[1-side]-initial[1-side],margin=banks[side]-banks[1-side])
    if any(row.get(k)!=v for k,v in expected.items()) or row.get('rewards')!=banks:
        raise ValueError('Initial cash or net profit mismatch')


def accept_job(out,plan,cfg):
    """Read bounded receipts and hash committed files; no NPZ/model decoding."""
    key=plan['game']; folder=out/'job-results'
    paths={name:folder/(key+'.'+name+'.json') for name in ('row','audit','complete','storage')}
    row,certificate,complete,storage=(read(paths[name]) for name in ('row','audit','complete','storage'))
    if (not storage['passed'] or storage['error_kind'] is not None or storage['output']['refused'] != 0):
        raise ValueError('Worker storage receipt failed')
    validate_receipts(plan,row,certificate,complete,sha(paths['row']),sha(paths['audit']),cfg)
    verify_economy(row)
    task,digest=task_identity(out,plan,cfg)
    if sha(task)!=digest or sha(ROOT/cfg['crop_checkpoint'])!=cfg['crop_checkpoint_sha256']:
        raise ValueError('Model identity changed')
    if sha(out/'games'/key/'replay.json.gz')!=row['replay_gzip_sha256'] or sha(out/'traces'/(key+'.jsonl.gz'))!=row['trace_sha256']:
        raise ValueError('Replay or trace identity changed')
    return row,dict(game=key,row_sha256=sha(paths['row']),certificate_sha256=sha(paths['audit']),
                    complete_sha256=sha(paths['complete']),storage_sha256=sha(paths['storage']),
                    replay_sha256=row['replay_gzip_sha256'],trace_sha256=row['trace_sha256'],
                    task_checkpoint_sha256=digest,official_transitions=719)


def marker(out,value,state):
    path=(out/'games-started.json').resolve()
    if not path.is_relative_to(OUT.resolve()): raise ValueError('Marker outside experiment')
    with path.open('x',encoding='utf-8',newline='\n') as f:
        state['created']=True
        json.dump(value,f,indent=2,allow_nan=False); f.write('\n')
    state['complete']=True


def run(out=OUT):
    out=Path(out).resolve(); cfg=ready(out); verify_commit(out)
    if any((out/name).exists() for name in ('games-started.json','games-failure.local.json','games-result.json',
        'job-inputs','job-results','games','traces','agents','game-ledger.json','game-audit-ledger.json')):
        raise ValueError('Panel attempt already consumed')
    queue=plans(cfg['test_seeds']); state=dict(created=False,complete=False)
    rows=[]; audits=[]; attempted=0; active=None; supervision=None
    error=None; spawn_entered=False
    try:
        marker(out,dict(attempt=1,freeze_sha256=sha(out/'freeze.json'),retry_allowed=False),state)
        for plan in queue:
            deadline_check(); ready(out)
            bot=make_wrapper(out,plan,cfg); rival=out/'sources/round2.py'
            _,digest=task_identity(out,plan,cfg)
            item=dict(plan,bot_sha256=sha(bot),rival_sha256=sha(rival),task_checkpoint_sha256=digest)
            write_new(out/'job-inputs'/(plan['game']+'.json'),item)
            active=plan['game']; attempted+=1; spawn_entered=True; supervision=None
            supervision=supervise(WindowsAPI(),[sys.executable,'-B','-m','learning.task_teacher_choice_eval_v2.games',
                '--out',str(out),'--key',plan['game']],ROOT,cfg['official_job_supervision'])
            if (not supervision['passed'] or not supervision['launcher_reaped'] or not supervision['job_members_zero']):
                raise RuntimeError('Worker or Job cleanup unconfirmed')
            row,audit=accept_job(out,item,cfg)
            audit['job_supervision']=supervision
            rows.append(row); audits.append(audit); active=None; spawn_entered=False
        write_new(out/'game-ledger.json',rows); write_new(out/'game-audit-ledger.json',audits)
        result=dict(passed=True,complete_games=8,live_official_transitions=8*719,primary_replay_transitions=8*719,
            game_ledger_sha256=sha(out/'game-ledger.json'),audit_ledger_sha256=sha(out/'game-audit-ledger.json'),
            independent_replay_not_yet_completed=True,final_economic_screening_completed=False,
            worker_stdout_stderr_discarded=True,retry_allowed=False)
        write_new(out/'games-result.json',result)
        return result
    except BaseException as exc:
        error=type(exc).__name__
    finally:
        if error and state['created']:
            failure=dict(error_kind=error,marker=state,attempted_games=attempted,completed_games=len(rows),
                confirmed_completed_live_transitions=len(rows)*719,confirmed_completed_primary_replay_transitions=len(rows)*719,
                uncertain_current_game=active,current_game_additional_transitions='unknown' if active else 0,
                child_lifecycle='supervision_recorded' if supervision is not None else 'unknown_after_spawn_attempt' if spawn_entered else 'not_started',
                job_supervision=supervision,retry_allowed=False,partial_work_not_refunded=True)
            try: write_new(out/'games-failure.local.json',failure)
            except BaseException as exc: raise CampaignFailure(error,type(exc).__name__) from None
    if error: raise CampaignFailure(error)
