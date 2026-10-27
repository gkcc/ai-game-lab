"""One registered official panel followed by one bounded teacher-data sample."""
import json
import os
from pathlib import Path
import secrets
import sys
from learning.task_job_sampling_race_v1.sampling import supervise
from learning.task_training_job_supervision_v1.job import WindowsAPI
from learning.task_teacher_dev_deviation_v1.membership import ENV_KEY, NamedWindowsAPI, token_hash
from learning.task_probe_failure_reporting_v1.report import classify, safe_job
from learning.task_teacher_choice_eval_v2.campaign import verify_economy
from . import games
from .experiment import OUT, ROOT, deadline_check, read, ready, sha, verify_commit, write_new


class ReceiptFailure(RuntimeError):
    def __init__(self,primary,receipt):
        super().__init__('Primary failure and failure-receipt error')
        self.primary_error_kind,self.receipt_error_kind=primary,receipt


def marker(out,name,value,state):
    """Record exclusive creation even if serialization or close then fails."""
    path=(Path(out)/name).resolve()
    if not path.is_relative_to(OUT.resolve()):
        raise ValueError('Marker outside experiment')
    with path.open('x',encoding='utf-8',newline='\n') as stream:
        state['created']=True
        json.dump(value,stream,indent=2,allow_nan=False)
        stream.write('\n')
    state['complete']=True


def job_ok(job):
    value = safe_job(job)
    return (value is not None and value['passed'] is True and value['assigned'] is True
        and value['resumed'] is True and value['launcher_reaped'] is True
        and value['job_members_zero'] is True and value['exit_code'] == 0
        and value['error_kind'] is None and value['cleanup_error_kind'] is None and value['stop_reason'] is None)


def launch_named(name,module,limits):
    previous = os.environ.get(ENV_KEY)
    try:
        os.environ[ENV_KEY] = name
        return supervise(NamedWindowsAPI(name),[sys.executable,'-B','-m',module],ROOT,limits)
    finally:
        if previous is None:
            os.environ.pop(ENV_KEY,None)
        else:
            os.environ[ENV_KEY] = previous


def accept_game(out,plan,cfg):
    key = plan['game']
    folder = out/'job-results'
    paths = {name:folder/(key+'.'+name+'.json') for name in ('row','audit','complete','storage')}
    row,audit,complete,storage = (read(paths[name]) for name in ('row','audit','complete','storage'))
    games.validate_receipts(plan,row,audit,complete,storage,sha(paths['row']),sha(paths['audit']),cfg)
    verify_economy(row)
    replay,trace = out/'games'/key/'replay.json.gz',out/'traces'/(key+'.jsonl.gz')
    if (sha(out/'agents'/(key+'.py')) != plan['bot_sha256']
        or sha(out/'sources/round2.py') != plan['rival_sha256']
        or sha(ROOT/cfg['crop_checkpoint']) != cfg['crop_checkpoint_sha256']
        or sha(replay) != row['replay_gzip_sha256'] or sha(trace) != row['trace_sha256']):
        raise ValueError('Official artifact identity changed')
    return row,dict(game=key,row_sha256=sha(paths['row']),audit_sha256=sha(paths['audit']),
        complete_sha256=sha(paths['complete']),storage_sha256=sha(paths['storage']),
        replay_sha256=row['replay_gzip_sha256'],trace_sha256=row['trace_sha256'],
        trace_size=trace.stat().st_size,official_transitions=719)


def run_games():
    out=OUT.resolve(); cfg=ready(out); verify_commit(out)
    names=('games-started.json','games-result.json','games-failure.json','job-inputs','job-results',
           'games','traces','agents','game-ledger.json','game-audit-ledger.json','source-manifest.json')
    if any((out/name).exists() for name in names):
        raise ValueError('Official panel already attempted')
    plans=games.plans(cfg)
    rows, audits = [], []
    state=dict(created=False,complete=False); attempted=0; active=None; supervision=None; spawn_entered=False
    try:
        marker(out,'games-started.json',dict(attempt=1,freeze_sha256=sha(out/'freeze.json'),retry_allowed=False),state)
        for plan in plans:
            deadline_check(); ready(out)
            bot=games.make_wrapper(out,plan,cfg); rival=out/'sources/round2.py'
            item=dict(plan,bot_sha256=sha(bot),rival_sha256=sha(rival),task_checkpoint_sha256=None)
            games.validate_plan(item,cfg,job=True)
            write_new(out/'job-inputs'/(plan['game']+'.json'),item)
            active=plan['game']; attempted+=1; spawn_entered=True; supervision=None
            supervision=supervise(WindowsAPI(),[sys.executable,'-B','-m','learning.task_teacher_refresh_v1.games',
                '--out',str(out),'--key',plan['game']],ROOT,cfg['official_job_supervision'])
            if not job_ok(supervision):
                raise RuntimeError('Official Job failed or cleanup unconfirmed')
            row,audit=accept_game(out,item,cfg)
            audit['job_supervision']=safe_job(supervision)
            rows.append(row); audits.append(audit); active=None; spawn_entered=False
        write_new(out/'game-ledger.json',rows)
        write_new(out/'game-audit-ledger.json',audits)
        sources=[dict(game=r['game'],seed=r['seed'],side=r['side'],
            trace=dict(path='traces/'+r['game']+'.jsonl.gz',size=a['trace_size'],sha256=a['trace_sha256']),
            replay_sha256=a['replay_sha256'],audit_sha256=a['audit_sha256']) for r,a in zip(rows,audits)]
        write_new(out/'source-manifest.json',sources)
        result=dict(passed=True,complete_games=4,live_official_transitions=4*719,
            primary_replay_transitions=4*719,game_ledger_sha256=sha(out/'game-ledger.json'),
            audit_ledger_sha256=sha(out/'game-audit-ledger.json'),source_manifest_sha256=sha(out/'source-manifest.json'),
            additional_independent_replay_completed=False,retry_allowed=False)
        write_new(out/'games-result.json',result)
        return result
    except BaseException as exc:
        if state['created']:
            try:
                write_new(out/'games-failure.json',dict(passed=False,error_kind=type(exc).__name__,marker=state,
                    attempted_games=attempted,completed_games=len(rows),
                    confirmed_completed_live_transitions=len(rows)*719,confirmed_completed_primary_replay_transitions=len(rows)*719,
                    uncertain_current_game=active,current_game_additional_transitions='unknown' if active else 0,
                    child_lifecycle='supervision_recorded' if supervision is not None else 'unknown_after_spawn_attempt' if spawn_entered else 'not_started',
                    job_supervision=safe_job(supervision) if supervision is not None else None,retry_allowed=False))
            except BaseException as receipt:
                raise ReceiptFailure(type(exc).__name__,type(receipt).__name__) from None
        raise


def validate_panel(out,cfg):
    from .collector import validate_sources
    result, rows, audits, sources = (read(out/name) for name in
        ('games-result.json','game-ledger.json','game-audit-ledger.json','source-manifest.json'))
    if (result.get('passed') is not True or result.get('complete_games') != 4
        or result.get('live_official_transitions') != 4*719 or result.get('primary_replay_transitions') != 4*719
        or result.get('game_ledger_sha256') != sha(out/'game-ledger.json')
        or result.get('audit_ledger_sha256') != sha(out/'game-audit-ledger.json')
        or result.get('source_manifest_sha256') != sha(out/'source-manifest.json')
        or len(rows) != 4 or len(audits) != 4 or len(sources) != 4):
        raise ValueError('Incomplete or changed panel')
    validate_sources(sources,cfg)
    for plan,row,audit,source in zip(games.plans(cfg),rows,audits,sources):
        key=plan['game']; folder=out/'job-results'
        paths={name:folder/(key+'.'+name+'.json') for name in ('row','audit','complete','storage')}
        for name in ('row','audit','complete','storage'):
            if sha(paths[name]) != audit[name+'_sha256']:
                raise ValueError('Individual game receipt changed')
        stored_row,stored_audit,complete,storage=(read(paths[name]) for name in ('row','audit','complete','storage'))
        item=dict(plan,bot_sha256=row['bot_sha256'],rival_sha256=row['rival_sha256'],
                  task_checkpoint_sha256=row['task_checkpoint_sha256'])
        games.validate_receipts(item,stored_row,stored_audit,complete,storage,
                                audit['row_sha256'],audit['audit_sha256'],cfg)
        verify_economy(stored_row)
        if (any(row.get(k) != v for k,v in plan.items()) or audit['game'] != plan['game']
            or stored_row != row or audit.get('official_transitions') != 719
            or source['game'] != plan['game'] or source['seed'] != plan['seed'] or source['side'] != plan['side']
            or source['replay_sha256'] != row['replay_gzip_sha256'] or source['audit_sha256'] != audit['audit_sha256']
            or source['trace']['sha256'] != row['trace_sha256'] or source['trace']['size'] != audit['trace_size']
            or not job_ok(audit['job_supervision'])):
            raise ValueError('Source and audit identity differ')
    return result


def run_data():
    out=OUT.resolve(); cfg=ready(out); verify_commit(out); validate_panel(out,cfg)
    names=('data-started.json','data-result.json','probe-started.json','probe-result.json',
           'data-worker-started.json','data-worker-entered.json','data-worker-result.json','data-worker-failure.json')
    if any((out/name).exists() for name in names):
        raise ValueError('Data collection already attempted')
    state=dict(created=False,complete=False); probe_job=None; verdict=None; worker_job=None; worker_entered=False
    try:
        freeze_sha=sha(out/'freeze.json')
        marker(out,'data-started.json',dict(attempt=1,freeze_sha256=freeze_sha,retry_allowed=False),state)
        probe_name='Local\\TeacherDeviation-'+secrets.token_hex(16)
        write_new(out/'probe-started.json',dict(attempt=1,freeze_sha256=freeze_sha,
            job_token_sha256=token_hash(probe_name),retry_allowed=False))
        probe_marker_sha=sha(out/'probe-started.json')
        probe_job=launch_named(probe_name,'learning.task_teacher_refresh_v1.member_probe',cfg['probe_job_supervision'])
        verdict=classify(probe_job,lambda:read(out/'probe-result.json'),freeze_sha,probe_marker_sha)
        if not verdict['worker_allowed']:
            raise RuntimeError('Probe rejected; data worker not launched')
        if sha(out/'probe-started.json') != probe_marker_sha or sha(out/'freeze.json') != freeze_sha:
            raise ValueError('Probe identity changed')
        deadline_check(); ready(out); validate_panel(out,cfg)
        worker_name='Local\\TeacherDeviation-'+secrets.token_hex(16)
        write_new(out/'data-worker-started.json',dict(attempt=1,freeze_sha256=freeze_sha,
            job_token_sha256=token_hash(worker_name),source_manifest_sha256=sha(out/'source-manifest.json'),
            schema_sha256=sha(out/'schema.json'),retry_allowed=False))
        worker_entered=True
        worker_job=launch_named(worker_name,'learning.task_teacher_refresh_v1.data_worker',cfg['data_job_supervision'])
        if not job_ok(worker_job):
            raise RuntimeError('Data Job failed or cleanup unconfirmed')
        worker=read(out/'data-worker-result.json')
        raw=worker.get('raw_reads',{})
        if (worker.get('passed') is not True or worker.get('freeze_sha256') != freeze_sha
            or worker.get('source_manifest_sha256') != sha(out/'source-manifest.json')
            or worker.get('schema_sha256') != cfg['schema_sha256'] or worker.get('collection_attempt') != 1
            or worker.get('sampling_algorithm') != cfg['sampling_algorithm']
            or raw.get('files_opened') != 4 or raw.get('files_completed') != 4
            or raw.get('raw_decode_passes_per_file') != 1
            or worker.get('array_sha256') != sha(out/'private/new-teacher.npz')):
            raise ValueError('Worker result or data identity changed')
        ready(out); validate_panel(out,cfg); deadline_check()
        result=dict(passed=True,probe=verdict,worker_job=safe_job(worker_job),
            worker_result_sha256=sha(out/'data-worker-result.json'),freeze_sha256=freeze_sha,
            array_sha256=worker['array_sha256'],groups=worker['groups'],groups_by_seed=worker['groups_by_seed'],
            selected_operations=worker['selected_operations'],teacher_choices_checked=worker['teacher_choices_checked'],
            candidate_rows=worker['candidate_rows'],raw_reads=raw,retry_allowed=False,
            training_started=False,official_test_games_started=False,
            interpretation='Development teacher choices; task completion and competitive improvement are not proven')
        write_new(out/'data-result.json',result)
        return result
    except BaseException as exc:
        if state['created']:
            try:
                write_new(out/'data-result.json',dict(passed=False,error_kind=type(exc).__name__,marker=state,
                    probe=verdict,probe_job=safe_job(probe_job) if probe_job is not None else None,
                    worker_job=safe_job(worker_job) if worker_job is not None else None,
                    worker_launch_entered=worker_entered,raw_trace_reads='unknown' if worker_entered else 'not_started_by_runner',
                    retry_allowed=False,training_started=False))
            except BaseException as receipt:
                raise ReceiptFailure(type(exc).__name__,type(receipt).__name__) from None
        raise


if __name__ == '__main__':
    if len(sys.argv) != 2 or sys.argv[1] not in ('games','data'):
        raise SystemExit('Expected games or data')
    (run_games if sys.argv[1] == 'games' else run_data)()
