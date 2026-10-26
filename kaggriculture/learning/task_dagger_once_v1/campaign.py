"""One collection, one fixed update, conditional development and holdout."""
import argparse
from collections import Counter
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from learning.bc_crop.agent import ROOT,sha
from learning.task_bc_holdout_v1.games import write_new,terminate_owned
from learning.task_dagger_once_v1.experiment import OUT,read,ready,verify,model_identity
from learning.task_dagger_once_v1.games import wrapper,task_identity,control_difference
from learning.task_dagger_once_v1.data import summarize_game,build_cache
from learning.task_dagger_once_v1.statistics import early_gate,development_gate,holdout_statistics


def plans(cfg, split, seed, arms):
    assert seed in cfg[{'train':'train_seeds','dev':'validation_seeds','holdout':'test_seeds'}[split]]
    return [dict(split=split,seed=seed,opponent=opp,side=side,policy=arm,
        game=f'{split}-{seed}-{opp}-{side}-{arm}',
        parity_kind='student' if split=='train' and seed==cfg['train_seeds'][0]
            and opp==cfg['opponents'][0] and arm=='baseline' else None)
        for opp in cfg['opponents'] for arm in arms for side in cfg['seats']]


class Runner:
    def __init__(self,out,cfg):
        self.out,self.cfg=Path(out),cfg
        self.start=time.monotonic()
        self.attempts,self.rows,self.certificates=[],[],[]
        self.environment=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')

    def limits(self):
        if (self.out/'campaign-cancel.local.json').exists():
            raise InterruptedError('Campaign cancelled by owner')
        assert time.monotonic()-self.start<self.cfg['campaign_seconds'],'Campaign deadline'
        assert len(self.attempts)<=self.cfg['max_games']
        counts=Counter()
        for cert in self.certificates:counts.update(cert['total_model_counts'])
        assert counts['task_forwards']<=self.cfg['task_forward_budget']
        assert counts['crop_forwards']<=self.cfg['crop_forward_budget']
        assert counts['task_rows']+counts['crop_rows']<=self.cfg['neural_input_row_budget']
        assert sum(c['policy_turns'] for c in self.certificates)<=self.cfg['policy_turn_budget']
        return dict(counts)

    def reserve(self,job):
        future=self.attempts+[job]
        neural=sum(j['policy']!='teacher_task' for j in future)
        teacher=sum(j['policy']=='teacher_task' for j in future)
        parity=sum(j['parity_kind']=='student' for j in future)
        caps=self.cfg['game_limits']; n=len(future)
        assert neural<=96 and teacher<=40 and parity<=2 and n<=136
        assert (2*neural+parity)*caps['task_forwards']<=self.cfg['task_forward_budget']
        assert (2*n+parity)*caps['crop_forwards']<=self.cfg['crop_forward_budget']
        assert (2*n+parity)*caps['neural_input_rows']<=self.cfg['neural_input_row_budget']
        assert 2*n*(caps['controller_calls']+719)+parity*719<=self.cfg['policy_turn_budget']

    def execute(self,jobs):
        active={}; queue=list(jobs); completed={}; cursor=0
        try:
            while active or cursor<len(queue):
                self.limits()
                for key,state in list(active.items()):
                    process=state['process']
                    if process.poll() is None and time.monotonic()-state['start']>=self.cfg['per_job_timeout_seconds']:
                        raise TimeoutError('Owned official job exceeded deadline')
                    code=process.poll()
                    if code is None:continue
                    process.wait();state['log'].close();del active[key]
                    assert code==0,'Official job failure; retain private log and stop'
                    rp=self.out/'job-results'/(key+'.row.json');ap=self.out/'job-results'/(key+'.audit.json')
                    marker=read(self.out/'job-results'/(key+'.complete.json'))
                    assert marker['passed'] and marker['row_sha256']==sha(rp) and marker['certificate_sha256']==sha(ap)
                    row,cert=read(rp),read(ap);assert cert['passed']
                    self.rows.append(row);self.certificates.append(cert);completed[key]=row
                    print(json.dumps(dict(completed=len(self.rows),attempted=len(self.attempts),game=key)),flush=True)
                self.limits()
                while len(active)<self.cfg['max_workers'] and cursor<len(queue):
                    verify(self.out)
                    job=queue[cursor];self.reserve(job);cursor+=1
                    key=job['game'];bot=wrapper(self.out,job,self.cfg)
                    _task,digest=task_identity(self.out,job,self.cfg)
                    item=dict(job,bot_sha256=sha(bot),rival_sha256=sha(self.out/'sources'/(job['opponent']+'.py')),
                              task_checkpoint_sha256=digest)
                    write_new(self.out/'job-inputs'/(key+'.json'),item);self.attempts.append(item)
                    path=self.out/'job-local'/(key+'.log');path.parent.mkdir(exist_ok=True);log=path.open('xb')
                    try:
                        opts=dict(creationflags=subprocess.CREATE_NO_WINDOW) if sys.platform=='win32' else dict(start_new_session=True)
                        process=subprocess.Popen([sys.executable,'-B','-X','utf8',str(Path(__file__).with_name('games.py')),
                            '--out',str(self.out),'--key',key],cwd=ROOT.parent,env=self.environment,stdout=log,stderr=subprocess.STDOUT,**opts)
                    except BaseException:
                        log.close();raise
                    active[key]=dict(process=process,log=log,start=time.monotonic())
                if active:time.sleep(.2)
        finally:
            for state in active.values():
                terminate_owned(state['process']);state['log'].close()
        return completed

    def collect(self):
        summaries=[]
        for index,seed in enumerate(self.cfg['train_seeds']):
            rows=self.execute(plans(self.cfg,'train',seed,['baseline']))
            if index==0:
                summaries=[summarize_game(self.out,row,self.cfg) for row in rows.values()]
                write_new(self.out/'early-collection-summary.json',summaries)
                gate=early_gate(summaries,self.cfg);write_new(self.out/'early-gate.json',gate)
                if not gate['passed']:return False
        for seed in self.cfg['validation_seeds']:
            self.execute(plans(self.cfg,'dev',seed,['baseline']))
        self.limits()
        build_cache(self.out,self.rows,self.cfg)
        self.limits()
        return True

    def gpu_fit(self):
        ready(self.out)
        write_new(self.out/'training-launch.json',dict(fit_count=1,input_sha256={name:sha(self.out/name)
            for name in ('registration.json','freeze.json','pre-execution-commit.json','cache/manifest.json')}))
        launch=read(self.out/'gpu-launch.local.json')
        command=['wsl.exe','-d',launch['distro'],'--cd',launch['workdir'],'--',launch['python'],
            '-B','-m','learning.task_dagger_once_v1.gpu_guard','--out',self.out.relative_to(ROOT).as_posix()]
        tick=time.monotonic();process=None
        with (self.out/'training-process.local.log').open('xb') as log:
            try:
                process=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,
                    creationflags=subprocess.CREATE_NO_WINDOW if sys.platform=='win32' else 0)
                # Process identity is local-only; it is not part of public evidence.
                write_new(self.out/'training-process.local.json',dict(pid=process.pid,command_kind='owned WSL session guard'))
                while process.poll() is None:
                    self.limits()
                    if time.monotonic()-tick>self.cfg['gpu_guard']['process_seconds']:
                        raise TimeoutError('GPU process exceeded registered total deadline')
                    time.sleep(.2)
                process.wait()
                assert process.returncode==0,'GPU fit failed; inspect retained private log'
            finally:
                if process is not None and process.poll() is None:
                    marker=self.out/'training-cancel.local.json'
                    if not marker.exists():write_new(marker,dict(cancel=True,reason='owned parent stopping'))
                    # The Linux guard terminates its owned session and reaps the
                    # child before acknowledging. Never equate killing wsl.exe
                    # with successful Linux cancellation.
                    try:process.wait(timeout=30)
                    except subprocess.TimeoutExpired as exc:
                        raise RuntimeError('Owned Linux session cancellation not yet acknowledged') from exc
        elapsed=time.monotonic()-tick
        assert elapsed<=self.cfg['gpu_guard']['process_seconds']
        assert read(self.out/'gpu-guard-result.json')['completed']
        digest=model_identity(self.out)
        write_new(self.out/'training-process-result.json',dict(passed=True,process_seconds=elapsed,
            checkpoint_sha256=digest,linux_guard_completed=True,fit_count=1))
        self.limits()
        return read(self.out/'offline-gate.json')['passed']

    def add_control_evidence(self,split,seed):
        rows=[r for r in self.rows if r['split']==split and r['seed']==seed]
        index={(r['opponent'],r['side'],r['policy']):r for r in rows}
        for row in rows:
            if row['policy']=='candidate':
                baseline=index[(row['opponent'],row['side'],'baseline')]
                row['first_request_difference_step']=control_difference(self.out,row,baseline)

    def finish(self,status,**extra):
        verify(self.out,history=True)
        counts=self.limits()
        write_new(self.out/'ledger.json',self.rows)
        write_new(self.out/'audit.json',self.certificates)
        fit_path=self.out/'fit-result.json'
        write_new(self.out/'run-summary.json',dict(status=status,completed_games=len(self.rows),
            attempted_games=len(self.attempts),all_attempts_completed=len(self.rows)==len(self.attempts),
            live_official_transitions=len(self.rows)*719,primary_official_transitions=len(self.certificates)*719,
            game_model_counts=counts,policy_turns=sum(c['policy_turns'] for c in self.certificates),
            machine_seconds=time.monotonic()-self.start,fit_attempts=int((self.out/'training-started.json').exists()),
            fit_completed=fit_path.exists(),fit_result_sha256=sha(fit_path) if fit_path.exists() else None,
            default_changed=False,automatic_teacher_admission=False,registration_sha256=sha(self.out/'registration.json'),
            freeze_sha256=sha(self.out/'freeze.json'),**extra))


def run(out=OUT):
    out=Path(out).resolve();cfg=ready(out)
    write_new(out/'campaign-started.json',dict(registration_sha256=sha(out/'registration.json'),
        freeze_sha256=sha(out/'freeze.json'),started_utc=datetime.now(timezone.utc).isoformat()))
    runner=Runner(out,cfg)
    try:
        if not runner.collect():
            runner.finish('stopped_early_supervision',holdout_run=False,gpu_operations=0);return
        if not runner.gpu_fit():
            runner.finish('stopped_offline_gate',holdout_run=False,actual_gpu_training=True);return
        for seed in cfg['validation_seeds']:
            runner.execute(plans(cfg,'dev',seed,['candidate','teacher_task']))
            runner.add_control_evidence('dev',seed)
        gate=development_gate([r for r in runner.rows if r['split']=='dev'],cfg)
        write_new(out/'development-gate.json',dict(gate,checkpoint_sha256=model_identity(out)))
        if not gate['passed']:
            runner.finish('stopped_development_gate',holdout_run=False,actual_gpu_training=True);return
        for seed in cfg['test_seeds']:
            runner.execute(plans(cfg,'holdout',seed,cfg['arms']))
            runner.add_control_evidence('holdout',seed)
        statistics=holdout_statistics([r for r in runner.rows if r['split']=='holdout'],cfg)
        write_new(out/'holdout-statistics.json',dict(statistics,checkpoint_sha256=model_identity(out)))
        runner.finish('completed_holdout',holdout_run=True,actual_gpu_training=True,
            research_positive=statistics['research_positive'],teacher_retention=statistics['teacher_retention'],
            candidate_admission=statistics['admission'])
    except BaseException as exc:
        write_new(out/'failure.json',dict(kind=type(exc).__name__,attempts=runner.attempts,
            attempted_games=len(runner.attempts),completed_games=len(runner.rows),
            retry_allowed=False,partial_work_not_complete_accounting=True))
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('command',choices=['run'])
    parser.add_argument('--out',type=Path,default=OUT)
    args=parser.parse_args();run(args.out)
