"""One anchored update, then conditional fresh development and holdout games."""
import argparse
from collections import Counter
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import shutil
import sys
import time

from learning.bc_crop.agent import ROOT,sha
from .experiment import OUT,read,write_new,ready,verify,model_identity
from .games import wrapper,task_identity,control_difference
from .statistics import development_gate,holdout_statistics
from .storage import owned_writes
from .processes import CapturedProcess

FOREIGN_RECEIPT_RESERVE = 4*1024**2


def plans(cfg,split,seed):
    assert split in ('dev','holdout')
    assert seed in cfg['validation_seeds' if split=='dev' else 'test_seeds']
    return [dict(split=split,seed=seed,opponent=opp,side=side,policy=arm,parity_kind=None,
        game=f'{split}-{seed}-{opp}-{side}-{arm}')
        for opp in cfg['opponents'] for arm in cfg['arms'] for side in cfg['seats']]


def owned_size(out):
    return sum(p.stat().st_size for p in Path(out).rglob('*') if p.is_file())


class Runner:
    def __init__(self,out,cfg,parent_budget):
        self.out,self.cfg,self.parent_budget=Path(out),cfg,parent_budget
        self.start=time.monotonic();self.attempts=[];self.rows=[];self.certificates=[]
        self.environment=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')

    def limits(self):
        if (self.out/'campaign-cancel.local.json').exists():raise InterruptedError('Owned campaign cancelled')
        assert time.monotonic()-self.start<self.cfg['campaign_seconds'],'Campaign deadline'
        assert len(self.attempts)<=self.cfg['max_games']
        counts=Counter()
        for cert in self.certificates:counts.update(cert['total_model_counts'])
        assert counts['task_forwards']<=self.cfg['task_forward_budget']
        assert counts['crop_forwards']<=self.cfg['crop_forward_budget']
        assert counts['task_rows']+counts['crop_rows']<=self.cfg['neural_input_row_budget']
        assert sum(c['policy_turns'] for c in self.certificates)<=self.cfg['policy_turn_budget']
        return dict(counts)

    def storage_ready(self,inflight_games=0,training=False):
        limits=self.cfg['storage']
        pending=(inflight_games+1)*limits['game_write_bytes'] if not training else limits['fit_write_bytes']
        # Worker output is not the only concurrent writer: reserve all remaining
        # parent writes and the conservative full child-receipt allowance too.
        pending+=self.parent_budget.remaining_bytes+FOREIGN_RECEIPT_RESERVE
        assert owned_size(self.out)<=limits['total_bytes'],'Owned output exceeded registered ceiling'
        assert shutil.disk_usage(self.out).free>=limits['free_floor_bytes']+pending,'Insufficient output space with in-flight reserves'

    def reserve(self,job):
        future=self.attempts+[job]
        neural=sum(j['policy']!='teacher_task' for j in future)
        teacher=sum(j['policy']=='teacher_task' for j in future)
        caps=self.cfg['game_limits'];n=len(future);storage=self.cfg['storage']
        assert neural<=80 and teacher<=40 and n<=120
        assert all(j['parity_kind'] is None for j in future)
        assert 2*neural*caps['task_forwards']<=self.cfg['task_forward_budget']
        assert 2*n*caps['crop_forwards']<=self.cfg['crop_forward_budget']
        assert 2*n*caps['neural_input_rows']<=self.cfg['neural_input_row_budget']
        assert 2*n*(caps['controller_calls']+719)<=self.cfg['policy_turn_budget']
        assert (n*storage['game_write_bytes']+storage['fit_write_bytes']
                +storage['primary_parent_write_bytes']+storage['independent_reserve_bytes'])<=storage['total_bytes']

    def execute(self,jobs):
        active={};queue=list(jobs);cursor=0
        try:
            while active or cursor<len(queue):
                self.limits()
                for key,state in list(active.items()):
                    child=state['child'];child.check();process=child.process
                    if process.poll() is None and time.monotonic()-state['start']>=self.cfg['per_job_timeout_seconds']:
                        raise TimeoutError('Owned official job deadline')
                    code=process.poll()
                    if code is None:continue
                    capture=child.finish();del active[key]
                    assert code==0,'Official job failed; no retry'
                    rp=self.out/'job-results'/(key+'.row.json');ap=self.out/'job-results'/(key+'.audit.json')
                    marker=read(self.out/'job-results'/(key+'.complete.json'))
                    storage=read(self.out/'job-results'/(key+'.storage.json'))
                    assert marker['passed'] and marker['row_sha256']==sha(rp) and marker['certificate_sha256']==sha(ap)
                    assert storage['passed'] and storage['output']['refused']==0
                    assert storage['output']['limit_bytes']==self.cfg['storage']['game_write_bytes']
                    assert storage['output']['used_bytes']<=self.cfg['storage']['game_write_bytes']
                    row,cert=read(rp),read(ap);assert cert['passed']
                    cert.update(storage_receipt_sha256=sha(self.out/'job-results'/(key+'.storage.json')),
                        output_write_bytes=storage['output']['used_bytes'],capture=capture)
                    self.rows.append(row);self.certificates.append(cert)
                    print(json.dumps(dict(completed=len(self.rows),attempted=len(self.attempts),game=key)),flush=True)
                self.limits()
                while len(active)<self.cfg['max_workers'] and cursor<len(queue):
                    verify(self.out);self.storage_ready(len(active))
                    job=queue[cursor];self.reserve(job);cursor+=1
                    key=job['game'];bot=wrapper(self.out,job,self.cfg);_task,digest=task_identity(self.out,job,self.cfg)
                    item=dict(job,bot_sha256=sha(bot),rival_sha256=sha(self.out/'sources'/(job['opponent']+'.py')),
                              task_checkpoint_sha256=digest)
                    write_new(self.out/'job-inputs'/(key+'.json'),item);self.attempts.append(item)
                    child=CapturedProcess([sys.executable,'-B','-X','utf8',str(Path(__file__).with_name('games.py')),
                        '--out',str(self.out),'--key',key],cwd=ROOT.parent,env=self.environment,
                        log=self.out/'job-local'/(key+'.log'),limit_bytes=self.cfg['storage']['process_log_bytes'])
                    active[key]=dict(child=child,start=time.monotonic())
                if active:time.sleep(.2)
        finally:
            for state in active.values():state['child'].stop()

    def gpu_fit(self):
        ready(self.out);self.storage_ready(training=True)
        write_new(self.out/'training-launch.json',dict(fit_count=1,input_sha256={n:sha(self.out/n)
            for n in ('registration.json','freeze.json','pre-execution-commit.json','reused-cache-inputs.json')}))
        launch=read(self.out/'gpu-launch.local.json')
        command=['wsl.exe','-d',launch['distro'],'--cd',launch['workdir'],'--',launch['python'],
            '-B','-m','learning.task_anchor_once_v1.gpu_guard','--out',self.out.relative_to(ROOT).as_posix()]
        tick=time.monotonic();child=None
        try:
            child=CapturedProcess(command,cwd=ROOT,log=self.out/'training-process.local.log',
                limit_bytes=self.cfg['storage']['process_log_bytes'])
            write_new(self.out/'training-process.local.json',dict(pid=child.process.pid,kind='owned WSL supervisor'))
            while child.process.poll() is None:
                self.limits();child.check()
                if time.monotonic()-tick>self.cfg['gpu_guard']['process_seconds']:raise TimeoutError('GPU process deadline')
                time.sleep(.2)
            capture=child.finish();assert child.process.returncode==0,'Single fit failed'
        finally:
            if child is not None:
                try:
                    if child.process.poll() is None:
                        try:
                            marker=self.out/'training-cancel.local.json'
                            if not marker.exists():write_new(marker,dict(cancel=True,reason='owned parent stopping'))
                        finally:
                            # Even a refused cancellation write must wait for
                            # the already bounded Linux supervisor. Killing
                            # only the Windows wrapper proves no Linux cleanup.
                            remaining=self.cfg['gpu_guard']['process_seconds']-(time.monotonic()-tick)
                            child.process.wait(timeout=max(.01,remaining))
                finally:
                    child.thread.join(timeout=5)
                    if child.thread.is_alive():raise RuntimeError('GPU output reader remains active')
        elapsed=time.monotonic()-tick
        assert elapsed<=self.cfg['gpu_guard']['process_seconds']
        assert read(self.out/'gpu-guard-result.json')['completed']
        receipt=read(self.out/'training-storage.json')
        assert receipt['fit_completed'] and receipt['worker_output']['refused']==0
        assert receipt['worker_output']['used_bytes']<=self.cfg['storage']['fit_write_bytes']
        digest=model_identity(self.out)
        write_new(self.out/'training-process-result.json',dict(passed=True,process_seconds=elapsed,
            checkpoint_sha256=digest,fit_count=1,linux_guard_completed=True,capture=capture))
        self.limits()
        return read(self.out/'offline-gate.json')['passed']

    def add_control(self,split,seed):
        rows=[r for r in self.rows if r['split']==split and r['seed']==seed]
        index={(r['opponent'],r['side'],r['policy']):r for r in rows}
        for row in rows:
            if row['policy']=='candidate':
                row['first_request_difference_step']=control_difference(self.out,row,index[row['opponent'],row['side'],'baseline'])

    def finish(self,status,**extra):
        verify(self.out,history=True);counts=self.limits()
        write_new(self.out/'ledger.json',self.rows);write_new(self.out/'audit.json',self.certificates)
        write_new(self.out/'run-summary.json',dict(status=status,completed_games=len(self.rows),attempted_games=len(self.attempts),
            all_attempts_completed=len(self.rows)==len(self.attempts),live_official_transitions=len(self.rows)*719,
            primary_official_transitions=len(self.certificates)*719,game_model_counts=counts,
            policy_turns=sum(c['policy_turns'] for c in self.certificates),machine_seconds=time.monotonic()-self.start,
            fit_attempts=int((self.out/'training-started.json').exists()),fit_completed=(self.out/'fit-result.json').exists(),
            fit_result_sha256=sha(self.out/'fit-result.json') if (self.out/'fit-result.json').exists() else None,
            default_changed=False,automatic_teacher_admission=False,registration_sha256=sha(self.out/'registration.json'),
            freeze_sha256=sha(self.out/'freeze.json'),**extra))


def run(out=OUT):
    out=Path(out).resolve();cfg=ready(out);storage=cfg['storage']
    initial=owned_size(out)
    assert not (out/'campaign-started.json').exists()
    # Outside the primary Python guard: <=120*16KiB worker receipts, fit64KiB,
    # supervisor64KiB, worker receipt16KiB, final parent receipt16KiB.
    foreign=FOREIGN_RECEIPT_RESERVE
    allowance=storage['primary_parent_write_bytes']-initial-foreign
    assert allowance>0
    parent=owned_writes(out,allowance)
    runner=None;error=None
    try:
        with parent:
            write_new(out/'campaign-started.json',dict(registration_sha256=sha(out/'registration.json'),
                freeze_sha256=sha(out/'freeze.json'),started_utc=datetime.now(timezone.utc).isoformat()))
            runner=Runner(out,cfg,parent)
            if not runner.gpu_fit():
                runner.finish('stopped_offline_gate',holdout_run=False,actual_gpu_training=True);return
            for seed in cfg['validation_seeds']:
                runner.execute(plans(cfg,'dev',seed));runner.add_control('dev',seed)
            gate=development_gate([r for r in runner.rows if r['split']=='dev'],cfg)
            write_new(out/'development-gate.json',dict(gate,checkpoint_sha256=model_identity(out)))
            if not gate['passed']:
                runner.finish('stopped_development_gate',holdout_run=False,actual_gpu_training=True);return
            for seed in cfg['test_seeds']:
                runner.execute(plans(cfg,'holdout',seed));runner.add_control('holdout',seed)
            stats=holdout_statistics([r for r in runner.rows if r['split']=='holdout'],cfg)
            write_new(out/'holdout-statistics.json',dict(stats,checkpoint_sha256=model_identity(out)))
            runner.finish('completed_holdout',holdout_run=True,actual_gpu_training=True,
                research_positive=stats['research_positive'],teacher_retention=stats['teacher_retention'],
                candidate_admission=stats['admission'])
    except BaseException as exc:
        error=type(exc).__name__
        raise
    finally:
        with owned_writes(out,storage['small_receipt_bytes'],allowed_paths=['campaign-storage.json','failure.json']):
            write_new(out/'campaign-storage.json',dict(passed=error is None,output=parent.snapshot(),
                initial_owned_bytes=initial,foreign_receipts_reserved_bytes=foreign,
                actual_owned_bytes_at_exit=owned_size(out),review_reserved_bytes=storage['independent_reserve_bytes']))
            if error is not None:
                write_new(out/'failure.json',dict(kind=error,
                    attempted_games=len(runner.attempts) if runner else 0,
                    attempt_records='job-inputs/*.json are the immutable attempted-job records',
                    completed_games=len(runner.rows) if runner else 0,retry_allowed=False,
                    partial_work_not_complete_accounting=True))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['run'])
    parser.add_argument('--out',type=Path,default=OUT);args=parser.parse_args();run(args.out)
