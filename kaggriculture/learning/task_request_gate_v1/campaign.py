"""One whole-request train/dev/three-arm holdout, with a two-seed screen."""
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
from learning.task_request_gate_v1.experiment import OUT,read,ready,verify,gate_identity
from learning.task_request_gate_v1.games import wrapper,pair
from learning.task_request_gate_v1.policy import RequestModel
from learning.task_request_gate_v1.statistics import early_signal,training_signal,development_gate,holdout_statistics


def plan_cells(cfg,split):
    seeds=cfg[{'train':'train_seeds','dev':'validation_seeds','holdout':'test_seeds'}[split]]
    return [dict(split=split,seed=seed,opponent=opponent,side=side,
        window=None if split=='holdout' else (i+j+side)%3,
        student_parity=split=='train' and i==0 and j==0,
        teacher_parity=split=='holdout' and i==0 and j==0 and side==0)
        for i,seed in enumerate(seeds) for j,opponent in enumerate(cfg['opponents']) for side in (0,1)]


def plan(cell,policy):
    kind=('student' if cell['student_parity'] and policy=='baseline' else
          'teacher_task' if cell['teacher_parity'] and policy=='teacher_task' else None)
    return dict(cell,policy=policy,game=f"{cell['split']}-{cell['seed']}-{cell['opponent']}-{cell['side']}-{policy}",parity_kind=kind)


class Runner:
    def __init__(self,out,cfg):
        self.out=Path(out);self.cfg=cfg;self.start=time.monotonic()
        self.attempts=[];self.rows=[];self.certificates=[];self.extra_gate_evaluations=0
        self.environment=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')

    def limits(self):
        assert time.monotonic()-self.start<self.cfg['campaign_seconds'],'Campaign deadline'
        assert len(self.attempts)<=self.cfg['max_games']
        counts=Counter()
        for c in self.certificates:counts.update(c['total_model_counts'])
        assert counts['task_forwards']<=self.cfg['task_forward_budget']
        assert counts['crop_forwards']<=self.cfg['crop_forward_budget']
        assert counts['task_rows']+counts['crop_rows']<=self.cfg['neural_input_row_budget']
        assert counts['gate_evaluations']+self.extra_gate_evaluations<=self.cfg['gate_evaluation_budget']
        assert sum(c['policy_turns'] for c in self.certificates)<=self.cfg['policy_turn_budget']
        return dict(counts)

    def reserve(self,job):
        """All-work worst-case proof includes in-flight jobs before dispatch."""
        future=self.attempts+[job]
        neural=sum(p['policy']!='teacher_task' for p in future)
        teacher=sum(p['policy']=='teacher_task' for p in future)
        ps=sum(p['parity_kind']=='student' for p in future);pt=sum(p['parity_kind']=='teacher_task' for p in future)
        assert neural<=144 and teacher<=32 and ps<=2 and pt<=1
        caps=self.cfg['game_limits'];n=len(future)
        assert (2*neural+ps)*caps['task_forwards']<=self.cfg['task_forward_budget']
        assert (2*n+ps+pt)*caps['crop_forwards']<=self.cfg['crop_forward_budget']
        assert (2*n+ps+pt)*caps['neural_input_rows']<=self.cfg['neural_input_row_budget']
        assert 2*n*(caps['controller_calls']+719)+(ps+pt)*719<=self.cfg['policy_turn_budget']
        assert 2*sum(p['policy']=='gated' for p in future)*3+100<=self.cfg['gate_evaluation_budget']

    def execute(self,jobs):
        active={};queue=list(jobs);completed={};cursor=0
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
                    assert code==0,'Official job failure; inspect retained private log'
                    marker=read(self.out/'job-results'/(key+'.complete.json'))
                    rp=self.out/'job-results'/(key+'.row.json');ap=self.out/'job-results'/(key+'.audit.json')
                    assert marker['passed'] and marker['row_sha256']==sha(rp) and marker['certificate_sha256']==sha(ap)
                    row=read(rp);cert=read(ap);assert cert['passed']
                    row['complete_request_changes']=cert['complete_request_changes']
                    completed[key]=row;self.rows.append(row);self.certificates.append(cert)
                    print(json.dumps(dict(completed=len(self.rows),attempted=len(self.attempts),game=key)),flush=True)
                self.limits()
                while len(active)<self.cfg['max_workers'] and cursor<len(queue):
                    verify(self.out)
                    job=queue[cursor];self.reserve(job);cursor+=1
                    assert len(self.attempts)<self.cfg['max_games']
                    key=job['game'];bot=wrapper(self.out,job,self.cfg)
                    item=dict(job,bot_sha256=sha(bot),rival_sha256=sha(self.out/'sources'/(job['opponent']+'.py')),
                        gate_model_sha256=gate_identity(self.out) if job['policy']=='gated' else None)
                    write_new(self.out/'job-inputs'/(key+'.json'),item);self.attempts.append(item)
                    lp=self.out/'job-local'/(key+'.log');lp.parent.mkdir(exist_ok=True);log=lp.open('xb')
                    try:
                        opts=dict(creationflags=subprocess.CREATE_NO_WINDOW) if sys.platform=='win32' else dict(start_new_session=True)
                        process=subprocess.Popen([sys.executable,'-B','-X','utf8',str(Path(__file__).with_name('games.py')),
                            '--out',str(self.out),'--key',key],cwd=ROOT.parent,env=self.environment,stdout=log,stderr=subprocess.STDOUT,**opts)
                    except BaseException:log.close();raise
                    active[key]=dict(process=process,log=log,start=time.monotonic())
                if active:time.sleep(.2)
        finally:
            for state in active.values():terminate_owned(state['process']);state['log'].close()
        return completed

    def collect(self,split):
        cells=plan_cells(self.cfg,split);pairs=[]
        seeds=self.cfg['train_seeds' if split=='train' else 'validation_seeds']
        for i,seed in enumerate(seeds):
            current=[c for c in cells if c['seed']==seed]
            base=self.execute([plan(c,'baseline') for c in current])
            alternatives=[]
            for cell in current:
                name=plan(cell,'baseline')['game']
                if read(self.out/'job-results'/(name+'.audit.json'))['opportunities']:
                    alternatives.append(plan(cell,'intervention'))
            changed=self.execute(alternatives)
            pairs.extend(pair(self.out,base[plan(c,'baseline')['game']],changed.get(plan(c,'intervention')['game'])) for c in current)
            if split=='train' and i==1:
                gate=early_signal(pairs,seeds[:2]);write_new(self.out/'early-signal.json',gate)
                if not gate['passed']:
                    write_new(self.out/'train-pairs.json',pairs)
                    return pairs,False
        write_new(self.out/(split+'-pairs.json'),pairs)
        return pairs,True

    def finish(self,status,**extra):
        verify(self.out,history=True);counts=self.limits()
        write_new(self.out/'ledger.json',self.rows);write_new(self.out/'audit.json',self.certificates)
        write_new(self.out/'run-summary.json',dict(status=status,completed_games=len(self.rows),attempted_games=len(self.attempts),
            all_attempts_completed=len(self.rows)==len(self.attempts),live_official_transitions=len(self.rows)*719,
            primary_official_transitions=len(self.certificates)*719,model_counts=counts,
            extra_gate_evaluations=self.extra_gate_evaluations,policy_turns=sum(c['policy_turns'] for c in self.certificates),
            campaign_seconds=time.monotonic()-self.start,neural_weight_updates=0,gpu_operations=0,
            gate_fits=int((self.out/'gate-model.json').exists()),default_changed=False,automatic_teacher_admission=False,
            registration_sha256=sha(self.out/'registration.json'),freeze_sha256=sha(self.out/'freeze.json'),**extra))


def reload_gate(out):
    model=RequestModel.load(out/'gate-model.json',gate_identity(out));pairs=read(out/'train-pairs.json')
    predictions=[float(model.predict(p['x8'])) for p in pairs if p['eligible']]
    write_new(out/'gate-reload-parity.json',dict(gate_sha256=sha(out/'gate-model.json'),predictions=predictions,
        process='fresh interpreter',neural_forwards=0,gpu_operations=0))


def run(out=OUT):
    out=Path(out).resolve();cfg=ready(out)
    write_new(out/'campaign-started.json',dict(registration_sha256=sha(out/'registration.json'),freeze_sha256=sha(out/'freeze.json'),
        started_utc=datetime.now(timezone.utc).isoformat()))
    runner=Runner(out,cfg)
    try:
        train,screen=runner.collect('train')
        if not screen:
            runner.finish('stopped_early_signal',early_signal_passed=False,training_signal_evaluated=False,holdout_run=False);return
        signal=training_signal(train,cfg['train_seeds']);write_new(out/'training-signal.json',signal)
        if not signal['passed']:
            runner.finish('stopped_training_signal',early_signal_passed=True,training_signal_passed=False,holdout_run=False);return
        eligible=[p for p in train if p['eligible']]
        write_new(out/'fit-started.json',dict(training_pairs_sha256=sha(out/'train-pairs.json'),fit_count=1))
        tick=time.monotonic();model=RequestModel.fit([p['x8'] for p in eligible],[p['delta_margin']/1000 for p in eligible])
        fit_seconds=time.monotonic()-tick;assert fit_seconds<=cfg['training']['seconds'];export_start=time.monotonic()
        model.save(out/'gate-model.json')
        write_new(out/'gate-model-freeze.json',dict(gate_sha256=sha(out/'gate-model.json'),training_pairs_sha256=sha(out/'train-pairs.json'),fit_count=1))
        predictions=[float(model.predict(p['x8'])) for p in eligible];runner.extra_gate_evaluations+=len(predictions)
        subprocess.run([sys.executable,'-B','-m','learning.task_request_gate_v1.campaign','reload','--out',str(out)],
            cwd=ROOT,env=runner.environment,check=True,timeout=60,creationflags=subprocess.CREATE_NO_WINDOW if sys.platform=='win32' else 0)
        parity=read(out/'gate-reload-parity.json');runner.extra_gate_evaluations+=len(parity['predictions'])
        assert predictions==parity['predictions']
        write_new(out/'fit-result.json',dict(gate_sha256=sha(out/'gate-model.json'),samples=len(eligible),fit_seconds=fit_seconds,
            export_and_reload_seconds=time.monotonic()-export_start,parameter_count=9,solver='CPU float64 ridge one solve',reload_parity=True,
            training_mse=sum((p['delta_margin']/1000-y)**2 for p,y in zip(eligible,predictions))/len(eligible)))
        dev,_screen=runner.collect('dev')
        for p in dev:
            p['prediction']=float(model.predict(p['x8'])) if p['eligible'] else None;runner.extra_gate_evaluations+=int(p['eligible'])
        write_new(out/'dev-predictions.json',dev)
        gate=development_gate(dev,cfg['validation_seeds']);gate['gate_sha256']=gate_identity(out)
        write_new(out/'development-gate.json',gate)
        if not gate['passed']:
            runner.finish('stopped_development_gate',training_signal_passed=True,development_gate_passed=False,holdout_run=False);return
        results=[]
        for seed in cfg['test_seeds']:
            jobs=[plan(c,arm) for c in plan_cells(cfg,'holdout') if c['seed']==seed for arm in cfg['holdout_arms']]
            results.extend(runner.execute(jobs).values())
        statistics=holdout_statistics(results,cfg);write_new(out/'holdout-statistics.json',statistics)
        runner.finish('completed_holdout',training_signal_passed=True,development_gate_passed=True,holdout_run=True,
            research_positive=statistics['research_positive'],teacher_retention=statistics['teacher_retention'],teacher_superiority=statistics['teacher_superiority'])
    except BaseException as exc:
        write_new(out/'failure.json',dict(kind=type(exc).__name__,attempts=runner.attempts,completed_games=len(runner.rows),
            attempted_games=len(runner.attempts),retry_allowed=False,partial_work_not_complete_accounting=True))
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['run','reload']);p.add_argument('--out',type=Path,default=OUT)
    a=p.parse_args();reload_gate(a.out) if a.command=='reload' else run(a.out)
