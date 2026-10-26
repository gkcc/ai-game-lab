"""One bounded train/dev/holdout campaign; no retries, tuning, or promotion."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from learning.bc_crop.agent import ROOT, sha
from learning.task_bc_holdout_v1.games import write_new, terminate_owned
from learning.task_reward_gate_v1.experiment import OUT, read, ready, verify, gate_identity
from learning.task_reward_gate_v1.games import wrapper, pair
from learning.task_reward_gate_v1.policy import GateModel
from learning.task_reward_gate_v1.statistics import training_signal, development_gate, holdout_statistics


def plan_cells(cfg,split):
    seeds=cfg[{'train':'train_seeds','dev':'validation_seeds','holdout':'test_seeds'}[split]]
    return [dict(split=split,seed=seed,opponent=opponent,side=side,
                 window=None if split=='holdout' else (i+j+side)%3,
                 baseline_parity=split=='train' and i==0 and j==0)
            for i,seed in enumerate(seeds) for j,opponent in enumerate(cfg['opponents']) for side in (0,1)]


def plan(cell,policy):
    return dict(cell,policy=policy,game=f"{cell['split']}-{cell['seed']}-{cell['opponent']}-{cell['side']}-{policy}",
                baseline_parity=cell['baseline_parity'] and policy=='baseline')


class Runner:
    def __init__(self,out,cfg):
        self.out=Path(out);self.cfg=cfg;self.start=time.monotonic()
        self.attempts=[];self.rows=[];self.certificates=[];self.extra_gate_evaluations=0
        self.environment=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')

    def limits(self):
        assert time.monotonic()-self.start<self.cfg['campaign_seconds'],'Campaign deadline'
        start=datetime.fromisoformat(read(self.out/'implementation-start.json')['started_utc'])
        assert (datetime.now(timezone.utc)-start).total_seconds()<14400,'Phase deadline'
        assert len(self.attempts)<=self.cfg['max_games']
        counts=Counter()
        for c in self.certificates:counts.update(c['total_model_counts'])
        assert counts['task_forwards']<=self.cfg['task_forward_budget']
        assert counts['crop_forwards']<=self.cfg['crop_forward_budget']
        assert counts['task_rows']+counts['crop_rows']<=self.cfg['neural_input_row_budget']
        assert counts['gate_evaluations']+self.extra_gate_evaluations<=self.cfg['gate_evaluation_budget']
        assert sum(c['policy_turns'] for c in self.certificates)<=self.cfg['policy_turn_budget']
        return dict(counts)

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
                    job=queue[cursor];cursor+=1;assert len(self.attempts)<self.cfg['max_games']
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
            for state in active.values():
                terminate_owned(state['process']);state['log'].close()
        return completed

    def collect(self,split):
        cells=plan_cells(self.cfg,split)
        base=self.execute([plan(c,'baseline') for c in cells])
        alternatives=[]
        for cell in cells:
            name=plan(cell,'baseline')['game']
            if read(self.out/'job-results'/(name+'.audit.json'))['opportunities']:
                alternatives.append(plan(cell,'intervention'))
        changed=self.execute(alternatives)
        pairs=[pair(self.out,base[plan(c,'baseline')['game']],changed.get(plan(c,'intervention')['game'])) for c in cells]
        write_new(self.out/(split+'-pairs.json'),pairs)
        return pairs

    def finish(self,status,**extra):
        cfg=self.cfg
        verify(self.out,history=True)
        counts=self.limits()
        write_new(self.out/'ledger.json',self.rows)
        write_new(self.out/'audit.json',self.certificates)
        write_new(self.out/'run-summary.json',dict(status=status,completed_games=len(self.rows),attempted_games=len(self.attempts),
            all_attempts_completed=len(self.rows)==len(self.attempts),
            live_official_transitions=len(self.rows)*719,primary_official_transitions=len(self.certificates)*719,
            model_counts=counts,extra_gate_evaluations=self.extra_gate_evaluations,
            policy_turns=sum(c['policy_turns'] for c in self.certificates),
            campaign_seconds=time.monotonic()-self.start,neural_weight_updates=0,gpu_operations=0,
            gate_fits=int((self.out/'gate-model.json').exists()),default_changed=False,teacher_admission=False,
            registration_sha256=sha(self.out/'registration.json'),freeze_sha256=sha(self.out/'freeze.json'),**extra))


def reload_gate(out):
    model=GateModel.load(out/'gate-model.json',gate_identity(out));pairs=read(out/'train-pairs.json')
    predictions=[float(model.predict(p['x12'])) for p in pairs if p['eligible']]
    write_new(out/'gate-reload-parity.json',dict(gate_sha256=sha(out/'gate-model.json'),predictions=predictions,
              process='fresh interpreter',neural_forwards=0,gpu_operations=0))


def run(out=OUT):
    out=Path(out).resolve();cfg=ready(out)
    write_new(out/'campaign-started.json',dict(registration_sha256=sha(out/'registration.json'),freeze_sha256=sha(out/'freeze.json'),
              started_utc=datetime.now(timezone.utc).isoformat()))
    runner=Runner(out,cfg)
    try:
        train=runner.collect('train');signal=training_signal(train,cfg['train_seeds'])
        write_new(out/'training-signal.json',signal)
        if not signal['passed']:
            runner.finish('stopped_training_signal',training_signal_passed=False,holdout_run=False);return
        eligible=[p for p in train if p['eligible']];tick=time.monotonic()
        write_new(out/'fit-started.json',dict(training_pairs_sha256=sha(out/'train-pairs.json'),fit_count=1))
        model=GateModel.fit([p['x12'] for p in eligible],[p['delta_own']/1000 for p in eligible])
        fit_seconds=time.monotonic()-tick
        assert fit_seconds<=cfg['training']['seconds']
        export_start=time.monotonic()
        model.save(out/'gate-model.json')
        write_new(out/'gate-model-freeze.json',dict(gate_sha256=sha(out/'gate-model.json'),
            training_pairs_sha256=sha(out/'train-pairs.json'),fit_count=1))
        predictions=[float(model.predict(p['x12'])) for p in eligible]
        runner.extra_gate_evaluations+=len(predictions)
        subprocess.run([sys.executable,'-B','-m','learning.task_reward_gate_v1.campaign','reload','--out',str(out)],
            cwd=ROOT,env=runner.environment,check=True,timeout=60,
            creationflags=subprocess.CREATE_NO_WINDOW if sys.platform=='win32' else 0)
        parity=read(out/'gate-reload-parity.json');runner.extra_gate_evaluations+=len(parity['predictions'])
        assert predictions==parity['predictions']
        write_new(out/'fit-result.json',dict(gate_sha256=sha(out/'gate-model.json'),samples=len(eligible),
            fit_seconds=fit_seconds,export_and_reload_seconds=time.monotonic()-export_start,
            parameter_count=13,solver='CPU float64 ridge one solve',
            reload_parity=True,training_mse=sum((p['delta_own']/1000-y)**2 for p,y in zip(eligible,predictions))/len(eligible)))
        dev=runner.collect('dev')
        for p in dev:
            p['prediction']=float(model.predict(p['x12'])) if p['eligible'] else None
            runner.extra_gate_evaluations+=int(p['eligible'])
        write_new(out/'dev-predictions.json',dev)
        gate=development_gate(dev,cfg['validation_seeds'])
        gate['gate_sha256']=gate_identity(out)
        write_new(out/'development-gate.json',gate)
        if not gate['passed']:
            runner.finish('stopped_development_gate',training_signal_passed=True,development_gate_passed=False,holdout_run=False);return
        holdout_jobs=[plan(c,arm) for c in plan_cells(cfg,'holdout') for arm in ('baseline','gated')]
        rows=runner.execute(holdout_jobs)
        statistics=holdout_statistics(list(rows.values()),cfg);write_new(out/'holdout-statistics.json',statistics)
        runner.finish('completed_holdout',training_signal_passed=True,development_gate_passed=True,holdout_run=True,
                      research_positive=statistics['research_positive'])
    except BaseException as exc:
        write_new(out/'failure.json',dict(kind=type(exc).__name__,attempts=runner.attempts,
                  completed_games=len(runner.rows),attempted_games=len(runner.attempts),retry_allowed=False))
        # Any partial raw traces and the originally frozen source survive. Do not
        # claim complete accounting for a killed, failed or partially audited job.
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['run','reload'])
    parser.add_argument('--out',type=Path,default=OUT);args=parser.parse_args()
    reload_gate(args.out) if args.command=='reload' else run(args.out)
