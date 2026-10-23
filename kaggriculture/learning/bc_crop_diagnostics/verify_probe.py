"""Real historical developer observations test the single diagnostic override.

No new games/training. These regressions verify representation and intervention
boundaries, not the probe's economic value.
"""
import argparse
from collections import Counter
import gzip
import json
from pathlib import Path
import numpy as np
from learning.bc_crop.agent import CropAgent, sha
from learning.bc_crop.features import CROPS
from learning.bc_crop.audit import stable
from learning.bc_crop.experiment import read, save, integrity
from decision.audit_market_teacher import observation
from .probe import StrawberryCorrection


def dispatch_regression():
    from .panel import bounded_jobs
    from threading import Barrier, Lock
    barrier=Barrier(4);lock=Lock();started=[];finished=[];recorded=[]
    def execute(job):
        with lock:started.append(job[0])
        barrier.wait(timeout=10)
        if job[0]==0:raise RuntimeError('deliberate worker failure')
        # Hold the other in-flight jobs until the failing future is observable.
        import time
        time.sleep(.05)
        with lock:finished.append(job[0])
        return job[0]
    errors=bounded_jobs([(i,) for i in range(36)],execute,lambda job,value:recorded.append(value),4)
    assert sorted(started)==[0,1,2,3] and sorted(finished)==sorted(recorded)==[1,2,3]
    assert errors==[{'game':0,'exception_type':'RuntimeError'}]
    success=[]
    assert bounded_jobs([(i,) for i in range(7)],lambda job:job[0],lambda job,value:success.append(value),3)==[]
    assert sorted(success)==list(range(7))
    return dict(failed_panel_started=4,failed_panel_successes_retained=3,unstarted_after_failure=32,
                successful_panel_completed=7,all_owned_threads_joined=True)


def run(source, destination):
    integrity(source)
    assert not destination.exists()
    model=source/'model/trained.npz';digest=sha(model)
    rows=[r for r in read(source/'trained-smoke-ledger.json') if r['policy']=='trained']
    assert len(rows)==2
    counts=Counter();files={}
    for row in rows:
        raw=source/'games'/row['game']/'replay.json.gz'
        assert sha(raw)==row['replay_gzip_sha256'];files[row['game']]=sha(raw)
        with gzip.open(raw,'rt',encoding='utf-8') as f: replay=json.load(f)
        student=CropAgent('model',model,digest);probe=StrawberryCorrection(model,digest)
        network=probe.network
        for before in replay['steps'][:-1]:
            o=observation(before,row['side']);cfg=replay['configuration']
            baseline=student(o,cfg);actual=probe(o,cfg)
            counts['observations']+=1
            assert probe.network is network and probe.last['checkpoint_sha256']==digest
            for record in probe.records:
                predicted=CROPS[int(np.argmax(record['model_logits']))]
                expected='STRAWBERRY' if record['teacher']=='STRAWBERRY' else predicted
                assert record['model_prediction']==predicted and record['chosen']==expected
                assert record['correction_applied']==(expected!=predicted)
                counts['corrected_calls' if record['correction_applied'] else 'unchanged_calls']+=1
            if not any(r['correction_applied'] for r in probe.records):
                assert baseline==actual
                records=[]
                for record in probe.records:
                    copy=dict(record)
                    for key in ('model_logits','model_prediction','correction_applied','logits_semantics'):copy.pop(key)
                    copy['source']='model';records.append(copy)
                assert stable(records)==stable(student.records)
                counts['identical_nonintervention_turns']+=1
        # Reusing the instance after an entire other trajectory must match fresh.
        fresh=StrawberryCorrection(model,digest)
        for frame in replay['steps'][:20]:
            o=observation(frame,row['side'])
            assert probe(o,cfg)==fresh(o,cfg) and stable(probe.last)==stable(fresh.last)
        counts['cross_game_reset_checks']+=1
    assert counts['corrected_calls']>0 and counts['unchanged_calls']>0 and counts['identical_nonintervention_turns']>0
    integrity(source)
    result=dict(passed=True,counts=dict(counts),dispatch=dispatch_regression(),source_replay_sha256=files,checkpoint_sha256=digest,
                probe_sha256=sha(Path(__file__).with_name('probe.py')),new_games=0,training_runs=0,
                scope='Original student developer observations, both seats. Conditional teacher override only; no claim of official economic gain.')
    save(destination,result);print(json.dumps(result))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();run(a.source.resolve(),a.out.resolve())
