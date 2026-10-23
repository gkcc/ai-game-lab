"""Fresh native process verifies saved candidate tensors, predictions and gates."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import numpy as np
from learning.bc_crop.agent import ROOT,Network,sha
from learning.bc_crop.experiment import read,save
from .experiment import verify
from .metrics import masks,evaluate,gate,compare_metrics


def run(out):
    cfg=verify(out);source=ROOT/cfg['original_experiment'];dest=out/'model';report=read(dest/'report.json')
    assert not (out/'reload-verification.json').exists()
    parent=source/'model/trained.npz';path=dest/'trained.npz';assert sha(path)==report['checkpoint_sha256']
    with np.load(parent,allow_pickle=False) as old,np.load(path,allow_pickle=False) as new:
        frozen={k:np.array_equal(old[k],new[k]) for k in cfg['training']['frozen']}
        assert all(frozen.values())
        changed=[k for k in ('w0','b0','w1','b1','w2','b2','mean','scale') if not np.array_equal(old[k],new[k])]
    assert set(changed)<=set(['w2','b2'])
    with np.load(source/'data/dataset.npz',allow_pickle=False) as z:x,y,val=[z[k] for k in ('x','y','validation')]
    with gzip.open(source/'data/samples.jsonl.gz','rt',encoding='utf-8') as f:samples=[json.loads(l) for l in f]
    names=read(source/'data/manifest.json')['features'];a=Network(parent,sha(parent)).logits(x,names);b=Network(path,sha(path)).logits(x,names)
    groups=masks(x,y,val,a.argmax(1),[s['outcome'] for s in samples])
    baseline=evaluate(a,y,groups);candidate=evaluate(b,y,groups)
    ce_errors={name:compare_metrics(actual,report[name]) for name,actual in [('baseline',baseline),('candidate',candidate)]}
    assert {name:hashlib.sha256(logits.argmax(1).astype('<i8').tobytes()).hexdigest() for name,logits in [('baseline',a),('candidate',b)]}==report['prediction_sha256']
    assert hashlib.sha256(np.flatnonzero(groups['selected_strawberry_errors']).astype('<i8').tobytes()).hexdigest()==report['selected_error_row_indices_sha256']
    checks=gate(baseline,candidate,report['best_epoch'])
    assert all(report['offline_gates'][k]==v for k,v in checks.items())
    # Fresh reload validates inference even if the development gate was negative.
    save(out/'reload-verification.json',dict(passed=True,fresh_native_process=True,samples=len(y),
        changed_tensors=changed,frozen_tensors_identical=frozen,statistics_and_gate_recomputed=True,
        checkpoint_sha256=sha(path),changed_predictions=int(np.count_nonzero(a.argmax(1)!=b.argmax(1))),
        per_row_predictions_exact=True,cross_platform_ce_max_errors=ce_errors,ce_tolerance=dict(atol=1e-12,rtol=1e-12),
        offline_gate_passed=report['offline_gate_passed'],teacher_correction_enabled=False))
    print(json.dumps(read(out/'reload-verification.json')))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True)
    run(p.parse_args().out.resolve())
