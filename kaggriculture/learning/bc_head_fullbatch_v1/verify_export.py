"""Native reload: independently select and verify all21 retained output heads."""
import argparse
from pathlib import Path
import numpy as np
from learning.bc_crop.agent import ROOT,Network,sha
from learning.bc_crop.experiment import read,save
from learning.bc_head_diagnostics.experiment import load
from learning.bc_head_v1.metrics import compare_metrics,evaluate,gate,ce
from .experiment import verify
from .metrics import summarize,prediction_hash


def run(out):
    cfg=verify(out);dest=out/'model';report=read(dest/'report.json');history=read(dest/'history.json')
    assert not (out/'reload-verification.json').exists()
    assert sha(dest/'history.json')==report['history_sha256'] and sha(dest/'trained.npz')==report['checkpoint_sha256']
    assert len(history)==21 and [row['epoch'] for row in history]==list(range(21))
    x,y,val,seeds,net,normalized,h,H,theta,baseline_logits,groups=load(cfg)
    with np.load(ROOT/cfg['original_experiment']/'model/trained.npz',allow_pickle=False) as f:parent={k:f[k].copy() for k in f.files}
    weights=np.asarray(report['class_weights'],np.float32);counts=np.bincount(y[~val],minlength=5)
    expected_weights=np.zeros(5,np.float32);present=counts>0;expected_weights[present]=1/np.sqrt(counts[present]);expected_weights[present]/=expected_weights[present].mean()
    assert np.array_equal(weights,expected_weights) and report['class_counts']==counts.tolist()
    best=0;best_loss=float('inf');selected=None;previous=None;changed_steps=0;maximum_metric_error=0.;receipts=[]
    for epoch,row in enumerate(history):
        path=dest/row['head_file'];assert path.parent==dest/'heads' and sha(path)==row['head_sha256']
        with np.load(path,allow_pickle=False) as f:
            assert int(f['step'])==epoch
            w,b=f['w2'].copy(),f['b2'].copy();saved_prediction=f['prediction'].copy()
        assert w.shape==(64,5) and b.shape==(5,) and w.dtype==b.dtype==np.float32
        assert np.isfinite(w).all() and np.isfinite(b).all()
        if epoch==0:assert np.array_equal(w,parent['w2']) and np.array_equal(b,parent['b2'])
        else:
            assert any(not np.array_equal(a,c) for a,c in zip(previous,(w,b)));changed_steps+=1
        previous=(w,b)
        logits=h @ w.astype(np.float64)+b.astype(np.float64)
        assert np.array_equal(logits.argmax(1),saved_prediction) and prediction_hash(logits)==row['prediction_sha256']
        actual=summarize(logits,y,val,seeds,groups,weights)
        maximum_metric_error=max(maximum_metric_error,compare_metrics(actual['groups'],row['groups']))
        assert np.isclose(actual['train_weighted_cross_entropy'],row['train_weighted_cross_entropy'],atol=1e-12,rtol=1e-12)
        for name,margin in actual['strawberry_melon_margins'].items():
            for key,value in margin.items():
                saved=row['strawberry_melon_margins'][name][key]
                if key in ('minimum','mean','maximum'):assert np.isclose(value,saved,atol=1e-12,rtol=1e-12)
                else:assert value==saved
        changed=sum(not np.array_equal(a,c) for a,c in zip((parent['w2'],parent['b2']),(w,b)))
        assert changed==row['changed_head_tensors'] and row['optimizer_updates']==epoch
        loss=ce(logits[val],y[val]);is_best=loss<best_loss
        assert is_best==row['selected_at_epoch']
        if is_best:best=epoch;best_loss=loss;selected=(w,b)
        receipts.append(dict(epoch=epoch,head_sha256=sha(path),prediction_sha256=prediction_hash(logits),validation_ce=loss))
    assert best==report['best_epoch'] and changed_steps==20
    with np.load(dest/'trained.npz',allow_pickle=False) as f:
        for key in cfg['training']['frozen']:
            assert f[key].dtype==parent[key].dtype and np.array_equal(f[key],parent[key]),key
        assert np.array_equal(f['w2'],selected[0]) and np.array_equal(f['b2'],selected[1])
    names=read(ROOT/cfg['original_experiment']/'data/manifest.json')['features']
    candidate=Network(dest/'trained.npz',sha(dest/'trained.npz')).logits(x,names)
    assert prediction_hash(candidate)==report['prediction_sha256']['candidate']
    baseline_metrics=evaluate(baseline_logits,y,groups);candidate_metrics=evaluate(candidate,y,groups)
    compare_metrics(baseline_metrics,report['baseline']);compare_metrics(candidate_metrics,report['candidate'])
    gates=gate(baseline_metrics,candidate_metrics,best)
    assert all(report['offline_gates'][k]==v for k,v in gates.items())
    assert report['offline_gate_passed']==all(report['offline_gates'].values())
    verify(out)
    save(out/'reload-verification.json',dict(passed=True,fresh_native_process=True,samples=len(y),heads=21,changed_steps=changed_steps,
        best_epoch=best,checkpoint_sha256=sha(dest/'trained.npz'),all_heads_and_predictions_recomputed=True,
        changed_predictions=int(np.count_nonzero(candidate.argmax(1)!=baseline_logits.argmax(1))),
        frozen_arrays_exact=True,ce_only_metric_tolerance=dict(atol=1e-12,rtol=1e-12),maximum_metric_ce_error=maximum_metric_error,
        strict_selection_reproduced=True,offline_gate_passed=report['offline_gate_passed'],heads_receipts=receipts))
    print({'passed':True,'heads':21,'changed_steps':20,'best_epoch':best,'offline_gate_passed':report['offline_gate_passed']})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);run(p.parse_args().out.resolve())
