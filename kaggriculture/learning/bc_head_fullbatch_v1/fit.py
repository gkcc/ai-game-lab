"""One GPU fullbatch head fit, retaining every head and its developer metrics."""
import argparse
import json
import os
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE','false')
from pathlib import Path
import time
import numpy as np
import jax
import jax.numpy as jnp
import optax
from learning.bc_crop.agent import ROOT,Network,sha
from learning.bc_crop.experiment import read,save
from learning.bc_head_diagnostics.experiment import load
from learning.bc_head_v1.metrics import evaluate,gate,ce
from .experiment import verify
from .metrics import summarize,prediction_hash


def train(out):
    cfg=verify(out);settings=cfg['training']
    assert settings['epochs']==cfg['optimizer_update_budget']==20 and settings['batch_size']==9127
    assert (out/'pre-training-commit.json').exists()
    dest=out/'model';dest.mkdir(exist_ok=False);heads=dest/'heads';heads.mkdir()
    x,y,val,seeds,network,normalized64,hidden64,H,theta,baseline_logits,groups=load(cfg)
    source=ROOT/cfg['original_experiment'];parent=source/'model/trained.npz'
    with np.load(parent,allow_pickle=False) as z:original={k:z[k].copy() for k in z.files}
    device=jax.devices('gpu')[0]
    save(dest/'device-local.json',dict(device_kind=device.device_kind,device=str(device),platform=device.platform))
    counts=np.bincount(y[~val],minlength=5);weights=np.zeros(5,np.float32);present=counts>0
    weights[present]=1/np.sqrt(counts[present]);weights[present]/=weights[present].mean()
    initial=(original['w2'],original['b2']);best=tuple(a.copy() for a in initial)
    best_epoch=0;best_loss=ce(baseline_logits[val],y[val]);history=[]
    def retain(epoch,current,logits,selected):
        path=heads/f'step-{epoch:02d}.npz';assert not path.exists()
        assert all(np.isfinite(a).all() for a in current)
        np.savez_compressed(path,w2=current[0],b2=current[1],prediction=logits.argmax(1).astype(np.int8),step=np.array(epoch))
        row=dict(epoch=epoch,optimizer_updates=epoch,head_file=path.relative_to(dest).as_posix(),head_sha256=sha(path),
            selected_at_epoch=selected,changed_head_tensors=sum(not np.array_equal(a,b) for a,b in zip(initial,current)),
            **summarize(logits,y,val,seeds,groups,weights))
        history.append(row);save(dest/'history.json',history)
    retain(0,initial,baseline_logits,True)
    total_start=time.perf_counter();training_seconds=0.;nonzero=0;grad_devices=set()
    with jax.default_matmul_precision('highest'):
        normalized=((x-original['mean'])/original['scale']).astype(np.float32)
        hidden=jax.device_put(normalized,device)
        for i in (0,1):hidden=jax.nn.relu(hidden @ jax.device_put(original['w'+str(i)],device)+jax.device_put(original['b'+str(i)],device))
        hidden=np.asarray(hidden)
        ht=jax.device_put(hidden[~val],device);yt=jax.device_put(y[~val].astype(np.int32),device);cw=jax.device_put(weights,device)
        head=jax.device_put(initial,device);optimizer=optax.adam(settings['learning_rate']);state=optimizer.init(head)
        def loss(p):
            per=optax.softmax_cross_entropy_with_integer_labels(ht @ p[0]+p[1],yt)
            weight=cw[yt];return jnp.sum(per*weight)/jnp.sum(weight)
        @jax.jit
        def step(p,s):
            value,grad=jax.value_and_grad(loss)(p)
            updates,s=optimizer.update(grad,s,p)
            return optax.apply_updates(p,updates),s,value,optax.global_norm(grad)
        for epoch in range(1,21):
            start=time.perf_counter();head,state,value,norm=step(head,state)
            current=tuple(np.asarray(a).copy() for a in head);value=float(np.asarray(value));norm_value=float(np.asarray(norm))
            training_seconds+=time.perf_counter()-start
            assert np.isfinite(value) and np.isfinite(norm_value) and norm_value>0
            nonzero+=1;grad_devices.update(str(d) for d in norm.devices())
            logits=hidden64 @ current[0].astype(np.float64)+current[1].astype(np.float64)
            score=ce(logits[val],y[val]);selected=score<best_loss
            if selected:best=current;best_epoch=epoch;best_loss=score
            retain(epoch,current,logits,selected)
            history[-1].update(gpu_pre_update_weighted_loss=value,gpu_gradient_norm=norm_value)
            save(dest/'history.json',history)
            save(out/'training-progress.json',dict(epoch=epoch,updates=epoch,best_epoch=best_epoch))
            if epoch%5==0:print(f'fullbatch step{epoch}/20 CE{score:.8f} best{best_epoch}',flush=True)
        assert nonzero==20 and len(history)==21 and any(not np.array_equal(a,b) for a,b in zip(initial,current))
        payload={k:v.copy() for k,v in original.items()};payload['w2'],payload['b2']=best
        metadata=json.loads(str(original['metadata']));metadata.update(experiment=cfg['experiment'],training=settings,
            parent_checkpoint_sha256=sha(parent),best_epoch=best_epoch,selected_updates=best_epoch,
            selection=cfg['selection'],source_sha256={p.name:sha(p) for p in Path(__file__).parent.glob('*.py')})
        payload['metadata']=np.asarray(json.dumps(metadata));np.savez_compressed(dest/'trained.npz',**payload)
        exported=Network(dest/'trained.npz',sha(dest/'trained.npz')).logits(x,read(source/'data/manifest.json')['features'])
        assert np.array_equal(exported,hidden64 @ best[0].astype(np.float64)+best[1].astype(np.float64))
        fp32=np.asarray(jax.device_put(hidden,device) @ jax.device_put(best[0],device)+jax.device_put(best[1],device))
        export_error=float(np.max(abs(exported-fp32)));disagreement=int(np.count_nonzero(exported.argmax(1)!=fp32.argmax(1)))
        frozen={k:np.array_equal(original[k],payload[k]) for k in settings['frozen']};assert all(frozen.values())
    baseline=evaluate(baseline_logits,y,groups);candidate=evaluate(exported,y,groups)
    gates=gate(baseline,candidate,best_epoch)
    gates.update(frozen_tensors=all(frozen.values()),gpu_gradients_and_updates=nonzero==20,
                 export_fp32_parity=export_error<1e-3 and disagreement==0,all21_heads_saved=len(list(heads.glob('*.npz')))==21)
    report=dict(experiment=cfg['experiment'],development_only=True,new_independent_evidence=False,
        cuda_training=True,framework=dict(jax=jax.__version__,optax=optax.__version__,numpy=np.__version__),
        gpu_gradient_devices=sorted(grad_devices),gpu_batch_device=str(next(iter(ht.devices()))),
        optimizer_updates=20,nonzero_gradient_updates=nonzero,data_traversals=20,actual_batch_samples=len(ht),
        best_epoch=best_epoch,selected_checkpoint_updates=best_epoch,history_sha256=sha(dest/'history.json'),
        frozen_tensors_equal=frozen,selected_head_tensors_changed=sum(not np.array_equal(a,b) for a,b in zip(initial,best)),
        original_checkpoint_sha256=sha(parent),checkpoint_sha256=sha(dest/'trained.npz'),
        fixed_data_sha256=sha(source/'data/dataset.npz'),class_counts=counts.tolist(),class_weights=weights.tolist(),
        baseline=baseline,candidate=candidate,offline_gates=gates,offline_gate_passed=all(gates.values()),
        prediction_sha256=dict(baseline=prediction_hash(baseline_logits),candidate=prediction_hash(exported)),
        export_validation=dict(samples=len(y),cpu_reload_exact_logits=True,float32_max_abs_logit_error=export_error,argmax_disagreements=disagreement),
        synchronized_training_seconds=training_seconds,total_fit_seconds=time.perf_counter()-total_start,
        official_simulation_on_gpu=False,teacher_correction_enabled=False)
    save(dest/'report.json',report);save(dest/'checkpoint.json',dict(file='trained.npz',sha256=sha(dest/'trained.npz'),scope='crop choice only; fullbatch development candidate'))
    verify(out)
    print(json.dumps({'updates':20,'best_epoch':best_epoch,'offline_gate_passed':all(gates.values()),'gates':gates}))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);train(p.parse_args().out.resolve())
