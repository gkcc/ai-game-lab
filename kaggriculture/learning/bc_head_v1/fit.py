"""One GPU output-head fit, frozen features/data/controller and epoch0 selection."""
import argparse
import gzip
import hashlib
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
from learning.bc_crop.features import CROPS
from learning.bc_crop.experiment import read,save
from .experiment import verify
from .metrics import masks,evaluate,gate,ce


def train(out):
    cfg=verify(out);settings=cfg['training'];source=ROOT/cfg['original_experiment']
    dest=out/'model';dest.mkdir(exist_ok=False)
    with np.load(source/'data/dataset.npz',allow_pickle=False) as z:
        x,y,val,seeds=[z[k].copy() for k in ('x','y','validation','seed')]
    assert len(y)==11483 and int((~val).sum())==9127 and int(val.sum())==2356
    assert set(seeds[~val])==set(cfg['train_seeds']) and set(seeds[val])==set(cfg['validation_seeds'])
    with gzip.open(source/'data/samples.jsonl.gz','rt',encoding='utf-8') as f:samples=[json.loads(l) for l in f]
    assert len(samples)==len(y)
    assert all(s['seed']==int(seed) and s['label']==CROPS[int(label)] for s,seed,label in zip(samples,seeds,y))
    manifest=read(source/'data/manifest.json');parent=source/'model/trained.npz'
    network=Network(parent,sha(parent));baseline_logits=network.logits(x,manifest['features'])
    # This is the exact prior160 eligible training error set; no candidate-based filtering.
    groups=masks(x,y,val,baseline_logits.argmax(1),[s['outcome'] for s in samples])
    assert int(groups['selected_strawberry_errors'].sum())==160
    assert int(groups['validation_unseen'].sum())==1324
    indices=np.flatnonzero(groups['selected_strawberry_errors']).astype('<i8')
    baseline=evaluate(baseline_logits,y,groups)
    with np.load(parent,allow_pickle=False) as z:original={k:z[k].copy() for k in z.files}
    device=jax.devices('gpu')[0]
    save(dest/'device-local.json',dict(device_kind=device.device_kind,device=str(device),platform=device.platform))
    # No GPU discovery is counted as training: actual gradients/updates are recorded below.
    hidden64=(x.astype(np.float64)-network.mean)/network.scale
    for w,b in network.weights[:2]:hidden64=np.maximum(hidden64@w+b,0)
    tick=time.perf_counter()
    with jax.default_matmul_precision('highest'):
        normalized=((x-original['mean'])/original['scale']).astype(np.float32)
        hidden=jax.device_put(normalized,device)
        for i in (0,1):
            hidden=jax.nn.relu(hidden@jax.device_put(original['w'+str(i)],device)+jax.device_put(original['b'+str(i)],device))
        hidden=np.asarray(hidden)  # synchronizes preparation, not a fitted representation
        ht=jax.device_put(hidden[~val],device);yt=jax.device_put(y[~val].astype(np.int32),device)
        counts=np.bincount(y[~val],minlength=5);weights=np.zeros(5,np.float32);present=counts>0
        weights[present]=1/np.sqrt(counts[present]);weights[present]/=weights[present].mean()
        cw=jax.device_put(weights,device)
        initial=(original['w2'],original['b2']);head=jax.device_put(initial,device)
        optimizer=optax.adam(settings['learning_rate']);state=optimizer.init(head)
        def loss(p,bx,by,mask):
            per=optax.softmax_cross_entropy_with_integer_labels(bx@p[0]+p[1],by)
            weight=cw[by]*mask;return jnp.sum(per*weight)/jnp.sum(weight)
        @jax.jit
        def step(p,s,ids,mask):
            value,grad=jax.value_and_grad(loss)(p,ht[ids],yt[ids],mask)
            updates,s=optimizer.update(grad,s,p)
            return optax.apply_updates(p,updates),s,value,optax.global_norm(grad)
        rng=np.random.default_rng(settings['seed']);updates=nonzero=0;history=[]
        best=tuple(v.copy() for v in initial);best_epoch=0;best_loss=ce(baseline_logits[val],y[val])
        history.append(dict(epoch=0,validation_ce=best_loss,selected=True))
        training_seconds=0.;evaluation_seconds=0.;grad_devices=set();batch=settings['batch_size']
        for epoch in range(1,settings['epochs']+1):
            ids=rng.permutation(len(ht));start=time.perf_counter()
            for offset in range(0,len(ids),batch):
                part=ids[offset:offset+batch];n=len(part)
                padded=np.pad(part,(0,batch-n),mode='edge').astype(np.int32);mask=(np.arange(batch)<n).astype(np.float32)
                head,state,value,norm=step(head,state,jax.device_put(padded,device),jax.device_put(mask,device))
                value,norm_value=float(np.asarray(value)),float(np.asarray(norm))
                assert np.isfinite(value) and np.isfinite(norm_value)
                updates+=1;nonzero+=norm_value>0;grad_devices.update(str(d) for d in norm.devices())
            current=tuple(np.asarray(a).copy() for a in head);training_seconds+=time.perf_counter()-start
            start=time.perf_counter();logits=hidden64@current[0].astype(np.float64)+current[1].astype(np.float64)
            validation_ce=ce(logits[val],y[val]);selected=validation_ce<best_loss
            if selected:best=current;best_epoch=epoch;best_loss=validation_ce
            history.append(dict(epoch=epoch,validation_ce=validation_ce,selected_at_epoch=selected))
            evaluation_seconds+=time.perf_counter()-start
            save(out/'training-progress.json',dict(epoch=epoch,updates=updates,best_epoch=best_epoch))
            if epoch%5==0:print(f'head epoch {epoch}/20, developer CE {validation_ce:.6f}, best {best_epoch}',flush=True)
        assert updates==cfg['optimizer_update_budget']==720 and nonzero>0
        assert any(not np.array_equal(a,b) for a,b in zip(initial,current))
        payload={k:v.copy() for k,v in original.items()};payload['w2'],payload['b2']=best
        metadata=json.loads(str(original['metadata']))
        metadata.update(training=settings,selection='minimum deployment float64 unweighted developer validation CE including epoch0',
                        parent_checkpoint_sha256=sha(parent),best_epoch=best_epoch,experiment='bc-output-head-v1',
                        training_matmul_precision='highest',source_sha256={p.name:sha(p) for p in Path(__file__).parent.glob('*.py')})
        payload['metadata']=np.asarray(json.dumps(metadata))
        np.savez_compressed(dest/'trained.npz',**payload)
        loaded=Network(dest/'trained.npz',sha(dest/'trained.npz'));exported=loaded.logits(x,manifest['features'])
        selected_logits=hidden64@best[0].astype(np.float64)+best[1].astype(np.float64)
        assert np.array_equal(exported,selected_logits)
        frozen={k:np.array_equal(original[k],payload[k]) for k in settings['frozen']};assert all(frozen.values())
        fp32=np.asarray(jax.device_put(hidden,device)@jax.device_put(best[0],device)+jax.device_put(best[1],device))
        export_error=float(np.max(np.abs(exported-fp32)));disagreement=int(np.count_nonzero(exported.argmax(1)!=fp32.argmax(1)))
        candidate=evaluate(exported,y,groups);gates=gate(baseline,candidate,best_epoch)
        gates.update(frozen_tensors=all(frozen.values()),gpu_gradients_and_updates=nonzero>0,
                     export_fp32_parity=export_error<1e-3 and disagreement==0)
    report=dict(experiment='bc-output-head-v1',development_only=True,new_independent_evidence=False,
        cuda_training=True,framework=dict(jax=jax.__version__,optax=optax.__version__,numpy=np.__version__),
        gpu_gradient_devices=sorted(grad_devices),gpu_batch_device=str(next(iter(ht.devices()))),
        optimizer_updates=updates,nonzero_gradient_updates=nonzero,best_epoch=best_epoch,
        selected_checkpoint_updates=best_epoch*36,frozen_tensors_equal=frozen,
        selected_head_tensors_changed=sum(not np.array_equal(a,b) for a,b in zip(initial,best)),
        final_head_tensors_changed=sum(not np.array_equal(a,b) for a,b in zip(initial,current)),
        original_checkpoint_sha256=sha(parent),checkpoint_sha256=sha(dest/'trained.npz'),
        selected_error_row_indices_sha256=hashlib.sha256(indices.tobytes()).hexdigest(),
        fixed_data_sha256=sha(source/'data/dataset.npz'),class_counts=counts.tolist(),class_weights=weights.tolist(),
        baseline=baseline,candidate=candidate,history=history,offline_gates=gates,offline_gate_passed=all(gates.values()),
        prediction_sha256={name:hashlib.sha256(logits.argmax(1).astype('<i8').tobytes()).hexdigest() for name,logits in [('baseline',baseline_logits),('candidate',exported)]},
        by_seed=[dict(seed=int(s),role='developer-validation' if s in cfg['validation_seeds'] else 'original-training',
            baseline=evaluate(baseline_logits,y,{'seed':seeds==s})['seed'],candidate=evaluate(exported,y,{'seed':seeds==s})['seed']) for s in sorted(set(seeds))],
        export_validation=dict(samples=len(y),cpu_reload_exact_logits=True,float32_max_abs_logit_error=export_error,argmax_disagreements=disagreement),
        synchronized_training_seconds=training_seconds,selection_evaluation_seconds=evaluation_seconds,
        total_fit_seconds=time.perf_counter()-tick,official_simulation_on_gpu=False,teacher_correction_enabled=False)
    save(dest/'report.json',report)
    save(dest/'checkpoint.json',dict(file='trained.npz',sha256=sha(dest/'trained.npz'),scope='crop choice only; output-head development candidate'))
    verify(out)
    print(json.dumps({'best_epoch':best_epoch,'updates':updates,'offline_gate_passed':all(gates.values()),'gates':gates}))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True)
    train(p.parse_args().out.resolve())
