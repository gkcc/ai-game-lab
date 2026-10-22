"""Bounded GPU MLP fit, using JAX/Optax after a recorded PyTorch transfer block."""
import argparse
import hashlib
import json
import os
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
from pathlib import Path
import time
import numpy as np
import jax
import jax.numpy as jnp
import optax
from .agent import Network, TEACHER_SHA, sha
from .experiment import read, save, integrity
from .features import CROPS, SCHEMA_VERSION


def forward(params, x):
    for i,(w,b) in enumerate(params):
        x=x@w+b
        if i<2:x=jax.nn.relu(x)
    return x


def host(tree):
    # Blocking host copies avoid the known WSL GetReadyFuture issue.
    return jax.tree_util.tree_map(lambda a:np.asarray(a),tree)


def weights_hash(params):
    h=hashlib.sha256()
    for a in jax.tree_util.tree_leaves(params):h.update(np.asarray(a).tobytes())
    return h.hexdigest()


def export(path, params, meta, mean, scale):
    arrays={'metadata':np.asarray(json.dumps(meta)), 'mean':mean, 'scale':scale}
    for i,(w,b) in enumerate(params):arrays['w'+str(i)]=np.asarray(w);arrays['b'+str(i)]=np.asarray(b)
    np.savez_compressed(path,**arrays)


def classification(y, pred):
    cm=np.zeros((5,5),np.int64);np.add.at(cm,(y,pred),1)
    return dict(accuracy=float(np.mean(y==pred)),confusion=cm.tolist(),
                recall=[float(cm[i,i]/cm[i].sum()) if cm[i].sum() else None for i in range(5)])


def train(out, smoke=False, allow_cpu=False):
    cfg=integrity(out);settings=cfg['training']
    assert (out/'backend-decision.json').is_file()
    data=out/('smoke-data' if smoke else 'data');dest=out/('smoke-model' if smoke else 'model');dest.mkdir(exist_ok=False)
    manifest=read(data/'manifest.json');assert sha(data/'dataset.npz')==manifest['dataset_sha256']
    tick=time.perf_counter();z=np.load(data/'dataset.npz',allow_pickle=False)
    x,y,val,seeds,games=[z[k] for k in ('x','y','validation','seed','game')]
    assert not set(seeds[~val])&set(seeds[val]) and not set(seeds)&set(cfg['test_seeds'])
    mu=x[~val].mean(0,dtype=np.float64).astype(np.float32);scale=np.maximum(x[~val].std(0,dtype=np.float64),1.).astype(np.float32)
    X=((x-mu)/scale).astype(np.float32);counts=np.bincount(y[~val],minlength=5)
    weights=np.zeros(5,np.float32);present=counts>0;weights[present]=1/np.sqrt(counts[present]);weights[present]/=weights[present].mean()
    try:device=jax.devices('gpu')[0]
    except RuntimeError:
        if not allow_cpu:raise
        device=jax.devices('cpu')[0]
    rng=np.random.default_rng(settings['seed']);dims=settings['architecture']
    initial=tuple((rng.uniform(-1/np.sqrt(a),1/np.sqrt(a),(a,b)).astype(np.float32),
                   rng.uniform(-1/np.sqrt(a),1/np.sqrt(a),(b,)).astype(np.float32)) for a,b in zip(dims,dims[1:]))
    meta=dict(schema_version=SCHEMA_VERSION,features=manifest['features'],schema_sha256=manifest['schema_sha256'],
        crops=list(CROPS),teacher_sha256=TEACHER_SHA,engine_sha256=cfg['engine_sha256'],engine_commit=cfg['engine_commit'],
        data_manifest_sha256=sha(data/'manifest.json'),training=settings,smoke=smoke,architecture=dims,
        framework='JAX/Optax',training_matmul_precision='JAX default',verification_matmul_precision='highest',
        initialization='NumPy seeded uniform +/- 1/sqrt(fan_in), including biases',
        normalization='train-only mean/std, floor 1',selection='initial before updates',
        source_sha256={p.name:sha(p) for p in sorted(Path(__file__).parent.glob('*.py'))})
    export(dest/'initial.npz',initial,meta,mu,scale);initial_sha=sha(dest/'initial.npz')
    params=jax.device_put(initial,device);xt=jax.device_put(X[~val],device);yt=jax.device_put(y[~val].astype(np.int32),device)
    xv=jax.device_put(X[val],device);yv=jax.device_put(y[val].astype(np.int32),device);cw=jax.device_put(weights,device)
    optimizer=optax.adam(settings['learning_rate']);state=optimizer.init(params)
    def loss(p,bx,by,mask):
        per=optax.softmax_cross_entropy_with_integer_labels(forward(p,bx),by)
        w=cw[by]*mask
        return jnp.sum(per*w)/jnp.sum(w)
    @jax.jit
    def step(p,s,xx,yy,ids,mask):
        value,grad=jax.value_and_grad(loss)(p,xx[ids],yy[ids],mask)
        updates,s=optimizer.update(grad,s,p)
        return optax.apply_updates(p,updates),s,value,optax.global_norm(grad)
    @jax.jit
    def evaluate(p,xx,yy):
        return jnp.mean(optax.softmax_cross_entropy_with_integer_labels(forward(p,xx),yy))
    host((params,xt,yt,xv,yv,state));preparation=time.perf_counter()-tick
    history=[];best_loss=float('inf');best=None;best_epoch=None;updates=nonzero=0
    train_seconds=val_seconds=compile_seconds=0.;batch=settings['batch_size']
    epochs=cfg['smoke']['epochs'] if smoke else settings['epochs'];gradient_device=None
    for epoch in range(epochs):
        permutation=rng.permutation(len(xt));t=time.perf_counter();losses=[]
        for start in range(0,len(xt),batch):
            ids=permutation[start:start+batch];n=len(ids)
            padded=np.pad(ids,(0,batch-n),mode='edge').astype(np.int32);mask=(np.arange(batch)<n).astype(np.float32)
            first=updates==0;compile_tick=time.perf_counter()
            params,state,value,norm=step(params,state,xt,yt,jax.device_put(padded,device),jax.device_put(mask,device))
            v,g=host((value,norm));gradient_device=str(next(iter(norm.devices())))
            if first:compile_seconds+=time.perf_counter()-compile_tick
            if not np.isfinite(v) or not np.isfinite(g):raise ValueError('Non-finite optimization')
            nonzero+=g>0;updates+=1;losses.append(float(v)*n)
        host(params);train_seconds+=time.perf_counter()-t
        t=time.perf_counter();vl=float(np.asarray(evaluate(params,xv,yv)));val_seconds+=time.perf_counter()-t
        history.append(dict(epoch=epoch+1,train_weighted_loss=sum(losses)/len(xt),validation_unweighted_loss=vl))
        if vl<best_loss:best_loss=vl;best_epoch=epoch+1;best=host(params)
        if (epoch+1)%10==0 or epoch+1==epochs:print(f'epoch {epoch+1}/{epochs}: validation CE {vl:.6f}',flush=True)
    assert updates and nonzero and weights_hash(initial)!=weights_hash(best)
    export(dest/'trained.npz',best,{**meta,'selection':'lowest validation unweighted cross entropy','best_epoch':best_epoch},mu,scale)
    assert sha(dest/'initial.npz')==initial_sha
    reports={};tick=time.perf_counter()
    for name,p in [('initial',initial),('trained',best)]:
        net=Network(dest/(name+'.npz'),sha(dest/(name+'.npz')));logits=net.logits(x,manifest['features']);pred=logits.argmax(1)
        # CUDA's default reduced-precision matmul can flip near-tied initial
        # logits. Compare deployment against the same weights in full FP32.
        with jax.default_matmul_precision('highest'):
            fp32=np.asarray(forward(jax.device_put(p,device),jax.device_put(X,device)))
        error=float(np.max(np.abs(logits-fp32)));disagree=int(np.count_nonzero(pred!=fp32.argmax(1)))
        assert error<1e-3 and disagree==0
        reports[name]=dict(train=classification(y[~val],pred[~val]),validation=classification(y[val],pred[val]),
            weights_sha256=weights_hash(p),checkpoint_sha256=sha(dest/(name+'.npz')),export_sha256=sha(dest/(name+'.npz')),
            export_validation=dict(samples=len(y),float32_max_abs_logit_error=error,argmax_disagreements=disagree,tolerance_float32=1e-3,
                deployment_precision='float64 CPU from float32 learned weights; first class wins exact tie',
                near_ties_under_1e_5=int(np.count_nonzero(np.diff(np.sort(logits,axis=1)[:,-2:],axis=1).ravel()<1e-5))),
            by_seed=[dict(seed=int(s),split='validation' if s in cfg['validation_seeds'] else 'train',samples=int(np.count_nonzero(seeds==s)),
                          **classification(y[seeds==s],pred[seeds==s])) for s in sorted(set(seeds))],
            by_game=[dict(game=manifest['games'][int(g)]['game'],samples=int(np.count_nonzero(games==g)),
                          **classification(y[games==g],pred[games==g])) for g in sorted(set(games))])
    save(dest/'report.json',dict(jax=jax.__version__,optax=optax.__version__,numpy=np.__version__,device_type=device.platform,
        model_device=str(next(iter(params[0][0].devices()))),batch_device=str(next(iter(xt.devices()))),gradient_device=gradient_device,
        cuda_training=device.platform=='gpu',optimizer_updates=updates,nonzero_gradient_updates=int(nonzero),
        weight_tensors_changed=sum(not np.array_equal(a,b) for a,b in zip(jax.tree_util.tree_leaves(initial),jax.tree_util.tree_leaves(best))),
        epochs=epochs,best_epoch=best_epoch,selected_checkpoint_updates=best_epoch*int(np.ceil(len(xt)/batch)),
        training_matmul_precision='JAX default',verification_matmul_precision='highest',
        history=history,class_counts=counts.tolist(),class_weights=weights.tolist(),
        majority_class=CROPS[int(counts.argmax())],majority_validation_accuracy=float(np.mean(y[val]==counts.argmax())),
        preparation_seconds=preparation,training_seconds_synchronized=train_seconds,first_compiled_update_seconds=compile_seconds,
        training_seconds_excluding_first_update=train_seconds-compile_seconds,validation_seconds_synchronized=val_seconds,
        export_verification_seconds=time.perf_counter()-tick,synchronization='blocking host copies',official_simulation_on_gpu=False,reports=reports))
    save(dest/'device-local.json',dict(device=device.device_kind,device_id=str(device)))
    print(json.dumps({k:read(dest/'report.json')[k] for k in ('cuda_training','optimizer_updates','best_epoch','training_seconds_synchronized')}))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--smoke',action='store_true');p.add_argument('--allow-cpu',action='store_true')
    a=p.parse_args();train(a.out.resolve(),a.smoke,a.allow_cpu)
