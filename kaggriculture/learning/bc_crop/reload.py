"""Fresh-process reload and deployment failure contracts; no fitting here."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import jax
from .agent import Network, sha
from .experiment import read, save
from .train import forward, weights_hash


def run(out, scratch):
    data=out/'data';dest=out/'model';manifest=read(data/'manifest.json')
    with np.load(data/'dataset.npz') as z:x=z['x'];y=z['y'];val=z['validation']
    reports={};predictions={}
    for name in ('initial','trained'):
        jax.config.update('jax_enable_x64',True)
        with np.load(dest/(name+'.npz'),allow_pickle=False) as z:
            params=tuple((z['w'+str(i)].copy(),z['b'+str(i)].copy()) for i in range(3))
        expected=read(dest/'report.json')['reports'][name]
        assert weights_hash(params)==expected['weights_sha256']
        net=Network(dest/(name+'.npz'),expected['export_sha256'])
        with jax.default_device(jax.devices('cpu')[0]):
            reference=np.asarray(forward(tuple((a.astype(np.float64),b.astype(np.float64)) for a,b in params),
                                         jax.numpy.asarray((x.astype(np.float64)-net.mean)/net.scale)))
        tick=time.perf_counter();logits=net.logits(x,manifest['features']);elapsed=time.perf_counter()-tick
        assert np.max(np.abs(logits-reference))<1e-9
        assert np.array_equal(logits.argmax(1),reference.argmax(1))
        predictions[name]=logits.argmax(1)
        reports[name]=dict(weight_identity_matches=True,export_identity_matches=True,
            checkpoint_sha256=sha(dest/(name+'.npz')),export_sha256=sha(dest/(name+'.npz')),
            samples=len(y),max_abs_error=float(np.max(np.abs(logits-reference))),batch_inference_seconds=elapsed)
    different=int(np.count_nonzero(predictions['initial']!=predictions['trained']))
    assert different>0
    failures=[]
    def fails(name, fn):
        try:fn()
        except (FileNotFoundError,ValueError):failures.append(name)
        else:raise AssertionError('Expected fail-closed behavior: '+name)
    path=dest/'trained.npz'
    fails('missing_weights',lambda:Network(scratch/'missing.npz','0'*64))
    fails('identity_mismatch',lambda:Network(path,'0'*64))
    with np.load(path,allow_pickle=False) as z:payload={k:z[k].copy() for k in z.files}
    bad={k:v.copy() for k,v in payload.items()};m=json.loads(str(bad['metadata']));m['schema_version']='incompatible'
    bad['metadata']=np.asarray(json.dumps(m));p=scratch/'bad-schema.npz';np.savez_compressed(p,**bad)
    fails('schema_mismatch',lambda:Network(p,sha(p)))
    bad={k:v.copy() for k,v in payload.items()};bad['w0'][0,0]=np.nan
    p=scratch/'nan.npz';np.savez_compressed(p,**bad)
    fails('nonfinite_weights',lambda:Network(p,sha(p)))
    net=Network(path,sha(path));bad_x=x[0].copy();bad_x[0]=np.nan
    fails('nonfinite_output',lambda:net.logits(bad_x,manifest['features']))
    fails('runtime_feature_mismatch',lambda:net.logits(x[0],list(reversed(manifest['features']))))
    save(out/'reload-verification.json',dict(passed=True,fresh_process=True,models=reports,
        changed_argmax_on_same_examples=different,total_examples=len(y),failure_contracts=failures,
        fallback_behavior='raise; no teacher substitution'))
    print(json.dumps({'reload_passed':True,'changed_argmax':different,'failure_contracts':len(failures)}))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--scratch',type=Path,required=True)
    a=p.parse_args();run(a.out.resolve(),a.scratch.resolve())
