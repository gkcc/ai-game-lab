"""Recompute saved epoch evidence; never fit, change selection, or run games."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
from pathlib import Path
from collections import Counter
import hashlib
import json
import numpy as np

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[1]
OPS=('DIG','FERTILIZE','HARVEST','PLANT','WATER')
FIELDS=('shared','candidates','offsets','labels','selected_ops','candidate_ops','candidate_identity','seeds')
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,v):
    assert not p.exists()
    p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

def independent_moments(a):
    # Different reduction path from training: parallel variance combination.
    mean=np.zeros(a.shape[1],np.float64);m2=mean.copy();count=0
    for start in range(0,len(a),4096):
        x=np.array(a[start:start+4096],np.float64);n=len(x);mu=x.mean(axis=0)
        delta=mu-mean;total=count+n
        m2+=np.square(x-mu).sum(axis=0)+np.square(delta)*count*n/total
        mean+=delta*n/total;count=total
    return mean,np.sqrt(m2/count)

def independent_logits(z,a):
    n=len(a['candidates']);result=np.empty(n,np.float64)
    for lo in range(0,n,8192):
        hi=min(n,lo+8192);owner=np.searchsorted(a['offsets'][1:],np.arange(lo,hi),side='right')
        x=np.hstack((a['shared'][owner],a['candidates'][lo:hi])).astype(np.float64)
        x=(x-z['mean'])/z['scale']
        for i in range(3):
            x=np.matmul(x,z['w'+str(i)].astype(np.float64))+z['b'+str(i)].astype(np.float64)
            if i<2:x=np.maximum(x,0.)
        result[lo:hi]=x[:,0]
    assert np.isfinite(result).all()
    return result

def independent_metrics(logits,a,weights):
    offsets=a['offsets'];sizes=np.diff(offsets);n=len(sizes);ids=np.arange(n)
    label=a['labels'];actual=a['selected_ops'];positions=offsets[:-1]+label
    owner=np.repeat(ids,sizes)
    maxima=np.maximum.reduceat(logits,offsets[:-1])
    logsum=np.log(np.add.reduceat(np.exp(logits-maxima[owner]),offsets[:-1]))
    ce=logsum-(logits[positions]-maxima)
    pred=np.asarray([int(np.argmax(logits[lo:hi])) for lo,hi in zip(offsets[:-1],offsets[1:])])
    predop=a['candidate_ops'][offsets[:-1]+pred]
    multi=sizes>1
    cross=np.asarray([len(np.unique(a['candidate_ops'][lo:hi]))>1 for lo,hi in zip(offsets[:-1],offsets[1:])])
    w=weights[actual].astype(np.float64)
    def detail(p,mask):
        pop=a['candidate_ops'][offsets[:-1]+p]
        identity=a['candidate_identity'][offsets[:-1]+p];expected=a['candidate_identity'][positions]
        def sub(s):
            count=int(s.sum())
            if not count:return {'groups':0}
            return dict(groups=count,exact_top1=float(np.count_nonzero(p[s]==label[s])/count),
                unit_exact=float(np.count_nonzero(identity[s,0]==expected[s,0])/count),
                tile_exact=float(np.count_nonzero(np.all(identity[s,1:]==expected[s,1:],axis=1))/count),
                operation_exact=float(np.count_nonzero(pop[s]==actual[s])/count))
        confusion=np.zeros((5,5),int)
        for target in range(5):
            for output in range(5):confusion[target,output]=np.count_nonzero(mask&(actual==target)&(pop==output))
        return dict(sub(mask),operation_confusion_rows_actual_columns_predicted=confusion.tolist(),
            per_operation={name:sub(mask&(actual==i)) for i,name in enumerate(OPS)})
    allmask=np.ones(n,bool);recall={}
    for i,name in enumerate(OPS):
        mask=cross&(actual==i);assert mask.any()
        recall[name]=dict(groups=int(mask.sum()),recall=float(np.count_nonzero(predop[mask]==i)/mask.sum()))
    details=detail(pred,allmask);details['multicandidate']=detail(pred,multi);details['cross_operation']=detail(pred,cross)
    seeds={}
    for seed in sorted(set(a['seeds'].tolist())):
        mask=a['seeds']==seed
        seeds[str(seed)]=dict(detail(pred,mask),unweighted_ce=float(ce[mask].mean()),weighted_ce=float(np.sum(w[mask]*ce[mask])/w[mask].sum()))
    byop={}
    for i,name in enumerate(OPS):
        mask=actual==i
        byop[name]=dict(unweighted_ce=float(ce[mask].mean()),weighted_ce=float(np.sum(w[mask]*ce[mask])/w[mask].sum()))
    water=[]
    for lo,hi in zip(offsets[:-1],offsets[1:]):
        positions_water=np.flatnonzero(a['candidate_ops'][lo:hi]==4)
        water.append(int(positions_water[0]) if len(positions_water) else 0)
    baseline={}
    for name,p in [('canonical_first',np.zeros(n,int)),('water_first_else_canonical_first',np.asarray(water))]:
        baseline[name]=dict(all=detail(p,allmask),multicandidate=detail(p,multi),cross_operation=detail(p,cross),not_used_for_checkpoint_selection=True)
    return dict(groups=n,multicandidate_groups=int(multi.sum()),cross_operation_groups=int(cross.sum()),
        weighted_ce=float(np.sum(w*ce)/w.sum()),unweighted_ce=float(ce.mean()),
        multicandidate_exact_top1=float(np.count_nonzero(pred[multi]==label[multi])/multi.sum()),
        cross_operation_macro_recall=float(sum(v['recall'] for v in recall.values())/5),
        cross_operation_recalls=recall,decision_details=details,by_seed=seeds,by_operation_ce=byop,
        descriptive_baselines=baseline,prediction_sha256=hashlib.sha256(pred.astype('<i8').tobytes()).hexdigest())

def compare(a,b,path='',errors=None):
    if errors is None:errors=[]
    if isinstance(a,dict):
        assert set(a)==set(b),(path,'keys')
        for k in a:compare(a[k],b[k],path+'/'+k,errors)
    elif isinstance(a,list):
        assert len(a)==len(b),path
        for i,(x,y) in enumerate(zip(a,b)):compare(x,y,path+'/'+str(i),errors)
    elif isinstance(a,float) and path.endswith(('weighted_ce','unweighted_ce')):
        errors.append(abs(a-b));assert np.isclose(a,b,atol=1e-12,rtol=1e-12),(path,a,b)
    elif isinstance(a,float) and path.endswith('/cross_operation_macro_recall'):
        assert abs(a-b)<=1e-15,(path,a,b) # Python sum vs NumPy mean; components remain exact.
    else:assert a==b,(path,a,b)
    return errors

def gate(m,initial):
    return dict(multicandidate_top1=m['multicandidate_exact_top1']>=.6,
        improves_initial_ten_points=m['multicandidate_exact_top1']-initial['multicandidate_exact_top1']>=.10,
        cross_operation_macro_recall=m['cross_operation_macro_recall']>=.7,
        all_cross_operation_recalls=all(r['recall']>=.5 for r in m['cross_operation_recalls'].values()),
        weighted_ce_improves=m['weighted_ce']<initial['weighted_ce'])

def main():
    cfg=read(OUT/'registration.json');cache=OUT/'cache';model=OUT/'model'
    report=read(model/'report.json');history=read(model/'history.json');manifest=read(cache/'manifest.json')
    assert report['completed'] and report['epochs']==8 and len(history)==9
    assert not (OUT/'training-failed.json').exists()
    assert sha(model/'history.json')==report['history_sha256']
    assert sha(cache/'manifest.json')==report['cache_manifest_sha256']
    assert report['optimizer_updates']==report['expected_updates']==3392
    assert report['synchronized_training_seconds']<=1800 and report['total_fit_seconds']<2700
    assert report['gpu_training'] and report['platform']=='gpu' and report['nonzero_gradient_updates']>0
    local_device=read(model/'device-local.json');assert local_device['platform']=='gpu'
    a={s:{k:np.load(cache/s/(k+'.npy'),mmap_mode='r',allow_pickle=False) for k in FIELDS} for s in ('train','validation')}
    for s,files in manifest['arrays'].items():
        for k,info in files.items():assert sha(cache/info['path'])==info['sha256']
    counts=np.bincount(a['train']['selected_ops'],minlength=5)
    w=np.clip(np.sqrt(len(a['train']['labels'])/(5*counts)),.5,5.);w=(w/(np.dot(counts,w)/counts.sum())).astype(np.float32)
    assert report['class_counts']==counts.tolist() and report['class_weights']==w.tolist()
    schema=read(cache/'schema.json');names=schema['shared_feature_names']+schema['feature_names']
    sm,ss=independent_moments(a['train']['shared']);cm,cs=independent_moments(a['train']['candidates'])
    mean=np.r_[sm,cm];scale=np.maximum(1.,np.r_[ss,cs])
    binary=[name=='planning.final_day' or name.endswith(('.present','.watered_today','.fed_today','.cared_today','.fertilizer_available','.pending_care_bonus')) or name.startswith('task.op.') or any(p in name for p in ('.kind.','.crop.','.animal.','.below_func.','.above_func.')) for name in names]
    mean[binary]=0.;scale[binary]=1.
    rng=np.random.default_rng(cfg['model_seed']);initial={}
    for i,(inputs,outputs) in enumerate(((373,128),(128,64),(64,1))):
        initial['w'+str(i)]=(rng.standard_normal((inputs,outputs))*np.sqrt(2/inputs)).astype(np.float32)
        initial['b'+str(i)]=np.zeros(outputs,np.float32)
    errors=[];summary=[];best=None;bestce=float('inf');first_metrics=None;previous=None;normalizer_errors=[];source_hashes={}
    for epoch in range(9):
        path=model/'checkpoints'/f'epoch-{epoch:02d}.npz';h=history[epoch]
        assert h['epoch']==epoch and h['updates']==424*epoch and sha(path)==h['checkpoint_sha256']
        source_hashes[path.relative_to(OUT).as_posix()]=sha(path)
        with np.load(path,allow_pickle=False) as archive:z={k:archive[k].copy() for k in archive.files}
        metadata=json.loads(str(z['metadata']));assert metadata['epoch']==epoch and metadata['model_seed']==cfg['model_seed']
        assert metadata['schema_sha256']==schema['schema_sha256'] and metadata['cache_manifest_sha256']==sha(cache/'manifest.json')
        np.testing.assert_allclose(z['mean'],mean,atol=1e-9,rtol=1e-11)
        np.testing.assert_allclose(z['scale'],scale,atol=1e-8,rtol=1e-9)
        normalizer_errors.append((float(np.max(abs(z['mean']-mean))),float(np.max(abs(z['scale']-scale)))))
        for k in initial:assert np.isfinite(z[k]).all() and z[k].dtype==np.float32
        if epoch==0:
            assert all(np.array_equal(z[k],v) for k,v in initial.items())
            assert sha(path)==sha(model/'initial.npz')
        else:
            assert h['groups_visited']==108310
            assert any(not np.array_equal(z[k],previous[k]) for k in initial)
        logits=independent_logits(z,a['validation']);m=independent_metrics(logits,a['validation'],w)
        errors.extend(compare(m,h['metrics']))
        if epoch==0:first_metrics=m
        else:
            gates=gate(m,first_metrics);assert gates==h['gates']
            if all(gates.values()) and m['weighted_ce']<bestce:best,bestce=epoch,m['weighted_ce']
            assert h['selected_at_epoch']==(best==epoch)
        summary.append(dict(epoch=epoch,weighted_ce=m['weighted_ce'],multicandidate_exact_top1=m['multicandidate_exact_top1'],cross_operation_macro_recall=m['cross_operation_macro_recall'],passes=False if epoch==0 else all(gates.values()),checkpoint_sha256=sha(path)))
        previous=z
        print('independent epoch checked',epoch,flush=True)
    assert best==report['best_epoch'] and (best is not None)==report['offline_metrics_passed']
    changed=sum(not np.array_equal(previous[k],initial[k]) for k in initial)
    assert changed==report['changed_final_parameter_tensors']
    export=report['export_validation'];offline=(best is not None and export['passed'])
    assert offline==report['offline_gate_passed']
    if best is None:assert not (model/'trained.npz').exists()
    else:
        selected=read(model/'selected-export.json');path=model/'checkpoints'/f'epoch-{best:02d}.npz'
        assert selected['epoch']==best and selected['checkpoint_sha256']==sha(path)
        assert sha(model/'selected-gpu-logits.npy')==selected['gpu_logits_sha256']
        gpu=np.load(model/'selected-gpu-logits.npy',allow_pickle=False)
        with np.load(path,allow_pickle=False) as archive:z={k:archive[k].copy() for k in archive.files}
        cpu=independent_logits(z,a['validation']);offset=a['validation']['offsets']
        disagree=sum(np.argmax(cpu[lo:hi])!=np.argmax(gpu[lo:hi]) for lo,hi in zip(offset[:-1],offset[1:]))
        close=bool(np.allclose(cpu,gpu,atol=1e-4,rtol=1e-5))
        assert int(disagree)==export['candidate_id_disagreements'] and (close and disagree==0)==export['passed']
        if offline:
            assert sha(model/'trained.npz')==sha(path)
            reload=read(OUT/'reload.json')
            assert reload['passed'] and reload['standalone_python_module_process'] and reload['epoch']==best
            for key,p in [('checkpoint_sha256',path),('training_report_sha256',model/'report.json'),('selected_export_sha256',model/'selected-export.json'),('cache_manifest_sha256',cache/'manifest.json'),('gpu_logits_sha256',model/'selected-gpu-logits.npy'),('cpu_logits_sha256',model/'reloaded-cpu-logits.npy')]:
                assert reload[key]==sha(p)
            fresh=np.load(model/'reloaded-cpu-logits.npy',allow_pickle=False)
            assert fresh.shape==cpu.shape and np.allclose(fresh,cpu,atol=1e-12,rtol=1e-12)
            assert np.allclose(fresh,gpu,atol=1e-4,rtol=1e-5)
            ids=hashlib.sha256()
            for lo,hi in zip(offset[:-1],offset[1:]):
                j=int(np.argmax(fresh[lo:hi]));assert j==np.argmax(cpu[lo:hi])==np.argmax(gpu[lo:hi])
                k=lo+j;unit,x,y=map(int,a['validation']['candidate_identity'][k]);op=OPS[int(a['validation']['candidate_ops'][k])]
                ids.update(f'u{unit}:x{x}:y{y}:{op}\n'.encode())
            assert ids.hexdigest()==reload['cpu_candidate_id_digest']==reload['gpu_candidate_id_digest']
    # Historical evidence remains immutable; publication append occurs later.
    protected=read(OUT/'protection-before.json');public=ROOT.parent/'publication/ai-game-lab'
    for rel,h in protected['runtime'].items():assert sha(ROOT/rel)==h
    for rel,h in protected['public'].items():assert sha(public/rel)==h
    evidence={n:sha(OUT/n) for n in ('model/report.json','model/history.json','cache/manifest.json','registration.json','freeze.json','model/initial.npz')}
    evidence.update(source_hashes)
    if best is not None:
        for n in ('model/selected-export.json','model/selected-gpu-logits.npy'):evidence[n]=sha(OUT/n)
    if offline:
        for n in ('model/trained.npz','reload.json','model/reloaded-cpu-logits.npy'):evidence[n]=sha(OUT/n)
    result=dict(passed=True,offline_entry_passed=bool(offline),script_sha256=sha(Path(__file__)),evidence_sha256=evidence,
        all_epochs_recomputed=9,validation_groups_per_epoch=27096,epochs=summary,best_epoch=best,
        maximum_ce_reduction_difference=max(errors),normalizer_max_mean_difference=max(x[0] for x in normalizer_errors),normalizer_max_scale_difference=max(x[1] for x in normalizer_errors),
        gpu_evidence={'recorded_platform':'gpu','private_device_record_sha256':sha(model/'device-local.json'),'changed_final_parameter_tensors':changed,'nonzero_gradient_updates':report['nonzero_gradient_updates'],'optimizer_updates':3392},
        protected_runtime_files=len(protected['runtime']),protected_public_files=len(protected['public']),
        reviewer_training_runs=0,reviewer_optimizer_updates=0,new_games=0,
        fresh_cpu_reload_verified=bool(offline),game_control_and_privacy_check_still_required=True,
        limitations=['Independent CPU recomputation checks saved weights, metrics, normalization, selection and receipts, not a second optimizer trajectory replay.',
                    'GPU execution is supported by frozen training code, private runtime record, completed updates and saved changed checkpoints; no hardware identity is published.',
                    'Development validation and strict engineering gates do not demonstrate economic or independent-test improvement.'])
    save(OUT/'independent-training-review.json',result)
    evidence=dict(evidence,**{'independent-training-review.json':sha(OUT/'independent-training-review.json'),'independent-training-review.py':sha(Path(__file__))})
    save(OUT/'post-training-review.json',dict(passed=True,offline_entry_passed=bool(offline),evidence_sha256=evidence,fresh_standalone_reload_verified=bool(offline),games_allowed_by_this_receipt_alone=False))
    print(json.dumps({k:v for k,v in result.items() if k not in ('evidence_sha256','epochs')}))

if __name__=='__main__':main()
