"""Register and run one fixed-model diagnostic; no fitting or games."""
import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
import numpy as np
from learning.bc_crop.agent import ROOT, Network, sha
from learning.bc_crop.features import CROPS
from learning.bc_crop.experiment import read, save
from learning.bc_head_v1.experiment import verify as verify_head
from learning.bc_head_v1.metrics import masks
from .math import objective, direction_stats, first_adam, cosine, feature_hash, collision_summary, nearest
from .regression import run as regression


def prepare(out, public):
    previous = ROOT/'artifacts/bc-head-v1-20261023'
    old = verify_head(previous)
    assert read(previous/'acceptance.json')['stage_execution_passed']
    source = ROOT/old['original_experiment']; probe = ROOT/old['diagnostics_experiment']
    assert not out.exists()
    assert not subprocess.check_output(['git','-C',str(public),'status','--porcelain'])
    inputs = [source/'data/dataset.npz',source/'data/manifest.json',source/'data/samples.jsonl.gz',source/'model/trained.npz',
              previous/'model/report.json',previous/'acceptance.json',probe/'panel-ledger.json']
    for row in read(probe/'panel-ledger.json'):
        if row['policy']=='probe':
            trace = probe/'traces'/(row['game']+'.jsonl.gz')
            assert sha(trace)==row['trace_sha256']; inputs.append(trace)
    cfg = dict(experiment='bc-head-zero-update-diagnostics-v1',created_utc=datetime.now(timezone.utc).isoformat(),
        baseline_head=subprocess.check_output(['git','-C',str(public),'rev-parse','HEAD'],text=True).strip(),
        previous_experiment=previous.relative_to(ROOT).as_posix(),original_experiment=source.relative_to(ROOT).as_posix(),
        probe_experiment=probe.relative_to(ROOT).as_posix(),
        budgets=dict(training_runs=0,optimizer_updates=0,new_games=0,new_checkpoints=0),
        precision='fixed original deployment float64 normalization, hidden activations and logits; not a reconstruction of GPU FP32 training',
        objectives='unweighted and original float32 class-weighted training; unweighted developer validation and each seed; fixed160 errors; each train seed/class and raw-x deduplicated training, descriptive only',
        batches=dict(seed=2026102321,size=256,epochs=1,expected_batches=36,expected_last_real_samples=167,
                     state='each batch independently evaluated at same original theta and zero Adam moments; no update'),
        adam=dict(learning_rate=.0003,epsilon=1e-8,beta1=.9,beta2=.999,epsilon_root=0,nesterov=False,
                  formula='negative lr * gradient / (abs(gradient) + epsilon), ideal first-step bias correction'),
        sgd='negative lr * gradient using same nominal lr, no step executed',
        directions='full weighted/unweighted train SGD and ideal first Adam; all36 weighted first-epoch batches both directions',
        directional_sign_tolerance=1e-12,
        neighbors=dict(pool='unique training raw-float32 x, first occurrence; no val/probe fitting',
                       query='23 unique fixed160 error x;12 unique corrected probe x',
                       metric='Euclidean raw64D deployment hidden activations; report original-model-normalized289D input distance for selected neighbor',
                       ties='exact numeric hidden distance then lexicographic raw-float32 feature SHA256',
                       comparison='nearest different teacher label; nearest same teacher label excluding identical raw x; no threshold'),
        hidden_collisions='exact float64 hidden numerical vectors, signed zero canonicalized; train and train+developer-validation separately',
        acceptance='synthetic math regressions; full11483 deployment logits parity; fixed160 and23 unique errors;12 query x and52 calls;36 batches; finite derivatives/curvature; unchanged source/weights/history; independent recomputation',
        interpretation='local derivatives/curvature are not actual loss changes or causal attribution of previous failure; no collisions does not prove separability; no independent generalization claim',
        stop='one fixed diagnostic, no actual parameter perturbation, line search, fitting, model export or games; future training requires a separately frozen bounded plan',
        input_sha256={p.relative_to(ROOT).as_posix():sha(p) for p in inputs})
    protected = read(previous/'protection-before.json')['runtime'].copy()
    # All previous phase artifacts and sources are retained, not only model tensors.
    for base in (previous,ROOT/'learning/bc_head_v1'):
        for p in base.rglob('*'):
            if p.is_file() and '__pycache__' not in p.parts:
                protected[p.relative_to(ROOT.parent).as_posix()] = sha(p)
    public_names = subprocess.check_output(['git','-C',str(public),'ls-files','-z']).decode().split('\0')
    out.mkdir(parents=True,exist_ok=False)
    save(out/'protection-before.json',dict(runtime=protected,public={n:sha(public/n) for n in public_names if n}))
    save(out/'registration.json',cfg)
    save(out/'freeze.json',dict(registration_sha256=sha(out/'registration.json'),
        code={p.relative_to(ROOT).as_posix():sha(p) for p in Path(__file__).parent.glob('*.py')},
        results_absent=True,local_freeze_not_external_timestamp=True))
    print(json.dumps({'prepared':True,'budgets':cfg['budgets']}))


def verify(out):
    frozen=read(out/'freeze.json'); cfg=read(out/'registration.json')
    assert sha(out/'registration.json')==frozen['registration_sha256']
    for name,digest in frozen['code'].items(): assert sha(ROOT/name)==digest,name
    verify_head(ROOT/cfg['previous_experiment'])
    for name,digest in cfg['input_sha256'].items(): assert sha(ROOT/name)==digest,name
    for name,digest in read(out/'protection-before.json')['runtime'].items(): assert sha(ROOT.parent/name)==digest,name
    return cfg


def load(cfg):
    source=ROOT/cfg['original_experiment']; manifest=read(source/'data/manifest.json')
    with np.load(source/'data/dataset.npz',allow_pickle=False) as z:
        x,y,val,seeds=[z[k].copy() for k in ('x','y','validation','seed')]
    with gzip.open(source/'data/samples.jsonl.gz','rt',encoding='utf-8') as f: samples=[json.loads(line) for line in f]
    assert len(samples)==len(y)==11483 and (~val).sum()==9127 and val.sum()==2356
    assert all(s['seed']==int(seed) and s['label']==CROPS[int(label)] for s,seed,label in zip(samples,seeds,y))
    path=source/'model/trained.npz'; net=Network(path,sha(path))
    normalized=(x.astype(np.float64)-net.mean)/net.scale
    h=normalized.copy()
    for w,b in net.weights[:2]: h=np.maximum(h@w+b,0)
    H=np.c_[h,np.ones(len(h))]; theta=np.vstack(net.weights[-1]); theta.flags.writeable=False
    logits=H@theta
    # BLAS may accumulate the augmented constant differently from separate bias addition.
    deploy=net.logits(x,manifest['features'])
    assert np.allclose(logits,deploy,atol=1e-12,rtol=1e-12) and np.array_equal(logits.argmax(1),deploy.argmax(1))
    groups=masks(x,y,val,deploy.argmax(1),[s['outcome'] for s in samples])
    assert groups['selected_strawberry_errors'].sum()==160
    return x,y,val,seeds,net,normalized,h,H,theta,deploy,groups


def analyze(out):
    cfg=verify(out); assert not (out/'results.json').exists()
    assert (out/'pre-analysis-commit.json').exists()
    checks=regression()
    x,y,val,seeds,net,normalized,h,H,theta,logits,groups=load(cfg)
    theta_before=theta.tobytes(); train=~val
    signatures=np.asarray([feature_hash(row) for row in x])
    counts=np.bincount(y[train],minlength=5); weights=np.zeros(5,np.float32); present=counts>0
    weights[present]=1/np.sqrt(counts[present]); weights[present]/=weights[present].mean()
    _,unique=np.unique(signatures[train],return_index=True)
    train_ids=np.flatnonzero(train); unique_ids=train_ids[np.sort(unique)]
    dedup=np.zeros(len(y),bool);dedup[unique_ids]=True
    specs={'train_weighted':(train,weights[y]),'train_unweighted':(train,np.ones(len(y))),
           'train_unique_unweighted':(dedup,np.ones(len(y))),
           'validation':(val,np.ones(len(y))),
           'selected_errors':(groups['selected_strawberry_errors'],np.ones(len(y)))}
    for split,mask in (('train',train),('validation',val)):
        for seed in sorted(set(seeds[mask])):
            specs[f'{split}_seed_{seed}']=(mask&(seeds==seed),np.ones(len(y)))
    for label in np.flatnonzero(present): specs['train_class_'+CROPS[label]]=(train&(y==label),np.ones(len(y)))
    values={};summary={}
    for name,(mask,a) in specs.items():
        loss,g,p=objective(H[mask],y[mask],theta,a[mask]); values[name]=(g,p)
        summary[name]=dict(samples=int(mask.sum()),loss=loss,gradient_norm=float(np.linalg.norm(g)))
    targets=['train_weighted','train_unweighted','validation','selected_errors']+[n for n in specs if n.startswith('validation_seed_')]
    lr=cfg['adam']['learning_rate'];eps=cfg['adam']['epsilon']
    def describe(g):
        result={}
        for kind,d in [('sgd',-lr*g),('zero_state_adam',first_adam(g,lr,eps))]:
            result[kind]=dict(norm=float(np.linalg.norm(d)),targets={})
            for name in targets:
                mask,a=specs[name]; grad,p=values[name]
                result[kind]['targets'][name]=direction_stats(H[mask],a[mask],p,grad,d)
        return result
    full={name:describe(values[name][0]) for name in ('train_weighted','train_unweighted')}
    alignment={name:{target:cosine(values[name][0],values[target][0]) for target in targets} for name in specs}
    permutation=np.random.default_rng(cfg['batches']['seed']).permutation(len(train_ids))
    batches=[]
    for offset in range(0,len(permutation),cfg['batches']['size']):
        ids=train_ids[permutation[offset:offset+cfg['batches']['size']]]
        loss,g,_=objective(H[ids],y[ids],theta,weights[y[ids]])
        batches.append(dict(batch=len(batches),samples=len(ids),row_indices_sha256=hashlib.sha256(ids.astype('<i8').tobytes()).hexdigest(),
                            weighted_loss=loss,directions=describe(g)))
    assert len(batches)==36 and batches[-1]['samples']==167
    tolerance=cfg['directional_sign_tolerance'];signs={}
    for kind in ('sgd','zero_state_adam'):
        signs[kind]={}
        for target in targets:
            v=np.array([b['directions'][kind]['targets'][target]['derivative'] for b in batches])
            signs[kind][target]=dict(descent=int((v < -tolerance).sum()),ascent=int((v > tolerance).sum()),neutral=int((abs(v)<=tolerance).sum()))
    selected=groups['selected_strawberry_errors']; query={}
    for idx in np.flatnonzero(selected):
        key=('selected_errors',signatures[idx])
        if key not in query: query[key]=dict(x=x[idx],label=int(y[idx]),calls=0)
        query[key]['calls']+=1
    probe=ROOT/cfg['probe_experiment'];probe_games=0
    for row in read(probe/'panel-ledger.json'):
        if row['policy']!='probe':continue
        probe_games+=1
        with gzip.open(probe/'traces'/(row['game']+'.jsonl.gz'),'rt',encoding='utf-8') as f:
            for line in f:
                for record in json.loads(line)['records']:
                    if not record.get('correction_applied'):continue
                    qx=np.asarray(record['x'],np.float32);key=('probe',feature_hash(qx));label=CROPS.index(record['teacher'])
                    if key not in query:query[key]=dict(x=qx,label=label,calls=0)
                    assert query[key]['label']==label;query[key]['calls']+=1
    assert probe_games==12
    assert sum(k[0]=='selected_errors' for k in query)==23 and sum(k[0]=='probe' for k in query)==12
    assert sum(v['calls'] for k,v in query.items() if k[0]=='probe')==52
    neighbors=[];pool_hash=signatures[unique_ids];pool_y=y[unique_ids]
    for (group,signature),q in sorted(query.items()):
        nx=(q['x'].astype(float)-net.mean)/net.scale; hidden=nx.copy()
        for w,b in net.weights[:2]:hidden=np.maximum(hidden@w+b,0)
        score=hidden @ net.weights[2][0]+net.weights[2][1]
        result=dict(group=group,feature_sha256=signature,calls=q['calls'],teacher=CROPS[q['label']],prediction=CROPS[int(score.argmax())],
                    strawberry_minus_melon_margin=float(score[3]-score[4]),exact_x_in_train=bool((pool_hash==signature).any()))
        for name,mask in [('different_label',pool_y!=q['label']),('same_label_other_x',(pool_y==q['label'])&(pool_hash!=signature))]:
            found=nearest(hidden,h[unique_ids],pool_hash,mask)
            if found is None:result[name]=None;continue
            index,distance=found;original=unique_ids[index]
            result[name]=dict(feature_sha256=str(pool_hash[index]),label=CROPS[int(pool_y[index])],hidden_distance=distance,
                              normalized_input_distance=float(np.linalg.norm(normalized[original]-nx)))
        neighbors.append(result)
    assert theta.tobytes()==theta_before and not theta.flags.writeable
    result=dict(experiment=cfg['experiment'],budget_used=cfg['budgets'],precision=cfg['precision'],development_only=True,
        regression=checks,model_sha256=sha(ROOT/cfg['original_experiment']/'model/trained.npz'),
        registration_sha256=sha(out/'registration.json'),freeze_sha256=sha(out/'freeze.json'),
        classes=list(CROPS),class_weights=weights.tolist(),objectives=summary,gradient_cosines=alignment,
        full_train_directions=full,independent_zero_state_batches=batches,batch_direction_sign_counts=signs,
        collisions=dict(train=collision_summary(h[train],y[train]),train_and_developer_validation=collision_summary(h,y)),
        unique_training_x=len(unique_ids),neighbors=neighbors,parameters_unchanged=True,
        no_training_no_games_no_checkpoint=True,limitations=cfg['interpretation'])
    verify(out);save(out/'results.json',result)
    print(json.dumps({'passed':True,'budget_used':cfg['budgets'],'collisions':result['collisions'],'batch_signs':signs}))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['prepare','analyze']);p.add_argument('--out',type=Path,required=True);p.add_argument('--public',type=Path)
    a=p.parse_args()
    if a.mode=='prepare':prepare(a.out.resolve(),a.public.resolve())
    else:analyze(a.out.resolve())
