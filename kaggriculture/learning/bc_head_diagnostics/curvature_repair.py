"""Append corrected float64 curvature evidence; retain frozen initial analysis."""
import argparse
from copy import deepcopy
from pathlib import Path
import numpy as np
from learning.bc_crop.agent import ROOT,sha
from learning.bc_crop.experiment import read,save
from .experiment import verify,load
from .math import objective,first_adam,direction_stats as historical_direction_stats


def direction_stats(h,weight,p,grad,direction):
    # Both numerator and denominator must use the same float64 sample weights
    # as objective(), even when supplied the original float32 class weights.
    weight=np.asarray(weight,dtype=np.float64)
    if not np.isfinite(weight).all() or (weight<0).any() or weight.sum()<=0:
        raise ValueError('Invalid curvature weights')
    projected=h @ direction
    mean=(p*projected).sum(axis=1,keepdims=True)
    variance=(p*(projected-mean)**2).sum(axis=1)
    return dict(derivative=float(np.sum(grad*direction)),
                curvature=float(np.dot(weight,variance)/weight.sum()))


def regression():
    h=np.array([[1.,2.],[3.,-1.],[.2,.7]])
    y=np.array([0,1,0]);theta=np.array([[.3,-.5],[.7,.2]])
    d=np.array([[.1,.7],[-.4,.3]])
    weights=np.array([.2,.7,1.3],np.float32)
    _,g,p=objective(h,y,theta,weights)
    actual=direction_stats(h,weights,p,g,d)
    expected=direction_stats(h,weights.astype(np.float64),p,g,d)
    assert actual==expected
    old=historical_direction_stats(h,weights,p,g,d)
    assert old['curvature']!=actual['curvature'] and old['derivative']==actual['derivative']
    e=1e-5
    numeric=np.sum(d*(objective(h,y,theta+e*d,weights)[1]-objective(h,y,theta-e*d,weights)[1]))/(2*e)
    assert np.isclose(actual['curvature'],numeric,atol=1e-9,rtol=1e-7)
    padded=direction_stats(np.vstack([h,h[:1]]),np.r_[weights,np.float32(0)],np.vstack([p,p[:1]]),g,d)
    assert padded==actual
    return dict(passed=True,float32_input_equals_float64=True,old_failure_reproduced=True,finite_difference=True,zero_padding=True)


def prepare(out):
    verify(out);assert not (out/'repair-registration.json').exists()
    save(out/'repair-registration.json',dict(original_result_sha256=sha(out/'results.json'),
        original_freeze_sha256=sha(out/'freeze.json'),repair_source_sha256=sha(Path(__file__)),
        correction='cast sample weights to float64 before curvature numerator and denominator reduction',
        expected_changes='Only76 train_weighted target curvature values:4 full-direction and72 batch-direction evaluations',
        unchanged='all gradients, directional derivatives, signs, query rows, collision results and historical evidence',
        regression=regression(),training_runs=0,optimizer_updates=0,new_games=0,new_checkpoints=0,
        local_freeze_not_external_timestamp=True))


def apply(out):
    cfg=verify(out);registration=read(out/'repair-registration.json')
    assert sha(Path(__file__))==registration['repair_source_sha256']
    assert sha(out/'results.json')==registration['original_result_sha256']
    assert sha(out/'freeze.json')==registration['original_freeze_sha256']
    assert (out/'pre-repair-commit.json').exists() and not (out/'results-corrected.json').exists()
    regression()
    x,y,val,seeds,net,normalized,h,H,theta,logits,groups=load(cfg)
    old=read(out/'results.json');result=deepcopy(old)
    weights=np.asarray(old['class_weights'],np.float32);ones=np.ones(len(y))
    specs={'train_weighted':(~val,weights[y]),'train_unweighted':(~val,ones),
           'validation':(val,ones),'selected_errors':(groups['selected_strawberry_errors'],ones)}
    for seed in sorted(set(seeds[val])):specs[f'validation_seed_{seed}']=(val&(seeds==seed),ones)
    values={name:objective(H[mask],y[mask],theta,a[mask])[1:] for name,(mask,a) in specs.items()}
    lr=cfg['adam']['learning_rate'];epsilon=cfg['adam']['epsilon'];changed=[]
    def replace(records,g,label):
        for kind,d in [('sgd',-lr*g),('zero_state_adam',first_adam(g,lr,epsilon))]:
            for name,(mask,a) in specs.items():
                grad,p=values[name];actual=direction_stats(H[mask],a[mask],p,grad,d)
                prior=records[kind]['targets'][name]
                assert actual['derivative']==prior['derivative']
                if actual['curvature']!=prior['curvature']:
                    assert name=='train_weighted'
                    changed.append(dict(location=label+'/'+kind+'/'+name,old=prior['curvature'],corrected=actual['curvature']))
                records[kind]['targets'][name]=actual
    for name in ('train_weighted','train_unweighted'):replace(result['full_train_directions'][name],values[name][0],name)
    train_ids=np.flatnonzero(~val);permutation=np.random.default_rng(cfg['batches']['seed']).permutation(len(train_ids))
    for batch,offset in zip(result['independent_zero_state_batches'],range(0,len(train_ids),cfg['batches']['size'])):
        ids=train_ids[permutation[offset:offset+cfg['batches']['size']]]
        _,g,_=objective(H[ids],y[ids],theta,weights[y[ids]])
        replace(batch['directions'],g,'batch_'+str(batch['batch']))
    assert len(changed)==76
    # Revert only the allowed field locations in a copy to prove all else exact.
    reverted=deepcopy(result)
    for name in reverted['full_train_directions']:
        for kind in reverted['full_train_directions'][name]:
            for target in specs:
                reverted['full_train_directions'][name][kind]['targets'][target]['curvature']=old['full_train_directions'][name][kind]['targets'][target]['curvature']
    for b,original in zip(reverted['independent_zero_state_batches'],old['independent_zero_state_batches']):
        for kind in b['directions']:
            for target in specs:b['directions'][kind]['targets'][target]['curvature']=original['directions'][kind]['targets'][target]['curvature']
    assert reverted==old
    result['numerical_correction']=dict(registration_sha256=sha(out/'repair-registration.json'),**registration)
    verify(out);save(out/'results-corrected.json',result)
    save(out/'curvature-repair-certificate.json',dict(passed=True,original_result_sha256=sha(out/'results.json'),
        corrected_result_sha256=sha(out/'results-corrected.json'),repair_registration_sha256=sha(out/'repair-registration.json'),
        changed_curvatures=changed,other_fields_exactly_unchanged=True,regression=regression(),
        training_runs=0,optimizer_updates=0,new_games=0,new_checkpoints=0))
    print('Corrected76 curvature values; all other original fields exact;0 training/games')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['regression','prepare','apply']);p.add_argument('--out',type=Path)
    a=p.parse_args()
    if a.mode=='regression':print(regression())
    elif a.mode=='prepare':prepare(a.out.resolve())
    else:apply(a.out.resolve())
