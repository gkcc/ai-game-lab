"""Per-step evidence in deployment precision; original gates remain unchanged."""
import hashlib
import numpy as np
from learning.bc_crop.features import CROPS
from learning.bc_head_v1.metrics import evaluate


def prediction_hash(logits):
    return hashlib.sha256(logits.argmax(1).astype('<i8').tobytes()).hexdigest()


def summarize(logits,y,validation,seeds,groups,weights):
    shifted=logits-logits.max(1,keepdims=True)
    losses=np.log(np.exp(shifted).sum(1))-shifted[np.arange(len(y)),y]
    train=~validation;weight=np.asarray(weights,dtype=np.float64)[y[train]]
    detailed=dict(groups)
    for s in sorted(set(seeds)):detailed['seed_'+str(s)]=seeds==s
    for split,mask in [('train',train),('validation',validation)]:
        for i,crop in enumerate(CROPS):detailed[split+'_class_'+crop]=mask&(y==i)
    metrics=evaluate(logits,y,detailed)
    margin=logits[:,3]-logits[:,4];margins={}
    for name in ('selected_strawberry_errors','train_class_STRAWBERRY','validation_class_STRAWBERRY'):
        mask=detailed[name]
        margins[name]=dict(samples=int(mask.sum()),minimum=float(margin[mask].min()),mean=float(margin[mask].mean()),
            maximum=float(margin[mask].max()),positive=int((margin[mask]>0).sum()),
            note='STRAWBERRY minus MELON only; positive does not imply STRAWBERRY wins all5 classes')
    return dict(groups=metrics,train_weighted_cross_entropy=float(np.dot(weight,losses[train])/weight.sum()),
                strawberry_melon_margins=margins,prediction_sha256=prediction_hash(logits))
