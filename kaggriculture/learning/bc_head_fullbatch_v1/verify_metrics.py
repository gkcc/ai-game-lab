"""Protect float32-weight normalization and5-class margin interpretation."""
import numpy as np
from .metrics import summarize
from learning.bc_head_v1.metrics import masks


def run():
    x=np.arange(8,dtype=np.float32).reshape(8,1)
    y=np.array([3,3,0,4,3,4,0,1]);val=np.array([False]*4+[True]*4)
    logits=np.zeros((8,5));logits[:,0]=3;logits[:,3]=1;logits[:,4]=.5
    weights=np.array([.2,.7,.9,1.3,.4],np.float32)
    groups=masks(x,y,val,logits.argmax(1),['plant_request']*8)
    r=summarize(logits,y,val,np.array([1]*4+[2]*4),groups,weights)
    rw=summarize(logits,y,val,np.array([1]*4+[2]*4),groups,weights.astype(np.float64))
    assert r==rw
    shifted=logits[:4]-logits[:4].max(1,keepdims=True)
    loss=np.log(np.exp(shifted).sum(1))-shifted[np.arange(4),y[:4]]
    expected=sum(float(w)*float(v) for w,v in zip(weights[y[:4]],loss))/sum(float(w) for w in weights[y[:4]])
    assert abs(r['train_weighted_cross_entropy']-expected)<1e-15
    assert r['strawberry_melon_margins']['selected_strawberry_errors']['positive']==2
    assert r['groups']['selected_strawberry_errors']['errors']==2
    assert r['groups']['seed_1']['samples']==4 and r['groups']['seed_2']['samples']==4
    print({'passed':True,'checks':['float32 versus float64 class weights','independent weighted normalization','pairwise margin is not5-class correctness','seed partitions'],'training_runs':0})


if __name__=='__main__':run()
