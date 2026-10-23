"""Focused regressions for CE, fixed error membership and development gates."""
import json
import numpy as np
from .metrics import ce,masks,evaluate,gate,compare_metrics


def run():
    assert abs(ce(np.zeros((3,5)),np.array([0,2,4]))-np.log(5))<1e-12
    assert ce(np.array([[10000.,0.,0.,0.,0.]]),np.array([0]))==0
    x=np.asarray([[1],[1],[2],[3]],np.float32);y=np.array([3,3,3,3]);val=np.array([False,True,False,True])
    groups=masks(x,y,val,np.array([4,4,4,4]),['plant_request','plant_request','move_with_seed','request_seed'])
    assert np.flatnonzero(groups['selected_strawberry_errors']).tolist()==[0]
    assert np.flatnonzero(groups['validation_unseen']).tolist()==[3]
    base_logits=np.zeros((4,5));base_logits[:,4]=1
    new_logits=base_logits.copy();new_logits[:,3]=2
    baseline=evaluate(base_logits,y,groups);candidate=evaluate(new_logits,y,groups)
    assert all(gate(baseline,candidate,1).values()) and not gate(baseline,candidate,0)['nonzero_epoch']
    assert baseline['selected_strawberry_errors']['samples']==candidate['selected_strawberry_errors']['samples']==1
    bad={k:dict(v) for k,v in candidate.items()};bad['validation']['accuracy']=baseline['validation']['accuracy']-.011
    assert not gate(baseline,bad,1)['validation_accuracy_guard']
    rounded={k:dict(v) for k,v in baseline.items()};rounded['validation']['cross_entropy']+=1e-14
    assert 0<compare_metrics(rounded,baseline)<1e-12
    rounded['validation']['errors']+=1
    try:compare_metrics(rounded,baseline)
    except AssertionError:pass
    else:raise AssertionError('Discrete errors must not receive numeric tolerance')
    print(json.dumps(dict(passed=True,checks=['stable CE','fixed original error subset','train-only exact overlap','same denominator','epoch0 rejection','validation accuracy guard','CE-only tolerance and exact discrete statistics'],training_runs=0,new_games=0)))


if __name__=='__main__':run()
