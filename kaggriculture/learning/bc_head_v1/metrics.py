"""Deployment-precision developer metrics, with no original test-game inputs."""
import hashlib
import numpy as np
from learning.bc_crop.features import CROPS


def ce(logits,y):
    shifted=logits-logits.max(1,keepdims=True)
    return float(np.mean(np.log(np.exp(shifted).sum(1))-shifted[np.arange(len(y)),y]))


def masks(x,y,validation,baseline,outcomes):
    signatures=[hashlib.sha256(np.asarray(v,dtype=np.float32).tobytes()).digest() for v in x]
    seen={h for h,v in zip(signatures,validation) if not v}
    strawberry=(~validation)&(y==CROPS.index('STRAWBERRY'))&(baseline!=y)&np.isin(outcomes,['request_seed','plant_request'])
    return dict(train=~validation,validation=validation,
                validation_unseen=validation&np.asarray([h not in seen for h in signatures]),
                selected_strawberry_errors=strawberry)


def evaluate(logits,y,groups):
    prediction=logits.argmax(1);result={}
    for name,mask in groups.items():
        cm=np.zeros((5,5),np.int64);np.add.at(cm,(y[mask],prediction[mask]),1)
        result[name]=dict(samples=int(mask.sum()),errors=int(np.count_nonzero(prediction[mask]!=y[mask])),
            accuracy=float(np.mean(prediction[mask]==y[mask])) if mask.any() else None,
            cross_entropy=ce(logits[mask],y[mask]) if mask.any() else None,confusion=cm.tolist())
    return result


def gate(baseline,candidate,epoch):
    return dict(nonzero_epoch=epoch>0,
        selected_train_errors_halved=candidate['selected_strawberry_errors']['errors']<=baseline['selected_strawberry_errors']['errors']/2,
        validation_accuracy_guard=candidate['validation']['accuracy']>=baseline['validation']['accuracy']-.01,
        unseen_validation_accuracy_guard=candidate['validation_unseen']['accuracy']>=baseline['validation_unseen']['accuracy']-.02)


def compare_metrics(actual,expected):
    """Discrete results exact; only cross-platform CE rounding is tolerated."""
    assert actual.keys()==expected.keys();error=0.
    for name,group in actual.items():
        other=expected[name]
        assert {k:v for k,v in group.items() if k!='cross_entropy'}=={k:v for k,v in other.items() if k!='cross_entropy'}
        a,b=group['cross_entropy'],other['cross_entropy']
        if a is None or b is None:assert a==b
        else:
            assert np.isclose(a,b,atol=1e-12,rtol=1e-12)
            error=max(error,abs(a-b))
    return error
