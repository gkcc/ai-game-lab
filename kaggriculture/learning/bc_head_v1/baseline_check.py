"""Cross-platform regression on the original checkpoint; never trains."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
from learning.bc_crop.agent import ROOT,Network,sha
from learning.bc_crop.experiment import read,save,integrity
from .metrics import masks,evaluate,compare_metrics


def run(out,reference=None):
    source=ROOT/'artifacts/bc-crop-v1-20261023-run1';integrity(source);assert not out.exists()
    model=source/'model/trained.npz'
    with np.load(source/'data/dataset.npz',allow_pickle=False) as z:x,y,val=[z[k] for k in ('x','y','validation')]
    with gzip.open(source/'data/samples.jsonl.gz','rt',encoding='utf-8') as f:samples=[json.loads(l) for l in f]
    names=read(source/'data/manifest.json')['features'];logits=Network(model,sha(model)).logits(x,names)
    groups=masks(x,y,val,logits.argmax(1),[s['outcome'] for s in samples]);metrics=evaluate(logits,y,groups)
    result=dict(platform=sys.platform,numpy=np.__version__,checkpoint_sha256=sha(model),dataset_sha256=sha(source/'data/dataset.npz'),
                prediction_sha256=hashlib.sha256(logits.argmax(1).astype('<i8').tobytes()).hexdigest(),metrics=metrics,training_runs=0,new_games=0)
    if reference:
        old=read(reference)
        assert all(old[k]==result[k] for k in ('checkpoint_sha256','dataset_sha256','prediction_sha256'))
        result.update(cross_platform_comparison_passed=True,max_cross_entropy_difference=compare_metrics(metrics,old['metrics']),reference_sha256=sha(reference))
    save(out,result);print(json.dumps({k:v for k,v in result.items() if k!='metrics'}))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);p.add_argument('--reference',type=Path)
    a=p.parse_args();run(a.out.resolve(),a.reference.resolve() if a.reference else None)
