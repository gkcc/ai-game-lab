"""Standalone CPU process: reload the one selected epoch against saved GPU logits.

No head reselection, training, new games or tolerance adjustment occurs here.
Run as ``python -B -m learning.task_bc_v1.reload --out ...`` after the GPU fit.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

# Process-local BLAS cap only; do not change machine settings or training env.
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
import numpy as np

from learning.bc_crop.agent import sha
from learning.bc_crop.experiment import read, save
from learning.task_choice_v1.features import OPS, candidate_id
from .model import TaskNetwork

ATOL = 1e-4
RTOL = 1e-5


def load_array(cache, manifest, name):
    info = manifest['arrays']['validation'][name]
    path = cache/info['path']
    assert path.resolve() == (cache/'validation'/(name+'.npy')).resolve()
    assert sha(path) == info['sha256'], name
    value = np.load(path, mmap_mode='r', allow_pickle=False)
    assert list(value.shape) == info['shape'] and str(value.dtype) == info['dtype']
    return value


def comparison(cpu, gpu):
    """Predeclared tolerance and first-argmax agreement, including exact ties."""
    cpu, gpu = np.asarray(cpu, np.float64), np.asarray(gpu, np.float64)
    if cpu.ndim != 1 or cpu.shape != gpu.shape or not len(cpu):
        raise ValueError('Mismatched group logits')
    if not np.isfinite(cpu).all() or not np.isfinite(gpu).all():
        raise ValueError('Nonfinite export logits')
    close = np.isclose(cpu,gpu,atol=ATOL,rtol=RTOL)
    return dict(tolerance_failures=int((~close).sum()),
        max_absolute_difference=float(np.max(np.abs(cpu-gpu))),
        cpu_index=int(np.argmax(cpu)),gpu_index=int(np.argmax(gpu)))


def run(out, *, standalone=False):
    if not standalone or __name__ != '__main__':
        raise RuntimeError('Export acceptance requires a new standalone Python module process')
    from .experiment import ready, verify
    out=Path(out).resolve(); ready(out,phase='train'); tick=time.perf_counter()
    model=out/'model'; cache=out/'cache'
    report=read(model/'report.json')
    assert report['completed'] and report['epochs']==8 and report['offline_metrics_passed']
    epoch=report['best_epoch']
    assert type(epoch) is int and 1<=epoch<=8
    selected=read(model/'selected-export.json')
    assert selected['epoch']==epoch
    checkpoint=model/selected['checkpoint_file']
    assert checkpoint.resolve()==(model/'checkpoints'/f'epoch-{epoch:02d}.npz').resolve()
    assert sha(checkpoint)==selected['checkpoint_sha256']
    if (model/'trained.npz').exists():
        assert sha(model/'trained.npz')==selected['checkpoint_sha256']
    gpu_path=model/selected['gpu_logits_file']
    assert gpu_path.resolve()==(model/'selected-gpu-logits.npy').resolve()
    assert sha(gpu_path)==selected['gpu_logits_sha256']
    assert sha(cache/'manifest.json')==selected['cache_manifest_sha256']
    manifest=read(cache/'manifest.json')
    assert manifest['passed'] and sha(cache/'schema.json')==manifest['schema_sha256']
    schema=read(cache/'schema.json')
    network=TaskNetwork(checkpoint,selected['checkpoint_sha256'])
    assert network.metadata['schema_sha256']==schema['schema_sha256']
    assert network.metadata['shared_feature_names']==schema['shared_feature_names']
    assert network.metadata['feature_names']==schema['feature_names']
    # Identity arrays are used exclusively to compare the two selected IDs.
    # Seeds, labels and selected teacher operations are never loaded or scored.
    shared=load_array(cache,manifest,'shared'); candidates=load_array(cache,manifest,'candidates')
    offsets=load_array(cache,manifest,'offsets')
    identity=load_array(cache,manifest,'candidate_identity'); ops=load_array(cache,manifest,'candidate_ops')
    n,m=len(shared),len(candidates)
    assert shared.shape==(n,275) and candidates.shape==(m,98)
    assert offsets.shape==(n+1,) and offsets[0]==0 and offsets[-1]==m
    assert np.all(np.diff(offsets)>0) and np.max(np.diff(offsets))<=65536
    assert identity.shape==(m,3) and ops.shape==(m,)
    assert np.all((ops>=0)&(ops<len(OPS)))
    gpu=np.load(gpu_path,mmap_mode='r',allow_pickle=False)
    assert gpu.shape==(m,)
    assert not (out/'reload-started.json').exists() and not (out/'reload.json').exists()
    cpu_path=model/'reloaded-cpu-logits.npy'
    assert not cpu_path.exists()
    save(out/'reload-started.json',dict(standalone_python_module_process=True,
        process_id=os.getpid(),epoch=epoch,checkpoint_sha256=selected['checkpoint_sha256'],
        registration_sha256=sha(out/'registration.json'),attempts=1,training_runs=0))
    cpu_out=np.lib.format.open_memmap(cpu_path,mode='w+',dtype='<f8',shape=(m,))
    tolerance_failures=groups_failed=id_disagreements=0; max_abs=0.
    cpu_id_hash=hashlib.sha256();gpu_id_hash=hashlib.sha256()
    for group in range(n):
        start,end=map(int,offsets[group:group+2])
        # At most one complete group x373 is materialized; never all Mx373.
        values=network.score(shared[group],candidates[start:end])
        cpu_out[start:end]=values
        checked=comparison(values,gpu[start:end])
        tolerance_failures+=checked['tolerance_failures']
        groups_failed+=checked['tolerance_failures']>0
        max_abs=max(max_abs,checked['max_absolute_difference'])
        def selected_id(position):
            k=start+position;unit,x,y=map(int,identity[k])
            return candidate_id(unit,(x,y),OPS[int(ops[k])])
        cpu_id=selected_id(checked['cpu_index']);gpu_id=selected_id(checked['gpu_index'])
        id_disagreements+=cpu_id!=gpu_id
        cpu_id_hash.update((cpu_id+'\n').encode('utf-8'))
        gpu_id_hash.update((gpu_id+'\n').encode('utf-8'))
    cpu_out.flush();cpu_out._mmap.close()
    # Bind both summaries to the exact original immutable local reference files.
    assert sha(checkpoint)==selected['checkpoint_sha256'] and sha(gpu_path)==selected['gpu_logits_sha256']
    result=dict(passed=tolerance_failures==0 and id_disagreements==0,
        standalone_python_module_process=True,process_id=os.getpid(),
        epoch=epoch,checkpoint_sha256=selected['checkpoint_sha256'],
        selected_export_sha256=sha(model/'selected-export.json'),
        training_report_sha256=sha(model/'report.json'),
        cache_manifest_sha256=sha(cache/'manifest.json'),
        gpu_logits_sha256=selected['gpu_logits_sha256'],cpu_logits_sha256=sha(cpu_path),
        all_validation_groups=n,all_validation_candidates=m,atol=ATOL,rtol=RTOL,
        tolerance_failures=tolerance_failures,groups_with_tolerance_failure=groups_failed,
        max_absolute_difference=max_abs,candidate_id_disagreements=id_disagreements,
        cpu_candidate_id_digest=cpu_id_hash.hexdigest(),gpu_candidate_id_digest=gpu_id_hash.hexdigest(),
        raw_logits='local only',input_arrays=['shared','candidates'],
        metadata_use='candidate_identity and candidate_ops only compare output IDs; no seeds, labels or teacher scores scored',
        wall_seconds=time.perf_counter()-tick,device='CPU NumPy float64 reload of saved epoch weights',
        new_games=0,training_runs=0,checkpoint_reselection=False,
        failure_rule='Any tolerance or ID mismatch blocks game admission; retain this sole selected head and all old evidence, never try another epoch.')
    save(out/'reload.json',result);verify(out)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--out',type=Path,required=True)
    run(parser.parse_args().out,standalone=True)
