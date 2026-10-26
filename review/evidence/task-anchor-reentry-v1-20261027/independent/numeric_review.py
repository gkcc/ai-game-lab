"""One reserved independent numeric review; no training or game entrypoint."""
import argparse
from datetime import datetime, timezone, timedelta
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time

for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS'):
    os.environ[key] = '1'
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
HERE = Path(__file__).resolve().parent
OUT = HERE.parent
ROOT = OUT.parent.parent
sys.path.insert(0, str(ROOT))
from learning.task_anchor_once_v1.storage import owned_writes

CAP = 256 * 1024**2
PARTS = {'prepare': 32768, 'run': 32*1024**2, 'guard': 65536, 'dispatch': 65536}
used = 0
refused = False
mode = None
PARAMS = ('w0', 'b0', 'w1', 'b1', 'w2', 'b2')
SPLITS = {'old_train': ('old_cache', 'train'), 'old_validation': ('old_cache', 'validation'),
          'dagger_train': ('dagger_cache', 'train'), 'known_dev': ('dagger_cache', 'validation')}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while block := stream.read(1024*1024):
            h.update(block)
    return h.hexdigest()


def now():
    return datetime.now(timezone.utc)


def save(name, value):
    global used, refused
    body = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False)+'\n').encode()
    assert not refused and used+len(body) <= PARTS[mode]
    assert sum(p.stat().st_size for p in HERE.rglob('*') if p.is_file())+len(body) <= CAP
    budget = owned_writes(HERE, PARTS[mode]-used, allowed_paths=[name])
    try:
        with budget:
            target = HERE/name
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open('xb') as stream:
                stream.write(body)
    finally:
        used += budget.used_bytes
        refused = refused or bool(budget.snapshot()['refused'])


def array_sha(x):
    h = hashlib.sha256()
    h.update(str(x.dtype).encode('ascii'))
    h.update(json.dumps(list(x.shape)).encode('ascii'))
    h.update(x.tobytes(order='C'))
    return h.hexdigest()


def check_deadline(res):
    assert now() < datetime.fromisoformat(res['deadline_utc']), 'Independent review deadline'


def prepare():
    import numpy as np
    cfg = read(OUT/'registration.json')
    groups = read(OUT/'model/numerical-check-groups.json')
    assert not (HERE/'numeric-reservation.json').exists()
    assert read(OUT/'training-process-result.json')['passed']
    assert read(OUT/'fit-result.json')['completed']
    expected = '799f3f3238f10b5ee66166c91f529f162be59bc0cdb5cc1883f6d7d8b1150ea5'
    assert sha(OUT/'model/trained.npz') == expected
    selected = {}
    for name, (cachekey, split) in SPLITS.items():
        folder = ROOT/cfg[cachekey]
        manifest = read(folder/'manifest.json')
        assert sha(folder/'manifest.json') == cfg[cachekey+'_manifest_sha256']
        meta = manifest['arrays'][split]
        offsets_path = folder/meta['offsets']['path']
        assert sha(offsets_path) == meta['offsets']['sha256']
        offsets = np.load(offsets_path, mmap_mode='r', allow_pickle=False)
        n = meta['labels']['shape'][0]
        ids = [(2*i+1)*n//32 for i in range(16)]
        rows = sum(int(offsets[i+1]-offsets[i]) for i in ids)
        value = dict(groups=16, group_indices=ids, candidate_rows=rows, offset_array_sha256=array_sha(offsets))
        assert n >= 16 and rows <= 6250 and value == groups['splits'][name]
        selected[name] = value
    total = 4*sum(v['candidate_rows'] for v in selected.values())
    assert total == groups['candidate_row_presentations'] == 12776 and total <= 100000
    # Conservative result-review clock: this predates all result-review planning.
    start = datetime.fromisoformat(read(OUT/'pre-execution-commit.json')['created_utc'])
    deadline = start+timedelta(seconds=3600)
    assert now() < deadline
    assert shutil.disk_usage(HERE).free >= cfg['storage']['free_floor_bytes']+CAP
    names = ['registration.json', 'freeze.json', 'pre-review.json', 'pre-execution-commit.json',
             'fit-result.json', 'offline-gate.json', 'model-freeze.json', 'training-storage.json',
             'gpu-guard-result.json', 'training-process-result.json', 'model/numerical-check-groups.json',
             'model/initial.npz', 'model/trained.npz']
    names += ['model/'+x for x in ('initial-known-dev-cpu-logits.npy', 'final-known-dev-cpu-logits.npy',
              'initial-old-validation-cpu-logits.npy', 'final-old-validation-cpu-logits.npy',
              'final-known-dev-gpu-logits.npy')]
    save('numeric-reservation.json', dict(schema='independent-anchor-numeric-reservation-v1',
        created_utc=now().isoformat(), reviewer='independent_subagent', source_sha256=sha(__file__),
        input_sha256={name:sha(OUT/name) for name in names}, splits=selected,
        checkpoint_sha256=expected, cpu_rows_reserved=total//2, gpu_rows_reserved=total//2,
        total_rows_reserved=total, row_limit=100000, retry_allowed=False, replacement_allowed=False,
        review_started_upper_bound_utc=start.isoformat(), deadline_utc=deadline.isoformat(),
        time_limit_seconds=3600, output_limit_bytes=CAP, numerical_process_seconds=900,
        output_partitions=PARTS, native_log_bytes=1048576,
        rule='Four fixed midpoint complete-group samples; initial/final x independent float64 NumPy/float32 JAX.',
        full_saved_logits='Recalculate metrics only; no additional full-data model forward.',
        independent_cpu_saved_logit_atol=1e-4, independent_cpu_saved_logit_rtol=1e-5,
        cpu_gpu_atol=1e-4, cpu_gpu_rtol=1e-5, first_argmax_disagreements_required=0,
        metric_comparison_abs_tol=1e-12, metrics_use_independent_fsum=True,
        pre_reservation_offset_array_opens=4, model_array_decodes_before_reservation=0,
        official_reexecutions_authorized=False, neural_rows_before_reservation=0))
    print(json.dumps(dict(reservation_sha256=sha(HERE/'numeric-reservation.json'), total_rows=total,
                         cpu_rows=total//2, gpu_rows=total//2, passed=True)), flush=True)


def run_numeric():
    import numpy as np
    import jax
    import jax.numpy as jnp
    from learning.task_bc_v1.train import load_split
    from importlib.metadata import version
    res = read(HERE/'numeric-reservation.json')
    cfg, fit, gate = (read(OUT/name) for name in ('registration.json','fit-result.json','offline-gate.json'))
    debited = dict(cpu=0,gpu=0)
    completed = dict(cpu=0,gpu=0)
    calls = 0
    started = time.monotonic()
    check_deadline(res)
    assert sha(__file__) == res['source_sha256']
    assert all(sha(OUT/name) == digest for name,digest in res['input_sha256'].items())
    assert all(version(k) == v for k,v in cfg['gpu_environment']['versions'].items())
    assert sha(ROOT/cfg['task_checkpoint']) == cfg['task_checkpoint_sha256'] == sha(OUT/'model/initial.npz')
    assert not (HERE/'numeric-worker-started.json').exists()
    save('numeric-worker-started.json', dict(reservation_sha256=sha(HERE/'numeric-reservation.json'),
         created_utc=now().isoformat(), retry_allowed=False))
    device = jax.devices('gpu')[0]
    expected = read(OUT/'gpu-launch.local.json')['expected_device_substring']
    assert device.platform == 'gpu' and expected.lower() in str(device.device_kind).lower()
    save('numeric-device.local.json',dict(device_kind=str(device.device_kind),device=str(device),
         platform=device.platform,jax=jax.__version__))
    try:
        caches = {}
        checked_arrays = 0
        for key in ('old_cache','dagger_cache'):
            folder = ROOT/cfg[key]
            manifest = read(folder/'manifest.json')
            assert sha(folder/'manifest.json') == cfg[key+'_manifest_sha256']
            assert sha(folder/'schema.json') == manifest['schema_sha256']
            for split, fields in manifest['arrays'].items():
                for field, info in fields.items():
                    check_deadline(res)
                    assert info['path'] == split+'/'+field+'.npy'
                    assert sha(folder/info['path']) == info['sha256']
                    checked_arrays += 1
            for split in ('train','validation'):
                caches[key,split] = load_split(folder,split)
        data = {name:caches[key,split] for name,(key,split) in SPLITS.items()}
        assert set(np.unique(data['dagger_train']['seeds']).tolist()) == set(cfg['training_source_seeds'])
        assert set(np.unique(data['known_dev']['seeds']).tolist()) == set(cfg['known_screen_seeds'])
        roles = [set(np.unique(d['seeds']).tolist()) for d in data.values()]
        roles += [set(cfg['validation_seeds']),set(cfg['test_seeds'])]
        assert all(not a & b for i,a in enumerate(roles) for b in roles[i+1:])
        arrays = {}
        for name, path in [('initial','initial.npz'),('final','trained.npz')]:
            with np.load(OUT/'model'/path,allow_pickle=False) as z:
                arrays[name] = {k:np.array(z[k],copy=True) for k in (*PARAMS,'mean','scale')}
                arrays[name]['metadata'] = json.loads(str(z['metadata']))
            d = arrays[name]
            for i,(left,right) in enumerate(zip((373,128,64),(128,64,1))):
                assert d['w'+str(i)].shape == (left,right) and d['b'+str(i)].shape == (right,)
            assert all(np.isfinite(d[k]).all() for k in (*PARAMS,'mean','scale'))
            assert d['mean'].shape == d['scale'].shape == (373,) and (d['scale']>=1).all()
            for k in PARAMS:
                assert np.array_equal(d[k], d[k].astype(np.float32))
                actual = array_sha(d[k].astype(np.float32))
                assert actual == fit[name+'_parameter_sha256'][k]
        for k in ('mean','scale'):
            a,b = arrays['initial'][k],arrays['final'][k]
            assert a.dtype == b.dtype and a.shape == b.shape and a.tobytes()==b.tobytes()
            assert array_sha(a) == fit['frozen_normalizer_array_sha256'][k]
        schema = read(ROOT/cfg['old_cache']/'schema.json')
        for name in arrays:
            assert all(arrays[name]['metadata'][k]==schema[k] for k in
                       ('schema_version','schema_sha256','shared_feature_names','feature_names','ops'))
        changed = sum(not np.array_equal(arrays['initial'][k],arrays['final'][k]) for k in PARAMS)
        assert changed == fit['changed_parameter_tensors'] == 6
        counts = np.bincount(data['old_train']['selected_ops'],minlength=5)
        weights = np.minimum(5.,np.maximum(.5,np.sqrt(counts.sum()/(5.*counts))))
        weights = (weights/(np.dot(counts,weights)/counts.sum())).astype(np.float32)
        assert counts.tolist()==fit['old_training_class_counts'] and weights.tolist()==fit['old_training_class_weights']
        metrics,saved = {},{}
        for model in ('initial','final'):
            for split, filename in [('known_dev',model+'-known-dev-cpu-logits.npy'),
                                    ('old_validation',model+'-old-validation-cpu-logits.npy')]:
                check_deadline(res)
                scores=np.load(OUT/'model'/filename,mmap_mode='r',allow_pickle=False)
                d=data[split]; assert scores.shape==(len(d['candidates']),) and np.isfinite(scores).all()
                loss_terms=[]; group_weights=[]; predictions=[]
                for g,label in enumerate(d['labels']):
                    lo,hi=map(int,d['offsets'][g:g+2]);v=np.asarray(scores[lo:hi]);shift=v-v.max()
                    loss=float(np.log(np.exp(shift).sum())-shift[int(label)])
                    w=float(weights[int(d['selected_ops'][g])]);loss_terms.append(w*loss);group_weights.append(w)
                    predictions.append(int(np.argmax(v)))
                pred=np.asarray(predictions,dtype=np.int64)
                metric=dict(groups=len(pred),candidates=len(scores),weighted_ce=math.fsum(loss_terms)/math.fsum(group_weights),
                    exact_top1=float(np.mean(pred==d['labels'])),prediction_sha256=array_sha(pred))
                expected_metric=gate['metrics'][model+'_'+split]
                assert all(metric[k]==expected_metric[k] for k in ('groups','candidates','exact_top1','prediction_sha256'))
                assert math.isclose(metric['weighted_ce'],expected_metric['weighted_ce'],rel_tol=0,abs_tol=1e-12)
                metrics[model+'_'+split]=metric;saved[model,split]=scores
        saved_gpu=np.load(OUT/'model/final-known-dev-gpu-logits.npy',mmap_mode='r',allow_pickle=False)
        cpu=saved['final','known_dev'];offsets=data['known_dev']['offsets']
        disagreements=sum(int(np.argmax(cpu[int(lo):int(hi)]))!=int(np.argmax(saved_gpu[int(lo):int(hi)]))
                          for lo,hi in zip(offsets[:-1],offsets[1:]))
        full_parity=dict(max_absolute_error=float(np.max(np.abs(cpu-saved_gpu))),first_argmax_disagreements=disagreements,
                         close=bool(np.allclose(cpu,saved_gpu,atol=1e-4,rtol=1e-5)))
        assert np.isfinite(saved_gpu).all() and full_parity['close'] and disagreements==0
        assert full_parity['max_absolute_error']==gate['export_parity']['max_absolute_logit_error']
        ratios=dict(known_dev=metrics['final_known_dev']['weighted_ce']/metrics['initial_known_dev']['weighted_ce'],
                    old_validation=metrics['final_old_validation']['weighted_ce']/metrics['initial_old_validation']['weighted_ce'])
        assert ratios['known_dev']<=.95 and ratios['old_validation']<=1.02 and gate['passed']

        def cpu_forward(x,p):
            h0=np.maximum(np.matmul(x,p['w0'].astype(np.float64))+p['b0'].astype(np.float64),0.)
            h1=np.maximum(np.matmul(h0,p['w1'].astype(np.float64))+p['b1'].astype(np.float64),0.)
            return (np.matmul(h1,p['w2'].astype(np.float64))+p['b2'].astype(np.float64))[:,0]

        @jax.jit
        def gpu_forward(x,p):
            h0=jnp.maximum(jnp.matmul(x,p['w0'])+p['b0'],jnp.float32(0))
            h1=jnp.maximum(jnp.matmul(h0,p['w1'])+p['b1'],jnp.float32(0))
            return (jnp.matmul(h1,p['w2'])+p['b2'])[:,0]

        numeric=[]
        for split,d in data.items():
            selected=res['splits'][split];ids=selected['group_indices']
            assert ids==[(2*i+1)*len(d['labels'])//32 for i in range(16)]
            assert array_sha(d['offsets'])==selected['offset_array_sha256']
            blocks=[];ranges=[];lengths=[]
            for g in ids:
                lo,hi=map(int,d['offsets'][g:g+2]);lengths.append(hi-lo);ranges.append((lo,hi))
                blocks.append(np.hstack((np.repeat(d['shared'][g:g+1],hi-lo,axis=0),d['candidates'][lo:hi])))
            raw=np.concatenate(blocks).astype(np.float64);rows=len(raw)
            assert rows==selected['candidate_rows'] and rows<=6250 and np.isfinite(raw).all()
            local=np.r_[0,np.cumsum(lengths)]
            for model,p in arrays.items():
                x=(raw-p['mean'].astype(np.float64))/p['scale'].astype(np.float64)
                values={}
                for platform in ('cpu','gpu'):
                    check_deadline(res);assert sum(debited.values())+rows<=res['total_rows_reserved']
                    debited[platform]+=rows
                    call=dict(call=calls,split=split,model=model,platform=platform,rows=rows,
                              debited=dict(debited),completed_before=dict(completed),created_utc=now().isoformat())
                    save('calls/%02d-debit.json'%calls,call)
                    if platform=='cpu':scored=cpu_forward(x,p)
                    else:
                        params=jax.device_put({k:p[k].astype(np.float32) for k in PARAMS},device)
                        with jax.default_matmul_precision('highest'):
                            output=gpu_forward(jax.device_put(x.astype(np.float32),device),params)
                            output.block_until_ready()
                        assert {v.platform for v in output.devices()}=={'gpu'}
                        scored=np.asarray(output)
                    assert scored.shape==(rows,) and np.isfinite(scored).all()
                    completed[platform]+=rows;values[platform]=scored
                    save('calls/%02d-done.json'%calls,dict(completed=dict(completed),score_array_sha256=array_sha(scored)))
                    calls+=1
                argmax_diff=sum(int(np.argmax(values['cpu'][lo:hi]))!=int(np.argmax(values['gpu'][lo:hi])) for lo,hi in zip(local[:-1],local[1:]))
                assert np.allclose(values['cpu'],values['gpu'],atol=1e-4,rtol=1e-5) and argmax_diff==0
                item=dict(split=split,model=model,groups=16,candidates=rows,first_argmax_disagreements=argmax_diff,
                          max_cpu_gpu_absolute_error=float(np.max(np.abs(values['cpu']-values['gpu']))))
                if (model,split) in saved:
                    previous=np.concatenate([saved[model,split][lo:hi] for lo,hi in ranges])
                    assert np.allclose(previous,values['cpu'],atol=1e-4,rtol=1e-5)
                    assert all(np.argmax(previous[lo:hi])==np.argmax(values['cpu'][lo:hi]) for lo,hi in zip(local[:-1],local[1:]))
                    item['saved_cpu_max_absolute_error']=float(np.max(np.abs(previous-values['cpu'])))
                if model=='final' and split=='known_dev':
                    previous=np.concatenate([saved_gpu[lo:hi] for lo,hi in ranges])
                    assert np.allclose(previous,values['gpu'],atol=1e-4,rtol=1e-5)
                    assert all(np.argmax(previous[lo:hi])==np.argmax(values['gpu'][lo:hi]) for lo,hi in zip(local[:-1],local[1:]))
                    item['saved_gpu_max_absolute_error']=float(np.max(np.abs(previous-values['gpu'])))
                numeric.append(item)
        assert debited==completed=={'cpu':6388,'gpu':6388} and calls==16
        assert fit['offline_rows_total_used']+sum(debited.values())<=cfg['training']['offline_row_limit']
        assert all(sha(OUT/name)==digest for name,digest in res['input_sha256'].items())
        check_deadline(res)
        save('numeric-result.json',dict(schema='independent-anchor-numeric-result-v1',passed=True,
            created_utc=now().isoformat(),reservation_sha256=sha(HERE/'numeric-reservation.json'),source_sha256=sha(__file__),
            checkpoint_sha256=res['checkpoint_sha256'],tensor_identity_verified=True,changed_parameter_tensors=changed,
            normalizer_bits_preserved=True,known_screens_independent=False,metrics=metrics,ce_ratios=ratios,
            saved_full_gpu_cpu_parity=full_parity,independent_numeric=numeric,debited_rows=debited,completed_rows=completed,
            total_debited_rows=sum(debited.values()),total_completed_rows=sum(completed.values()),forward_calls=calls,
            cache_array_file_hashes_verified=checked_arrays,cache_array_memmap_opens=32,model_array_archive_decodes=2,
            saved_logit_array_opens=5,full_dataset_neural_forwards=0,official_reexecutions=0,training_updates=0,
            primary_plus_independent_offline_rows=fit['offline_rows_total_used']+sum(debited.values()),
            worker_seconds=time.monotonic()-started,worker_output_bytes_before_receipt=used,
            limitation='Independent numerical sampling and saved-logit metric recomputation; no economic or future-holdout claim.'))
    except BaseException as exc:
        try:save('numeric-failure.json',dict(passed=False,error_kind=type(exc).__name__,debited_rows=debited,
            completed_rows=completed,calls_completed=calls,retry_allowed=False))
        except BaseException:pass
        raise


def guard():
    from learning.task_dagger_once_v1.gpu_guard import supervise
    res=read(HERE/'numeric-reservation.json');check_deadline(res)
    assert sha(__file__)==res['source_sha256']
    save('numeric-guard-started.json',dict(created_utc=now().isoformat(),source_sha256=sha(__file__)))
    remaining=(datetime.fromisoformat(res['deadline_utc'])-now()).total_seconds()
    result=supervise([sys.executable,'-B',str(Path(__file__).resolve()),'run'],min(885.,remaining-15),
                     lambda:(HERE/'numeric-cancel.local.json').exists())
    save('numeric-guard-result.json',result)
    if not result['completed']:raise RuntimeError('Independent numeric worker failed; no retry')


def dispatch():
    from learning.task_anchor_once_v1.processes import CapturedProcess
    res=read(HERE/'numeric-reservation.json');check_deadline(res)
    assert sha(__file__)==res['source_sha256']
    save('numeric-dispatch-started.json',dict(created_utc=now().isoformat(),reservation_sha256=sha(HERE/'numeric-reservation.json')))
    launch=read(OUT/'gpu-launch.local.json')
    command=['wsl.exe','-d',launch['distro'],'--cd',launch['workdir'],'--',launch['python'],'-B',
             Path(__file__).resolve().relative_to(ROOT).as_posix(),'guard']
    child=None;error=None;started=time.monotonic();capture=None;transport=reader=None
    try:
        child=CapturedProcess(command,cwd=ROOT,log=HERE/'numeric-process.local.log',limit_bytes=1048576)
        while child.process.poll() is None:
            check_deadline(res);child.check()
            assert time.monotonic()-started<900,'Independent numeric process deadline'
            time.sleep(.2)
        capture=child.finish();assert child.process.returncode==0
    except BaseException as exc:error=exc
    finally:
        if child is not None:
            try:
                if child.process.poll() is None:
                    save('numeric-cancel.local.json',dict(cancel=True))
                    child.process.wait(timeout=max(15.,900-(time.monotonic()-started)))
                transport=child.process.poll() is not None
            except BaseException:transport=False
            try:child.thread.join(timeout=5);reader=not child.thread.is_alive()
            except BaseException:reader=False
    linux=read(HERE/'numeric-guard-result.json') if (HERE/'numeric-guard-result.json').exists() else {}
    clean=all(linux.get(k) is True for k in ('direct_child_reaped','owned_process_group_signalled','no_live_owned_session_members_before_reap'))
    passed=error is None and transport is True and reader is True and clean and time.monotonic()-started<=900
    save('numeric-dispatch-result.json',dict(passed=passed,error_kind=type(error).__name__ if error else None,
         transport_reaped=transport,reader_exited=reader,linux_session_cleanup=clean,
         elapsed_seconds=time.monotonic()-started,capture=capture,retry_allowed=False))
    if error is not None:raise error
    assert passed
    result=read(HERE/'numeric-result.json')
    print(json.dumps(dict(passed=result['passed'],debited_rows=result['debited_rows'],completed_rows=result['completed_rows'],
                         result_sha256=sha(HERE/'numeric-result.json'),cleanup_passed=passed)),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['prepare','run','guard','dispatch'])
    mode=parser.parse_args().mode
    {'prepare':prepare,'run':run_numeric,'guard':guard,'dispatch':dispatch}[mode]()
