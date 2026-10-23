"""One registered, bounded GPU group-softmax fit; no automatic retries or games.

Shared normalization counts training groups once; local normalization counts real
training candidates once. Whole groups are never split or truncated. Accumulated
microbatch gradients share the complete effective batch weight denominator.
"""
import argparse
import hashlib
import json
import math
import multiprocessing
import os
from pathlib import Path
import shutil
import time

os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
import numpy as np

from .model import DIMENSIONS, TaskNetwork, continuous_mask, numpy_logits, sha
from learning.task_choice_v1.features import SCHEMA_VERSION, OPS, schema_hash

FIXED = dict(epochs=8, maximum_updates=3400, maximum_synchronized_training_minutes=30,
             maximum_process_minutes_including_compile=45, effective_batch_decision_groups=256,
             microbatch_padded_candidate_budget=65536, learning_rate=0.0003, gradient_norm_clip=1)


def save(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, sort_keys=True, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def load_split(cache, split):
    root = Path(cache)/split
    fields = ('shared','candidates','offsets','labels','selected_ops','candidate_ops','candidate_identity','seeds')
    data = {k: np.load(root/(k+'.npy'), mmap_mode='r', allow_pickle=False) for k in fields}
    n, m = len(data['shared']), len(data['candidates'])
    offsets, labels = data['offsets'], data['labels']
    if (data['shared'].shape != (n,275) or data['candidates'].shape != (m,98)
            or offsets.shape != (n+1,) or labels.shape != (n,)
            or data['selected_ops'].shape != (n,) or data['candidate_ops'].shape != (m,)
            or data['candidate_identity'].shape != (m,3) or data['seeds'].shape != (n,)
            or offsets[0] != 0 or offsets[-1] != m):
        raise ValueError('Invalid ragged cache shape')
    size = np.diff(offsets)
    if not n or np.any(size < 1) or np.any(size > 65536) or np.any(labels < 0) or np.any(labels >= size):
        raise ValueError('Empty, oversized or invalid whole candidate group')
    if np.any(data['candidate_ops'] < 0) or np.any(data['candidate_ops'] >= 5):
        raise ValueError('Invalid cached operation')
    if not np.array_equal(data['candidate_ops'][offsets[:-1]+labels], data['selected_ops']):
        raise ValueError('Selected operation label mismatch')
    return data


def verify_cache(cache):
    manifest=read(Path(cache)/'manifest.json')
    if not manifest['passed'] or sha(Path(cache)/'schema.json') != manifest['schema_sha256']:
        raise ValueError('Invalid cache manifest or schema identity')
    for split in ('train','validation'):
        for name, info in manifest['arrays'][split].items():
            path=Path(cache)/info['path']
            if path.resolve() != (Path(cache)/split/(name+'.npy')).resolve() or sha(path) != info['sha256']:
                raise ValueError('Cache array path or identity mismatch')
            array=np.load(path,mmap_mode='r',allow_pickle=False)
            if list(array.shape)!=info['shape'] or str(array.dtype)!=info['dtype']:
                raise ValueError('Cache array shape or dtype mismatch')
    return manifest


def moments(array):
    total = np.zeros(array.shape[1], np.float64)
    squares = total.copy()
    for start in range(0, len(array), 32768):
        x = np.asarray(array[start:start+32768], dtype=np.float64)
        if not np.isfinite(x).all():
            raise ValueError('Nonfinite training data')
        total += x.sum(axis=0)
        squares += np.square(x).sum(axis=0)
    mean = total / len(array)
    return mean, np.sqrt(np.maximum(0, squares/len(array)-mean*mean))


def normalization(data, schema):
    shared_mean, shared_std = moments(data['shared'])
    local_mean, local_std = moments(data['candidates'])
    mean = np.concatenate((shared_mean, local_mean))
    scale = np.maximum(1., np.concatenate((shared_std, local_std)))
    mask = continuous_mask(schema['shared_feature_names']+schema['feature_names'])
    mean[~mask], scale[~mask] = 0., 1.
    return mean, scale


def class_weights(selected):
    counts = np.bincount(selected, minlength=5)
    if np.any(counts == 0):
        raise ValueError('All five operations require training labels')
    weights = np.clip(np.sqrt(len(selected)/(5*counts)), .5, 5.)
    weights /= np.dot(counts, weights)/len(selected)
    return counts, weights.astype(np.float32)


def microbatches(data, ids, budget=65536):
    """Deterministic power-of-two buckets; each emitted group remains whole."""
    buckets = {}
    for identifier in ids:
        size = int(data['offsets'][identifier+1]-data['offsets'][identifier])
        width = 1 << (size-1).bit_length()
        if width > budget:
            raise ValueError('Whole group padding exceeds candidate budget')
        buckets.setdefault(width, []).append(int(identifier))
    for width, group_ids in sorted(buckets.items()):
        capacity = min(256, budget//width)
        for start in range(0, len(group_ids), capacity):
            yield group_ids[start:start+capacity], capacity, width


def padded(data, ids, capacity, width, mean, scale, weights):
    x = np.zeros((capacity, width, 373), np.float32)
    mask = np.zeros((capacity, width), bool)
    # Empty padding rows have one harmless dummy candidate and group weight 0.
    mask[:, 0] = True
    labels = np.zeros(capacity, np.int32)
    group_weights = np.zeros(capacity, np.float32)
    for row, identifier in enumerate(ids):
        lo, hi = map(int, data['offsets'][identifier:identifier+2])
        length = hi-lo
        if length > width:
            raise ValueError('Candidate truncation is forbidden')
        x[row,:length,:275] = (data['shared'][identifier]-mean[:275])/scale[:275]
        x[row,:length,275:] = (data['candidates'][lo:hi]-mean[275:])/scale[275:]
        mask[row,:length] = True
        labels[row] = data['labels'][identifier]
        group_weights[row] = weights[data['selected_ops'][identifier]]
    return x, mask, labels, group_weights


def weighted_group_loss(logits, mask, labels, group_weights, denominator, xp=np):
    """Backend-generic stable loss used by the actual differentiated GPU path."""
    scores = xp.where(mask, logits, -xp.inf)
    shifted = scores-xp.max(scores,axis=1,keepdims=True)
    losses = xp.log(xp.sum(xp.exp(shifted),axis=1))-shifted[xp.arange(len(labels)),labels]
    return xp.sum(losses*group_weights)/denominator


def group_metrics(logits, data, weights):
    losses, predictions, actual_ops, predicted_ops, multi, cross, water_predictions = [], [], [], [], [], [], []
    for i, label in enumerate(data['labels']):
        lo, hi = map(int, data['offsets'][i:i+2])
        z = np.asarray(logits[lo:hi], dtype=np.float64)
        shifted = z-z.max()
        prediction = int(np.argmax(z))
        losses.append(float(np.log(np.exp(shifted).sum())-shifted[int(label)]))
        predictions.append(prediction)
        actual_ops.append(int(data['selected_ops'][i]))
        predicted_ops.append(int(data['candidate_ops'][lo+prediction]))
        multi.append(hi-lo > 1)
        cross.append(len(set(data['candidate_ops'][lo:hi].tolist())) > 1)
        water=np.flatnonzero(data['candidate_ops'][lo:hi] == OPS.index('WATER'))
        water_predictions.append(int(water[0]) if len(water) else 0)
    losses, predictions = np.asarray(losses), np.asarray(predictions)
    actual_ops, predicted_ops = np.asarray(actual_ops), np.asarray(predicted_ops)
    multi, cross = np.asarray(multi), np.asarray(cross)
    if not multi.any() or not cross.any():
        raise ValueError('Validation requires multicandidate and cross-operation decisions')
    recall = {}
    for op, name in enumerate(OPS):
        mask = cross & (actual_ops == op)
        if not mask.any():
            raise ValueError('Every operation needs cross-operation validation examples')
        recall[name] = dict(groups=int(mask.sum()), recall=float(np.mean(predicted_ops[mask] == op)))
    w = weights[actual_ops].astype(np.float64)
    def details(prediction, selected_mask):
        predicted_positions=data['offsets'][:-1]+prediction
        expected_positions=data['offsets'][:-1]+data['labels']
        actual_identity=data['candidate_identity'][expected_positions]
        predicted_identity=data['candidate_identity'][predicted_positions]
        predicted_operation=data['candidate_ops'][predicted_positions]
        confusion=np.zeros((5,5),int)
        np.add.at(confusion,(actual_ops[selected_mask],predicted_operation[selected_mask]),1)
        def summarize(mask):
            if not mask.any():return dict(groups=0)
            return dict(groups=int(mask.sum()),
                exact_top1=float(np.mean(prediction[mask]==data['labels'][mask])),
                unit_exact=float(np.mean(predicted_identity[mask,0]==actual_identity[mask,0])),
                tile_exact=float(np.mean(np.all(predicted_identity[mask,1:]==actual_identity[mask,1:],axis=1))),
                operation_exact=float(np.mean(predicted_operation[mask]==actual_ops[mask])))
        result=summarize(selected_mask)
        result['operation_confusion_rows_actual_columns_predicted']=confusion.tolist()
        result['per_operation']={name:summarize(selected_mask & (actual_ops==op)) for op,name in enumerate(OPS)}
        return result
    all_groups=np.ones(len(losses),bool)
    summary=details(predictions,all_groups)
    summary['multicandidate']=details(predictions,multi)
    summary['cross_operation']=details(predictions,cross)
    seed_summary={}
    for seed in sorted(set(data['seeds'].tolist())):
        mask=data['seeds']==seed
        seed_summary[str(seed)]=dict(details(predictions,mask),unweighted_ce=float(losses[mask].mean()),
                                    weighted_ce=float(np.dot(w[mask],losses[mask])/w[mask].sum()))
    operation_ce={name:dict(unweighted_ce=float(losses[actual_ops==op].mean()),
                           weighted_ce=float(np.dot(w[actual_ops==op],losses[actual_ops==op])/w[actual_ops==op].sum()))
                  for op,name in enumerate(OPS)}
    baselines={}
    for name,prediction in [('canonical_first',np.zeros(len(losses),int)),
                            ('water_first_else_canonical_first',np.asarray(water_predictions))]:
        baselines[name]=dict(all=details(prediction,all_groups),multicandidate=details(prediction,multi),
                             cross_operation=details(prediction,cross),not_used_for_checkpoint_selection=True)
    return dict(groups=len(losses), multicandidate_groups=int(multi.sum()), cross_operation_groups=int(cross.sum()),
                weighted_ce=float(np.dot(w, losses)/w.sum()),unweighted_ce=float(losses.mean()),
                multicandidate_exact_top1=float(np.mean(predictions[multi] == data['labels'][multi])),
                cross_operation_macro_recall=float(np.mean([v['recall'] for v in recall.values()])),
                cross_operation_recalls=recall,decision_details=summary,by_seed=seed_summary,
                by_operation_ce=operation_ce,descriptive_baselines=baselines,
                prediction_sha256=hashlib.sha256(predictions.astype('<i8').tobytes()).hexdigest())


def metric_gate(metrics, initial):
    return dict(multicandidate_top1=metrics['multicandidate_exact_top1'] >= .6,
                improves_initial_ten_points=metrics['multicandidate_exact_top1']-initial['multicandidate_exact_top1'] >= .10,
                cross_operation_macro_recall=metrics['cross_operation_macro_recall'] >= .7,
                all_cross_operation_recalls=all(v['recall'] >= .5 for v in metrics['cross_operation_recalls'].values()),
                weighted_ce_improves=metrics['weighted_ce'] < initial['weighted_ce'])


def deployment_logits(params, data, mean, scale):
    output = np.empty(len(data['candidates']), np.float64)
    # Stream candidates; never materialize the complete M x 373 matrix.
    for start in range(0, len(output), 8192):
        stop = min(start+8192, len(output))
        owner = np.searchsorted(data['offsets'][1:], np.arange(start, stop), side='right')
        x = np.concatenate((data['shared'][owner], data['candidates'][start:stop]), axis=1).astype(np.float64)
        output[start:stop] = numpy_logits(params, (x-mean)/scale)
    return output


def checkpoint(path, params, mean, scale, schema, epoch, cfg, cache_sha):
    metadata = dict(schema_version=SCHEMA_VERSION, schema_sha256=schema['schema_sha256'],
                    shared_feature_names=schema['shared_feature_names'], feature_names=schema['feature_names'],
                    ops=list(OPS), dimensions=list(DIMENSIONS), epoch=epoch,
                    model_seed=cfg['model_seed'], training=FIXED,
                    cache_manifest_sha256=cache_sha, decision='first canonical maximum over real teacher opportunities')
    if path.exists():
        raise FileExistsError('Do not overwrite checkpoints')
    np.savez_compressed(path, **params, mean=mean, scale=scale, metadata=np.asarray(json.dumps(metadata,sort_keys=True)))


def _fit(out, cfg):
    import jax
    import jax.numpy as jnp
    import optax

    started = time.monotonic()
    train_seconds, updates, nonzero = 0., 0, 0
    out = Path(out)
    model = out/'model'
    model.mkdir(exist_ok=False)
    heads = model/'checkpoints'
    heads.mkdir()
    cache = out/'cache'
    verify_cache(cache)
    schema = read(cache/'schema.json')
    if schema['schema_sha256'] != schema_hash(schema['feature_names'],schema['shared_feature_names']):
        raise ValueError('Cache schema mismatch')
    training, validation = load_split(cache, 'train'), load_split(cache, 'validation')
    expected_updates = 8*math.ceil(len(training['labels'])/256)
    if expected_updates > 3400:
        raise ValueError('Complete eight epochs exceed update budget')
    mean, scale = normalization(training, schema)
    counts, weights = class_weights(training['selected_ops'])
    device = jax.devices('gpu')[0]
    if device.platform != 'gpu':
        raise ValueError('Registered fit requires the locally verified GPU')
    save(model/'device-local.json', dict(device_kind=device.device_kind, device=str(device), platform=device.platform))
    rng = np.random.default_rng(cfg['model_seed'])
    host_params = {}
    for layer, (a,b) in enumerate(zip(DIMENSIONS[:-1],DIMENSIONS[1:])):
        host_params['w'+str(layer)] = (rng.standard_normal((a,b))*np.sqrt(2/a)).astype(np.float32)
        host_params['b'+str(layer)] = np.zeros(b,np.float32)
    initial_params = {k:v.copy() for k,v in host_params.items()}
    params = jax.device_put(host_params, device)
    optimizer = optax.chain(optax.clip_by_global_norm(1.), optax.adam(3e-4))
    optimizer_state = optimizer.init(params)

    def forward(p, x):
        for layer in range(3):
            x = x @ p['w'+str(layer)]+p['b'+str(layer)]
            if layer < 2:
                x = jax.nn.relu(x)
        return x[...,0]

    def weighted_loss(p, x, mask, labels, group_weights, denominator):
        return weighted_group_loss(forward(p,x),mask,labels,group_weights,denominator,jnp)

    loss_grad = jax.jit(jax.value_and_grad(weighted_loss))
    infer = jax.jit(forward)

    @jax.jit
    def apply(p, state, gradient):
        change, state = optimizer.update(gradient, state, p)
        return optax.apply_updates(p, change), state, optax.global_norm(gradient)

    cache_sha = sha(cache/'manifest.json')
    cpu_preparation_seconds=time.monotonic()-started
    validation_tick=time.monotonic()
    initial_logits = deployment_logits(host_params, validation, mean, scale)
    initial_metrics = group_metrics(initial_logits, validation, weights)
    cpu_validation_seconds=time.monotonic()-validation_tick
    checkpoint(heads/'epoch-00.npz', host_params, mean, scale, schema, 0, cfg, cache_sha)
    shutil.copyfile(heads/'epoch-00.npz',model/'initial.npz')
    history = [dict(epoch=0, updates=0, checkpoint_sha256=sha(heads/'epoch-00.npz'), metrics=initial_metrics)]
    save(model/'history.json',history)
    best_epoch, best_ce = None, float('inf')
    gradient_devices, loss_values, norms = set(), [], []
    packing_seconds=compile_seconds=synchronized_update_seconds=0.
    first_update_wall_seconds=None
    compiled_losses={}
    compiled_apply=None
    with jax.default_matmul_precision('highest'):
        for epoch in range(1,9):
            order = rng.permutation(len(training['labels']))
            visited = 0
            for start in range(0,len(order),256):
                if train_seconds >= 1800 or time.monotonic()-started >= 2700 or updates >= 3400:
                    raise TimeoutError('Registered complete-fit budget exhausted')
                tick = time.monotonic()
                ids = order[start:start+256]
                denominator = np.float32(weights[training['selected_ops'][ids]].sum(dtype=np.float64))
                accumulated = jax.tree_util.tree_map(jnp.zeros_like,params)
                batch_loss = jnp.asarray(0.,dtype=jnp.float32)
                batch_compile_seconds=batch_packing_seconds=0.
                for batch_ids,capacity,width in microbatches(training,ids):
                    pack_tick=time.monotonic()
                    arrays = padded(training,batch_ids,capacity,width,mean,scale,weights)
                    batch_packing_seconds+=time.monotonic()-pack_tick
                    placed = [jax.device_put(a,device) for a in arrays]
                    divisor=jax.device_put(denominator,device)
                    key=(capacity,width)
                    if key not in compiled_losses:
                        compile_tick=time.monotonic()
                        compiled_losses[key]=loss_grad.lower(params,*placed,divisor).compile()
                        batch_compile_seconds+=time.monotonic()-compile_tick
                    value, gradient = compiled_losses[key](params,*placed,divisor)
                    accumulated = jax.tree_util.tree_map(lambda a,b:a+b,accumulated,gradient)
                    batch_loss = batch_loss+value
                if compiled_apply is None:
                    compile_tick=time.monotonic()
                    compiled_apply=apply.lower(params,optimizer_state,accumulated).compile()
                    batch_compile_seconds+=time.monotonic()-compile_tick
                params,optimizer_state,norm = compiled_apply(params,optimizer_state,accumulated)
                norm.block_until_ready()
                value,norm_value = float(batch_loss),float(norm)
                elapsed=time.monotonic()-tick
                train_seconds += elapsed
                packing_seconds+=batch_packing_seconds
                compile_seconds+=batch_compile_seconds
                synchronized_update_seconds+=elapsed-batch_packing_seconds-batch_compile_seconds
                if first_update_wall_seconds is None:first_update_wall_seconds=elapsed
                if not np.isfinite(value) or not np.isfinite(norm_value):
                    raise ValueError('Nonfinite GPU loss or gradients')
                gradient_devices.update(str(d) for d in norm.devices())
                nonzero += int(norm_value > 0)
                loss_values.append(value); norms.append(norm_value)
                updates += 1; visited += len(ids)
                if train_seconds > 1800 or time.monotonic()-started >= 2700:
                    raise TimeoutError('Registered complete-fit time exhausted')
            assert visited == len(training['labels'])
            host_params = {k:np.asarray(v).copy() for k,v in params.items()}
            if not all(np.isfinite(v).all() for v in host_params.values()):
                raise ValueError('Nonfinite GPU-trained weights')
            path = heads/f'epoch-{epoch:02d}.npz'
            checkpoint(path,host_params,mean,scale,schema,epoch,cfg,cache_sha)
            validation_tick=time.monotonic()
            logits = deployment_logits(host_params,validation,mean,scale)
            metrics = group_metrics(logits,validation,weights)
            cpu_validation_seconds+=time.monotonic()-validation_tick
            gates = metric_gate(metrics,initial_metrics)
            if all(gates.values()) and metrics['weighted_ce'] < best_ce:
                best_epoch,best_ce = epoch,metrics['weighted_ce']
            history.append(dict(epoch=epoch,updates=updates,checkpoint_sha256=sha(path),metrics=metrics,
                                gates=gates,selected_at_epoch=best_epoch==epoch,groups_visited=visited,
                                synchronized_training_seconds=train_seconds))
            save(model/'history.json',history)
            save(out/'training-progress.json',dict(epoch=epoch,updates=updates,best_epoch=best_epoch))
            print(f'task epoch {epoch}/8 updates {updates} CE {metrics["weighted_ce"]:.6f} selected {best_epoch}',flush=True)
    assert updates == expected_updates and len(history) == 9
    final_changed = sum(not np.array_equal(initial_params[k],host_params[k]) for k in host_params)
    if not final_changed or nonzero == 0:
        raise ValueError('No actual GPU learning update')
    export = dict(checked=False,passed=False)
    if best_epoch is not None:
        selected_path = heads/f'epoch-{best_epoch:02d}.npz'
        network = TaskNetwork(selected_path,sha(selected_path))
        cpu_logits = deployment_logits(network.params,validation,network.mean,network.scale)
        gpu_params = jax.device_put({k:v.astype(np.float32) for k,v in network.params.items()},device)
        gpu_logits = np.empty_like(cpu_logits)
        with jax.default_matmul_precision('highest'):
            for start in range(0,len(cpu_logits),8192):
                end = min(start+8192,len(cpu_logits))
                owner = np.searchsorted(validation['offsets'][1:],np.arange(start,end),side='right')
                raw = np.concatenate((validation['shared'][owner],validation['candidates'][start:end]),axis=1)
                normalized = ((raw-network.mean)/network.scale).astype(np.float32)
                gpu_logits[start:end] = np.asarray(infer(gpu_params,jax.device_put(normalized,device)))
        np.save(model/'selected-gpu-logits.npy',gpu_logits,allow_pickle=False)
        save(model/'selected-export.json',dict(checkpoint_file=selected_path.relative_to(model).as_posix(),
             checkpoint_sha256=sha(selected_path),gpu_logits_file='selected-gpu-logits.npy',
             gpu_logits_sha256=sha(model/'selected-gpu-logits.npy'),cache_manifest_sha256=cache_sha,epoch=best_epoch))
        disagreements = sum(int(np.argmax(cpu_logits[lo:hi])) != int(np.argmax(gpu_logits[lo:hi]))
                            for lo,hi in zip(validation['offsets'][:-1],validation['offsets'][1:]))
        close = bool(np.allclose(cpu_logits,gpu_logits,atol=1e-4,rtol=1e-5))
        export = dict(checked=True,passed=close and disagreements==0,all_validation_candidates=len(cpu_logits),
                      validation_groups=len(validation['labels']),atol=1e-4,rtol=1e-5,
                      max_absolute_logit_error=float(np.max(np.abs(cpu_logits-gpu_logits))),
                      candidate_id_disagreements=disagreements)
        if export['passed']:
            shutil.copyfile(selected_path,model/'trained.npz')
            save(model/'checkpoint.json',dict(file='trained.npz',sha256=sha(model/'trained.npz'),epoch=best_epoch))
    report = dict(completed=True,epochs=8,optimizer_updates=updates,expected_updates=expected_updates,
                  initial_and_all_epochs_saved=True,best_epoch=best_epoch,
                  offline_metrics_passed=best_epoch is not None,export_validation=export,
                  offline_gate_passed=best_epoch is not None and export['passed'],
                  game_entry_requires_action_or_reservation_change=True,
                  actual_game_control_verified=False,new_process_numpy_reload_verified=False,
                  competitive_improvement_proven=False,
                  class_counts=counts.tolist(),class_weights=weights.tolist(),
                  normalization=dict(shared_training_groups=len(training['shared']),
                                     local_training_candidates=len(training['candidates']),padding_included=False,
                                     binary_preserved=True,standard_deviation_floor=1),
                  gpu_training=True,platform=device.platform,
                  gradient_devices=sorted(gradient_devices),nonzero_gradient_updates=nonzero,
                  changed_final_parameter_tensors=final_changed,
                  gradient_norm_min=float(min(norms)),gradient_norm_max=float(max(norms)),
                  training_loss_min=float(min(loss_values)),training_loss_max=float(max(loss_values)),
                  synchronized_training_seconds=train_seconds,total_fit_seconds=time.monotonic()-started,
                  timings=dict(cpu_preparation_seconds=cpu_preparation_seconds,
                               cpu_validation_seconds=cpu_validation_seconds,
                               cpu_training_pack_seconds=packing_seconds,
                               explicit_gpu_compilation_seconds=compile_seconds,
                               synchronized_update_and_transfer_seconds=synchronized_update_seconds,
                               first_update_wall_seconds=first_update_wall_seconds,
                               budgeted_training_loop_seconds=train_seconds,
                               budget_interpretation='Training limit includes packing, explicit compilation, transfers and synchronized updates; not CPU validation. Process limit includes everything.'),
                  history_sha256=sha(model/'history.json'),cache_manifest_sha256=cache_sha,
                  framework=dict(jax=jax.__version__,optax=optax.__version__,numpy=np.__version__),
                  official_simulation_on_gpu=False,teacher_fallback=False)
    save(model/'report.json',report)


def _worker(out,cfg):
    try:
        _fit(out,cfg)
    except BaseException as exc:
        save(Path(out)/'training-failed.json',dict(type=type(exc).__name__,automatic_retry_allowed=False,
                                                 fit_complete=False,games_allowed=False))
        raise


def train(out):
    from .experiment import ready
    out = Path(out).resolve()
    cfg = ready(out,phase='train')
    if any(cfg['training'].get(k) != value for k,value in FIXED.items()) or not isinstance(cfg.get('model_seed'),int):
        raise ValueError('Missing or altered registered training settings')
    if not (out/'pre-training-commit.json').is_file():
        raise ValueError('Missing frozen preregistration receipt')
    started = out/'training-started.json'
    if started.exists():
        raise FileExistsError('Single-fit start marker prohibits retries')
    save(started,dict(runs=1,maximum_process_seconds=2700,maximum_synchronized_training_seconds=1800))
    process = multiprocessing.get_context('spawn').Process(target=_worker,args=(str(out),cfg))
    process.start()
    try:
        process.join(timeout=2700)
    except BaseException:
        process.terminate();process.join(timeout=10)
        if process.is_alive():process.kill();process.join(timeout=10)
        save(out/'training-failed.json',dict(type='ParentInterrupted',automatic_retry_allowed=False,
                                           fit_complete=False,games_allowed=False))
        raise
    if process.is_alive():
        process.terminate();process.join(timeout=10)
        if process.is_alive():
            process.kill();process.join(timeout=10)
        save(out/'training-failed.json',dict(type='HardProcessTimeout',automatic_retry_allowed=False,
                                           fit_complete=False,games_allowed=False))
        raise TimeoutError('Registered 45-minute process budget reached')
    if process.exitcode != 0:
        raise RuntimeError('Registered task fit failed; preserve partial evidence without retry')
    if not read(out/'model/report.json')['completed']:
        raise RuntimeError('Incomplete registered task fit')


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    train(parser.parse_args().out)
