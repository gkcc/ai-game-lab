"""Read-only independent recomputation; emits one retained sanitized receipt."""
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import numpy as np

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
sys.path.insert(0, str(ROOT))
from learning.bc_head_fullbatch_v1.experiment import verify
from learning.bc_head_v1.experiment import model_gate


def read(p):
    return json.loads(p.read_text(encoding='utf-8'))


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def run():
    target = OUT/'independent-review-public-summary.json'
    assert not target.exists()
    cfg = verify(OUT)
    src = ROOT/cfg['original_experiment']
    model = OUT/'model'
    report = read(model/'report.json')
    history = read(model/'history.json')
    receipt = read(OUT/'pre-training-commit.json')
    assert receipt['registration_sha256'] == sha(OUT/'registration.json')
    assert receipt['freeze_sha256'] == sha(OUT/'freeze.json')
    public = ROOT.parent/'publication/ai-game-lab'
    for filename in (ROOT/'learning/bc_head_fullbatch_v1').glob('*.py'):
        relative = filename.relative_to(ROOT.parent).as_posix()
        blob = subprocess.check_output(['git', '-C', str(public), 'show', receipt['commit']+':'+relative])
        assert blob == filename.read_bytes()
    assert sha(model/'history.json') == report['history_sha256']
    with np.load(src/'data/dataset.npz', allow_pickle=False) as data:
        x,y,val,seeds = [data[k].copy() for k in ('x','y','validation','seed')]
    with np.load(src/'model/trained.npz', allow_pickle=False) as data:
        parent = {k:data[k].copy() for k in data.files}
    with gzip.open(src/'data/samples.jsonl.gz', 'rt', encoding='utf-8') as stream:
        samples = [json.loads(line) for line in stream]
    names = ['WHEAT','CARROT','TOMATO','STRAWBERRY','MELON']
    assert len(samples) == len(x) == 11483 and (~val).sum() == 9127
    assert all(s['seed'] == int(seed) and s['label'] == names[int(label)] for s,seed,label in zip(samples,seeds,y))
    hidden = (x.astype(np.float64)-parent['mean'].astype(np.float64))/parent['scale'].astype(np.float64)
    for index in (0,1):
        hidden = np.maximum(np.matmul(hidden,parent['w'+str(index)].astype(np.float64))+parent['b'+str(index)].astype(np.float64),0)
    original_logits = hidden @ parent['w2'].astype(np.float64)+parent['b2'].astype(np.float64)
    base_prediction = original_logits.argmax(axis=1)
    keys = [row.tobytes() for row in x]
    seen = {key for key,validation in zip(keys,val) if not validation}
    selected = (~val)&(y==3)&(base_prediction!=3)&np.isin([s['outcome'] for s in samples],['request_seed','plant_request'])
    assert selected.sum() == 160
    groups = dict(train=~val,validation=val,validation_unseen=val & np.array([key not in seen for key in keys]),selected_strawberry_errors=selected)
    for seed in sorted(set(seeds)): groups['seed_'+str(seed)] = seeds==seed
    for split,mask in [('train',~val),('validation',val)]:
        for index,name in enumerate(names): groups[split+'_class_'+name] = mask & (y==index)
    counts = np.bincount(y[~val],minlength=5)
    weights = (1/np.sqrt(counts)).astype(np.float32)
    weights /= weights.mean()
    assert weights.tolist() == report['class_weights'] and counts.tolist() == report['class_counts']
    checked_groups = 0
    max_error = 0.
    best_loss = float('inf')
    best = None
    prior = None
    steps = []
    all_metrics = []
    assert len(history) == 21
    for epoch,row in enumerate(history):
        assert row['epoch'] == row['optimizer_updates'] == epoch
        file = model/row['head_file']
        assert file.parent == model/'heads' and sha(file) == row['head_sha256']
        with np.load(file,allow_pickle=False) as data:
            w,b,prediction = [data[k].copy() for k in ('w2','b2','prediction')]
            assert int(data['step']) == epoch
        assert w.dtype == b.dtype == np.float32 and w.shape == (64,5) and b.shape == (5,)
        assert np.isfinite(w).all() and np.isfinite(b).all()
        if epoch == 0:
            assert np.array_equal(w,parent['w2']) and np.array_equal(b,parent['b2'])
        else:
            assert not (np.array_equal(w,prior[0]) and np.array_equal(b,prior[1]))
            assert np.isfinite(row['gpu_pre_update_weighted_loss']) and np.isfinite(row['gpu_gradient_norm']) and row['gpu_gradient_norm'] > 0
        prior = (w,b)
        assert row['changed_head_tensors'] == sum(not np.array_equal(a,c) for a,c in zip((w,b),(parent['w2'],parent['b2'])))
        logits = hidden @ w.astype(np.float64)+b.astype(np.float64)
        pred = logits.argmax(axis=1)
        assert np.array_equal(pred,prediction)
        pred_sha = hashlib.sha256(pred.astype('<i8').tobytes()).hexdigest()
        assert pred_sha == row['prediction_sha256']
        shifted = logits-logits.max(axis=1,keepdims=True)
        losses = np.logaddexp.reduce(shifted,axis=1)-shifted[np.arange(len(y)),y]
        metrics = {}
        assert set(row['groups']) == set(groups)
        for name,mask in groups.items():
            truth,guess = y[mask],pred[mask]
            cm = np.bincount(5*truth+guess,minlength=25).reshape(5,5).tolist()
            result = dict(samples=int(mask.sum()),errors=int((guess!=truth).sum()),accuracy=float((guess==truth).mean()) if mask.any() else None,cross_entropy=float(losses[mask].mean()) if mask.any() else None,confusion=cm)
            saved = row['groups'][name]
            for key,value in result.items():
                if key == 'cross_entropy' and value is not None:
                    max_error = max(max_error,abs(value-saved[key]));assert np.isclose(value,saved[key],atol=1e-12,rtol=1e-12)
                else: assert value == saved[key],(epoch,name,key)
            metrics[name] = result
            checked_groups += 1
        weighted = np.average(losses[~val],weights=weights[y[~val]].astype(np.float64))
        assert np.isclose(weighted,row['train_weighted_cross_entropy'],atol=1e-12,rtol=1e-12)
        margin = logits[:,3]-logits[:,4]
        for name,saved in row['strawberry_melon_margins'].items():
            values = margin[groups[name]]
            assert saved['samples'] == len(values) and saved['positive'] == int((values>0).sum())
            for key,value in [('minimum',values.min()),('mean',values.mean()),('maximum',values.max())]: assert np.isclose(value,saved[key],atol=1e-12,rtol=1e-12)
        score = metrics['validation']['cross_entropy']
        winner = score < best_loss
        assert row['selected_at_epoch'] == winner
        if winner: best,best_loss,best_weights = epoch,score,(w,b)
        steps.append(dict(epoch=epoch,head_sha256=sha(file),prediction_sha256=pred_sha,validation_ce=score,selected_errors=metrics['selected_strawberry_errors']['errors']))
        all_metrics.append(metrics)
    assert best == report['best_epoch'] == 6
    assert sha(model/'trained.npz') == report['checkpoint_sha256']
    with np.load(model/'trained.npz',allow_pickle=False) as data:
        for key in cfg['training']['frozen']: assert data[key].dtype == parent[key].dtype and data[key].tobytes() == parent[key].tobytes()
        assert np.array_equal(data['w2'],best_weights[0]) and np.array_equal(data['b2'],best_weights[1])
    base,candidate = all_metrics[0],all_metrics[best]
    for field,actual in [('baseline',base),('candidate',candidate)]:
        for name,saved in report[field].items():
            assert {k:v for k,v in saved.items() if k!='cross_entropy'} == {k:v for k,v in actual[name].items() if k!='cross_entropy'}
            assert np.isclose(saved['cross_entropy'],actual[name]['cross_entropy'],atol=1e-12,rtol=1e-12)
    gates = dict(nonzero_epoch=best>0,selected_train_errors_halved=candidate['selected_strawberry_errors']['errors']<=80,validation_accuracy_guard=candidate['validation']['accuracy']>=base['validation']['accuracy']-.01,unseen_validation_accuracy_guard=candidate['validation_unseen']['accuracy']>=base['validation_unseen']['accuracy']-.02)
    assert all(report['offline_gates'][k] == v for k,v in gates.items()) and not all(gates.values()) and report['offline_gate_passed'] is False
    rejected = False
    try: model_gate(OUT)
    except AssertionError: rejected = True
    assert rejected
    absent = ['candidate-freeze.json','games','dev-ledger.json','entry-dev-ledger.json','results.json']
    assert not any((OUT/name).exists() for name in absent)
    protection = read(OUT/'protection-before.json')
    for name,digest in protection['runtime'].items(): assert sha(ROOT.parent/name) == digest,name
    same = 0;appended = []
    for name,digest in protection['public'].items():
        if sha(public/name) == digest: same += 1;continue
        assert name in ('review/PROGRESS.md','review/WEBCHAT_REVIEW.md')
        before = subprocess.check_output(['git','-C',str(public),'show',cfg['baseline_head']+':'+name])
        assert hashlib.sha256(before).hexdigest() == digest and (public/name).read_bytes().startswith(before)
        appended.append(name)
    device = read(model/'device-local.json')
    assert device['platform'] == 'gpu' and bool(device['device_kind'])  # Model identity verified only in private original review.
    assert report['gpu_gradient_devices'] == ['cuda:0'] and report['gpu_batch_device'] == 'cuda:0'
    assert report['optimizer_updates'] == report['nonzero_gradient_updates'] == 20
    result = dict(passed=True,script_sha256=sha(Path(__file__)),pre_training_commit=receipt['commit'],registration_sha256=sha(OUT/'registration.json'),freeze_sha256=sha(OUT/'freeze.json'),checkpoint_sha256=sha(model/'trained.npz'),history_sha256=sha(model/'history.json'),heads_recomputed=21,adjacent_head_changes=20,group_metric_checks=checked_groups,maximum_ce_difference=max_error,best_epoch=best,selected_errors=candidate['selected_strawberry_errors']['errors'],offline_gates=gates,offline_gate_passed=False,actual_model_gate_rejected=True,protected_runtime_files=len(protection['runtime']),public_byte_identical=same,public_append_only=appended,new_games=0,review_training_runs=0,review_parameter_updates=0,steps=steps,limitations='Independent CPU recomputation of every retained head, metrics and gates. GPU execution relies on frozen code and run evidence; no rerun of training, no optimizer-state reconstruction, no independent economic or generalization claim.')
    target.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='steps'}))


if __name__ == '__main__': run()
