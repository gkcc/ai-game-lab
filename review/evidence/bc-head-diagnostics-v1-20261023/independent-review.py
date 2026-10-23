"""Independent read-only recomputation; writes one new, sanitized audit receipt."""
import collections
from copy import deepcopy
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

RUN = Path(__file__).resolve().parent
ROOT = RUN.parents[1]
sys.path.insert(0, str(ROOT))
from learning.bc_crop.agent import Network, sha


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def signature(x):
    return hashlib.sha256(np.asarray(x, dtype='<f4').tobytes()).hexdigest()


def collisions(hidden, labels):
    bins = collections.defaultdict(set)
    for row, label in zip(hidden, labels):
        value = row.copy()
        value[value == 0] = 0
        bins[value.astype('<f8').tobytes()].add(int(label))
    return dict(unique_hidden=len(bins), different_label_groups=sum(len(v) > 1 for v in bins.values()))


def main():
    destination = RUN / 'independent-review-summary.json'
    assert not destination.exists(), 'Retain earlier independent review receipts'
    cfg, freeze = read(RUN/'registration.json'), read(RUN/'freeze.json')
    old, result = read(RUN/'results.json'), read(RUN/'results-corrected.json')
    repair, certificate = read(RUN/'repair-registration.json'), read(RUN/'curvature-repair-certificate.json')
    assert sha(RUN/'registration.json') == freeze['registration_sha256']
    for name, digest in {**freeze['code'], **cfg['input_sha256']}.items():
        assert sha(ROOT/name) == digest, name
    assert sha(RUN/'results.json') == repair['original_result_sha256'] == certificate['original_result_sha256']
    assert sha(RUN/'freeze.json') == repair['original_freeze_sha256']
    assert sha(ROOT/'learning/bc_head_diagnostics/curvature_repair.py') == repair['repair_source_sha256']
    assert sha(RUN/'results-corrected.json') == certificate['corrected_result_sha256']
    reverted = deepcopy(result)
    reverted.pop('numerical_correction')
    changes = []
    def restore(a, b, prefix):
        for kind in a:
            for target in a[kind]['targets']:
                actual, prior = a[kind]['targets'][target], b[kind]['targets'][target]
                assert actual['derivative'] == prior['derivative']
                if actual['curvature'] != prior['curvature']:
                    assert target == 'train_weighted'
                    changes.append(dict(location=prefix+'/'+kind+'/'+target, old=prior['curvature'], corrected=actual['curvature']))
                actual['curvature'] = prior['curvature']
    for name in reverted['full_train_directions']:
        restore(reverted['full_train_directions'][name], old['full_train_directions'][name], name)
    for a, b in zip(reverted['independent_zero_state_batches'], old['independent_zero_state_batches']):
        restore(a['directions'], b['directions'], 'batch_'+str(a['batch']))
    assert reverted == old and len(changes) == 76
    assert changes == certificate['changed_curvatures']

    source = ROOT/cfg['original_experiment']
    with np.load(source/'data/dataset.npz', allow_pickle=False) as z:
        x, y, val, seeds = [z[k].copy() for k in ('x', 'y', 'validation', 'seed')]
    with gzip.open(source/'data/samples.jsonl.gz', 'rt', encoding='utf-8') as stream:
        samples = [json.loads(line) for line in stream]
    net = Network(source/'model/trained.npz', sha(source/'model/trained.npz'))
    normalized = (x.astype('float64')-net.mean)/net.scale
    h = normalized.copy()
    for w, b in net.weights[:2]:
        h = np.maximum(h @ w+b, 0)
    H = np.column_stack([h, np.ones(len(h))])
    theta = np.vstack(net.weights[-1])
    theta.flags.writeable = False
    theta_bytes = theta.tobytes()
    prediction = net.logits(x, net.meta['features']).argmax(1)
    train = ~val
    ids = np.flatnonzero(train)
    signatures = np.array([signature(row) for row in x])
    _, first = np.unique(signatures[train], return_index=True)
    unique_ids = ids[np.sort(first)]
    dedup = np.zeros(len(y), bool)
    dedup[unique_ids] = True
    selected = train & (y == 3) & (prediction != 3) & np.isin([s['outcome'] for s in samples], ['request_seed', 'plant_request'])
    assert selected.sum() == 160
    counts = np.bincount(y[train], minlength=5)
    class_weights = (1/np.sqrt(counts)).astype('float32')
    class_weights /= class_weights.mean()
    assert class_weights.tolist() == result['class_weights']
    specs = {}
    for name in result['objectives']:
        if name == 'train_unique_unweighted': mask = dedup
        elif name.startswith('train_seed_'): mask = train & (seeds == int(name.rsplit('_', 1)[1]))
        elif name.startswith('validation_seed_'): mask = val & (seeds == int(name.rsplit('_', 1)[1]))
        elif name.startswith('train_class_'): mask = train & (y == net.meta['crops'].index(name[len('train_class_'):]))
        elif name == 'validation': mask = val
        elif name == 'selected_errors': mask = selected
        else: mask = train
        specs[name] = (mask, class_weights[y] if name == 'train_weighted' else np.ones(len(y)))

    def independent_objective(mask, weights):
        design, label = H[mask], y[mask]
        weight = weights[mask].astype('float64')
        weight /= weight.sum()
        z = design @ theta
        z -= z.max(1)[:, None]
        exponent = np.exp(z)
        probability = exponent/exponent.sum(1)[:, None]
        loss = float(np.dot(weight, np.log(exponent.sum(1))-z[np.arange(len(label)), label]))
        gradient = np.einsum('ij,ik,i->jk', design, probability-np.eye(5)[label], weight)
        return loss, gradient, probability, weight

    values = {name: independent_objective(*spec) for name, spec in specs.items()}
    for name, (loss, gradient, _, _) in values.items():
        observed = result['objectives'][name]
        assert observed['samples'] == int(specs[name][0].sum())
        assert np.isclose(loss, observed['loss'], atol=1e-14, rtol=1e-12)
        assert np.isclose(np.linalg.norm(gradient), observed['gradient_norm'], atol=1e-13, rtol=1e-11)
    for name, targets in result['gradient_cosines'].items():
        for target, expected in targets.items():
            a, b = values[name][1], values[target][1]
            cosine = np.dot(a.ravel(), b.ravel())/(np.linalg.norm(a)*np.linalg.norm(b))
            assert np.isclose(cosine, expected, atol=1e-13, rtol=1e-11)
    checked = 0
    def check_directions(gradient, report):
        nonlocal checked
        for kind, data in report.items():
            direction = -.0003*gradient if kind == 'sgd' else -.0003*gradient/(abs(gradient)+1e-8)
            assert np.isclose(np.linalg.norm(direction), data['norm'], atol=1e-13, rtol=1e-10)
            for target, expected in data['targets'].items():
                _, target_gradient, probability, weight = values[target]
                projection = H[specs[target][0]] @ direction
                derivative = np.dot(target_gradient.ravel(), direction.ravel())
                curvature = np.dot(weight, (probability*projection**2).sum(1)-(probability*projection).sum(1)**2)
                assert np.isfinite(derivative) and np.isfinite(curvature) and curvature >= 0
                assert np.isclose(derivative, expected['derivative'], atol=1e-13, rtol=1e-10)
                assert np.isclose(curvature, expected['curvature'], atol=1e-15, rtol=1e-10)
                checked += 1
    for name, data in result['full_train_directions'].items():
        check_directions(values[name][1], data)
    permutation = np.random.default_rng(2026102321).permutation(len(ids))
    for i, batch in enumerate(result['independent_zero_state_batches']):
        chosen = ids[permutation[i*256:(i+1)*256]]
        assert hashlib.sha256(chosen.astype('<i8').tobytes()).hexdigest() == batch['row_indices_sha256']
        mask = np.zeros(len(y), bool)
        mask[chosen] = True
        loss, gradient, _, _ = independent_objective(mask, class_weights[y])
        assert len(chosen) == batch['samples']
        assert np.isclose(loss, batch['weighted_loss'], atol=1e-14, rtol=1e-12)
        check_directions(gradient, batch['directions'])
    for kind, targets in result['batch_direction_sign_counts'].items():
        for target, expected in targets.items():
            d = np.array([b['directions'][kind]['targets'][target]['derivative'] for b in result['independent_zero_state_batches']])
            assert dict(descent=int((d < -1e-12).sum()), ascent=int((d > 1e-12).sum()), neutral=int((abs(d) <= 1e-12).sum())) == expected

    def one_hidden(raw):
        hidden = (raw.astype('float64')-net.mean)/net.scale
        for w, b in net.weights[:2]: hidden = np.maximum(hidden @ w+b, 0)
        return hidden
    single = np.array([one_hidden(row) for row in x])
    same_x_splits = {}
    for name, mask in [('train', train), ('all', np.ones(len(y), bool))]:
        buckets = collections.defaultdict(list)
        for i in np.flatnonzero(mask): buckets[signatures[i]].append(i)
        split = [indices for indices in buckets.values() if len({h[i].tobytes() for i in indices}) > 1]
        same_x_splits[name] = dict(groups=len(split), max_abs_difference=max(float(np.max(abs(h[q]-h[q[0]]))) for q in split))
    query = {}
    for i in np.flatnonzero(selected):
        key = ('selected_errors', signatures[i])
        query.setdefault(key, dict(x=x[i], label=int(y[i]), calls=0))['calls'] += 1
    probe = ROOT/cfg['probe_experiment']
    for row in read(probe/'panel-ledger.json'):
        if row['policy'] != 'probe': continue
        with gzip.open(probe/'traces'/(row['game']+'.jsonl.gz'), 'rt', encoding='utf-8') as stream:
            for turn in map(json.loads, stream):
                for record in turn['records']:
                    if not record['correction_applied']: continue
                    raw = np.asarray(record['x'], np.float32)
                    key = ('probe', signature(raw))
                    query.setdefault(key, dict(x=raw, label=net.meta['crops'].index(record['teacher']), calls=0))['calls'] += 1
    assert len(query) == len(result['neighbors']) == 35
    neighbor_changes, distance_difference = 0, 0.
    pool_hash, pool_y = signatures[unique_ids], y[unique_ids]
    for observed in result['neighbors']:
        q = query[observed['group'], observed['feature_sha256']]
        hidden = one_hidden(q['x'])
        score = hidden @ net.weights[2][0]+net.weights[2][1]
        assert q['calls'] == observed['calls'] and net.meta['crops'][q['label']] == observed['teacher']
        assert net.meta['crops'][int(score.argmax())] == observed['prediction']
        assert float(score[3]-score[4]) == observed['strawberry_minus_melon_margin']
        for name in ('different_label', 'same_label_other_x'):
            candidates = [i for i in range(len(unique_ids)) if (pool_y[i] != q['label'] if name == 'different_label' else pool_y[i] == q['label'] and pool_hash[i] != observed['feature_sha256'])]
            distance = np.sqrt(np.sum((h[unique_ids]-hidden)**2, axis=1))
            best = min(candidates, key=lambda i: (distance[i], pool_hash[i]))
            actual = observed[name]
            assert actual['feature_sha256'] == pool_hash[best] and actual['label'] == net.meta['crops'][int(pool_y[best])]
            assert np.isclose(actual['hidden_distance'], distance[best], atol=1e-14, rtol=1e-13)
            raw_distance = np.linalg.norm(normalized[unique_ids[best]]-(q['x'].astype('float64')-net.mean)/net.scale)
            assert np.isclose(actual['normalized_input_distance'], raw_distance, atol=1e-14, rtol=1e-13)
            canonical = np.sqrt(np.sum((single[unique_ids]-hidden)**2, axis=1))
            other = min(candidates, key=lambda i: (canonical[i], pool_hash[i]))
            neighbor_changes += best != other
            distance_difference = max(distance_difference, abs(float(canonical[other]-distance[best])))
    canonical_collisions = {}
    for name, mask in [('train', train), ('train_and_developer_validation', np.ones(len(y), bool))]:
        assert collisions(h[mask], y[mask]) == result['collisions'][name]
        canonical_collisions[name] = collisions(single[mask], y[mask])
    single_logits = single @ net.weights[2][0]+net.weights[2][1]
    assert np.array_equal(single_logits.argmax(1), prediction)
    assert theta.tobytes() == theta_bytes and not theta.flags.writeable

    protected = read(RUN/'protection-before.json')
    for name, digest in protected['runtime'].items(): assert sha(ROOT.parent/name) == digest, name
    public = ROOT.parent/'publication/ai-game-lab'
    exact, appended = 0, []
    for name, digest in protected['public'].items():
        current = (public/name).read_bytes()
        if hashlib.sha256(current).hexdigest() == digest: exact += 1
        else:
            assert name in ('review/PROGRESS.md', 'review/WEBCHAT_REVIEW.md')
            original = subprocess.check_output(['git', '-C', str(public), 'show', cfg['baseline_head']+':'+name])
            assert hashlib.sha256(original).hexdigest() == digest and current.startswith(original)
            appended.append(name)
    summary = dict(passed=True, reviewer='independent sub-agent', script_sha256=sha(Path(__file__)),
        original_result_sha256=sha(RUN/'results.json'), corrected_result_sha256=sha(RUN/'results-corrected.json'),
        source_freeze_sha256=sha(RUN/'freeze.json'), repair_registration_sha256=sha(RUN/'repair-registration.json'),
        changed_curvatures=76, all_other_original_fields_exact=True, independent_objectives=len(values),
        derivative_and_curvature_target_checks=checked, independent_batches=36, queries=35, neighbor_checks=70,
        canonical_single_vector_collisions=canonical_collisions, same_raw_x_batch_splitting=same_x_splits,
        batch_vs_single_max_hidden_difference=float(np.max(abs(h-single))), changed_predictions=0,
        canonical_neighbor_identity_changes=int(neighbor_changes), canonical_neighbor_max_distance_difference=distance_difference,
        protected_runtime_files=len(protected['runtime']), original_public_files=len(protected['public']),
        public_byte_identical=exact, public_append_only=appended,
        real_parameter_updates=0, training_runs=0, new_games=0, new_checkpoints=0,
        limitations='Local float64 derivatives only; no reconstruction of GPU training, no separability proof or economic causal attribution.')
    destination.write_text(json.dumps(summary, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == '__main__':
    main()
