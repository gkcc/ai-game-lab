"""Six preregistered synthetic cases; no original data or checkpoint loader.

Only run_regressions executes. Repeated sample consumption is counted each time,
including rejected rows. The caller freezes this file before running it once.
"""
from copy import deepcopy
import gzip
import hashlib
import json
from pathlib import Path
import pickle
import shutil
import time

import numpy as np

from learning.task_choice_v1.features import OPS, SCHEMA_VERSION, candidate_id, schema_hash
from . import pipeline as p


CASES = ('sparse_equivalence_and_train_isolation', 'duplicate_identity',
         'missing_identity', 'reordered_identity', 'misaligned_identity',
         'derived_decode_budget')


class SyntheticModel:
    sha = hashlib.sha256(b'stratified-fixed-synthetic-float64-scorer-v1').hexdigest()
    total_calls = 0
    total_candidates = 0

    def __init__(self):
        self.metadata = dict(schema_version=SCHEMA_VERSION,
            shared_feature_names=['shared.' + str(i) for i in range(275)],
            feature_names=['task.op.PLANT'] + ['candidate.' + str(i) for i in range(1, 98)],
            ops=list(OPS))
        self.metadata['schema_sha256'] = schema_hash(self.metadata['feature_names'],
                                                    self.metadata['shared_feature_names'])

    def score(self, shared, candidates):
        assert shared.dtype == np.float64 and candidates.dtype == np.float64
        assert shared.shape == (275,) and candidates.shape == (2, 98)
        SyntheticModel.total_calls += 1
        SyntheticModel.total_candidates += len(candidates)
        return candidates[:, 1] + shared[0]


def _input(model, step, side, *, novel=False):
    result = deepcopy(model.metadata)
    result['shared_x'] = [0.] * 275
    result['shared_x'][0] = 10. if novel else 0.
    result['observation'] = dict(step=step, player=side)
    result['candidates'] = []
    for target, op, value in (([0, 0], 'PLANT', 1.), ([1, 0], 'WATER', 2.)):
        local = [0.] * 98; local[1] = value
        result['candidates'].append(dict(id=candidate_id(0, target, op), unit=0,
                                         target=target, op=op, x=local))
    return result


def _json(path, value):
    Path(path).write_bytes(p.canonical(value) + b'\n')


def _gzip(path, rows):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as raw:
        with gzip.GzipFile(filename='', mode='wb', fileobj=raw, mtime=0) as stream:
            for row in rows:
                stream.write(p.canonical(row) + b'\n')


def _sample(root):
    root = Path(root); root.mkdir(parents=True, exist_ok=False)
    model = SyntheticModel()
    # Original manifest order deliberately starts with validation. Two training
    # games share a seed and have different seats/opponents. Empty groups stay.
    identities = [('validation', 'validation_a', 101, 0, 'round2'),
                  ('train', 'train_a', 1, 0, 'v1'),
                  ('train', 'train_b', 1, 1, 'market')]
    identities += [('train', 'empty_train_' + str(i), 2 + i // 2, i % 2,
                    ('v1', 'market', 'round2')[i % 3]) for i in range(14)]
    identities += [('validation', 'empty_validation_' + str(i), 101 + (i + 1) // 2,
                    (i + 1) % 2, ('v1', 'market', 'round2')[i % 3]) for i in range(3)]
    points = {'validation_a': [(4, 0, 5, 0)],
              'train_a': [(7, 240, 2, 0), (11, 240, 7, 1)],
              'train_b': [(9, 480, 3, 1)]}
    groups, rows_by_game = [], {}
    for split, game, seed, side, opponent in identities:
        rows, selections = [], []
        for source_row_index, step, decision, label_index in points.get(game, []):
            inputs = _input(model, step, side, novel=split == 'validation')
            label = dict(index=label_index, teacher_label=inputs['candidates'][label_index]['id'])
            meta = dict(game=game, split=split, seed=seed, side=side, opponent=opponent,
                        step=step, decision=decision,
                        input_sha256=hashlib.sha256(p.canonical(inputs)).hexdigest())
            rows.append(dict(inputs=inputs, label=label, meta=meta,
                             source_row_index=source_row_index))
            selections.append(dict(source_row_index=source_row_index, step=step,
                decision=decision, teacher_class=inputs['candidates'][label_index]['op'],
                phase_index=step // 240, candidate_count=2))
        indices = [r['source_row_index'] for r in rows]
        path = 'sample/' + game + '.jsonl.gz'
        _gzip(root / path, rows)
        groups.append(dict(split=split, game=game, seed=seed, side=side, opponent=opponent,
            path=path, rows=len(rows), candidates=2 * len(rows), sha256=p.sha(root / path),
            source_row_indices=indices,
            source_indices_sha256=hashlib.sha256(p.canonical(indices)).hexdigest(),
            source_sha256={kind: hashlib.sha256((game + ':' + kind).encode()).hexdigest()
                           for kind in ('inputs', 'labels', 'index')}, selections=selections))
        rows_by_game[game] = rows
    manifest = dict(schema=p.SAMPLE_SCHEMA, sample_only=True,
        source_manifest_sha256=hashlib.sha256(b'synthetic-source-manifest-only').hexdigest(),
        groups=groups)
    _json(root / 'sample-manifest.json', manifest)
    cfg = dict(sample_root=str(root.resolve()), sample_manifest_sha256=p.sha(root / 'sample-manifest.json'),
        model_sha256=SyntheticModel.sha,
        freeze_sha256=hashlib.sha256(b'registered-stratified-synthetic-freeze').hexdigest(),
        previous_receipt_sha256=hashlib.sha256(b'registered-stratified-synthetic-first-receipt').hexdigest(),
        registered_groups=[{k: g[k] for k in p.IDENTITY_FIELDS} for g in groups],
        max_evaluated_groups=0, max_evaluated_candidates=0,
        max_decoded_groups=0, max_decoded_candidates=0)
    return cfg, manifest, rows_by_game


def _caps(cfg, decoded, evaluated=0):
    return dict(cfg, max_decoded_groups=decoded, max_decoded_candidates=2 * decoded,
                max_evaluated_groups=evaluated, max_evaluated_candidates=2 * evaluated)


def _reject(call, expected):
    try:
        call()
    except (ValueError, AssertionError) as error:
        assert any(part in str(error) for part in expected), (type(error).__name__, str(error))
        return type(error).__name__
    raise AssertionError('Expected fail-closed rejection: ' + repr(expected))


def _corrupt_sample(source, target, cfg, manifest, rows, mutate):
    target.mkdir(parents=True, exist_ok=False)
    case_manifest = deepcopy(manifest)
    changed_rows = deepcopy(rows['train_a'])
    mutate(changed_rows)
    for group in case_manifest['groups']:
        path = target / group['path']
        if group['game'] == 'train_a':
            _gzip(path, changed_rows)
            group['sha256'] = p.sha(path)
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source / group['path'], path)
    _json(target / 'sample-manifest.json', case_manifest)
    return dict(cfg, sample_root=str(target.resolve()),
                sample_manifest_sha256=p.sha(target / 'sample-manifest.json'))


def _corrupt_compact(source, target, cfg, mutation):
    # Compact diagnostics contain no numeric candidate arrays and never score.
    shutil.copytree(source, target)
    path = target / 'shards' / 'train_a.jsonl.gz'
    compact = list(p._lines(path, None))
    mutation(compact)
    replacement = path.with_name('replacement.jsonl.gz')
    _gzip(replacement, compact)
    replacement.replace(path)
    receipt_path = path.with_name('train_a.receipt.json')
    receipt = p._read(receipt_path)
    receipt.update(compact_sha256=p.sha(path), compact_bytes=path.stat().st_size)
    _json(receipt_path, receipt)
    changed = deepcopy(cfg)
    changed['shard_receipt_sha256']['train_a'] = p.sha(receipt_path)
    aggregate_path = target / 'aggregate-receipt.json'
    aggregate = p._read(aggregate_path)
    aggregate['shard_receipt_sha256'] = changed['shard_receipt_sha256']
    _json(aggregate_path, aggregate)
    changed['aggregate_receipt_sha256'] = p.sha(aggregate_path)
    return changed


def run_regressions(out, deadline=None):
    """Execute exactly six tiny cases; caller owns lifecycle and single-run gate."""
    out = Path(out); out.mkdir(parents=True, exist_ok=False)
    deadline = min(time.monotonic() + 120., deadline) if deadline is not None else time.monotonic() + 120.
    calls_before, candidates_before = SyntheticModel.total_calls, SyntheticModel.total_candidates
    cfg, manifest, rows = _sample(out / 'sample')
    reference_out, candidate_out = out / 'reference', out / 'candidate'
    reference = p.reference(reference_out, _caps(cfg, 7, 4), deadline, SyntheticModel())
    index = p.build_index(candidate_out, _caps(cfg, 3), deadline, SyntheticModel())
    current = dict(cfg, support_index_sha256=index['support_index_sha256'],
                   support_index_receipt_sha256=index['support_index_receipt_sha256'],
                   previous_receipt_sha256=index['receipt_sha256'])
    receipts, all_counts = {}, [reference['counts'], index['counts']]
    for group in manifest['groups']:
        result = p.score_shard(candidate_out, _caps(current, group['rows'], group['rows']),
                               deadline, group['game'], SyntheticModel())
        receipts[group['game']] = result['receipt_sha256']
        current['previous_receipt_sha256'] = result['receipt_sha256']
        all_counts.append(result['counts'])
    current['shard_receipt_sha256'] = receipts
    current['shard_order'] = [group['game'] for group in manifest['groups']]
    aggregate = p.aggregate(candidate_out, current, deadline)
    all_counts.append(aggregate['counts'])
    current.update(aggregate_receipt_sha256=aggregate['receipt_sha256'],
        reference_receipt_sha256=reference['receipt_sha256'], reference_freeze_sha256=cfg['freeze_sha256'])
    exact = p.compare_reference(candidate_out, reference_out, current, deadline)
    assert exact['exact_compact_rows'] == 4 and exact['exact_candidates'] == 8
    assert list(receipts)[0] == 'validation_a'
    with (candidate_out / 'support-index.pkl').open('rb') as stream:
        support = pickle.load(stream)['index']
    assert support['groups'] == 3 and support['max'][0] == 0.
    assert len(support['source_files_sha256']) == 16 * 3
    compact = list(p._lines(reference_out / 'reference.jsonl.gz', deadline))
    assert [r['source_row_index'] for r in compact] == [7, 11, 9, 4]
    assert compact[0]['diagnostic']['training_label_indices64'] == [0, 1]
    assert not compact[-1]['diagnostic']['train_seen64']
    assert reference['counts']['actual_score_calls'] < sum(
        counts['actual_score_calls'] for counts in all_counts[2:-1])
    cases = [dict(case=CASES[0], passed=True)]

    corruptions = [
        (CASES[1], lambda items: items.__setitem__(1, deepcopy(items[0])),
         ['reordered source row'], ['reordered compact row identity']),
        (CASES[2], lambda items: items.pop(), ['Incomplete source group'],
         ['Comparison shard population mismatch']),
        (CASES[3], lambda items: items.reverse(), ['reordered source row'],
         ['reordered compact row identity']),
    ]
    for name, mutate, source_error, compact_error in corruptions:
        sample_cfg = _corrupt_sample(out / 'sample', out / (name + '-sample'),
                                     cfg, manifest, rows, mutate)
        failed_out = out / (name + '-source-reject')
        _reject(lambda: p.build_index(failed_out, _caps(sample_cfg, 3), deadline, SyntheticModel()),
                source_error)
        all_counts.append(p._read(failed_out / 'progress' / 'build-index.json')['counts'])
        changed = _corrupt_compact(candidate_out, out / (name + '-compact'), current, mutate)
        _reject(lambda: p.compare_reference(out / (name + '-compact'), reference_out, changed, deadline),
                compact_error)
        cases.append(dict(case=name, passed=True))

    name = CASES[4]
    def misalign(items):
        items[0]['inputs']['shared_x'][0] = 123.
        # Preserve the old official input hash: the source decoder must reject.
    sample_cfg = _corrupt_sample(out / 'sample', out / (name + '-sample'), cfg, manifest, rows, misalign)
    failed_out = out / (name + '-source-reject')
    try:
        p.build_index(failed_out, _caps(sample_cfg, 3), deadline, SyntheticModel())
    except AssertionError:
        pass
    else:
        raise AssertionError('Official triple decoder accepted an incorrect input hash')
    all_counts.append(p._read(failed_out / 'progress' / 'build-index.json')['counts'])
    # Wrong sparse receipt identity remains wrong even with freshly bound hashes.
    changed_out = out / (name + '-receipt')
    shutil.copytree(candidate_out, changed_out)
    path = changed_out / 'shards' / 'train_a.receipt.json'
    receipt = p._read(path); receipt['source_row_indices'] = [0, 1]
    _json(path, receipt)
    changed = deepcopy(current); changed['shard_receipt_sha256']['train_a'] = p.sha(path)
    aggregate_path = changed_out / 'aggregate-receipt.json'
    aggregate_value = p._read(aggregate_path)
    aggregate_value['shard_receipt_sha256'] = changed['shard_receipt_sha256']
    _json(aggregate_path, aggregate_value)
    changed['aggregate_receipt_sha256'] = p.sha(aggregate_path)
    _reject(lambda: p.compare_reference(changed_out, reference_out, changed, deadline),
            ['Comparison shard source/index binding mismatch'])
    cases.append(dict(case=name, passed=True))

    name = CASES[5]; failed_out = out / name
    _reject(lambda: p.build_index(failed_out, _caps(cfg, 1), deadline, SyntheticModel()),
            ['Allocated derived decode budget exhausted before group'])
    counts = p._read(failed_out / 'progress' / 'build-index.json')['counts']
    assert counts['decoded_groups'] == counts['decoded_candidates'] == counts['evaluated_groups'] == 0
    all_counts.append(counts); cases.append(dict(case=name, passed=True))

    total = {field: sum(counts[field] for counts in all_counts) for field in
             ('decoded_groups', 'decoded_candidates', 'evaluated_groups', 'evaluated_candidates',
              'actual_score_calls', 'actual_score_candidates')}
    assert total['decoded_groups'] == 19 and total['decoded_candidates'] == 38
    assert total['evaluated_groups'] == 8 and total['evaluated_candidates'] == 16
    assert total['actual_score_calls'] == SyntheticModel.total_calls - calls_before
    assert total['actual_score_candidates'] == SyntheticModel.total_candidates - candidates_before
    # Conservative combined numeric-processing ceiling counts evaluate again,
    # in addition to its already-counted input JSON/decoder consumption.
    assert total['decoded_groups'] + total['evaluated_groups'] <= 40
    assert total['decoded_candidates'] + total['evaluated_candidates'] <= 192
    assert [row['case'] for row in cases] == list(CASES)
    result = dict(schema='task-stratified-pipeline-regression-v1', passed=True,
        synthetic_only=True, cases=cases, counts=total,
        synthetic_numeric_processing_groups=total['decoded_groups'] + total['evaluated_groups'],
        synthetic_numeric_processing_candidates=total['decoded_candidates'] + total['evaluated_candidates'],
        exact_comparison=exact, original_streams_read=0, frozen_model_calls=0,
        training_runs=0, new_games=0, official_transitions=0)
    p.write_new(out / 'regression-result.json', result)
    return result
