"""Registered, tiny synthetic acceptance cases; never load historical data.

Only run_regressions(out) executes work. The caller supplies an owned directory,
registers this fixed matrix first, and cleans that directory after acceptance.
No engine, checkpoint loader, source dataset, or subprocess is used here.
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

from learning.task_bc_error_diagnostics_v1.numeric import _build_and_summarize, new_index
from learning.task_choice_v1.features import OPS, SCHEMA_VERSION, candidate_id, schema_hash
from . import pipeline as p


class SyntheticModel:
    sha = hashlib.sha256(b'capacity-fixed-synthetic-float64-scorer-v1').hexdigest()
    total_calls = 0
    total_candidates = 0

    def __init__(self, nonfinite=False):
        self.metadata = dict(schema_version=SCHEMA_VERSION,
            shared_feature_names=['shared.' + str(i) for i in range(275)],
            feature_names=['task.op.PLANT'] + ['candidate.' + str(i) for i in range(1, 98)],
            ops=list(OPS))
        self.metadata['schema_sha256'] = schema_hash(self.metadata['feature_names'],
                                                    self.metadata['shared_feature_names'])
        self.nonfinite = nonfinite

    def score(self, shared, candidates):
        assert shared.dtype == np.float64 and candidates.dtype == np.float64
        assert shared.shape == (275,) and candidates.ndim == 2 and candidates.shape[1] == 98
        SyntheticModel.total_calls += 1
        SyntheticModel.total_candidates += len(candidates)
        if self.nonfinite:
            return np.full(len(candidates), np.nan)
        return candidates[:, 1] + shared[0]


def _input(model, *, second=1., singleton=False, context=None):
    result = deepcopy(model.metadata)
    result['shared_x'] = [0.] * 275
    result['observation'] = dict(step=0, player=0)
    if context is not None:
        result['observation']['fixture_context'] = context
    result['candidates'] = []
    for target, op, value in (([0, 0], 'PLANT', 1.), ([1, 0], 'WATER', second)):
        local = [0.] * 98; local[1] = value
        result['candidates'].append(dict(id=candidate_id(0, target, op), unit=0,
                                         target=target, op=op, x=local))
    if singleton:
        result['candidates'] = result['candidates'][:1]
    return result


def _fixture_rows(model):
    base = _input(model)
    novel = _input(model); novel['shared_x'][1] = 3.; novel['candidates'][0]['x'][0] = 1.
    unique = _input(model); unique['shared_x'][0] = 2.
    train = [(base, 0), (_input(model, singleton=True), 0), (deepcopy(base), 1),
             (_input(model, context='other'), 1), (_input(model, second=1. + 2. ** -30), 1),
             (unique, 0)]
    validation = [(novel, 1), (deepcopy(base), 1)]
    return train, validation


def _gzip(path, rows, *, newline=True, allow_nan=False):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('wb') as raw:
        with gzip.GzipFile(filename='', mode='wb', fileobj=raw, mtime=0) as stream:
            for i, row in enumerate(rows):
                encoded = json.dumps(row, sort_keys=True, ensure_ascii=False,
                                     separators=(',', ':'), allow_nan=allow_nan).encode('utf-8')
                stream.write(encoded + (b'\n' if newline or i + 1 < len(rows) else b''))


def _json(path, value):
    Path(path).write_bytes(p.canonical(value) + b'\n')


def _read_rows(path):
    return list(p._lines(path, None))


def _make_sample(root, train, validation, *, boundary=2):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=False)
    groups = []
    declarations = [('train', 'train_a', 1, train[:boundary]),
                    ('train', 'train_b', 2, train[boundary:])]
    if validation:
        declarations.append(('validation', 'validation_a', 3, validation))
    for split, game, seed, records in declarations:
        rows = []
        for source_row_index, (inputs, label_index) in enumerate(records):
            label = dict(index=label_index, teacher_label=inputs['candidates'][label_index]['id'])
            meta = dict(game=game, split=split, seed=seed, side=0, opponent='v1',
                        step=0, decision=source_row_index,
                        input_sha256=hashlib.sha256(p.canonical(inputs)).hexdigest())
            rows.append(dict(inputs=deepcopy(inputs), label=label, meta=meta,
                             source_row_index=source_row_index))
        path = 'sample/' + game + '.jsonl.gz'
        _gzip(root / path, rows)
        groups.append(dict(split=split, game=game, seed=seed, side=0, opponent='v1', path=path,
                           rows=len(rows), candidates=sum(len(row['inputs']['candidates']) for row in rows),
                           sha256=p.sha(root / path), source_row_start=0, source_row_stop=len(rows),
                           source_sha256={kind: hashlib.sha256((game + ':' + kind).encode()).hexdigest()
                                          for kind in ('inputs', 'labels', 'index')}))
    manifest = dict(schema=p.SAMPLE_SCHEMA, sample_only=True,
                    source_manifest_sha256=hashlib.sha256(b'synthetic-source-manifest-only').hexdigest(),
                    groups=groups)
    _json(root / 'sample-manifest.json', manifest)
    return _config(root, manifest), manifest


def _config(root, manifest=None):
    return dict(sample_root=str(Path(root).resolve()),
                sample_manifest_sha256=p.sha(Path(root) / 'sample-manifest.json'),
                model_sha256=SyntheticModel.sha,
                freeze_sha256=hashlib.sha256(b'registered-synthetic-freeze').hexdigest(),
                previous_receipt_sha256=hashlib.sha256(b'registered-synthetic-first-receipt').hexdigest(),
                max_evaluated_groups=32, max_evaluated_candidates=64)


def _new_run(out, cfg, manifest, deadline):
    index = p.build_index(out, cfg, deadline, SyntheticModel())
    current = dict(cfg, support_index_sha256=index['support_index_sha256'],
                   support_index_receipt_sha256=index['support_index_receipt_sha256'],
                   previous_receipt_sha256=index['receipt_sha256'])
    receipts = {}
    for group in manifest['groups']:
        result = p.score_shard(out, current, deadline, group['game'], SyntheticModel())
        receipts[group['game']] = result['receipt_sha256']
        current['previous_receipt_sha256'] = result['receipt_sha256']
    current['shard_receipt_sha256'] = receipts
    result = p.aggregate(out, current, deadline)
    current['previous_receipt_sha256'] = result['receipt_sha256']
    current['aggregate_receipt_sha256'] = result['receipt_sha256']
    return current, result


def _reject(call, message):
    expected = {
        'reordered compact rows': 'reordered compact row identity',
        'duplicate compact row': 'reordered compact row identity',
        'truncated final compact line': 'Truncated or blank JSONL row',
        'nonfinite compact value': 'Nonfinite JSON constant',
        'missing shard': 'Missing or extra committed shard files',
        'extra duplicate shard': 'Missing or extra committed shard files',
        'reordered shard manifest': 'reordered shard receipt manifest',
        'wrong index hash': 'Support index hash mismatch before unpickle',
        'wrong shard receipt hash': 'Shard receipt hash mismatch',
        'source chain mismatch': 'Shard source/index binding mismatch',
        'wrong model hash': 'Uncommitted or mismatched index receipt',
        'reordered original rows': 'reordered source row',
        'duplicate original row': 'reordered source row',
        'truncated original line': 'Truncated or blank JSONL row',
        'nonfinite original value': 'Nonfinite JSON constant',
        'cache hits cannot bypass evaluation budget': 'Allocated evaluation budget exhausted before evaluate',
        'nonfinite frozen model logits': 'Invalid frozen logits',
        'sample file hash mismatch': 'Sample file hash mismatch',
        'sample manifest hash mismatch': 'Sample manifest hash mismatch',
        'duplicate source group': 'Invalid or duplicate game',
        'wrong comparison reference receipt hash': 'Comparison reference completion receipt hash mismatch',
        'wrong comparison aggregate receipt hash': 'Comparison aggregate completion receipt hash mismatch',
    }.get(message)
    try:
        call()
    except (ValueError, AssertionError, EOFError, OSError, pickle.UnpicklingError,
            KeyError, TypeError) as error:
        if expected is not None:
            assert expected in str(error), (message, type(error).__name__, str(error))
        elif message == 'uncommitted index':
            assert isinstance(error, FileNotFoundError)
        elif message == 'completed action cannot be retried':
            assert isinstance(error, FileExistsError)
        else:
            raise AssertionError('Rejection fixture lacks an expected cause: ' + message) from error
        return message
    raise AssertionError('Expected fail-closed rejection: ' + message)


def _clone_outputs(source, target):
    source, target = Path(source), Path(target)
    target.mkdir(parents=True, exist_ok=False)
    for name in ('support-index.pkl', 'support-index-receipt.json'):
        shutil.copy2(source / name, target / name)
    shutil.copytree(source / 'shards', target / 'shards')


def _rewrite_shard(folder, game, cfg, mutate, *, newline=True, allow_nan=False):
    path = Path(folder) / 'shards' / (game + '.jsonl.gz')
    rows = _read_rows(path)
    mutate(rows)
    _gzip(path, rows, newline=newline, allow_nan=allow_nan)
    receipt_path = path.with_name(game + '.receipt.json')
    receipt = p._read(receipt_path)
    receipt.update(compact_sha256=p.sha(path), compact_bytes=path.stat().st_size)
    _json(receipt_path, receipt)
    cfg = deepcopy(cfg)
    cfg['shard_receipt_sha256'][game] = p.sha(receipt_path)
    return cfg


def _clone_sample(source, target, mutate, *, newline=True, allow_nan=False):
    shutil.copytree(source, target)
    path = Path(target) / 'sample' / 'train_a.jsonl.gz'
    rows = _read_rows(path)
    mutate(rows)
    _gzip(path, rows, newline=newline, allow_nan=allow_nan)
    manifest_path = Path(target) / 'sample-manifest.json'
    manifest = p._read(manifest_path)
    manifest['groups'][0]['sha256'] = p.sha(path)
    _json(manifest_path, manifest)
    return _config(target)


def run_regressions(out):
    """Run this fixed synthetic matrix only after caller registration.

    Returns aggregate counts/hashes, never fixture inputs or local file paths.
    Failed fixtures are intentional, retained under out for the caller's owned
    cleanup, and never resumed or fed into a succeeding job.
    """
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    deadline = time.monotonic() + 120.
    calls_before = SyntheticModel.total_calls
    candidates_before = SyntheticModel.total_candidates
    model = SyntheticModel()
    train, validation = _fixture_rows(model)
    cfg, manifest = _make_sample(out / 'sample', train, validation)
    reference_out, new_out = out / 'reference', out / 'new'
    reference = p.reference(reference_out, cfg, deadline, SyntheticModel())
    new_cfg, _ = _new_run(new_out, cfg, manifest, deadline)
    new_cfg['reference_receipt_sha256'] = reference['receipt_sha256']
    new_cfg['reference_freeze_sha256'] = cfg['freeze_sha256']
    exact = p.compare_reference(new_out, reference_out, new_cfg, deadline)
    shard_counts = [p._read(new_out / 'shards' / (g['game'] + '.receipt.json'))['counts']
                    for g in manifest['groups']]
    assert sum(c['evaluated_groups'] for c in shard_counts) == reference['counts']['evaluated_groups']
    assert sum(c['actual_score_calls'] for c in shard_counts) > reference['counts']['actual_score_calls']
    assert sum(c['cache_hits'] + c['cache_misses'] for c in shard_counts) == len(train) + len(validation)

    # Independent direct old entrypoint, with no replacement of its analyzer.
    def legacy_reader(split):
        for inputs, label in train if split == 'train' else validation:
            yield inputs, dict(index=label, teacher_label=inputs['candidates'][label]['id'])
    legacy_model = SyntheticModel()
    direct = _build_and_summarize(new_index(legacy_model), legacy_model, legacy_reader)
    assert p.canonical(direct) + b'\n' == (new_out / 'summary.json').read_bytes()
    direct_groups = len(train) + len(validation)
    direct_candidates = sum(len(inputs['candidates']) for inputs, _ in train + validation)

    compact = _read_rows(reference_out / 'reference.jsonl.gz')
    early = compact[0]['diagnostic']
    assert early['training_label_indices64'] == [0, 1]
    assert all(early[k] for k in ('training_label_conflict64',
        'training_same_context_label_conflict64', 'training_different_context_label_conflict64'))
    close = compact[4]['diagnostic']
    assert close['key32'] == early['key32'] and close['key64'] != early['key64']
    assert close['alias32'] and not close['alias64'] and close['pred_index'] == 1
    assert compact[1]['diagnostic']['teacher_margin'] is None
    assert early['pred_index'] == 0 and early['teacher_tied_at_max']
    assert compact[-2]['diagnostic']['range_exceeded']
    assert compact[-2]['diagnostic']['unseen_discrete_features'] == [275]
    with (new_out / 'support-index.pkl').open('rb') as stream:
        index = pickle.load(stream)['index']
    assert index['max'][1] == 0. and index['discrete'][275] == {0.}
    assert index['groups'] == len(train)

    # A different cache/shard boundary preserves diagnostic rows and sums.
    alt_cfg, alt_manifest = _make_sample(out / 'alternate-sample', train, validation, boundary=1)
    alt_out = out / 'alternate-new'
    _, alt_receipt = _new_run(alt_out, alt_cfg, alt_manifest, deadline)
    assert (new_out / 'summary.json').read_bytes() == (alt_out / 'summary.json').read_bytes()
    alternate = [row['diagnostic'] for group in alt_manifest['groups']
                 for row in _read_rows(alt_out / 'shards' / (group['game'] + '.jsonl.gz'))]
    assert p.canonical(alternate) == p.canonical([row['diagnostic'] for row in compact])

    # An empty validation split retains the old null denominators and recalls.
    empty_cfg, empty_manifest = _make_sample(out / 'empty-validation-sample', train, [])
    empty_ref, empty_new = out / 'empty-reference', out / 'empty-new'
    empty_reference = p.reference(empty_ref, empty_cfg, deadline, SyntheticModel())
    empty_new_cfg, _ = _new_run(empty_new, empty_cfg, empty_manifest, deadline)
    empty_new_cfg['reference_receipt_sha256'] = empty_reference['receipt_sha256']
    empty_new_cfg['reference_freeze_sha256'] = empty_cfg['freeze_sha256']
    p.compare_reference(empty_new, empty_ref, empty_new_cfg, deadline)
    empty = p._read(empty_new / 'summary.json')['validation']
    assert empty['raw_weighted']['denominator'] == 0
    assert empty['raw_weighted']['rates']['candidate_mismatch'] is None
    assert empty['raw_weighted']['class_recall']['DIG'] is None
    assert empty['unique_weighted64']['denominator'] == 0

    checks = ['exact old/new compact records and global summary bytes',
              'direct old _build_and_summarize equality',
              'complete late training support and cross-shard label/context conflicts',
              'validation-only extremes and categories excluded from train index',
              'float32 collision retains distinct float64 choices',
              'single-candidate null margin and first-argmax max ties',
              'cache hits count as evaluations and shard cache misses are measured separately',
              'changed shard boundary preserves original diagnostic order and summaries',
              'empty validation retains null rates and class recalls']

    # Receipt rejection performs no evaluation and leaves the fixed scoring budget unchanged.
    for label, field in (
            ('wrong comparison reference receipt hash', 'reference_receipt_sha256'),
            ('wrong comparison aggregate receipt hash', 'aggregate_receipt_sha256')):
        case_cfg = dict(new_cfg, **{field: '0' * 64})
        checks.append(_reject(lambda: p.compare_reference(new_out, reference_out, case_cfg, deadline), label))

    # All corruption attempts get a fresh action claim and cannot publish a summary.
    for label, mutate, newline, nonfinite in (
            ('reordered compact rows', lambda rows: rows.reverse(), True, False),
            ('duplicate compact row', lambda rows: rows.__setitem__(1, deepcopy(rows[0])), True, False),
            ('truncated final compact line', lambda rows: None, False, False),
            ('nonfinite compact value', lambda rows: rows[0]['diagnostic'].__setitem__('logit_gap', float('nan')),
             True, True)):
        case = out / ('reject-' + label.replace(' ', '-'))
        _clone_outputs(new_out, case)
        case_cfg = _rewrite_shard(case, 'train_a', new_cfg, mutate, newline=newline, allow_nan=nonfinite)
        checks.append(_reject(lambda: p.aggregate(case, case_cfg, deadline), label))
        assert not (case / 'summary.json').exists()

    for label in ('missing shard', 'extra duplicate shard', 'reordered shard manifest',
                  'wrong index hash', 'uncommitted index', 'wrong shard receipt hash',
                  'source chain mismatch', 'wrong model hash'):
        case = out / ('reject-' + label.replace(' ', '-'))
        _clone_outputs(new_out, case)
        case_cfg = deepcopy(new_cfg)
        if label == 'missing shard':
            (case / 'shards' / 'train_a.jsonl.gz').unlink()
        elif label == 'extra duplicate shard':
            shutil.copy2(case / 'shards' / 'train_a.jsonl.gz', case / 'shards' / 'duplicate.jsonl.gz')
        elif label == 'reordered shard manifest':
            case_cfg['shard_receipt_sha256'] = dict(reversed(list(case_cfg['shard_receipt_sha256'].items())))
        elif label == 'wrong index hash':
            with (case / 'support-index.pkl').open('ab') as stream:
                stream.write(b'not-a-committed-index')
        elif label == 'uncommitted index':
            (case / 'support-index-receipt.json').unlink()
        elif label == 'wrong shard receipt hash':
            case_cfg['shard_receipt_sha256']['train_a'] = '0' * 64
        elif label == 'source chain mismatch':
            receipt_path = case / 'shards' / 'train_a.receipt.json'
            receipt = p._read(receipt_path)
            receipt['source_sha256']['inputs'] = '0' * 64
            _json(receipt_path, receipt)
            case_cfg['shard_receipt_sha256']['train_a'] = p.sha(receipt_path)
        elif label == 'wrong model hash':
            case_cfg['model_sha256'] = '0' * 64
        checks.append(_reject(lambda: p.aggregate(case, case_cfg, deadline), label))
        assert not (case / 'summary.json').exists()

    for label, mutate, newline, nonfinite in (
            ('reordered original rows', lambda rows: rows.reverse(), True, False),
            ('duplicate original row', lambda rows: rows.__setitem__(1, deepcopy(rows[0])), True, False),
            ('truncated original line', lambda rows: None, False, False),
            ('nonfinite original value', lambda rows: rows[0]['inputs']['shared_x'].__setitem__(0, float('nan')),
             True, True)):
        case_sample = out / ('sample-reject-' + label.replace(' ', '-'))
        case_cfg = _clone_sample(out / 'sample', case_sample, mutate, newline=newline, allow_nan=nonfinite)
        case = out / ('reject-' + label.replace(' ', '-'))
        checks.append(_reject(lambda: p.build_index(case, case_cfg, deadline, SyntheticModel()), label))
        assert not (case / 'support-index-receipt.json').exists()

    for label in ('sample file hash mismatch', 'sample manifest hash mismatch', 'duplicate source group'):
        case_sample = out / ('sample-reject-' + label.replace(' ', '-'))
        shutil.copytree(out / 'sample', case_sample)
        case_cfg = _config(case_sample)
        if label == 'sample file hash mismatch':
            with (case_sample / 'sample' / 'train_a.jsonl.gz').open('ab') as stream:
                stream.write(b'changed')
        elif label == 'sample manifest hash mismatch':
            with (case_sample / 'sample-manifest.json').open('ab') as stream:
                stream.write(b' ')
        else:
            changed = p._read(case_sample / 'sample-manifest.json')
            changed['groups'].append(deepcopy(changed['groups'][0]))
            _json(case_sample / 'sample-manifest.json', changed)
            case_cfg = _config(case_sample)
        case = out / ('reject-' + label.replace(' ', '-'))
        checks.append(_reject(lambda: p.build_index(case, case_cfg, deadline, SyntheticModel()), label))
        assert not (case / 'support-index-receipt.json').exists()

    # Beforecall accounting includes a second evaluation which would hit cache.
    case = out / 'reject-cache-hit-budget'
    case.mkdir()
    for name in ('support-index.pkl', 'support-index-receipt.json'):
        shutil.copy2(new_out / name, case / name)
    limited = dict(new_cfg, max_evaluated_groups=1, max_evaluated_candidates=2)
    checks.append(_reject(lambda: p.score_shard(case, limited, deadline, 'train_b', SyntheticModel()),
                          'cache hits cannot bypass evaluation budget'))
    progress = p._read(case / 'progress' / 'score-train_b.json')
    assert progress['counts']['evaluated_groups'] == 1
    assert progress['counts']['actual_score_calls'] == 1
    assert not (case / 'shards' / 'train_b.receipt.json').exists()

    case = out / 'reject-nonfinite-model'
    case.mkdir()
    for name in ('support-index.pkl', 'support-index-receipt.json'):
        shutil.copy2(new_out / name, case / name)
    checks.append(_reject(lambda: p.score_shard(case, new_cfg, deadline, 'train_a', SyntheticModel(True)),
                          'nonfinite frozen model logits'))
    assert not (case / 'shards' / 'train_a.receipt.json').exists()

    checks.append(_reject(lambda: p.build_index(new_out, cfg, deadline, SyntheticModel()),
                          'completed action cannot be retried'))
    # No fixture progress is copied, so each file is exactly one attempted action.
    stages = [p._read(path) for path in out.rglob('progress/*.json')]
    evaluated_groups = direct_groups + sum(s['counts']['evaluated_groups'] for s in stages)
    evaluated_candidates = direct_candidates + sum(s['counts']['evaluated_candidates'] for s in stages)
    assert evaluated_groups <= 16384 and evaluated_candidates <= 1048576
    result = dict(schema=p.SCHEMA, passed=True, public_safe=True, sample_only=True,
        fixture='fixed tiny synthetic scorer and registered corruption matrix',
        original_data_rows=0, real_model_calls=0,
        synthetic_evaluated_groups=evaluated_groups,
        synthetic_evaluated_candidates=evaluated_candidates,
        synthetic_actual_score_calls=SyntheticModel.total_calls - calls_before,
        synthetic_actual_score_candidates=SyntheticModel.total_candidates - candidates_before,
        exact_reference_rows=exact['exact_compact_rows'],
        summary_sha256=reference['summary_sha256'],
        alternate_summary_sha256=alt_receipt['summary_sha256'],
        checks=checks, training_runs=0, new_games=0, official_transitions=0)
    p.write_new(out / 'synthetic-verification.json', result)
    return result
