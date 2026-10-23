"""Offline conditional scoring and train-only support; never run the game engine.

Numeric keys preserve candidate count and order and normalize signed zero. The
float32 key describes training-cache support, NOT deployed indistinguishability.
Only original float64 inputs are scored. Indexes and individual compact records
are local artifacts; only aggregate summaries are suitable for publication.
"""
from collections import Counter, OrderedDict
from contextlib import ExitStack
import gzip
import hashlib
from itertools import zip_longest
import json
from pathlib import Path
import time

import numpy as np

from learning.task_bc_v1.model import continuous_mask
from learning.task_choice_v1.features import OPS, candidate_id, schema_hash

SCHEMA = 'task-error-numeric-v1'
FLAGS = ('candidate_mismatch', 'op_mismatch', 'unit_mismatch', 'target_mismatch',
         'train_seen32', 'train_seen64', 'candidate_semantics_seen', 'alias32',
         'alias64', 'teacher_later_alias32', 'teacher_later_alias64',
         'float32_only_alias', 'cross_op', 'multicandidate', 'max_tied',
         'teacher_tied_at_max', 'candidate_count_out_of_range',
         'candidate_count_unseen', 'range_exceeded', 'discrete_unseen')


def _require(ok, message):
    if not ok:
        raise ValueError(message)


def _digest(value):
    raw = json.dumps(value, sort_keys=True, separators=(',', ':'),
                     ensure_ascii=False, allow_nan=False).encode('utf-8')
    return hashlib.sha256(raw).hexdigest()


def _sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def _deadline(deadline):
    """deadline is an absolute time.monotonic() value, or None in tests."""
    if deadline is not None and time.monotonic() >= deadline:
        raise TimeoutError('Numeric diagnostics deadline exceeded; no retry')


def numeric_key(shared, candidates, precision):
    """Full ordered bytes, equivalent to NumericEquivalence for float32."""
    _require(precision in (32, 64), 'Unknown numeric precision')
    with np.errstate(over='ignore', invalid='ignore'):
        shared = np.asarray(shared, dtype='<f' + str(precision // 8)).copy()
        candidates = np.asarray(candidates, dtype='<f' + str(precision // 8)).copy()
    _require(shared.ndim == 1 and candidates.ndim == 2 and len(candidates) > 0,
             'Invalid numeric key shape')
    _require(np.isfinite(shared).all() and np.isfinite(candidates).all(),
             'Nonfinite numeric input or float32 overflow')
    shared[shared == 0] = 0.; candidates[candidates == 0] = 0.
    h = hashlib.sha256(np.asarray([len(candidates)], dtype='<i8').tobytes())
    h.update(shared.tobytes()); h.update(candidates.tobytes())
    return h.hexdigest()


def _aliases(candidates, precision):
    values = np.asarray(candidates, dtype='<f' + str(precision // 8)).copy()
    values[values == 0] = 0.
    first = {}; aliases = []
    for i, row in enumerate(values):
        aliases.append(first.setdefault(row.tobytes(), i))
    return aliases


def _parts(inputs, model):
    meta = model.metadata
    for field in ('schema_version', 'schema_sha256', 'shared_feature_names', 'feature_names'):
        _require(inputs[field] == meta[field], 'Frozen input schema mismatch: ' + field)
    _require(inputs['schema_sha256'] == schema_hash(inputs['feature_names'], inputs['shared_feature_names']),
             'Invalid feature schema digest')
    shared = np.asarray(inputs['shared_x'], dtype='<f8')
    candidates = inputs['candidates']
    local = np.asarray([c['x'] for c in candidates], dtype='<f8')
    _require(shared.shape == (275,) and local.shape == (len(candidates), 98) and len(candidates) > 0,
             'Invalid original task input shape')
    _require(np.isfinite(shared).all() and np.isfinite(local).all(), 'Nonfinite original input')
    ids = [c['id'] for c in candidates]
    _require(all(isinstance(i, str) for i in ids) and len(ids) == len(set(ids)), 'Invalid candidate IDs')
    for c in candidates:
        _require(type(c['unit']) is int and 0 <= c['unit'] < 32768 and
                 len(c['target']) == 2 and all(type(t) is int and 0 <= t < 32768 for t in c['target']) and
                 c['op'] in OPS and c['id'] == candidate_id(c['unit'], c['target'], c['op']),
                 'Invalid semantic candidate identity')
    ordering = [(c['unit'], c['target'][1], c['target'][0], c['op']) for c in candidates]
    _require(ordering == sorted(ordering), 'Noncanonical candidate order')
    semantic = _digest([[c['id'], c['op'], c['unit'], c['target']] for c in candidates])
    return shared, local, ids, numeric_key(shared, local, 32), numeric_key(shared, local, 64), semantic


def new_index(model):
    """Empty train reference, exposed for small offline regression fixtures."""
    names = model.metadata['shared_feature_names'] + model.metadata['feature_names']
    return dict(schema=SCHEMA, checkpoint_sha256=model.sha, metadata=model.metadata,
                keys32={}, keys64={}, semantics=set(), candidate_counts=Counter(),
                min=np.full(373, np.inf), max=np.full(373, -np.inf),
                discrete={int(i): set() for i in np.flatnonzero(~continuous_mask(names))},
                groups=0, candidates=0, source_files_sha256={})


def add_training(index, inputs, label_index, model):
    """Add one original training call, without retaining candidate matrices."""
    shared, local, ids, key32, key64, semantic = _parts(inputs, model)
    _require(type(label_index) is int and 0 <= label_index < len(ids), 'Invalid label index')
    context = _digest(inputs)
    for field, key in (('keys32', key32), ('keys64', key64)):
        group = index[field].setdefault(key, dict(labels=Counter(), semantics=set(), contexts={}, size=len(ids)))
        group['labels'][label_index] += 1
        group['semantics'].add(semantic)
        group['contexts'].setdefault(context, Counter())[label_index] += 1
    index['semantics'].add(semantic)
    low = np.concatenate((shared, local.min(axis=0)))
    high = np.concatenate((shared, local.max(axis=0)))
    np.minimum(index['min'], low, out=index['min'])
    np.maximum(index['max'], high, out=index['max'])
    for i, values in index['discrete'].items():
        values.update([float(shared[i])] if i < 275 else np.unique(local[:, i - 275]).tolist())
    index['candidate_counts'][len(ids)] += 1
    index['groups'] += 1; index['candidates'] += len(ids)


class RecordAnalyzer:
    """Return compact local evidence. No raw inputs or priorities are returned.

    Factual origins: trained asserts model choice == chosen; teacher_task asserts
    chosen == teacher_label. historical_train/validation are teacher records.
    Bounded score cache is keyed ONLY by full float64 numeric input.
    """
    def __init__(self, index, model):
        _require(index['schema'] == SCHEMA and index['checkpoint_sha256'] == model.sha and
                 index['metadata'] == model.metadata, 'Support/model identity mismatch')
        self.index, self.model, self.cache = index, model, OrderedDict()

    def evaluate(self, record, origin):
        _require(origin in ('trained', 'teacher_task', 'historical_train', 'historical_validation'),
                 'Unknown record origin')
        inputs = record['inputs']
        shared, local, ids, key32, key64, semantic = _parts(inputs, self.model)
        _require(record['teacher_label'] in ids and record['chosen'] in ids, 'Unknown factual/teacher ID')
        label = ids.index(record['teacher_label'])
        if key64 in self.cache:
            logits = self.cache.pop(key64)
        else:
            logits = np.asarray(self.model.score(shared, local), dtype=np.float64)
            _require(logits.shape == (len(ids),) and np.isfinite(logits).all(), 'Invalid frozen logits')
        self.cache[key64] = logits
        if len(self.cache) > 256:
            self.cache.popitem(last=False)
        pred = int(np.argmax(logits))
        if origin == 'trained':
            _require(ids[pred] == record['chosen'], 'Factual trained/model choice mismatch')
        else:
            _require(record['chosen'] == record['teacher_label'], 'Factual teacher/label mismatch')
        teacher, predicted = inputs['candidates'][label], inputs['candidates'][pred]
        aliases32, aliases64 = _aliases(local, 32), _aliases(local, 64)
        index = self.index
        _require(index['groups'] > 0, 'Empty training support')
        low = np.concatenate((shared, local.min(axis=0)))
        high = np.concatenate((shared, local.max(axis=0)))
        outside = np.flatnonzero((low < index['min']) | (high > index['max'])).tolist()
        discrete = [i for i, seen in index['discrete'].items()
                    if any(float(v) not in seen for v in
                           ([shared[i]] if i < 275 else np.unique(local[:, i - 275])))]
        n = len(ids); counts = index['candidate_counts']
        context_key = _digest(inputs)
        known = index['keys64'].get(key64)
        training_labels = set(known['labels']) if known else set()
        same_context_labels = set(known['contexts'].get(context_key, {})) if known else set()
        result = dict(schema=SCHEMA, origin=origin, key32=key32, key64=key64,
            semantic_key=semantic, context_key=context_key, teacher_index=label, pred_index=pred,
            candidate_count=n, candidate_mismatch=ids[label] != ids[pred],
            op_mismatch=teacher['op'] != predicted['op'],
            unit_mismatch=teacher['unit'] != predicted['unit'],
            target_mismatch=teacher['target'] != predicted['target'],
            train_seen32=key32 in index['keys32'], train_seen64=key64 in index['keys64'],
            candidate_semantics_seen=semantic in index['semantics'],
            training_label_conflict64=len(training_labels) > 1,
            label_disagrees_with_training64=bool(training_labels - {label}),
            training_same_context_label_conflict64=bool(same_context_labels - {label}),
            training_different_context_label_conflict64=bool(known and any(
                set(labels) - {label} for key, labels in known['contexts'].items() if key != context_key)),
            training_label_indices64=sorted(training_labels),
            alias32=len(set(aliases32)) < n, alias64=len(set(aliases64)) < n,
            teacher_later_alias32=aliases32[label] != label,
            teacher_later_alias64=aliases64[label] != label,
            float32_only_alias=any(a != b for a, b in zip(aliases32, aliases64)),
            logit_gap=float(logits[pred] - logits[label]),
            teacher_margin=(float(logits[label] - np.max(np.delete(logits, label))) if n > 1 else None),
            max_tie_count=int(np.count_nonzero(logits == logits[pred])),
            teacher_tied_at_max=bool(logits[label] == logits[pred] and np.count_nonzero(logits == logits[pred]) > 1),
            max_tied=bool(np.count_nonzero(logits == logits[pred]) > 1),
            out_of_range_features=outside, unseen_discrete_features=discrete,
            range_exceeded=bool(outside), discrete_unseen=bool(discrete),
            candidate_count_out_of_range=n < min(counts) or n > max(counts),
            candidate_count_unseen=n not in counts,
            multicandidate=n > 1, cross_op=len({c['op'] for c in inputs['candidates']}) > 1)
        for prefix, selected in (('teacher', teacher), ('pred', predicted)):
            for field in ('id', 'op', 'unit', 'target'):
                result[prefix + '_' + field] = selected[field]
        return result


class SummaryAccumulator:
    """Streaming aggregation; unique rates average each group's call-level rate.

    Conflicting labels and differing semantic IDs within a numeric group are
    retained. No representative call is selected or silently dropped.
    """
    def __init__(self):
        self.raw = self._empty()
        self.unique = {'key32': {}, 'key64': {}}
        self.ranges, self.discrete, self.sizes = Counter(), Counter(), Counter()
        self.slices = {}

    @staticmethod
    def _empty():
        return dict(n=0, flags=Counter(), confusion=Counter(), gap_sum=0., candidates=0)

    @staticmethod
    def _add(target, record):
        target['n'] += 1
        target['flags'].update({k: int(record[k]) for k in FLAGS})
        target['confusion'][(record['teacher_op'], record['pred_op'])] += 1
        target['gap_sum'] += record['logit_gap']
        target['candidates'] += record['candidate_count']

    def add(self, record):
        self._add(self.raw, record)
        for field, groups in self.unique.items():
            group = groups.setdefault(record[field], dict(stats=self._empty(), labels=Counter(),
                                                         semantics=set(), contexts={}))
            self._add(group['stats'], record)
            group['labels'][record['teacher_index']] += 1
            group['semantics'].add(record['semantic_key'])
            group['contexts'].setdefault(record['context_key'], set()).add(record['teacher_index'])
        self.ranges.update(record['out_of_range_features'])
        self.discrete.update(record['unseen_discrete_features'])
        self.sizes[record['candidate_count']] += 1
        for field in ('multicandidate', 'cross_op', 'train_seen32', 'train_seen64'):
            key = field + '=' + str(bool(record[field])).lower()
            self._add(self.slices.setdefault(key, self._empty()), record)

    @staticmethod
    def _report(stats):
        n = stats['n']; confusion = stats['confusion']
        return dict(denominator=n, counts=dict(stats['flags']),
            rates={k: stats['flags'].get(k, 0) / n if n else None for k in FLAGS},
            total_candidates=stats['candidates'],
            mean_logit_gap=stats['gap_sum'] / n if n else None,
            confusion={a: {b: confusion.get((a, b), 0) for b in OPS} for a in OPS},
            class_recall={a: (confusion.get((a, a), 0) / sum(confusion.get((a, b), 0) for b in OPS)
                             if sum(confusion.get((a, b), 0) for b in OPS) else None) for a in OPS})

    def result(self):
        result = dict(schema=SCHEMA, raw_weighted=self._report(self.raw),
                      range_exceedance_calls_by_feature=dict(sorted(self.ranges.items())),
                      unseen_discrete_calls_by_feature=dict(sorted(self.discrete.items())),
                      candidate_count_calls=dict(sorted(self.sizes.items())),
                      slices={k: self._report(v) for k, v in self.slices.items()})
        for field, groups in self.unique.items():
            total = self._empty(); total['n'] = len(groups)
            for group in groups.values():
                s = group['stats']; n = s['n']
                total['flags'].update({k: v / n for k, v in s['flags'].items()})
                total['confusion'].update({k: v / n for k, v in s['confusion'].items()})
                total['gap_sum'] += s['gap_sum'] / n
                total['candidates'] += s['candidates'] / n
            report = self._report(total)
            report['weighting'] = 'One unit per numeric group; average all original calls inside each group, preserving conflicting labels.'
            report['counts_unit'] = 'Fractional group equivalents; not original-call counts'
            report['duplicate_groups'] = sum(g['stats']['n'] > 1 for g in groups.values())
            report['calls_in_duplicate_groups'] = sum(g['stats']['n'] for g in groups.values() if g['stats']['n'] > 1)
            report['conflicting_label_groups'] = sum(len(g['labels']) > 1 for g in groups.values())
            report['calls_in_conflicting_label_groups'] = sum(g['stats']['n'] for g in groups.values() if len(g['labels']) > 1)
            report['multiple_semantic_groups'] = sum(len(g['semantics']) > 1 for g in groups.values())
            report['same_structured_context_conflict_groups'] = sum(any(len(v) > 1 for v in g['contexts'].values()) for g in groups.values())
            report['different_structured_context_conflict_groups'] = sum(len(g['labels']) > 1 and len(g['contexts']) > 1 for g in groups.values())
            result['unique_weighted' + field[-2:]] = report
        result['interpretation'] = 'Logit gaps are classification diagnostics, not reward regret. Exact novelty alone does not establish harmful distribution shift.'
        return result


def summarize(records):
    accumulator = SummaryAccumulator()
    for record in records:
        accumulator.add(record)
    return accumulator.result()


def _build_and_summarize(index, model, records_for_split):
    """Fit complete train support, then stream each split against that reference.

    The reader is reopened twice for train and once for validation. No original
    input matrices are retained between rows or passes.
    """
    for inputs, label in records_for_split('train'):
        add_training(index, inputs, label['index'], model)
    analyzer = RecordAnalyzer(index, model)
    summaries = {}
    for split in ('train', 'validation'):
        acc = SummaryAccumulator()
        for inputs, label in records_for_split(split):
            record = dict(inputs=inputs, teacher_label=label['teacher_label'], chosen=label['teacher_label'])
            acc.add(analyzer.evaluate(record, 'historical_' + split))
        summaries[split] = acc.result()
    return summaries


def build_support(source_root, model, deadline):
    """Complete train-only support before scoring train and validation.

    Returns (pickle-safe local index, public-safe summary). Hash-verifies all
    180 compressed files and validates original line alignment via cache.decode.
    Decompresses train twice (fit, score) and validation once (score only).
    Never reads a holdout replay, fits statistics on validation, or changes files.
    """
    from learning.task_bc_v1.cache import decode
    source = Path(source_root).resolve()
    read = lambda name: json.loads((source / name).read_text(encoding='utf-8'))
    manifest = read('dataset-manifest.json'); registration = read('registration.json')
    _require(_sha(source / 'registration.json') == manifest['registration_sha256'], 'Registration hash mismatch')
    _require(_sha(source / 'reconstruction-ledger.json') == manifest['reconstruction_ledger_sha256'], 'Ledger hash mismatch')
    _require(_sha(source / 'coverage.json') == manifest['coverage_sha256'], 'Coverage hash mismatch')
    coverage = read('coverage.json'); ledger = read('reconstruction-ledger.json')['rows']
    rows = {r['game']: r for r in ledger}
    _require(coverage['passed'] and manifest['examples'] == 135406 and
             len(rows) == len(ledger) == len(manifest['files']) == 60 and
             len({f['game'] for f in manifest['files']}) == 60 and
             set(rows) == {f['game'] for f in manifest['files']}, 'Original population mismatch')
    train, validation = registration['train_seeds'], registration['validation_seeds']
    _require(len(set(train)) == len(train) == 8 and len(set(validation)) == len(validation) == 2 and
             not set(train) & set(validation), 'Original 8/2 split mismatch')
    expected = {(seed, op, side) for seed in train + validation for op in ('v1', 'market', 'round2') for side in (0, 1)}
    _require({(r['seed'], r['opponent'], r['side']) for r in ledger} == expected, 'Original game grid mismatch')
    index = new_index(model)
    sources = {'train': [], 'validation': []}
    for file_group in manifest['files']:
        _deadline(deadline)
        split = file_group['split']; game = file_group['game']; identity = rows[game]
        _require(split in sources and identity['split'] == split and
                 identity['seed'] in registration[split + '_seeds'], 'Split reassignment')
        paths = {}
        for kind in ('inputs', 'labels', 'index'):
            info = file_group['files'][kind]
            path = (source / info['path']).resolve()
            _require(path.is_relative_to(source / 'data') and info == identity['files'][kind] and
                     info['rows'] == file_group['files']['inputs']['rows'], 'Unaligned source declaration')
            _require(info['path'] not in index['source_files_sha256'], 'Duplicate source path')
            actual_sha = _sha(path)
            _require(actual_sha == info['sha256'], 'Original source hash mismatch')
            index['source_files_sha256'][info['path']] = actual_sha; paths[kind] = path
        sources[split].append((file_group, identity, paths))
    schema = None

    def records_for_split(split):
        nonlocal schema
        split_groups = split_candidates = 0
        for file_group, identity, paths in sources[split]:
            _deadline(deadline)
            game = file_group['game']
            count = 0; previous = None
            with ExitStack() as stack:
                streams = [stack.enter_context(gzip.open(paths[k], 'rt', encoding='utf-8')) for k in ('inputs', 'labels', 'index')]
                for lines in zip_longest(*streams):
                    _deadline(deadline)
                    _require(all(line is not None for line in lines), 'Unaligned source lengths')
                    inputs, label, meta = [json.loads(line) for line in lines]
                    current = (meta['step'], meta['decision'])
                    _require(previous is None or current > previous, 'Nonmonotone decision order')
                    _require(current[1] == (0 if previous is None or previous[0] != current[0] else previous[1] + 1), 'Missing decision')
                    previous = current
                    schema, *_ = decode(inputs, label, meta, game=game, split=split,
                        expected_seed=identity['seed'], expected_side=identity['side'],
                        expected_opponent=identity['opponent'], schema=schema)
                    count += 1; split_groups += 1; split_candidates += len(inputs['candidates'])
                    _require(split_groups <= coverage['splits'][split]['decisions'] and
                             split_candidates <= 6900116,
                             'Historical population cap exceeded')
                    yield inputs, label
            _require(count == file_group['files']['inputs']['rows'], 'Source row count mismatch')
        _require(split_groups == coverage['splits'][split]['decisions'] and
                 split_candidates == sum(v['candidate_count'] for v in coverage['splits'][split]['tasks'].values()),
                 'Split coverage mismatch')
    summaries = _build_and_summarize(index, model, records_for_split)
    total_groups = sum(s['raw_weighted']['denominator'] for s in summaries.values())
    total_candidates = sum(s['raw_weighted']['total_candidates'] for s in summaries.values())
    _require(total_groups == 135406 and total_candidates == 6900116 and
             len(index['source_files_sha256']) == 180 and schema['schema_sha256'] == manifest['schema_sha256'],
             'Historical final coverage mismatch')
    _deadline(deadline)
    summary = dict(schema=SCHEMA, passed=True, source_manifest_sha256=_sha(source / 'dataset-manifest.json'),
        checkpoint_sha256=model.sha, source_files_sha256=index['source_files_sha256'],
        groups=total_groups, candidates=total_candidates, source_file_count=180,
        train_seed_count=8, validation_seed_count=2, splits=summaries, finite=True,
        train_feature_min=index['min'].tolist(), train_feature_max=index['max'].tolist(),
        train_discrete_values={str(k): sorted(v) for k, v in index['discrete'].items()},
        train_candidate_count_frequencies=dict(sorted(index['candidate_counts'].items())),
        scoring_precision='Original float64 inputs; no float32 reconstruction',
        support_precision='Separate float32 training-cache and float64 deployed-input keys',
        training_runs=0, new_games=0, official_transitions=0)
    return index, summary
