"""Bounded, sample-only support construction, scoring, and ordered aggregation.

The caller registers and prepares the sample, loads the frozen model, allocates
the shared evaluation budget, and owns every worker.  This module never reads
the original dataset or a checkpoint, launches a worker, resumes, or retries.
Old numeric functions remain the sole source of scoring/support semantics.
"""
from contextlib import contextmanager
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import pickle
import re
import time

import numpy as np

from learning.task_bc_error_diagnostics_v1.numeric import (
    FLAGS, SCHEMA as NUMERIC_SCHEMA, RecordAnalyzer, SummaryAccumulator,
    add_training, new_index,
)
from learning.task_bc_v1.cache import decode
from learning.task_choice_v1.features import candidate_id


SCHEMA = 'task-diagnostics-capacity-v1'
SAMPLE_SCHEMA = 'task-capacity-sample-v1'
SPLITS = ('train', 'validation')
HEX = re.compile(r'[0-9a-f]{64}\Z')
TOKEN = re.compile(r'[A-Za-z0-9_-]+\Z')
LIMITS = dict(train=1024, validation=256, groups=1280, candidates=80000)


def require(ok, message):
    if not ok:
        raise ValueError(message)


def check_deadline(deadline):
    if deadline is not None and time.monotonic() >= deadline:
        raise TimeoutError('Capacity pipeline deadline exceeded; no retry')


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False,
                      separators=(',', ':'), allow_nan=False).encode('utf-8')


def _finite_float(text):
    value = float(text)
    require(math.isfinite(value), 'Nonfinite JSON number')
    return value


def strict_json(raw):
    def invalid(_):
        raise ValueError('Nonfinite JSON constant')
    return json.loads(raw, parse_constant=invalid, parse_float=_finite_float)


def sha(path, deadline=None):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            check_deadline(deadline)
            digest.update(chunk)
    check_deadline(deadline)
    return digest.hexdigest()


def _hash(value, field):
    require(isinstance(value, str) and HEX.fullmatch(value) is not None,
            'Missing or invalid SHA256: ' + field)
    return value


def _read(path):
    return strict_json(Path(path).read_bytes())


@contextmanager
def _new_file(path, *, replace=False):
    """Fsync first; hard-link commits without overwriting an existing result.

    A failed partial is retained as local failure evidence and is never read by
    consumers.  The exclusive stage claim prohibits using it for a retry.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + '.partial')
    with partial.open('xb') as stream:
        yield stream
        stream.flush()
        os.fsync(stream.fileno())
    if replace:
        os.replace(partial, path)
    else:
        os.link(partial, path)
        partial.unlink()


def write_new(path, value):
    with _new_file(path) as stream:
        stream.write(canonical(value) + b'\n')


def _write_progress(path, value):
    with _new_file(path, replace=True) as stream:
        stream.write(canonical(value) + b'\n')


def _emit(stream, value):
    stream.write(canonical(value) + b'\n')


@contextmanager
def _gzip_output(path):
    with _new_file(path) as raw:
        with gzip.GzipFile(filename='', mode='wb', fileobj=raw, mtime=0) as stream:
            yield stream


def _lines(path, deadline):
    with gzip.open(path, 'rb') as stream:
        for raw in stream:
            check_deadline(deadline)
            require(raw.endswith(b'\n') and raw.strip(),
                    'Truncated or blank JSONL row')
            value = strict_json(raw)
            require(isinstance(value, dict), 'JSONL row is not an object')
            yield value
    check_deadline(deadline)


def _bind(cfg, manifest):
    return dict(
        schema=SCHEMA, sample_only=True,
        source_manifest_sha256=manifest['source_manifest_sha256'],
        sample_manifest_sha256=_hash(cfg['sample_manifest_sha256'], 'sample_manifest'),
        model_sha256=_hash(cfg['model_sha256'], 'model'),
        freeze_sha256=_hash(cfg['freeze_sha256'], 'freeze'),
    )


class Stage:
    def __init__(self, out, cfg, key, deadline):
        require(TOKEN.fullmatch(key) is not None, 'Invalid stage identity')
        check_deadline(deadline)
        self.out, self.cfg, self.key = Path(out), cfg, key
        self.deadline, self.started = deadline, time.monotonic()
        self.counts = dict(evaluated_groups=0, evaluated_candidates=0,
                           completed_groups=0, completed_candidates=0,
                           actual_score_calls=0, actual_score_candidates=0,
                           cache_hits=0, cache_misses=0, decoded_groups=0,
                           decoded_candidates=0, index_loads=0)
        self.wall = dict(sample_hash=0., decode_validate=0., index_insert=0.,
                         index_commit=0., index_load=0., evaluate=0.,
                         model_score=0., compact_write=0., aggregate=0.)
        self.previous = _hash(cfg['previous_receipt_sha256'], 'previous_receipt')
        self.last_progress_groups = 0
        self.progress = self.out / 'progress' / (key + '.json')
        claim = self.out / 'stages' / (key + '.started.json')
        write_new(claim, dict(schema=SCHEMA, sample_only=True, stage=key,
                  freeze_sha256=_hash(cfg['freeze_sha256'], 'freeze'),
                  previous_receipt_sha256=self.previous,
                  progress_is_not_resume_permission=True))

    def snapshot(self, state, **extra):
        groups = max(self.counts['completed_groups'], self.counts['evaluated_groups'],
                     extra.get('indexed_groups', 0))
        if state not in ('started', 'completed', 'failed', 'group-completed') and \
                groups < self.last_progress_groups + 32:
            return
        _write_progress(self.progress, dict(schema=SCHEMA, sample_only=True,
            public_safe=False, stage=self.key, state=state,
            previous_receipt_sha256=self.previous, counts=self.counts,
            wall_seconds=self.wall, elapsed_seconds=time.monotonic() - self.started,
            progress_is_not_resume_permission=True, counters_are_progress_checkpoint=True, **extra))
        self.last_progress_groups = groups

    def receipt(self, binding, **extra):
        check_deadline(self.deadline)
        return dict(**binding, passed=True, stage=self.key,
                    previous_receipt_sha256=self.previous,
                    counts=dict(self.counts), wall_seconds=dict(self.wall),
                    elapsed_seconds=time.monotonic() - self.started,
                    training_runs=0, new_games=0, official_transitions=0,
                    policy_calls=0, **extra)

    def __enter__(self):
        self.snapshot('started')
        return self

    def __exit__(self, kind, error, traceback):
        if kind is not None:
            self.snapshot('failed', error_type=kind.__name__, error=str(error))
        return False


def _manifest(cfg, stage):
    require(__debug__, 'Pipeline requires Python assertions enabled for source decode')
    root = Path(cfg['sample_root'])
    require(root.is_absolute(), 'Sample root must be absolute')
    root = root.resolve()
    path = root / 'sample-manifest.json'
    started = time.monotonic()
    require(sha(path, stage.deadline) == _hash(cfg['sample_manifest_sha256'], 'sample_manifest'),
            'Sample manifest hash mismatch')
    manifest = _read(path)
    stage.wall['sample_hash'] += time.monotonic() - started
    require(manifest.get('schema') == SAMPLE_SCHEMA and manifest.get('sample_only') is True,
            'Only a registered sample-only manifest is accepted')
    _hash(manifest.get('source_manifest_sha256'), 'source_manifest')
    groups = manifest.get('groups')
    require(isinstance(groups, list), 'Missing sample groups')
    names, paths, seeds = set(), set(), set()
    counts = dict(train=0, validation=0, groups=0, candidates=0)
    for group in groups:
        require(isinstance(group, dict), 'Invalid sample group')
        game, split = group['game'], group['split']
        require(isinstance(game, str) and TOKEN.fullmatch(game) is not None and game not in names,
                'Invalid or duplicate game')
        require(split in SPLITS, 'Unknown original split')
        require(type(group['seed']) is int and type(group['side']) is int and group['side'] == 0
                and group['opponent'] == 'v1' and group['seed'] not in seeds,
                'Sample identity violates registered selection')
        require(type(group['rows']) is int and 0 < group['rows'] <= 128 and
                type(group['candidates']) is int and group['candidates'] >= group['rows'],
                'Invalid complete sample counts')
        require(group['source_row_start'] == 0 and type(group['source_row_stop']) is int
                and group['source_row_stop'] == group['rows'], 'Sample is not the original prefix')
        require(group['path'] == 'sample/' + game + '.jsonl.gz' and group['path'] not in paths,
                'Noncanonical or duplicate sample path')
        resolved = (root / group['path']).resolve()
        require(resolved.is_relative_to(root / 'sample'), 'Sample path escapes its directory')
        _hash(group['sha256'], 'sample_group')
        require(set(group['source_sha256']) == {'inputs', 'labels', 'index'},
                'Incomplete original source hash chain')
        for kind, digest in group['source_sha256'].items():
            _hash(digest, 'source_' + kind)
        names.add(game); paths.add(group['path']); seeds.add(group['seed'])
        counts[split] += group['rows']; counts['groups'] += group['rows']
        counts['candidates'] += group['candidates']
    require(all(counts[k] <= LIMITS[k] for k in LIMITS), 'Registered sample size limit exceeded')
    require(len(groups) <= 10, 'Registered source group count exceeded')
    return root, manifest, _bind(cfg, manifest)


def _check_model(model, cfg):
    require(model.sha == cfg['model_sha256'] and isinstance(model.metadata, dict),
            'Frozen model identity mismatch')


def _sample_rows(root, group, stage):
    started = time.monotonic()
    path = root / group['path']
    require(sha(path, stage.deadline) == group['sha256'], 'Sample file hash mismatch')
    stage.wall['sample_hash'] += time.monotonic() - started
    count = candidates = 0
    previous = None
    schema = None
    iterator = iter(_lines(path, stage.deadline))
    while True:
        started = time.monotonic()
        try:
            row = next(iterator)
        except StopIteration:
            stage.wall['decode_validate'] += time.monotonic() - started
            break
        require(set(row) == {'inputs', 'label', 'meta', 'source_row_index'},
                'Unexpected sample row fields')
        require(type(row['source_row_index']) is int and row['source_row_index'] == count,
                'Missing, duplicate, or reordered source row')
        require(count < group['rows'], 'Extra source row')
        inputs, label, meta = row['inputs'], row['label'], row['meta']
        current = (meta['step'], meta['decision'])
        require(previous is None or current > previous, 'Nonmonotone original decision')
        require(current[1] == (0 if previous is None or previous[0] != current[0] else previous[1] + 1),
                'Missing original decision')
        schema, *_ = decode(inputs, label, meta, game=group['game'], split=group['split'],
            expected_seed=group['seed'], expected_side=group['side'],
            expected_opponent=group['opponent'], schema=schema)
        previous = current
        count += 1; candidates += len(inputs['candidates'])
        require(candidates <= group['candidates'], 'Source candidate count exceeded')
        stage.counts['decoded_groups'] += 1
        stage.counts['decoded_candidates'] += len(inputs['candidates'])
        stage.wall['decode_validate'] += time.monotonic() - started
        yield row
    require(count == group['rows'] and candidates == group['candidates'],
            'Incomplete source group')


def _index_stats(index):
    return dict(groups=index['groups'], candidates=index['candidates'],
                unique32=len(index['keys32']), unique64=len(index['keys64']),
                contexts64=sum(len(v['contexts']) for v in index['keys64'].values()),
                candidate_counts={str(k): v for k, v in sorted(index['candidate_counts'].items())})


def _validate_index(index, manifest, model_sha, metadata=None):
    train = [group for group in manifest['groups'] if group['split'] == 'train']
    require(index['schema'] == NUMERIC_SCHEMA and index['checkpoint_sha256'] == model_sha,
            'Index/model identity mismatch')
    if metadata is not None:
        require(index['metadata'] == metadata, 'Index/model schema mismatch')
    require(index['groups'] == sum(g['rows'] for g in train) > 0 and
            index['candidates'] == sum(g['candidates'] for g in train),
            'Incomplete train-only index')
    require(np.shape(index['min']) == (373,) and np.shape(index['max']) == (373,)
            and np.isfinite(index['min']).all() and np.isfinite(index['max']).all()
            and np.all(index['min'] <= index['max']), 'Invalid index ranges')
    require(sum(index['candidate_counts'].values()) == index['groups'] and
            sum(k * v for k, v in index['candidate_counts'].items()) == index['candidates'],
            'Invalid index candidate frequencies')
    for field in ('keys32', 'keys64'):
        require(sum(sum(g['labels'].values()) for g in index[field].values()) == index['groups'],
                'Incomplete index numeric-key population')
        for key, group in index[field].items():
            _hash(key, field)
            require(sum(sum(labels.values()) for labels in group['contexts'].values()) ==
                    sum(group['labels'].values()), 'Incomplete index context population')
    require(all(math.isfinite(v) for values in index['discrete'].values() for v in values),
            'Nonfinite index discrete value')
    expected_sources = {g['game'] + '/' + kind: digest for g in train
                        for kind, digest in g['source_sha256'].items()}
    require(index['source_files_sha256'] == expected_sources, 'Index train source chain mismatch')


def _train_index(root, manifest, model, stage):
    index = new_index(model)
    for group in manifest['groups']:
        if group['split'] != 'train':
            continue
        for row in _sample_rows(root, group, stage):
            check_deadline(stage.deadline)
            started = time.monotonic()
            add_training(index, row['inputs'], row['label']['index'], model)
            stage.wall['index_insert'] += time.monotonic() - started
            stage.snapshot('indexing', game=group['game'], source_row_index=row['source_row_index'],
                           indexed_groups=index['groups'], indexed_candidates=index['candidates'])
        index['source_files_sha256'].update({group['game'] + '/' + k: v
                                            for k, v in group['source_sha256'].items()})
        stage.snapshot('group-completed', game=group['game'], indexed_groups=index['groups'],
                       indexed_candidates=index['candidates'])
    _validate_index(index, manifest, model.sha, model.metadata)
    return index


def build_index(out, cfg, deadline, model):
    """Commit complete sample train support. No model scoring is performed."""
    with Stage(out, cfg, 'build-index', deadline) as stage:
        root, manifest, binding = _manifest(cfg, stage)
        _check_model(model, cfg)
        index = _train_index(root, manifest, model, stage)
        payload = dict(**binding, groups_manifest=manifest['groups'], index=index)
        started = time.monotonic()
        path = stage.out / 'support-index.pkl'
        with _new_file(path) as stream:
            pickle.dump(payload, stream, protocol=pickle.HIGHEST_PROTOCOL)
        index_sha = sha(path, deadline)
        stage.wall['index_commit'] += time.monotonic() - started
        receipt = stage.receipt(binding, support_index_sha256=index_sha,
            index_bytes=path.stat().st_size, index_stats=_index_stats(index),
            source_files_sha256=index['source_files_sha256'],
            scoring_precision='original float64', support_precision='separate float32 and float64')
        receipt_path = stage.out / 'support-index-receipt.json'
        write_new(receipt_path, receipt)
        receipt_sha = sha(receipt_path, deadline)
        stage.snapshot('completed', support_index_sha256=index_sha,
                       support_index_receipt_sha256=receipt_sha)
        return dict(receipt, receipt_sha256=receipt_sha, support_index_receipt_sha256=receipt_sha)


def _load_index(out, cfg, manifest, binding, stage, model=None):
    started = time.monotonic()
    path, receipt_path = Path(out) / 'support-index.pkl', Path(out) / 'support-index-receipt.json'
    require(sha(receipt_path, stage.deadline) ==
            _hash(cfg['support_index_receipt_sha256'], 'support_index_receipt'),
            'Support index completion receipt hash mismatch')
    receipt = _read(receipt_path)
    require(all(receipt.get(k) == v for k, v in binding.items()) and receipt.get('passed') is True
            and receipt.get('stage') == 'build-index', 'Uncommitted or mismatched index receipt')
    index_sha = _hash(cfg['support_index_sha256'], 'support_index')
    require(receipt['support_index_sha256'] == index_sha and sha(path, stage.deadline) == index_sha,
            'Support index hash mismatch before unpickle')
    require(path.stat().st_size == receipt['index_bytes'], 'Support index byte size mismatch')
    # The complete bytes have been checked against the caller's committed hash.
    with path.open('rb') as stream:
        payload = pickle.load(stream)
        require(not stream.read(1), 'Trailing support index bytes')
    check_deadline(stage.deadline)
    require(all(payload.get(k) == v for k, v in binding.items()) and
            payload['groups_manifest'] == manifest['groups'], 'Index sample source binding mismatch')
    index = payload['index']
    _validate_index(index, manifest, cfg['model_sha256'], None if model is None else model.metadata)
    require(_index_stats(index) == receipt['index_stats'] or
            canonical(_index_stats(index)) == canonical(receipt['index_stats']),
            'Index statistics mismatch')
    stage.counts['index_loads'] += 1
    stage.wall['index_load'] += time.monotonic() - started
    return index


class MeteredModel:
    """Score receives only original numeric arrays; labels never enter this API."""
    def __init__(self, model, stage):
        self.model, self.stage = model, stage
        self.sha, self.metadata = model.sha, model.metadata

    def score(self, shared, candidates):
        stage = self.stage
        check_deadline(stage.deadline)
        require(shared.dtype == np.dtype('float64') and candidates.dtype == np.dtype('float64'),
                'Scoring precision changed')
        stage.counts['actual_score_calls'] += 1
        stage.counts['actual_score_candidates'] += len(candidates)
        stage.counts['cache_misses'] += 1
        stage.snapshot('model-score-started')
        started = time.monotonic()
        try:
            return self.model.score(shared, candidates)
        finally:
            stage.wall['model_score'] += time.monotonic() - started


def _evaluate(analyzer, row, group, stage):
    check_deadline(stage.deadline)
    count = len(row['inputs']['candidates'])
    groups_cap, candidates_cap = stage.cfg['max_evaluated_groups'], stage.cfg['max_evaluated_candidates']
    require(type(groups_cap) is int and type(candidates_cap) is int and groups_cap >= 0
            and candidates_cap >= 0, 'Invalid allocated evaluation budget')
    require(stage.counts['evaluated_groups'] + 1 <= groups_cap and
            stage.counts['evaluated_candidates'] + count <= candidates_cap,
            'Allocated evaluation budget exhausted before evaluate')
    stage.counts['evaluated_groups'] += 1
    stage.counts['evaluated_candidates'] += count
    stage.snapshot('evaluate-started', game=group['game'], source_row_index=row['source_row_index'])
    calls = stage.counts['actual_score_calls']
    started = time.monotonic()
    label = row['label']['teacher_label']
    compact = analyzer.evaluate(dict(inputs=row['inputs'], teacher_label=label, chosen=label),
                                'historical_' + group['split'])
    stage.wall['evaluate'] += time.monotonic() - started
    check_deadline(stage.deadline)
    stage.counts['cache_hits'] += int(stage.counts['actual_score_calls'] == calls)
    stage.counts['completed_groups'] += 1
    stage.counts['completed_candidates'] += count
    return dict(split=group['split'], game=group['game'], source_row_index=row['source_row_index'],
                diagnostic=compact)


def score_shard(out, cfg, deadline, game, model):
    """One original game, one read-only complete support index, one new cache."""
    require(isinstance(game, str) and TOKEN.fullmatch(game) is not None, 'Invalid shard game')
    with Stage(out, cfg, 'score-' + game, deadline) as stage:
        root, manifest, binding = _manifest(cfg, stage)
        _check_model(model, cfg)
        selected = [g for g in manifest['groups'] if g['game'] == game]
        require(len(selected) == 1, 'Shard is not in the frozen sample manifest')
        group = selected[0]
        index = _load_index(out, cfg, manifest, binding, stage, model)
        analyzer = RecordAnalyzer(index, MeteredModel(model, stage))
        path = stage.out / 'shards' / (game + '.jsonl.gz')
        with _gzip_output(path) as stream:
            for row in _sample_rows(root, group, stage):
                compact = _evaluate(analyzer, row, group, stage)
                started = time.monotonic()
                _emit(stream, compact)
                stage.wall['compact_write'] += time.monotonic() - started
                stage.snapshot('row-completed', game=game, source_row_index=row['source_row_index'])
        require(stage.counts['completed_groups'] == group['rows'] and
                stage.counts['completed_candidates'] == group['candidates'], 'Incomplete scoring shard')
        stage.snapshot('group-completed', game=game)
        receipt = stage.receipt(binding, game=game, split=group['split'],
            source_row_start=group['source_row_start'], source_row_stop=group['source_row_stop'],
            source_sha256=group['source_sha256'], sample_group_sha256=group['sha256'],
            support_index_sha256=cfg['support_index_sha256'],
            support_index_receipt_sha256=cfg['support_index_receipt_sha256'],
            compact_sha256=sha(path, deadline), compact_bytes=path.stat().st_size)
        receipt_path = path.with_name(game + '.receipt.json')
        write_new(receipt_path, receipt)
        receipt_sha = sha(receipt_path, deadline)
        stage.snapshot('completed', compact_sha256=receipt['compact_sha256'], receipt_sha256=receipt_sha)
        return dict(receipt, receipt_sha256=receipt_sha)


def _compact(record, group, expected_row):
    require(set(record) == {'split', 'game', 'source_row_index', 'diagnostic'} and
            record['split'] == group['split'] and record['game'] == group['game'] and
            type(record['source_row_index']) is int and record['source_row_index'] == expected_row,
            'Missing, duplicate, or reordered compact row identity')
    row = record['diagnostic']
    require(isinstance(row, dict) and row['schema'] == NUMERIC_SCHEMA and
            row['origin'] == 'historical_' + group['split'], 'Compact diagnostic schema mismatch')
    for field in ('key32', 'key64', 'semantic_key', 'context_key'):
        _hash(row[field], field)
    n = row['candidate_count']
    require(type(n) is int and n > 0 and all(type(row[f]) is int and 0 <= row[f] < n
            for f in ('teacher_index', 'pred_index')), 'Invalid compact candidate/selection count')
    require(all(type(row[f]) is bool for f in FLAGS), 'Invalid compact boolean flags')
    require(type(row['logit_gap']) in (int, float) and math.isfinite(row['logit_gap']) and
            row['logit_gap'] >= 0, 'Invalid compact logit gap')
    margin = row['teacher_margin']
    require((n == 1 and margin is None) or (n > 1 and type(margin) in (int, float)
            and math.isfinite(margin)), 'Invalid compact teacher margin')
    for prefix in ('teacher', 'pred'):
        require(row[prefix + '_id'] == candidate_id(row[prefix + '_unit'], row[prefix + '_target'],
                                                   row[prefix + '_op']), 'Compact semantic ID mismatch')
    canonical(row)  # Reject nonfinite values anywhere, including unused fields.
    return row


def _shard_records(out, cfg, manifest, binding, stage):
    expected = [g['game'] for g in manifest['groups']]
    receipt_hashes = cfg['shard_receipt_sha256']
    require(isinstance(receipt_hashes, dict) and list(receipt_hashes) == expected,
            'Missing, extra, duplicate, or reordered shard receipt manifest')
    folder = Path(out) / 'shards'
    require({p.name for p in folder.glob('*.jsonl.gz')} == {g + '.jsonl.gz' for g in expected}
            and {p.name for p in folder.glob('*.receipt.json')} == {g + '.receipt.json' for g in expected},
            'Missing or extra committed shard files')
    for group in manifest['groups']:
        game = group['game']
        path = folder / (game + '.jsonl.gz')
        receipt_path = folder / (game + '.receipt.json')
        require(sha(receipt_path, stage.deadline) == _hash(receipt_hashes[game], 'shard_receipt'),
                'Shard receipt hash mismatch')
        receipt = _read(receipt_path)
        required = dict(binding, passed=True, stage='score-' + game, game=game, split=group['split'],
                        source_row_start=group['source_row_start'], source_row_stop=group['source_row_stop'],
                        source_sha256=group['source_sha256'], sample_group_sha256=group['sha256'],
                        support_index_sha256=cfg['support_index_sha256'],
                        support_index_receipt_sha256=cfg['support_index_receipt_sha256'])
        require(all(receipt.get(k) == v for k, v in required.items()), 'Shard source/index binding mismatch')
        require(sha(path, stage.deadline) == receipt['compact_sha256'] and
                path.stat().st_size == receipt['compact_bytes'], 'Compact shard hash/size mismatch')
        count = candidates = 0
        for record in _lines(path, stage.deadline):
            require(count < group['rows'], 'Extra compact row')
            row = _compact(record, group, count)
            count += 1; candidates += row['candidate_count']
            yield record
        require(count == group['rows'] == receipt['counts']['completed_groups'] and
                candidates == group['candidates'] == receipt['counts']['completed_candidates'],
                'Incomplete compact shard population')
        stage.snapshot('group-completed', game=game)


def aggregate(out, cfg, deadline):
    """Replay every compact row in original manifest/row order, globally by split."""
    with Stage(out, cfg, 'aggregate', deadline) as stage:
        _, manifest, binding = _manifest(cfg, stage)
        _load_index(out, cfg, manifest, binding, stage)
        accumulators = {split: SummaryAccumulator() for split in SPLITS}
        started = time.monotonic()
        for record in _shard_records(out, cfg, manifest, binding, stage):
            row = record['diagnostic']
            accumulators[record['split']].add(row)
            stage.counts['completed_groups'] += 1
            stage.counts['completed_candidates'] += row['candidate_count']
            stage.snapshot('aggregating', game=record['game'], source_row_index=record['source_row_index'])
        summaries = {split: acc.result() for split, acc in accumulators.items()}
        stage.wall['aggregate'] += time.monotonic() - started
        _validate_summaries(summaries, manifest)
        path = stage.out / 'summary.json'
        write_new(path, summaries)
        receipt = stage.receipt(binding, support_index_sha256=cfg['support_index_sha256'],
            support_index_receipt_sha256=cfg['support_index_receipt_sha256'],
            shard_receipt_sha256=cfg['shard_receipt_sha256'], summary_sha256=sha(path, deadline),
            summary_bytes=path.stat().st_size)
        receipt_path = stage.out / 'aggregate-receipt.json'
        write_new(receipt_path, receipt)
        receipt_sha = sha(receipt_path, deadline)
        stage.snapshot('completed', receipt_sha256=receipt_sha)
        return dict(receipt, receipt_sha256=receipt_sha)


def _validate_summaries(summaries, manifest):
    for split in SPLITS:
        groups = [g for g in manifest['groups'] if g['split'] == split]
        raw = summaries[split]['raw_weighted']
        require(raw['denominator'] == sum(g['rows'] for g in groups) and
                raw['total_candidates'] == sum(g['candidates'] for g in groups),
                'Global summary denominator mismatch')
    canonical(summaries)


def reference(out, cfg, deadline, model):
    """Old pure-function order, plus a streaming per-row writer for comparison.

    Exactly as old _build_and_summarize: all training insertions first, then a
    single analyzer/cache and one accumulator per split, train then validation.
    No additional model evaluation is used to collect the compact records.
    """
    with Stage(out, cfg, 'reference', deadline) as stage:
        root, manifest, binding = _manifest(cfg, stage)
        _check_model(model, cfg)
        index = _train_index(root, manifest, model, stage)
        analyzer = RecordAnalyzer(index, MeteredModel(model, stage))
        summaries = {}
        path = stage.out / 'reference.jsonl.gz'
        with _gzip_output(path) as stream:
            for split in SPLITS:
                acc = SummaryAccumulator()
                for group in manifest['groups']:
                    if group['split'] != split:
                        continue
                    for row in _sample_rows(root, group, stage):
                        record = _evaluate(analyzer, row, group, stage)
                        started = time.monotonic()
                        _emit(stream, record)
                        stage.wall['compact_write'] += time.monotonic() - started
                        acc.add(record['diagnostic'])
                        stage.snapshot('row-completed', game=group['game'], source_row_index=row['source_row_index'])
                    stage.snapshot('group-completed', game=group['game'])
                summaries[split] = acc.result()
        _validate_summaries(summaries, manifest)
        summary_path = stage.out / 'summary.json'
        write_new(summary_path, summaries)
        receipt = stage.receipt(binding, index_stats=_index_stats(index),
            source_files_sha256=index['source_files_sha256'],
            compact_sha256=sha(path, deadline), compact_bytes=path.stat().st_size,
            summary_sha256=sha(summary_path, deadline), summary_bytes=summary_path.stat().st_size,
            reference='Old pure functions in _build_and_summarize order; original float64 inputs')
        receipt_path = stage.out / 'reference-receipt.json'
        write_new(receipt_path, receipt)
        receipt_sha = sha(receipt_path, deadline)
        stage.snapshot('completed', receipt_sha256=receipt_sha)
        return dict(receipt, receipt_sha256=receipt_sha)


def compare_reference(out, reference_out, cfg, deadline):
    """No scoring: exact canonical compact rows and byte-identical summaries."""
    from itertools import zip_longest
    out, reference_out = Path(out), Path(reference_out)
    require(sha(Path(cfg['sample_root']) / 'sample-manifest.json', deadline) == cfg['sample_manifest_sha256'],
            'Comparison sample manifest hash mismatch')
    manifest = _read(Path(cfg['sample_root']) / 'sample-manifest.json')
    require(sha(out / 'aggregate-receipt.json', deadline) ==
            _hash(cfg['aggregate_receipt_sha256'], 'aggregate_receipt'),
            'Comparison aggregate completion receipt hash mismatch')
    require(sha(reference_out / 'reference-receipt.json', deadline) ==
            _hash(cfg['reference_receipt_sha256'], 'reference_receipt'),
            'Comparison reference completion receipt hash mismatch')
    completed = _read(out / 'aggregate-receipt.json')
    ref = _read(reference_out / 'reference-receipt.json')
    require(completed.get('stage') == 'aggregate' and completed.get('freeze_sha256') ==
            _hash(cfg['freeze_sha256'], 'aggregate_freeze'),
            'Comparison aggregate stage/freeze mismatch')
    require(ref.get('stage') == 'reference' and ref.get('freeze_sha256') ==
            _hash(cfg['reference_freeze_sha256'], 'reference_freeze'),
            'Comparison reference stage/freeze mismatch')
    for key in ('source_manifest_sha256', 'sample_manifest_sha256', 'model_sha256'):
        require(completed[key] == ref[key], 'Comparison provenance mismatch: ' + key)
    require(completed['sample_manifest_sha256'] == cfg['sample_manifest_sha256'] and
            completed['model_sha256'] == cfg['model_sha256'] and
            completed['source_manifest_sha256'] == manifest['source_manifest_sha256'],
            'Comparison provenance does not match the caller')
    for key in ('support_index_sha256', 'support_index_receipt_sha256', 'shard_receipt_sha256'):
        require(completed[key] == cfg[key], 'Comparison completed index/shard binding mismatch')
    require(sha(out / 'support-index.pkl', deadline) == cfg['support_index_sha256'] and
            sha(out / 'support-index-receipt.json', deadline) == cfg['support_index_receipt_sha256'],
            'Comparison support index changed')
    require(completed['passed'] is True and ref['passed'] is True, 'Comparison requires completed outputs')
    for folder, receipt in ((out, completed), (reference_out, ref)):
        require(sha(folder / 'summary.json', deadline) == receipt['summary_sha256'],
                'Comparison summary hash mismatch')
    require((out / 'summary.json').read_bytes() == (reference_out / 'summary.json').read_bytes(),
            'New and old global summaries differ')
    require(sha(reference_out / 'reference.jsonl.gz', deadline) == ref['compact_sha256'],
            'Comparison reference compact hash mismatch')

    def ordered():
        for split in SPLITS:
            for group in manifest['groups']:
                if group['split'] != split:
                    continue
                game = group['game']
                receipt_path = out / 'shards' / (game + '.receipt.json')
                require(sha(receipt_path, deadline) == completed['shard_receipt_sha256'][game],
                        'Comparison shard receipt hash mismatch')
                receipt = _read(receipt_path)
                path = out / 'shards' / (game + '.jsonl.gz')
                require(sha(path, deadline) == receipt['compact_sha256'], 'Comparison compact hash mismatch')
                count = 0
                for record in _lines(path, deadline):
                    _compact(record, group, count)
                    count += 1
                    yield record
                require(count == group['rows'], 'Comparison shard row count mismatch')

    count = candidates = 0
    for new, old in zip_longest(ordered(), _lines(reference_out / 'reference.jsonl.gz', deadline)):
        require(new is not None and old is not None and canonical(new) == canonical(old),
                'New and old compact row differ or are incomplete')
        count += 1; candidates += new['diagnostic']['candidate_count']
    require(count == sum(g['rows'] for g in manifest['groups']) and
            candidates == sum(g['candidates'] for g in manifest['groups']), 'Comparison population mismatch')
    return dict(schema=SCHEMA, passed=True, sample_only=True, exact_compact_rows=count,
                exact_candidates=candidates, byte_identical_summary=True,
                model_calls=0, evaluated_groups=0, evaluated_candidates=0,
                sample_manifest_sha256=cfg['sample_manifest_sha256'], model_sha256=cfg['model_sha256'],
                freeze_sha256=cfg['freeze_sha256'],
                reference_freeze_sha256=cfg['reference_freeze_sha256'],
                aggregate_receipt_sha256=cfg['aggregate_receipt_sha256'],
                reference_receipt_sha256=cfg['reference_receipt_sha256'])
