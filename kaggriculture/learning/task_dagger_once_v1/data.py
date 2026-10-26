"""Student-visited task labels, kept outside deployment inputs.

Only complete frozen-student training/development episodes are accepted. Each
conditional label belongs to its own callback's candidate/reservation context;
there is no matching to a separately diverged teacher trajectory.
"""
from contextlib import contextmanager
import gzip
import json
from pathlib import Path
import re

import numpy as np

from learning.bc_crop.agent import sha
from learning.task_bc_holdout_v1.games import write_new
from learning.task_bc_v1.cache import decode
from learning.task_bc_v1.train import verify_cache
from learning.task_choice_v1.dataset import digest
from learning.task_choice_v1.features import OPS, SCHEMA_VERSION


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def require(value, message):
    if not value:
        raise ValueError(message)


def validate_role(row, cfg):
    """Reject held-out/other-policy input before touching its files."""
    sets = []
    for key, count in (('train_seeds', 4), ('validation_seeds', 2), ('test_seeds', 8)):
        values = cfg[key]
        require(len(values) == count and len(set(values)) == count
                and all(type(s) is int and 0 <= s < 2**32 for s in values), 'Invalid seed partition')
        sets.append(set(values))
    require(not any(sets[i] & sets[j] for i in range(3) for j in range(i)), 'Overlapping seed partitions')
    require(row['split'] in ('train', 'dev') and row['policy'] == 'baseline',
            'Only collection/development frozen-student labels are allowed')
    split = 'train' if row['split'] == 'train' else 'validation'
    require(type(row['seed']) is int and row['seed'] in cfg[split + '_seeds'], 'Wrong split seed')
    require(cfg['opponents'] == ['round2', 'market'] and cfg['seats'] == [0, 1], 'Unexpected panel')
    require(row['opponent'] in cfg['opponents'] and type(row['side']) is int
            and row['side'] in cfg['seats'], 'Unknown opponent or seat')
    require(isinstance(row['game'], str) and re.fullmatch(r'[A-Za-z0-9_-]+', row['game']),
            'Invalid game identifier')
    return split


def source_identity(out, row, cfg):
    split = validate_role(row, cfg)
    out = Path(out).resolve(); game = row['game']
    paths = {name: out / 'job-results' / (game + suffix) for name, suffix in
             (('row', '.row.json'), ('audit', '.audit.json'), ('complete', '.complete.json'))}
    paths['trace'] = out / 'traces' / (game + '.jsonl.gz')
    require(all(p.resolve().is_relative_to(out) for p in paths.values()), 'Source path escaped run')
    marker, stored, certificate = read(paths['complete']), read(paths['row']), read(paths['audit'])
    require(marker['passed'] is True and marker['game'] == game
            and marker['row_sha256'] == sha(paths['row'])
            and marker['certificate_sha256'] == sha(paths['audit']), 'Unbound completion certificate')
    require(all(key in row and row[key] == value for key, value in stored.items()), 'Changed aggregate row')
    require(row['passed'] is True and row['states'] == 720 and row['action_turns'] == 719
            and row['status'] == ['DONE', 'DONE'] and row['agent_errors'] == 0, 'Incomplete source game')
    require(certificate['passed'] is True and certificate['game'] == game
            and certificate['trace_sha256'] == row['trace_sha256'] == sha(paths['trace']),
            'Changed or unverified source trace')
    require(row['task_checkpoint_sha256'] == cfg['task_checkpoint_sha256']
            and row['crop_checkpoint_sha256'] == cfg['crop_checkpoint_sha256'], 'Unexpected source model')
    return split, paths['trace']


def decode_record(record, row, step, decision, schema=None):
    """Preserve current candidate order, but label the teacher's conditional ID."""
    inputs = record['inputs']; candidates = inputs['candidates']
    ids = [candidate['id'] for candidate in candidates]
    teacher, chosen = record['teacher_label'], record['chosen']
    require(type(teacher) is str and teacher in ids, 'Teacher label is absent from current candidates')
    require(type(chosen) is str and chosen in ids and record['source'] == 'callback',
            'Invalid frozen-student choice')
    selected = candidates[ids.index(chosen)]
    require(type(record['unit']) is int and record['unit'] == selected['unit']
            and record['target'] == selected['target'] and record['op'] == selected['op'],
            'Chosen candidate metadata mismatch')
    label = {'teacher_label': teacher, 'index': ids.index(teacher)}
    index = dict(game=row['game'], split=row['split'], seed=row['seed'], side=row['side'],
                 opponent=row['opponent'], step=step, decision=decision, input_sha256=digest(inputs))
    return decode(inputs, label, index, game=row['game'], split=row['split'],
                  expected_seed=row['seed'], expected_side=row['side'],
                  expected_opponent=row['opponent'], schema=schema)


def records(out, row, cfg):
    """Read one complete bound trace; no input file from another role is opened."""
    split, path = source_identity(out, row, cfg)
    group_cap = min(cfg['data_limits'][split + '_groups'], 4000)
    candidate_cap = min(cfg['data_limits'][split + '_candidates'], 160000)
    groups = total = 0; schema = None
    with gzip.open(path, 'rt', encoding='utf-8') as stream:
        for step in range(719):
            line = stream.readline(); require(bool(line), 'Missing trace turn')
            event = json.loads(line)
            require(type(event['step']) is int and event['step'] == step
                    and event['schema_version'] == SCHEMA_VERSION, 'Trace turn/schema mismatch')
            require(event['checkpoint_sha256'] == cfg['crop_checkpoint_sha256']
                    and event['task_checkpoint_sha256'] == cfg['task_checkpoint_sha256'],
                    'Trace model identity mismatch')
            require(all(event[name] == 0 for name in ('load_errors', 'inference_errors', 'fallbacks')),
                    'Trace inference failure')
            require(type(event['records']) is list, 'Invalid callback sequence')
            for decision, record in enumerate(event['records']):
                count = len(record['inputs']['candidates'])
                require(groups + 1 <= group_cap and total + count <= candidate_cap, 'Per-game data cap')
                values = decode_record(record, row, step, decision, schema)
                schema = values[0]; groups += 1; total += count
                yield record, values
        require(not stream.readline(), 'Extra trace turn')
    require(sha(path) == row['trace_sha256'], 'Trace changed while consuming')


def summarize_game(out, row, cfg):
    groups = candidates = disagreements = 0; schema = None
    perop = {op: dict(groups=0, candidate_rows=0, teacher_ID_disagreements=0) for op in OPS}
    for record, values in records(out, row, cfg):
        schema, _shared, local, label, ops, _ids = values
        count = len(local); changed = int(record['teacher_label'] != record['chosen'])
        groups += 1; candidates += count; disagreements += changed
        item = perop[OPS[int(ops[label])]]
        item['groups'] += 1; item['candidate_rows'] += count; item['teacher_ID_disagreements'] += changed
    require(groups > 0 and schema is not None, 'No task supervision in source game')
    return dict(game=row['game'], split=row['split'], policy=row['policy'], seed=row['seed'],
                opponent=row['opponent'], side=row['side'], passed=True, label_binding_passed=True,
                states=720, action_turns=719, status=['DONE', 'DONE'], agent_errors=0,
                groups=groups, candidate_rows=candidates, teacher_ID_disagreements=disagreements,
                perop=perop, trace_sha256=row['trace_sha256'], schema_sha256=schema['schema_sha256'])


def ordered_population(rows, cfg):
    by_cell = {}
    for row in rows:
        split = validate_role(row, cfg)
        key = split, row['seed'], row['opponent'], row['side']
        require(key not in by_cell, 'Duplicate collection cell')
        by_cell[key] = row
    expected = [(split, seed, opponent, side) for split in ('train', 'validation')
                for seed in cfg[split + '_seeds'] for opponent in cfg['opponents'] for side in cfg['seats']]
    require(set(by_cell) == set(expected) and len({r['game'] for r in rows}) == len(rows),
            'Missing, extra or aliased collection cell')
    return {split: [by_cell[key] for key in expected if key[0] == split]
            for split in ('train', 'validation')}


@contextmanager
def arrays(folder, n, m):
    definitions = dict(shared=('<f4', (n, 275)), candidates=('<f4', (m, 98)),
                       offsets=('<i8', (n + 1,)), labels=('<i8', (n,)), selected_ops=('i1', (n,)),
                       candidate_ops=('i1', (m,)), candidate_identity=('<i2', (m, 3)), seeds=('<i8', (n,)))
    values = {}
    try:
        for name, (dtype, shape) in definitions.items():
            values[name] = np.lib.format.open_memmap(folder / (name + '.npy'), mode='w+', dtype=dtype, shape=shape)
        yield values
    finally:
        for value in values.values():
            value.flush(); value._mmap.close()


def build_cache(out, rows, cfg):
    """Materialize one immutable ragged cache in registration order, never by outcome."""
    out = Path(out); rows = list(rows); population = ordered_population(rows, cfg)
    require(not (out / 'cache-started.json').exists() and not (out / 'cache').exists(), 'Existing cache run')
    write_new(out / 'cache-started.json', dict(attempt=1, games=len(rows), role='student-state labels'))
    summaries = {row['game']: summarize_game(out, row, cfg) for row in rows}
    totals = {}
    for split, games in population.items():
        total = {name: sum(summaries[r['game']][name] for r in games)
                 for name in ('groups', 'candidate_rows', 'teacher_ID_disagreements')}
        require(0 < total['groups'] <= cfg['data_limits'][split + '_groups']
                and total['groups'] <= total['candidate_rows'] <= cfg['data_limits'][split + '_candidates'],
                'Aggregate data cap')
        totals[split] = dict(total, games=len(games))
    cache = out / 'cache'; cache.mkdir(); schema = None; manifest_arrays = {}; max_size = 0
    for split, games in population.items():
        folder = cache / split; folder.mkdir()
        n, m = totals[split]['groups'], totals[split]['candidate_rows']
        with arrays(folder, n, m) as values:
            group = offset = 0; values['offsets'][0] = 0
            for row in games:
                game_groups = game_rows = game_disagreements = 0
                for record, decoded in records(out, row, cfg):
                    current, shared, local, label, ops, identities = decoded
                    require(schema is None or schema == current, 'Cross-game schema mismatch')
                    schema = current; end = offset + len(local)
                    require(group < n and end <= m, 'Source exceeded counted shape')
                    values['shared'][group] = shared; values['candidates'][offset:end] = local
                    values['labels'][group] = label; values['offsets'][group + 1] = end
                    values['selected_ops'][group] = ops[label]; values['candidate_ops'][offset:end] = ops
                    values['candidate_identity'][offset:end] = identities; values['seeds'][group] = row['seed']
                    group += 1; offset = end; game_groups += 1; game_rows += len(local)
                    game_disagreements += int(record['teacher_label'] != record['chosen'])
                    max_size = max(max_size, len(local))
                summary = summaries[row['game']]
                require((game_groups, game_rows, game_disagreements) ==
                        tuple(summary[k] for k in ('groups', 'candidate_rows', 'teacher_ID_disagreements')),
                        'Source changed between count and materialization')
            require(group == n and offset == m, 'Incomplete cache')
            shapes = {name: dict(shape=list(value.shape), dtype=str(value.dtype)) for name, value in values.items()}
        manifest_arrays[split] = {name: dict(path=f'{split}/{name}.npy', sha256=sha(folder / (name + '.npy')), **shape)
                                  for name, shape in shapes.items()}
    write_new(cache / 'schema.json', schema)
    source_hashes = {f'traces/{r["game"]}.jsonl.gz': r['trace_sha256'] for r in rows}
    summary = dict(passed=True, splits=totals, games=[summaries[r['game']] for split in population for r in population[split]],
                   all_groups_retained=True, labels='Conditional teacher_label on each actual student callback',
                   numeric_model_inputs=['shared', 'candidates'], metric_only_arrays=['seeds', 'candidate_identity', 'selected_ops', 'candidate_ops'],
                   seed_splits_disjoint=True, development_used_for_gradients=False, holdout_read=False,
                   normalization_recomputed=False, cache_build_source_trace_passes_per_game=2,
                   note='Counting and materialization both validate source identity; any earlier caller summaries are separate reads.')
    write_new(out / 'cache-summary.json', summary)
    write_new(cache / 'manifest.json', dict(passed=True, schema_sha256=sha(cache / 'schema.json'),
              numeric_feature_schema_sha256=schema['schema_sha256'], arrays=manifest_arrays,
              source_files_sha256=source_hashes, max_group_size=max_size, split_summary=totals,
              summary_sha256=sha(out / 'cache-summary.json'),
              privacy='Local numeric arrays only; labels and identity metadata are not deployment inputs.'))
    verify_cache(cache)
    return summary
