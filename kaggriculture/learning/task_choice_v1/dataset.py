"""Rebuild task-choice demonstrations from the original 60 official replays.

This offline consumer never creates new games or trains a model. Inputs, labels
and identifying metadata are separate, line-aligned local files. Only aggregate
coverage, hashes and reconstruction certificates are suitable for publication.
"""
import argparse
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from contextlib import ExitStack
import gzip
import hashlib
import io
import json
from pathlib import Path
import time

from learning.bc_crop.agent import ROOT, sha
from learning.bc_crop.experiment import read, save, integrity
from decision.audit_market_teacher import observation
from decision.full_audit import reexecute

OPS = ('PLANT', 'DIG', 'WATER', 'HARVEST', 'FERTILIZE')


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'),
                      ensure_ascii=False, allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode('utf-8')).hexdigest()


def load_agent(path):
    namespace = {'__name__': 'frozen_demonstration_actor', '__file__': str(path)}
    exec(path.read_bytes(), namespace)
    return namespace['agent']


def selected_rows(cfg):
    """Exact original Cartesian population; no replacement or split reassignment."""
    source = ROOT / cfg['demonstration_root']
    original = integrity(source)
    for key in ('train_seeds', 'validation_seeds'):
        assert cfg[key] == original[key], key
    assert len(cfg['train_seeds']) == 8 and len(cfg['validation_seeds']) == 2
    assert not set(cfg['train_seeds']) & set(cfg['validation_seeds'])
    assert cfg['ledger_names'] == ['collect-ledger.json', 'collect-smoke-ledger.json']
    rows = [row for name in cfg['ledger_names'] for row in read(source / name)]
    wanted = {(s, op, side) for s in cfg['train_seeds'] + cfg['validation_seeds']
              for op in ('v1', 'market', 'round2') for side in (0, 1)}
    actual = [(r['seed'], r['opponent'], r['side']) for r in rows]
    assert len(rows) == len(set(actual)) == 60 and set(actual) == wanted
    assert len({r['game'] for r in rows}) == 60
    for row in rows:
        assert row['policy'] == 'teacher' and row['passed'] is True
        assert row['states'] == 720 and row['action_turns'] == 719
        assert row['final_status'] == ['DONE', 'DONE'] and row['agent_errors'] == 0
        assert row['engine_sha256'] == cfg['engine_sha256']
    return source, sorted(rows, key=lambda r: r['game'])


def _writer(stack, path):
    raw = stack.enter_context(path.open('xb'))
    zipped = stack.enter_context(gzip.GzipFile(filename='', mode='wb', fileobj=raw, mtime=0))
    return stack.enter_context(io.TextIOWrapper(zipped, encoding='utf-8', newline='\n'))


def check(job):
    """Reconstruct one recorded trajectory and reexecute its official actions."""
    from .agent import TaskAgent
    out_text, source_text, row, split = job
    out, source = Path(out_text), Path(source_text)
    side = row['side']
    replay_path = source / 'games' / row['game'] / 'replay.json.gz'
    assert sha(replay_path) == row['replay_gzip_sha256']
    assert sha(ROOT / row['agent_relative']) == row['bot_sha256']
    assert sha(source / 'sources' / (row['opponent'] + '.py')) == row['rival_sha256']
    with gzip.open(replay_path, 'rt', encoding='utf-8') as stream:
        replay = json.load(stream)
    assert len(replay['steps']) == 720
    assert [s['status'] for s in replay['steps'][-1]] == ['DONE', 'DONE']
    agent = TaskAgent()
    teacher = load_agent(source / 'sources' / 'teacher.py')
    rival = load_agent(source / 'sources' / (row['opponent'] + '.py'))
    folder = out / 'data' / split
    paths = {kind: folder / (row['game'] + '.' + kind + '.jsonl.gz')
             for kind in ('inputs', 'labels', 'index')}
    counts = {op: Counter() for op in OPS}
    unique = {op: {'candidate': set(), 'selected': set()} for op in OPS}
    decision_count = 0
    competitive_decisions = 0
    schemas = set()
    with ExitStack() as stack:
        streams = {kind: _writer(stack, path) for kind, path in paths.items()}
        for step, (before, after) in enumerate(zip(replay['steps'], replay['steps'][1:])):
            obs = observation(before, side)
            action = agent(obs, replay['configuration'])
            assert action == after[side]['action'], (row['game'], step, 'task reconstruction')
            assert teacher(observation(before, side), replay['configuration']) == action
            assert rival(observation(before, 1-side), replay['configuration']) == after[1-side]['action']
            for ordinal, record in enumerate(agent.last['records']):
                inputs = record['inputs']
                candidates = inputs['candidates']
                ids = [c['id'] for c in candidates]
                assert ids and len(ids) == len(set(ids))
                label = record['teacher_label']
                assert label in ids and record['chosen'] == label
                picked = candidates[ids.index(label)]
                assert picked['op'] == record['op']
                assert picked['unit'] == record['unit'] and picked['target'] == record['target']
                assert record['unit_request'] == [action['farmer'], *action['hands']][record['unit']]
                assert set(c['op'] for c in candidates) <= set(OPS)
                schemas.add((inputs['schema_version'], inputs['schema_sha256']))
                competitive = len({c['op'] for c in candidates}) > 1
                competitive_decisions += competitive
                context_hash = digest(inputs)
                for candidate in candidates:
                    op = candidate['op']
                    # Full legal context + candidate identity, not a teacher score
                    # or label. Multiple repeated contexts count only once here.
                    encoded = digest([context_hash, candidate['id']])
                    counts[op]['candidate_count'] += 1
                    counts[op]['candidate_competitive_count'] += competitive
                    unique[op]['candidate'].add(encoded)
                    if candidate['id'] == label:
                        counts[op]['selected_count'] += 1
                        counts[op]['selected_competitive_count'] += competitive
                        unique[op]['selected'].add(encoded)
                streams['inputs'].write(canonical(inputs) + '\n')
                streams['labels'].write(canonical({'teacher_label': label, 'index': ids.index(label)}) + '\n')
                streams['index'].write(canonical(dict(game=row['game'], seed=row['seed'],
                    opponent=row['opponent'], side=side, step=step, decision=ordinal,
                    split=split, input_sha256=context_hash)) + '\n')
                decision_count += 1
    assert len(schemas) == 1
    audit = reexecute(replay)
    assert audit['official_transitions_reexecuted'] == 719
    assert audit['semantic_sha256'] == row['semantic_replay_sha256']
    assert audit['semantic_sha256_v2'] == row['semantic_replay_sha256_v2']
    assert audit['final_rewards'] == row['banks'] == row['rewards']
    public = dict(game=row['game'], seed=row['seed'], opponent=row['opponent'], side=side,
        split=split, passed=True, reconstructed_actor_actions=719,
        original_teacher_actions_checked=719, reproduced_rival_actions=719,
        replay_gzip_sha256=row['replay_gzip_sha256'], original_bot_sha256=row['bot_sha256'],
        rival_sha256=row['rival_sha256'], decisions=decision_count,
        cross_task_competition_decisions=competitive_decisions, audit=audit,
        files={kind: dict(path=path.relative_to(out).as_posix(), sha256=sha(path), rows=decision_count)
               for kind, path in paths.items()},
        schema_version=next(iter(schemas))[0], schema_sha256=next(iter(schemas))[1],
        per_task={op: {**dict(counts[op]),
                      'unique_candidate_encodings': len(unique[op]['candidate']),
                      'unique_selected_encodings': len(unique[op]['selected'])} for op in OPS})
    # Unique hashes stay in process IPC only; do not publish per-context identities.
    return public, unique


def run(out, workers=4):
    from .experiment import ready, verify
    out = Path(out).resolve()
    cfg = ready(out)
    assert isinstance(workers, int) and 1 <= workers <= min(4, cfg['workers'])
    source, rows = selected_rows(cfg)
    assert not (out / 'reconstruction-started.json').exists(), 'One attempt only; no automatic restart'
    for name in ('coverage.json', 'dataset-manifest.json', 'reconstruction-ledger.json'):
        assert not (out / name).exists()
    save(out / 'reconstruction-started.json', dict(games=60, workers=workers,
        new_games=0, training_runs=0, official_existing_replay_transitions=60*719,
        registration_sha256=sha(out / 'registration.json')))
    for split in ('train', 'validation'):
        (out / 'data' / split).mkdir(parents=True, exist_ok=False)
    jobs = [(str(out), str(source), row,
             'train' if row['seed'] in cfg['train_seeds'] else 'validation') for row in rows]
    completed = []
    aggregate = {split: {op: {'counts': Counter(), 'candidate_seeds': set(),
        'selected_seeds': set(), 'candidate_unique': set(), 'selected_unique': set()}
        for op in OPS} for split in ('train', 'validation')}
    failures = []
    started = 0
    tick = time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        pending = {}
        def dispatch():
            nonlocal started
            job = jobs[started]
            pending[pool.submit(check, job)] = job[2]['game']
            started += 1
        while started < min(workers, len(jobs)):
            dispatch()
        while pending:
            done, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                game = pending.pop(future)
                try:
                    record, unique = future.result()
                    completed.append(record)
                    for op in OPS:
                        acc = aggregate[record['split']][op]
                        per = record['per_task'][op]
                        for key in ('candidate_count', 'selected_count',
                                    'candidate_competitive_count', 'selected_competitive_count'):
                            acc['counts'][key] += per.get(key, 0)
                        if per.get('candidate_count', 0): acc['candidate_seeds'].add(record['seed'])
                        if per.get('selected_count', 0): acc['selected_seeds'].add(record['seed'])
                        acc['candidate_unique'].update(unique[op]['candidate'])
                        acc['selected_unique'].update(unique[op]['selected'])
                except Exception as exc:
                    failures.append(dict(game=game, exception=type(exc).__name__))
            # Inspect every completion in this batch before refilling. An error
            # drains already-running jobs; no retry, replacement or extra dispatch.
            if not failures:
                while len(pending) < workers and started < len(jobs): dispatch()
            print(f'task reconstruction: {len(completed)}/60; failed={len(failures)}', flush=True)
    ledger = dict(passed=not failures and len(completed) == 60,
        games=len(completed), games_dispatched=started, failures=failures,
        new_games=0, training_runs=0, official_existing_replay_transitions=len(completed)*719,
        workers=workers, elapsed_seconds=time.perf_counter()-tick,
        rows=sorted(completed, key=lambda r: r['game']))
    save(out / 'reconstruction-ledger.json', ledger)
    if failures:
        save(out / 'reconstruction-failed.json', dict(failures=failures, no_retry=True))
        raise RuntimeError('Reconstruction failed; retained local partial evidence; no retry')
    assert len(completed) == 60
    schemas = {(r['schema_version'], r['schema_sha256']) for r in completed}
    assert len(schemas) == 1
    coverage = {}
    for split, by_task in aggregate.items():
        subset = [r for r in completed if r['split'] == split]
        coverage[split] = dict(games=len(subset), seeds=cfg[split + '_seeds'],
            decisions=sum(r['decisions'] for r in subset),
            cross_task_competition_decisions=sum(r['cross_task_competition_decisions'] for r in subset),
            tasks={op: {**dict(acc['counts']),
                       'candidate_seeds': sorted(acc['candidate_seeds']),
                       'selected_seeds': sorted(acc['selected_seeds']),
                       'unique_candidate_encodings': len(acc['candidate_unique']),
                       'unique_selected_encodings': len(acc['selected_unique'])}
                   for op, acc in by_task.items()})
    gates = dict(all_60_full_action_parities=True,
        each_task_selected_in_both_splits=all(coverage[s]['tasks'][op].get('selected_count', 0) > 0
                                            for s in coverage for op in OPS),
        each_task_selected_with_competing_type_in_both_splits=all(
            coverage[s]['tasks'][op].get('selected_competitive_count', 0) > 0
            for s in coverage for op in OPS),
        both_splits_have_cross_task_competition=all(coverage[s]['cross_task_competition_decisions'] > 0
                                                   for s in coverage))
    save(out / 'coverage.json', dict(passed=all(gates.values()), gates=gates, splits=coverage,
        unique_encoding_definition='Canonical complete selector input plus candidate ID; includes legally visible clock and current reservations. Structural uniqueness is not statistical independence.',
        interpretation='Minimum structural coverage only, not sufficient examples, quality, or learned strength. Teacher opportunity filters remain heuristic, not official legality masks.',
        stop='If coverage fails, retain gaps; no automatic extra demonstrations, changed splits, relaxed gates, training or new games.'))
    version, schema_sha = next(iter(schemas))
    save(out / 'dataset-manifest.json', dict(schema_version=version, schema_sha256=schema_sha,
        source='Original 60 frozen-teacher official demonstration replays only',
        source_ledgers={name: sha(source/name) for name in cfg['ledger_names']},
        teacher_sha256=cfg['teacher_sha256'], registration_sha256=sha(out/'registration.json'),
        reconstruction_ledger_sha256=sha(out/'reconstruction-ledger.json'),
        coverage_sha256=sha(out/'coverage.json'),
        inputs='data/<split>/<game>.inputs.jsonl.gz: one variable-candidate selector input per line',
        labels='data/<split>/<game>.labels.jsonl.gz: separate teacher candidate ID and its zero-based input position',
        metadata='data/<split>/<game>.index.jsonl.gz: offline identities only, never model features; line-aligned with inputs and labels',
        privacy='All raw data files remain local; publish aggregate results and file hashes only.',
        training_runs=0, new_games=0, examples=sum(r['decisions'] for r in completed),
        files=[dict(game=r['game'], split=r['split'], files=r['files']) for r in ledger['rows']]))
    verify(out)
    return ledger


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    run(args.out, args.workers)
