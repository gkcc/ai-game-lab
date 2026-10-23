"""Diagnose crop BC errors using training/validation and development ONLY.

Reconstructs choices on stored observations, not new games or engine transitions.
Teacher scores are diagnostic preferences, never official rewards or model input.
"""
import argparse
import ast
from collections import Counter, defaultdict
from copy import deepcopy
import gzip
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from learning.bc_crop.agent import ROOT, CropAgent, Network, sha
from learning.bc_crop.features import CROPS, SHOPS, SHAPES
from learning.bc_crop.experiment import read, save, integrity
from learning.bc_crop.audit import stable
from decision.audit_market_teacher import observation


def extracted_score_function():
    raw = (ROOT/'bots/teacher_fertilizer_terminal_v2.py').read_bytes()
    tree = ast.parse(raw)
    functions = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == 'crop_score']
    if len(functions) != 1:
        raise ValueError('Unexpected frozen teacher score definition')
    namespace = {'__name__': 'offline_frozen_teacher'}
    exec(compile(raw, '<frozen-teacher>', 'exec'), namespace)
    exec(compile(ast.Module(body=[deepcopy(functions[0])], type_ignores=[]), '<extracted-teacher-score>', 'exec'), namespace)
    return namespace


def feature_scores(x, names, namespace):
    """Reconstruct score inputs from deployment features; never train on scores."""
    f = dict(zip(names, map(float, x)))
    params = {}
    for crop in CROPS:
        prefix = 'market.param.' + crop + '.'
        params[crop] = {key: f[prefix+key] for key in ('base', 'I0', 'T', 'below_target', 'above_target')}
        if not any(params[crop].values()) and not any(f[prefix+d+'.'+s] for d in ('below_func','above_func') for s in SHAPES):
            # Actual official observations omit private market parameters. In
            # those recorded contexts the frozen teacher uses its fixed defaults.
            # This is a reconstruction convention, not a claim that an explicit
            # all-zero custom parameter object is universally distinguishable.
            params.pop(crop)
            continue
        for direction in ('below_func', 'above_func'):
            active = [shape for shape in SHAPES if f[prefix+direction+'.'+shape] == 1]
            if len(active) != 1:
                raise ValueError('Market shape not represented uniquely by encoded features')
            params[crop][direction] = active[0]
    demand = {c: 1. for c in CROPS}
    for shop in SHOPS:
        products = namespace['SHOPS'][shop]
        for crop in products:
            if crop in demand:
                demand[crop] += f['town.'+shop] * f['config.turnsPerDay'] / f['config.townShopSellInterval'] * (2 if len(products) == 1 else 1)
    namespace.update(last_plant_age=int(f['planning.last_plant_age']), demand=demand, params=params,
                     market={'inventory': {c: f['market.stock.'+c] for c in CROPS}},
                     supply={c: f['planned_supply.'+c] for c in CROPS})
    return [float(namespace['crop_score'](crop)) for crop in CROPS]


def aggregate(rows):
    confusion = np.zeros((5, 5), dtype=np.int64)
    for row in rows:
        confusion[CROPS.index(row['teacher']), CROPS.index(row['prediction'])] += 1
    counts = confusion.sum(1)
    return dict(calls=len(rows), errors=int(confusion.sum()-np.trace(confusion)),
                accuracy=float(np.trace(confusion)/max(1, confusion.sum())),
                teacher_counts=dict(zip(CROPS, map(int, counts))), confusion=confusion.tolist(),
                recall={c: float(confusion[i, i]/counts[i]) if counts[i] else None for i, c in enumerate(CROPS)},
                exact_feature_vectors=len({r['feature_sha256'] for r in rows}),
                contiguous_decision_chains=len({r['chain'] for r in rows}),
                disagreement_by_outcome=dict(Counter(r['outcome'] for r in rows if r['teacher'] != r['prediction'])),
                mean_teacher_score_gap=float(np.mean([r['teacher_score_gap'] for r in rows])) if rows else None)


def run(source, out):
    plan = read(out/'analysis-plan.json')
    for name, digest in plan['allowed_inputs'].items():
        if sha(source/name) != digest:
            raise ValueError('Frozen diagnostic input changed: '+name)
    if (out/'existing-diagnostics.json').exists():
        raise FileExistsError('Do not overwrite diagnostic results')
    cfg = integrity(source)
    manifest = read(source/'data/manifest.json'); names = manifest['features']
    model = Network(source/'model/trained.npz', plan['allowed_inputs']['model/trained.npz'])
    rows = read(source/'collect-smoke-ledger.json') + read(source/'collect-ledger.json')
    rows += [r for r in read(source/'trained-smoke-ledger.json') if r['policy'] == 'trained']
    if len(rows) != 62 or any(r['seed'] in cfg['test_seeds'] for r in rows):
        raise ValueError('Unexpected diagnosis panel or formal-test contamination')
    ns = extracted_score_function()
    results = []; per_game = []; seen = defaultdict(set); raw_names = None
    source_map = []; start = time.perf_counter()
    for index, row in enumerate(rows):
        game = row['game']; side = row['side']
        replay_path = source/'games'/game/'replay.json.gz'; trace_path = source/'traces'/(game+'.jsonl.gz')
        if sha(replay_path) != row['replay_gzip_sha256'] or sha(trace_path) != row['trace_sha256']:
            raise ValueError('Raw diagnostic evidence hash mismatch')
        with gzip.open(replay_path, 'rt', encoding='utf-8') as file: replay = json.load(file)
        with gzip.open(trace_path, 'rt', encoding='utf-8') as file: traces = [json.loads(line) for line in file]
        is_student = row['policy'] == 'trained'
        split = 'student_development' if is_student else 'train' if row['seed'] in cfg['train_seeds'] else 'validation'
        agent = CropAgent('model' if is_student else 'teacher', source/'model/trained.npz' if is_student else None,
                          model.sha if is_student else None)
        original = agent.select; current = []; chains = {}; serial = 0
        def select(ctx):
            nonlocal serial
            scores = [float(ctx['crop_score'](c)) for c in CROPS]
            choice = original(ctx); record = agent.records[-1]
            x = np.asarray(record['x'], dtype=np.float32)
            prediction = CROPS[int(np.argmax(model.logits(x, names)))]
            reconstructed = feature_scores(x, names, ns)
            feature_label = max(CROPS, key=lambda c: (reconstructed[CROPS.index(c)], c))
            fingerprint = hashlib.sha256(x.tobytes()).hexdigest()
            seen[fingerprint].add(record['teacher'])
            previous = chains.get(record['unit'])
            target = tuple(record['target']); day = int(ctx['observation']['day'])
            if (previous is None or previous['step'] != record['step']-1 or previous['target'] != target
                    or previous['day'] != day or previous['terminal']):
                serial += 1; chain = f'{game}:{serial}'
            else:
                chain = previous['chain']
            chains[record['unit']] = dict(step=record['step'], target=target, day=day, chain=chain,
                                          terminal=record['outcome'] in ('plant_request', 'economic_reject'))
            current.append(dict(game=game, seed=row['seed'], opponent=row['opponent'], side=side, split=split,
                step=record['step'], day=day, unit=record['unit'], decision=record['decision'], chain=chain,
                teacher=record['teacher'], prediction=prediction, actual_choice=choice, outcome=record['outcome'],
                feature_sha256=fingerprint, teacher_score_gap=max(scores)-scores[CROPS.index(prediction)],
                score_reconstruction_max_error=max(abs(a-b) for a,b in zip(scores,reconstructed)),
                score_reconstruction_label_mismatch=feature_label != record['teacher'],
                z_outside_train_scale_3=int(np.count_nonzero(np.abs((x-model.mean)/model.scale)>3)),
                unit_request=record['unit_request'] if 'unit_request' in record else None))
            return choice
        agent.controller.__globals__['_bc_select'] = select
        if len(traces) != 719 or len(replay['steps']) != 720:
            raise ValueError('Incomplete source game')
        for step in range(719):
            action = agent(observation(replay['steps'][step], side), replay['configuration'])
            if action != replay['steps'][step+1][side]['action'] or stable(agent.last) != stable(traces[step]):
                raise ValueError('Decision reconstruction differs from source action/telemetry')
        for record in current:
            actual = traces[record['step']]['records'][record['decision']]
            record['unit_request'] = actual['unit_request']
            record['buy_seed_order_count'] = len(actual['buy_seed_orders'])
        per_game.append(dict(game=game, seed=row['seed'], opponent=row['opponent'], side=side, split=split, **aggregate(current)))
        results.extend(current)
        source_map.append(dict(game=game, replay_sha256=row['replay_gzip_sha256'], trace_sha256=row['trace_sha256']))
        if (index+1)%4 == 0 or index == len(rows)-1:
            print(f'Diagnostic reconstruction: {index+1}/{len(rows)}', flush=True)
    groups = {split: aggregate([r for r in results if r['split']==split]) for split in ('train','validation','student_development')}
    seed_groups = [dict(split=split, seed=seed, **aggregate([r for r in results if r['split']==split and r['seed']==seed]))
                   for split in groups for seed in sorted({r['seed'] for r in results if r['split']==split})]
    candidates = []
    for crop in CROPS:
        eligible = [r for r in results if r['split']=='train' and r['teacher']==crop and r['prediction']!=crop
                    and r['outcome'] in ('request_seed','plant_request')]
        counts = Counter(r['seed'] for r in eligible)
        values = [sum(r['teacher_score_gap'] for r in eligible if r['seed']==seed)/max(1,sum(r['split']=='train' and r['seed']==seed and r['outcome'] in ('request_seed','plant_request') for r in results))
                  for seed in cfg['train_seeds']]
        candidates.append(dict(crop=crop, eligible_disagreements=len(eligible), seeds_with_at_least_three=sum(n>=3 for n in counts.values()),
                               mean_seed_preference_gap=float(np.mean(values)), eligible_for_single_probe=sum(n>=3 for n in counts.values())>=2))
    ranked = sorted((c for c in candidates if c['eligible_for_single_probe']), key=lambda c:(-c['mean_seed_preference_gap'],CROPS.index(c['crop'])))
    report = dict(passed=True, source_games=62, new_games=0, training_runs=0, engine_reexecutions=0,
                  decision_reconstruction_turns=62*719, model_sha256=model.sha, crops=list(CROPS), groups=groups,
                  by_seed=seed_groups, by_game=per_game, source_hashes=source_map,
                  exact_feature_label_conflicts=sum(len(labels)>1 for labels in seen.values()),
                  score_reconstruction_label_mismatches=sum(r['score_reconstruction_label_mismatch'] for r in results),
                  score_reconstruction_max_error=max(r['score_reconstruction_max_error'] for r in results),
                  candidate_strata=candidates, selected_probe_teacher_crop=ranked[0]['crop'] if ranked else None,
                  limitations=['Teacher scores are heuristic preferences, not official rewards or causal regret',
                               'Exact-vector and contiguous unit/target/day chains are descriptive counts, not independent samples',
                               'Student development contains only one old training seed and one opponent, not broad on-policy coverage',
                               'No original formal-test per-game input used; no training or new game executed'],
                  wall_seconds=time.perf_counter()-start)
    with gzip.open(out/'diagnostic-decisions-private.jsonl.gz', 'wt', encoding='utf-8') as file:
        for row in results: file.write(json.dumps(row, separators=(',', ':'))+'\n')
    save(out/'existing-diagnostics.json', report)
    print(json.dumps({k:report[k] for k in ('source_games','new_games','exact_feature_label_conflicts','score_reconstruction_label_mismatches','score_reconstruction_max_error','selected_probe_teacher_crop')}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    run(args.source.resolve(), args.out.resolve())
