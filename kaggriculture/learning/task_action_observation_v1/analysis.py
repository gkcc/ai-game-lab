"""Single-pass analysis of this task's 24 new development games only.

The older diagnostic supplies pure comparison semantics, never its run function
or its historical input directory. Completion and causation remain unknown.
"""
from pathlib import Path, PurePosixPath
import stat
import time

from learning.task_action_impact_diagnostic_v1.analysis import (
    ARMS, Meter, compact_replay, read_json, trace_lines, validate_trace,
    compare_callbacks, first_request_difference, descriptor, finish_pair, aggregate)


def raw_path(out, identity, kind):
    """Accept canonical task-local inputs, without opening their contents."""
    relative = identity['path']
    if (type(relative) is not str or not relative or '\\' in relative or ':' in relative
            or any(part in ('', '.', '..') for part in relative.split('/'))):
        raise ValueError('Noncanonical raw path')
    parts = PurePosixPath(relative).parts
    if len(parts) < 2 or parts[0] != {'replay': 'games', 'trace': 'traces'}[kind]:
        raise ValueError('Raw input is outside its registered directory')
    if (type(identity['size']) is not int or identity['size'] <= 0
            or type(identity['sha256']) is not str or len(identity['sha256']) != 64
            or any(c not in '0123456789abcdef' for c in identity['sha256'])):
        raise ValueError('Invalid raw identity')
    root = Path(out).resolve()
    path = root.joinpath(*parts)
    if not path.resolve().is_relative_to(root / parts[0]):
        raise ValueError('Raw path escapes the new task output')
    for part in (path, *path.parents):
        if part == root.parent:
            break
        try:
            info = part.lstat()
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(info.st_mode) or getattr(info, 'st_file_attributes', 0) & 0x400:
            raise ValueError('Linked raw input refused')
    return path


def validate_manifest(out, manifest):
    if manifest.get('schema') != 'task-action-observation-input-manifest-v1':
        raise ValueError('Wrong observation manifest schema')
    games, pairs = manifest['games'], manifest['pairs']
    if type(games) is not list or type(pairs) is not list or (len(games), len(pairs)) != (24, 8):
        raise ValueError('Expected 24 games and 8 triads')
    indexed, paths, keys = {}, {}, set()
    for game in games:
        if (type(game['seed']) is not int or type(game['side']) is not int
                or game['side'] not in (0, 1) or game['arm'] not in ARMS
                or type(game['opponent']) is not str or not game['opponent']
                or game['split'] != 'development' or type(game['key']) is not str
                or not game['key'] or game['id'] != game['key']):
            raise ValueError('Invalid development game identity')
        key = (game['seed'], game['opponent'], game['side'], game['arm'])
        if key in indexed or game['key'] in keys:
            raise ValueError('Duplicate game identity')
        indexed[key] = game
        keys.add(game['key'])
        for kind in ('replay', 'trace'):
            path = raw_path(out, game[kind], kind)
            if path in paths:
                raise ValueError('Duplicate raw input')
            paths[path] = game[kind]['path']
    seeds = {g['seed'] for g in games}
    opponents = {g['opponent'] for g in games}
    expected = {(seed, opponent, side) for seed in seeds for opponent in opponents for side in (0, 1)}
    if len(seeds) != 2 or len(opponents) != 2 or len(paths) != 48:
        raise ValueError('Expected the fixed 2 seed by 2 opponent design')
    seen = set()
    for pair in pairs:
        key = (pair['seed'], pair['opponent'], pair['side'])
        if key in seen or key not in expected or type(pair['side']) is not int:
            raise ValueError('Duplicate or unregistered triad')
        seen.add(key)
        ids = pair['game_ids']
        if set(ids) != set(ARMS) or any(indexed[(*key, arm)]['key'] != ids[arm] for arm in ARMS):
            raise ValueError('Triad game ids do not match their identities')
        first = pair['first_candidate_baseline_request_difference']
        if first is not None and (type(first) is not int or not 0 <= first < 719):
            raise ValueError('Invalid accepted first request step')
    if seen != expected:
        raise ValueError('Incomplete fixed triad design')
    return indexed, set(paths.values())


def observation_aggregate(rows):
    # Reuse numerical/group semantics, not the old 32-primary-pair screen.
    summary = aggregate(rows)
    summary.pop('screen')
    primary = [row for row in rows if row['reference'] == 'teacher_task']
    summary['observation_denominators'] = dict(
        paired_comparisons=len(rows), teacher_task_pairs=len(primary),
        baseline_pairs=sum(row['reference'] == 'baseline' for row in rows),
        primary_identifiable_pairs=sum(row['identifiable_chain_or_null_chain'] for row in primary),
        primary_unknown_pairs=sum(row['partition'] == 'unknown' for row in primary),
        completion_unknown_pairs=len(rows), unique_seeds=len({row['seed'] for row in rows}),
        unique_opponents=len({row['opponent'] for row in rows}), seats=sorted({row['side'] for row in rows}))
    summary['sample_role'] = 'development_observation_only'
    summary['screen_applied'] = False
    summary['confidence_intervals_estimated'] = False
    summary['promotion_authorized'] = False
    return summary


def campaign_binding(out, manifest):
    """Bind the new raw-file plan to its accepted campaign and frozen panel."""
    from .experiment import read, sha
    cfg, completed = read(out / 'registration.json'), read(out / 'campaign-result.json')
    identities = {name: sha(out / name) for name in ('input-manifest.json', 'ledger.json', 'audit.json')}
    if (completed.get('passed') is not True
            or completed.get('input_manifest_sha256') != identities['input-manifest.json']
            or manifest.get('ledger_sha256') != identities['ledger.json']
            or manifest.get('audit_sha256') != identities['audit.json']):
        raise ValueError('New campaign metadata identity mismatch')
    if ({game['seed'] for game in manifest['games']} != set(cfg['validation_seeds'])
            or {game['opponent'] for game in manifest['games']} != set(cfg['opponents'])):
        raise ValueError('Manifest panel differs from the frozen registration')
    return identities


def run(out):
    from .experiment import read, write_new
    started, cpu = time.monotonic(), time.process_time()
    out = Path(out).resolve()
    meter, rows, game_descriptors = Meter(compressed_limit=2**30, decoded_limit=12 * 2**30), [], []
    try:
        manifest = read(out / 'input-manifest.json')
        bound_metadata = campaign_binding(out, manifest)
        indexed, identities = validate_manifest(out, manifest)
        for spec in manifest['pairs']:
            selected = {arm: indexed[(spec['seed'], spec['opponent'], spec['side'], arm)] for arm in ARMS}
            compact = {}
            for arm in ARMS:
                game = selected[arm]
                compact[arm] = compact_replay(read_json(raw_path(out, game['replay'], 'replay'), game['replay'], meter), game)
            current = {}
            for reference in ('teacher_task', 'baseline'):
                first = first_request_difference(compact['candidate'], compact[reference], spec['side'])
                if reference == 'baseline' and first != spec['first_candidate_baseline_request_difference']:
                    raise ValueError('Accepted candidate-baseline first request step mismatch')
                current[reference] = dict(seed=spec['seed'], opponent=spec['opponent'], side=spec['side'],
                    reference=reference, first_request_difference_step=first, first_choice_difference=None,
                    first_request_turn_choice_linkage=None, matched_context_calls=0,
                    equal_context_choice_disagreement_turns=0, disagreement_same_complete_request_turns=0,
                    disagreement_different_complete_request_turns=0, unalignable_common_turns=0,
                    common_prefix_turns=0, common_prefix_active=True, first_unalignable_reason=None)
            streams = {arm: trace_lines(raw_path(out, selected[arm]['trace'], 'trace'), selected[arm]['trace'], meter)
                       for arm in ARMS}
            descriptors = {arm: descriptor() for arm in ARMS}
            try:
                for step in range(719):
                    traces = {}
                    for arm in ARMS:
                        try:
                            traces[arm] = next(streams[arm])
                        except StopIteration as exc:
                            raise ValueError('Trace ended before 719 turns') from exc
                        validate_trace(traces[arm], compact[arm], step, descriptors[arm])
                    for reference, pair in current.items():
                        a, b = compact['candidate'], compact[reference]
                        if not pair['common_prefix_active']:
                            continue
                        if a['physical'][step] != b['physical'][step]:
                            pair['common_prefix_active'] = False
                            if step == 0:
                                pair['unalignable_common_turns'] += 1
                                pair['first_unalignable_reason'] = 'initial_observation_difference'
                            continue
                        pair['common_prefix_turns'] += 1
                        comparison = compare_callbacks(traces['candidate']['records'], traces[reference]['records'])
                        pair['matched_context_calls'] += comparison['matched_context_calls']
                        if comparison['context_stop']:
                            pair['unalignable_common_turns'] += 1
                            pair['first_unalignable_reason'] = pair['first_unalignable_reason'] or comparison['context_stop']
                        if comparison['choice'] is not None:
                            if step == pair['first_request_difference_step']:
                                pair['first_request_turn_choice_linkage'] = dict(step=step, **comparison['choice'])
                            pair['equal_context_choice_disagreement_turns'] += 1
                            if pair['first_choice_difference'] is None:
                                pair['first_choice_difference'] = dict(step=step, **comparison['choice'])
                            same = a['requests'][step][spec['side']] == b['requests'][step][spec['side']]
                            pair['disagreement_same_complete_request_turns' if same else 'disagreement_different_complete_request_turns'] += 1
                for stream in streams.values():
                    try:
                        next(stream)
                    except StopIteration:
                        pass
                    else:
                        raise ValueError('Trace contains extra turns')
            finally:
                for stream in streams.values():
                    stream.close()
            for arm in ARMS:
                desc = descriptors[arm]
                desc.pop('previous')
                desc.pop('previous_day')
                game_descriptors.append(dict(game=selected[arm]['key'], seed=spec['seed'],
                    opponent=spec['opponent'], side=spec['side'], arm=arm, **desc,
                    completion_observability='unknown_in_saved_schema', geometry_is_descriptive_only=True))
            rows.extend(finish_pair(pair, compact['candidate'], compact[reference]) for reference, pair in current.items())
            del compact, traces, descriptors
        if len(rows) != 16 or len({(r['seed'], r['opponent'], r['side'], r['reference']) for r in rows}) != 16:
            raise ValueError('Wrong final comparison identities')
        if {r['path'] for r in meter.receipts} != identities or len(meter.receipts) != 48:
            raise ValueError('Not all fixed files completed exactly once')
        result = dict(schema='task-action-observation-result-v1', complete=True,
            **observation_aggregate(rows), input_games=24, triads=8, raw_files=48, terminal_checks=24,
            accepted_campaign_metadata_sha256=bound_metadata,
            accepted_first_request_checks=8, raw_reads=meter.snapshot(),
            analysis_wall_seconds=time.monotonic() - started, analysis_cpu_seconds=time.process_time() - cpu,
            gpu_seconds=0, neural_forwards=0, official_transitions=0, training_updates=0, new_games=0,
            decompressed_copies_on_disk=0)
        write_new(out / 'paired-rows.json', rows)
        write_new(out / 'game-descriptors.json', game_descriptors)
        write_new(out / 'raw-read-receipts.json', meter.receipts)
        write_new(out / 'analysis-result.json', result)
        return result
    except BaseException as exc:
        write_new(out / 'analysis-failure.local.json', dict(error_type=type(exc).__name__,
            message=str(exc)[:1000], **meter.snapshot(), completed_comparisons=len(rows),
            analysis_wall_seconds=time.monotonic() - started, analysis_cpu_seconds=time.process_time() - cpu,
            attempt_consumed=True, no_retry=True))
        raise
