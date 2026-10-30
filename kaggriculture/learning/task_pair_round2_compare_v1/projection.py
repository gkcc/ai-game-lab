"""Allowlisted descriptive projection for the frozen round2 development panel.

Callers must authenticate game/model identities, original receipts and saved
replays. Nothing in this module runs a game, model, training or inference.
"""
import hashlib
import json
import shutil
from statistics import mean

from learning.task_pair_dev_campaign_v1.projection import _decision_counts
from learning.task_pair_dev_metrics_v1.panel import _game
from learning.task_pair_dev_metrics_v1.panel import paired
from learning.task_pair_saved_compare_v1.analysis import compare as compare_saved


ALIASES = {'S02': 'S01', 'S03': 'S04'}
ARMS = ('baseline', 'control')
OPS = ('DIG', 'FERTILIZE', 'HARVEST', 'PLANT', 'WATER')
ECONOMY = ('own_net_profit', 'opponent_net_profit', 'margin', 'score',
           'own_sales_units', 'opponent_sales_units',
           'own_sales_revenue', 'opponent_sales_revenue')
STOP_REASONS = (None, 'frame_difference', 'input_difference', 'choice_difference',
                'callback_count_difference', 'joint_action_difference')
REQUEST_REASONS = {'observed': ('actor_request_difference',),
                   'unknown': ('frame_difference', 'opponent_request_difference',
                               'simultaneous_request_difference', 'terminal_frame_difference'),
                   'no_difference': ('all_requests_equal',)}


def require(value):
    if not value:
        raise ValueError('Invalid development result')


def _detail(value):
    require(type(value) is dict and value.get('scope') == 'strict_common_recorded_history'
            and type(value.get('trace_turns_per_game')) is int and value['trace_turns_per_game'] == 719
            and type(value.get('matched_selection_records')) is int and value['matched_selection_records'] >= 0
            and value.get('stop_reason') in STOP_REASONS)
    stop, step = value['stop_reason'], value.get('first_stop_step')
    require((stop is None and step is None) or
            (stop is not None and type(step) is int and 0 <= step < 719))
    choice = value.get('first_different_recorded_choice')
    if stop == 'choice_difference':
        require(type(choice) is dict and choice.get('baseline_op') in OPS and choice.get('control_op') in OPS)
        keys = ('same_unit', 'same_target', 'selected_unit_request_ordered_equal',
                'complete_actor_request_equal', 'complete_opponent_request_equal', 'both_model_callbacks')
        require(all(type(choice.get(key)) is bool for key in keys))
        choice = {key: choice[key] for key in ('baseline_op', 'control_op', *keys)}
    else:
        require(choice is None)
    prefix = value.get('complete_request_prefix')
    require(type(prefix) is dict and prefix.get('scope') == 'same_complete_history'
            and prefix.get('status') in REQUEST_REASONS
            and prefix.get('reason') in REQUEST_REASONS[prefix['status']])
    pstep = prefix.get('step')
    require((prefix['status'] == 'observed' and type(pstep) is int and 0 <= pstep < 719)
            or (prefix['status'] != 'observed' and pstep is None))
    return dict(scope='strict_common_recorded_history', trace_turns_per_game=719,
                matched_selection_records=value['matched_selection_records'], stop_reason=stop,
                first_stop_step=step, first_different_recorded_choice=choice,
                complete_request_prefix={key: prefix[key] for key in ('scope', 'status', 'reason', 'step')})


def project(*, rows, seeds, comparisons, details):
    """Project exactly eight accepted games and four authenticated pairs.

    The seed values in ``seeds`` and raw identities in the arguments never
    appear in the returned object. The gate is an exploratory eligibility rule,
    not a statistical test, teacher promotion or permission to spend a holdout.
    """
    require(type(seeds) is dict and set(seeds) == set(ALIASES)
            and all(type(seed) is int and 0 <= seed < 2**32 for seed in seeds.values())
            and len(set(seeds.values())) == 2 and type(rows) is list and len(rows) == 8
            and type(comparisons) is list and len(comparisons) == 4 and type(details) is dict)
    games, identities = {}, {'baseline': set(), 'control': set(), 'crop': set()}
    for row in rows:
        require(type(row) is dict and row.get('schema') == 'task-pair-dev-game-metrics-v1')
        alias, side, arm = row.get('seed_alias'), row.get('side'), row.get('arm')
        require(alias in ALIASES and type(side) is int and side in (0, 1) and arm in ARMS
                and type(row.get('seed')) is int and row['seed'] == seeds[alias] and row.get('opponent') == 'round2'
                and row.get('game') == f'dev-{alias}-round2-{side}-{arm}')
        key = alias, side, arm
        require(key not in games)
        metrics = _game(row)
        decisions = _decision_counts(row['result'], side)
        require(decisions['by_source']['teacher'] == 0)
        games[key] = dict(**{name: metrics[name] for name in ECONOMY}, decision_counts=decisions)
        for role, field in ((arm, 'task_checkpoint_sha256'), ('crop', 'crop_checkpoint_sha256')):
            value = row.get(field)
            require(type(value) is str and len(value) == 64 and all(c in '0123456789abcdef' for c in value))
            identities[role].add(value)
    expected = {(alias, side, arm) for alias in ALIASES for side in (0, 1) for arm in ARMS}
    require(set(games) == expected and all(len(values) == 1 for values in identities.values())
            and identities['baseline'] != identities['control'])
    indexed = {}
    for item in comparisons:
        require(type(item) is dict and item.get('reference') == 'baseline' and item.get('candidate') == 'control')
        alias, side = item.get('seed_alias'), item.get('side')
        require(alias in ALIASES and type(side) is int and side in (0, 1)
                and item.get('seed') == seeds[alias] and item.get('opponent') == 'round2')
        result = item.get('result')
        require(type(result) is dict and result.get('comparison_complete') is True
                and type(result.get('matched_calls')) is int and result['matched_calls'] >= 0
                and result.get('stop_reason') in STOP_REASONS)
        key = alias, side
        require(key not in indexed)
        indexed[key] = result
    require(set(indexed) == {(alias, side) for alias in ALIASES for side in (0, 1)}
            and set(details) == set(indexed))
    pairs = []
    for alias in ALIASES:
        for side in (0, 1):
            ref, candidate = games[alias, side, 'baseline'], games[alias, side, 'control']
            detail = _detail(details[alias, side])
            old = indexed[alias, side]
            require(detail['matched_selection_records'] == old['matched_calls']
                    and detail['stop_reason'] == old['stop_reason']
                    and detail['matched_selection_records'] <= ref['decision_counts']['total']
                    and detail['matched_selection_records'] <= candidate['decision_counts']['total'])
            if detail['first_different_recorded_choice'] is not None:
                require(detail['first_different_recorded_choice']['both_model_callbacks'] is True)
            pairs.append(dict(seed_alias=ALIASES[alias], side=side, opponent='round2',
                              baseline=ref, control=candidate,
                              delta={key: candidate[key] - ref[key] for key in ECONOMY},
                              comparison=detail))
    summary = {key: mean(pair['delta'][key] for pair in pairs) for key in ECONOMY}
    seed_gate = {}
    for alias in ALIASES.values():
        subset = [pair for pair in pairs if pair['seed_alias'] == alias]
        base = mean(pair['baseline']['own_net_profit'] for pair in subset)
        gain = mean(pair['delta']['own_net_profit'] for pair in subset)
        seed_gate[alias] = dict(mean_own_delta=gain, baseline_own_mean=base,
            own_loss_within_one_percent=(gain >= -0.01 * base if base > 0 else None),
            actor_request_difference_observed=any(
                pair['comparison']['complete_request_prefix']['status'] == 'observed' for pair in subset))
    checks = dict(mean_own_profit_positive=summary['own_net_profit'] > 0,
                  mean_margin_nonnegative=summary['margin'] >= 0,
                  mean_score_nonnegative=summary['score'] >= 0,
                  each_seed_own_loss_within_one_percent=all(
                      seed_gate[s]['own_loss_within_one_percent'] is True for s in seed_gate),
                  each_seed_actor_request_difference_observed=all(
                      seed_gate[s]['actor_request_difference_observed'] for s in seed_gate))
    return dict(schema='task-pair-round2-development-result-v1', scope='consumed_development_only',
                games=8, seed_clusters=2, pairs=pairs, mean_delta=summary, seed_checks=seed_gate,
                exploration_gate=dict(checks=checks, passed=all(checks.values()),
                                      interpretation='Exploratory only; not promotion or proof of improvement.'),
                new_training_updates=0, new_seed_generation=False,
                independent_improvement=False, statistical_significance='unknown', promotion=False)


def _json(value):
    return (json.dumps(value, ensure_ascii=False, separators=(',', ':'), allow_nan=False) + '\n').encode()


def _same(left, right):
    # Normalize integer keys in the existing paired summary, exactly as its
    # persisted JSON does. Comparison is in memory and no identity is printed.
    return json.loads(_json(left)) == json.loads(_json(right))


def save_result(ctx, driver, payload):
    """Recheck accepted receipts, compare all four saved pairs and write once.

    The outer authenticated caller owns the stage, Job, exclusive markers,
    identity and execution budget. The stored private result retains receipt
    identities; the separate public projection contains no raw seed or hash.
    Guard checks during replay parsing are sampled, not an async memory cap.
    """
    require(type(payload) is dict and payload.get('attempted') == 8
            and payload.get('official_replay_passes') == 8 and payload.get('training') is False
            and payload.get('promotion') is False and payload.get('independent_improvement') is False
            and type(payload.get('rows')) is list and len(payload['rows']) == 8
            and type(payload.get('comparisons')) is list and len(payload['comparisons']) == 4)
    modules, stage = ctx['modules'], ctx['stage']
    dmod, acceptance, child = (modules[k] for k in ('driver', 'acceptance', 'child'))
    check = ctx['check']
    require(callable(check) and check() is True and len(ctx['plans']) == 8)
    rows, ledger = [], []
    for plan, prior in zip(ctx['plans'], payload['rows']):
        require(check() is True)
        game, bundle = plan['game'], driver.read_bundle(plan)
        supervision_raw = dmod._read(stage, stage / 'job-results' / (game + '.supervision.local.json'),
                                     acceptance.MAX_RECEIPT_BYTES)
        accepted_raw = dmod._read(stage, stage / 'job-results' / (game + '.acceptance.local.json'),
                                  acceptance.MAX_RECEIPT_BYTES)
        supervision = acceptance._json(supervision_raw, acceptance.MAX_RECEIPT_BYTES)
        accepted = acceptance._json(accepted_raw, acceptance.MAX_RECEIPT_BYTES)
        require(supervision.get('schema') == 'task-pair-dev-job-supervision-v1'
                and supervision.get('game') == game and supervision.get('retry_allowed') is False
                and accepted.get('schema') == 'task-pair-dev-job-acceptance-v1'
                and accepted.get('game') == game and accepted.get('passed') is True
                and accepted.get('retry_allowed') is False
                and accepted.get('supervision_sha256') == hashlib.sha256(supervision_raw).hexdigest()
                and accepted.get('job_input_sha256') == dmod._hash(
                    stage, stage / 'job-inputs' / (game + '.json'), child.worker.MAX_JOB_BYTES)
                and accepted.get('receipt_sha256') == {k: hashlib.sha256(v).hexdigest() for k, v in bundle.items()})
        row = acceptance.accept(ctx['manifest'].panel, plan, bundle, supervision.get('supervision'),
                                driver.verify_files)
        require(_same(row, prior))
        markers = {}
        for kind in ('permit', 'started', 'consumed'):
            relative, expected = ctx['marker'](game, kind)
            raw = dmod._read(stage, stage / relative, acceptance.MAX_RECEIPT_BYTES)
            require(raw == _json(expected))
            markers[kind] = hashlib.sha256(raw).hexdigest()
        rows.append(row)
        ledger.append(dict(game=game, supervision_sha256=hashlib.sha256(supervision_raw).hexdigest(),
                           acceptance_sha256=hashlib.sha256(accepted_raw).hexdigest(), marker_sha256=markers))
    seeds = dict(zip(ALIASES, ctx['manifest'].panel.seeds))
    comparisons, details = [], {}
    indexed = {(row['seed_alias'], row['side'], row['arm']): row for row in rows}
    require(len(indexed) == 8)
    for item in payload['comparisons']:
        alias, side = item.get('seed_alias'), item.get('side')
        require(alias in ALIASES and type(side) is int and side in (0, 1) and (alias, side) not in details)
        left, right = indexed[alias, side, 'baseline'], indexed[alias, side, 'control']
        path = dmod._comparison_file(stage, left['game'], 'control')
        raw = dmod._read(stage, path, acceptance.MAX_RECEIPT_BYTES)
        receipt = acceptance._json(raw, acceptance.MAX_RECEIPT_BYTES)
        require(receipt.get('schema') == 'task-pair-dev-comparison-v1'
                and receipt.get('reference_game') == left['game'] and receipt.get('candidate_game') == right['game']
                and receipt.get('reference_replay_sha256') == left['replay_sha256']
                and receipt.get('candidate_replay_sha256') == right['replay_sha256']
                and receipt.get('reference_trace_sha256') == left['trace_sha256']
                and receipt.get('candidate_trace_sha256') == right['trace_sha256']
                and _same(receipt.get('comparison'), item))
        # Full source/asset checks are done before and after each pair. A
        # lightweight guard avoids hashing hundreds of assets on every line.
        def sample():
            import psutil
            require(ctx['time_check']() is True)
            available = psutil.virtual_memory().available
            require(type(available) is int and available >= 0)
            ctx['last_available_bytes'] = available
            if available < ctx['limits']['between_available_bytes']:
                raise MemoryError('Comparison memory floor reached')
            return True
        require(check() is True and driver.preflight_comparison(ctx['limits'], 1.0) is True and sample())
        ref = dmod._replay(stage, stage / 'games' / left['game'] / 'replay.json.gz', left['replay_sha256'])
        require(sample() and check() is True)
        new = dmod._replay(stage, stage / 'games' / right['game'] / 'replay.json.gz', right['replay_sha256'])
        require(sample() and check() is True)
        lt = dmod._trace(stage, stage / 'traces' / (left['game'] + '.jsonl.gz'), left['trace_sha256'])
        rt = dmod._trace(stage, stage / 'traces' / (right['game'] + '.jsonl.gz'), right['trace_sha256'])
        try:
            details[alias, side] = compare_saved(ref, new, lt, rt, seed=seeds[alias], side=side, check=sample)
        finally:
            lt.close()
            rt.close()
        del ref, new
        require(check() is True)
        comparisons.append(item)
    expected = paired(rows, seeds=list(seeds.values()), opponents=['round2'],
                      reference='baseline', candidate='control', comparisons=comparisons)
    require(_same(payload['paired'], expected))
    public = project(rows=rows, seeds=seeds, comparisons=comparisons, details=details)
    private = dict(schema='task-pair-round2-compare-private-result-v1', passed=True, games=8,
                   official_replay_passes=8, comparisons=4, seeds=seeds, ledger=ledger,
                   paired=expected, training=False, independent_improvement=False, promotion=False)
    size = len(_json(public)) + len(_json(private))
    used, free = dmod._owned_size(stage), shutil.disk_usage(stage).free
    require(size <= dmod.MAX_SUMMARY and used + size <= ctx['limits']['total_output_bytes']
            and free >= ctx['limits']['disk_free_floor_bytes'] + size and check() is True)
    dmod._write_new(stage, stage / 'comparison-result.local.json', private)
    dmod._write_new(stage, stage / 'projected-summary.local.json', public)
    require(check() is True)
    return True
