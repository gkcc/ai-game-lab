"""Preregistered eight-whole-seed paired holdout analysis; no model selection.

Consumes the existing official economic/control audit output. Entry repetitions
are rejected, not silently included or dropped. This module never fits models,
runs matches, changes a checkpoint, or changes the default Bot.
"""
import hashlib
import math
from statistics import mean

import numpy as np

from learning.bc_crop_economic_diagnostic.analysis import (
    FAMILIES, METRICS, OPPONENTS, number, quantity, require,
)


def _check_row(row, cfg):
    require(row.get('stage') == 'holdout' and not row.get('is_entry', False),
            'Non-holdout or entry row is not admissible')
    require(isinstance(row.get('game'), str) and row['game']
            and not row['game'].startswith('entry'), 'Invalid/entry game identity')
    require(row.get('passed') is True and row.get('policy') == 'trained', 'Unverified worker role')
    family = row.get('model_family')
    require(family in FAMILIES, 'Unknown model family')
    require(type(row.get('seed')) is int and type(row.get('side')) is int,
            'Invalid paired coordinate type')
    require(row.get('checkpoint_sha256') == cfg['model_sha256'][family], 'Checkpoint identity mismatch')
    control, economic_audit = row.get('control_audit', {}), row.get('economic_audit', {})
    require(control.get('checkpoint_sha256') == row['checkpoint_sha256'], 'Control checkpoint mismatch')
    for key in ('game', 'seed', 'side', 'opponent', 'policy'):
        require(control.get(key) == row[key], 'Control source coordinate mismatch: ' + key)
    require(control.get('reproduced_actor_actions') == 719
            and control.get('reproduced_rival_actions') == 719
            and control.get('raw_decision_traces_reproduced') is True,
            'Full source/action/trace reproduction required')
    require(all(control.get(k) == 0 for k in ('fallbacks', 'load_errors', 'inference_errors')),
            'Control execution errors')
    require(economic_audit.get('passed') is True and economic_audit == control.get('audit'),
            'Both matching official replay audits required')
    require(economic_audit.get('official_transitions_reexecuted') == 719
            and economic_audit.get('both_public_and_private_states_match_each_step') is True
            and economic_audit.get('inventory_order_checked_each_step') is True,
            'Incomplete official ordered-state certificate')
    if 'engine_sha256' in cfg:
        require(economic_audit.get('engine_sha256') == cfg['engine_sha256'], 'Official rules identity mismatch')
    require(row.get('official_replay_passes') == 2
            and row.get('official_transitions_reexecuted') == 1438,
            'Incorrect replay-pass accounting')
    for metric in METRICS:
        number(row[metric], metric)
    economic = row.get('economic', {})
    own, rival = economic.get('own', {}), economic.get('opponent', {})
    for player in (own, rival):
        require(player.get('cash_reconciled') is True, 'Unreconciled player cash')
        start = number(player['initial_cash'], 'initial_cash')
        end = number(player['final_cash'], 'final_cash')
        require(player['net_profit'] == end - start, 'Player cash/net profit mismatch')
        require(number(player['market_cash_delta'], 'market cash')
                + number(player['unit_cash_delta'], 'unit cash') == end - start,
                'Cash effects mismatch')
    require(row['net_profit'] == own['net_profit']
            and row['opponent_net_profit'] == rival['net_profit'], 'Net profit summary mismatch')
    require(row['margin'] == own['final_cash'] - rival['final_cash'], 'Margin summary mismatch')
    expected_score = 1. if row['margin'] > 0 else .5 if row['margin'] == 0 else 0.
    require(row['match_score'] == expected_score, 'Match score summary mismatch')
    rewards = [None, None]
    require(row['side'] in (0, 1), 'Invalid seat')
    rewards[row['side']], rewards[1-row['side']] = own['final_cash'], rival['final_cash']
    require(economic_audit.get('final_rewards') == rewards, 'Reward/cash summary mismatch')
    for metric in METRICS[4:10]:
        require(row[metric] == own[metric] and row['opponent_' + metric] == rival[metric],
                'Execution aggregate mismatch: ' + metric)
        if metric != 'sales_revenue':
            quantity(row[metric], metric)
            quantity(row['opponent_' + metric], 'opponent_' + metric)
        else:
            require(row[metric] >= 0 and row['opponent_' + metric] >= 0, 'Negative sales revenue')


def summarize(rows, cfg):
    """Exactly 96 audited rows -> 48 pairs -> eight seed means -> one CI."""
    seeds = cfg['test_seeds']
    require(len(seeds) == 8 and all(type(s) is int and 0 <= s < 2**32 for s in seeds)
            and len(set(seeds)) == 8, 'Exactly eight distinct test seeds required')
    require(len(cfg['opponents']) == 3 and set(cfg['opponents']) == set(OPPONENTS), 'Wrong opponent panel')
    require(len(cfg['seats']) == 2 and set(cfg['seats']) == {0, 1}
            and all(type(s) is int for s in cfg['seats']), 'Wrong seat panel')
    require(set(cfg['model_sha256']) == set(FAMILIES), 'Wrong checkpoint map')
    require(all(isinstance(s, str) and len(s) == 64 and all(c in '0123456789abcdef' for c in s)
                for s in cfg['model_sha256'].values()), 'Malformed checkpoint SHA256')
    require(len(set(cfg['model_sha256'].values())) == 2, 'Distinct frozen model identities required')
    bootstrap = cfg['bootstrap']
    expected_bootstrap = dict(draws=20000, seed=2026102326, unit='whole seed',
                              method='percentile', confidence=.95, quantile_method='linear')
    require(bootstrap == expected_bootstrap and type(bootstrap.get('draws')) is int
            and type(bootstrap.get('seed')) is int, 'Preregistered bootstrap configuration mismatch')
    require(len(rows) == 96, 'Exactly 96 holdout rows required')
    expected = {(f, s, o, side) for f in FAMILIES for s in seeds for o in OPPONENTS for side in (0, 1)}
    index, games = {}, set()
    for row in rows:
        _check_row(row, cfg)
        key = (row['model_family'], row['seed'], row['opponent'], row['side'])
        require(key in expected and key not in index, 'Duplicate or unexpected panel coordinate')
        game = (row['model_family'], row['game'])
        require(game not in games, 'Duplicate family/game identity')
        games.add(game)
        index[key] = row
    require(set(index) == expected, 'Incomplete paired panel')
    pairs = []
    for seed in seeds:
        for opponent in OPPONENTS:
            for side in (0, 1):
                baseline, candidate = (index[f, seed, opponent, side] for f in FAMILIES)
                pairs.append(dict(seed=seed, opponent=opponent, side=side,
                    original_student_game=baseline['game'], fixed_epoch6_game=candidate['game'],
                    original_student={m: baseline[m] for m in METRICS},
                    fixed_epoch6={m: candidate[m] for m in METRICS},
                    delta={m: candidate[m] - baseline[m] for m in METRICS}))
    def aggregate(part):
        return {m: mean(row['delta'][m] for row in part) for m in METRICS}
    by_seed = {str(s): aggregate([p for p in pairs if p['seed'] == s]) for s in seeds}
    by_opponent = {o: aggregate([p for p in pairs if p['opponent'] == o]) for o in OPPONENTS}
    overall = {m: mean(s[m] for s in by_seed.values()) for m in METRICS}
    by_model = {}
    for family in FAMILIES:
        part = [r for r in rows if r['model_family'] == family]
        by_model[family] = dict(games=48, means={m: mean(r[m] for r in part) for m in METRICS},
            wins=sum(r['margin'] > 0 for r in part), draws=sum(r['margin'] == 0 for r in part),
            losses=sum(r['margin'] < 0 for r in part))
    baseline_profit = by_model['original_student']['means']['net_profit']
    baseline_positive = baseline_profit > 0
    seed_means = np.asarray([by_seed[str(s)]['net_profit'] for s in seeds], dtype=np.float64)
    # One draw matrix, preserving every opponent and both seats inside a seed.
    indices = np.random.default_rng(bootstrap['seed']).integers(0, 8, size=(20000, 8), dtype=np.int64)
    sampled_means = seed_means[indices].mean(axis=1)
    low, high = np.quantile(sampled_means, [.025, .975], method='linear')
    require(math.isfinite(float(low)) and math.isfinite(float(high)), 'Non-finite bootstrap interval')
    ci = dict(lower=float(low), upper=float(high))
    threshold = .01 * baseline_profit if baseline_positive else None
    gates = dict(original_student_mean_profit_positive=baseline_positive,
                 net_profit_ci95_lower_positive=ci['lower'] > 0,
                 net_profit_at_least_one_percent=baseline_positive and overall['net_profit'] >= threshold,
                 overall_margin_nonnegative=overall['margin'] >= 0,
                 overall_score_nonnegative=overall['match_score'] >= 0,
                 market_profit_nonnegative=by_opponent['market']['net_profit'] >= 0,
                 market_margin_nonnegative=by_opponent['market']['margin'] >= 0,
                 market_score_nonnegative=by_opponent['market']['match_score'] >= 0)
    loso = []
    for excluded in seeds:
        remaining = [s for s in seeds if s != excluded]
        loso.append(dict(excluded_seed=excluded, remaining_seed_clusters=7,
                         delta={m: mean(by_seed[str(s)][m] for s in remaining) for m in METRICS}))
    return dict(evaluation_role='independent-holdout', development_only=False,
                holdout_games=96, entry_repeats_in_statistics=0, whole_seed_clusters=8,
                paired_games=48, training_runs=0, default_changed=False, automatic_deployment=False,
                learned_student_gate_passed=all(gates.values()), gate_checks=gates,
                original_student_mean_net_profit=baseline_profit,
                practical_increment_threshold=threshold,
                relative_net_profit_increment=overall['net_profit']/baseline_profit if baseline_positive else None,
                fixed_epoch6_minus_original_student=overall, net_profit_ci95=ci,
                bootstrap={**bootstrap, 'seed_order': list(seeds), 'sampled_clusters_per_draw': 8,
                           'pairs_per_seed': 6, 'row_count_is_not_sample_size': True,
                           'index_matrix_shape': [20000, 8], 'index_matrix_hash_dtype': 'little-endian int64',
                           'index_matrix_sha256': hashlib.sha256(indices.astype('<i8').tobytes(order='C')).hexdigest()},
                by_seed=by_seed, by_opponent=by_opponent, by_model=by_model, paired=pairs,
                leave_one_seed_out_descriptive=loso,
                limitations=['Eight seed clusters, not 96 independent samples.',
                             'Only net profit receives a confidence interval; margin/score guards use point estimates.',
                             'Market subgroup point-estimate guards are not a statistical noninferiority guarantee.',
                             'Nonpositive reference profit is a failed practical gate, not an invalid match; relative improvement is undefined.',
                             'Leave-one-seed-out means are descriptive and never select or remove a seed.',
                             'Original fullbatch offline failure remains unchanged; no teacher-superiority claim.',
                             'Inference is conditional on these frozen opponents and the registered seed distribution.'])
