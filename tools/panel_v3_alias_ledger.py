"""Project a complete panel ledger to public aliases, then recompute statistics.

This pure helper never reads an experiment or writes files. The caller must
first verify the complete official panel, extra replay and outer cleanup.
"""
import copy
import math


FIELDS = (
    'game', 'seed_alias', 'opponent', 'side', 'policy', 'split', 'parity_kind',
    'passed', 'states', 'action_turns', 'status', 'agent_errors', 'bank',
    'opponent_bank', 'initial_bank', 'opponent_initial_bank', 'net_profit',
    'opponent_net_profit', 'margin', 'score',
)
ALIASES = ('S01', 'S02', 'S03', 'S04')
OPPONENTS = ('round2', 'market', 'teacher')
POLICIES = ('merged', 'unweighted', 'teacher_task')
ECONOMIC = ('bank', 'opponent_bank', 'initial_bank', 'opponent_initial_bank',
            'net_profit', 'opponent_net_profit', 'margin', 'score')
# These numbers are ephemeral identifiers for calling the frozen evaluator;
# they are never passed to a game and never included in the output ledger.
SYNTHETIC_SEEDS = (101, 102, 103, 104)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def require_non_economic_completion(games, extra, verification, coordinator, outer, launcher_exit):
    """Check necessary self-reported completion flags before economic reads.

    A caller must separately bind and validate the actual official panel and
    extra replay through the frozen v3 context before projecting any rows.
    """
    _require(all(isinstance(x, dict) for x in (games, extra, verification, coordinator, outer)),
             'Missing completion receipt')
    _require(games.get('passed') is True and type(games.get('complete_games')) is int
             and games['complete_games'] == 72
             and type(games.get('live_official_transitions')) is int
             and games['live_official_transitions'] == 72 * 719
             and type(games.get('primary_replay_transitions')) is int
             and games['primary_replay_transitions'] == 72 * 719
             and games.get('retry_allowed') is False, 'Panel not complete')
    _require(extra.get('passed') is True and extra.get('same_official_engine') is True
             and extra.get('independent_rules_implementation') is False
             and extra.get('retry_allowed') is False, 'Extra replay not complete')
    job = extra.get('supervision')
    _require(isinstance(job, dict)
             and all(job.get(key) is True for key in ('passed', 'assigned', 'resumed', 'launcher_reaped',
                                                       'job_members_zero'))
             and all(key in job and job[key] is None for key in ('error_kind', 'cleanup_error_kind', 'stop_reason'))
             and type(job.get('exit_code')) is int and job['exit_code'] == 0, 'Extra replay job not clean')
    _require(verification.get('passed') is True and type(verification.get('official_transitions')) is int
             and verification['official_transitions'] == 719
             and type(verification.get('sample_count')) is int and verification['sample_count'] == 20
             and type(verification.get('matched')) is int and verification['matched'] == 20
             and all(verification.get(key) is True for key in ('both_public_and_private_states_match_each_step',
                 'inventory_order_checked_each_step', 'final_rewards_match', 'terminal_status_done',
                 'no_model_inference', 'no_new_games')) and verification.get('retry_allowed') is False,
             'Extra replay verification incomplete')
    _require(coordinator.get('passed') is True and coordinator.get('economic_evaluation_completed') is True
             and coordinator.get('retry_allowed') is False, 'Coordinator did not finish')
    _require(type(launcher_exit) is int and launcher_exit == 0 and outer.get('passed') is True
             and all(outer.get(key) is True for key in ('child_reaped', 'job_members_zero', 'handles_closed',
                                                         'cleanup_known', 'cache_prefix_absent'))
             and type(outer.get('exit_code')) is int and outer['exit_code'] == 0
             and all(key in outer and outer[key] is None for key in ('error_kind', 'stop_reason'))
             and outer.get('scratch_retained') is False and outer.get('retry_allowed') is False,
             'Outer process or cleanup not confirmed')


def require_completion_receipts(games, extra, verification, evaluation, coordinator, outer, launcher_exit):
    """Also screen the final statistics container; source validation remains required."""
    require_non_economic_completion(games, extra, verification, coordinator, outer, launcher_exit)
    _require(isinstance(evaluation, dict) and evaluation.get('passed') is True
             and type(evaluation.get('seed_cluster_count')) is int
             and evaluation['seed_cluster_count'] == 4 and evaluation.get('automatic_promotion') is False
             and evaluation.get('no_independent_rules_engine') is True
             and evaluation.get('retry_allowed') is False
             and all(isinstance(evaluation.get(key), dict) for key in ('pairs', 'seed_clusters', 'per_opponent',
                 'overall', 'bootstrap_intervals', 'gates'))
             and all(isinstance(evaluation.get(key), list) for key in ('summaries', 'gate_failures'))
             and all(type(evaluation.get(key)) is bool for key in ('exploratory_gate_passed',
                                                                  'exploratory_positive_signal')),
             'Final evaluation incomplete')


def project(rows, evaluate):
    """Return a 72-row allowlisted ledger and its frozen evaluator output."""
    _require(isinstance(rows, list) and len(rows) == 72 and callable(evaluate), 'Incomplete panel')
    expected = {(alias, opponent, side, policy) for alias in ALIASES
                for opponent in OPPONENTS for side in (0, 1) for policy in POLICIES}
    seen, public, temporary = set(), [], []
    for row in rows:
        _require(isinstance(row, dict) and all(key in row for key in FIELDS), 'Incomplete row')
        alias, opponent, side, policy = (row[key] for key in ('seed_alias', 'opponent', 'side', 'policy'))
        _require(isinstance(alias, str) and alias in ALIASES and isinstance(opponent, str)
                 and opponent in OPPONENTS and type(side) is int and side in (0, 1)
                 and isinstance(policy, str) and policy in POLICIES, 'Invalid panel identity')
        identity = (alias, opponent, side, policy)
        _require(identity not in seen and identity in expected, 'Duplicate panel identity')
        seen.add(identity)
        _require(row['game'] == f'holdout-{alias}-{opponent}-{side}-{policy}'
                 and row['split'] == 'holdout' and row['parity_kind'] is None, 'Invalid game identity')
        _require(row['passed'] is True and type(row['states']) is int and row['states'] == 720
                 and type(row['action_turns']) is int and row['action_turns'] == 719
                 and row['status'] == ['DONE', 'DONE']
                 and type(row['agent_errors']) is int and row['agent_errors'] == 0, 'Incomplete game')
        _require(all(type(row[key]) in (int, float) and math.isfinite(row[key]) for key in ECONOMIC),
                 'Invalid economic field')
        projected = {key: copy.deepcopy(row[key]) for key in FIELDS}
        public.append(projected)
        sample = dict(projected)
        sample['seed'] = SYNTHETIC_SEEDS[ALIASES.index(alias)]
        temporary.append(sample)
    _require(seen == expected, 'Panel identities are incomplete')
    # The frozen evaluator checks accounting, score, pairing and whole-seed
    # bootstrap. Synthetic values are local labels, not game randomness.
    statistics = evaluate(temporary, list(SYNTHETIC_SEEDS))
    _require(isinstance(statistics, dict), 'Invalid statistics result')
    return public, statistics
