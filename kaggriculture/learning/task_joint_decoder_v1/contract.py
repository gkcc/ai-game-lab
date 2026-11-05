"""Four consumed-source games and a thin view of the old frozen panel.

This changes the declared policy identity, never the registered models or
official rules.  It does not read private data or launch work at import.
"""
import math
import re


POLICY = 'joint_decoder'
CASES = tuple((role, alias, opponent, side)
              for role, alias, opponent in (('F01', 'S02', 'market'),
                                             ('F02', 'S03', 'round2'))
              for side in (0, 1))
PLAN_KEYS = frozenset(('game', 'seed_alias', 'seed', 'opponent', 'side',
                       'policy', 'split', 'parity_kind'))
JOB_KEYS = PLAN_KEYS | frozenset(('bot_sha256', 'rival_sha256', 'task_checkpoint_sha256'))
HEX = re.compile(r'[0-9a-f]{64}\Z')
GIB = 1024**3
MIB = 1024**2
LIMITS = dict(initial_available_bytes=6*GIB, between_available_bytes=6*GIB,
    running_available_bytes=4*GIB, readiness_samples=3, readiness_interval_seconds=10,
    readiness_timeout_seconds=300, max_rss_bytes=2*GIB, max_commit_bytes=3*GIB,
    job_wall_seconds=900, cleanup_seconds=10, poll_seconds=1,
    total_wall_seconds=2*60*60, disk_floor_bytes=4*GIB,
    per_game_reserve_bytes=160*MIB, per_stage_output_bytes=GIB,
    total_output_bytes=GIB, private_result_bytes=16*MIB,
    max_games=4, max_official_replays=4, max_attempts=1)


def require(value):
    if not value:
        raise ValueError('Joint decoder development contract differs')


def sources():
    return [dict(role=r, seed_alias=a, opponent=o, side=s) for r, a, o, s in CASES]


class JointPanel:
    """Parameter adapter accepted by the old isolated Worker and consumer.

    The original panel remains untouched. Both live and audit actors use its
    baseline task checkpoint; the new policy name identifies the decoder.
    """
    def __init__(self, base_panel):
        require(type(base_panel.seeds) is tuple and len(base_panel.seeds) == 2
                and all(type(x) is int and 0 <= x < 2**32 for x in base_panel.seeds)
                and len(set(base_panel.seeds)) == 2
                and type(base_panel.models) is tuple and len(base_panel.models) == 3
                and type(base_panel.rivals) is tuple and len(base_panel.rivals) == 2)
        self.base = base_panel
        self.crop = base_panel.crop
        self.worker_sha = base_panel.worker_sha

    def __repr__(self):
        return '<JointDecoderPanel>'

    def plans(self):
        return tuple(dict(game=f'dev-{alias}-{opponent}-{side}-{POLICY}',
            seed_alias=alias, seed=self.base.seeds[('S02', 'S03').index(alias)],
            opponent=opponent, side=side, policy=POLICY, split='dev', parity_kind=None)
            for _role, alias, opponent, side in CASES)

    def validate_plan(self, plan, *, job=False):
        require(type(job) is bool and type(plan) is dict
                and set(plan) == (JOB_KEYS if job else PLAN_KEYS))
        requested = {key: plan[key] for key in PLAN_KEYS}
        matches = [p for p in self.plans() if all(type(requested[k]) is type(v)
                   and requested[k] == v for k, v in p.items())]
        require(len(matches) == 1)
        if job:
            require(type(plan['bot_sha256']) is str and HEX.fullmatch(plan['bot_sha256']) is not None
                    and plan['rival_sha256'] == self.base.rivals[('market', 'round2').index(plan['opponent'])]
                    and plan['task_checkpoint_sha256'] == self.base.models[0][1])
        return dict(plan)

    def task_model(self, plan):
        self.validate_plan({key: plan[key] for key in PLAN_KEYS})
        return self.base.models[0]

    def worker_config(self):
        return self.base.worker_config()


FIELDS = ('own', 'opponent', 'margin', 'score')


def project(rows):
    """Complete four-case descriptive comparison; no statistical promotion.

    The caller has already authenticated complete games, official replays and
    the old control identities. Here a positive development signal also needs
    an actual request/state change and no hidden market-combination decline.
    """
    require(type(rows) is list and len(rows) == 4)
    deltas = []
    for row, (role, _alias, opponent, side) in zip(rows, CASES):
        require(type(row) is dict and row.get('role') == role
                and row.get('opponent') == opponent and type(row.get('side')) is int
                and row['side'] == side)
        for arm in ('baseline', 'candidate'):
            metrics = row.get(arm)
            require(type(metrics) is dict and set(metrics) == set(FIELDS)
                    and all(type(x) in (int, float) and math.isfinite(x) for x in metrics.values())
                    and metrics['score'] == (1. if metrics['margin'] > 0 else .5 if metrics['margin'] == 0 else 0.))
        delta = {k: row['candidate'][k]-row['baseline'][k] for k in FIELDS}
        require(all(math.isfinite(x) for x in delta.values())
                and delta['margin'] == delta['own']-delta['opponent'])
        require(type(row.get('decoder_changed_callbacks')) is int and row['decoder_changed_callbacks'] >= 0)
        for key in ('first_request_difference', 'first_state_difference'):
            value = row.get(key)
            require(value is None or type(value) is int and 0 <= value < (719 if key == 'first_request_difference' else 720))
        deltas.append(delta)
    def mean(items):
        return {k: math.fsum(v[k] for v in items)/len(items) for k in FIELDS}
    overall = mean(deltas)
    groups = {role: mean(deltas[i:i+2]) for role, i in (('F01', 0), ('F02', 2))}
    changed = sum(r['decoder_changed_callbacks'] for r in rows)
    requests = sum(r['first_request_difference'] is not None for r in rows)
    states = sum(r['first_state_difference'] is not None for r in rows)
    promising = (changed > 0 and requests > 0 and states > 0 and overall['own'] > 0
                 and overall['margin'] >= 0 and overall['score'] >= 0
                 and all(g['own'] >= 0 and g['margin'] >= 0 and g['score'] >= 0 for g in groups.values()))
    return dict(cases=4,seed_opponent_combinations=2,case_mean_delta=overall,
        seed_combination_mean_delta=mean(list(groups.values())),combination_means=groups,
        positive_zero_negative={k:dict(positive=sum(d[k]>0 for d in deltas),
            zero=sum(d[k]==0 for d in deltas),negative=sum(d[k]<0 for d in deltas)) for k in FIELDS},
        decoder_changed_callbacks=changed,games_with_request_difference=requests,
        games_with_state_difference=states,development_signal=promising,
        independent_evaluation=False,statistical_significance=False,promotion=False)
