"""Explicit seed-role adapter to frozen three-arm economic/statistical consumers."""
from learning.task_dagger_once_v1 import statistics as previous

ARMS=previous.ARMS
BOOTSTRAP=previous.BOOTSTRAP
DEVELOPMENT=previous.DEVELOPMENT
HOLDOUT=previous.HOLDOUT


def validate_roles(cfg):
    roles={}
    for name,count in (('training_source_seeds',4),('known_screen_seeds',2),
                       ('validation_seeds',2),('test_seeds',8)):
        values=cfg[name]
        if (not isinstance(values,list) or len(values)!=count or
                any(type(v) is not int or not 0<=v<2**32 for v in values) or len(set(values))!=count):
            raise ValueError('Invalid explicit seed role: '+name)
        roles[name]=values
    if len({v for values in roles.values() for v in values})!=16:
        raise ValueError('Reused sources/screens and fresh official panels must be disjoint')
    if cfg.get('known_screens_independent') is not False:
        raise ValueError('Reused screens are known development, not independent evidence')
    return roles


def _legacy_view(cfg):
    roles=validate_roles(cfg)
    # The old consumer checks four training identities but never uses training
    # outcomes. This explicit view does not relabel them as fresh or run its
    # early collection gate. The actual new panels remain dev2/test8.
    view=dict(cfg,train_seeds=roles['training_source_seeds'],early_screen=dict(previous.EARLY))
    previous._configuration(view)
    return view


def development_gate(rows,cfg):
    result=previous.development_gate(rows,_legacy_view(cfg))
    result.update(schema='task-anchor-once-development-v1',reused_screen_outcomes_in_panel=False)
    return result


def holdout_statistics(rows,cfg):
    result=previous.holdout_statistics(rows,_legacy_view(cfg))
    result.update(schema='task-anchor-once-holdout-statistics-v1',reused_screen_outcomes_in_panel=False)
    return result
