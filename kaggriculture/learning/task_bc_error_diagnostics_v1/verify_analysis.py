"""Reject the identified empty-compact and duplicated-turn acceptance bugs."""
import time
from .analysis import validate_compact,validate_turns


def rejected(fn):
    try:fn()
    except (AssertionError,ValueError):return
    raise AssertionError('Invalid coverage accepted')


def run():
    deadline=time.monotonic()+30
    row=dict(game='synthetic',seed=1,side=0,opponent='v1',policy='trained')
    base=dict(game='synthetic',seed=1,side=0,opponent='v1',origin='trained')
    expected=dict(factual_groups=1,factual_candidates=2,shadow_groups=1,turns=719)
    compact=dict(**base,step=0,decision=0,time_third=0,candidate_count=2)
    assert validate_compact([compact],row,expected,deadline)['factual_groups']==1
    rejected(lambda:validate_compact([],row,expected,deadline))
    rejected(lambda:validate_compact([dict(compact,seed=2)],row,expected,deadline))
    rejected(lambda:validate_compact([dict(compact,decision=1)],row,expected,deadline))
    turns=[dict(**base,step=i,time_third=i//240,factual_task_calls=int(i==0),shadow_task_calls=int(i==0)) for i in range(719)]
    assert validate_turns(turns,row,expected,deadline)['turns']==719
    rejected(lambda:validate_turns([turns[0]]*719,row,expected,deadline))
    rejected(lambda:validate_turns(turns[:-1],row,expected,deadline))
    rejected(lambda:validate_turns([dict(r,game='wrong') for r in turns],row,expected,deadline))
    return dict(passed=True,fixture='Synthetic coverage invariants',rejected_cases=6,
                policy_calls=0,official_transitions=0,new_games=0,private_holdout_reads=0)


if __name__=='__main__':
    import json
    print(json.dumps(run()))
