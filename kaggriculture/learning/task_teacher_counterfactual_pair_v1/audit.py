"""Add explicit two-player shed-order checks to the unchanged official audit.

The original engine and audit perform transitions. Only the local replay
auditor's factory is scoped-wrapped, and restored even on failure. The caller
must run this in a separately supervised, single-threaded child process.
"""
from decision import full_audit
from learning.task_bc_error_diagnostics_v1.shadow import first_difference


def _shed(frame,side):
    private=frame[side]['observation']['private']
    shed=private['shed']
    if not isinstance(shed,dict):
        raise ValueError('Invalid private shed')
    return [[key,value] for key,value in shed.items()]


def reexecute(replay,*,guard,sink=None):
    """Return the official receipt plus explicit 720-frame shed coverage."""
    if (not __debug__ or not callable(guard) or type(replay) is not dict
            or type(replay.get('steps')) is not list or len(replay['steps'])!=720
            or (sink is not None and not callable(sink))):
        raise ValueError('Invalid official replay audit contract')
    original=full_audit.make
    count=[0]
    created=[False]
    def check(frame,index):
        guard()
        if (index>=720 or any(first_difference(_shed(frame,side),
                        _shed(replay['steps'][index],side)) is not None for side in (0,1))):
            raise ValueError('Official shed order differs')
        count[0]+=1
    def make(*args,**kwargs):
        if created[0]: raise ValueError('Unexpected extra official environment')
        created[0]=True
        env=original(*args,**kwargs)
        check(env.state,0)
        old_step=env.step
        def step(*step_args,**step_kwargs):
            result=old_step(*step_args,**step_kwargs)
            check(env.state,count[0])
            return result
        env.step=step
        return env
    full_audit.make=make
    try:
        receipt=full_audit.reexecute(replay,trace=True,label_sink=sink)
    finally:
        full_audit.make=original
    if (type(receipt) is not dict or receipt.get('passed') is not True
            or receipt.get('official_transitions_reexecuted')!=719
            or receipt.get('both_public_and_private_states_match_each_step') is not True
            or receipt.get('inventory_order_checked_each_step') is not True
            or count[0]!=720 or not created[0]):
        raise ValueError('Official replay audit incomplete')
    guard()
    return dict(official=receipt,shed_order_frames_checked=count[0],
                comparison='v2_canonical_python_plus_typed_ordered_shed',
                scope='one_separately_supervised_official_replay')
