"""Pure synthetic cap regression: no models, observations or policy calls."""
from unittest.mock import patch
import subprocess

from .runner import BudgetJoint, run_owned, timeout_seconds
from learning.task_bc_v1.deployment import JointAgent


def verify():
    with patch('learning.task_bc_error_diagnostics_v1.runner.time.monotonic', return_value=100):
        assert timeout_seconds(1000, 900) == 900
        assert timeout_seconds(105, 900) == 5
        try:
            timeout_seconds(100, 900)
        except TimeoutError:
            pass
        else:
            raise AssertionError('Expired run dispatched a job')
        actor = object.__new__(BudgetJoint)
        actor.remaining = dict(shadow_groups=1, shadow_candidates=3)
        actor.deadline = 101
        actor.used = dict(shadow_groups=0, shadow_candidates=0)
        with patch.object(JointAgent, 'select', return_value='unchanged') as base:
            assert actor.select({'choices': [1, 2, 3]}) == 'unchanged'
            assert actor.used == dict(shadow_groups=1, shadow_candidates=3)
            assert base.call_count == 1
            try:
                actor.select({'choices': [1]})
            except ValueError:
                pass
            else:
                raise AssertionError('Group cap did not fail before scoring')
            assert base.call_count == 1
        actor.used = dict(shadow_groups=0, shadow_candidates=0)
        with patch.object(JointAgent, 'select') as base:
            try:
                actor.select({'choices': [1, 2, 3, 4]})
            except ValueError:
                pass
            else:
                raise AssertionError('Candidate cap did not fail before scoring')
            assert not base.called and actor.used == dict(shadow_groups=0, shadow_candidates=0)
            actor.deadline = 100
            try:
                actor.select({'choices': [1]})
            except TimeoutError:
                pass
            else:
                raise AssertionError('Expired selector reached scoring')
            assert not base.called
    class OwnedChild:
        def __init__(self, error):
            self.error, self.alive, self.reaped = error, True, False
        def wait(self, timeout=None):
            if timeout is not None and self.error is not None:
                raise self.error
            self.alive, self.reaped = False, True
            return 0
        def poll(self):
            return None if self.alive else 0
        def kill(self):
            self.alive = False
    cases = [('success', None), ('timeout', subprocess.TimeoutExpired('synthetic', 1)),
             ('exception', RuntimeError('synthetic')), ('interrupt', KeyboardInterrupt())]
    for label, error in cases:
        child = OwnedChild(error); lifecycle = {}
        with patch('learning.task_bc_error_diagnostics_v1.runner.subprocess.Popen', return_value=child):
            try:
                result = run_owned(['synthetic'], cwd=None, env=None, stream=None,
                                   timeout=1, lifecycle=lifecycle)
            except BaseException as actual:
                assert actual is error
            else:
                assert error is None and result == 0
        assert not child.alive and child.reaped and lifecycle['owned_child_closed'], label
    return dict(passed=True, scope='Synthetic scheduler caps and owned-child lifecycle',
                owned_child_cases=[label for label, _ in cases], owned_child_cases_reaped=4,
                source_observations_read=0, policy_calls=0, training_runs=0, new_games=0,
                official_transitions=0)


if __name__ == '__main__':
    import json
    print(json.dumps(verify(), sort_keys=True))
