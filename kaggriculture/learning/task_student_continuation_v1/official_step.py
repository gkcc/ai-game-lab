"""Compare two complete requests from one authenticated official replay frame.

This privileged offline helper accepts a replay already authenticated by its
caller. It never calls a policy, publishes a replay, or returns observations,
seed values, private inventories, actions, or next states. Its boolean result
is a one-transition comparison, not a complete game or a future-return claim.
The caller must keep replay data and any result derived from private state in
the trusted evaluation process, and pass only an official single-player
observation and scrubbed configuration to any separate policy.
"""
from copy import deepcopy
import hashlib
from pathlib import Path

from decision.semantics import canonical
from decision.verify_mechanisms import ENGINE_SHA, make, rules
from kaggle_environments import core as official_core
from learning.panel_eval_v3.experiment import ENGINE_CORE_SHA256
from learning.task_request_gate_v1.policy import same_request


class OfficialStepError(ValueError):
    """The requested comparison cannot be authenticated or executed."""


def _require(condition, message):
    if not condition:
        raise OfficialStepError(message)


def _request(value):
    """Require a full, finite, type-preserving JSON request."""
    _require(type(value) is dict and set(value) == {'farmer', 'hands', 'market'}
             and all(type(value[name]) is list for name in ('farmer', 'hands', 'market')),
             'A complete request is required')
    try:
        _require(same_request(value, value), 'Invalid request')
    except (TypeError, ValueError):
        raise OfficialStepError('Invalid request') from None
    return deepcopy(value)


def _physical(state):
    """Compare already-canonical state effects, excluding the request itself."""
    value = deepcopy(state)
    for player in value:
        player.pop('action', None)
    return value


def compare_step(replay, *, step, side, baseline_request, intervention_request, guard=None):
    """Run a validated factual turn and one alternative official turn.

    ``replay`` must be an already authenticated, complete 720-frame replay of
    the fixed official engine. ``step`` indexes its pre-action frame (0..718);
    both original requests are in frame ``step + 1``. The supplied baseline
    must exactly match the recorded request for ``side``. The opponent always
    receives the recorded request in both arms.

    The engine is restored with the replay's original configuration, its
    privileged ``info`` and a deep copy of the selected frame. A placeholder
    prefix supplies the framework's required ``len(env.steps)`` only; it is not
    history and must never be exported as a replay. The factual output must
    equal the saved next frame under the existing v2 order-aware semantics
    before the alternative is run. Timing/overage metadata is excluded by v2.

    ``guard`` is a no-argument cooperative resource check called between major
    operations. A single official ``step`` cannot be interrupted by it; callers
    must additionally own process, wall-time and memory limits where needed.
    Guard exceptions and official engine OSError, MemoryError, and TimeoutError
    propagate unchanged, so the caller can stop the entire experiment.
    Any identity, replay, request, factual or execution failure raises. The
    returned fixed-field summary is privileged evaluation data; it is not an
    observation, action-value label, match result, or proof of future profit.
    """
    _require(guard is None or callable(guard), 'Invalid guard')
    check = guard if guard is not None else lambda: None
    _require(type(replay) is dict and replay.get('name') == 'kaggriculture'
             and type(replay.get('configuration')) is dict and type(replay.get('info')) is dict
             and type(replay.get('steps')) is list and len(replay['steps']) == 720
             and type(step) is int and 0 <= step < 719 and type(side) is int and side in (0, 1),
             'Invalid replay coordinate')
    configuration, info = deepcopy(replay['configuration']), deepcopy(replay['info'])
    _require(configuration.get('episodeSteps') == 720 and configuration.get('seed') is None
             and type(info.get('seed')) is int and 0 <= info['seed'] < 2**32,
             'Invalid official replay metadata')
    before, expected = deepcopy(replay['steps'][step]), deepcopy(replay['steps'][step + 1])
    _require(type(before) is list and len(before) == 2 and type(expected) is list and len(expected) == 2
             and all(type(player) is dict and type(player.get('observation')) is dict for player in before + expected)
             and type(before[0]['observation'].get('step')) is int and before[0]['observation']['step'] == step
             and type(expected[0]['observation'].get('step')) is int and expected[0]['observation']['step'] == step + 1
             and all(player.get('status') == 'ACTIVE' for player in before), 'Invalid replay frames')
    baseline, alternative = _request(baseline_request), _request(intervention_request)
    actions = [_request(player.get('action')) for player in expected]
    _require(same_request(baseline, actions[side]), 'Baseline request differs from replay')
    check()
    _require(hashlib.sha256(Path(rules.__file__).read_bytes()).hexdigest() == ENGINE_SHA,
             'Official rule identity differs')
    _require(hashlib.sha256(Path(official_core.__file__).read_bytes()).hexdigest() == ENGINE_CORE_SHA256,
             'Official core identity differs')

    def engine_call(operation):
        """Do not hide resource failures; sanitize other engine exceptions."""
        try:
            return operation()
        except OfficialStepError:
            raise
        except (OSError, MemoryError, TimeoutError):
            raise
        except Exception:
            raise OfficialStepError('Official transition failed') from None

    def run_one(joint):
        check()

        def restore():
            environment = make('kaggriculture', configuration=deepcopy(configuration),
                               info=deepcopy(info), steps=[None] * step + [deepcopy(before)],
                               debug=False)
            _require(environment.interpreter is rules.interpreter and len(environment.steps) == step + 1
                     and canonical(environment.state) == canonical(before),
                     'Restored frame differs')
            _require(environment.configuration.get('seed') is None
                     and canonical(environment.configuration) == canonical(configuration)
                     and environment.info == info,
                     'Restored metadata differs')
            return environment

        environment = engine_call(restore)
        check()
        engine_call(lambda: environment.step(deepcopy(joint)))
        check()

        def result():
            _require(not any(player.status in ('ERROR', 'INVALID', 'TIMEOUT')
                             for player in environment.state), 'Official action failed')
            return canonical(environment.state)

        return engine_call(result)

    factual = run_one(actions)
    _require(factual == canonical(expected), 'Factual next frame differs')
    # Verify the saved opponent request is exactly the one accepted by the
    # official framework before considering the intervention.
    _require(same_request(factual[1 - side]['action'], actions[1 - side]),
             'Factual opponent request differs')
    changed_actions = deepcopy(actions)
    changed_actions[side] = alternative
    intervention = run_one(changed_actions)
    _require(same_request(intervention[1 - side]['action'], actions[1 - side]),
             'Intervention opponent request differs')
    check()
    return dict(factual_reproduced=True,
                request_changed=not same_request(baseline, alternative),
                state_changed=_physical(factual) != _physical(intervention),
                scope='one_official_transition', complete_game=False,
                future_return_evaluated=False)
