"""A small reward gate; the frozen scorer and controller remain unchanged.

The gate sees only the old scorer's top-two logit gap, visible turn, and their
operation identities. It chooses between existing economic opportunities, not
the full official legal action space. Traces and supervision are output only.
"""
from copy import deepcopy
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np

from decision.api import observe
from learning.task_bc_v1.model import TaskNetwork
from learning.task_choice_v1.agent import TaskAgent
from learning.task_choice_v1.features import OPS


WINDOWS = ((80, 119), (260, 299), (440, 479))
FEATURES = ('top1_minus_top2_logit', 'visible_step_div_719',
            *(f'top1_op.{op}' for op in OPS), *(f'top2_op.{op}' for op in OPS))
DEFAULT_LIMITS = dict(task_forwards=4000, crop_forwards=4000,
                      neural_input_rows=140000, gate_evaluations=3,
                      max_task_forwards_per_turn=32, max_crop_forwards_per_turn=32,
                      max_neural_input_rows_per_turn=2000)
MODEL_SCHEMA = 'task-reward-gate-v1'


def _finite_array(value, shape=None):
    result = np.asarray(value, dtype=np.float64)
    if (shape is not None and result.shape != shape) or not np.isfinite(result).all():
        raise ValueError('Nonfinite or incorrectly shaped reward-gate array')
    return result


class GateModel:
    """Twelve coefficients plus intercept, CPU float64, one fixed ridge solve."""

    def __init__(self, mean, scale, coefficients, intercept):
        self.mean = _finite_array(mean, (2,)).copy()
        self.scale = _finite_array(scale, (2,)).copy()
        self.coefficients = _finite_array(coefficients, (12,)).copy()
        self.intercept = float(_finite_array(intercept, ()))
        if not (self.scale >= 1).all():
            raise ValueError('Reward-gate normalization scale must be at least one')

    @classmethod
    def fit(cls, x, target):
        """min mean((y-prediction)^2) + sum(nonintercept_weight^2).

        Only these training rows set the first two normalization columns. The
        ten operation indicators remain uncentered and unscaled. No selection,
        target clipping, iteration, or hyperparameter search occurs here.
        """
        x = _finite_array(x)
        if x.ndim != 2 or x.shape[1] != 12 or len(x) == 0:
            raise ValueError('Training requires nonempty N by 12 gate features')
        target = _finite_array(target, (len(x),))
        mean = x[:, :2].mean(axis=0)
        scale = np.maximum(x[:, :2].std(axis=0), 1.)
        normalized = x.copy()
        normalized[:, :2] = (normalized[:, :2] - mean) / scale
        design = np.column_stack((np.ones(len(x), dtype=np.float64), normalized))
        penalty = np.diag([0.] + [1.] * 12)
        weights = np.linalg.solve(design.T @ design / len(x) + penalty,
                                  design.T @ target / len(x))
        return cls(mean, scale, weights[1:], weights[0])

    def predict(self, x):
        values = _finite_array(x)
        if values.shape == (12,):
            single = True
            values = values[None, :]
        elif values.ndim == 2 and values.shape[1] == 12:
            single = False
        else:
            raise ValueError('Prediction requires a 12-vector or N by 12 matrix')
        normalized = values.copy()
        normalized[:, :2] = (normalized[:, :2] - self.mean) / self.scale
        result = _finite_array(normalized @ self.coefficients + self.intercept)
        return float(result[0]) if single else result

    def to_dict(self):
        return dict(schema=MODEL_SCHEMA, features=list(FEATURES), ops=list(OPS),
                    mean=self.mean.tolist(), scale=self.scale.tolist(),
                    coefficients=self.coefficients.tolist(), intercept=self.intercept,
                    ridge_lambda=1., target='own_terminal_cash_delta_div_1000')

    @classmethod
    def from_dict(cls, value):
        keys = {'schema', 'features', 'ops', 'mean', 'scale', 'coefficients',
                'intercept', 'ridge_lambda', 'target'}
        if (not isinstance(value, dict) or set(value) != keys
                or value['schema'] != MODEL_SCHEMA or value['features'] != list(FEATURES)
                or value['ops'] != list(OPS) or type(value['ridge_lambda']) not in (int, float)
                or value['ridge_lambda'] != 1.
                or value['target'] != 'own_terminal_cash_delta_div_1000'):
            raise ValueError('Incompatible reward-gate model')
        return cls(value['mean'], value['scale'], value['coefficients'], value['intercept'])

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('x', encoding='utf-8', newline='\n') as stream:
            json.dump(self.to_dict(), stream, indent=2, allow_nan=False)
            stream.write('\n')

    @classmethod
    def load(cls, path, expected_sha=None):
        raw = Path(path).read_bytes()
        if expected_sha is not None and hashlib.sha256(raw).hexdigest() != expected_sha:
            raise ValueError('Reward-gate model identity mismatch')
        def nonfinite(_value):
            raise ValueError('Nonfinite JSON token in reward-gate model')
        return cls.from_dict(json.loads(raw, parse_constant=nonfinite))


def gate_features(step, top1, top2, gap):
    if type(step) is not int or not 0 <= step <= 719:
        raise ValueError('Invalid visible turn')
    if top1['op'] not in OPS or top2['op'] not in OPS:
        raise ValueError('Unknown candidate operation')
    gap = float(gap)
    if not np.isfinite(gap) or gap < 0:
        raise ValueError('Invalid top-two score gap')
    return [gap, step / 719., *(float(top1['op'] == op) for op in OPS),
            *(float(top2['op'] == op) for op in OPS)]


class GateSelector:
    """One existing scorer call per callback; at most one gate per window.

    Only callback zero of a turn may consume a window. If that callback has
    one candidate, later callbacks of the same turn do not become eligible.
    A consumed window stays consumed even when the baseline/top1 is selected.
    New games must begin at step zero (or call reset explicitly).
    """

    def __init__(self, network, mode='baseline', assigned_window=None, model=None, limits=None):
        if mode not in ('baseline', 'intervention', 'gated'):
            raise ValueError('Unknown gate mode')
        if assigned_window is not None and (type(assigned_window) is not int
                                            or not 0 <= assigned_window < len(WINDOWS)):
            raise ValueError('Unknown assigned window')
        if mode == 'intervention' and assigned_window is None:
            raise ValueError('One-shot intervention requires one assigned window')
        if (mode == 'gated') != (model is not None):
            raise ValueError('Only gated mode requires a reward model')
        if not callable(network):
            raise TypeError('Frozen task network must be callable')
        self.network, self.mode = network, mode
        self.assigned_window, self.model = assigned_window, model
        if limits is not None and not set(limits) <= set(DEFAULT_LIMITS):
            raise ValueError('Unknown computation limit')
        self.limits = dict(DEFAULT_LIMITS, **({} if limits is None else limits))
        if any(type(v) is not int or v < 1 for v in self.limits.values()):
            raise ValueError('Explicit positive limits required for every counter')
        self.reset()

    def reset(self):
        self.step, self.player, self.callback = None, None, 0
        self.consumed, self.events = set(), []
        self.counts = dict(task_forwards=0, task_rows=0, crop_forwards=0,
                           crop_rows=0, gate_evaluations=0)
        self.turn_counts = dict(task_forwards=0, task_rows=0, crop_forwards=0, crop_rows=0)

    def begin_turn(self, step, player):
        if type(step) is not int or not 0 <= step <= 719 or type(player) is not int or player not in (0, 1):
            raise ValueError('Invalid visible step or player')
        if step == 0:
            self.reset()
        elif self.step is not None and (step <= self.step or player != self.player):
            raise ValueError('Nonincreasing turn or changed seat requires a new game reset')
        self.step, self.player, self.callback, self.events = step, player, 0, []
        self.turn_counts = dict(task_forwards=0, task_rows=0, crop_forwards=0, crop_rows=0)

    def _count_forward(self, kind, rows):
        if self.step is None:
            raise ValueError('begin_turn must precede a network forward')
        if (self.counts[kind + '_forwards'] + 1 > self.limits[kind + '_forwards']
                or self.counts['task_rows'] + self.counts['crop_rows'] + rows > self.limits['neural_input_rows']
                or self.turn_counts[kind + '_forwards'] + 1 > self.limits['max_' + kind + '_forwards_per_turn']
                or self.turn_counts['task_rows'] + self.turn_counts['crop_rows'] + rows > self.limits['max_neural_input_rows_per_turn']):
            raise RuntimeError('Frozen network computation budget exhausted')
        self.counts[kind + '_forwards'] += 1
        self.counts[kind + '_rows'] += rows
        self.turn_counts[kind + '_forwards'] += 1
        self.turn_counts[kind + '_rows'] += rows

    def count_crop_forward(self):
        """A crop forward has one input example and five output crop logits."""
        self._count_forward('crop', 1)

    def __call__(self, inputs):
        if self.step is None:
            raise ValueError('begin_turn must precede task selection')
        obs = inputs['observation']
        if (obs['step'], obs['player']) != (self.step, self.player):
            raise ValueError('Callback observation disagrees with current turn')
        candidates = inputs['candidates']
        if not candidates:
            raise ValueError('Task selection requires at least one candidate')
        ids = [candidate['id'] for candidate in candidates]
        if any(not isinstance(i, str) for i in ids) or len(set(ids)) != len(ids):
            raise ValueError('Candidate IDs must be unique strings')
        self._count_forward('task', len(candidates))
        old_choice = self.network(inputs)  # Includes frozen schema checks, one float64 forward.
        logits = _finite_array(self.network.last['logits'], (len(candidates),))
        ranking = np.argsort(-logits, kind='stable')
        first = int(ranking[0])
        if old_choice != ids[first] or self.network.last['chosen'] != old_choice:
            raise ValueError('Frozen scorer choice disagrees with its canonical logits')
        ordinal = self.callback
        self.callback += 1
        windows = range(len(WINDOWS)) if self.assigned_window is None else (self.assigned_window,)
        eligible = next((w for w in windows if WINDOWS[w][0] <= self.step <= WINDOWS[w][1]
                         and w not in self.consumed), None)
        if ordinal != 0 or len(candidates) < 2 or eligible is None:
            return old_choice

        self.consumed.add(eligible)
        second = int(ranking[1])
        gap = float(logits[first] - logits[second])
        x = gate_features(self.step, candidates[first], candidates[second], gap)
        prediction = None
        chosen = old_choice
        if self.mode == 'intervention':
            chosen = ids[second]
        elif self.mode == 'gated':
            if self.counts['gate_evaluations'] >= self.limits['gate_evaluations']:
                raise RuntimeError('Reward-gate computation budget exhausted')
            self.counts['gate_evaluations'] += 1
            prediction = float(self.model.predict(x))
            if not np.isfinite(prediction):
                raise ValueError('Nonfinite reward-gate prediction')
            if prediction > 0:
                chosen = ids[second]
        self.events.append(dict(step=self.step, player=self.player, window=eligible,
                                callback=ordinal, top1=ids[first], top2=ids[second],
                                top1_op=candidates[first]['op'], top2_op=candidates[second]['op'],
                                top1_logit=float(logits[first]), top2_logit=float(logits[second]),
                                gap=gap, x12=x, prediction=prediction, chosen=chosen,
                                changed_id=chosen != old_choice))
        return chosen


class GateAgent(TaskAgent):
    """The existing joint student with a scoped gate and local-only trace."""

    def __init__(self, crop_path, crop_sha, task_path, task_sha, mode='baseline',
                 assigned_window=None, model=None, trace=None, limits=None):
        self.selector = GateSelector(TaskNetwork(task_path, task_sha), mode,
                                     assigned_window, model, limits)
        self.task_sha, self.trace = task_sha, Path(trace) if trace else None
        super().__init__(self.selector, 'model', crop_path, crop_sha)
        crop_logits = self.crop.network.logits

        def counted_crop_logits(x, names):
            self.selector.count_crop_forward()
            return crop_logits(x, names)

        self.crop.network.logits = counted_crop_logits

    def __call__(self, observation, configuration):
        # Use exactly the existing visibility filter, including own private data.
        # Gate state stores only counters/windows, never an opponent's inventory.
        view = observe(observation, configuration)
        self.selector.begin_turn(view.observation['step'], view.observation['player'])
        return super().__call__(view.observation, view.configuration)

    def finish(self, context, result):
        result = super().finish(context, result)
        self.last['task_checkpoint_sha256'] = self.task_sha
        self.last['gate_mode'] = self.selector.mode
        self.last['gate_events'] = deepcopy(self.selector.events)
        self.last['gate_counts'] = dict(self.selector.counts)
        self.last['turn_gate_counts'] = dict(self.selector.turn_counts)
        if self.trace:
            self.trace.parent.mkdir(parents=True, exist_ok=True)
            with gzip.open(self.trace, 'at', encoding='utf-8') as stream:
                stream.write(json.dumps(self.last, separators=(',', ':'), allow_nan=False) + '\n')
        return result
