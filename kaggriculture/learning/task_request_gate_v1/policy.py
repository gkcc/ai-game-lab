"""Select one complete proposed request; never mix controller decisions.

Both controllers use the same frozen crop weights. The teacher alternative uses
rule task selection, not a fully rule-based teacher. Inputs and proposal traces
remain local; only eight contemporaneous proposal aggregates enter the gate.
"""
from collections import Counter
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
FEATURES = ('visible_step_div_719', 'student_task_record_count',
            *(f'teacher_minus_student_chosen_count.{op}' for op in OPS),
            'teacher_minus_student_chosen_distance_sum')
MODEL_SCHEMA = 'task-request-gate-v1'
TARGET = 'terminal_cash_margin_delta_div_1000'
DEFAULT_LIMITS = dict(task_forwards=4000, crop_forwards=8000, neural_input_rows=160000,
                      gate_evaluations=3, controller_calls=839, teacher_proposals=120,
                      max_task_forwards_per_turn=32, max_crop_forwards_per_turn=64,
                      max_neural_input_rows_per_turn=4096, max_controller_calls_per_turn=2)


def _array(value, shape=None):
    result = np.asarray(value, dtype=np.float64)
    if (shape is not None and result.shape != shape) or not np.isfinite(result).all():
        raise ValueError('Nonfinite or incorrectly shaped request-gate array')
    return result


class RequestModel:
    """Eight normalized features plus unpenalized intercept; fixed CPU ridge."""

    def __init__(self, mean, scale, coefficients, intercept):
        self.mean = _array(mean, (8,)).copy()
        self.scale = _array(scale, (8,)).copy()
        self.coefficients = _array(coefficients, (8,)).copy()
        self.intercept = float(_array(intercept, ()))
        if not (self.scale >= 1).all():
            raise ValueError('Request-gate scales must be at least one')

    @classmethod
    def fit(cls, x, target):
        """One solve: mean squared target error + squared nonintercept weights."""
        x = _array(x)
        if x.ndim != 2 or x.shape[1] != 8 or not len(x):
            raise ValueError('Training requires a nonempty N by 8 feature matrix')
        target = _array(target, (len(x),))
        mean = x.mean(axis=0)
        scale = np.maximum(x.std(axis=0), 1.)
        design = np.column_stack((np.ones(len(x), dtype=np.float64), (x - mean) / scale))
        penalty = np.diag([0.] + [1.] * 8)
        weights = np.linalg.solve(design.T @ design / len(x) + penalty,
                                  design.T @ target / len(x))
        return cls(mean, scale, weights[1:], weights[0])

    def predict(self, x):
        x = _array(x)
        if x.shape == (8,):
            single, x = True, x[None, :]
        elif x.ndim == 2 and x.shape[1] == 8:
            single = False
        else:
            raise ValueError('Prediction requires an 8-vector or N by 8 matrix')
        result = _array(((x - self.mean) / self.scale) @ self.coefficients + self.intercept)
        return float(result[0]) if single else result

    def to_dict(self):
        return dict(schema=MODEL_SCHEMA, features=list(FEATURES), ops=list(OPS), target=TARGET,
                    ridge_lambda=1., mean=self.mean.tolist(), scale=self.scale.tolist(),
                    coefficients=self.coefficients.tolist(), intercept=self.intercept)

    @classmethod
    def from_dict(cls, value):
        keys = {'schema', 'features', 'ops', 'target', 'ridge_lambda',
                'mean', 'scale', 'coefficients', 'intercept'}
        if (not isinstance(value, dict) or set(value) != keys or value['schema'] != MODEL_SCHEMA
                or value['features'] != list(FEATURES) or value['ops'] != list(OPS)
                or value['target'] != TARGET or type(value['ridge_lambda']) not in (int, float)
                or value['ridge_lambda'] != 1.):
            raise ValueError('Incompatible request-gate model schema')
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
            raise ValueError('Request-gate model identity mismatch')
        def reject_constant(_value):
            raise ValueError('Nonfinite token in request-gate JSON')
        return cls.from_dict(json.loads(raw, parse_constant=reject_constant))


def _typed(value):
    """JSON values with explicit scalar types and ordered lists."""
    kind = type(value)
    if kind is dict:
        if any(type(key) is not str for key in value):
            raise ValueError('Request dictionary keys must be strings')
        return ['dict', [[key, _typed(value[key])] for key in sorted(value)]]
    if kind is list:
        return ['list', [_typed(item) for item in value]]
    if kind is float:
        if not np.isfinite(value):
            raise ValueError('Requests must contain finite JSON values')
        return ['float', 0. if value == 0 else value]
    if kind in (str, int, bool, type(None)):
        return [kind.__name__, value]
    raise ValueError('Request contains an unsupported non-JSON value')


def same_request(left, right):
    """Dict order is irrelevant; scalar types and all list ordering are exact."""
    return _typed(left) == _typed(right)


def request_digest(request):
    raw = json.dumps(_typed(request), separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode('utf-8')
    return hashlib.sha256(raw).hexdigest()


def _assignments(trace):
    counts, distance = Counter(), 0.
    records = trace['records']
    if not isinstance(records, list):
        raise ValueError('Expected controller assignment records')
    for record in records:
        inputs = record['inputs']
        names = inputs['feature_names']
        if names.count('task.distance') != 1:
            raise ValueError('Missing or ambiguous task.distance feature')
        candidates = inputs['candidates']
        selected = [c for c in candidates if c['id'] == record['chosen']]
        if len(selected) != 1:
            raise ValueError('Chosen assignment must identify exactly one own candidate')
        chosen = selected[0]
        if chosen['op'] not in OPS or chosen['op'] != record['op']:
            raise ValueError('Unknown or inconsistent chosen task operation')
        x = _array(chosen['x'], (len(names),))
        value = float(x[names.index('task.distance')])
        if value < 0:
            raise ValueError('Chosen task distance must be nonnegative')
        counts[chosen['op']] += 1
        distance += value
    if not np.isfinite(distance):
        raise ValueError('Nonfinite chosen-distance sum')
    return len(records), counts, distance


def request_features(step, student_trace, teacher_trace):
    """Each proposal is aggregated independently; callbacks are never aligned."""
    if type(step) is not int or not 0 <= step <= 719:
        raise ValueError('Invalid visible step')
    if student_trace['step'] != step or teacher_trace['step'] != step:
        raise ValueError('Both proposal traces must describe the same visible turn')
    number, student, student_distance = _assignments(student_trace)
    _number, teacher, teacher_distance = _assignments(teacher_trace)
    values = [step / 719., float(number), *(float(teacher[op] - student[op]) for op in OPS),
              teacher_distance - student_distance]
    return _array(values, (8,)).tolist()


class _Budget:
    """One per-agent budget shared by both frozen controllers and their heads."""

    def __init__(self, limits):
        if limits is not None and not set(limits) <= set(DEFAULT_LIMITS):
            raise ValueError('Unknown request-gate computation limit')
        self.limits = dict(DEFAULT_LIMITS, **({} if limits is None else limits))
        if any(type(value) is not int or value < 1 for value in self.limits.values()):
            raise ValueError('All request-gate limits must be positive integers')
        self.reset()

    def reset(self):
        self.counts = dict(task_forwards=0, task_rows=0, crop_forwards=0, crop_rows=0,
                           gate_evaluations=0, controller_calls=0, teacher_proposals=0)
        self.begin_turn()

    def begin_turn(self):
        self.turn = {key: 0 for key in self.counts}

    def controller(self, proposal=False):
        if (self.counts['controller_calls'] >= self.limits['controller_calls']
                or self.turn['controller_calls'] >= self.limits['max_controller_calls_per_turn']
                or (proposal and self.counts['teacher_proposals'] >= self.limits['teacher_proposals'])):
            raise RuntimeError('Controller/proposal computation budget exhausted')
        self.counts['controller_calls'] += 1
        self.turn['controller_calls'] += 1
        if proposal:
            self.counts['teacher_proposals'] += 1
            self.turn['teacher_proposals'] += 1

    def forward(self, kind, rows):
        if (self.counts[kind + '_forwards'] >= self.limits[kind + '_forwards']
                or self.turn[kind + '_forwards'] >= self.limits['max_' + kind + '_forwards_per_turn']
                or self.counts['task_rows'] + self.counts['crop_rows'] + rows > self.limits['neural_input_rows']
                or self.turn['task_rows'] + self.turn['crop_rows'] + rows > self.limits['max_neural_input_rows_per_turn']):
            raise RuntimeError('Shared neural computation budget exhausted')
        self.counts[kind + '_forwards'] += 1
        self.counts[kind + '_rows'] += rows
        self.turn[kind + '_forwards'] += 1
        self.turn[kind + '_rows'] += rows

    def gate(self):
        if self.counts['gate_evaluations'] >= self.limits['gate_evaluations']:
            raise RuntimeError('Request-gate prediction budget exhausted')
        self.counts['gate_evaluations'] += 1
        self.turn['gate_evaluations'] += 1


class RequestAgent:
    """First whole-request disagreement in each allowed fixed window.

    A baseline without an assigned window is pure student (no teacher proposal).
    The teacher_task arm is rule task choice plus the same frozen crop network;
    it constructs no task network. Counterfactual features use cached proposals.
    """

    def __init__(self, crop_path, crop_sha, task_path, task_sha, mode='baseline',
                 assigned_window=None, model=None, trace=None, limits=None):
        if mode not in ('baseline', 'intervention', 'gated', 'teacher_task'):
            raise ValueError('Unknown request selection mode')
        if assigned_window is not None and (type(assigned_window) is not int or not 0 <= assigned_window < 3):
            raise ValueError('Unknown assigned request window')
        if mode == 'intervention' and assigned_window is None:
            raise ValueError('One-turn intervention needs one assigned window')
        if mode == 'teacher_task' and assigned_window is not None:
            raise ValueError('Full teacher_task arm has no request window')
        if (mode == 'gated') != (model is not None):
            raise ValueError('Only gated mode requires a request model')
        if model is not None and not isinstance(model, RequestModel):
            raise ValueError('Expected the new request-gate model schema')
        self.mode, self.assigned_window, self.model = mode, assigned_window, model
        self.task_sha, self.trace = task_sha, Path(trace) if trace else None
        self.budget = _Budget(limits)
        self.student, self.teacher, self.task_network = None, None, None
        if mode != 'teacher_task':
            self.task_network = TaskNetwork(task_path, task_sha)
            def task_selector(inputs):
                self.budget.forward('task', len(inputs['candidates']))
                return self.task_network(inputs)
            self.student = TaskAgent(task_selector, 'model', crop_path, crop_sha)
            self._count_crop(self.student)
        if mode != 'baseline' or assigned_window is not None:
            self.teacher = TaskAgent(None, 'model', crop_path, crop_sha)
            self._count_crop(self.teacher)
        self.reset()

    def _count_crop(self, controller):
        original = controller.crop.network.logits
        def counted(x, names):
            self.budget.forward('crop', 1)  # One example, not five output logits.
            return original(x, names)
        controller.crop.network.logits = counted

    def reset(self):
        self.step, self.player, self.last = None, None, None
        self.consumed, self.window_proposals = set(), Counter()
        self.budget.reset()
        for controller in (self.student, self.teacher):
            if controller is not None:
                controller.records, controller.last = [], None
                controller.crop.records, controller.crop.last = [], None
        if self.task_network is not None:
            self.task_network.last = None

    def _begin_turn(self, step, player):
        if type(step) is not int or not 0 <= step <= 719 or type(player) is not int or player not in (0, 1):
            raise ValueError('Invalid visible turn or player')
        if step == 0:
            self.reset()
        elif self.step is not None and (step <= self.step or player != self.player):
            raise ValueError('Nonincreasing step or changed player requires a new episode')
        self.step, self.player, self.last = step, player, None
        self.budget.begin_turn()

    def _run(self, controller, observation, configuration, *, proposal=False, student=False):
        self.budget.controller(proposal)
        action = controller(deepcopy(observation), deepcopy(configuration))
        if (type(action) is not dict or set(action) != {'farmer', 'hands', 'market'}
                or any(type(action[key]) is not list for key in action)):
            raise ValueError('Frozen controller must propose one complete request')
        request_digest(action)  # Reject nonfinite or non-JSON requests even outside windows.
        trace = deepcopy(controller.last)
        if trace['step'] != self.step or any(trace[key] for key in ('fallbacks', 'load_errors', 'inference_errors')):
            raise ValueError('Controller trace failed execution checks')
        trace['task_checkpoint_sha256'] = self.task_sha if student else None
        return action, trace

    def __call__(self, observation, configuration):
        view = observe(observation, configuration)
        obs, cfg = view.observation, view.configuration
        self._begin_turn(obs['step'], obs['player'])
        student_action = teacher_action = student_trace = teacher_trace = proposals = None
        events = []
        selected_teacher = self.mode == 'teacher_task'
        if selected_teacher:
            teacher_action, teacher_trace = self._run(self.teacher, obs, cfg)
        else:
            student_action, student_trace = self._run(self.student, obs, cfg, student=True)
            windows = (() if self.mode == 'baseline' and self.assigned_window is None
                       else range(3) if self.assigned_window is None else (self.assigned_window,))
            window = next((w for w in windows if WINDOWS[w][0] <= self.step <= WINDOWS[w][1]
                           and w not in self.consumed), None)
            if window is not None:
                if self.window_proposals[window] >= 40:
                    raise RuntimeError('Per-window teacher proposal budget exhausted')
                self.window_proposals[window] += 1
                teacher_action, teacher_trace = self._run(self.teacher, obs, cfg, proposal=True)
                proposals = dict(student=deepcopy(student_action), teacher=deepcopy(teacher_action))
                if not same_request(student_action, teacher_action):
                    self.consumed.add(window)
                    x8 = request_features(self.step, student_trace, teacher_trace)
                    prediction = None
                    selected_teacher = self.mode == 'intervention'
                    if self.mode == 'gated':
                        self.budget.gate()
                        prediction = float(self.model.predict(x8))
                        if not np.isfinite(prediction):
                            raise ValueError('Nonfinite request-gate prediction')
                        selected_teacher = prediction > 0
                    events.append(dict(step=self.step, player=self.player, window=window, x8=x8,
                                       prediction=prediction, selected_teacher=selected_teacher,
                                       changed_request=selected_teacher, different_request=True,
                                       student_request_sha256=request_digest(student_action),
                                       teacher_request_sha256=request_digest(teacher_action)))
        selected_action = teacher_action if selected_teacher else student_action
        selected_trace = teacher_trace if selected_teacher else student_trace
        self.last = deepcopy(selected_trace)
        self.last.update(request_schema_version=MODEL_SCHEMA, request_mode=self.mode,
                         selected_teacher=selected_teacher, selected_trace=deepcopy(selected_trace),
                         student_trace=student_trace, teacher_trace=teacher_trace, proposals=proposals,
                         request_events=events, request_counts=dict(self.budget.counts),
                         turn_request_counts=dict(self.budget.turn),
                         request_student_checkpoint_sha256=self.task_sha if self.student else None)
        if self.trace:
            self.trace.parent.mkdir(parents=True, exist_ok=True)
            with gzip.open(self.trace, 'at', encoding='utf-8') as stream:
                stream.write(json.dumps(self.last, separators=(',', ':'), allow_nan=False) + '\n')
        return deepcopy(selected_action)
