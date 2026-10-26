"""Frozen controller with a SHA-bound task head and bounded trace collection.

Conditional teacher labels are written by the existing controller after the
selector call. The selector sees only its ordinary legal encoded inputs.
"""
from copy import deepcopy
import gzip
import json
from pathlib import Path

from decision.api import observe
from learning.task_choice_v1.agent import TaskAgent
from learning.task_bc_v1.model import TaskNetwork
from learning.task_request_gate_v1.policy import _Budget


LIMITS = dict(task_forwards=4000, crop_forwards=8000, neural_input_rows=160000,
              gate_evaluations=1, controller_calls=719, teacher_proposals=1,
              max_task_forwards_per_turn=32, max_crop_forwards_per_turn=64,
              max_neural_input_rows_per_turn=4096, max_controller_calls_per_turn=1)


class BoundedAgent:
    def __init__(self, crop_path, crop_sha, task_path=None, task_sha=None,
                 trace=None, limits=None):
        if (task_path is None) != (task_sha is None):
            raise ValueError('Task checkpoint and identity must both be supplied')
        self.budget = _Budget(LIMITS if limits is None else limits)
        self.network = TaskNetwork(task_path, task_sha) if task_path is not None else None
        self.task_sha = task_sha

        def select(inputs):
            if any(k in inputs for k in ('teacher_label', 'chosen', 'teacher_priority')):
                raise ValueError('Training-only label exposed to selector')
            self.budget.forward('task', len(inputs['candidates']))
            return self.network(inputs)

        self.controller = TaskAgent(select if self.network is not None else None,
                                    'model', crop_path, crop_sha)
        original_crop = self.controller.crop.network.logits

        def crop(x, names):
            self.budget.forward('crop', 1)
            return original_crop(x, names)

        self.controller.crop.network.logits = crop
        self.trace = Path(trace) if trace is not None else None
        self.last = None
        self.previous = None

    def __call__(self, observation, configuration):
        self.last = None
        view = observe(observation, configuration)
        step, player = view.observation['step'], view.observation['player']
        if type(step) is not int or not 0 <= step < 719:
            raise ValueError('Expected a live official action turn')
        if step == 0:
            if self.trace is not None and self.trace.exists():
                raise FileExistsError('A new episode needs a new trace destination')
            self.budget.reset()
            self.previous = None
        if self.previous is None:
            if step != 0:
                raise ValueError('Full-episode agent must begin at zero')
        elif (player, step - 1) != self.previous:
            raise ValueError('Cross-episode or out-of-order agent input')
        self.budget.begin_turn()
        self.budget.controller()
        request = self.controller(view.observation, view.configuration)
        trace = deepcopy(self.controller.last)
        if any(trace[k] != 0 for k in ('load_errors', 'inference_errors', 'fallbacks')):
            raise ValueError('Frozen controller reported an error')
        if set(request) != {'farmer', 'hands', 'market'}:
            raise ValueError('Incomplete official request')
        trace.update(model_counts=dict(self.budget.counts),
                     turn_model_counts=dict(self.budget.turn),
                     task_checkpoint_sha256=self.task_sha)
        json.dumps(request, allow_nan=False)
        json.dumps(trace, allow_nan=False)
        self.last = trace
        self.previous = (player, step)
        if self.trace is not None:
            self.trace.parent.mkdir(parents=True, exist_ok=True)
            with gzip.open(self.trace, 'at', encoding='utf-8', newline='\n') as stream:
                stream.write(json.dumps(trace, separators=(',', ':'), allow_nan=False) + '\n')
        return request
