"""Frozen task/crop models with the bounded virtual HARVEST choice seam.

Only the controller construction differs from the old bounded actor. Its whole
episode call/ordering, budgets and trace writer are inherited. No extra forward
is used to record task scores, and no harvest preference is added to the model.
"""
from copy import deepcopy
import math
from pathlib import Path

from learning.task_bc_v1.model import TaskNetwork
from learning.task_choice_v1.agent import TaskAgent
from learning.task_dagger_once_v1.policy import BoundedAgent
from learning.task_harvest_choice_v1.adapter import HarvestChoiceAgent
from learning.task_pair_dev_games_v1.panel import GAME_LIMITS
from learning.task_request_gate_v1.policy import _Budget


class _ModelHarvestController(HarvestChoiceAgent):
    def __init__(self, network, budget, crop_path, crop_sha, *, enabled):
        self.enabled, self.task_network = enabled, network

        def select(inputs):
            if any(k in inputs for k in ('teacher_label', 'chosen', 'teacher_priority')):
                raise ValueError('Training-only label exposed to selector')
            budget.forward('task', len(inputs['candidates']))
            return network(inputs)

        # Deliberately bypass HarvestChoiceAgent.__init__: that micro-scene
        # constructor fixes crop_mode='teacher'. Both frozen heads stay active.
        TaskAgent.__init__(self, select, 'model', crop_path, crop_sha)
        if self.crop.mode != 'model' or self.crop.network is None:
            raise ValueError('Harvest model controller requires the crop model')

    def select(self, context):
        choice = super().select(context)
        record = self.records[-1]
        evidence = deepcopy(self.task_network.last)
        candidates = record['inputs']['candidates']
        if (type(evidence) is not dict
                or set(evidence) != {'checkpoint_sha256', 'chosen', 'logits', 'fallbacks'}
                or evidence['checkpoint_sha256'] != self.task_network.sha
                or evidence['chosen'] != record['chosen']
                or type(evidence['fallbacks']) is not int or evidence['fallbacks'] != 0):
            raise ValueError('Missing or inconsistent task model evidence')
        logits = evidence['logits']
        if (type(logits) is not list or len(logits) != len(candidates)
                or any(type(value) is not float or not math.isfinite(value) for value in logits)):
            raise ValueError('Invalid task model evidence logits')
        # Pure comparison of the already computed logits; first exact tie wins,
        # matching the original TaskNetwork. This is not a second model call.
        index = max(range(len(logits)), key=logits.__getitem__)
        if candidates[index]['id'] != record['chosen']:
            raise ValueError('Task model evidence does not match its first argmax')
        evidence['argmax_index'] = index
        record['task_model'] = evidence
        return choice


class BoundedHarvestAgent(BoundedAgent):
    """Same bounded actor API, with both SHA-bound model paths required.

    ``enabled=False`` uses the original candidate set through TaskAgent.select.
    The full request, crop model and reservation logic remain the old actor's;
    records additionally retain the existing task forward's scores.
    """
    def __init__(self, crop_path, crop_sha, task_path, task_sha,
                 trace=None, limits=None, *, enabled=True):
        if type(enabled) is not bool:
            raise TypeError('enabled must be a boolean')
        if task_path is None or task_sha is None:
            raise ValueError('Both task checkpoint and identity are required; no teacher fallback')
        self.budget = _Budget(GAME_LIMITS if limits is None else limits)
        self.network = TaskNetwork(task_path, task_sha)
        self.task_sha = task_sha
        self.controller = _ModelHarvestController(self.network, self.budget, crop_path, crop_sha,
                                                  enabled=enabled)
        original_crop = self.controller.crop.network.logits

        def crop(x, names):
            self.budget.forward('crop', 1)
            return original_crop(x, names)

        self.controller.crop.network.logits = crop
        self.trace = Path(trace) if trace is not None else None
        self.last = None
        self.previous = None
