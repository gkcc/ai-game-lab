"""Re-decode one existing, budgeted task forward; keep the frozen controller.

The provisional matching never becomes a multi-action request. The controller
consumes one ID, updates its live reservations/resources and calls again. This
module neither introduces opportunities nor makes official legality claims.
"""
from copy import deepcopy
import gzip
import json
from pathlib import Path

from decision.api import observe
from learning.task_dagger_once_v1.policy import BoundedAgent
from .decoder import DecoderError, candidate_key, choose


SCHEMA = 'task-joint-decoder-v1'


class JointAgent:
    """Same task/crop checkpoints and computation budgets, different decoding.

    ``network.last.chosen`` stays the original network argmax. The returned ID
    and the ordinary controller trace contain the decoder's actual choice.
    ``decoder_records`` explicitly records both, once for each task callback.
    """

    def __init__(self, crop_path, crop_sha, task_path, task_sha, trace=None, limits=None):
        if task_path is None or task_sha is None:
            raise ValueError('Joint decoding requires a frozen task checkpoint')
        self.base = BoundedAgent(crop_path, crop_sha, task_path, task_sha,
                                 trace=None, limits=limits)
        self.network, self.controller = self.base.network, self.base.controller
        self.budget, self.task_sha = self.base.budget, self.base.task_sha
        self.trace = Path(trace) if trace is not None else None
        self.last, self.decoder_records = None, []
        original_selector = self.controller.task_selector

        def select(inputs):
            # Use the existing closure so each attempt is charged before its
            # one network call. A failure must never consume a previous last.
            self.network.last = None
            original = original_selector(inputs)
            scored = self.network.last
            if (type(scored) is not dict or scored.get('chosen') != original
                    or scored.get('checkpoint_sha256') != self.task_sha
                    or scored.get('fallbacks') != 0):
                raise DecoderError('Missing or inconsistent current task forward')
            candidates, logits = inputs['candidates'], scored.get('logits')
            selection = choose(candidates, logits)
            keys = [candidate_key(candidate) for candidate in candidates]
            if keys != sorted(keys):
                raise DecoderError('Expected the controller canonical candidate order')
            first_argmax = max(range(len(logits)), key=lambda i: float(logits[i]))
            if original != candidates[first_argmax]['id']:
                raise DecoderError('Current task choice disagrees with its logits')
            self.decoder_records.append(dict(
                ordinal=len(self.decoder_records), original_network_choice=original,
                decoder_choice=selection.choice_id, matching_ids=list(selection.matching_ids),
                matching_cardinality=selection.cardinality,
                matching_logit_sum=selection.total_logit, logits=[float(v) for v in logits]))
            return selection.choice_id

        self.controller.task_selector = select

    def __call__(self, observation, configuration):
        self.last, self.decoder_records = None, []
        self.base.last, self.network.last = None, None
        view = observe(observation, configuration)
        # BoundedAgent owns the turn/episode checks and counters. Its writer is
        # disabled, so preserve its step-zero overwrite rejection here.
        if view.observation['step'] == 0 and self.trace is not None and self.trace.exists():
            raise FileExistsError('A new episode needs a new trace destination')
        request = self.base(view.observation, view.configuration)
        trace = deepcopy(self.base.last)
        if (len(trace['records']) != len(self.decoder_records)
                or any(record['chosen'] != decoded['decoder_choice']
                       for record, decoded in zip(trace['records'], self.decoder_records))):
            raise DecoderError('Controller trace disagrees with current decoded choices')
        trace.update(decoder_schema_version=SCHEMA, decoder_records=deepcopy(self.decoder_records))
        encoded = json.dumps(trace, separators=(',', ':'), allow_nan=False) + '\n'
        if self.trace is not None:
            self.trace.parent.mkdir(parents=True, exist_ok=True)
            with gzip.open(self.trace, 'at', encoding='utf-8', newline='\n') as stream:
                stream.write(encoded)
        self.last = trace
        return request
