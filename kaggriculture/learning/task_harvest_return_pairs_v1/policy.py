"""One selector-level H/W intervention, with truthful model proposal evidence.

The H arm and every non-target callback use the frozen model controller. The W
callback alone returns its original WATER candidate through HarvestChoiceAgent,
so normal tile claiming, later reservations and market generation still run.
"""
from copy import deepcopy

from learning.task_harvest_choice_v1.adapter import HarvestChoiceAgent
from learning.task_harvest_model_dev_v1.policy import BoundedHarvestAgent
from .selection import ordered_hash, require, validate_binding, validate_proposal, validate_target


PROOF_SCHEMA = 'task-harvest-return-selector-proof-v1'
CHOICE_SCHEMA = 'task-harvest-return-choice-v1'


class BoundedPairAgent(BoundedHarvestAgent):
    def __init__(self, crop_path, crop_sha, task_path, task_sha,
                 trace=None, limits=None, *, target, arm, enabled=True):
        self.target = validate_target(target)
        require(type(arm) is str and arm in ('H', 'W') and enabled is True,
                'Return pair requires H/W with the same enabled candidate extension')
        require(task_sha == self.target['task_checkpoint_sha256']
                and crop_sha == self.target['crop_checkpoint_sha256'], 'Return-pair model identity differs')
        self.arm, self.override_count, self.target_proof = arm, 0, None
        super().__init__(crop_path, crop_sha, task_path, task_sha, trace, limits, enabled=True)
        # TaskAgent creates a fresh hash-pinned AST namespace per controller.
        # Bind only this instance; no shared module/global method is replaced.
        namespace = self.controller.controller.__globals__
        self._original_select, self._original_finish = namespace['_task_select'], namespace['_task_finish']
        namespace['_task_select'], namespace['_task_finish'] = self._select, self._finish

    def _budget(self):
        return dict(total=dict(self.budget.counts), turn=dict(self.budget.turn))

    def _select(self, context):
        obs = context['observation']
        require(obs['player'] == self.target['side'], 'Return-pair seat differs')
        ordinal = len(self.controller.records)
        if (obs['step'], ordinal) != (self.target['step'], self.target['ordinal']):
            return self._original_select(context)
        require(self.target_proof is None and self.override_count == 0, 'Return-pair target encountered twice')
        before = self._budget()
        # This is a selector-time prefix. Late unit_request fields are attached
        # by the ordinary finish hook only after all assignments have completed.
        prefix = ordered_hash(self.controller.records)
        if self.arm == 'H':
            choice = self._original_select(context)
            record = self.controller.records[-1]
            validate_binding(self.target, record['inputs'])
            proposal = validate_proposal(record['inputs'], record['task_model'], self.task_sha)
            require(record['chosen'] == self.target['harvest_choice'], 'Factual target is not the bound new H')
            selected_inputs = deepcopy(record['inputs'])
        else:
            original_selector, received = self.controller.task_selector, []

            def choose(inputs):
                validate_binding(self.target, inputs)
                proposed = original_selector(inputs)  # Exactly one ordinary, charged task forward.
                evidence = validate_proposal(inputs, self.network.last, self.task_sha, raw=True)
                require(proposed == evidence['chosen'] == self.target['harvest_choice'],
                        'W intervention original model proposal is not the bound H')
                received.append((deepcopy(inputs), evidence))
                return self.target['water_choice']

            self.controller.task_selector = choose
            try:
                # Only this callback bypasses the frozen model controller's
                # proposal==actual assertion: W is explicitly an intervention.
                # The ordinary HarvestChoiceAgent consumes the returned W slot.
                choice = HarvestChoiceAgent.select(self.controller, context)
            finally:
                self.controller.task_selector = original_selector
            require(len(received) == 1, 'W target did not make exactly one model proposal')
            selected_inputs, proposal = received[0]
            record = self.controller.records[-1]
            require(record['chosen'] == self.target['water_choice'] and record['op'] == 'WATER',
                    'W intervention did not consume the normal WATER choice')
            record['task_model'] = deepcopy(proposal)  # Truthful H proposal, not fabricated W logits.
            record['pair_choice'] = dict(schema_version=CHOICE_SCHEMA, arm='W',
                input_sha256=self.target['input_sha256'], proposed_choice=proposal['chosen'],
                actual_choice=record['chosen'], override_applied=True)
            self.override_count += 1
        extension = record['harvest_choice_extension']
        require(self.target['harvest_choice'] in extension['added_candidate_ids']
                and extension['selected_virtual_harvest'] is (self.arm == 'H'),
                'Bound target was not a virtual HARVEST choice')
        self.target_proof = dict(schema_version=PROOF_SCHEMA, arm=self.arm, target=deepcopy(self.target),
            inputs=selected_inputs, input_sha256=self.target['input_sha256'],
            original_proposal=deepcopy(proposal), actual_choice=record['chosen'],
            override_applied=self.arm == 'W', override_count=self.override_count,
            prefix_records_sha256=prefix, budget_before=before, budget_after=self._budget(), unit_request=None)
        return choice

    def _finish(self, context, result):
        result = self._original_finish(context, result)
        obs = context['observation']
        require(obs['player'] == self.target['side'], 'Return-pair seat differs')
        if obs['step'] >= self.target['step']:
            require(self.target_proof is not None and self.override_count == (1 if self.arm == 'W' else 0),
                    'Bound return-pair callback was never reached')
        if obs['step'] == self.target['step']:
            record = self.controller.last['records'][self.target['ordinal']]
            expected = ['HARVEST'] if self.arm == 'H' else ['WATER']
            require(record['unit'] == self.target['unit'] and record['target'] == self.target['target']
                    and record['unit_request'] == expected
                    and context['actions'][self.target['unit']] == expected,
                    'Selector choice did not reach the normal final unit request')
            self.target_proof['unit_request'] = deepcopy(record['unit_request'])
        return result
