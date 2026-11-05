"""One frozen HARVEST choice, with the original model forward and budget.

This is a research intervention over the existing controller's opportunities,
not an official legality mask. A HARVEST alternative is not a teacher label.
"""
from copy import deepcopy

from learning.task_bc_error_diagnostics_v1.shadow import ordered_hash
from learning.task_teacher_counterfactual_pair_v1.gate import (
    OneShotSelector, _snapshot, _check_single_forward,
)


TARGET_KEYS = frozenset(('side', 'step', 'ordinal', 'input_sha256',
                         'original_choice', 'harvest_choice'))


def require(condition, message='Harvest window contract differs'):
    if not condition:
        raise ValueError(message)


def validate_target(target):
    require(type(target) is dict and set(target) == TARGET_KEYS)
    require(type(target['side']) is int and target['side'] in (0, 1)
            and type(target['step']) is int and 0 <= target['step'] <= 671
            and type(target['ordinal']) is int and 0 <= target['ordinal'] < 32)
    digest = target['input_sha256']
    require(type(digest) is str and len(digest) == 64
            and all(c in '0123456789abcdef' for c in digest))
    require(all(type(target[key]) is str and target[key]
                for key in ('original_choice', 'harvest_choice'))
            and target['original_choice'] != target['harvest_choice'])
    return deepcopy(target)


def _harvest_candidate(inputs, expected):
    candidates = {c['id']: c for c in inputs['candidates']}
    require(expected['original_choice'] in candidates and expected['harvest_choice'] in candidates,
            'Frozen candidate missing')
    original, harvest = (candidates[expected[key]]
                         for key in ('original_choice', 'harvest_choice'))
    unit, target = harvest.get('unit'), harvest.get('target')
    observation = inputs['observation']
    require(observation.get('player') == expected['side'] and harvest.get('op') == 'HARVEST'
            and type(unit) is int and unit >= 0 and type(original.get('unit')) is int
            and original['unit'] == unit
            and type(target) is list and len(target) == 2
            and all(type(x) is int and x >= 0 for x in target), 'Invalid HARVEST opportunity')
    farm = observation['farms'][expected['side']]
    units = [farm['farmer'], *farm['hands']]
    require(unit < len(units) and list(units[unit]) == target, 'HARVEST is not at the current unit position')
    x, y = target
    require(y < len(farm['tiles']) and x < len(farm['tiles'][y]))
    tile = farm['tiles'][y][x]
    require(type(tile) is dict and tile.get('kind') == 'PLANT'
            and type(tile.get('yield_units')) is int and tile['yield_units'] > 0,
            'HARVEST has no visible standing yield')
    # There is no official backpack-capacity restriction on HARVEST.
    # Source selection checks crop production age before freezing the digest.
    return deepcopy(harvest)


class HarvestSelector:
    """Keep the old one-shot ordering/forward contract without teacher relabeling."""
    def __init__(self, original, index_reader, budget_reader, target, mode):
        self.target = validate_target(target)
        require(mode in ('control', 'intervene') and all(callable(x)
                for x in (original, index_reader, budget_reader)))
        self.original, self.index_reader, self.budget_reader = original, index_reader, budget_reader
        self.mode, self.next_turn, self.turn_calls = mode, 0, 0
        self.hit = self.verified = self.failed = False
        self.evidence = None

    def __call__(self, inputs):
        try:
            require(not self.failed, 'Failed HARVEST selector reused')
            step, ids = OneShotSelector._input(inputs)
            ordinal = self.index_reader()
            require(type(ordinal) is int and ordinal == self.turn_calls and step == self.next_turn,
                    'Out-of-order selector callback')
            target = step == self.target['step'] and ordinal == self.target['ordinal']
            require(step <= self.target['step'] or self.verified, 'Frozen callback was not verified')
            if target:
                require(not self.hit and ordered_hash(inputs) == self.target['input_sha256'],
                        'Frozen callback input differs')
                harvest = _harvest_candidate(inputs, self.target)
                before = _snapshot(self.budget_reader)
                original = self.original(inputs)
                after = _snapshot(self.budget_reader)
                require(ordered_hash(inputs) == self.target['input_sha256'], 'Selector mutated input')
                _check_single_forward(before, after, len(ids))
                require(original == self.target['original_choice'], 'Frozen original choice differs')
                returned = original if self.mode == 'control' else self.target['harvest_choice']
                self.evidence = dict(mode=self.mode, step=step, ordinal=ordinal,
                    input_sha256=self.target['input_sha256'], original_choice=original,
                    harvest_choice=self.target['harvest_choice'], returned_choice=returned,
                    harvest_unit=harvest['unit'], budget_before=before, budget_after=after)
                self.hit = True
            else:
                require(step != self.target['step'] or ordinal < self.target['ordinal'] or self.hit,
                        'Frozen callback skipped')
                returned = self.original(inputs)
                require(type(returned) is str and returned in ids, 'Original selector returned invalid candidate')
            self.turn_calls += 1
            return returned
        except BaseException:
            self.failed = True
            raise

    def verify_turn(self, trace):
        try:
            require(not self.failed and type(trace) is dict and trace.get('step') == self.next_turn)
            records = trace.get('records')
            require(type(records) is list and len(records) == self.turn_calls
                    and all(type(r) is dict and r.get('source') == 'callback' for r in records))
            if self.next_turn == self.target['step']:
                require(self.hit and not self.verified and self.evidence is not None
                        and len(records) > self.target['ordinal'], 'Frozen callback missing')
                record = records[self.target['ordinal']]
                require(record.get('chosen') == self.evidence['returned_choice']
                        and ordered_hash(record.get('inputs')) == self.target['input_sha256'])
                # Preserve the actual teacher label independently of the HARVEST choice.
                self.evidence['teacher_label'] = record.get('teacher_label')
                self.evidence['unit_request'] = deepcopy(record.get('unit_request'))
                if self.mode == 'intervene':
                    require(record.get('unit_request') == ['HARVEST'], 'Selected HARVEST was not requested')
                self.verified = True
            elif self.next_turn > self.target['step']:
                require(self.verified)
            self.next_turn += 1
            self.turn_calls = 0
        except BaseException:
            self.failed = True
            raise

    def finish(self):
        require(not self.failed and self.hit and self.verified and self.turn_calls == 0
                and self.next_turn == self.target['step'] + 48, 'HARVEST window incomplete')
        return self.proof()

    def proof(self):
        require(not self.failed and self.verified and self.evidence is not None,
                'HARVEST callback has not been verified')
        return deepcopy(self.evidence)
