"""One controlled task-selector substitution; no engine or storage access.

The caller supplies a frozen, private expected-input digest and candidate IDs.
The only runtime input is the actor-visible selector input already produced by
TaskAgent. No opponent private state or future trajectory is read here. The
original selector is called once and charged by its existing budget wrapper.
This module does not establish official action legality or economic benefit.
"""
from copy import deepcopy

from learning.task_bc_error_diagnostics_v1.shadow import ordered_hash


TARGET_STEP = 2
TARGET_RECORD_INDEX = 5
ACTION_TURNS = 719
BUDGET_KEYS = frozenset(('task_forwards', 'task_rows', 'crop_forwards', 'crop_rows',
                         'gate_evaluations', 'controller_calls', 'teacher_proposals'))


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _sha256(value):
    return (isinstance(value, str) and len(value) == 64
            and all(character in '0123456789abcdef' for character in value))


def _snapshot(reader):
    """Copy the two _Budget counters without exposing their owner."""
    value = reader()
    _require(isinstance(value, dict) and set(value) == {'total', 'turn'}, 'Invalid budget snapshot')
    for scope in ('total', 'turn'):
        counts = value[scope]
        _require(isinstance(counts, dict) and set(counts) == BUDGET_KEYS
                 and all(type(item) is int and item >= 0 for item in counts.values()),
                 'Invalid budget counters')
    return deepcopy(value)


def _check_single_forward(before, after, rows):
    for scope in ('total', 'turn'):
        for key in BUDGET_KEYS:
            expected_delta = 1 if key == 'task_forwards' else rows if key == 'task_rows' else 0
            _require(after[scope][key] - before[scope][key] == expected_delta,
                     'Original selector did not use the expected budget')


class OneShotSelector:
    """Wrap an existing model selector for exactly one fixed assignment.

    expected has exactly input_sha256, original_choice, teacher_choice. These
    fields are private experimental input and must not be exposed to the model.
    index_reader returns len(agent.controller.records) during the callback.
    budget_reader returns {'total': agent.budget.counts, 'turn': agent.budget.turn}.

    After each BoundedAgent call and BEFORE returning the action to the game,
    the adapter must call verify_turn(agent.last). At episode end it must call
    verify_complete(); proof() returns a copy only after the target turn passes.
    """

    def __init__(self, original, index_reader, budget_reader, expected, mode):
        _require(callable(original) and callable(index_reader) and callable(budget_reader),
                 'Selector and readers must be callable')
        _require(mode in ('control', 'intervene'), 'Invalid intervention mode')
        _require(isinstance(expected, dict)
                 and set(expected) == {'input_sha256', 'original_choice', 'teacher_choice'},
                 'Invalid expected target')
        _require(_sha256(expected['input_sha256'])
                 and isinstance(expected['original_choice'], str) and bool(expected['original_choice'])
                 and isinstance(expected['teacher_choice'], str) and bool(expected['teacher_choice'])
                 and expected['original_choice'] != expected['teacher_choice'],
                 'Invalid target identities')
        self._original = original
        self._index_reader = index_reader
        self._budget_reader = budget_reader
        self._expected = dict(expected)
        self.mode = mode
        self._next_turn = 0
        self._turn_calls = 0
        self._hit = False
        self._verified = False
        self._failed = False
        self._evidence = None

    def _healthy(self):
        _require(not self._failed, 'Intervention already failed')

    @staticmethod
    def _input(inputs):
        _require(isinstance(inputs, dict) and isinstance(inputs.get('observation'), dict),
                 'Invalid selector input')
        _require(not any(key in inputs for key in ('teacher_label', 'chosen', 'teacher_priority')),
                 'Training-only label exposed to model')
        step = inputs['observation'].get('step')
        _require(type(step) is int and 0 <= step < ACTION_TURNS, 'Invalid selector step')
        candidates = inputs.get('candidates')
        _require(isinstance(candidates, list) and bool(candidates)
                 and all(isinstance(c, dict) and isinstance(c.get('id'), str) and c['id']
                         for c in candidates), 'Invalid candidate set')
        ids = [candidate['id'] for candidate in candidates]
        _require(len(ids) == len(set(ids)), 'Duplicate candidate ID')
        return step, ids

    def __call__(self, inputs):
        try:
            self._healthy()
            step, ids = self._input(inputs)
            index = self._index_reader()
            _require(type(index) is int and index >= 0 and step == self._next_turn
                     and index == self._turn_calls, 'Out-of-order selector callback')
            _require(step <= TARGET_STEP or self._verified, 'Target was not verified')
            at_target = step == TARGET_STEP and index == TARGET_RECORD_INDEX
            if at_target:
                _require(not self._hit, 'Duplicate target callback')
                _require(ordered_hash(inputs) == self._expected['input_sha256'], 'Target input differs')
                _require(self._expected['original_choice'] in ids and self._expected['teacher_choice'] in ids,
                         'Target candidate missing')
                before = _snapshot(self._budget_reader)
                original = self._original(inputs)
                after = _snapshot(self._budget_reader)
                _require(ordered_hash(inputs) == self._expected['input_sha256'], 'Selector mutated target input')
                _check_single_forward(before, after, len(ids))
                _require(isinstance(original, str) and original == self._expected['original_choice'],
                         'Original selection differs')
                returned = original if self.mode == 'control' else self._expected['teacher_choice']
                self._evidence = dict(mode=self.mode, step=step, selection_record_index=index,
                    input_sha256=self._expected['input_sha256'], original_choice=original,
                    teacher_choice=self._expected['teacher_choice'], returned_choice=returned,
                    budget_before=before, budget_after=after)
                self._hit = True
            else:
                _require(step != TARGET_STEP or index < TARGET_RECORD_INDEX or self._hit,
                         'Target callback was skipped')
                original = self._original(inputs)
                _require(isinstance(original, str) and original in ids, 'Original selection invalid')
                returned = original
            self._turn_calls += 1
            return returned
        except BaseException:
            self._failed = True
            raise

    def verify_turn(self, trace):
        """Check a completed controller turn before its action is submitted."""
        try:
            self._healthy()
            _require(isinstance(trace, dict) and type(trace.get('step')) is int
                     and trace['step'] == self._next_turn and self._next_turn < ACTION_TURNS,
                     'Out-of-order controller turn')
            records = trace.get('records')
            _require(isinstance(records, list) and len(records) == self._turn_calls
                     and all(isinstance(r, dict) and r.get('source') == 'callback' for r in records),
                     'Task selection records differ')
            if self._next_turn == TARGET_STEP:
                _require(self._hit and not self._verified and self._evidence is not None
                         and len(records) > TARGET_RECORD_INDEX, 'Target callback missing')
                record = records[TARGET_RECORD_INDEX]
                _require(record.get('teacher_label') == self._expected['teacher_choice']
                         and record.get('chosen') == self._evidence['returned_choice']
                         and ordered_hash(record.get('inputs')) == self._expected['input_sha256'],
                         'Recorded teacher or selection differs')
                self._evidence['teacher_label_verified'] = True
                self._verified = True
            elif self._next_turn > TARGET_STEP:
                _require(self._verified, 'Target was not verified')
            self._next_turn += 1
            self._turn_calls = 0
        except BaseException:
            self._failed = True
            raise

    def proof(self):
        try:
            self._healthy()
            _require(self._verified and self._evidence is not None, 'Target is not verified')
            return deepcopy(self._evidence)
        except BaseException:
            self._failed = True
            raise

    def verify_complete(self):
        try:
            self._healthy()
            _require(self._verified and self._hit and self._next_turn == ACTION_TURNS and self._turn_calls == 0,
                     'Episode or intervention incomplete')
            return True
        except BaseException:
            self._failed = True
            raise
