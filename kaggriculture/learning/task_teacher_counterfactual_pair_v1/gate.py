"""One parameterized task substitution with the original budgeted forward.

The expected choices and digest belong to a private experiment contract. Only
the actor's own live selector input is passed to the original model.
"""
from copy import deepcopy

from learning.task_bc_error_diagnostics_v1.shadow import ordered_hash

ACTION_TURNS=719
BUDGET_KEYS=frozenset(('task_forwards','task_rows','crop_forwards','crop_rows',
                       'gate_evaluations','controller_calls','teacher_proposals'))


def _require(condition,message):
    if not condition: raise ValueError(message)


def _snapshot(reader):
    value=reader()
    _require(type(value) is dict and set(value)=={'total','turn'},'Invalid budget snapshot')
    for scope in ('total','turn'):
        counts=value[scope]
        _require(type(counts) is dict and set(counts)==BUDGET_KEYS
            and all(type(x) is int and x>=0 for x in counts.values()),'Invalid budget counters')
    return deepcopy(value)


def _check_single_forward(before,after,rows):
    for scope in ('total','turn'):
        for key in BUDGET_KEYS:
            delta=1 if key=='task_forwards' else rows if key=='task_rows' else 0
            _require(after[scope][key]-before[scope][key]==delta,'Unexpected model budget increment')


class OneShotSelector:
    """Retain the verified one-shot lifecycle with a private dynamic target."""
    def __init__(self, original, index_reader, budget_reader, expected, mode, *, step, ordinal):
        if (type(step) is not int or not 0 <= step < ACTION_TURNS
                or type(ordinal) is not int or not 0 <= ordinal < 32):
            raise ValueError('Invalid private intervention coordinate')
        _require(callable(original) and callable(index_reader) and callable(budget_reader)
            and mode in ('control','intervene') and type(expected) is dict
            and set(expected)=={'input_sha256','original_choice','teacher_choice'},'Invalid private target')
        digest=expected['input_sha256']
        _require(type(digest) is str and len(digest)==64 and all(c in '0123456789abcdef' for c in digest)
            and all(type(expected[k]) is str and bool(expected[k]) for k in ('original_choice','teacher_choice'))
            and expected['original_choice']!=expected['teacher_choice'],'Invalid private target')
        self._original,self._index_reader,self._budget_reader=original,index_reader,budget_reader
        self._expected,self.mode=dict(expected),mode
        self._next_turn=self._turn_calls=0
        self._hit=self._verified=self._failed=False
        self._evidence=None
        self.target_step,self.target_ordinal=step,ordinal

    def _healthy(self): _require(not self._failed,'Intervention already failed')

    @staticmethod
    def _input(inputs):
        _require(type(inputs) is dict and type(inputs.get('observation')) is dict
            and not any(k in inputs for k in ('teacher_label','chosen','teacher_priority')),
            'Invalid selector input')
        step=inputs['observation'].get('step');candidates=inputs.get('candidates')
        _require(type(step) is int and 0<=step<ACTION_TURNS and type(candidates) is list and bool(candidates)
            and all(type(c) is dict and type(c.get('id')) is str and c['id'] for c in candidates),
            'Invalid selector input')
        ids=[candidate['id'] for candidate in candidates]
        _require(len(ids)==len(set(ids)),'Duplicate candidate ID')
        return step,ids

    def __call__(self,inputs):
        try:
            self._healthy()
            step,ids=self._input(inputs)
            index=self._index_reader()
            _require(type(index) is int and index>=0 and step==self._next_turn
                           and index==self._turn_calls,'Out-of-order selector callback')
            _require(step<=self.target_step or self._verified,'Target was not verified')
            target=step==self.target_step and index==self.target_ordinal
            if target:
                _require(not self._hit and ordered_hash(inputs)==self._expected['input_sha256'],
                               'Target input differs')
                _require(self._expected['original_choice'] in ids and self._expected['teacher_choice'] in ids,
                               'Target candidate missing')
                before=_snapshot(self._budget_reader)
                original=self._original(inputs)
                after=_snapshot(self._budget_reader)
                _require(ordered_hash(inputs)==self._expected['input_sha256'],
                               'Selector mutated target input')
                _check_single_forward(before,after,len(ids))
                _require(type(original) is str and original==self._expected['original_choice'],
                               'Original selection differs')
                returned=original if self.mode=='control' else self._expected['teacher_choice']
                self._evidence=dict(mode=self.mode,step=step,selection_record_index=index,
                    input_sha256=self._expected['input_sha256'],original_choice=original,
                    teacher_choice=self._expected['teacher_choice'],returned_choice=returned,
                    budget_before=before,budget_after=after)
                self._hit=True
            else:
                _require(step!=self.target_step or index<self.target_ordinal or self._hit,
                               'Target callback was skipped')
                returned=self._original(inputs)
                _require(type(returned) is str and returned in ids,'Original selection invalid')
            self._turn_calls+=1
            return returned
        except BaseException:
            self._failed=True
            raise

    def verify_turn(self,trace):
        """Verify the live teacher and selection before the action is submitted."""
        try:
            self._healthy()
            _require(type(trace) is dict and type(trace.get('step')) is int
                           and trace['step']==self._next_turn and self._next_turn<ACTION_TURNS,
                           'Out-of-order controller turn')
            records=trace.get('records')
            _require(type(records) is list and len(records)==self._turn_calls
                           and all(type(r) is dict and r.get('source')=='callback' for r in records),
                           'Task selection records differ')
            if self._next_turn==self.target_step:
                _require(self._hit and not self._verified and self._evidence is not None
                               and len(records)>self.target_ordinal,'Target callback missing')
                record=records[self.target_ordinal]
                _require(record.get('teacher_label')==self._expected['teacher_choice']
                    and record.get('chosen')==self._evidence['returned_choice']
                    and ordered_hash(record.get('inputs'))==self._expected['input_sha256'],
                    'Recorded teacher or selection differs')
                self._evidence['teacher_label_verified']=True
                self._verified=True
            elif self._next_turn>self.target_step:
                _require(self._verified,'Target was not verified')
            self._next_turn+=1
            self._turn_calls=0
        except BaseException:
            self._failed=True
            raise

    def proof(self):
        try:
            self._healthy()
            _require(self._verified and self._evidence is not None,'Target is not verified')
            return deepcopy(self._evidence)
        except BaseException:
            self._failed=True
            raise

    def verify_complete(self):
        try:
            self._healthy()
            _require(self._verified and self._hit and self._next_turn==ACTION_TURNS and self._turn_calls==0,
                     'Episode or intervention incomplete')
            return True
        except BaseException:
            self._failed=True
            raise
