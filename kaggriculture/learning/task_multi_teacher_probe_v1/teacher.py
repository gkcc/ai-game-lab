"""Capture one frozen teacher's own task labels without changing its choice.

The caller authenticates the exact immutable source bytes against a trusted
completed-game receipt and owns process/memory/time limits. A source hash is an
identity check, not a sandbox for untrusted code. Returned observations and
records are private and must be discarded or reduced to approved local data.
No official engine, task network, crop network or historical seed file is read.
"""
import ast
from copy import deepcopy
import hashlib
import math
import re

from decision.api import observe
from learning.task_choice_v1.features import CANDIDATE_SCOPE, OPS, SCHEMA_VERSION, candidate_id, encode_candidates


ALIASES = ('market', 'round2')
MAX_SOURCE_BYTES = 1024 * 1024
MAX_CANDIDATES = 256
MAX_CALLBACKS_PER_TURN = 32
WATCHED = ('observation', 'configuration', 'choices', 'tasks', 'units', 'actions', 'available',
           'seeds', 'needed_seeds', 'supply', 'projected_shed', 'remaining', 'today',
           'last_plant_age', 'final_day')
RESERVED = '_probe_task_select'


class ProbeError(ValueError):
    """A source, callback or private policy view failed validation."""


def _require(value, message):
    if not value:
        raise ProbeError(message)


def _snapshot(value):
    """Compare structured controller state, preserving order and scalar types."""
    if value is None:
        return ('none',)
    if type(value) is bool:
        return ('bool', value)
    if type(value) is int:
        return ('int', value)
    if type(value) is float:
        _require(math.isfinite(value), 'nonfinite controller state')
        return ('float', value.hex())
    if type(value) is str:
        return ('str', value)
    if type(value) in (list, tuple):
        return ('list' if type(value) is list else 'tuple', tuple(_snapshot(item) for item in value))
    if type(value) is dict:
        return ('dict', tuple((_snapshot(key), _snapshot(item)) for key, item in value.items()))
    if type(value) in (set, frozenset):
        values = [_snapshot(item) for item in value]
        return ('set' if type(value) is set else 'frozenset', tuple(sorted(values, key=repr)))
    raise ProbeError('unsupported controller state')


def _watched(context):
    _require(type(context) is dict and all(key in context for key in WATCHED),
             'missing controller context')
    return tuple(_snapshot(context[key]) for key in WATCHED)


def _task_call(node):
    return (isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == 'max'
            and len(node.args) >= 1 and isinstance(node.args[0], ast.Name) and node.args[0].id == 'choices')


def _reserved_reference(node):
    """Reject static bindings/references, including literal globals access."""
    if isinstance(node, ast.Name):
        return node.id == RESERVED
    if isinstance(node, ast.arg):
        return node.arg == RESERVED
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return node.name == RESERVED
    if isinstance(node, ast.alias):
        return node.asname == RESERVED or RESERVED in node.name.split('.')
    if isinstance(node, ast.ImportFrom):
        return RESERVED in (node.module or '').split('.')
    if isinstance(node, (ast.Global, ast.Nonlocal)):
        return RESERVED in node.names
    if isinstance(node, (ast.ExceptHandler, ast.MatchAs, ast.MatchStar)):
        return node.name == RESERVED
    if isinstance(node, ast.MatchMapping):
        return node.rest == RESERVED
    if isinstance(node, ast.MatchClass):
        return RESERVED in node.kwd_attrs
    if isinstance(node, ast.Attribute):
        return node.attr == RESERVED
    if isinstance(node, ast.keyword):
        return node.arg == RESERVED
    return isinstance(node, ast.Constant) and node.value == RESERVED


def _instrument(source):
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError, TypeError, UnicodeError):
        raise ProbeError('invalid teacher source') from None
    _require(not any(_reserved_reference(node) for node in ast.walk(tree)),
             'teacher source uses reserved callback name')
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'agent']
    _require(len(functions) == 1, 'teacher agent entry differs')
    agent = functions[0]
    args = agent.args
    _require([item.arg for item in args.args] == ['observation', 'configuration']
             and not (args.posonlyargs or args.vararg or args.kwonlyargs or args.kwarg or args.defaults),
             'teacher agent signature differs')
    calls = [node for node in ast.walk(tree) if _task_call(node)]
    _require(len(calls) == 1 and len(calls[0].args) == 1 and not calls[0].keywords,
             'teacher task choice point differs')
    assignments = [node for node in ast.walk(agent) if isinstance(node, ast.Assign) and node.value is calls[0]]
    _require(len(assignments) == 1, 'teacher task assignment differs')
    assignment = assignments[0]
    # Pass the source's actual max binding. This preserves ordinary Python name
    # resolution even if a future frozen source shadows the builtin.
    assignment.value = ast.Call(func=ast.Name(id='_probe_task_select', ctx=ast.Load()),
        args=[ast.Call(func=ast.Name(id='locals', ctx=ast.Load()), args=[], keywords=[]),
              ast.Name(id='max', ctx=ast.Load())], keywords=[])
    try:
        return compile(ast.fix_missing_locations(tree), '<private-frozen-teacher>', 'exec')
    except (SyntaxError, ValueError, TypeError):
        raise ProbeError('teacher source compilation failed') from None


class TeacherProbe:
    """One stateless-at-call-boundary, hash-pinned rule teacher probe."""

    def __init__(self, source_bytes, expected_sha256, alias, *, guard=lambda: None):
        _require(type(source_bytes) is bytes and 0 < len(source_bytes) <= MAX_SOURCE_BYTES
                 and type(expected_sha256) is str and re.fullmatch('[0-9a-f]{64}', expected_sha256)
                 and hashlib.sha256(source_bytes).hexdigest() == expected_sha256
                 and type(alias) is str and alias in ALIASES and callable(guard), 'teacher source identity differs')
        self.alias, self._guard, self._running = alias, guard, False
        self._records, self.last = [], None
        code = _instrument(source_bytes)
        callback = self._select
        namespace = {'__name__': 'private_frozen_teacher', RESERVED: callback}
        exec(code, namespace)
        _require(namespace.get(RESERVED) is callback and callable(namespace.get('agent')),
                 'teacher agent entry or callback identity differs')
        self._agent = namespace['agent']

    def __repr__(self):
        return '<PrivateTeacherProbe>'

    def clear(self):
        """Release this probe's current private records when no turn is active.

        Callers must also discard any record references they retained.
        """
        _require(not self._running, 'cannot clear an active teacher probe')
        self.last, self._records = None, []

    def _select(self, context, original_max):
        _require(self._running and callable(original_max), 'teacher callback outside a turn')
        self._guard()
        _require(len(self._records) < MAX_CALLBACKS_PER_TURN, 'teacher callback limit exceeded')
        _require(type(context) is dict and type(context.get('choices')) is list
                 and 0 < len(context['choices']) <= MAX_CANDIDATES,
                 'teacher candidate width exceeds limit')
        before = _watched(context)
        choice = original_max(context['choices'])
        _require(type(choice) is tuple and len(choice) >= 2, 'invalid teacher choice')
        unit, task_index = choice[-2:]
        _require(type(unit) is int and type(task_index) is int and type(context['tasks']) is list
                 and 0 <= unit < len(context['units']) and 0 <= task_index < len(context['tasks']),
                 'teacher choice is outside current context')
        task = context['tasks'][task_index]
        _require(type(task) in (tuple, list) and len(task) == 3, 'invalid teacher task')
        target, operation, _priority = task
        _require(type(target) in (tuple, list) and len(target) == 2
                 and all(type(value) is int for value in target)
                 and type(operation) is list and operation and operation[0] in OPS,
                 'invalid teacher candidate')
        label = candidate_id(unit, target, operation[0])
        # ``locals()`` also includes live state such as demand and planning
        # estimates that the encoder does not currently use. Never expose a
        # live mutable reference, even for an unlisted future encoder field.
        # Exclude functions (notably crop_score, which closes over live state).
        isolated = deepcopy({key: value for key, value in context.items() if not callable(value)})
        isolated_before = _watched(isolated)
        inputs, lookup = encode_candidates(isolated)
        _require(_watched(isolated) == isolated_before, 'feature encoding mutated copied reservations')
        _require(_watched(context) == before, 'feature encoding mutated controller state')
        _require(type(inputs) is dict and inputs.get('schema_version') == SCHEMA_VERSION
                 and inputs.get('candidate_scope') == CANDIDATE_SCOPE
                 and type(inputs.get('candidates')) is list and len(inputs['candidates']) == len(context['choices'])
                 and type(lookup) is dict and len(lookup) == len(context['choices']),
                 'teacher candidate encoding differs')
        candidates = inputs['candidates']
        ids = [item.get('id') if type(item) is dict else None for item in candidates]
        _require(all(type(item) is str and item for item in ids) and len(set(ids)) == len(ids)
                 and set(ids) == set(lookup) and label in lookup and _snapshot(lookup[label]) == _snapshot(choice),
                 'teacher label not in current candidates')
        selected = candidates[ids.index(label)]
        _require(selected.get('unit') == unit and selected.get('target') == list(target)
                 and selected.get('op') == operation[0], 'teacher candidate identity differs')
        self._records.append(dict(inputs=inputs, teacher_label=label, chosen=label, unit=unit,
                                  target=list(target), op=operation[0], source='teacher'))
        self._guard()
        return choice

    def __call__(self, observation, configuration):
        _require(not self._running, 'teacher probe is already active')
        self.last, self._records = None, []
        self._running = True
        try:
            self._guard()
            _require(type(observation) is dict and type(configuration) is dict
                     and type(observation.get('player')) is int and observation['player'] in (0, 1)
                     and type(observation.get('step')) is int and 0 <= observation['step'] < 719,
                     'invalid single-player observation')
            view = observe(observation, configuration)
            action = self._agent(view.observation, view.configuration)
            _require(type(action) is dict and set(action) == {'farmer', 'hands', 'market'}
                     and all(type(action[key]) is list for key in ('farmer', 'hands', 'market')),
                     'teacher returned an invalid joint request')
            units = [action['farmer'], *action['hands']]
            for record in self._records:
                _require(0 <= record['unit'] < len(units), 'teacher unit request missing')
                record['unit_request'] = deepcopy(units[record['unit']])
            self._guard()
            self.last = dict(schema_version=SCHEMA_VERSION, step=view.observation['step'],
                             teacher_alias=self.alias, records=self._records)
            return deepcopy(action)
        except BaseException:
            self.last, self._records = None, []
            raise
        finally:
            self._running = False
