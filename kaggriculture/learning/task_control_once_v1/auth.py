"""Authenticate a separately registered, one-game development scope.

This helper does not read private seeds/models or start a game. The caller
authenticates this file and supplies an isolated identity module. Existing
closed attempts remain immutable; structural Panel aliases are explicit.
"""
import hashlib
from pathlib import Path
import re
import threading
from types import ModuleType


REGISTRATION = 'review/evidence/task-control-once-v1-20261030/registration.json'
STAGE_NAME = 'task-control-once-v1-20261030'
SCHEMA = 'task-control-once-dev-prereg-v1'
PREFIX = 'kaggriculture/learning/task_prefix_diagnostic_v2/runner.py'
OLD_REGISTRATION = 'review/evidence/task-pair-dev-games-v1-20261030/registration.json'
PANEL = dict(scope='previously_consumed_s01_single_development',
             actual_seed_role='S01', actual_opponent='teacher', actual_policy='control', actual_side=0,
             internal_seed_role='S02', internal_opponent='market', internal_policy='control', internal_side=0,
             inactive_seed_role='S03', inactive_actual_seed_role='S02',
             inactive_opponent='round2', inactive_policies=['baseline', 'auxiliary'],
             games=1)
EVALUATION = dict(official_games=1, official_replays=1, comparisons=0, retries=False,
                  new_seed_generation=False, training_updates=0, independent_evaluation=False, promotion=False)
PRIOR = (
    ('b97687f6eefd93c8caf390e8b36775f20a9d190d',
     (OLD_REGISTRATION, 'review/evidence/task-pair-dev-games-v1-20261030/REGISTRATION.zh-CN.md', PREFIX)),
    ('eb20f608d4961d2bab5aad2acadb3c91562c2061',
     ('review/evidence/task-pair-dev-games-v1-failure-20261030/certificate.json',
      'review/evidence/task-pair-dev-games-v1-failure-20261030/FAILURE.zh-CN.md')),
    ('4d7f480fa41cadcbae8427722e600472cfd2fe8f',
     ('review/evidence/task-pair-toy-smoke-v1-20261030/registration.json',
      'review/evidence/task-pair-toy-smoke-v1-20261030/REGISTRATION.zh-CN.md')),
    ('78f14984f32ef68028fd1a8798c51d51e027cb81',
     ('review/evidence/task-pair-toy-smoke-v1-failure-20261030/certificate.json',
      'review/evidence/task-pair-toy-smoke-v1-failure-20261030/FAILURE.zh-CN.md')),
)
EXTRA_SOURCES = frozenset((
    'kaggriculture/learning/task_control_once_v1/__init__.py',
    'kaggriculture/learning/task_control_once_v1/auth.py',
    'kaggriculture/learning/task_control_once_v1/bindings.py',
    'kaggriculture/learning/task_control_once_v1/child_scope.py',
    'kaggriculture/learning/task_control_once_v1/budget_observer.py',
    'kaggriculture/learning/task_control_once_v1/runner.py',
    'kaggriculture/learning/task_control_once_v1/entry.py',
    'kaggriculture/bots/teacher_fertilizer_terminal_v2.py',
    'review/evidence/task-control-once-v1-20261030/REGISTRATION.zh-CN.md',
    *(path for _commit, paths in PRIOR for path in paths),
))
_LOCK = threading.Lock()


def _require(value, message):
    if not value:
        raise ValueError(message)


def _typed_equal(value, expected):
    if type(value) is not type(expected):
        return False
    if type(expected) is dict:
        return value.keys() == expected.keys() and all(_typed_equal(value[k], v) for k, v in expected.items())
    if type(expected) is list:
        return len(value) == len(expected) and all(_typed_equal(a, b) for a, b in zip(value, expected))
    return value == expected


def _panel(reg, identity, required):
    _require(type(reg) is dict and reg.get('schema') == SCHEMA
             and reg.get('stage_name') == STAGE_NAME
             and _typed_equal(reg.get('panel'), PANEL)
             and _typed_equal(reg.get('evaluation'), EVALUATION), 'Single-game registration differs')
    fields = {'engine_commit': 'ENGINE_COMMIT', 'engine_sha256': 'ENGINE_SHA',
              'previous_training_result_commit': 'RESULT_COMMIT', 'historical_panel_result_commit': 'HISTORICAL_RESULT'}
    _require(all(type(reg.get(key)) is str and type(getattr(identity, attr, None)) is str
                 and reg[key] == getattr(identity, attr) for key, attr in fields.items()),
             'Historical or engine identity differs')
    _require(type(reg.get('public_sources')) is dict and required.issubset(reg['public_sources']),
             'Required sources missing')


def _check_prior_checker(reg, prior, source):
    prior_sources = prior.get('public_sources') if type(prior) is dict else None
    new_sources = reg.get('public_sources') if type(reg) is dict else None
    expected = prior_sources.get(PREFIX) if type(prior_sources) is dict else None
    _require(type(expected) is str and re.fullmatch(r'[0-9a-f]{64}', expected) is not None
             and type(new_sources) is dict and new_sources.get(PREFIX) == expected
             and type(source) is bytes and bool(source) and hashlib.sha256(source).hexdigest() == expected,
             'Historical checker identity changed')


def _closed_attempts(identity, repo, registration_commit, reg):
    for commit, paths in PRIOR:
        identity._git(repo, 'merge-base', '--is-ancestor', commit, registration_commit)
        for relative in paths:
            committed = identity._git(repo, 'show', commit + ':' + relative)
            current = identity._read(identity._file(Path(repo), relative))
            _require(committed == current, 'Closed attempt evidence changed')
    prior = identity._json(identity._read(identity._file(Path(repo), OLD_REGISTRATION)))
    source = identity._read(identity._file(Path(repo), PREFIX))
    _check_prior_checker(reg, prior, source)


def verify_public(identity, *, repo, game_root, registration_commit):
    """Delegate full source/engine authentication to an isolated verifier."""
    _require(type(identity) is ModuleType, 'Independent identity module required')
    _require(_LOCK.acquire(blocking=False), 'Single-game verifier already active')
    try:
        old_registration, old_panel, old_required = identity.REGISTRATION, identity._panel, identity._REQUIRED
        _require(type(old_registration) is str and old_registration != REGISTRATION
                 and callable(old_panel) and type(old_required) is frozenset
                 and callable(identity.verify_public), 'Invalid identity module')
        required = old_required | EXTRA_SOURCES
        try:
            identity.REGISTRATION = REGISTRATION
            identity._panel = lambda reg: _panel(reg, identity, required)
            identity._REQUIRED = required
            result = identity.verify_public(repo=repo, game_root=game_root, registration_commit=registration_commit)
            _require(type(result) is tuple and len(result) == 3, 'Invalid verifier result')
            _panel(result[0], identity, required)
            _closed_attempts(identity, repo, registration_commit, result[0])
            return result
        finally:
            identity.REGISTRATION, identity._panel, identity._REQUIRED = old_registration, old_panel, old_required
    finally:
        _LOCK.release()
