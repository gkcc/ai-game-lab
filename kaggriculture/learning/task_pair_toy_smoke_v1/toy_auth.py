"""Authenticate a separate, explicitly registered synthetic two-game smoke.

Only the standard library is imported.  The caller must independently
authenticate this helper and the supplied, exclusively owned identity module
before invoking it.  The identity module verifies committed source, historic
lineage, runtime and official rules; this helper constrains the new toy
registration and restores the module even on failure.  No files are read here
except by the caller-supplied verifier.  This is not a cross-process lock and
does not start games, load weights, or verify resource limits.
"""
import re
import threading
from pathlib import Path
from types import ModuleType


REGISTRATION = 'review/evidence/task-pair-toy-smoke-v1-20261030/registration.json'
STAGE_NAME = 'task-pair-toy-smoke-v1-20261030'
SCHEMA = 'task-pair-toy-smoke-prereg-v1'
FAILED_REGISTRATION_COMMIT = 'b97687f6eefd93c8caf390e8b36775f20a9d190d'
FAILURE_CERTIFICATE_COMMIT = 'eb20f608d4961d2bab5aad2acadb3c91562c2061'
FAILED_REGISTRATION_FILES = (
    'review/evidence/task-pair-dev-games-v1-20261030/registration.json',
    'review/evidence/task-pair-dev-games-v1-20261030/REGISTRATION.zh-CN.md',
)
FAILURE_CERTIFICATE_FILES = (
    'review/evidence/task-pair-dev-games-v1-failure-20261030/certificate.json',
    'review/evidence/task-pair-dev-games-v1-failure-20261030/FAILURE.zh-CN.md',
)
EXTRA_SOURCES = frozenset((
    'kaggriculture/learning/task_pair_toy_smoke_v1/__init__.py',
    'kaggriculture/learning/task_pair_toy_smoke_v1/toy_auth.py',
    'kaggriculture/learning/task_pair_toy_smoke_v1/toy_weights.py',
    'kaggriculture/learning/task_pair_toy_smoke_v1/runner.py',
    'kaggriculture/learning/task_pair_toy_smoke_v1/entry.py',
    'review/evidence/bc-crop-v1-20261023/data-manifest.json',
    'review/evidence/task-bc-v1-20261023-run1/cache/schema.json',
    'kaggriculture/bots/heuristic_v1_market.py',
    'kaggriculture/bots/heuristic_v1.py',
    *FAILED_REGISTRATION_FILES,
    *FAILURE_CERTIFICATE_FILES,
))
_SHA256 = re.compile(r'[0-9a-f]{64}\Z')
_LOCK = threading.Lock()


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _typed_equal(value, expected):
    """Exact JSON structure and types: in particular bool is never an int."""
    if type(value) is not type(expected):
        return False
    if type(expected) is dict:
        return (value.keys() == expected.keys()
                and all(_typed_equal(value[key], item) for key, item in expected.items()))
    if type(expected) is list:
        return len(value) == len(expected) and all(_typed_equal(a, b) for a, b in zip(value, expected))
    return value == expected


def _panel(reg, identity, required):
    """Validate the complete toy plan and mandatory source identities."""
    _require(type(reg) is dict, 'Invalid toy registration')
    panel = dict(synthetic_seeds={'S02': 11017, 'S03': 22029},
                 active_seed_role='S02', active_opponent='market', active_side=0,
                 active_policies=['baseline', 'control'], games=2,
                 scope='public_synthetic_not_previous_panel',
                 placeholder_roles=['S03', 'auxiliary', 'round2'])
    evaluation = dict(official_games=2, official_replays=2, comparisons=1,
                      retry_allowed=False, training_updates=0, promotion=False,
                      original_24_panel_resumed=False)
    _require(_typed_equal(reg.get('schema'), SCHEMA)
             and _typed_equal(reg.get('stage_name'), STAGE_NAME)
             and _typed_equal(reg.get('failed_registration_commit'), FAILED_REGISTRATION_COMMIT)
             and _typed_equal(reg.get('failure_certificate_commit'), FAILURE_CERTIFICATE_COMMIT)
             and _typed_equal(reg.get('panel'), panel)
             and _typed_equal(reg.get('evaluation'), evaluation),
             'Toy panel or evaluation differs')
    fields = {'engine_commit': 'ENGINE_COMMIT', 'engine_sha256': 'ENGINE_SHA',
              'previous_training_result_commit': 'RESULT_COMMIT',
              'historical_panel_result_commit': 'HISTORICAL_RESULT'}
    _require(all(type(reg.get(key)) is str and type(getattr(identity, attr, None)) is str
                 and reg[key] == getattr(identity, attr) for key, attr in fields.items()),
             'Toy historical or engine identity differs')
    models = reg.get('toy_model_sha256')
    _require(type(models) is dict and set(models) == {'crop', 'baseline', 'control', 'auxiliary'}
             and all(type(value) is str and _SHA256.fullmatch(value) is not None
                     for value in models.values())
             and len(set(models.values())) == 4, 'Invalid synthetic model identities')
    sources = reg.get('public_sources')
    _require(type(sources) is dict and required.issubset(sources), 'Toy sources missing')


def _failure_lineage(identity, repo, registration_commit):
    """Bind the closed attempt to its original committed public evidence."""
    for commit, paths in ((FAILED_REGISTRATION_COMMIT, FAILED_REGISTRATION_FILES),
                          (FAILURE_CERTIFICATE_COMMIT, FAILURE_CERTIFICATE_FILES)):
        identity._git(repo, 'merge-base', '--is-ancestor', commit, registration_commit)
        for relative in paths:
            committed = identity._git(repo, 'show', commit + ':' + relative)
            current = identity._read(identity._file(Path(repo), relative))
            _require(committed == current, 'Closed development evidence differs')


def verify_public(identity, *, repo, game_root, registration_commit):
    """Verify a toy registration with a separately authenticated identity module.

    The caller supplies an independent ``ModuleType`` and must not use it in a
    different thread or process during the call.  The verifier's three mutable
    attributes are restored in ``finally`` even if verification is interrupted.
    Its returned ``(registration, historical_registration, freeze)`` is passed
    through after another structural check.  Model bytes and resource limits
    still need their separate caller-side verification.
    """
    _require(type(identity) is ModuleType, 'Independent identity module required')
    _require(_LOCK.acquire(blocking=False), 'Toy identity verification is already active')
    try:
        old_registration = identity.REGISTRATION
        old_panel = identity._panel
        old_required = identity._REQUIRED
        _require(type(old_registration) is str and old_registration != REGISTRATION
                 and callable(old_panel) and type(old_required) is frozenset
                 and callable(identity.verify_public), 'Invalid identity module')
        required = old_required | EXTRA_SOURCES
        try:
            identity.REGISTRATION = REGISTRATION
            identity._panel = lambda reg: _panel(reg, identity, required)
            identity._REQUIRED = required
            result = identity.verify_public(repo=repo, game_root=game_root,
                                            registration_commit=registration_commit)
            _require(type(result) is tuple and len(result) == 3, 'Invalid verifier result')
            _panel(result[0], identity, required)
            _failure_lineage(identity, repo, registration_commit)
            return result
        finally:
            identity.REGISTRATION = old_registration
            identity._panel = old_panel
            identity._REQUIRED = old_required
    finally:
        _LOCK.release()
