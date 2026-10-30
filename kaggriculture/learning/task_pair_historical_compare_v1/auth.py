"""Public preregistration and immutable historical identity checks.

The private model and seed values are authenticated separately by bindings.
This verifies bytes and ancestry, not a third-party timestamp or filesystem lock.
"""
import threading


REGISTRATION = 'review/evidence/task-pair-historical-compare-v1-20261030/registration.json'
STAGE_NAME = 'task-pair-historical-compare-v1-20261030'
SCHEMA = 'task-pair-historical-compare-prereg-v1'
PANEL = dict(actual_seed_roles=['S01', 'S04'], internal_seed_roles=['S02', 'S03'],
             opponents=['market'], policies=['baseline', 'control'], sides=[0, 1],
             games=8, scope='consumed_development_only')
EVALUATION = dict(official_games=8, official_replays=8, comparisons=4,
                  new_seed_generation=False, training_updates=0, retries=False,
                  independent_evaluation=False, statistical_significance=False, promotion=False)
PREFIX = 'kaggriculture/learning/task_pair_historical_compare_v1/'
REQUIRED = frozenset((
    PREFIX + '__init__.py', PREFIX + 'auth.py', PREFIX + 'bindings.py',
    PREFIX + 'entry.py', PREFIX + 'runner.py', PREFIX + 'projection.py', PREFIX + 'request_prefix.py',
    'review/evidence/task-pair-historical-compare-v1-20261030/REGISTRATION.zh-CN.md',
    'review/evidence/task-control-once-v1-20261030/registration.json',
    'review/evidence/task-control-once-v1-result-20261030/RESULTS.zh-CN.md',
    'review/evidence/task-control-once-v1-result-20261030/certificate.json',
))
PRIOR = (
    ('88f74949de33def1adac43799b43ba3d00e518fb',
     ('review/evidence/task-control-once-v1-20261030/registration.json',)),
    ('3e4cc9975664239f7b1038ce57cce3aff4c6afa8',
     ('review/evidence/task-control-once-v1-result-20261030/RESULTS.zh-CN.md',
      'review/evidence/task-control-once-v1-result-20261030/certificate.json')),
)
_LOCK = threading.Lock()


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _panel(reg, identity, once_auth, required):
    _require(type(reg) is dict and reg.get('schema') == SCHEMA
             and reg.get('stage_name') == STAGE_NAME
             and once_auth._typed_equal(reg.get('panel'), PANEL)
             and once_auth._typed_equal(reg.get('evaluation'), EVALUATION),
             'Historical comparison scope differs')
    fields = {'engine_commit': 'ENGINE_COMMIT', 'engine_sha256': 'ENGINE_SHA',
              'previous_training_result_commit': 'RESULT_COMMIT',
              'historical_panel_result_commit': 'HISTORICAL_RESULT'}
    _require(all(type(reg.get(key)) is str and reg[key] == getattr(identity, attr, None)
                 for key, attr in fields.items()), 'Historical identity differs')
    _require(type(reg.get('public_sources')) is dict and required.issubset(reg['public_sources']),
             'Registered sources missing')


def verify_public(identity, once_auth, *, repo, game_root, registration_commit):
    """Use the existing source, rule, package and prior-evidence verifier."""
    _require(_LOCK.acquire(blocking=False), 'Comparison verifier already active')
    try:
        old_registration, old_panel, old_required = identity.REGISTRATION, identity._panel, identity._REQUIRED
        _require(type(old_registration) is str and callable(old_panel)
                 and type(old_required) is frozenset and callable(identity.verify_public)
                 and callable(once_auth._closed_attempts), 'Invalid existing verifier')
        required = old_required | once_auth.EXTRA_SOURCES | REQUIRED
        try:
            identity.REGISTRATION = REGISTRATION
            identity._panel = lambda reg: _panel(reg, identity, once_auth, required)
            identity._REQUIRED = required
            result = identity.verify_public(repo=repo, game_root=game_root,
                                            registration_commit=registration_commit)
            _require(type(result) is tuple and len(result) == 3, 'Invalid verifier result')
            _panel(result[0], identity, once_auth, required)
            once_auth._closed_attempts(identity, repo, registration_commit, result[0])
            for commit, paths in PRIOR:
                identity._git(repo, 'merge-base', '--is-ancestor', commit, registration_commit)
                for relative in paths:
                    original = identity._git(repo, 'show', commit + ':' + relative)
                    current = identity._read(identity._file(repo, relative))
                    _require(original == current, 'Previously completed evidence changed')
            return result
        finally:
            identity.REGISTRATION, identity._panel, identity._REQUIRED = old_registration, old_panel, old_required
    finally:
        _LOCK.release()
