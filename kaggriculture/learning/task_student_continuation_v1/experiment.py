"""One bounded diagnostic on already-consumed official replay/trace sources.

The preparatory binding and output are private. Only the result projection is
eligible for review and publication after a separate privacy check. This entry
never trains or runs a complete game. Historical binding validation indirectly
parses the private exclusion inventory containing reserved seed values; those
values are not used for policy input, sampling, or reserved-panel matches.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import secrets
import subprocess
import time

from . import analysis, samples
from learning.task_pair_market_consumed_v1.entry import physical_available


REPO = Path(__file__).resolve().parents[3]
PRIVATE = REPO.parent / 'private' / 'task-student-continuation-v1-20261031'
FRESH = REPO.parent / 'private' / 'task-pair-fresh-multi-v1-20261031'
MARKET = REPO.parent / 'private' / 'task-pair-market-consumed-v1-20261031'
REGISTRATION = REPO / 'review/evidence/task-student-continuation-v1-20261031/registration.json'
NOTE = REPO / 'review/evidence/task-student-continuation-v1-20261031/REGISTRATION.zh-CN.md'
BASE_COMMIT = '0e9734a0109a0ee4822ef1310e8b56fac21eaa48'
FRESH_COMMIT = '40f3f393819bbdc62af1007877cd5fbdf3a9a42f'
OFFICIAL_CORE_SHA256 = '0922c4599a1b6e0d8c3dadf06ae5297f98138d859d686f00daee6f36e6d45d0e'
SOURCE_PREFIX = 'kaggriculture/learning/task_student_continuation_v1/'
MAX_METADATA = 16 * 1024 * 1024
INITIAL_MEMORY, RUN_MEMORY = 5 * 1024**3, 4 * 1024**3
WALL_SECONDS = 1800
MAX_COMPARISONS, MAX_CALLS, MAX_TRANSITIONS = 48, 96, 96


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _read(path):
    with Path(path).open('rb') as source:
        raw = source.read(MAX_METADATA + 1)
    if not raw or len(raw) > MAX_METADATA:
        raise ValueError('Missing or oversized private metadata')
    return raw


def _document(raw):
    def pairs(values):
        result = {}
        for key, value in values:
            if key in result:
                raise ValueError('Duplicate metadata key')
            result[key] = value
        return result
    def invalid(_value):
        raise ValueError('Nonfinite metadata')
    value = json.loads(raw.decode('utf-8-sig'), object_pairs_hook=pairs, parse_constant=invalid)
    if type(value) is not dict:
        raise ValueError('Metadata must be an object')
    return value


def _encoded(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + '\n').encode('utf-8')


def _write_new(path, value):
    with Path(path).open('xb') as target:
        raw = _encoded(value)
        if target.write(raw) != len(raw):
            raise OSError('Incomplete local evidence write')


def _git(*args):
    return subprocess.check_output(['git', '-C', str(REPO), *args], stderr=subprocess.DEVNULL).decode().strip()


def prepare():
    """Bind already-existing private evidence with a secret salt before running."""
    if _git('rev-parse', 'HEAD') != BASE_COMMIT or REGISTRATION.exists() or PRIVATE.exists():
        raise ValueError('Unexpected preparation state')
    old = _document(subprocess.check_output(['git', '-C', str(REPO), 'show',
        FRESH_COMMIT + ':review/evidence/task-pair-fresh-multi-v1-20261031/registration.json']))
    fresh_raw = _read(FRESH / 'binding.local.json')
    fresh = _document(fresh_raw)
    market_raw = _read(MARKET / 'binding.local.json')
    campaign_raw, closed_raw = _read(FRESH / 'campaign-result.local.json'), _read(FRESH / 'closed.local.json')
    campaign, closed = _document(campaign_raw), _document(closed_raw)
    if (old.get('binding_commitment_sha256') != _sha(fresh_raw)
            or fresh.get('market_binding_sha256') != _sha(market_raw)
            or campaign.get('registration_commit') != FRESH_COMMIT
            or campaign.get('binding_sha256') != _sha(fresh_raw)
            or (campaign.get('games'), campaign.get('official_replay_passes')) != (32, 32)
            or closed.get('registration_commit') != FRESH_COMMIT
            or closed.get('binding_sha256') != _sha(fresh_raw)
            or closed.get('outcome') != 'completed' or closed.get('retry_allowed') is not False):
        raise ValueError('Prior completed campaign binding differs')
    binding = dict(schema='student-continuation-private-binding-v1', salt=secrets.token_hex(32),
                   prior_commit=BASE_COMMIT, fresh_registration_commit=FRESH_COMMIT,
                   fresh_binding_sha256=_sha(fresh_raw), market_binding_sha256=_sha(market_raw),
                   campaign_result_sha256=_sha(campaign_raw), closed_sha256=_sha(closed_raw))
    raw = _encoded(binding)
    registration = dict(schema='student-continuation-prereg-v1', base_commit=BASE_COMMIT,
        fresh_registration_commit=FRESH_COMMIT, binding_commitment_sha256=_sha(raw),
        source='eight selected games from the consumed 32-game exploration',
        roles=list(samples.ROLES), opponents=list(samples.OPPONENTS), windows=[list(w) for w in samples.WINDOWS],
        max_games_read=8, max_windows=24, max_comparisons=MAX_COMPARISONS,
        max_controller_calls=MAX_CALLS, max_official_transitions=MAX_TRANSITIONS,
        max_wall_seconds=WALL_SECONDS, initial_memory_bytes=INITIAL_MEMORY, run_memory_bytes=RUN_MEMORY,
        candidate_width_max=256, official_core_sha256=OFFICIAL_CORE_SHA256,
        max_attempts=1, training=False, new_complete_games=0,
        independent_evaluation=False, promotion=False)
    PRIVATE.mkdir(parents=False, exist_ok=False)
    with (PRIVATE / 'binding.local.json').open('xb') as target:
        if target.write(raw) != len(raw):
            raise OSError('Incomplete binding write')
    _write_new(REGISTRATION, registration)
    print('Private binding and public exploratory registration prepared.')


def _runtime_binding():
    head = _git('rev-parse', 'HEAD')
    if (_git('status', '--porcelain') or _git('rev-parse', 'origin/main') != head
            or _git('rev-parse', 'HEAD^') != BASE_COMMIT
            or _git('merge-base', '--is-ancestor', BASE_COMMIT, head)):
        raise ValueError('The exact preregistered commit must be clean and published')
    reg_raw = _read(REGISTRATION)
    committed = subprocess.check_output(['git', '-C', str(REPO), 'show',
                                         head + ':' + REGISTRATION.relative_to(REPO).as_posix()])
    if reg_raw != committed:
        raise ValueError('Registration differs from committed bytes')
    reg = _document(reg_raw)
    expected = dict(schema='student-continuation-prereg-v1', base_commit=BASE_COMMIT,
        fresh_registration_commit=FRESH_COMMIT, roles=list(samples.ROLES), opponents=list(samples.OPPONENTS),
        windows=[list(w) for w in samples.WINDOWS], max_games_read=8, max_windows=24,
        max_comparisons=MAX_COMPARISONS, max_controller_calls=MAX_CALLS,
        max_official_transitions=MAX_TRANSITIONS, max_wall_seconds=WALL_SECONDS,
        initial_memory_bytes=INITIAL_MEMORY, run_memory_bytes=RUN_MEMORY,
        candidate_width_max=256, official_core_sha256=OFFICIAL_CORE_SHA256,
        max_attempts=1, training=False, new_complete_games=0,
        independent_evaluation=False, promotion=False)
    if any(reg.get(k) != v for k, v in expected.items()):
        raise ValueError('Registration scope differs')
    binding_raw = _read(PRIVATE / 'binding.local.json')
    if _sha(binding_raw) != reg.get('binding_commitment_sha256'):
        raise ValueError('Private binding commitment differs')
    binding = _document(binding_raw)
    if (binding.get('schema') != 'student-continuation-private-binding-v1'
            or binding.get('prior_commit') != BASE_COMMIT or binding.get('fresh_registration_commit') != FRESH_COMMIT
            or binding.get('fresh_binding_sha256') != _sha(_read(FRESH / 'binding.local.json'))
            or binding.get('market_binding_sha256') != _sha(_read(MARKET / 'binding.local.json'))
            or binding.get('closed_sha256') != _sha(_read(FRESH / 'closed.local.json'))):
        raise ValueError('Historical private evidence differs')
    return head, binding


def _guard(deadline):
    if time.monotonic() >= deadline:
        raise TimeoutError('Diagnostic wall budget exhausted')
    if physical_available() < RUN_MEMORY:
        raise MemoryError('Diagnostic memory floor reached')


def run():
    """Run the fixed, single-attempt consumed-data diagnostic."""
    if os.name != 'nt' or physical_available() < INITIAL_MEMORY:
        raise MemoryError('Windows initial memory floor not met')
    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[name] = '1'
    head, binding = _runtime_binding()
    deadline = time.monotonic() + WALL_SECONDS
    guard = lambda: _guard(deadline)
    from decision.api import observe
    from decision.audit_market_teacher import observation
    from learning.task_dagger_once_v1.policy import BoundedAgent
    from learning.task_pair_dev_games_v1.panel import GAME_LIMITS
    from . import compare, official_step, sources

    guard()
    context = sources.open_context(campaign_sha256=binding['campaign_result_sha256'], guard=guard)
    _write_new(PRIVATE / 'attempt.local.json', dict(schema='student-continuation-attempt-v1',
        commit=head, max_comparisons=MAX_COMPARISONS, retry_allowed=False))
    rows = []
    comparisons = calls = transitions = 0
    try:
        for role in samples.ROLES:
            for opponent in samples.OPPONENTS:
                guard()
                with sources.read_game(context, role=role, opponent=opponent, guard=guard) as game:
                    identity = game.identity
                    if identity['game_limits'] != GAME_LIMITS:
                        raise ValueError('Original game limits differ')
                    for window, selected in enumerate(game.selected):
                        guard()
                        row = {**samples.source_coordinates(role, opponent), 'window': window,
                               'selection': 'empty' if selected is None else 'selected',
                               'teacher': None, 'third': None}
                        if selected is None:
                            rows.append(row)
                            continue
                        record = selected['document']['records'][selected['ordinal']]
                        width = len(record['inputs']['candidates'])
                        if width > 256:
                            raise ValueError('Candidate width exceeds registered resource budget')
                        step, side = selected['step'], identity['side']
                        row.update(step=step, ordinal=selected['ordinal'], candidate_width=width)
                        original = game.replay['steps'][step + 1][side]['action']
                        view = observe(observation(game.replay['steps'][step], side),
                                       game.replay['configuration'])
                        if 'seed' in view.configuration or 'info' in view.observation:
                            raise ValueError('Policy view contains privileged metadata')

                        def factory():
                            return BoundedAgent(identity['crop_path'], identity['crop_sha256'],
                                                identity['task_path'], identity['task_sha256'],
                                                limits=GAME_LIMITS)

                        for label, alternative in (('teacher', 'teacher'), ('third', selected['third_candidate'])):
                            if alternative is None:
                                continue
                            if comparisons >= MAX_COMPARISONS or calls + 2 > MAX_CALLS or transitions + 2 > MAX_TRANSITIONS:
                                raise ValueError('Registered work limit exceeded')
                            comparisons += 1
                            calls += 2
                            result = compare.compare_turn(selected['document'], view.observation,
                                view.configuration, original, selected['ordinal'],
                                task_sha256=identity['task_sha256'], crop_sha256=identity['crop_sha256'],
                                game_limits=GAME_LIMITS, agent_factory=factory, guard=guard,
                                previous_document=selected['previous_document'], alternative=alternative)
                            if result.reference_action is None or result.alternative_action is None:
                                raise ValueError('Comparison lacks verified complete actions')
                            transitions += 2
                            factual = official_step.compare_step(game.replay, step=step, side=side,
                                baseline_request=result.reference_action,
                                intervention_request=result.alternative_action, guard=guard)
                            public = result.public_summary()
                            public['official'] = factual
                            row[label] = public
                        rows.append(row)
                        del record, selected, original, view, result, factory
                    del identity
                del game
        guard()
        summary = analysis.summarize(rows)
        summary.update(comparisons=comparisons, controller_call_limit_reserved=calls,
                       official_transition_limit_reserved=transitions, source_games=8,
                       registration_commit=head)
        _write_new(PRIVATE / 'result.local.json', dict(summary=summary, rows=rows))
        _write_new(PRIVATE / 'closed.local.json', dict(outcome='completed', retry_allowed=False))
        print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    except BaseException as exc:
        _write_new(PRIVATE / 'failed.local.json', dict(outcome='failed', error_type=type(exc).__name__,
            comparisons_started=comparisons, controller_calls_reserved=calls,
            official_transitions_reserved=transitions, retry_allowed=False))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare', 'run'))
    args = parser.parse_args()
    prepare() if args.command == 'prepare' else run()


if __name__ == '__main__':
    main()
