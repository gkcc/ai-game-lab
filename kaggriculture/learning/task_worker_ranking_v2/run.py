"""Run the separately registered ranking comparison with exact Git bytes.

The frozen v1 data, eligibility, scoring and result functions are reused. No
source, model or output from the old failed attempt is changed or reopened.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import types


GAME_SOURCE = Path(__file__).resolve().parents[2]
REPO = GAME_SOURCE.parent
REGISTRATION = REPO / 'review/evidence/task-worker-ranking-v2-20261030/registration.json'
V1_RELATIVE = 'kaggriculture/learning/task_worker_ranking_v1/run.py'
V1_SHA256 = '1a071a23064e4f1e13ae146805123ef898a5ee3de2b9427388258a6acb30e4c2'
V2_SOURCE_PATHS = frozenset((
    'kaggriculture/learning/task_worker_ranking_v2/__init__.py',
    'kaggriculture/learning/task_worker_ranking_v2/run.py',
    'kaggriculture/learning/task_worker_ranking_v2/test_run.py',
    'kaggriculture/learning/task_worker_ranking_v2/README.zh-CN.md',
))
REGISTRATION_CONTRACT = {
    'schema': 'task-worker-ranking-registration-v2',
    'scope': {
        'source': 'two_frozen_sampled_conditional_teacher_archives',
        'encoded_steps': 'all',
        'purpose': 'development_only',
        'new_training': False,
        'new_games': False,
    },
    'selection': {
        'teacher_operation': 'PLANT',
        'candidate_operation': 'PLANT',
        'target': 'same_as_teacher',
        'min_distinct_units': 2,
        'score_input': 'entire_candidate_group',
        'max_candidates_per_group': 256,
        'tie_break': 'first_canonical_candidate',
    },
    'gate': {
        'min_eligible_groups': 20,
        'source_clusters_exact': 4,
        'min_eligible_groups_per_cluster': 5,
        'no_models_if_insufficient': True,
    },
    'model_a': 'early_bc',
    'model_b': 'merged_bc',
    'metrics': {
        'cluster_aggregation': 'group_mean_within_cluster_then_equal_clusters',
        'primary': 'conditional_teacher_top1',
        'direction': 'model_a_minus_model_b',
        'global_guard': {
            'metrics': ['overall_teacher_top1', 'overall_same_target_plant'],
            'minimum_difference_pp': -5,
            'comparison': 'greater_or_equal',
            'role': 'exploratory',
        },
        'primary_difference_bands': ['<-20pp', '[-20,-5)pp', '[-5,+5]pp',
                                     '(+5,+20]pp', '>+20pp'],
        'public_statistics': ['conditional_teacher_top1_difference_band', 'global_guard'],
    },
    'minimum_initial_available_bytes': 4294967296,
    'groups_expected': 5104,
    'candidates_expected': 322151,
    'max_model_calls': 10208,
    'max_scored_rows': 644302,
    'sampled_deadline_seconds': 300,
}
REGISTRATION_LIMIT = 4 * 1024 * 1024
STAGES = frozenset(('arguments', 'registration', 'sources', 'guard',
                    'output', 'evaluation', 'publish'))
CURRENT_STAGE = 'arguments'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def set_stage(stage):
    global CURRENT_STAGE
    require(stage in STAGES, 'Invalid safe stage')
    CURRENT_STAGE = stage


def strict_contract_equal(value, expected):
    """Compare JSON trees without treating bool, float and int as equivalent."""
    if type(value) is not type(expected):
        return False
    if isinstance(expected, dict):
        return (value.keys() == expected.keys()
                and all(strict_contract_equal(value[k], expected[k]) for k in expected))
    if isinstance(expected, list):
        return (len(value) == len(expected)
                and all(strict_contract_equal(a, b) for a, b in zip(value, expected)))
    return value == expected


def verify_registration_commit(commit):
    """Require HEAD and compare committed and worktree registration bytes exactly."""
    require(isinstance(commit, str) and len(commit) == 40
            and all(c in '0123456789abcdef' for c in commit), 'Invalid registration commit')
    head = subprocess.check_output(['git', '-C', str(REPO), 'rev-parse', 'HEAD'],
                                   stderr=subprocess.DEVNULL).decode().strip()
    require(head == commit, 'Registration commit is not HEAD')
    relative = REGISTRATION.relative_to(REPO).as_posix()
    registered = subprocess.check_output(['git', '-C', str(REPO), 'show', commit + ':' + relative],
                                         stderr=subprocess.DEVNULL)
    with REGISTRATION.open('rb') as stream:
        workspace = stream.read(REGISTRATION_LIMIT + 1)
    require(len(registered) <= REGISTRATION_LIMIT and len(workspace) <= REGISTRATION_LIMIT,
            'Registration exceeds size limit')
    require(registered == workspace, 'Registration is not committed exactly')
    return workspace


def parse_registration(raw):
    """Reject duplicate keys and nonstandard numeric constants at any depth."""
    require(isinstance(raw, bytes) and len(raw) <= REGISTRATION_LIMIT,
            'Invalid registration bytes')

    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, 'Duplicate registration field')
            result[key] = value
        return result

    def reject_constant(value):
        raise ValueError('Invalid registration numeric constant')

    try:
        return json.loads(raw.decode('utf-8-sig'), object_pairs_hook=unique_object,
                          parse_constant=reject_constant)
    except (UnicodeDecodeError, ValueError, TypeError):
        raise ValueError('Invalid registration encoding') from None


def load_frozen_v1():
    """Compile exactly the SHA-bound v1 entry into its own non-main namespace."""
    path = REPO / V1_RELATIVE
    require(path.is_file() and path.resolve(strict=True).is_relative_to(REPO.resolve(strict=True)),
            'Frozen entry is missing or outside repository')
    raw = path.read_bytes()
    require(hashlib.sha256(raw).hexdigest() == V1_SHA256, 'Frozen entry identity differs')
    module = types.ModuleType('task_worker_ranking_v1_frozen_entry')
    module.__file__ = str(path)
    exec(compile(raw, '<frozen-task-worker-ranking-v1>', 'exec', dont_inherit=True), module.__dict__)
    return module


def verify_sources(registration, frozen):
    """Bind all v2 files before adapting the immutable v1 source validator."""
    require(type(registration) is dict
            and set(registration) == set(REGISTRATION_CONTRACT) | {'source_sha256'}
            and all(strict_contract_equal(registration[name], expected)
                    for name, expected in REGISTRATION_CONTRACT.items()),
            'Registration contract differs')
    require(V2_SOURCE_PATHS.isdisjoint(frozen.SOURCE_PATHS), 'Source identity sets overlap')
    entries = registration.get('source_sha256')
    require(isinstance(entries, dict) and set(entries) == frozen.SOURCE_PATHS | V2_SOURCE_PATHS,
            'Registered source identity set differs')
    require(entries.get(V1_RELATIVE) == V1_SHA256, 'Frozen entry registration differs')
    for relative in V2_SOURCE_PATHS:
        expected = entries[relative]
        require(isinstance(expected, str) and len(expected) == 64
                and all(c in '0123456789abcdef' for c in expected), 'Invalid registered source identity')
        require(frozen.sha(frozen.local_file(REPO, relative)) == expected,
                'Registered source identity changed')

    # The v2 contract has checked the declared behavior. The v1 validator
    # checks its original source set and fixed numeric budgets; model bytes are
    # checked later, if the coverage gate permits models to be loaded.
    adapted = dict(registration)
    adapted['schema'] = 'task-worker-ranking-registration-v1'
    adapted['source_sha256'] = {name: entries[name] for name in frozen.SOURCE_PATHS}
    frozen.verify_sources(adapted)
    return adapted


class QuietParser(argparse.ArgumentParser):
    def error(self, message):
        raise ValueError('Invalid command line')


def main():
    set_stage('arguments')
    parser = QuietParser(description=__doc__)
    parser.add_argument('--game-root', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--registration-commit', required=True)
    args = parser.parse_args()

    set_stage('registration')
    registered = verify_registration_commit(args.registration_commit)
    registration = parse_registration(registered)
    set_stage('sources')
    frozen = load_frozen_v1()
    adapted = verify_sources(registration, frozen)
    set_stage('guard')
    frozen.bind_runtime(adapted)
    frozen.reject_shadows(GAME_SOURCE / 'learning' / 'task_worker_ranking_v2', ('run',))

    root = args.game_root.resolve(strict=True)
    artifacts = (root / 'artifacts').resolve(strict=True)
    frozen.require_direct_artifacts(root, artifacts)
    default_bot = frozen.local_file(root, 'main.py')
    require(frozen.sha(default_bot) == registration['source_sha256']['kaggriculture/main.py'],
            'Local default Bot differs')
    set_stage('output')
    output = args.output_dir.resolve(strict=False)
    require(output.parent == artifacts and output.name == 'task-worker-ranking-v2-20261030',
            'Output is outside the registered location')
    output.mkdir(exist_ok=False)
    try:
        with (output / 'started.json').open('x', encoding='utf-8') as stream:
            json.dump({'schema': 'task-worker-ranking-start-v2'}, stream)
        set_stage('evaluation')
        result = frozen.run(root, adapted)
        require(isinstance(result, dict) and result.get('schema') == 'task-worker-ranking-result-v1',
                'Frozen result schema differs')
        set_stage('guard')
        require(frozen.sha(default_bot) == registration['source_sha256']['kaggriculture/main.py'],
                'Local default Bot changed')
        set_stage('registration')
        require(verify_registration_commit(args.registration_commit) == registered,
                'Registration changed during run')
        set_stage('sources')
        verify_sources(registration, frozen)
        result = dict(result, schema='task-worker-ranking-result-v2',
                      registration_commit=args.registration_commit)
        set_stage('publish')
        frozen.publish_result(output, result)
        print(json.dumps({'status': 'completed', 'coverage': result['coverage']}))
    except Exception as exc:
        try:
            with (output / 'failure.json').open('x', encoding='utf-8') as stream:
                json.dump({'schema': 'task-worker-ranking-failure-v2',
                           'stage': CURRENT_STAGE, 'error_type': type(exc).__name__}, stream)
        except OSError:
            # Preserve the original failure if the local certificate cannot be written.
            pass
        raise


def cli():
    """Emit only an exception class, never argument or local-path text."""
    set_stage('arguments')
    try:
        main()
    except Exception as exc:
        print(json.dumps({'status': 'failed', 'stage': CURRENT_STAGE,
                          'error_type': type(exc).__name__}), file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(cli())
