"""Run the registered, CPU-only comparison of existing task-head checkpoints.

Raw archives, cluster identities and model weights stay local.  The result is a
coarse development projection, not an independent game or profit evaluation.
"""
import argparse
import hashlib
import importlib.machinery
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import types

GAME_SOURCE = Path(__file__).resolve().parents[2]
REPO = GAME_SOURCE.parent
REGISTRATION = REPO / 'review/evidence/task-worker-ranking-v1-20261030/registration.json'
SOURCE_PATHS = frozenset((
    'kaggriculture/main.py',
    'kaggriculture/learning/task_worker_ranking_v1/__init__.py',
    'kaggriculture/learning/task_worker_ranking_v1/core.py',
    'kaggriculture/learning/task_worker_ranking_v1/run.py',
    'kaggriculture/learning/task_worker_ranking_v1/test_core.py',
    'kaggriculture/learning/task_worker_ranking_v1/test_run.py',
    'kaggriculture/learning/task_worker_ranking_v1/README.zh-CN.md',
    'kaggriculture/learning/task_worker_geometry_diagnostic_v1/__init__.py',
    'kaggriculture/learning/task_worker_geometry_diagnostic_v1/archive_io.py',
    'kaggriculture/learning/task_worker_geometry_diagnostic_v1/training_summary.py',
    'kaggriculture/learning/task_teacher_choice_bc_v1/numeric.py',
    'kaggriculture/learning/task_teacher_merge_v1/merge.py',
    'kaggriculture/learning/task_choice_v1/__init__.py',
    'kaggriculture/learning/task_choice_v1/features.py',
    'kaggriculture/learning/task_bc_v1/__init__.py',
    'kaggriculture/learning/task_bc_v1/model.py',
    'kaggriculture/learning/bc_crop/features.py',
    'kaggriculture/decision/__init__.py',
    'kaggriculture/decision/api.py',
    'kaggriculture/learning/task_single_intervention_v1/trusted_imports.py',
    'kaggriculture/learning/panel_eval_v3/bootstrap.py',
    'review/evidence/task-choice-v1-20261023/registration.json',
    'review/evidence/task-bc-v1-20261023-run1/registration.json',
    'review/evidence/task-bc-v1-20261023-run1/model/selected-export.json',
    'review/evidence/task-teacher-merge-bc-v1-20261028/registration.json',
    'review/evidence/task-teacher-merge-bc-v1-20261028/model-seal.json',
    'review/evidence/task-teacher-merged-eval-v3-formal-20261030/registration.json',
))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path, maximum=4 * 1024 * 1024):
    with Path(path).open('rb') as stream:
        raw = stream.read(maximum + 1)
    require(len(raw) <= maximum, 'Metadata exceeds size limit')
    return json.loads(raw.decode('utf-8-sig'))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def local_file(root, relative):
    root = Path(root).resolve(strict=True)
    rel = Path(relative)
    require(not rel.is_absolute() and '..' not in rel.parts, 'Invalid local input')
    path = (root / rel).resolve(strict=True)
    require(path.is_relative_to(root) and path.is_file(), 'Input outside local root')
    return path


def require_direct_artifacts(root, resolved_artifacts):
    require(resolved_artifacts == root / 'artifacts' and resolved_artifacts.is_dir(),
            'Artifacts directory is not directly under the local game root')


def source_roles(early, archive):
    old, new = archive['old_train_seeds'], archive['new_train_seeds']
    initial = early['train_seeds'] + early['validation_seeds']
    for items, count in ((old, 2), (new, 2), (initial, 10)):
        require(isinstance(items, list) and len(items) == count
                and all(type(x) is int and 0 <= x <= 0xffffffff for x in items)
                and len(set(items)) == count, 'Invalid registered source roles')
    require(not (set(old) & set(new)) and not ((set(old) | set(new)) & set(initial)),
            'Registered source roles overlap')
    return old, new, initial


def require_early_training_match(demonstration, training):
    require(all(demonstration[key] == training[key] for key in ('train_seeds', 'validation_seeds')),
            'Early model training roles differ from demonstration roles')


def verify_sources(registration):
    require(registration['schema'] == 'task-worker-ranking-registration-v1', 'Invalid registration')
    entries = registration['source_sha256']
    require(isinstance(entries, dict) and set(entries) == SOURCE_PATHS,
            'Registered source identity set differs')
    require(registration['groups_expected'] == 5104 and registration['candidates_expected'] == 322151
            and registration['max_model_calls'] == 10208 and registration['max_scored_rows'] == 644302
            and registration['sampled_deadline_seconds'] == 300,
            'Registered resource limits differ')
    for relative, expected in entries.items():
        require(isinstance(relative, str) and isinstance(expected, str)
                and len(expected) == 64, 'Invalid registered source identity')
        path = local_file(REPO, relative)
        require(sha(path) == expected, 'Registered source identity changed')


def verify_registration_commit(commit):
    require(isinstance(commit, str) and len(commit) == 40
            and all(c in '0123456789abcdef' for c in commit), 'Invalid registration commit')
    head = subprocess.check_output(['git', '-C', str(REPO), 'rev-parse', 'HEAD'],
                                   stderr=subprocess.DEVNULL).decode().strip()
    require(head == commit, 'Registration commit is not HEAD')
    relative = REGISTRATION.relative_to(REPO).as_posix()
    registered = subprocess.check_output(['git', '-C', str(REPO), 'show', commit + ':' + relative],
                                         stderr=subprocess.DEVNULL)
    require(registered == REGISTRATION.read_bytes().replace(b'\r\n', b'\n'),
            'Registration is not committed exactly')


def reject_shadows(package, modules, exists=None):
    """Reject local files that could be preferred over the frozen sources."""
    if exists is None:
        def exists(path):
            try:
                path.lstat()
                return True
            except FileNotFoundError:
                return False
    suffixes = ('.pyc', *importlib.machinery.EXTENSION_SUFFIXES)
    require(package.is_dir() and package.resolve(strict=True) == package,
            'Unexpected package directory')
    for suffix in ('.py', *suffixes):
        require(not exists(package.parent / (package.name + suffix)), 'Package shadow exists')
    for suffix in suffixes:
        require(not exists(package / ('__init__' + suffix)), 'Package initializer shadow exists')
    for name in modules:
        for suffix in suffixes:
            require(not exists(package / (name + suffix)), 'Module shadow exists')
        for suffix in ('.py', *suffixes):
            require(not exists(package / name / ('__init__' + suffix)), 'Module package shadow exists')


def check_import_origin(module, relative):
    expected = local_file(REPO, relative)
    require(isinstance(module.__loader__, importlib.machinery.SourceFileLoader)
            and Path(module.__file__).resolve(strict=True) == expected, 'Imported source differs')


def bind_runtime(registration):
    """Reuse the existing isolated project-import guard before project imports."""
    require(sys.flags.isolated and sys.flags.dont_write_bytecode and __debug__,
            'Expected isolated non-optimized interpreter without bytecode writes')
    # Load third-party dependencies before adding the project root to sys.path.
    import numpy  # noqa: F401
    import psutil  # noqa: F401
    guard_path = local_file(REPO, 'kaggriculture/learning/task_single_intervention_v1/trusted_imports.py')
    raw = guard_path.read_bytes()
    require(hashlib.sha256(raw).hexdigest() == registration['source_sha256'][
        'kaggriculture/learning/task_single_intervention_v1/trusted_imports.py'], 'Import guard changed')
    guard = types.ModuleType('task_ranking_import_guard')
    exec(compile(raw, '<task-ranking-import-guard>', 'exec', dont_inherit=True), guard.__dict__)
    guard.bind(GAME_SOURCE)
    for name, modules in (('task_worker_ranking_v1', ('core',)),
                          ('task_worker_geometry_diagnostic_v1', ('archive_io', 'training_summary'))):
        reject_shadows(GAME_SOURCE / 'learning' / name, modules)


class ScoreBudget:
    """Shared sampled resource checks, not a hard real-time process limit."""
    def __init__(self, registration, started):
        self.max_calls = registration['max_model_calls']
        self.max_rows = registration['max_scored_rows']
        self.deadline = started + registration['sampled_deadline_seconds']
        self.calls = self.rows = 0
        self.failure = None

    def check(self):
        if time.monotonic() >= self.deadline:
            self.failure = 'deadline'
            raise TimeoutError('Sampled deadline exceeded')

    def wrapped(self, scorer):
        budget = self

        class BoundedScorer:
            metadata = scorer.metadata

            def score(self, shared, candidates):
                budget.check()
                budget.calls += 1
                budget.rows += len(candidates)
                if budget.calls > budget.max_calls or budget.rows > budget.max_rows:
                    budget.failure = 'scoring_budget'
                    raise ValueError('Scoring budget exceeded')
                result = scorer.score(shared, candidates)
                budget.check()
                return result

        return BoundedScorer()


def run(game_root, registration):
    """Decode only registered data, gate coverage, then load frozen CPU models."""
    import psutil
    from learning.task_worker_geometry_diagnostic_v1 import archive_io, training_summary
    from learning.task_teacher_choice_bc_v1 import numeric
    from learning.task_teacher_merge_v1 import merge as merge_module
    from learning.task_worker_ranking_v1 import core
    from learning.task_choice_v1 import features

    for module, relative in (
        (archive_io, 'kaggriculture/learning/task_worker_geometry_diagnostic_v1/archive_io.py'),
        (training_summary, 'kaggriculture/learning/task_worker_geometry_diagnostic_v1/training_summary.py'),
        (numeric, 'kaggriculture/learning/task_teacher_choice_bc_v1/numeric.py'),
        (merge_module, 'kaggriculture/learning/task_teacher_merge_v1/merge.py'),
        (core, 'kaggriculture/learning/task_worker_ranking_v1/core.py'),
        (features, 'kaggriculture/learning/task_choice_v1/features.py'),
        (sys.modules['learning.bc_crop.features'], 'kaggriculture/learning/bc_crop/features.py'),
        (sys.modules['decision.api'], 'kaggriculture/decision/api.py'),
        (sys.modules['learning.task_worker_ranking_v1'],
         'kaggriculture/learning/task_worker_ranking_v1/__init__.py'),
        (sys.modules['learning.task_worker_geometry_diagnostic_v1'],
         'kaggriculture/learning/task_worker_geometry_diagnostic_v1/__init__.py'),
    ):
        check_import_origin(module, relative)

    started = time.monotonic()
    budget = ScoreBudget(registration, started)
    require(psutil.virtual_memory().available >= 4 * 1024**3, 'Insufficient available memory')
    game_root = Path(game_root).resolve(strict=True)
    archive_config, schema, paths = archive_io.registered_inputs(REPO, game_root)
    early_source = read_json(REPO / 'review/evidence/task-choice-v1-20261023/registration.json')
    early_training = read_json(REPO / 'review/evidence/task-bc-v1-20261023-run1/registration.json')
    require_early_training_match(early_source, early_training)
    old_seeds, new_seeds, early_seeds = source_roles(early_source, archive_config)
    require(sum(archive_config['expected_groups'].values()) == registration['groups_expected']
            and sum(archive_config['expected_candidates'].values()) == registration['candidates_expected'],
            'Registered archive dimensions changed')
    limits = archive_config['data_limits']
    arrays = {}
    for source in ('old', 'new'):
        arrays[source] = numeric.read(paths[source], limits['max_groups'], limits['max_candidates'],
            limits['max_archive_bytes'], limits['max_uncompressed_bytes'], archive_config['archive_sha256'][source])
        require(len(arrays[source]['shared']) == archive_config['expected_groups'][source]
                and len(arrays[source]['candidates']) == archive_config['expected_candidates'][source],
                'Archive dimensions differ')
        budget.check()
    # The earlier-model development roles are supplied only to check that the
    # two sources are disjoint; they are not an independent holdout claim.
    merged = merge_module.merge(arrays['old'], arrays['new'], old_seeds=old_seeds, new_seeds=new_seeds,
        holdout_seeds=early_seeds, max_groups=registration['groups_expected'],
        max_candidates=registration['candidates_expected'])
    del arrays
    args = (merged['shared'], merged['candidates'], merged['offsets'],
            merged['teacher_choice_label'], merged['ops'], merged['seed'],
            schema['shared_feature_names'], schema['feature_names'])
    sufficient = core.assess_coverage(*args, max_candidates_per_group=256)
    budget.check()
    if not sufficient:
        return {'schema': 'task-worker-ranking-result-v1', 'coverage': 'insufficient',
                'models_loaded': False, 'model_inference_executed': False,
                'model_a': 'early_bc', 'model_b': 'merged_bc'}

    from learning.task_bc_v1 import model
    check_import_origin(model, 'kaggriculture/learning/task_bc_v1/model.py')
    early = read_json(REPO / 'review/evidence/task-bc-v1-20261023-run1/model/selected-export.json')
    current = read_json(REPO / 'review/evidence/task-teacher-merged-eval-v3-formal-20261030/registration.json')
    seal = read_json(REPO / 'review/evidence/task-teacher-merge-bc-v1-20261028/model-seal.json')
    require(current['merged_checkpoint_sha256'] == seal['checkpoint_sha256']
            and early['checkpoint_sha256'] != seal['checkpoint_sha256'], 'Model identities differ')
    first = local_file(game_root, Path('artifacts/task-bc-v1-20261023-run1/model') / early['checkpoint_file'])
    second = local_file(game_root, current['merged_checkpoint'])
    scorer_a = model.TaskNetwork(first, early['checkpoint_sha256'])
    scorer_b = model.TaskNetwork(second, seal['checkpoint_sha256'])
    try:
        private_result = core.evaluate_private(*args, budget.wrapped(scorer_a), budget.wrapped(scorer_b),
                                               max_candidates_per_group=256)
    except ValueError:
        if budget.failure == 'deadline':
            raise TimeoutError('Sampled deadline exceeded') from None
        raise
    budget.check()
    require(0 < budget.calls <= budget.max_calls and 0 < budget.rows <= budget.max_rows,
            'Scoring was not executed')
    projected = core.project_public(private_result)
    require(projected['coverage'] == 'sufficient', 'Coverage changed between checks')
    require(sha(first) == early['checkpoint_sha256'] and sha(second) == seal['checkpoint_sha256'],
            'Model identity changed after scoring')
    return {'schema': 'task-worker-ranking-result-v1', **projected,
            'models_loaded': True, 'model_inference_executed': True,
            'model_a': 'early_bc', 'model_b': 'merged_bc'}


class QuietParser(argparse.ArgumentParser):
    def error(self, message):
        raise ValueError('Invalid command line')


def publish_result(output, result):
    """Publish only after a complete pending write and exact readback."""
    pending, final = output / 'result.pending', output / 'result.json'
    raw = (json.dumps(result, ensure_ascii=False, indent=2) + '\n').encode('utf-8')
    with pending.open('xb') as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    require(pending.read_bytes() == raw and json.loads(raw) == result, 'Result readback failed')
    require(not final.exists(), 'Result already exists')
    pending.rename(final)


def main():
    parser = QuietParser(description=__doc__)
    parser.add_argument('--game-root', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--registration-commit', required=True)
    args = parser.parse_args()
    verify_registration_commit(args.registration_commit)
    registration = read_json(REGISTRATION)
    verify_sources(registration)
    bind_runtime(registration)
    root = args.game_root.resolve(strict=True)
    artifacts = (root / 'artifacts').resolve(strict=True)
    require_direct_artifacts(root, artifacts)
    default_bot = local_file(root, 'main.py')
    require(sha(default_bot) == registration['source_sha256']['kaggriculture/main.py'],
            'Local default Bot differs')
    output = args.output_dir.resolve(strict=False)
    require(output.parent == artifacts and output.name == 'task-worker-ranking-v1-20261030',
            'Output is outside the registered location')
    output.mkdir(exist_ok=False)
    with (output / 'started.json').open('x', encoding='utf-8') as stream:
        json.dump({'schema': 'task-worker-ranking-start-v1'}, stream)
    try:
        result = run(root, registration)
        require(sha(default_bot) == registration['source_sha256']['kaggriculture/main.py'],
                'Local default Bot changed')
        verify_registration_commit(args.registration_commit)
        verify_sources(registration)
        result['registration_commit'] = args.registration_commit
        publish_result(output, result)
        print(json.dumps({'status': 'completed', 'coverage': result['coverage']}))
    except Exception as exc:
        with (output / 'failure.json').open('x', encoding='utf-8') as stream:
            json.dump({'schema': 'task-worker-ranking-failure-v1',
                       'error_type': type(exc).__name__}, stream)
        raise


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        print(json.dumps({'status': 'failed', 'error_type': type(exc).__name__}), file=sys.stderr)
        sys.exit(1)
