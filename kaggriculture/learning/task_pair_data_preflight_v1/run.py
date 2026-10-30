"""Read-only preparation of previously consumed task demonstrations.

Original arrays are authenticated with streaming hashes and opened as read-only
memory maps. Conditional NPZ inputs use their existing authenticated reader.
This entry never loads checkpoint weights, trains, scores a model or runs a game.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import types

GAME_SOURCE = Path(__file__).resolve().parents[2]
REPO = GAME_SOURCE.parent
REGISTRATION = REPO / 'review/evidence/task-pair-data-preflight-v1-20261030/registration.json'
V2_RELATIVE = 'kaggriculture/learning/task_worker_ranking_v2/run.py'
V2_SHA256 = '80576c8030ebaf097f8c64419e04aba05f4bd3f205f4595b40c0f49083dbc2ef'
DATA_SOURCE_PATHS = frozenset((
    'kaggriculture/learning/task_pair_data_preflight_v1/__init__.py',
    'kaggriculture/learning/task_pair_data_preflight_v1/core.py',
    'kaggriculture/learning/task_pair_data_preflight_v1/projection.py',
    'kaggriculture/learning/task_pair_data_preflight_v1/run.py',
    'kaggriculture/learning/task_pair_data_preflight_v1/test_core.py',
    'kaggriculture/learning/task_pair_data_preflight_v1/test_projection.py',
    'kaggriculture/learning/task_pair_data_preflight_v1/test_run.py',
    'kaggriculture/learning/task_pair_data_preflight_v1/README.zh-CN.md',
    'kaggriculture/learning/task_bc_v1/train.py',
    'review/evidence/task-bc-v1-20261023-run1/cache/manifest.json',
    'review/evidence/task-bc-v1-20261023-run1/cache-summary.json',
    'review/evidence/task-choice-v1-20261023/dataset-manifest.json',
))
CONTRACT = {
    'schema': 'task-pair-data-preflight-registration-v1',
    'roles': {'train': 8, 'teacher_dev': 2, 'conditional_dev': 4},
    'groups': {'train': 108310, 'teacher_dev': 27096, 'conditional_dev': 5104},
    'candidates': {'train': 5546854, 'teacher_dev': 1353262, 'conditional_dev': 322151},
    'cache_file_count': 16,
    'cache_total_bytes': 2905480098,
    'max_group_width': 256,
    'train_qualifying_groups_min': 20,
    'train_qualifying_clusters_min': 2,
    'dev_qualifying_groups_min': 20,
    'dev_qualifying_groups_per_cluster_min': 5,
    'operation_groups_min': 20,
    'operation_clusters_min': 2,
    'conditional_source_order': ['old', 'new'],
    'conditional_source_primary_policy': 'each_source_required',
    'overlap_identity': 'float32_ordered_full_group_signed_zero_normalized_with_numeric_confirmation',
    'overlap_public_bands': ['below_20', 'at_least_20'],
    'historical_teacher_dev_seen_groups_expected': 3978,
    'historical_cross_split_label_conflicts_expected': 0,
    'checkpoint_weights_loaded': False,
    'model_inference': False,
    'training': False,
    'new_games': False,
    'minimum_initial_available_bytes': 4294967296,
    'minimum_sampled_available_bytes': 2147483648,
    'sampled_deadline_seconds': 900,
}
STAGES = frozenset(('arguments', 'registration', 'sources', 'guard', 'output',
                    'load', 'analysis', 'publish'))
CURRENT_STAGE = 'arguments'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def set_stage(stage):
    global CURRENT_STAGE
    require(stage in STAGES, 'Invalid safe stage')
    CURRENT_STAGE = stage


def load_v2():
    path = REPO / V2_RELATIVE
    require(path.is_file() and path.resolve(strict=True).is_relative_to(REPO.resolve(strict=True)),
            'Frozen entry is missing or outside repository')
    with path.open('rb') as stream:
        raw = stream.read(1024 * 1024 + 1)
    require(len(raw) <= 1024 * 1024 and hashlib.sha256(raw).hexdigest() == V2_SHA256,
            'Frozen entry identity differs')
    module = types.ModuleType('task_ranking_v2_preflight_entry')
    module.__file__ = str(path)
    exec(compile(raw, '<frozen-task-ranking-v2>', 'exec', dont_inherit=True), module.__dict__)
    return module


def verify_sources(registration, v2, frozen):
    require(type(registration) is dict and set(registration) == set(CONTRACT) | {'source_sha256'}
            and all(v2.strict_contract_equal(registration[key], value) for key, value in CONTRACT.items()),
            'Registered data contract differs')
    expected = frozen.SOURCE_PATHS | v2.V2_SOURCE_PATHS | DATA_SOURCE_PATHS
    require(len(expected) == len(frozen.SOURCE_PATHS) + len(v2.V2_SOURCE_PATHS) + len(DATA_SOURCE_PATHS),
            'Source sets overlap')
    hashes = registration['source_sha256']
    require(type(hashes) is dict and set(hashes) == expected, 'Registered source identity set differs')
    require(hashes[V2_RELATIVE] == V2_SHA256, 'Frozen entry registration differs')
    for relative, expected_sha in hashes.items():
        require(type(expected_sha) is str and len(expected_sha) == 64
                and all(c in '0123456789abcdef' for c in expected_sha), 'Invalid source identity')
        require(frozen.sha(frozen.local_file(REPO, relative)) == expected_sha, 'Source identity changed')


class Budget:
    """Sampled resource checks; neither deadline nor memory is a hard limit."""
    def __init__(self, psutil):
        self.psutil = psutil
        self.deadline = time.monotonic() + CONTRACT['sampled_deadline_seconds']
        self.calls = 0
        require(psutil.virtual_memory().available >= CONTRACT['minimum_initial_available_bytes'],
                'Insufficient available memory')

    def check(self):
        self.calls += 1
        if time.monotonic() >= self.deadline:
            raise TimeoutError('Sampled deadline exceeded')
        if self.calls % 256 == 0:
            require(self.psutil.virtual_memory().available >= CONTRACT['minimum_sampled_available_bytes'],
                    'Insufficient sampled memory')


def stream_sha(path, check):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while True:
            check()
            block = stream.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def load_data(root, frozen, check):
    """Authenticate historical roles, original mmap arrays and conditional NPZs."""
    from learning.task_bc_v1 import train
    from learning.task_pair_data_preflight_v1 import core
    from learning.task_teacher_choice_bc_v1 import numeric
    from learning.task_worker_geometry_diagnostic_v1 import archive_io
    import numpy as np

    for module, relative in (
        (train, 'kaggriculture/learning/task_bc_v1/train.py'),
        (core, 'kaggriculture/learning/task_pair_data_preflight_v1/core.py'),
        (numeric, 'kaggriculture/learning/task_teacher_choice_bc_v1/numeric.py'),
        (archive_io, 'kaggriculture/learning/task_worker_geometry_diagnostic_v1/archive_io.py'),
        (sys.modules['learning.task_pair_data_preflight_v1'],
         'kaggriculture/learning/task_pair_data_preflight_v1/__init__.py'),
    ):
        frozen.check_import_origin(module, relative)

    original = frozen.read_json(REPO / 'review/evidence/task-choice-v1-20261023/registration.json')
    trained = frozen.read_json(REPO / 'review/evidence/task-bc-v1-20261023-run1/registration.json')
    frozen.require_early_training_match(original, trained)
    conditional, conditional_schema, conditional_paths = archive_io.registered_inputs(REPO, root)
    old_seeds, new_seeds, _ = frozen.source_roles(original, conditional)
    roles = {'train': original['train_seeds'], 'teacher_dev': original['validation_seeds'],
             'conditional_dev': old_seeds + new_seeds}

    export = frozen.read_json(REPO / 'review/evidence/task-bc-v1-20261023-run1/model/selected-export.json')
    public_manifest = REPO / 'review/evidence/task-bc-v1-20261023-run1/cache/manifest.json'
    public_summary = REPO / 'review/evidence/task-bc-v1-20261023-run1/cache-summary.json'
    dataset_manifest = REPO / 'review/evidence/task-choice-v1-20261023/dataset-manifest.json'
    original_registration = REPO / 'review/evidence/task-choice-v1-20261023/registration.json'
    cache_manifest = frozen.read_json(public_manifest)
    dataset = frozen.read_json(dataset_manifest)
    require(frozen.sha(public_manifest) == export['cache_manifest_sha256']
            and frozen.sha(public_summary) == cache_manifest['summary_sha256']
            and frozen.sha(dataset_manifest) == cache_manifest['source_dataset_manifest_sha256']
            and frozen.sha(original_registration) == dataset['registration_sha256'], 'Historic source chain differs')
    cache = (root / 'artifacts/task-bc-v1-20261023-run1/cache').resolve(strict=True)
    require(cache.is_relative_to((root / 'artifacts').resolve(strict=True)) and cache.is_dir(),
            'Cache is outside artifacts')
    require((cache / 'manifest.json').read_bytes() == public_manifest.read_bytes(), 'Cache manifest differs')
    schema_path = frozen.local_file(root, 'artifacts/task-bc-v1-20261023-run1/cache/schema.json')
    require(frozen.sha(schema_path) == cache_manifest['schema_sha256'], 'Cache schema bytes differ')
    schema = frozen.read_json(schema_path)
    require(schema == conditional_schema and schema['schema_sha256'] == cache_manifest['numeric_feature_schema_sha256'],
            'Logical feature schema differs')
    require(cache_manifest['passed'] is True and cache_manifest['max_group_size'] <= CONTRACT['max_group_width'],
            'Cache is not eligible')

    fields = {'shared', 'candidates', 'offsets', 'labels', 'selected_ops',
              'candidate_ops', 'candidate_identity', 'seeds'}
    require(set(cache_manifest['arrays']) == {'train', 'validation'}, 'Cache split differs')
    paths = {}
    total_bytes = 0
    for split in ('train', 'validation'):
        infos = cache_manifest['arrays'][split]
        require(set(infos) == fields, 'Cache field set differs')
        for name, info in infos.items():
            require(info['path'] == split + '/' + name + '.npy', 'Unexpected cache array path')
            path = frozen.local_file(root, Path('artifacts/task-bc-v1-20261023-run1/cache') / info['path'])
            total_bytes += path.stat().st_size
            require(total_bytes <= CONTRACT['cache_total_bytes'], 'Cache bytes exceed budget')
            require(stream_sha(path, check) == info['sha256'], 'Cache array identity differs')
            paths[(split, name)] = path
    require(len(paths) == CONTRACT['cache_file_count'] and total_bytes == CONTRACT['cache_total_bytes'],
            'Cache files differ')

    views = {}
    for split, role in (('train', 'train'), ('validation', 'teacher_dev')):
        check()
        arrays = train.load_split(cache, split)
        for name, array in arrays.items():
            info = cache_manifest['arrays'][split][name]
            require(isinstance(array, np.memmap) and not array.flags.writeable
                    and list(array.shape) == info['shape'] and str(array.dtype) == info['dtype'],
                    'Cache array metadata differs')
        require(len(arrays['shared']) == CONTRACT['groups'][role]
                and len(arrays['candidates']) == CONTRACT['candidates'][role]
                and np.all(np.diff(arrays['offsets']) <= CONTRACT['max_group_width']),
                'Cache dimensions differ')
        views[role] = [core.RaggedView.from_cache(arrays, schema)]

    conditional_views = []
    limits = conditional['data_limits']
    for name, expected_seeds in (('old', old_seeds), ('new', new_seeds)):
        check()
        arrays = numeric.read(conditional_paths[name], limits['max_groups'], limits['max_candidates'],
                              limits['max_archive_bytes'], limits['max_uncompressed_bytes'],
                              conditional['archive_sha256'][name])
        require(len(arrays['shared']) == conditional['expected_groups'][name]
                and len(arrays['candidates']) == conditional['expected_candidates'][name]
                and set(map(int, np.unique(arrays['seed']))) == set(expected_seeds)
                and np.all(np.diff(arrays['offsets']) <= CONTRACT['max_group_width']),
                'Conditional source differs')
        conditional_views.append(core.RaggedView.from_conditional(arrays, schema))
    require(sum(len(view.shared) for view in conditional_views) == CONTRACT['groups']['conditional_dev']
            and sum(len(view.candidates) for view in conditional_views) == CONTRACT['candidates']['conditional_dev'],
            'Conditional dimensions differ')
    views['conditional_dev'] = conditional_views
    return views, roles


def save_masks(output, private, registration_commit):
    """Persist only local grouping masks; a result is not published before readback."""
    import numpy as np
    entries = []
    for role in ('teacher_dev', 'conditional_dev'):
        for source, group in enumerate(private.masks[role]):
            require(set(group) == {'seen_in_train', 'label_conflict', 'qualifying_plant'}
                    and len({len(group[key]) for key in group}) == 1, 'Invalid local mask set')
            for key in ('seen_in_train', 'label_conflict', 'qualifying_plant'):
                mask = group[key]
                require(isinstance(mask, np.ndarray) and mask.dtype == np.dtype(bool) and mask.ndim == 1,
                        'Invalid local mask')
                name = role + '-' + str(source) + '-' + key + '.npy'
                path = output / name
                with path.open('xb') as stream:
                    np.save(stream, mask, allow_pickle=False)
                    stream.flush()
                    os.fsync(stream.fileno())
                restored = np.load(path, allow_pickle=False)
                require(np.array_equal(restored, mask), 'Local mask readback failed')
                entries.append({'name': name, 'sha256': stream_sha(path, lambda: None), 'length': len(mask)})
    manifest = {'schema': 'task-pair-data-local-masks-v1', 'registration_commit': registration_commit,
                'conditional_source_order': ['old', 'new'], 'masks': entries}
    manifest_path = output / 'masks.local.json'
    with manifest_path.open('x', encoding='utf-8') as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    with manifest_path.open('r', encoding='utf-8') as stream:
        require(json.load(stream) == manifest, 'Local mask manifest readback failed')


def verify_historical_overlap(private, historic):
    reference = historic['train_validation_numeric_overlap']
    require(reference['validation_examples_with_train_numeric_match']
            == CONTRACT['historical_teacher_dev_seen_groups_expected']
            and reference['shared_numeric_groups_with_multiple_labels']
            == CONTRACT['historical_cross_split_label_conflicts_expected'],
            'Historical overlap reference differs')
    overlap = private.overlap['teacher_dev']
    require(len(overlap) == 1
            and overlap[0]['all']['same_label_only'] + overlap[0]['all']['label_conflict']
            == CONTRACT['historical_teacher_dev_seen_groups_expected']
            and overlap[0]['all']['label_conflict']
            == CONTRACT['historical_cross_split_label_conflicts_expected'],
            'Historical overlap was not reproduced')


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
    v2 = load_v2()
    v2.REGISTRATION = REGISTRATION
    raw = v2.verify_registration_commit(args.registration_commit)
    registration = v2.parse_registration(raw)
    set_stage('sources')
    frozen = v2.load_frozen_v1()
    verify_sources(registration, v2, frozen)
    set_stage('guard')
    frozen.bind_runtime(registration)
    frozen.reject_shadows(GAME_SOURCE / 'learning/task_pair_data_preflight_v1', ('core', 'projection'))
    root = args.game_root.resolve(strict=True)
    artifacts = (root / 'artifacts').resolve(strict=True)
    frozen.require_direct_artifacts(root, artifacts)
    default_bot = frozen.local_file(root, 'main.py')
    require(frozen.sha(default_bot) == registration['source_sha256']['kaggriculture/main.py'],
            'Local default Bot differs')
    set_stage('output')
    output = args.output_dir.resolve(strict=False)
    require(output.parent == artifacts and output.name == 'task-pair-data-preflight-v1-20261030',
            'Output is outside registered location')
    output.mkdir(exist_ok=False)
    try:
        with (output / 'started.json').open('x', encoding='utf-8') as stream:
            json.dump({'schema': 'task-pair-data-preflight-start-v1'}, stream)
        set_stage('load')
        import psutil
        from learning.task_pair_data_preflight_v1 import core, projection
        frozen.check_import_origin(projection, 'kaggriculture/learning/task_pair_data_preflight_v1/projection.py')
        budget = Budget(psutil)
        views, roles = load_data(root, frozen, budget.check)
        set_stage('analysis')
        private = core.analyze(views['train'], views['teacher_dev'], views['conditional_dev'],
                               roles, check=budget.check)
        historic = frozen.read_json(REPO / 'review/evidence/task-bc-v1-20261023-run1/cache-summary.json')
        verify_historical_overlap(private, historic)
        result = projection.project(private, views, roles)
        save_masks(output, private, args.registration_commit)
        budget.check()
        set_stage('registration')
        require(v2.verify_registration_commit(args.registration_commit) == raw, 'Registration changed during run')
        set_stage('sources')
        verify_sources(registration, v2, frozen)
        require(frozen.sha(default_bot) == registration['source_sha256']['kaggriculture/main.py'],
                'Local default Bot changed')
        result.update({'schema': 'task-pair-data-preflight-result-v1',
                       'registration_commit': args.registration_commit})
        set_stage('publish')
        frozen.publish_result(output, result)
        print(json.dumps({'status': 'completed',
                          'all_groups_primary_ready': result['all_groups_primary_ready'],
                          'unmatched_primary_ready': result['unmatched_primary_ready']}))
    except (Exception, KeyboardInterrupt) as exc:
        try:
            with (output / 'failure.json').open('x', encoding='utf-8') as stream:
                json.dump({'schema': 'task-pair-data-preflight-failure-v1', 'stage': CURRENT_STAGE,
                           'error_type': type(exc).__name__}, stream)
        except OSError:
            pass
        raise


def cli():
    try:
        main()
    except (Exception, KeyboardInterrupt) as exc:
        print(json.dumps({'status': 'failed', 'stage': CURRENT_STAGE,
                          'error_type': type(exc).__name__}), file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(cli())
