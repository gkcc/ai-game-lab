"""One-shot, source-bound recheck of previously generated private masks.

Registration is maintained separately and must be independently reviewed,
committed, and passed as HEAD before real data is read. This entry does not
load checkpoint weights, score a model, train, or run games.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import types


GAME_SOURCE = Path(__file__).resolve().parents[2]
REPO = GAME_SOURCE.parent
OLD_COMMIT = '27635128cee4653e661faeaa16aab28e3d78fc5f'
PUBLISHED_COMMIT = '54bdf5346e2d76f203edda3d1d91e7d0fac125d6'
OLD_REG_REL = 'review/evidence/task-pair-data-preflight-v1-20261030/registration.json'
OLD_RESULT_REL = 'review/evidence/task-pair-data-preflight-v1-result-20261030/result.json'
NEW_REG_REL = 'review/evidence/task-pair-mask-recheck-v1-20261030/registration.json'
PREFLIGHT_REL = 'kaggriculture/learning/task_pair_data_preflight_v1/run.py'
PREFLIGHT_SHA = 'c2775fcc5bcea0b12525c2bd8dd1f085a53c12149c9ff4f494bd7136d5793db5'
DEFAULT_BOT_SHA = '74943a2e174c1fe1f9ed2e434d8aef5f13aeddc5575d06c0949ebf6bde818459'
OLD_OUTPUT_NAME = 'task-pair-data-preflight-v1-20261030'
NEW_OUTPUT_NAME = 'task-pair-mask-recheck-v1-20261030'
METADATA_LIMIT = 4 * 1024 * 1024
MANIFEST_LIMIT = 65536
RUNTIME = {'platform': 'win32', 'python': '3.12.2', 'numpy': '2.5.4', 'psutil': '7.2.2'}
EXTRA_SOURCE_PATHS = frozenset((
    'kaggriculture/learning/task_pair_mask_recheck_v1/__init__.py',
    'kaggriculture/learning/task_pair_mask_recheck_v1/run.py',
    'kaggriculture/learning/task_pair_mask_recheck_v1/test_run.py',
    'kaggriculture/learning/task_pair_mask_recheck_v1/README.zh-CN.md',
    'kaggriculture/learning/task_pair_training_v1/__init__.py',
    'kaggriculture/learning/task_pair_training_v1/masks.py',
    'kaggriculture/learning/task_pair_training_v1/recompute.py',
    'kaggriculture/learning/task_pair_training_v1/test_masks.py',
    'kaggriculture/learning/task_pair_training_v1/test_recompute.py',
    OLD_REG_REL, OLD_RESULT_REL,
))
CONTRACT = {
    'schema': 'task-pair-mask-recheck-registration-v1',
    'previous_registration_commit': OLD_COMMIT,
    'previous_result_commit': PUBLISHED_COMMIT,
    'old_output_name': OLD_OUTPUT_NAME,
    'new_output_name': NEW_OUTPUT_NAME,
    'conditional_source_order': ['old', 'new'],
    'old_manifest_identity': 'temporary_current_snapshot_not_original_timestamp',
    'max_manifest_bytes': MANIFEST_LIMIT,
    'cache_file_count': 16,
    'cache_total_bytes': 2905480098,
    'max_group_width': 256,
    'minimum_initial_available_bytes': 4294967296,
    'minimum_sampled_available_bytes': 2147483648,
    'sampled_deadline_seconds': 900,
    'runtime': dict(RUNTIME),
    'checkpoint_weights_loaded': False,
    'model_inference': False,
    'training': False,
    'new_games': False,
}
STAGES = frozenset(('arguments', 'output', 'registration', 'sources', 'guard',
                    'load', 'analysis', 'old_masks', 'fresh_masks', 'publish'))
CURRENT_STAGE = 'arguments'
HEX_SHA = re.compile(r'[0-9a-f]{64}\Z')
HEX_COMMIT = re.compile(r'[0-9a-f]{40}\Z')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def stage(name):
    global CURRENT_STAGE
    require(name in STAGES, 'Invalid safe stage')
    CURRENT_STAGE = name


def bounded_file(root, relative, maximum):
    """Read a direct regular file snapshot under root with a strict byte cap."""
    root = Path(root).resolve(strict=True)
    relative = Path(relative)
    require(not relative.is_absolute() and '..' not in relative.parts, 'Invalid input path')
    path = root / relative
    require(not path.is_symlink(), 'Unexpected input link')
    resolved = path.resolve(strict=True)
    require(resolved.is_relative_to(root) and resolved.is_file(), 'Input outside root')
    with path.open('rb') as stream:
        raw = stream.read(maximum + 1)
    require(len(raw) <= maximum, 'Input exceeds size limit')
    return raw


def read_local_mask(directory, name, maximum):
    """Prevent shard traversal, renamed targets and reads past the byte cap."""
    require(type(name) is str and Path(name).name == name and '/' not in name and '\\' not in name,
            'Invalid mask name')
    directory = Path(directory).resolve(strict=True)
    path = directory / name
    require(not path.is_symlink(), 'Unexpected mask link')
    resolved = path.resolve(strict=True)
    require(resolved.parent == directory and resolved.name == name and resolved.is_file(),
            'Mask is outside its directory')
    with path.open('rb') as stream:
        raw = stream.read(maximum + 1)
    require(len(raw) <= maximum, 'Mask exceeds size limit')
    return raw


def git_blob(commit, relative, *, allowed=None):
    """Read a bounded Git blob; check its declared size before capturing bytes."""
    require(type(commit) is str and HEX_COMMIT.fullmatch(commit), 'Invalid commit')
    if allowed is None:
        allowed = frozenset((OLD_REG_REL, OLD_RESULT_REL, NEW_REG_REL))
    require(isinstance(allowed, (set, frozenset)) and 1 <= len(allowed) <= 256
            and type(relative) is str and relative in allowed
            and relative and not relative.startswith(('/', '-')) and '\\' not in relative
            and ':' not in relative and '..' not in Path(relative).parts,
            'Invalid registered blob')
    spec = commit + ':' + relative
    size_raw = subprocess.check_output(['git', '-C', str(REPO), 'cat-file', '-s', spec],
                                       stderr=subprocess.DEVNULL)
    require(len(size_raw) <= 32 and size_raw.strip().isdigit(), 'Invalid blob size')
    size = int(size_raw.strip())
    require(size <= METADATA_LIMIT, 'Registered blob exceeds size limit')
    raw = subprocess.check_output(['git', '-C', str(REPO), 'show', spec], stderr=subprocess.DEVNULL)
    require(len(raw) == size, 'Registered blob size changed')
    return raw


def is_ancestor(ancestor, descendant):
    require(type(ancestor) is str and HEX_COMMIT.fullmatch(ancestor)
            and type(descendant) is str and HEX_COMMIT.fullmatch(descendant), 'Invalid commit')
    return subprocess.run(['git', '-C', str(REPO), 'merge-base', '--is-ancestor', ancestor, descendant],
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0


def verify_old_history(v2, head):
    """Historical, new HEAD and worktree bytes must all agree."""
    require(is_ancestor(OLD_COMMIT, head) and is_ancestor(PUBLISHED_COMMIT, head),
            'Required historical commit is not an ancestor')
    old = git_blob(OLD_COMMIT, OLD_REG_REL)
    result = git_blob(PUBLISHED_COMMIT, OLD_RESULT_REL)
    require(old == git_blob(head, OLD_REG_REL) == bounded_file(REPO, OLD_REG_REL, METADATA_LIMIT)
            and result == git_blob(head, OLD_RESULT_REL) == bounded_file(REPO, OLD_RESULT_REL, METADATA_LIMIT),
            'Historical, HEAD and worktree bytes differ')
    return v2.parse_registration(old), v2.parse_registration(result)


def verify_new_registration(v2, commit):
    """Use v2's HEAD-only verifier solely for the future new registration."""
    raw = git_blob(commit, NEW_REG_REL)
    require(raw == bounded_file(REPO, NEW_REG_REL, METADATA_LIMIT),
            'New registration Git and worktree bytes differ')
    v2.REGISTRATION = REPO / NEW_REG_REL
    checked = v2.verify_registration_commit(commit)
    require(checked == raw, 'New registration changed during verification')
    return checked, v2.parse_registration(checked)


def verify_new_sources(registration, old_registration, v2, frozen, commit, *, extras=EXTRA_SOURCE_PATHS):
    """Bind every declared source to its bounded HEAD blob and worktree bytes."""
    require(type(registration) is dict and set(registration) == set(CONTRACT) | {'source_sha256'}
            and all(v2.strict_contract_equal(registration[key], value)
                    for key, value in CONTRACT.items()), 'New registration contract differs')
    require(type(old_registration) is dict and type(old_registration.get('source_sha256')) is dict,
            'Invalid historical source identities')
    old = old_registration['source_sha256']
    require(isinstance(extras, (set, frozenset)) and extras.isdisjoint(old)
            and len(extras) > 0, 'New source set differs')
    declared = registration['source_sha256']
    require(type(declared) is dict and set(declared) == set(old) | set(extras),
            'Registered source identity set differs')
    allowed = frozenset(declared)
    for name, expected in declared.items():
        require(type(name) is str and type(expected) is str and HEX_SHA.fullmatch(expected),
                'Invalid registered source identity')
        require(name not in old or expected == old[name], 'Historical source identity changed')
        frozen.local_file(REPO, name)
        workspace = bounded_file(REPO, name, METADATA_LIMIT)
        committed = git_blob(commit, name, allowed=allowed)
        require(hashlib.sha256(workspace).hexdigest() == expected
                and hashlib.sha256(committed).hexdigest() == expected and committed == workspace,
                'Source HEAD or worktree identity changed')


def load_preflight():
    """Bootstrap only the SHA-pinned historical entry, without importing data."""
    raw = bounded_file(REPO, PREFLIGHT_REL, 1024 * 1024)
    require(hashlib.sha256(raw).hexdigest() == PREFLIGHT_SHA, 'Historical entry changed')
    module = types.ModuleType('mask_recheck_preflight_entry')
    module.__file__ = str(REPO / PREFLIGHT_REL)
    exec(compile(raw, '<pinned-data-preflight>', 'exec', dont_inherit=True), module.__dict__)
    return module


def load_frozen_v1_bounded(v2):
    """Execute only the bounded, SHA-verified v1 snapshot; do not reread it."""
    raw = bounded_file(REPO, v2.V1_RELATIVE, METADATA_LIMIT)
    require(hashlib.sha256(raw).hexdigest() == v2.V1_SHA256, 'Frozen v1 entry differs')
    module = types.ModuleType('task_worker_ranking_v1_frozen_entry')
    module.__file__ = str(REPO / v2.V1_RELATIVE)
    exec(compile(raw, '<frozen-task-worker-ranking-v1>', 'exec', dont_inherit=True), module.__dict__)
    return module


def verify_sources_in_order(registration, old_registration, v2, frozen, commit, preflight):
    """Bound all source files before the historical streaming SHA validator."""
    verify_new_sources(registration, old_registration, v2, frozen, commit)
    preflight.verify_sources(old_registration, v2, frozen)


def verify_game_root(root):
    """Do not create an output under an unrelated project passed as game root."""
    raw = bounded_file(root, 'main.py', METADATA_LIMIT)
    require(hashlib.sha256(raw).hexdigest() == DEFAULT_BOT_SHA, 'Game root identity differs')


def runtime_identity():
    """Report only fixed software versions, never local paths or machine IDs."""
    import numpy
    import psutil
    return {'platform': sys.platform, 'python': '.'.join(map(str, sys.version_info[:3])),
            'numpy': numpy.__version__, 'psutil': psutil.__version__}


def verify_runtime():
    actual = runtime_identity()
    require(type(actual) is dict and actual.keys() == RUNTIME.keys()
            and all(type(actual[key]) is str and actual[key] == value for key, value in RUNTIME.items()),
            'Runtime version differs')


def compare_current_masks(preflight, core, projection, mask_reader, recompute, v2,
                          views, roles, historic, published, manifest, read_old, check,
                          *, set_stage=lambda name: None):
    """Recompute actual source masks and compare the old snapshot and projection."""
    private = core.analyze(views['train'], views['teacher_dev'], views['conditional_dev'],
                           roles, check=check)
    preflight.verify_historical_overlap(private, historic)
    projected = projection.project(private, views, roles)
    projected.update({'schema': 'task-pair-data-preflight-result-v1',
                      'registration_commit': OLD_COMMIT})
    require(v2.strict_contract_equal(projected, published), 'Published coarse result differs')
    lengths = {role: [len(view.labels) for view in views[role]]
               for role in ('teacher_dev', 'conditional_dev')}
    # This hash authenticates only the bytes just read, not the old creation time.
    snapshot_sha = hashlib.sha256(manifest).hexdigest()
    set_stage('old_masks')
    saved = mask_reader.load_masks(manifest, read_old, manifest_sha256=snapshot_sha,
                                   registration_commit=OLD_COMMIT, role_lengths=lengths)
    require(recompute.verify_masks(private.masks, saved), 'Old snapshot masks differ')
    return private, lengths


def create_fresh_masks(preflight, mask_reader, recompute, output, private, lengths, commit):
    """Write fixed mask names exclusively and decode only through bounded checks.

    The historical writer is deliberately not used: its initial ``np.load``
    would read a just-written file again before checking a replacement header.
    All actual masks here come from the already verified in-memory analysis.
    """
    import numpy as np

    del preflight  # Kept in the public call signature for the existing entry.
    kinds = ('seen_in_train', 'label_conflict', 'qualifying_plant')
    require(type(commit) is str and HEX_COMMIT.fullmatch(commit)
            and tuple(mask_reader.KINDS) == kinds and type(lengths) is dict
            and set(lengths) == {'teacher_dev', 'conditional_dev'}
            and type(private.masks) is dict and set(private.masks) == set(lengths),
            'Invalid fresh mask metadata')
    for role, shards in (('teacher_dev', 1), ('conditional_dev', 2)):
        require(type(lengths[role]) in (tuple, list) and len(lengths[role]) == shards
                and type(private.masks[role]) in (tuple, list) and len(private.masks[role]) == shards,
                'Invalid fresh mask shards')
        for group, length in zip(private.masks[role], lengths[role], strict=True):
            require(type(length) is int and 1 <= length <= 1_000_000
                    and type(group) is dict and set(group) == set(kinds), 'Invalid fresh mask fields')
            require(all(isinstance(group[kind], np.ndarray) and group[kind].dtype == np.dtype(bool)
                        and group[kind].shape == (length,) for kind in kinds), 'Invalid fresh mask array')
            require(not np.any(group['label_conflict']), 'Fresh label conflict')

    entries = []
    for role in ('teacher_dev', 'conditional_dev'):
        for source, group in enumerate(private.masks[role]):
            length = lengths[role][source]
            for kind in kinds:
                name = role + '-' + str(source) + '-' + kind + '.npy'
                with (output / name).open('xb') as stream:
                    np.save(stream, group[kind], allow_pickle=False)
                    stream.flush()
                    os.fsync(stream.fileno())
                raw = read_local_mask(output, name, length + MANIFEST_LIMIT)
                entries.append({'name': name, 'sha256': hashlib.sha256(raw).hexdigest(), 'length': length})
    require(len(entries) == 9, 'Invalid fresh mask count')
    write_json_exclusive(output / 'masks.local.json', {
        'schema': 'task-pair-data-local-masks-v1', 'registration_commit': commit,
        'conditional_source_order': ['old', 'new'], 'masks': entries,
    })
    manifest = read_local_mask(output, 'masks.local.json', MANIFEST_LIMIT)
    digest = hashlib.sha256(manifest).hexdigest()
    fresh = mask_reader.load_masks(manifest, lambda name, maximum: read_local_mask(output, name, maximum),
                                   manifest_sha256=digest, registration_commit=commit,
                                   role_lengths=lengths)
    require(recompute.verify_masks(private.masks, fresh), 'Fresh mask readback differs')
    return digest


def write_json_exclusive(path, value):
    raw = (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode('utf-8')
    with Path(path).open('xb') as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    with Path(path).open('rb') as stream:
        restored = stream.read(len(raw) + 1)
    require(restored == raw, 'Receipt readback differs')


def public_result(commit):
    """Coarse publication only; private manifest identity stays local."""
    require(type(commit) is str and HEX_COMMIT.fullmatch(commit), 'Invalid result commit')
    return {'schema': 'task-pair-mask-recheck-result-v1',
            'registration_commit': commit,
            'previous_registration_commit': OLD_COMMIT,
            'old_snapshot_matches_recomputation': True,
            'fresh_masks_match_recomputation': True,
            'historical_overlap_matches': True,
            'previous_public_projection_matches': True,
            'old_creation_time_proven': False,
            'checkpoint_weights_loaded': False, 'model_inference': False,
            'training': False, 'new_games': False}


class QuietParser(argparse.ArgumentParser):
    def error(self, message):
        raise ValueError('Invalid command line')


def main():
    stage('arguments')
    parser = QuietParser(description=__doc__)
    parser.add_argument('--game-root', required=True, type=Path)
    parser.add_argument('--registration-commit', required=True)
    args = parser.parse_args()
    root = args.game_root.resolve(strict=True)
    verify_game_root(root)
    artifacts = (root / 'artifacts').resolve(strict=True)
    require(artifacts == root / 'artifacts' and artifacts.is_dir(), 'Invalid artifacts root')
    stage('output')
    output = artifacts / NEW_OUTPUT_NAME
    require(output.resolve(strict=False) == output, 'Invalid output location')
    output.mkdir(exist_ok=False)
    try:
        write_json_exclusive(output / 'started.json', {'schema': 'task-pair-mask-recheck-start-v1'})
        stage('registration')
        preflight = load_preflight()
        v2 = preflight.load_v2()
        frozen = load_frozen_v1_bounded(v2)
        registered, registration = verify_new_registration(v2, args.registration_commit)
        old_registration, published = verify_old_history(v2, args.registration_commit)
        stage('sources')
        verify_sources_in_order(registration, old_registration, v2, frozen,
                                args.registration_commit, preflight)
        require(old_registration['source_sha256']['kaggriculture/main.py'] == DEFAULT_BOT_SHA,
                'Historical game root identity differs')
        require(all(v2.strict_contract_equal(CONTRACT[name], preflight.CONTRACT[name]) for name in (
            'cache_file_count', 'cache_total_bytes', 'max_group_width', 'minimum_initial_available_bytes',
            'minimum_sampled_available_bytes', 'sampled_deadline_seconds')),
                'Resource contract differs from historical preflight')
        stage('guard')
        frozen.bind_runtime(registration)
        verify_runtime()
        frozen.require_direct_artifacts(root, artifacts)
        frozen.reject_shadows(GAME_SOURCE / 'learning/task_pair_data_preflight_v1', ('core', 'projection'))
        frozen.reject_shadows(GAME_SOURCE / 'learning/task_pair_training_v1', ('masks', 'recompute'))
        frozen.reject_shadows(GAME_SOURCE / 'learning/task_pair_mask_recheck_v1', ('run',))
        default_bot = frozen.local_file(root, 'main.py')
        require(frozen.sha(default_bot) == registration['source_sha256']['kaggriculture/main.py'],
                'Local default Bot differs')
        old_output = artifacts / OLD_OUTPUT_NAME
        require(not old_output.is_symlink() and old_output.resolve(strict=True) == old_output
                and old_output.is_dir(), 'Invalid old output location')
        import psutil
        from learning.task_pair_data_preflight_v1 import core, projection
        from learning.task_pair_training_v1 import masks, recompute
        for module, relative in (
            (core, 'kaggriculture/learning/task_pair_data_preflight_v1/core.py'),
            (projection, 'kaggriculture/learning/task_pair_data_preflight_v1/projection.py'),
            (masks, 'kaggriculture/learning/task_pair_training_v1/masks.py'),
            (recompute, 'kaggriculture/learning/task_pair_training_v1/recompute.py'),
            (sys.modules['learning.task_pair_training_v1'], 'kaggriculture/learning/task_pair_training_v1/__init__.py'),
        ):
            frozen.check_import_origin(module, relative)
        budget = preflight.Budget(psutil)
        stage('load')
        views, roles = preflight.load_data(root, frozen, budget.check)
        historic = frozen.read_json(REPO / 'review/evidence/task-bc-v1-20261023-run1/cache-summary.json')
        stage('old_masks')
        old_manifest = read_local_mask(old_output, 'masks.local.json', MANIFEST_LIMIT)
        stage('analysis')
        private, lengths = compare_current_masks(
            preflight, core, projection, masks, recompute, v2, views, roles, historic, published,
            old_manifest, lambda name, maximum: read_local_mask(old_output, name, maximum), budget.check,
            set_stage=stage)
        stage('fresh_masks')
        fresh_sha = create_fresh_masks(preflight, masks, recompute, output, private, lengths,
                                       args.registration_commit)
        budget.check()
        stage('registration')
        require(verify_new_registration(v2, args.registration_commit)[0] == registered,
                'New registration changed during recheck')
        later_old, later_result = verify_old_history(v2, args.registration_commit)
        require(v2.strict_contract_equal(later_old, old_registration)
                and v2.strict_contract_equal(later_result, published), 'Historical record changed')
        stage('sources')
        verify_sources_in_order(registration, old_registration, v2, frozen,
                                args.registration_commit, preflight)
        verify_runtime()
        require(frozen.sha(default_bot) == registration['source_sha256']['kaggriculture/main.py'],
                'Local default Bot changed')
        # Both receipt and public result are required with an observed exit 0.
        stage('publish')
        write_json_exclusive(output / 'binding.local.json', {
            'schema': 'task-pair-mask-recheck-binding-v1',
            'registration_commit': args.registration_commit,
            'previous_registration_commit': OLD_COMMIT,
            'previous_result_commit': PUBLISHED_COMMIT,
            'fresh_manifest_sha256': fresh_sha,
            'conditional_source_order': ['old', 'new'],
            'old_snapshot_matches_recomputation': True,
            'fresh_masks_match_recomputation': True,
            'old_creation_time_proven': False,
        })
        frozen.publish_result(output, public_result(args.registration_commit))
        print(json.dumps({'status': 'completed'}))
    except (Exception, KeyboardInterrupt) as exc:
        try:
            write_json_exclusive(output / 'failure.json', {
                'schema': 'task-pair-mask-recheck-failure-v1',
                'stage': CURRENT_STAGE, 'error_type': type(exc).__name__,
            })
        except (OSError, ValueError):
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
