"""Bounded, read-only geometry and demonstration coverage diagnostic."""
import argparse
import gzip
import hashlib
import io
import importlib.machinery
import json
import os
from pathlib import Path
import sys
import time
import types

GAME_SOURCE = Path(__file__).resolve().parents[2]
REPO = GAME_SOURCE.parent
REGISTRATION = REPO / 'review/evidence/task-worker-geometry-diagnostic-v1-20261030/registration.json'
PREFIX_SOURCE_SHA = '510a8276b3901641a993b3763adbb69c25e6f46c10bbf34aa2bd69e019b2ae2a'
PREFIX_INIT_SHA = '7255ebf781b573f8aed02a0318327fe46f67f98fe0a3ba45b215f4908174b028'
GUARD_SHA = '329d3b6415e3fac1cabea5d27571db3510281cbabd9c5118c55cec9aa43b115a'
TRAIN_FREEZE_SHA = '1b7dee4cc8551067b30ee4266f7fe4fe89130142c8ce0dd97eaf01e3c09f1a2e'
PAIRS_SHA = '44cde22338157d2d1f62aa227a22eb837e2f2c620b77a88068965a3f0723f8a7'
TIME_LIMIT_SECONDS = 180
COMPRESSED_LIMIT = 512 * 1024**2
LINE_LIMIT = 16 * 1024**2


def require(condition, message):
    if not condition:
        raise ValueError(message)


class QuietParser(argparse.ArgumentParser):
    def error(self, message):
        raise ValueError('Invalid command line')


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def read_small(path, limit=4 * 1024**2):
    with path.open('rb') as stream:
        raw = stream.read(limit + 1)
    require(len(raw) <= limit, 'Input exceeds size limit')
    return raw


def deadline(started):
    if time.monotonic() - started > TIME_LIMIT_SECONDS:
        raise TimeoutError('Sampled diagnostic deadline exceeded')


def verify_sources():
    registry = json.loads(read_small(REGISTRATION).decode('utf-8-sig'))
    require(registry['schema'] == 'worker-geometry-registration-v1', 'Unexpected registration')
    root = GAME_SOURCE / 'learning/task_worker_geometry_diagnostic_v1'
    expected_names = {'__init__.py', 'run.py', 'archive_io.py', 'trace_summary.py', 'training_summary.py',
                      'test_run.py', 'test_archive_io.py', 'test_trace_summary.py',
                      'test_training_summary.py', 'README.zh-CN.md'}
    require(set(registry['source_sha256']) == expected_names, 'Registered source set differs')
    for name, expected in registry['source_sha256'].items():
        require(Path(name).name == name and sha(read_small(root / name)) == expected,
                'Registered source identity changed')
    return registry


def reject_shadows(package, modules, exists=None):
    """Reject nearby package or native-module shadows before ordinary import."""
    if exists is None:
        exists = lambda path: path.exists()
    suffixes = ('.pyc', *importlib.machinery.EXTENSION_SUFFIXES)
    for suffix in ('.py', *suffixes):
        require(not exists(package.parent / (package.name + suffix)), 'Package shadow exists')
    for suffix in suffixes:
        require(not exists(package / ('__init__' + suffix)), 'Package initializer shadow exists')
    for name in modules:
        for suffix in suffixes:
            require(not exists(package / (name + suffix)), 'Module shadow exists')
        for suffix in ('.py', *suffixes):
            require(not exists(package / name / ('__init__' + suffix)), 'Module package shadow exists')


def check_import_origin(module, expected):
    require(isinstance(module.__loader__, importlib.machinery.SourceFileLoader)
            and Path(module.__file__).resolve(strict=True) == expected.resolve(strict=True),
            'Imported module origin differs')


def bind_runtime():
    guard_path = GAME_SOURCE / 'learning/task_single_intervention_v1/trusted_imports.py'
    raw = read_small(guard_path)
    require(sha(raw) == GUARD_SHA, 'Import guard changed')
    guard = types.ModuleType('frozen_geometry_import_guard')
    exec(compile(raw, 'frozen-geometry-import-guard', 'exec', dont_inherit=True), guard.__dict__)
    guard.bind(GAME_SOURCE, check_public_prefix=True)
    learning = GAME_SOURCE / 'learning'
    for package_name, modules in (('task_prefix_diagnostic_v2', ('runner',)),
                                  ('task_worker_geometry_diagnostic_v1',
                                   ('archive_io', 'trace_summary', 'training_summary'))):
        package = learning / package_name
        require(package.is_dir() and package.resolve(strict=True) == package,
                'Unexpected package directory')
        reject_shadows(package, modules)
    prefix_path = GAME_SOURCE / 'learning/task_prefix_diagnostic_v2/runner.py'
    require(sha(read_small(prefix_path)) == PREFIX_SOURCE_SHA, 'Historical adapter changed')
    require(sha(read_small(prefix_path.parent / '__init__.py')) == PREFIX_INIT_SHA,
            'Historical adapter package changed')
    from learning.task_prefix_diagnostic_v2 import runner as prefix
    check_import_origin(prefix, prefix_path)
    check_import_origin(sys.modules['learning.task_prefix_diagnostic_v2'], prefix_path.parent / '__init__.py')
    prefix.verify_runtime()
    freeze_raw = read_small(REPO / 'review/evidence/task-teacher-merge-bc-v1-20261028/freeze.json')
    require(sha(freeze_raw) == TRAIN_FREEZE_SHA, 'Training freeze changed')
    freeze = json.loads(freeze_raw.decode('utf-8-sig'))
    for suffix in ('task_choice_v1/features.py', 'task_choice_v1/__init__.py',
                   'bc_crop/features.py', 'bc_crop/__init__.py'):
        matches = [name for name in freeze['public_runtime_sha256'] if name.endswith(suffix)]
        require(len(matches) == 1, 'Frozen feature source identity ambiguous')
        name = matches[0]
        require(sha(read_small(GAME_SOURCE / name)) == freeze['public_runtime_sha256'][name],
                'Frozen feature source changed')
    from learning.task_worker_geometry_diagnostic_v1 import archive_io, trace_summary, training_summary
    check_import_origin(sys.modules['learning.task_worker_geometry_diagnostic_v1'],
                        learning / 'task_worker_geometry_diagnostic_v1' / '__init__.py')
    for module, name in ((archive_io, 'archive_io'), (trace_summary, 'trace_summary'),
                         (training_summary, 'training_summary')):
        check_import_origin(module, learning / 'task_worker_geometry_diagnostic_v1' / (name + '.py'))
    return prefix, archive_io, trace_summary, training_summary


def validate_pairs(rows):
    require(isinstance(rows, list), 'Invalid preceding pair results')
    expected = {(alias, side, reference) for alias in ('S02', 'S03') for side in (0, 1)
                for reference in ('unweighted', 'teacher_task')}
    require(len(rows) == 8 and {(r['seed_alias'], r['side'], r['reference']) for r in rows} == expected,
            'Preceding pair identities differ')
    for row in rows:
        choice = row['first_choice_difference']
        require(row['first_branch_kind'] == 'actor' and row['first_branch_step'] == 2
                and row['first_actor_request_components_differ'] == {'farmer': False, 'hands': True, 'market': False}
                and row['choice_difference_at_branch'] is True and choice['step'] == 2
                and choice['selection_record_index'] == 5 and choice['candidate_op'] == 'PLANT'
                and choice['reference_op'] == 'PLANT' and choice['same_target'] is True
                and choice['same_unit'] is False, 'Preceding target selection differs')


def decode_target(raw, expected_size, expected_sha):
    require(len(raw) == expected_size and sha(raw) == expected_sha, 'Trace identity differs')
    with gzip.GzipFile(fileobj=io.BytesIO(raw), mode='rb') as stream:
        target = None
        for step in range(3):
            line = stream.readline(LINE_LIMIT + 1)
            require(len(line) <= LINE_LIMIT and line.endswith(b'\n'), 'Trace prefix incomplete')
            target = json.loads(line)
            require(type(target['step']) is int and target['step'] == step, 'Trace turn differs')
        require(all(target[name] == 0 for name in ('load_errors', 'inference_errors', 'fallbacks')),
                'Recorded trace errors')
        require(type(target['records']) is list and len(target['records']) > 5, 'Target selection missing')
        return target, target['records'][5]


def public_training_summary(by_source):
    """Publish only coarse existence thresholds, never numeric training counts."""
    result = {}
    fields = ('groups', 'teacher_plant_groups', 'same_target_unit_competition_groups',
              'teacher_has_farther_alternative_groups', 'equal_distance_competition_groups')
    for source in ('old', 'new'):
        full, early = by_source[source]['all_steps'], by_source[source]['encoded_step_2']
        require(all(set(row) == set(fields) and all(type(v) is int and v >= 0 for v in row.values())
                    for row in (full, early)), 'Invalid aggregate counts')
        for row in (full, early):
            groups, plant, competing, farther, equal = (row[field] for field in fields)
            overlap = farther + equal - competing
            require(groups >= plant >= competing >= max(farther, equal)
                    and 0 <= overlap <= min(farther, equal), 'Aggregate hierarchy differs')
        require(all(full[field] >= early[field] for field in fields), 'Time partition differs')
        def status(count):
            return 'at_least_20' if count >= 20 else 'below_20'
        # The two categories can overlap. Do not publish their union/count:
        # a union threshold can reveal that a suppressed small category exists.
        result[source] = {field: status(early[field]) for field in fields[3:]}
    return result


def analyze(root, panel, status, manifest):
    started = time.monotonic()
    registry = verify_sources()
    prefix, archive_io, trace_summary, training_summary = bind_runtime()
    cfg, schema, archive_paths = archive_io.registered_inputs(REPO, root)
    selected = prefix.checked_rows(panel, status)
    require(prefix.equal(manifest, prefix.input_snapshot(panel, status, selected)), 'Input snapshot differs')
    pair_path = REPO / 'review/evidence/task-prefix-diagnostic-v2-result-20261030/pairs.json'
    pair_raw = read_small(pair_path)
    require(sha(pair_raw) == PAIRS_SHA, 'Previous public pair results changed')
    validate_pairs(json.loads(pair_raw.decode('utf-8-sig')))
    categories = {'student_farther': 0, 'same_distance_teacher_lower_unit': 0}
    compressed = 0
    for alias in ('S02', 'S03'):
        for side in (0, 1):
            deadline(started)
            game = selected[(alias, 'round2', side, 'merged')]
            identity = game['trace']
            path = prefix.safe_file(panel, *identity['path'].split('/'))
            require(identity['size'] <= COMPRESSED_LIMIT - compressed, 'Trace input budget exceeded')
            raw = read_small(path, identity['size'])
            compressed += len(raw)
            trace, record = decode_target(raw, identity['size'], identity['sha256'])
            prefix.check_trace_identity(trace, game['row'])
            category = trace_summary.classify_record(record, side, schema)
            require(category in categories, 'Unknown geometry category')
            categories[category] += 1
    require(sum(categories.values()) == 4, 'Incomplete recorded geometry')
    by_source = {}
    for name in ('old', 'new'):
        deadline(started)
        arrays = archive_io.read_archive(archive_paths[name], cfg, name)
        by_source[name] = training_summary.summarize_training_geometry(
            arrays['shared'], arrays['candidates'], arrays['offsets'], arrays['teacher_choice_label'],
            arrays['ops'], schema['shared_feature_names'], schema['feature_names'])
        chosen_ops = arrays['ops'][arrays['offsets'][:-1] + arrays['teacher_choice_label']]
        counts = {op: int((chosen_ops == index).sum()) for index, op in enumerate(schema['ops'])}
        require(counts == cfg['expected_operations'][name], 'Registered teacher labels differ')
        del arrays, chosen_ops
    deadline(started)
    prefix.verify_runtime()
    require(verify_sources() == registry, 'Registered source or selection changed')
    public = {'schema': 'worker-geometry-diagnostic-result-v1', 'passed': True,
        'historical_seed_roles': ['S02', 'S03'], 'historical_roles_already_consumed': True,
        'historical_trace_count': 4, 'unique_trace_lines_per_trace': 3,
        'historical_choice_counts': categories, 'training_sources': ['old', 'new'],
        'training_encoded_step_2_thresholds': public_training_summary(by_source),
        'training_categories_may_overlap': True, 'training_step_is_encoded_feature': True,
        'models_called': 0, 'training_updates': 0, 'new_games': 0, 'official_transitions': 0,
        'gpu_operations': 0, 'privacy': 'four historical category totals are public without role identities; training counts use only coarse thresholds',
        'interpretation': 'Exploratory matching of coarse structure, not a source join or causal evidence.'}
    return public, by_source


def write_new(path, data):
    payload = (json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode('utf-8')
    with path.open('xb') as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    require(path.read_bytes() == payload, 'Output verification failed')


def main():
    parser = QuietParser(description=__doc__)
    parser.add_argument('--game-root', type=Path, required=True)
    parser.add_argument('--panel', type=Path, required=True)
    parser.add_argument('--status', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--stage', type=Path, required=True)
    args = parser.parse_args()
    root, panel, status, manifest_path = (path.resolve(strict=True) for path in
                                          (args.game_root, args.panel, args.status, args.manifest))
    stage = args.stage.resolve(strict=False)
    require(stage.parent == (root / 'artifacts').resolve(strict=True) and not stage.exists()
            and not stage.is_relative_to(panel) and not stage.is_relative_to(REPO), 'Invalid output stage')
    manifest = json.loads(read_small(manifest_path).decode('utf-8-sig'))
    public, details = analyze(root, panel, status, manifest)
    stage.mkdir(exist_ok=False)
    write_new(stage / 'training-aggregate.local.json', details)
    pending, complete = stage / 'result.pending.json', stage / 'result.json'
    write_new(pending, public)
    require(not complete.exists(), 'Result already exists')
    os.replace(pending, complete)
    print(json.dumps({'passed': True, 'historical_traces': 4, 'training_archives': 2,
                      'model_calls': 0, 'official_transitions': 0}), flush=True)


if __name__ == '__main__':
    try:
        main()
    except SystemExit:
        raise
    except BaseException as exc:
        print(json.dumps({'passed': False, 'error_type': type(exc).__name__}), flush=True)
        raise SystemExit(1) from None
