"""Compare a frozen sampler with matched-size uniform sampling, without training.

Only aggregate metadata from the two registered training archives is emitted.
Historical batch indices were not recorded, so this is a conditional reconstruction.
"""
import argparse
import ast
import hashlib
import io
import json
import os
from pathlib import Path
import struct
import time
import zipfile

import numpy as np

REPO = Path(__file__).resolve().parents[3]
EVIDENCE = REPO / 'review/evidence/task-teacher-merge-bc-v1-20261028'
IDENTITIES = {
    'registration.json': '9f95b190268158cb99f3f1a09c0dd0c58a24be8b1bfac7ae30c9ebe3c3ab55a5',
    'freeze.json': '1b7dee4cc8551067b30ee4266f7fe4fe89130142c8ce0dd97eaf01e3c09f1a2e',
    'fit-result.json': '110c4ed5a836c94fd101ac5f08bbcd3efc2877fee53b2704a2c87380fea71c38',
}
SOURCES = (
    ('public_runtime_sha256', 'task_teacher_choice_bc_v1/training.py'),
    ('public_runtime_sha256', 'task_bc_v1/model.py'),
    ('public_runtime_sha256', 'task_bc_v1/train.py'),
    ('public_runtime_sha256', 'task_teacher_merge_v1/merge.py'),
    ('public_runtime_sha256', 'task_teacher_choice_bc_v1/numeric.py'),
    ('new_source_sha256', 'task_teacher_merge_bc_v1/gpu_entry.py'),
)
BUDGET = 65536
MAX_GROUPS = 256
UPDATES = 128
TIME_LIMIT_SECONDS = 90
ARRAY_BYTES_LIMIT = 4 * 1024 * 1024
HEADER_LIMIT = 4096


def require(condition, message):
    if not condition:
        raise ValueError(message)


class QuietParser(argparse.ArgumentParser):
    def error(self, message):
        raise ValueError('Invalid command line')


def deadline(started):
    if time.monotonic() - started > TIME_LIMIT_SECONDS:
        raise TimeoutError('Diagnostic time limit exceeded')


def snapshot(path, maximum, started):
    with path.open('rb') as stream:
        raw = stream.read(maximum + 1)
    require(len(raw) <= maximum, 'Snapshot size exceeds limit')
    deadline(started)
    return raw


def literal_assignment(raw, name):
    tree = ast.parse(raw.decode('utf-8-sig'))
    matches = [node.value for node in tree.body if isinstance(node, ast.Assign)
               and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
               and node.targets[0].id == name]
    require(len(matches) == 1, 'Missing or ambiguous frozen literal')
    return ast.literal_eval(matches[0])


def padded_width(width):
    width = int(width)
    require(1 <= width <= BUDGET, 'Invalid complete group width')
    return 1 << (width - 1).bit_length()


def original_batch(widths, order, budget=BUDGET, max_groups=MAX_GROUPS):
    """Frozen order-dependent skip rule; retain later groups after a rejection."""
    require(0 < budget <= BUDGET and 0 < max_groups <= MAX_GROUPS, 'Invalid batch limit')
    selected, width, skipped = [], 0, 0
    for identifier in order:
        identifier = int(identifier)
        next_width = max(width, padded_width(widths[identifier]))
        if (1 << len(selected).bit_length()) * next_width > budget:
            skipped += 1
            continue
        selected.append(identifier)
        width = next_width
        if len(selected) == max_groups:
            break
    require(bool(selected), 'No group fits the batch')
    capacity = 1 << (len(selected) - 1).bit_length()
    return selected, skipped, capacity * width


def microbatch_geometry(widths, identifiers, budget=BUDGET):
    """Count the padded shapes of the existing whole-group width buckets."""
    buckets = {}
    for identifier in identifiers:
        width = padded_width(widths[int(identifier)])
        require(width <= budget, 'A complete group exceeds the budget')
        buckets[width] = buckets.get(width, 0) + 1
    padded, batches = 0, 0
    for width, count in buckets.items():
        capacity = min(MAX_GROUPS, budget // width)
        number = (count + capacity - 1) // capacity
        batches += number
        padded += number * capacity * width
    return batches, padded


def aggregate(population, baseline_counts, baseline_weights, uniform_counts,
              uniform_weights, masks, updates):
    total = len(population)
    baseline_total = int(baseline_counts.sum())
    uniform_total = int(uniform_counts.sum())
    sizes = {name: int(np.count_nonzero(mask)) for name, mask in masks.items()}
    # Suppress the whole dimension so totals cannot expose one omitted small cell.
    if any(0 < count < 20 for count in sizes.values()):
        return {'withheld': True, 'reason': 'insufficient_group_size'}
    rows = {}
    for name, mask in masks.items():
        count = sizes[name]
        if not count:
            continue
        rows[name] = {
            'training_groups': count,
            'population_share': round(count / total, 6),
            'reconstructed_exposures': int(baseline_counts[mask].sum()),
            'reconstructed_exposure_share': round(float(baseline_counts[mask].sum()) / baseline_total, 6),
            'reconstructed_mean_update_share': round(float(baseline_weights[mask].sum()) / updates, 6),
            'matched_uniform_exposures': int(uniform_counts[mask].sum()),
            'matched_uniform_exposure_share': round(float(uniform_counts[mask].sum()) / uniform_total, 6),
            'matched_uniform_mean_update_share': round(float(uniform_weights[mask].sum()) / updates, 6),
        }
    return rows


def reconstruct(widths, labels, old_count, train_seed, dimensions, operations, started):
    n = len(widths)
    require(n and 0 < old_count < n and len(labels) == n, 'Invalid population')
    require(all(padded_width(width) <= BUDGET for width in widths), 'Oversized group')
    rng = np.random.default_rng(train_seed)
    require(type(rng.bit_generator).__name__ == 'PCG64', 'Unexpected NumPy bit generator')
    # fit() consumes these normal arrays before generating its first permutation.
    for a, b in zip(dimensions[:-1], dimensions[1:]):
        rng.standard_normal((a, b))
    uniform_rng = np.random.default_rng(np.random.SeedSequence([train_seed, 0x53414D50]))
    baseline_counts = np.zeros(n, dtype=np.int32)
    uniform_counts = np.zeros(n, dtype=np.int32)
    baseline_weights = np.zeros(n, dtype=np.float64)
    uniform_weights = np.zeros(n, dtype=np.float64)
    batches, skipped, old_padding, new_padding, micro_count = [], [], [], [], []
    for _ in range(UPDATES):
        deadline(started)
        selected, omissions, padded = original_batch(widths, rng.permutation(n))
        k = len(selected)
        # A separate stream avoids conditioning this draw on the old ordering.
        uniform = uniform_rng.permutation(n)[:k]
        baseline_counts[selected] += 1
        uniform_counts[uniform] += 1
        baseline_weights[selected] += 1.0 / k
        uniform_weights[uniform] += 1.0 / k
        microbatches, micro_padded = microbatch_geometry(widths, uniform)
        batches.append(k)
        skipped.append(omissions)
        old_padding.append(padded)
        new_padding.append(micro_padded)
        micro_count.append(microbatches)
    require(int(baseline_counts.sum()) == int(uniform_counts.sum()) == sum(batches),
            'Exposure accounting mismatch')
    positions = np.arange(n)
    source_masks = {'old': positions < old_count, 'new': positions >= old_count}
    operation_masks = {name: labels == i for i, name in enumerate(operations)}
    width_masks = {'1-64': widths <= 64, '65-128': (widths > 64) & (widths <= 128),
                   '129-256': (widths > 128) & (widths <= 256), '257+': widths > 256}
    def by(masks):
        return aggregate(widths, baseline_counts, baseline_weights, uniform_counts,
                         uniform_weights, masks, UPDATES)
    def count_or_withheld(value):
        return None if 0 < value < 20 else value
    return {
        'population_groups': n,
        'updates_reconstructed': UPDATES,
        'batch_groups_min': min(batches), 'batch_groups_max': max(batches),
        'batch_groups_mean': round(float(np.mean(batches)), 4),
        'total_exposures_each_method': sum(batches),
        'groups_never_selected_reconstructed': count_or_withheld(int(np.count_nonzero(baseline_counts == 0))),
        'groups_never_selected_matched_uniform': count_or_withheld(int(np.count_nonzero(uniform_counts == 0))),
        'skipped_before_end_total': count_or_withheld(sum(skipped)),
        'old_padded_candidate_slots_total': sum(old_padding),
        'matched_uniform_microbatch_padded_slots_total': sum(new_padding),
        'matched_uniform_microbatches_total': sum(micro_count),
        'by_source': by(source_masks), 'by_teacher_operation': by(operation_masks),
        'by_group_width': by(width_masks),
    }


def load_training_metadata(path, expected_sha, expected_groups, expected_candidates,
                           expected_operations, operations, limits, started):
    require(path.is_file() and path.stat().st_size <= limits['max_archive_bytes'], 'Invalid archive')
    raw = snapshot(path, limits['max_archive_bytes'], started)
    return extract_training_metadata(raw, expected_sha, expected_groups, expected_candidates, expected_operations,
                                     operations, limits, started)


def extract_training_metadata(raw, expected_sha, expected_groups, expected_candidates, expected_operations,
                              operations, limits, started):
    require(hashlib.sha256(raw).hexdigest() == expected_sha, 'Training archive identity changed')
    names = ('offsets', 'teacher_choice_label', 'ops')
    expected_shapes = {'offsets': (expected_groups + 1,), 'teacher_choice_label': (expected_groups,),
                       'ops': (expected_candidates,)}
    expected_dtypes = {'offsets': np.dtype('int64'), 'teacher_choice_label': np.dtype('int64'),
                       'ops': np.dtype('uint8')}
    stream = io.BytesIO(raw)
    with zipfile.ZipFile(stream) as archive:
        members = archive.infolist()
        for name in names:
            matches = [member for member in members if member.filename == name + '.npy']
            require(len(matches) == 1 and matches[0].file_size <= ARRAY_BYTES_LIMIT,
                    'Invalid training metadata member')
            info = matches[0]
            require(not info.flag_bits & 1 and info.compress_type == zipfile.ZIP_STORED,
                    'Invalid training metadata encoding')
            with archive.open(info) as member:
                version = np.lib.format.read_magic(member)
                require(version in ((1, 0), (2, 0)), 'Unsupported metadata header')
                prefix_size = 2 if version == (1, 0) else 4
                prefix = member.read(prefix_size)
                require(len(prefix) == prefix_size, 'Truncated metadata header')
                header_length = struct.unpack('<H' if prefix_size == 2 else '<I', prefix)[0]
                require(header_length <= HEADER_LIMIT, 'Metadata header exceeds limit')
                header = member.read(header_length)
                require(len(header) == header_length, 'Truncated metadata header')
                reader = (np.lib.format.read_array_header_1_0 if version == (1, 0)
                          else np.lib.format.read_array_header_2_0)
                shape, fortran, dtype = reader(io.BytesIO(prefix + header), max_header_size=HEADER_LIMIT)
                require(shape == expected_shapes[name] and not fortran and dtype == expected_dtypes[name],
                        'Invalid training metadata header')
                require(member.tell() <= HEADER_LIMIT and member.tell() + int(np.prod(shape)) * dtype.itemsize
                        == info.file_size, 'Training metadata byte size inconsistent')
        require(sum(m.file_size for m in members) <= limits['max_uncompressed_bytes'],
                'Training archive uncompressed budget exceeded')
    stream.seek(0)
    with np.load(stream, allow_pickle=False) as archive:
        # Do not access shared/candidate features or the per-group game seed array.
        offsets, labels, ops = (archive[name].copy() for name in names)
    require(offsets.dtype == np.dtype('int64') and labels.dtype == np.dtype('int64')
            and ops.dtype == np.dtype('uint8') and all(array.ndim == 1
                for array in (offsets, labels, ops)), 'Invalid training metadata dtype')
    require(len(labels) == expected_groups and len(offsets) == expected_groups + 1
            and len(ops) == expected_candidates and int(offsets[0]) == 0
            and int(offsets[-1]) == expected_candidates and np.all(offsets[1:] > offsets[:-1]),
            'Training metadata shape changed')
    widths = np.diff(offsets)
    require(np.all((widths > 0) & (widths <= BUDGET)) and np.all(labels >= 0)
            and np.all(labels < widths) and np.all((ops >= 0) & (ops < len(operations))),
            'Training metadata values invalid')
    chosen = ops[offsets[:-1] + labels].astype(np.int64)
    counts = np.bincount(chosen, minlength=len(operations))
    require({name: int(counts[i]) for i, name in enumerate(operations)} == expected_operations,
            'Frozen operation counts changed')
    deadline(started)
    return widths.astype(np.int64), chosen


def analyze(game_root):
    started = time.monotonic()
    documents = {}
    for name, expected in IDENTITIES.items():
        path = EVIDENCE / name
        raw = snapshot(path, ARRAY_BYTES_LIMIT, started)
        require(hashlib.sha256(raw).hexdigest() == expected, 'Frozen evidence changed')
        documents[name] = json.loads(raw.decode('utf-8-sig'))
    cfg, freeze, result = (documents[name] for name in ('registration.json', 'freeze.json', 'fit-result.json'))
    require(result['passed'] is True and result['updates'] == UPDATES
            and cfg['training']['updates'] == UPDATES, 'Frozen fit not accepted')
    require(np.__version__ == cfg['wsl_software_versions']['numpy'], 'NumPy version differs')
    source_bytes = {}
    for section, suffix in SOURCES:
        matches = [name for name in freeze[section] if name.endswith(suffix)]
        require(len(matches) == 1, 'Frozen source identity ambiguous')
        name = matches[0]
        path = REPO / name if (REPO / name).exists() else REPO / 'kaggriculture' / name
        raw = snapshot(path, ARRAY_BYTES_LIMIT, started)
        require(hashlib.sha256(raw).hexdigest() == freeze[section][name], 'Frozen source changed')
        source_bytes[suffix] = raw
    dimensions = literal_assignment(source_bytes['task_bc_v1/model.py'], 'DIMENSIONS')
    features_path = REPO / 'kaggriculture/learning/task_choice_v1/features.py'
    matches = [name for name in freeze['public_runtime_sha256'] if name.endswith('task_choice_v1/features.py')]
    raw = snapshot(features_path, ARRAY_BYTES_LIMIT, started)
    require(len(matches) == 1 and hashlib.sha256(raw).hexdigest() == freeze['public_runtime_sha256'][matches[0]],
            'Frozen operation schema changed')
    operations = literal_assignment(raw, 'OPS')
    require(dimensions == (373, 128, 64, 1) and len(operations) == 5, 'Frozen model schema changed')
    artifacts = (game_root / 'artifacts').resolve(strict=True)
    arrays = []
    for name in ('old', 'new'):
        path = (game_root / cfg['archive_path'][name]).resolve(strict=True)
        require(path.is_relative_to(artifacts), 'Training archive outside artifacts')
        arrays.append(load_training_metadata(path, cfg['archive_sha256'][name], cfg['expected_groups'][name],
            cfg['expected_candidates'][name], cfg['expected_operations'][name], operations,
            cfg['data_limits'], started))
    widths = np.concatenate([item[0] for item in arrays])
    labels = np.concatenate([item[1] for item in arrays])
    seed = cfg['training']['training_seed']
    require(type(seed) is int and 0 <= seed <= 0xffffffff, 'Frozen RNG configuration invalid')
    summary = reconstruct(widths, labels, len(arrays[0][0]), seed, dimensions, operations, started)
    return {
        'schema': 'task-sampling-diagnostic-v1', 'passed': True,
        'scope': 'conditional source reconstruction; historical batch indices were not recorded',
        'numpy_version': np.__version__, 'bit_generator': 'PCG64',
        'source_order': ['old', 'new'], 'selected_archive_arrays': ['offsets', 'teacher_choice_label', 'ops'],
        'uniform_comparison': 'separate RNG stream, same reconstructed per-update group counts',
        'uniform_population_share_is_expectation': True,
        'small_categories_suppressed_below_groups': 20,
        'small_category_policy': 'withhold the full dimension if any nonempty category has fewer than 20 groups',
        'training_updates': 0, 'model_calls': 0, 'official_transitions': 0,
        'gpu_operations': 0, 'summary': summary,
    }


def main():
    parser = QuietParser(description=__doc__)
    parser.add_argument('--game-root', type=Path, required=True)
    parser.add_argument('--stage', type=Path, required=True)
    args = parser.parse_args()
    root = args.game_root.resolve(strict=True)
    parent = (root / 'artifacts').resolve(strict=True)
    stage = args.stage.resolve(strict=False)
    require(stage.parent == parent and not stage.exists(), 'Stage must be a new artifacts directory')
    result = analyze(root)
    stage.mkdir(exist_ok=False)
    pending = stage / 'summary.pending.json'
    completed = stage / 'summary.json'
    with pending.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    require(json.loads(pending.read_text(encoding='utf-8')) == result and not completed.exists(),
            'Pending result verification failed')
    os.replace(pending, completed)
    print(json.dumps({'passed': True, 'training_updates': 0, 'official_transitions': 0}), flush=True)


if __name__ == '__main__':
    try:
        main()
    except (Exception, KeyboardInterrupt) as exc:
        print(json.dumps({'passed': False, 'error_type': type(exc).__name__}), flush=True)
        raise SystemExit(1) from None
