"""One registered, streaming extraction of sparse historical task rows.

This module never scores a model or opens a replay. Original compressed files
are hashed once and decompressed once. Selection is fixed from the complete
index/label streams before any original input JSON is decoded. Failed partials
and counters remain local evidence; they cannot authorize a retry.
"""
from contextlib import ExitStack
import hashlib
import os
from pathlib import Path
import re
import time
import traceback
import zlib

from learning.bc_crop.agent import ROOT
from learning.task_bc_v1.cache import decode
from learning.task_choice_v1.features import candidate_id
from learning.task_bc_diagnostics_capacity_v1.pipeline import (
    _gzip_output, _write_progress, canonical, check_deadline, require,
    sha, strict_json, write_new,
)


STAGE = 'task-bc-stratified-error-sample-v1'
SCHEMA = 'task-stratified-collection-v1'
PHASES = ((0, 239), (240, 479), (480, 718))
CLASSES = ('DIG', 'FERTILIZE', 'HARVEST', 'PLANT', 'WATER')
KINDS = ('index', 'labels', 'inputs')
COUNTERS = (
    'compressed_source_bytes_hashed', 'index_metadata_rows', 'label_rows',
    'input_physical_lines_read', 'all_original_gzip_physical_lines',
    'input_json_rows_decoded', 'input_json_candidates_decoded',
    'decompressed_source_bytes',
)
ROW_COUNTER = dict(index='index_metadata_rows', labels='label_rows',
                   inputs='input_physical_lines_read')
TOKEN = re.compile(r'[A-Za-z0-9_-]+\Z')
HEX = re.compile(r'[0-9a-f]{64}\Z')
TEACHER_ID = re.compile(r'u(0|[1-9][0-9]*):x(0|[1-9][0-9]*):y(0|[1-9][0-9]*):([A-Z]+)\Z')
BLOCK_BYTES = 65536
MAX_LINE_BYTES = 16 * 1024 * 1024
MAX_CANDIDATES = 242
COUNTER_SEMANTICS = {
    'compressed_source_bytes_hashed': 'Bytes read by the hash pass, including failed hash attempts.',
    'decompressed_source_bytes': 'All bytes returned by inflate, including buffered lines not yet processed.',
    'physical_lines': 'Lines detected by the iterator, including rejected extra or unterminated rows; unprocessed buffered lines are not counted.',
    'input_json_rows_decoded': 'Charged parse attempts, including failed or deadline-cancelled attempts.',
    'input_json_candidates_decoded': 'Entries in successfully parsed candidate arrays, including subsequent failures.',
    'failed_actual_consumption_can_exceed_limit': True,
}


def _digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def _relative(base, value):
    require(isinstance(value, str) and value and '\\' not in value,
            'Expected a canonical relative source path')
    path = Path(value)
    require(not path.is_absolute() and ':' not in value and
            all(part not in ('', '.', '..') for part in value.split('/')),
            'Source path traversal or noncanonical path')
    resolved = (base / path).resolve()
    require(resolved.is_relative_to(base.resolve()), 'Source path escapes its root')
    return resolved


def _signature(path):
    stat = path.stat()
    return stat.st_size, stat.st_mtime_ns


class _Meter:
    def __init__(self, folder, limits, deadline):
        require(isinstance(limits, dict) and set(COUNTERS) <= set(limits),
                'Missing registered source scan limits')
        require(all(type(limits[k]) is int and limits[k] >= 0 for k in COUNTERS),
                'Invalid registered source scan limit')
        self.counts = dict.fromkeys(COUNTERS, 0)
        self.limits = {k: limits[k] for k in COUNTERS}
        self.folder, self.deadline = folder, deadline
        self.last_bytes = self.last_rows = 0
        self.started = time.monotonic()
        self.phase = 'started'

    def can(self, key, amount):
        check_deadline(self.deadline)
        require(type(amount) is int and amount >= 0 and
                self.counts[key] + amount <= self.limits[key],
                'Source scan allowance exhausted before consumption: ' + key)

    def record(self, **amounts):
        """Keep actual consumption even when its post-work check fails.

        Call can before a bounded read/allocation. Once work has happened, all
        counters belonging to that event are recorded before any limit or time
        check. An observed excess is evidence of failure, never permission for
        further consumption or a refund. JSON parse attempts are charged before
        parsing so an unsuccessful parse still retains its attempt charge.
        """
        require(amounts and all(key in self.counts and type(value) is int and value >= 0
                                for key, value in amounts.items()), 'Invalid consumed amount')
        for key, value in amounts.items():
            self.counts[key] += value
        for key in amounts:
            require(self.counts[key] <= self.limits[key],
                    'Source scan allowance exceeded after recorded consumption: ' + key)
        check_deadline(self.deadline)

    def snapshot(self, *, force=False):
        byte_count = (self.counts['compressed_source_bytes_hashed'] +
                      self.counts['decompressed_source_bytes'])
        row_count = self.counts['all_original_gzip_physical_lines']
        if not force and byte_count - self.last_bytes < 4 * 1024 * 1024 and \
                row_count - self.last_rows < 128:
            return
        _write_progress(self.folder / 'progress.json', dict(
            schema=SCHEMA, public_safe=False, phase=self.phase,
            counters=dict(self.counts), elapsed_seconds=time.monotonic() - self.started,
            counters_are_checkpoint_not_retry_permission=True))
        self.last_bytes, self.last_rows = byte_count, row_count


def _hash_source(path, info, meter):
    """Read the declared compressed bytes exactly once, in bounded chunks."""
    require(_signature(path)[0] == info['bytes'], 'Original compressed file size changed')
    signature = _signature(path)
    meter.can('compressed_source_bytes_hashed', info['bytes'])
    digest = hashlib.sha256()
    remaining = info['bytes']
    with path.open('rb') as stream:
        while remaining:
            requested = min(BLOCK_BYTES, remaining)
            meter.can('compressed_source_bytes_hashed', requested)
            raw = stream.read(requested)
            meter.record(compressed_source_bytes_hashed=len(raw))
            require(raw, 'Truncated original compressed file')
            digest.update(raw)
            remaining -= len(raw)
            meter.snapshot()
    require(_signature(path) == signature, 'Original compressed file changed while hashing')
    require(digest.hexdigest() == info['sha256'], 'Original compressed file SHA256 mismatch')
    return signature


def _physical_lines(path, info, kind, signature, meter):
    """Count actual zlib output, including skipped or incomplete physical rows.

    Output is limited before each decompression allocation, not after a complete
    gzip file has been read. The frozen original writer creates one gzip member;
    trailing bytes and additional members are rejected. The 16 MiB line bound is
    an explicit memory guard, never permission to truncate or substitute a row.
    """
    require(_signature(path) == signature, 'Source changed between hash and physical scan')
    inflater = zlib.decompressobj(16 + zlib.MAX_WBITS)
    pending, line = b'', bytearray()
    lines = raw_bytes = 0
    eof = False
    with path.open('rb') as stream:
        while True:
            check_deadline(meter.deadline)
            if not pending and not eof:
                raw = stream.read(BLOCK_BYTES)
                raw_bytes += len(raw)
                require(raw_bytes <= info['bytes'], 'Original gzip grew during scan')
                pending = raw
                eof = not raw
            remaining = meter.limits['decompressed_source_bytes'] - meter.counts['decompressed_source_bytes']
            require(remaining > 0, 'Decompressed source allowance exhausted before inflate')
            capacity = min(BLOCK_BYTES, remaining)
            meter.can('decompressed_source_bytes', capacity)
            piece = inflater.decompress(pending, capacity)
            pending = inflater.unconsumed_tail
            meter.record(decompressed_source_bytes=len(piece))
            for fragment in piece.splitlines(keepends=True):
                # splitlines also recognizes CR; only LF ends a physical JSONL
                # row, so any CR fragment remains in the same bounded line.
                require(len(line) + len(fragment) <= MAX_LINE_BYTES,
                        'Original JSONL line exceeds registered allocation bound')
                line.extend(fragment)
                if line.endswith(b'\n'):
                    # The complete line is now detected. Retain both counters
                    # together even if an extra row or the deadline rejects it.
                    # The full declared row allowance was checked before I/O.
                    meter.record(**{ROW_COUNTER[kind]: 1, 'all_original_gzip_physical_lines': 1})
                    lines += 1
                    require(lines <= info['rows'], 'Original stream has extra physical rows')
                    raw_line = bytes(line)
                    line.clear()
                    require(raw_line.strip(), 'Blank original JSONL row')
                    meter.snapshot()
                    yield raw_line
            meter.snapshot()
            if inflater.eof:
                require(not inflater.unused_data and not pending,
                        'Original gzip has trailing data or multiple members')
                require(raw_bytes == info['bytes'] and stream.read(1) == b'',
                        'Original gzip has trailing compressed bytes')
                break
            require(not (eof and not piece and not pending), 'Truncated original gzip member')
    if line:
        meter.record(**{ROW_COUNTER[kind]: 1, 'all_original_gzip_physical_lines': 1})
        raise ValueError('Original JSONL final physical row lacks newline')
    require(lines == info['rows'], 'Original stream physical row count mismatch')
    require(_signature(path) == signature, 'Source changed during physical scan')
    check_deadline(meter.deadline)


def _validate_meta(meta, group, previous):
    require(isinstance(meta, dict) and set(meta) == {
        'game', 'split', 'seed', 'side', 'opponent', 'step', 'decision', 'input_sha256'},
        'Invalid original index metadata schema')
    require(all(meta[k] == group[k] for k in ('game', 'split', 'seed', 'side', 'opponent')) and
            type(meta['seed']) is int and type(meta['side']) is int,
            'Original metadata identity mismatch')
    require(type(meta['step']) is int and 0 <= meta['step'] <= 718 and
            type(meta['decision']) is int and meta['decision'] >= 0,
            'Invalid original step or decision')
    require(isinstance(meta['input_sha256'], str) and HEX.fullmatch(meta['input_sha256']) is not None,
            'Invalid original input hash')
    current = meta['step'], meta['decision']
    require(previous is None or current > previous, 'Original index metadata is reordered')
    # Turns with no task decision need not appear in this dataset. Decisions
    # within every observed turn must start at zero and remain consecutive.
    require(current[1] == (0 if previous is None or current[0] != previous[0]
                           else previous[1] + 1), 'Original decision continuity mismatch')
    return current


def _teacher_class(label):
    require(isinstance(label, dict) and set(label) == {'index', 'teacher_label'} and
            type(label['index']) is int and 0 <= label['index'] < MAX_CANDIDATES and
            isinstance(label['teacher_label'], str), 'Invalid original label schema')
    match = TEACHER_ID.fullmatch(label['teacher_label'])
    require(match is not None, 'Original teacher ID is not canonical')
    unit, x, y = (int(match.group(k)) for k in (1, 2, 3))
    op = match.group(4)
    require(all(value < 32768 for value in (unit, x, y)) and op in CLASSES and
            candidate_id(unit, [x, y], op) == label['teacher_label'],
            'Original teacher ID or class is invalid')
    return op


class _Selector:
    """Fifteen fixed cells; at most two metadata/label records per cell."""
    def __init__(self, game):
        self.game = game
        self.populations = {(phase, op): 0 for phase in range(3) for op in CLASSES}
        self.selected = {key: [] for key in self.populations}

    def add(self, source_row_index, meta, label):
        phase = next(i for i, (start, stop) in enumerate(PHASES) if start <= meta['step'] <= stop)
        op = _teacher_class(label)
        key = phase, op
        self.populations[key] += 1
        rank = _digest([STAGE, self.game, source_row_index])
        bucket = self.selected[key]
        bucket.append((rank, source_row_index, meta, label))
        bucket.sort(key=lambda value: (value[0], value[1]))
        del bucket[2:]

    def result(self):
        strata, selected = [], []
        for (phase, op), population in self.populations.items():
            bucket = self.selected[phase, op]
            indices = sorted(value[1] for value in bucket)
            strata.append(dict(phase_index=phase, phase_step_range=list(PHASES[phase]),
                               teacher_class=op, population=population,
                               selected_count=len(indices), source_row_indices=indices))
            selected.extend(dict(source_row_index=index, meta=meta, label=label,
                                 phase_index=phase, teacher_class=op)
                            for _, index, meta, label in bucket)
        selected.sort(key=lambda value: value['source_row_index'])
        return selected, strata


def collect_group(out, cfg, deadline, group):
    """Extract one registered game; return and persist a complete local receipt.

    ``scan_limits`` is the caller's remaining source-scan allowance for this job.
    ``sample_rows_remaining`` and ``sample_candidates_remaining`` are the global
    materialized sample allowances remaining before this game. They do not grant
    permission to change membership: exceeding either stops the entire stage.
    """
    require(__debug__, 'The unchanged source decoder requires assertions enabled')
    out = Path(out).resolve()
    require(out.is_relative_to(ROOT.resolve()), 'Collection output must be inside the project')
    require(isinstance(group, dict) and isinstance(group.get('game'), str) and
            TOKEN.fullmatch(group['game']) is not None, 'Invalid registered game identity')
    folder = out / 'collection' / group['game']
    folder.mkdir(parents=True, exist_ok=False)
    counts = dict.fromkeys(COUNTERS, 0)
    meter = None
    started = time.monotonic()
    hashes, streams = 0, 0
    try:
        require(cfg.get('stage') == STAGE, 'Wrong registered sampling stage')
        meter = _Meter(folder, cfg['scan_limits'], deadline)
        counts = meter.counts
        write_new(folder / 'started.json', dict(schema=SCHEMA, stage=STAGE,
            game=group['game'], original_stream_passes_allowed=1,
            public_safe=False, retry_allowed=False, source_limits=meter.limits))
        meter.snapshot(force=True)
        source = _relative(ROOT.resolve(), cfg['source_root'])
        require(group.get('split') in ('train', 'validation') and
                type(group.get('seed')) is int and group['seed'] >= 0 and
                type(group.get('side')) is int and group['side'] in (0, 1) and
                group.get('opponent') in ('v1', 'market', 'round2'), 'Invalid registered source identity')
        require(set(group['source_files']) == set(KINDS), 'Missing registered original streams')
        require(type(cfg['sample_rows_remaining']) is int and cfg['sample_rows_remaining'] >= 0 and
                type(cfg['sample_candidates_remaining']) is int and cfg['sample_candidates_remaining'] >= 0,
                'Invalid remaining materialized sample allowance')
        paths, signatures = {}, {}
        for kind in KINDS:
            info = group['source_files'][kind]
            require(type(info['rows']) is int and info['rows'] >= 0 and type(info['bytes']) is int and
                    info['bytes'] > 0 and isinstance(info['sha256'], str) and HEX.fullmatch(info['sha256']) is not None,
                    'Invalid registered original stream counts or SHA256')
            expected = 'data/' + group['split'] + '/' + group['game'] + '.' + kind + '.jsonl.gz'
            require(info['path'] == expected, 'Original stream path does not match its identity')
            paths[kind] = _relative(source, info['path'])
            meter.can(ROW_COUNTER[kind], info['rows'])
        declared_rows = group['source_files']['index']['rows']
        require(all(info['rows'] == declared_rows for info in group['source_files'].values()),
                'Registered original streams are not aligned')
        meter.can('all_original_gzip_physical_lines', declared_rows * 3)
        meter.can('compressed_source_bytes_hashed', sum(info['bytes'] for info in group['source_files'].values()))
        for kind in KINDS:
            meter.phase = 'hash-' + kind
            signatures[kind] = _hash_source(paths[kind], group['source_files'][kind], meter)
            hashes += 1
        meter.phase = 'select-metadata'
        selector = _Selector(group['game'])
        previous = None
        with ExitStack() as stack:
            readers = {kind: _physical_lines(paths[kind], group['source_files'][kind], kind,
                                            signatures[kind], meter) for kind in ('index', 'labels')}
            for reader in readers.values():
                stack.callback(reader.close)
            for source_row_index in range(declared_rows):
                meta = strict_json(next(readers['index']))
                label = strict_json(next(readers['labels']))
                previous = _validate_meta(meta, group, previous)
                selector.add(source_row_index, meta, label)
            for kind, reader in readers.items():
                require(next(reader, None) is None, 'Original metadata has extra rows')
                streams += 1
        chosen, strata = selector.result()
        indices = [row['source_row_index'] for row in chosen]
        require(len(indices) <= 30 and len(indices) <= cfg['sample_rows_remaining'],
                'Fixed selection exceeds remaining row allowance; no truncation')
        meter.can('input_json_rows_decoded', len(indices))
        # The immutable source manifest bounds every original row at 242
        # candidates. Reserve that worst case before parsing selected JSON.
        meter.can('input_json_candidates_decoded', len(indices) * MAX_CANDIDATES)
        write_new(folder / 'selection-fixed.json', dict(schema=SCHEMA, stage=STAGE,
            game=group['game'], source_row_indices=indices, source_indices_sha256=_digest(indices),
            source_sha256={kind: group['source_files'][kind]['sha256'] for kind in KINDS},
            strata=strata, membership_fixed_before_input_json_decoding=True,
            no_logits_errors_returns_or_numeric_deduplication=True))
        meter.snapshot(force=True)
        meter.phase = 'extract-selected-inputs'
        target = out / 'sample' / (group['game'] + '.jsonl.gz')
        by_index = {row['source_row_index']: row for row in chosen}
        schema, candidates, emitted = None, 0, 0
        selections = []
        reader = _physical_lines(paths['inputs'], group['source_files']['inputs'], 'inputs',
                                 signatures['inputs'], meter)
        with ExitStack() as stack:
            stack.callback(reader.close)
            output = stack.enter_context(_gzip_output(target))
            for source_row_index, raw in enumerate(reader):
                if source_row_index not in by_index:
                    continue
                picked = by_index[source_row_index]
                meter.can('input_json_candidates_decoded', MAX_CANDIDATES)
                meter.can('input_json_rows_decoded', 1)
                meter.record(input_json_rows_decoded=1)
                inputs = strict_json(raw)
                require(isinstance(inputs, dict) and isinstance(inputs.get('candidates'), list),
                        'Selected original input has no candidate list')
                count = len(inputs['candidates'])
                meter.record(input_json_candidates_decoded=count)
                require(0 < count <= MAX_CANDIDATES, 'Original candidate count exceeds frozen population bound')
                schema, *_ = decode(inputs, picked['label'], picked['meta'],
                    game=group['game'], split=group['split'], expected_seed=group['seed'],
                    expected_side=group['side'], expected_opponent=group['opponent'], schema=schema)
                teacher = inputs['candidates'][picked['label']['index']]
                require(teacher['op'] == picked['teacher_class'], 'Selected teacher class differs from stratum')
                require(candidates + count <= cfg['sample_candidates_remaining'],
                        'Fixed selection exceeds remaining candidate allowance; no truncation')
                candidates += count
                selections.append(dict(source_row_index=source_row_index, step=picked['meta']['step'],
                    decision=picked['meta']['decision'], teacher_class=picked['teacher_class'],
                    phase_index=picked['phase_index'], candidate_count=count))
                output.write(canonical(dict(inputs=inputs, label=picked['label'], meta=picked['meta'],
                                            source_row_index=source_row_index)) + b'\n')
                emitted += 1
                meter.snapshot()
        streams += 1
        require(emitted == len(indices), 'An exact selected physical source row is missing')
        require(counts['all_original_gzip_physical_lines'] == 3 * declared_rows,
                'Original complete physical scan was not counted')
        result_group = {key: group[key] for key in ('game', 'split', 'seed', 'side', 'opponent')}
        result_group.update(path=target.relative_to(out).as_posix(), rows=emitted, candidates=candidates,
            sha256=sha(target, deadline), source_sha256={kind: group['source_files'][kind]['sha256'] for kind in KINDS},
            source_row_indices=indices, source_indices_sha256=_digest(indices),
            strata=strata, selections=selections, source_population_rows=declared_rows)
        meter.phase = 'completed'
        meter.snapshot(force=True)
        receipt = dict(schema=SCHEMA, stage=STAGE, passed=True, public_safe=False,
            game=group['game'], group=result_group, counters=dict(counts), counter_semantics=COUNTER_SEMANTICS,
            source_files_hashed=hashes, source_streams_completed=streams,
            selection_fixed_sha256=sha(folder / 'selection-fixed.json', deadline),
            elapsed_seconds=time.monotonic() - started, model_calls=0, training_runs=0,
            new_games=0, policy_calls=0, original_stream_passes=1, retry_allowed=False)
        write_new(folder / 'receipt.json', receipt)
        return receipt
    except BaseException as error:
        if meter is not None:
            meter.phase = 'failed'
            meter.snapshot(force=True)
        failure = dict(schema=SCHEMA, stage=STAGE, passed=False, public_safe=False,
            game=group['game'], counters=dict(counts), counter_semantics=COUNTER_SEMANTICS, source_files_hashed=hashes,
            source_streams_completed=streams, elapsed_seconds=time.monotonic() - started,
            error_type=type(error).__name__, retry_allowed=False, model_calls=0,
            training_runs=0, new_games=0, policy_calls=0)
        write_new(folder / 'receipt.json', failure)
        with (folder / 'local-error.txt').open('x', encoding='utf-8', newline='\n') as stream:
            stream.write(traceback.format_exc())
        raise
