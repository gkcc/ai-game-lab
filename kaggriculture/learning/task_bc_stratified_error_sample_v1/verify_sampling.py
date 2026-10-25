"""Six registered tiny sampler regressions; no original source or model I/O.

Call run_regressions only after the parent has frozen and committed this matrix.
Fixtures use 18 synthetic aligned rows and 36 synthetic candidates in total.
Two additional cases exercise in-memory accounting only and create no inputs.
"""
from copy import deepcopy
import gzip
import hashlib
from pathlib import Path
import time

from learning.bc_crop.agent import ROOT
from learning.task_choice_v1.features import SCHEMA_VERSION, candidate_id, schema_hash
from . import sampling as s


def _input(step, op):
    local_names = ['x' + str(i) for i in range(98)]
    shared_names = ['s' + str(i) for i in range(275)]
    return dict(schema_version=SCHEMA_VERSION, schema_sha256=schema_hash(local_names, shared_names),
        feature_names=local_names, shared_feature_names=shared_names, shared_x=[0.] * 275,
        observation=dict(step=step, player=0, synthetic_prediction=999., synthetic_return=-999.),
        candidates=[dict(id=candidate_id(0, [x, 0], op), unit=0, target=[x, 0], op=op,
                         x=[float(x)] * 98) for x in range(2)])


def _write_gzip(path, values):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as raw:
        with gzip.GzipFile(filename='', mode='wb', fileobj=raw, mtime=0) as stream:
            for value in values:
                stream.write(s.canonical(value) + b'\n')


def _fixture(folder, game, specs, mutation=None):
    source = folder / 'source'
    rows = []
    for row_index, (step, decision, op) in enumerate(specs):
        inputs = _input(step, op)
        label = dict(index=1, teacher_label=inputs['candidates'][1]['id'])
        meta = dict(game=game, split='train', seed=7, side=0, opponent='v1',
                    step=step, decision=decision, input_sha256=s._digest(inputs))
        rows.append(dict(inputs=inputs, label=label, meta=meta, source_row_index=row_index))
    if mutation is not None:
        mutation(rows)
    files = {}
    for kind, field in (('inputs', 'inputs'), ('labels', 'label'), ('index', 'meta')):
        path = source / 'data' / 'train' / (game + '.' + kind + '.jsonl.gz')
        _write_gzip(path, [row[field] for row in rows])
        files[kind] = dict(path=path.relative_to(source).as_posix(), rows=len(rows),
                           bytes=path.stat().st_size, sha256=s.sha(path))
    group = dict(game=game, split='train', seed=7, side=0, opponent='v1', source_files=files)
    limits = dict(compressed_source_bytes_hashed=sum(v['bytes'] for v in files.values()),
        index_metadata_rows=len(rows), label_rows=len(rows), input_physical_lines_read=len(rows),
        all_original_gzip_physical_lines=3 * len(rows), input_json_rows_decoded=len(rows),
        input_json_candidates_decoded=242 * len(rows), decompressed_source_bytes=2 * 1024 * 1024)
    cfg = dict(stage=s.STAGE, source_root=source.relative_to(ROOT).as_posix(), scan_limits=limits,
               sample_rows_remaining=30, sample_candidates_remaining=64)
    return cfg, group, rows


def _reject(out, cfg, group, expected, deadline):
    try:
        s.collect_group(out, cfg, deadline, group)
    except (ValueError, AssertionError) as error:
        s.require(expected in str(error) or (isinstance(error, AssertionError) and expected == 'source decode'),
                  'Synthetic sampler rejected for an unexpected reason')
        receipt = s.strict_json((out / 'collection' / group['game'] / 'receipt.json').read_bytes())
        s.require(receipt['passed'] is False and receipt['retry_allowed'] is False and
                  (out / 'collection' / group['game'] / 'local-error.txt').is_file(),
                  'Synthetic failure did not preserve local counters and error')
        return receipt
    raise AssertionError('Synthetic sampler accepted an invalid original stream')


def run_regressions(out, deadline=None):
    out = Path(out).resolve()
    s.require(out.is_relative_to(ROOT.resolve()), 'Synthetic output must be an owned project run')
    out.mkdir(parents=True, exist_ok=False)
    deadline = min(time.monotonic() + 60, deadline) if deadline is not None else time.monotonic() + 60
    cases, receipts = [], []
    s.write_new(out / 'started.json', dict(schema='task-stratified-sampling-regressions-v1',
        synthetic_only=True, expected_cases=6, synthetic_rows=18, synthetic_candidates=36,
        original_source_reads=0, model_calls=0, retry_allowed=False))

    specs = [(5, i, 'PLANT') for i in range(4)] + [(300, i, 'HARVEST') for i in range(3)] + \
            [(700, i, 'FERTILIZE') for i in range(2)]
    cfg, group, rows = _fixture(out / 'sparse', 'synthetic_sparse', specs)
    result = s.collect_group(out / 'sparse' / 'output', cfg, deadline, group)
    selected = result['group']
    expected = []
    for start, stop in ((0, 4), (4, 7), (7, 9)):
        ranked = sorted(range(start, stop), key=lambda i: (
            hashlib.sha256(s.canonical([s.STAGE, group['game'], i])).hexdigest(), i))
        expected.extend(ranked[:2])
    expected.sort()
    s.require(selected['source_row_indices'] == expected and selected['rows'] == 6 and
              selected['candidates'] == 12 and selected['source_indices_sha256'] == s._digest(expected),
              'Stable per-cell minimum-hash selection or sparse binding changed')
    with gzip.open(out / 'sparse' / 'output' / selected['path'], 'rb') as stream:
        saved = [s.strict_json(line) for line in stream]
    s.require(saved == [rows[i] for i in expected] and
              any(row['meta']['decision'] > 0 for row in saved),
              'Selected original sparse step/decision or values were rewritten')
    s.require(len(selected['strata']) == 15 and
              sum(cell['population'] for cell in selected['strata']) == 9 and
              sum(cell['population'] == 0 for cell in selected['strata']) == 12 and
              all(cell['selected_count'] == min(2, cell['population']) for cell in selected['strata']),
              'Empty cells were filled or quotas changed')
    # Selection consumes only metadata/label and physical identity. Change
    # dummy input predictions/returns while preserving each original identity;
    # no second source extraction, JSON decode, or model call is performed.
    selector = s._Selector(group['game'])
    for i, row in enumerate(rows):
        altered = deepcopy(row)
        altered['inputs']['observation'].update(synthetic_prediction=-777., synthetic_return=888.)
        selector.add(i, altered['meta'], altered['label'])
    repeated, _ = selector.result()
    s.require([row['source_row_index'] for row in repeated] == expected,
              'Prediction or reward values influenced selection')
    s.require(result['counters']['all_original_gzip_physical_lines'] == 27 and
              result['counters']['input_json_rows_decoded'] == 6 and
              result['counters']['input_json_candidates_decoded'] == 12,
              'Skipped rows or selected candidate decoding were miscounted')
    cases.append(dict(name='stable_sparse_selection_and_empty_strata', passed=True))
    receipts.append(result)

    short = [(5, 0, 'PLANT'), (300, 0, 'HARVEST'), (700, 0, 'FERTILIZE')]
    cfg, group, _ = _fixture(out / 'bad-identity', 'synthetic_identity', short,
        lambda values: values[1]['meta'].update(side=1))
    result = _reject(out / 'bad-identity' / 'output', cfg, group, 'identity mismatch', deadline)
    s.require(result['counters']['input_json_rows_decoded'] == 0,
              'Invalid full metadata reached selected input decode')
    cases.append(dict(name='foreign_metadata_identity_rejected_before_inputs', passed=True))
    receipts.append(result)

    cfg, group, _ = _fixture(out / 'bad-continuity', 'synthetic_continuity',
        [(5, 0, 'PLANT'), (5, 2, 'PLANT'), (700, 0, 'FERTILIZE')])
    result = _reject(out / 'bad-continuity' / 'output', cfg, group, 'continuity mismatch', deadline)
    s.require(result['counters']['input_json_rows_decoded'] == 0,
              'Invalid decision continuity reached selected input decode')
    cases.append(dict(name='full_metadata_missing_decision_rejected', passed=True))
    receipts.append(result)

    cfg, group, _ = _fixture(out / 'bad-teacher', 'synthetic_teacher', short,
        lambda values: values[0]['label'].update(index=0))
    result = _reject(out / 'bad-teacher' / 'output', cfg, group, 'source decode', deadline)
    s.require(result['counters']['input_json_rows_decoded'] == 1 and
              result['counters']['input_json_candidates_decoded'] == 2,
              'Failed selected triple decoding was not counted')
    cases.append(dict(name='selected_triple_teacher_alignment_rejected', passed=True))
    receipts.append(result)

    # No file reads, JSON parsing, or model calls in these two counter cases.
    # Pretend a bounded operation completed after its successful precheck.
    limits = dict.fromkeys(s.COUNTERS, 100)
    meter = s._Meter(out / 'unused-counter-folder', limits, deadline)
    completed = dict(compressed_source_bytes_hashed=7, decompressed_source_bytes=11,
                     index_metadata_rows=1, all_original_gzip_physical_lines=1)
    for key, value in completed.items():
        meter.can(key, value)
    meter.deadline = 0.0
    try:
        meter.record(**completed)
    except TimeoutError:
        pass
    else:
        raise AssertionError('Completed consumption did not reject an expired deadline')
    s.require(all(meter.counts[key] == value for key, value in completed.items()),
              'Deadline rejection lost consumed bytes or half of the physical-row event')
    cases.append(dict(name='consumed_bytes_and_row_survive_deadline_rejection', passed=True))
    deadline_counts = dict(meter.counts)

    limits = dict.fromkeys(s.COUNTERS, 100)
    limits['input_json_candidates_decoded'] = 1
    meter = s._Meter(out / 'unused-counter-folder', limits, deadline)
    meter.can('input_json_candidates_decoded', 1)
    try:
        meter.record(input_json_candidates_decoded=2)
    except ValueError as error:
        s.require('after recorded consumption' in str(error), 'Wrong post-consumption quota failure')
    else:
        raise AssertionError('Observed candidate consumption above quota was accepted')
    s.require(meter.counts['input_json_candidates_decoded'] == 2 and
              meter.limits['input_json_candidates_decoded'] == 1,
              'Rejected actual candidate consumption was refunded or the quota was extended')
    cases.append(dict(name='consumed_candidates_survive_quota_rejection', passed=True))

    input_rows = sum(r['counters']['input_json_rows_decoded'] for r in receipts)
    input_candidates = sum(r['counters']['input_json_candidates_decoded'] for r in receipts)
    readback_rows = len(saved)
    readback_candidates = sum(len(row['inputs']['candidates']) for row in saved)
    s.require(input_rows + readback_rows <= 24 and input_candidates + readback_candidates <= 64,
              'Sampler synthetic numeric-processing budget exceeded')
    result = dict(schema='task-stratified-sampling-regressions-v1', passed=True, synthetic_only=True,
        cases=cases, synthetic_rows=18, synthetic_candidates=36, original_source_reads=0,
        model_calls=0, original_model_loads=0,
        synthetic_input_rows_decoded=input_rows, synthetic_input_candidates_decoded=input_candidates,
        synthetic_readback_rows_decoded=readback_rows,
        synthetic_readback_candidates_decoded=readback_candidates,
        synthetic_numeric_processing_groups=input_rows+readback_rows,
        synthetic_numeric_processing_candidates=input_candidates+readback_candidates,
        frozen_model_calls=0, original_streams_read=0,
        counter_only_cases=2, counter_only_cases_parse_no_numeric_inputs=True,
        simulated_counter_outcomes=dict(expired_deadline=deadline_counts, exceeded_quota=dict(meter.counts)),
        fixture_counter_receipts=[dict(game=r['game'], passed=r['passed'], counters=r['counters'])
                                  for r in receipts])
    s.write_new(out / 'result.json', result)
    return result
