"""Verified, ragged numeric task inputs; never duplicate shared rows per candidate."""
import argparse
from collections import Counter, defaultdict
from contextlib import ExitStack
import gzip
import hashlib
from itertools import zip_longest
import json
from pathlib import Path
import time

import numpy as np

from learning.bc_crop.agent import ROOT, sha
from learning.bc_crop.experiment import read, save
from learning.task_choice_v1.dataset import canonical, digest
from learning.task_choice_v1.features import OPS, SCHEMA_VERSION, schema_hash, candidate_id

DIMENSIONS = {'shared': 275, 'candidates': 98}


def decode(inputs, label, index, *, game, split, expected_seed, expected_side, expected_opponent, schema=None):
    """Validate one aligned triple and return only numeric model/metric arrays."""
    assert index['game'] == game and index['split'] == split
    assert index['seed'] == expected_seed and index['side'] == expected_side
    assert index['opponent'] == expected_opponent
    assert type(index['step']) is int and 0 <= index['step'] < 719
    assert type(index['decision']) is int and index['decision'] >= 0
    assert index['input_sha256'] == digest(inputs)
    assert inputs['observation']['step'] == index['step']
    assert inputs['observation']['player'] == index['side']
    assert inputs['schema_version'] == SCHEMA_VERSION
    assert inputs['schema_sha256'] == schema_hash(inputs['feature_names'], inputs['shared_feature_names'])
    current_schema = dict(schema_version=SCHEMA_VERSION, schema_sha256=inputs['schema_sha256'],
        shared_feature_names=inputs['shared_feature_names'], feature_names=inputs['feature_names'], ops=list(OPS))
    if schema is not None: assert current_schema == schema
    assert len(inputs['shared_feature_names']) == 275 and len(inputs['feature_names']) == 98
    assert len(set(inputs['shared_feature_names'])) == 275 and len(set(inputs['feature_names'])) == 98
    shared = np.asarray(inputs['shared_x'], dtype='<f4')
    candidates = inputs['candidates']
    assert candidates
    local = np.asarray([c['x'] for c in candidates], dtype='<f4')
    assert shared.shape == (275,) and local.shape == (len(candidates), 98)
    assert np.isfinite(shared).all() and np.isfinite(local).all()
    ids = [c['id'] for c in candidates]
    assert len(ids) == len(set(ids))
    assert type(label['index']) is int and 0 <= label['index'] < len(ids)
    assert label['teacher_label'] == ids[label['index']]
    numeric_ids = []
    ops = []
    for c in candidates:
        unit, target, op = c['unit'], c['target'], c['op']
        assert type(unit) is int and 0 <= unit < 32768
        assert len(target) == 2 and all(type(t) is int and 0 <= t < 32768 for t in target)
        assert op in OPS and c['id'] == candidate_id(unit, target, op)
        numeric_ids.append([unit, *target]); ops.append(OPS.index(op))
    assert [(c['unit'], c['target'][1], c['target'][0], c['op']) for c in candidates] == sorted(
        (c['unit'], c['target'][1], c['target'][0], c['op']) for c in candidates)
    return current_schema, shared, local, label['index'], np.asarray(ops, dtype=np.int8), np.asarray(numeric_ids, dtype=np.int16)


class NumericEquivalence:
    """Optimistic exact-ID bound under equal numeric inputs and first-argmax ties.

    This is an upper bound, not proof a fixed-width MLP can attain the bound:
    shared scores across different candidate sets may impose more constraints.
    """
    def __init__(self):
        self.groups = {}
        self.group_sizes = Counter()
        self.examples = self.alias_selected = self.groups_with_aliases = 0

    def add(self, shared, candidates, label):
        shared = np.asarray(shared, dtype='<f4').copy()
        candidates = np.asarray(candidates, dtype='<f4').copy()
        # IEEE signed zero compares numerically equal and scores identically.
        shared[shared == 0] = 0.; candidates[candidates == 0] = 0.
        count = len(candidates)
        self.examples += 1; self.group_sizes[count] += 1
        h = hashlib.sha256()
        h.update(np.asarray([count], dtype='<i8').tobytes())
        h.update(shared.tobytes()); h.update(candidates.tobytes())
        identity = h.digest()
        first = {}
        aliases = []
        for row in candidates:
            value = row.tobytes()
            aliases.append(first.setdefault(value, len(aliases)))
        allowed = aliases[label] == label
        self.alias_selected += not allowed
        self.groups_with_aliases += len(first) != count
        record = self.groups.setdefault(identity, {'labels': Counter(), 'feasible': set(),
            'size': count, 'has_aliases': len(first) != count})
        record['labels'][label] += 1
        if allowed: record['feasible'].add(label)

    def result(self, *, multicandidate=False):
        records = [r for r in self.groups.values() if not multicandidate or r['size'] > 1]
        sizes = {k:v for k,v in self.group_sizes.items() if not multicandidate or k > 1}
        examples = sum(sizes.values())
        conflicts = [r for r in records if len(r['labels']) > 1]
        upper = sum(max((r['labels'][label] for label in r['feasible']), default=0)
                    for r in records)
        return dict(examples=examples, numeric_distinct_groups=len(records),
            numeric_duplicate_groups=sum(sum(r['labels'].values()) > 1 for r in records),
            duplicate_numeric_groups_with_different_labels=len(conflicts),
            examples_in_conflicting_groups=sum(sum(r['labels'].values()) for r in conflicts),
            groups_with_intragroup_numeric_aliases=sum(sum(r['labels'].values()) for r in records if r['has_aliases']),
            labels_unreachable_due_to_first_argmax_alias=sum(
                frequency for r in records for label,frequency in r['labels'].items() if label not in r['feasible']),
            optimistic_exact_id_correct_upper_bound=upper,
            optimistic_exact_id_accuracy_upper_bound=upper/examples if examples else None,
            group_size_counts={str(k): v for k, v in sorted(sizes.items())},
            min_group_size=min(sizes, default=0), max_group_size=max(sizes, default=0),
            total_candidates=sum(k*v for k,v in sizes.items()),
            population='candidate count > 1 only' if multicandidate else 'all groups including singletons',
            interpretation='Whole ordered numeric shared/candidate arrays; identical float32 encodings merge, signed zero normalized. Labels remain unaltered. Fixed first-argmax cannot choose a later identical candidate. Upper bound ignores additional neural-function constraints and is not an achievable network guarantee.')

    def merge(self, other):
        self.examples += other.examples
        self.alias_selected += other.alias_selected
        self.groups_with_aliases += other.groups_with_aliases
        self.group_sizes.update(other.group_sizes)
        for identity, record in other.groups.items():
            target = self.groups.setdefault(identity, {'labels': Counter(), 'feasible': set(),
                'size': record['size'], 'has_aliases': record['has_aliases']})
            target['labels'].update(record['labels']); target['feasible'].update(record['feasible'])


def _array(folder, name, dtype, shape):
    return np.lib.format.open_memmap(folder/(name+'.npy'), mode='w+', dtype=dtype, shape=shape)


def run(out):
    from .experiment import ready, verify
    out = Path(out).resolve(); cfg = ready(out); tick=time.perf_counter()
    source = ROOT/cfg['task_dataset_root']
    manifest = read(source/'dataset-manifest.json')
    coverage = read(source/'coverage.json')
    source_rows = read(source/'reconstruction-ledger.json')['rows']
    rows_by_game = {r['game']: r for r in source_rows}
    assert len(rows_by_game) == len(source_rows) == len(manifest['files']) == 60
    assert coverage['passed'] and manifest['examples'] == 135406
    assert len({r['game'] for r in manifest['files']}) == 60
    assert {r['game'] for r in manifest['files']} == set(rows_by_game)
    assert not (out/'cache-started.json').exists() and not (out/'cache').exists()
    save(out/'cache-started.json', dict(registration_sha256=sha(out/'registration.json'), attempts=1))
    cache = out/'cache'; cache.mkdir()
    schema = None; reports = {}; array_manifest = {}; source_hashes = {}
    combined = NumericEquivalence(); overlap = None
    for split in ('train', 'validation'):
        split_rows = [r for r in manifest['files'] if r['split'] == split]
        n = sum(r['files']['inputs']['rows'] for r in split_rows)
        m = sum(v['candidate_count'] for v in coverage['splits'][split]['tasks'].values())
        assert n == coverage['splits'][split]['decisions'] and n > 0 and m >= n
        folder = cache/split; folder.mkdir()
        arrays = dict(shared=_array(folder,'shared','<f4',(n,275)),
            candidates=_array(folder,'candidates','<f4',(m,98)),
            offsets=_array(folder,'offsets','<i8',(n+1,)), labels=_array(folder,'labels','<i8',(n,)),
            selected_ops=_array(folder,'selected_ops','i1',(n,)),
            candidate_ops=_array(folder,'candidate_ops','i1',(m,)),
            candidate_identity=_array(folder,'candidate_identity','<i2',(m,3)),
            seeds=_array(folder,'seeds','<i8',(n,)))
        arrays['offsets'][0] = 0
        group = start = 0; equivalent = NumericEquivalence()
        for file_group in split_rows:
            game = file_group['game']; identity = rows_by_game[game]
            assert identity['split'] == split and identity['seed'] in cfg[split+'_seeds']
            paths = {}
            for kind in ('inputs', 'labels', 'index'):
                info = file_group['files'][kind]
                assert info == identity['files'][kind]
                assert info['rows'] == file_group['files']['inputs']['rows']
                path = source/info['path']
                assert path.resolve().is_relative_to((source/'data').resolve())
                assert sha(path) == info['sha256']
                source_hashes[info['path']] = info['sha256']; paths[kind] = path
            count = 0; previous = None
            with ExitStack() as stack:
                streams = [stack.enter_context(gzip.open(paths[k],'rt',encoding='utf-8'))
                           for k in ('inputs','labels','index')]
                for lines in zip_longest(*streams):
                    assert all(line is not None for line in lines), 'Unaligned triplet length'
                    inputs, label, index = [json.loads(line) for line in lines]
                    current = (index['step'], index['decision'])
                    assert previous is None or current > previous
                    if previous is None or current[0] != previous[0]: assert current[1] == 0
                    else: assert current[1] == previous[1] + 1
                    previous = current
                    schema, shared, candidates, selected, ops, numeric_ids = decode(inputs,label,index,
                        game=game, split=split, expected_seed=identity['seed'], expected_side=identity['side'],
                        expected_opponent=identity['opponent'], schema=schema)
                    end = start + len(candidates)
                    assert group < n and end <= m
                    arrays['shared'][group] = shared; arrays['candidates'][start:end] = candidates
                    arrays['offsets'][group+1] = end; arrays['labels'][group] = selected
                    arrays['selected_ops'][group] = ops[selected]; arrays['candidate_ops'][start:end] = ops
                    arrays['candidate_identity'][start:end] = numeric_ids
                    arrays['seeds'][group] = identity['seed']
                    equivalent.add(shared,candidates,selected)
                    group += 1; start = end; count += 1
            assert count == file_group['files']['inputs']['rows']
            print(f'cache {split}: {group}/{n} groups',flush=True)
        assert group == n and start == m
        reports[split] = equivalent.result()
        reports[split]['multicandidate_only'] = equivalent.result(multicandidate=True)
        assert reports[split]['total_candidates'] == m
        if split == 'validation':
            shared_keys = set(combined.groups) & set(equivalent.groups)
            conflicts = [key for key in shared_keys if len(
                set(combined.groups[key]['labels']) | set(equivalent.groups[key]['labels'])) > 1]
            overlap = dict(numeric_groups_shared_between_train_and_validation=len(shared_keys),
                validation_examples_with_train_numeric_match=sum(
                    sum(equivalent.groups[key]['labels'].values()) for key in shared_keys),
                shared_numeric_groups_with_multiple_labels=len(conflicts),
                interpretation='Natural overlap in fixed original splits; reported, never deduplicated or reassigned. Numeric agreement is narrower than complete structured state equivalence.')
        combined.merge(equivalent)
        array_manifest[split] = {}
        for name, array in arrays.items():
            array.flush(); shape, dtype = list(array.shape), str(array.dtype)
            array._mmap.close()
            path = folder/(name+'.npy')
            array_manifest[split][name] = dict(path=path.relative_to(cache).as_posix(),
                sha256=sha(path), shape=shape, dtype=dtype)
        del arrays
    assert schema['schema_sha256'] == manifest['schema_sha256']
    assert len(source_hashes) == 180
    save(cache/'schema.json',schema)
    save(out/'cache-summary.json',dict(passed=True,splits=reports,overall=combined.result(),
        overall_multicandidate_only=combined.result(multicandidate=True),
        train_validation_numeric_overlap=overlap,source_files_verified=180,
        wall_seconds=time.perf_counter()-tick,device='CPU preparation; no training',
        all_groups_retained=True, candidate_truncation=False, candidate_group_splitting=False,
        finite=True, expanded_candidate_by_shared_matrix_created=False,
        metric_only_arrays=['candidate_identity','candidate_ops','selected_ops','seeds'],
        numeric_model_inputs=['shared','candidates'],
        normalization='None here. Train-only statistics and exact binary/presence column map belong to the frozen trainer.'))
    save(cache/'manifest.json',dict(passed=True,schema_sha256=sha(cache/'schema.json'),
        numeric_feature_schema_sha256=schema['schema_sha256'],
        source_dataset_manifest_sha256=sha(source/'dataset-manifest.json'),
        source_files_sha256=source_hashes,arrays=array_manifest,
        max_group_size=max(r['max_group_size'] for r in reports.values()),
        split_summary=reports,summary_sha256=sha(out/'cache-summary.json'),
        privacy='Arrays are local training data, not public evidence. Seed/unit/tile/op metadata are for metrics only; unit/tile/op also have declared legal observation feature columns, never added secretly.'))
    verify(out)
    return reports


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--out',required=True,type=Path)
    run(parser.parse_args().out)
