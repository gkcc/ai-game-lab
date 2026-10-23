"""Small pure numeric regressions; no engine, training, or private holdout reads."""
from copy import deepcopy
import json
from unittest.mock import patch

import numpy as np

from learning.task_bc_v1.model import continuous_mask
from learning.task_choice_v1.features import SCHEMA_VERSION, candidate_id, schema_hash
from .numeric import (_build_and_summarize, RecordAnalyzer, add_training, new_index,
                      numeric_key, summarize)


class FrozenFixture:
    """Deterministic scorer whose close float64 values have observable effects."""
    sha = 'synthetic-fixed-scorer'

    def __init__(self):
        self.metadata = dict(schema_version=SCHEMA_VERSION,
            shared_feature_names=['shared.' + str(i) for i in range(275)],
            feature_names=['task.op.PLANT'] + ['candidate.' + str(i) for i in range(1, 98)])
        self.metadata['schema_sha256'] = schema_hash(self.metadata['feature_names'], self.metadata['shared_feature_names'])
        self.calls = 0

    def score(self, shared, candidates):
        self.calls += 1
        assert shared.dtype == np.float64 and candidates.dtype == np.float64
        return candidates[:, 1] + shared[0]


def fixture(model, first=1., second=1.):
    inputs = deepcopy(model.metadata)
    inputs['shared_x'] = [0.] * 275
    inputs['candidates'] = []
    for target, op, value in (([0, 0], 'PLANT', first), ([1, 0], 'WATER', second)):
        x = [0.] * 98; x[1] = value
        inputs['candidates'].append(dict(id=candidate_id(0, target, op), op=op, unit=0, target=target, x=x))
    inputs['observation'] = {'step': 0}
    return inputs


def record(inputs, label, chosen=None):
    ids = [c['id'] for c in inputs['candidates']]
    return dict(inputs=inputs, teacher_label=ids[label], chosen=ids[label if chosen is None else chosen])


def fails(call):
    try:
        call()
    except (ValueError, OverflowError):
        return
    raise AssertionError('Expected fail-closed rejection')


def late_training_conflict():
    """The first scored row must see labels that appear later in training."""
    model = FrozenFixture(); first = fixture(model)
    other_context = deepcopy(first); other_context['observation']['step'] = 1
    validation = fixture(model)
    validation['shared_x'][1] = 2.
    validation['candidates'][0]['x'][0] = 1.
    rows = {'train': [(first, 0), (first, 1), (other_context, 1)],
            'validation': [(validation, 1)]}
    reads = {'train': 0, 'validation': 0}; order = []; evaluated = []
    index = new_index(model)

    def reader(split):
        reads[split] += 1; order.append(split)
        assert reads[split] <= (2 if split == 'train' else 1)
        if split == 'train' and reads[split] == 1:
            assert model.calls == 0
        else:
            assert index['groups'] == 3
        for inputs, label in rows[split]:
            yield inputs, dict(index=label, teacher_label=inputs['candidates'][label]['id'])

    evaluate = RecordAnalyzer.evaluate

    def capture(analyzer, row, origin):
        assert index['groups'] == 3
        result = evaluate(analyzer, row, origin)
        evaluated.append(result)
        return result

    with patch.object(RecordAnalyzer, 'evaluate', capture):
        summaries = _build_and_summarize(index, model, reader)
    early = evaluated[0]
    assert early['origin'] == 'historical_train'
    assert early['training_label_indices64'] == [0, 1]
    assert early['training_label_conflict64']
    assert early['training_same_context_label_conflict64']
    assert early['training_different_context_label_conflict64']
    assert reads == {'train': 2, 'validation': 1}
    assert order == ['train', 'train', 'validation']
    assert index['groups'] == 3 and index['candidates'] == 6
    assert index['keys64'][early['key64']]['labels'] == {0: 1, 1: 2}
    assert len(index['keys64']) == 1
    assert index['max'][1] == 0. and index['discrete'][275] == {0.}
    assert index['candidate_counts'] == {2: 3}
    assert not evaluated[-1]['train_seen64'] and evaluated[-1]['range_exceeded']
    assert evaluated[-1]['unseen_discrete_features'] == [275]
    assert summaries['train']['raw_weighted']['denominator'] == 3
    assert summaries['validation']['raw_weighted']['denominator'] == 1
    assert model.calls == 2
    return dict(passed=True, synthetic_train_rows=3, synthetic_validation_rows=1,
                source_passes=reads, rows_processed=7, scored_rows=4,
                synthetic_scorer_calls=model.calls,
                first_row_final_training_label_indices64=early['training_label_indices64'],
                first_row_conflicts={k: early[k] for k in (
                    'training_label_conflict64', 'training_same_context_label_conflict64',
                    'training_different_context_label_conflict64')},
                validation_excluded_from_support=True)


def run():
    model = FrozenFixture(); base = fixture(model)
    changed = fixture(model, second=1. + 2. ** -30)
    shared = base['shared_x']; local = [c['x'] for c in base['candidates']]
    changed_local = [c['x'] for c in changed['candidates']]
    assert numeric_key(shared, local, 32) == numeric_key(shared, changed_local, 32)
    assert numeric_key(shared, local, 64) != numeric_key(shared, changed_local, 64)
    assert numeric_key(shared, changed_local, 64) != numeric_key(shared, changed_local[::-1], 64)
    assert numeric_key(shared, local, 64) != numeric_key(shared, local[:1], 64)
    signed = np.asarray(local).copy(); signed[0, 0] = -0.
    assert numeric_key(shared, local, 64) == numeric_key(shared, signed, 64)
    signed_shared = np.asarray(shared).copy(); signed_shared[0] = -0.
    assert numeric_key(shared, local, 32) == numeric_key(signed_shared, signed, 32)
    for invalid in (float('nan'), float('inf'), -float('inf')):
        bad = np.asarray(local).copy(); bad[0, 1] = invalid
        fails(lambda: numeric_key(shared, bad, 64))
    too_large = np.asarray(local).copy(); too_large[0, 1] = 1e100
    fails(lambda: numeric_key(shared, too_large, 32))

    index = new_index(model)
    assert set(index['discrete']) == set(np.flatnonzero(~continuous_mask(
        model.metadata['shared_feature_names'] + model.metadata['feature_names'])))
    add_training(index, base, 0, model)
    analyzer = RecordAnalyzer(index, model)
    same = analyzer.evaluate(record(base, 0), 'trained')
    other = analyzer.evaluate(record(changed, 0, chosen=1), 'trained')
    assert same['pred_index'] == 0 and other['pred_index'] == 1 and model.calls == 2
    assert other['train_seen32'] and not other['train_seen64']
    assert same['alias64'] and other['alias32'] and not other['alias64']
    assert other['float32_only_alias'] and other['candidate_mismatch']
    assert other['range_exceeded'] and other['out_of_range_features'] == [276]
    assert other['cross_op'] and other['op_mismatch'] and other['target_mismatch']
    assert not other['unit_mismatch'] and other['logit_gap'] == 2. ** -30
    later = analyzer.evaluate(record(base, 1), 'teacher_task')
    assert later['teacher_later_alias64'] and later['max_tie_count'] == 2
    assert later['teacher_tied_at_max'] and later['candidate_mismatch'] and model.calls == 2
    fails(lambda: analyzer.evaluate(record(base, 1), 'trained'))
    fails(lambda: analyzer.evaluate(record(base, 0, 1), 'teacher_task'))
    reverse = deepcopy(changed); reverse['candidates'].reverse()
    fails(lambda: analyzer.evaluate(record(reverse, 0), 'teacher_task'))
    bad = deepcopy(base); bad['shared_x'][0] = float('nan')
    fails(lambda: analyzer.evaluate(record(bad, 0), 'teacher_task'))
    bad = record(base, 0); bad['teacher_label'] = 'unknown'
    fails(lambda: analyzer.evaluate(bad, 'teacher_task'))

    add_training(index, base, 1, model)
    assert index['keys64'][same['key64']]['labels'] == {0: 1, 1: 1}
    conflict = analyzer.evaluate(record(base, 0), 'teacher_task')
    assert conflict['training_label_conflict64'] and conflict['training_same_context_label_conflict64']
    changed_context = deepcopy(base); changed_context['observation']['step'] = 1
    assert analyzer.evaluate(record(changed_context, 0), 'teacher_task')['training_different_context_label_conflict64']
    novel = deepcopy(base); novel['shared_x'][1] = 2.
    assert not analyzer.evaluate(record(novel, 0), 'teacher_task')['train_seen32']
    discrete = deepcopy(base); discrete['candidates'][0]['x'][0] = 1.
    d = analyzer.evaluate(record(discrete, 0), 'teacher_task')
    assert d['unseen_discrete_features'] == [275]

    weighted = summarize([same, same, same, other])
    assert weighted['raw_weighted']['denominator'] == 4
    assert weighted['raw_weighted']['rates']['candidate_mismatch'] == .25
    assert weighted['unique_weighted64']['denominator'] == 2
    assert weighted['unique_weighted64']['rates']['candidate_mismatch'] == .5
    assert weighted['unique_weighted32']['denominator'] == 1
    assert weighted['unique_weighted32']['rates']['candidate_mismatch'] == .25
    conflicts = summarize([same, same, later])
    assert conflicts['unique_weighted64']['conflicting_label_groups'] == 1
    assert conflicts['unique_weighted64']['same_structured_context_conflict_groups'] == 1
    assert conflicts['raw_weighted']['confusion']['WATER']['PLANT'] == 1
    assert conflicts['raw_weighted']['class_recall']['DIG'] is None
    empty = summarize([])
    assert empty['raw_weighted']['rates']['candidate_mismatch'] is None
    assert empty['unique_weighted64']['denominator'] == 0
    assert 'inputs' not in other and 'shared_x' not in other and 'logits' not in other
    json.dumps(weighted, allow_nan=False)
    late_conflict = late_training_conflict()
    return dict(passed=True, fixture='synthetic fixed scorer', original_data_rows=0,
                late_training_conflict=late_conflict,
                synthetic_scorer_calls=model.calls + late_conflict['synthetic_scorer_calls'],
                training_runs=0, new_games=0, official_transitions=0,
                checks=['float32/float64 separation', 'float64-only scoring cache',
                        'order and count identity', 'signed zero', 'nonfinite rejection',
                        'factual parity', 'semantic aliases and first argmax ties',
                        'labels separate from inputs', 'training-only support',
                        'range/discrete novelty', 'raw and unique multiplicity',
                        'conflicts and undefined denominators',
                        'late training conflict uses complete train-only support',
                        'train two passes and validation one pass'])


if __name__ == '__main__':
    print(json.dumps(run(), indent=2))
