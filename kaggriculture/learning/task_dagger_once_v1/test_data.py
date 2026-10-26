"""Synthetic regressions for conditional teacher labels and split isolation."""
from copy import deepcopy
import io
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

from learning.task_choice_v1.features import SCHEMA_VERSION, candidate_id, schema_hash
from . import data


def config():
    return dict(train_seeds=[100, 101, 102, 103], validation_seeds=[200, 201],
                test_seeds=list(range(300, 308)), opponents=['round2', 'market'], seats=[0, 1],
                task_checkpoint_sha256='task', crop_checkpoint_sha256='crop',
                data_limits=dict(train_groups=64000, validation_groups=32000,
                                 train_candidates=2560000, validation_candidates=1280000))


def row(split='train', seed=100, opponent='round2', side=0):
    return dict(game=f'{split}-{seed}-{opponent}-{side}-baseline', split=split,
                policy='baseline', seed=seed, opponent=opponent, side=side, passed=True,
                states=720, action_turns=719, status=['DONE', 'DONE'], agent_errors=0,
                task_checkpoint_sha256='task', crop_checkpoint_sha256='crop', trace_sha256='trace')


def record(step=0, side=0):
    shared_names = ['s' + str(i) for i in range(275)]
    names = ['x' + str(i) for i in range(98)]
    candidates = []
    for unit, op in enumerate(('WATER', 'DIG')):
        target = [unit + 1, 1]
        candidates.append(dict(id=candidate_id(unit, target, op), unit=unit, target=target,
                               op=op, x=[float(unit)] * 98))
    inputs = dict(schema_version=SCHEMA_VERSION, feature_names=names, shared_feature_names=shared_names,
                  schema_sha256=schema_hash(names, shared_names), shared_x=[0.] * 275,
                  observation=dict(step=step, player=side), candidates=candidates)
    selected = candidates[1]
    return dict(inputs=inputs, teacher_label=candidates[0]['id'], chosen=selected['id'],
                unit=selected['unit'], target=selected['target'], op=selected['op'], source='callback')


def event(step, side=0):
    return dict(schema_version=SCHEMA_VERSION, step=step, checkpoint_sha256='crop',
                task_checkpoint_sha256='task', load_errors=0, inference_errors=0, fallbacks=0,
                records=[record(step, side)] if step == 0 else [])


def trace_text(side=0, turns=719):
    return ''.join(json.dumps(event(step, side)) + '\n' for step in range(turns))


class DataTests(unittest.TestCase):
    def test_teacher_label_not_student_choice_and_inputs_unchanged(self):
        item = record(); original = deepcopy(item)
        values = data.decode_record(item, row(), 0, 0)
        self.assertEqual(values[3], 0)
        self.assertEqual(item['chosen'], item['inputs']['candidates'][1]['id'])
        np.testing.assert_array_equal(values[2][0], np.zeros(98))
        np.testing.assert_array_equal(values[2][1], np.ones(98))
        self.assertEqual(item, original)

    def test_label_must_belong_to_this_callback(self):
        missing = record(); del missing['teacher_label']
        with self.assertRaises(KeyError):
            data.decode_record(missing, row(), 0, 0)
        item = record(); item['inputs']['candidates'] = item['inputs']['candidates'][1:]
        with self.assertRaisesRegex(ValueError, 'Teacher label'):
            data.decode_record(item, row(), 0, 1)
        item['teacher_label'] = item['chosen']
        self.assertEqual(data.decode_record(item, row(), 0, 1)[3], 0)

    def test_wrong_current_state_or_choice_metadata_rejected(self):
        for key, value in (('step', 1), ('player', 1)):
            with self.subTest(key=key):
                item = record(); item['inputs']['observation'][key] = value
                with self.assertRaises(AssertionError):
                    data.decode_record(item, row(), 0, 0)
        item = record(); item['target'] = [99, 99]
        with self.assertRaisesRegex(ValueError, 'metadata'):
            data.decode_record(item, row(), 0, 0)

    def test_ineligible_sources_rejected_before_file_access(self):
        bad = [dict(row(), split='holdout', seed=300), dict(row(), policy='candidate'),
               dict(row(), policy='teacher_task'), dict(row(), seed=200)]
        with patch.object(data, 'read', side_effect=AssertionError('Should not read a file')):
            for item in bad:
                with self.subTest(item=item):
                    with self.assertRaises(ValueError):
                        data.source_identity(Path('synthetic'), item, config())
        cfg = config(); cfg['test_seeds'][0] = 100
        with self.assertRaisesRegex(ValueError, 'Overlapping'):
            data.validate_role(row(), cfg)

    def test_complete_source_hashes_are_required(self):
        item = row(); game = item['game']
        stored = {game + '.row.json': item,
                  game + '.audit.json': dict(game=game, passed=True, trace_sha256='trace'),
                  game + '.complete.json': dict(game=game, passed=True, row_sha256='r', certificate_sha256='a')}
        def sha(path):
            return 'r' if path.name.endswith('.row.json') else 'a' if path.name.endswith('.audit.json') else 'trace'
        with patch.object(data, 'read', side_effect=lambda p: stored[p.name]), patch.object(data, 'sha', side_effect=sha):
            self.assertEqual(data.source_identity(Path('synthetic'), item, config())[0], 'train')
            stored[game + '.complete.json']['certificate_sha256'] = 'different'
            with self.assertRaisesRegex(ValueError, 'completion'):
                data.source_identity(Path('synthetic'), item, config())

    def test_full_trace_summary_and_turn_truncation(self):
        item = row()
        with patch.object(data, 'source_identity', return_value=('train', Path('synthetic'))), \
             patch.object(data, 'sha', return_value='trace'):
            with patch.object(data.gzip, 'open', return_value=io.StringIO(trace_text())):
                summary = data.summarize_game(Path('synthetic'), item, config())
                self.assertEqual((summary['groups'], summary['candidate_rows'], summary['teacher_ID_disagreements']), (1, 2, 1))
                self.assertTrue(summary['label_binding_passed'])
                self.assertEqual(summary['perop']['WATER']['groups'], 1)
                self.assertEqual(summary['perop']['DIG']['groups'], 0)
            for turns in (718, 720):
                with self.subTest(turns=turns), patch.object(data.gzip, 'open', return_value=io.StringIO(trace_text(turns=turns))):
                    with self.assertRaisesRegex(ValueError, 'trace turn'):
                        data.summarize_game(Path('synthetic'), item, config())

    def test_no_group_is_truncated_to_fit_budget(self):
        cfg = config(); cfg['data_limits']['train_candidates'] = 1
        with patch.object(data, 'source_identity', return_value=('train', Path('synthetic'))), \
             patch.object(data.gzip, 'open', return_value=io.StringIO(trace_text())), \
             patch.object(data, 'sha', return_value='trace'):
            with self.assertRaisesRegex(ValueError, 'data cap'):
                data.summarize_game(Path('synthetic'), row(), cfg)

    def test_complete_population_has_stable_partition_order(self):
        cfg = config()
        rows = [row(split, seed, opponent, side)
                for split, key in (('train', 'train_seeds'), ('dev', 'validation_seeds'))
                for seed in cfg[key] for opponent in cfg['opponents'] for side in cfg['seats']]
        population = data.ordered_population(list(reversed(rows)), cfg)
        self.assertEqual([r['seed'] for r in population['train']], [s for s in cfg['train_seeds'] for _ in range(4)])
        self.assertEqual({r['seed'] for r in population['validation']}, set(cfg['validation_seeds']))
        with self.assertRaisesRegex(ValueError, 'Missing'):
            data.ordered_population(rows[:-1], cfg)
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            data.ordered_population(rows + [rows[0]], cfg)


if __name__ == '__main__':
    unittest.main()
