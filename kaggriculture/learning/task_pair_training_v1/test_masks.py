import hashlib
import io
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from learning.task_pair_training_v1.masks import KINDS, load_masks


class MaskTests(unittest.TestCase):
    commit = 'a' * 40

    def fixture(self):
        lengths = {'teacher_dev': [3], 'conditional_dev': [2, 4]}
        files, entries = {}, []
        for role, shards in lengths.items():
            for index, length in enumerate(shards):
                for kind in KINDS:
                    name = f'{role}-{index}-{kind}.npy'
                    array = np.zeros(length, dtype=bool)
                    if kind == 'seen_in_train':
                        array[0] = True
                    stream = io.BytesIO()
                    np.save(stream, array, allow_pickle=False)
                    raw = stream.getvalue()
                    files[name] = raw
                    entries.append({'name': name, 'sha256': hashlib.sha256(raw).hexdigest(), 'length': length})
        document = {'schema': 'task-pair-data-local-masks-v1', 'registration_commit': self.commit,
                    'conditional_source_order': ['old', 'new'], 'masks': entries}
        return lengths, files, document

    def load(self, lengths, files, document):
        raw = json.dumps(document).encode()
        return load_masks(raw, lambda name, limit: files[name][:limit + 1],
                          manifest_sha256=hashlib.sha256(raw).hexdigest(),
                          registration_commit=self.commit, role_lengths=lengths)

    def test_round_trip_preserves_order_and_each_mask(self):
        lengths, files, document = self.fixture()
        result = self.load(lengths, files, document)
        self.assertEqual([len(shard['seen_in_train']) for shard in result['conditional_dev']], [2, 4])
        self.assertFalse(result['teacher_dev'][0]['label_conflict'][0])
        self.assertFalse(result['teacher_dev'][0]['seen_in_train'][1])

    def test_rejects_changed_mask_bytes_and_wrong_manifest_identity(self):
        lengths, files, document = self.fixture()
        name = document['masks'][0]['name']
        files[name] += b'changed'
        with self.assertRaises(ValueError):
            self.load(lengths, files, document)
        raw = json.dumps(document).encode()
        with self.assertRaises(ValueError):
            load_masks(raw, lambda name, limit: files[name], manifest_sha256='0' * 64,
                       registration_commit=self.commit, role_lengths=lengths)

    def test_rejects_duplicate_missing_or_wrong_order(self):
        lengths, files, document = self.fixture()
        document['masks'].append(document['masks'][0])
        with self.assertRaises(ValueError):
            self.load(lengths, files, document)
        lengths, files, document = self.fixture()
        document['masks'].pop()
        with self.assertRaises(ValueError):
            self.load(lengths, files, document)
        lengths, files, document = self.fixture()
        document['conditional_source_order'].reverse()
        with self.assertRaises(ValueError):
            self.load(lengths, files, document)

    def test_rejects_conflict_outside_overlap(self):
        lengths, files, document = self.fixture()
        name = 'teacher_dev-0-label_conflict.npy'
        stream = io.BytesIO()
        np.save(stream, np.asarray([False, True, False]), allow_pickle=False)
        files[name] = stream.getvalue()
        next(e for e in document['masks'] if e['name'] == name)['sha256'] = hashlib.sha256(files[name]).hexdigest()
        with self.assertRaises(ValueError):
            self.load(lengths, files, document)

    def test_rejects_conflict_inside_overlap(self):
        lengths, files, document = self.fixture()
        name = 'teacher_dev-0-label_conflict.npy'
        stream = io.BytesIO()
        np.save(stream, np.asarray([True, False, False]), allow_pickle=False)
        files[name] = stream.getvalue()
        next(e for e in document['masks'] if e['name'] == name)['sha256'] = hashlib.sha256(files[name]).hexdigest()
        with self.assertRaises(ValueError):
            self.load(lengths, files, document)

    def test_rejects_ambiguous_json_and_extra_array_data(self):
        lengths, files, document = self.fixture()
        raw = b'{"schema":1,"schema":2}'
        with self.assertRaises(ValueError):
            load_masks(raw, lambda name, limit: files[name], manifest_sha256=hashlib.sha256(raw).hexdigest(),
                       registration_commit=self.commit, role_lengths=lengths)
        name = document['masks'][0]['name']
        files[name] += b'extra'
        document['masks'][0]['sha256'] = hashlib.sha256(files[name]).hexdigest()
        with self.assertRaises(ValueError):
            self.load(lengths, files, document)

    def test_rejects_huge_declared_shape_before_allocation(self):
        lengths, files, document = self.fixture()
        name = document['masks'][0]['name']
        stream = io.BytesIO()
        np.lib.format.write_array_header_1_0(stream, {
            'descr': '|b1', 'fortran_order': False, 'shape': (1_000_000_000,),
        })
        files[name] = stream.getvalue()
        document['masks'][0]['sha256'] = hashlib.sha256(files[name]).hexdigest()
        # A ValueError after np.load attempts an allocation is too late.
        with patch('numpy.load', side_effect=AssertionError('unsafe loading')), \
                patch('numpy.frombuffer', side_effect=AssertionError('payload reached')), \
                patch('numpy.empty', side_effect=AssertionError('allocation reached')):
            with self.assertRaises(ValueError):
                self.load(lengths, files, document)


if __name__ == '__main__':
    unittest.main()
