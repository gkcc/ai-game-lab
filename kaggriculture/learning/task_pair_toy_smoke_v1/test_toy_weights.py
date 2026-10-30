"""Synthetic schema and loader tests; read public teacher source, no artifacts or games."""
import builtins
from copy import deepcopy
import hashlib
import io
import json
import os
import unittest
from unittest.mock import patch

import numpy as np

from decision.api import CROPS, PRODUCTS
from learning.bc_crop import agent as crop_agent
from learning.bc_crop import features as crop_features
from learning.task_bc_v1 import model as task_model
from learning.task_choice_v1 import features as task_features
from learning.task_choice_v1.agent import TaskAgent
from . import toy_weights


def public_synthetic_fixture():
    """Obtain exact public encoder names without a game or checkpoint."""
    farms = [dict(money=1000, tiles=[[None, None], [None, None]], farmer=[0, 0],
                  hands=[], unlocked_quadrants=[0], hires_today=0) for _ in range(2)]
    obs = dict(player=0, step=24, day=1, hour=0, farms=farms,
               private=dict(shed={}, seeds={c: 1 for c in CROPS}, inventories=[{}]),
               market=dict(inventory={p: 10000 for p in PRODUCTS},
                           prices={p: 50 for p in PRODUCTS}, params={}),
               town=dict(unlocked_shops=[]))
    cfg = dict(episodeSteps=720, boardSize=2, turnsPerDay=24, shedCapacity=80,
               townShopSellInterval=24, maxMarketOrdersPerTurn=8, farmHandCostMult=1,
               marketParams={})
    task_inputs, crop_inputs = [], []
    def first(inputs):
        task_inputs.append(deepcopy(inputs))
        return inputs['candidates'][0]['id']
    reference = TaskAgent(first, 'teacher')
    original = reference.controller.__globals__['_crop_select']
    def capture(context):
        x, names = crop_features.encode(context)
        crop_inputs.append((x, names))
        return original(context)
    reference.controller.__globals__['_crop_select'] = capture
    reference(obs, cfg)
    item = task_inputs[0]
    crop_x, crop_names = crop_inputs[0]
    crop_schema = dict(schema_version=crop_features.SCHEMA_VERSION, crops=list(crop_features.CROPS),
                       teacher_sha256=crop_agent.TEACHER_SHA, features=crop_names,
                       schema_sha256=crop_features.schema_hash(crop_names))
    task_schema = dict(schema_version=task_features.SCHEMA_VERSION, ops=list(task_features.OPS),
                       shared_feature_names=item['shared_feature_names'], feature_names=item['feature_names'],
                       schema_sha256=item['schema_sha256'])
    return crop_schema, task_schema, item, crop_x


def load_memory(module, loader, raw, digest):
    """Use the actual loader, replacing only file hash and file opening."""
    if hashlib.sha256(raw).hexdigest() != digest:
        raise ValueError('Synthetic archive digest differs')
    real_load = np.load
    with patch.object(module, 'sha', return_value=digest), \
         patch.object(np, 'load', side_effect=lambda _path, allow_pickle: real_load(
             io.BytesIO(raw), allow_pickle=allow_pickle)):
        return loader('synthetic-in-memory-only', digest)


class ToyWeightsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.crop_schema, cls.task_schema, cls.inputs, cls.crop_x = public_synthetic_fixture()

    def build(self):
        return toy_weights.build(crop_schema=self.crop_schema, task_schema=self.task_schema)

    def test_archives_are_repeatable_distinct_finite_and_do_not_open_files(self):
        with patch.object(builtins, 'open', side_effect=AssertionError('file opened')), \
             patch.object(io, 'open', side_effect=AssertionError('file opened')), \
             patch.object(os, 'open', side_effect=AssertionError('file opened')):
            first, second = self.build(), self.build()
        self.assertEqual(first, second)
        self.assertFalse(first['trained'])
        archive_list = [first['crop'], *first['tasks'].values()]
        self.assertEqual(len({archive['sha256'] for archive in archive_list}), 4)
        for archive in archive_list:
            self.assertEqual(hashlib.sha256(archive['data']).hexdigest(), archive['sha256'])
            with np.load(io.BytesIO(archive['data']), allow_pickle=False) as values:
                self.assertEqual(set(values.files), {'metadata', 'mean', 'scale',
                    'w0', 'b0', 'w1', 'b1', 'w2', 'b2'})
                metadata = json.loads(str(values['metadata']))
                self.assertIs(metadata['trained'], False)
                self.assertEqual(metadata['source'], 'synthetic_toy')
                for name in set(values.files) - {'metadata'}:
                    self.assertEqual(values[name].dtype, np.dtype('float32'))
                    self.assertTrue(np.isfinite(values[name]).all())

    def test_real_loaders_accept_toy_schema_and_same_input_flips_choice(self):
        output = self.build()
        models = {}
        for role, archive in output['tasks'].items():
            models[role] = load_memory(task_model, task_model.TaskNetwork,
                                       archive['data'], archive['sha256'])
            self.assertEqual(models[role].mean.shape, (373,))
            self.assertEqual([models[role].params['w'+str(i)].shape for i in range(3)],
                             [(373, 128), (128, 64), (64, 1)])
        chosen = {role: model(self.inputs) for role, model in models.items()}
        self.assertNotEqual(chosen['baseline'], chosen['control'])
        self.assertEqual(chosen['baseline'], chosen['auxiliary'])
        np.testing.assert_array_equal(models['baseline'].last['logits'],
                                      -np.asarray(models['control'].last['logits']))
        crop = output['crop']
        crop_network = load_memory(crop_agent, crop_agent.Network, crop['data'], crop['sha256'])
        self.assertEqual(crop_network.mean.shape, (289,))
        self.assertEqual([pair[0].shape for pair in crop_network.weights],
                         [(289, 128), (128, 64), (64, 5)])
        self.assertEqual(CROPS[int(np.argmax(crop_network.logits(self.crop_x,
                                                               self.crop_schema['features'])))], 'CARROT')

    def test_bad_lengths_duplicates_or_hashes_are_rejected(self):
        for item in ('crop_length', 'task_length', 'duplicate', 'crop_hash', 'task_hash',
                     'wrong_teacher', 'wrong_order', 'self_consistent_but_wrong_names'):
            crop, task = deepcopy(self.crop_schema), deepcopy(self.task_schema)
            if item == 'crop_length': crop['features'].pop()
            if item == 'task_length': task['feature_names'].pop()
            if item == 'duplicate': task['feature_names'][1] = task['feature_names'][0]
            if item == 'crop_hash': crop['schema_sha256'] = '0'*64
            if item == 'task_hash': task['schema_sha256'] = '0'*64
            if item == 'wrong_teacher': crop['teacher_sha256'] = '0'*64
            if item == 'wrong_order': task['ops'].reverse()
            if item == 'self_consistent_but_wrong_names':
                task['feature_names'][-1] = 'synthetic.wrong'
                task['schema_sha256'] = task_features.schema_hash(task['feature_names'],
                                                                  task['shared_feature_names'])
            with self.subTest(item=item), self.assertRaises(ValueError):
                toy_weights.build(crop_schema=crop, task_schema=task)

    def test_input_extras_are_not_copied_into_archives(self):
        crop, task = deepcopy(self.crop_schema), deepcopy(self.task_schema)
        crop['untrusted_extra'] = 'do-not-copy'
        task['untrusted_extra'] = 'do-not-copy'
        result = toy_weights.build(crop_schema=crop, task_schema=task)
        for archive in [result['crop'], *result['tasks'].values()]:
            with np.load(io.BytesIO(archive['data']), allow_pickle=False) as values:
                self.assertNotIn('do-not-copy', str(values['metadata']))


if __name__ == '__main__':
    unittest.main()
