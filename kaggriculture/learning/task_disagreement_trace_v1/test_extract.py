"""Synthetic conversion tests; no real trace, model, seed, or game."""
from copy import deepcopy
from dataclasses import FrozenInstanceError
import unittest

import numpy as np

from learning.task_choice_v1.features import candidate_id, schema_hash
from learning.task_multisource_rank_v1 import dedup
from learning.task_water_residual_v1.test_policy import metadata_and_inputs
from . import extract


def fixture():
    metadata, inputs = metadata_and_inputs()
    candidates = inputs['candidates']
    assert len(candidates) >= 2
    return dict(role='A03', step=inputs['observation']['step'], ordinal=0,
                inputs=inputs, teacher_label=candidates[0]['id'], chosen=candidates[1]['id'],
                metadata=metadata, guard=lambda: None)


class ExtractionTests(unittest.TestCase):
    def test_preserves_candidate_order_teacher_target_and_private_coordinates(self):
        args = fixture()
        args['inputs']['candidates'].reverse()
        candidates = args['inputs']['candidates']
        result = extract.extract_example(**args)
        self.assertIs(type(result.item), dedup.Item)
        self.assertEqual((result.role, result.item.pool, result.step, result.ordinal),
                         ('A03', 'new_train', args['step'], 0))
        self.assertEqual(result.item.target, [c['id'] for c in candidates].index(args['teacher_label']))
        self.assertEqual(result.student_index, [c['id'] for c in candidates].index(args['chosen']))
        self.assertNotEqual(result.item.target, result.student_index)
        self.assertEqual(result.item.shared.shape, (275,))
        self.assertEqual(result.item.candidates.shape, (len(candidates), 98))
        self.assertEqual(result.item.shared.dtype, np.dtype('float32'))
        self.assertEqual(result.item.candidates.dtype, np.dtype('float32'))
        np.testing.assert_array_equal(result.item.shared, np.asarray(args['inputs']['shared_x'], np.float32))
        for row, candidate in zip(result.item.candidates, candidates):
            np.testing.assert_array_equal(row, np.asarray(candidate['x'], np.float32))
        kept, report = dedup.deduplicate([result.item], mean=np.zeros(373, np.float64),
                                         scale=np.ones(373, np.float64), guard=lambda: None)
        self.assertEqual(kept, [result.item])
        self.assertEqual(report['retained_by_pool']['new_train'], 1)

    def test_role_split_disagreement_and_diagnostic_agreement(self):
        for role in ('A03', 'A04'):
            args = fixture(); args['role'] = role
            self.assertEqual(extract.extract_example(**args).item.pool, 'new_train')
            args['chosen'] = args['teacher_label']
            with self.assertRaises(extract.ExtractionError): extract.extract_example(**args)
        for role in ('A09', 'A10'):
            args = fixture(); args['role'] = role; args['chosen'] = args['teacher_label']
            result = extract.extract_example(**args)
            self.assertEqual(result.item.pool, 'new_dev')
            self.assertEqual(result.item.target, result.student_index)

    def test_arrays_and_container_do_not_retain_input_or_identifiers(self):
        args = fixture()
        result = extract.extract_example(**args)
        shared, local = result.item.shared.copy(), result.item.candidates.copy()
        for array in (result.item.shared, result.item.candidates):
            self.assertFalse(array.flags.writeable)
            with self.assertRaises(ValueError): array.setflags(write=True)
        with self.assertRaises(FrozenInstanceError): result.student_index = 0
        args['inputs']['shared_x'][0] += 12
        args['inputs']['candidates'][0]['x'][-1] += 12
        np.testing.assert_array_equal(result.item.shared, shared)
        np.testing.assert_array_equal(result.item.candidates, local)
        self.assertEqual(repr(result), '<PrivateDisagreementExample>')
        self.assertNotIn(args['teacher_label'], repr(result))
        self.assertNotIn(args['chosen'], repr(result))
        self.assertEqual(set(result.__slots__), {'item', 'step', 'ordinal', 'student_index'})
        self.assertEqual(set(result.item.__slots__), {'shared', 'candidates', 'target', 'pool', 'role'})

    def test_candidate_identity_onehot_and_order_collisions_fail(self):
        variants = []
        args = fixture(); args['inputs']['candidates'][0]['x'][args['metadata']['feature_names'].index('task.op.WATER')] = 0.5; variants.append(args)
        args = fixture(); args['inputs']['candidates'][0]['id'] = 'PRIVATE_IDENTIFIER'; variants.append(args)
        args = fixture(); args['inputs']['candidates'].append(deepcopy(args['inputs']['candidates'][0])); variants.append(args)
        args = fixture(); args['inputs']['candidates'][0]['x'][args['metadata']['feature_names'].index('unit.index')] = -1; variants.append(args)
        args = fixture(); args['inputs']['candidates'][0]['x'][args['metadata']['feature_names'].index('task.distance')] += 1; variants.append(args)
        args = fixture(); args['teacher_label'] = 'PRIVATE_IDENTIFIER'; variants.append(args)
        args = fixture(); args['chosen'] = 'PRIVATE_IDENTIFIER'; variants.append(args)
        for args in variants:
            with self.subTest(case=variants.index(args)):
                with self.assertRaises(extract.ExtractionError) as caught: extract.extract_example(**args)
                self.assertNotIn('PRIVATE_IDENTIFIER', str(caught.exception))

    def test_nonfinite_numeric_and_coordinate_bounds_fail(self):
        variants = []
        args = fixture(); args['inputs']['shared_x'][0] = float('nan'); variants.append(args)
        args = fixture(); args['inputs']['shared_x'][0] = True; variants.append(args)
        args = fixture(); args['inputs']['candidates'][0]['x'][0] = '1'; variants.append(args)
        args = fixture(); args['inputs']['candidates'][0]['x'][-1] = 1e300; variants.append(args)
        args = fixture(); args['step'] = 719; variants.append(args)
        args = fixture(); args['ordinal'] = 32; variants.append(args)
        args = fixture(); args['ordinal'] = True; variants.append(args)
        args = fixture(); args['role'] = 'A05'; variants.append(args)
        args = fixture(); args['inputs']['candidates'] *= 129; variants.append(args)
        args = fixture()
        candidate = args['inputs']['candidates'][0]
        candidate['target'][0] = args['inputs']['configuration']['boardSize']
        candidate['id'] = candidate_id(candidate['unit'], candidate['target'], candidate['op'])
        candidate['x'][args['metadata']['feature_names'].index('target.x')] = candidate['target'][0]
        args['teacher_label'] = candidate['id']
        variants.append(args)
        for args in variants:
            with self.subTest(case=variants.index(args)), self.assertRaises(extract.ExtractionError):
                extract.extract_example(**args)

    def test_recorded_numeric_features_must_round_trip_without_precision_loss(self):
        # The real encoder serializes float32 features; a valid float64 that
        # differs by one ULP must not be silently rounded into that feature.
        for location in ('shared', 'candidate'):
            args = fixture()
            if location == 'shared':
                vector, index = args['inputs']['shared_x'], 0
            else:
                vector, index = args['inputs']['candidates'][0]['x'], -1
            original = vector[index]
            changed = float(np.nextafter(np.float64(original), np.float64('inf')))
            self.assertNotEqual(changed, original)
            self.assertEqual(np.float32(changed), np.float32(original))
            vector[index] = changed
            with self.subTest(location=location), self.assertRaises(extract.ExtractionError):
                extract.extract_example(**args)

    def test_schema_hash_names_and_forbidden_features_fail_closed(self):
        variants = []
        args = fixture(); args['metadata']['schema_sha256'] = 'PRIVATE_SCHEMA'; variants.append(args)
        args = fixture(); args['inputs']['schema_sha256'] = 'PRIVATE_SCHEMA'; variants.append(args)
        args = fixture(); args['inputs']['feature_names'][0] = 'PRIVATE_SCHEMA'; variants.append(args)
        args = fixture(); args['metadata']['dimensions'] = [1]; variants.append(args)
        for forbidden in ('opponent.private.cash', 'info.seed', 'teacher.label'):
            args = fixture()
            args['metadata']['shared_feature_names'][0] = forbidden
            args['inputs']['shared_feature_names'][0] = forbidden
            digest = schema_hash(args['metadata']['feature_names'], args['metadata']['shared_feature_names'])
            args['metadata']['schema_sha256'] = args['inputs']['schema_sha256'] = digest
            variants.append(args)
        for args in variants:
            with self.subTest(case=variants.index(args)):
                with self.assertRaises(extract.ExtractionError) as caught: extract.extract_example(**args)
                self.assertNotIn('PRIVATE_SCHEMA', str(caught.exception))

    def test_extra_private_state_rng_and_model_fields_do_not_enter_output(self):
        variants = []
        args = fixture(); obs = args['inputs']['observation']; obs['farms'][1 - obs['player']]['private'] = {'SECRET': 1}; variants.append(args)
        args = fixture(); args['inputs']['observation']['info'] = {'seed': 'SECRET'}; variants.append(args)
        args = fixture(); args['inputs']['model_logits'] = ['SECRET']; variants.append(args)
        args = fixture(); args['inputs']['candidates'][0]['teacher_label'] = 'SECRET'; variants.append(args)
        for args in variants:
            with self.subTest(case=variants.index(args)):
                with self.assertRaises(extract.ExtractionError) as caught: extract.extract_example(**args)
                self.assertNotIn('SECRET', str(caught.exception))

    def test_guard_failure_propagates_without_result(self):
        args = fixture()
        signal = MemoryError('synthetic guard')
        def fail(): raise signal
        args['guard'] = fail
        with self.assertRaises(MemoryError) as caught: extract.extract_example(**args)
        self.assertIs(caught.exception, signal)
        calls = []
        def fail_midway():
            calls.append(None)
            if len(calls) == 3: raise signal
        args['guard'] = fail_midway
        with self.assertRaises(MemoryError) as caught: extract.extract_example(**args)
        self.assertIs(caught.exception, signal)
        args['guard'] = None
        with self.assertRaises(extract.ExtractionError): extract.extract_example(**args)


if __name__ == '__main__':
    unittest.main()
