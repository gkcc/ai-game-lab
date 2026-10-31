"""In-memory acceptance checks for transfer selection and bounded archives."""
import io
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import zipfile

import numpy as np

from learning.task_teacher_support_probe_v1.test_alignment import make_stream
from . import dataset, experiment as e
from .test_dataset import matrix


class MemoryFile(io.BytesIO):
    def __exit__(self, *unused):
        return False


def turn(step, hour, *, total=0, supported=0, retained=0):
    return dict(step=step, hour=hour, teacher_request_matched=True, terminal_request_equal=True,
        selected_ordinals=list(dataset.rank_ordinals('A01', 'market', 0, step, total)), retained_groups=retained,
        teacher_callbacks=total, terminal_callbacks=total, teacher_empty_turn=int(total == 0),
        terminal_empty_turn=int(total == 0), supported=supported, unsupported=0, unknown=total-supported,
        ambiguous=0, supported_target_features_equal=supported, supported_target_features_differ=0)


def case(raw, rows):
    return dict(schema='teacher-transfer-dataset-private-case-v1', registration_commit='a'*40,
        role='A01', opponent='market', learner_side=0, teacher_side=1, replay_states=720,
        sampled_turns=6, rule_calls=12, rows=rows, data_sha256=e.previous._sha(raw),
        official_transitions=0, training=False, passed=True)


class CaseTests(unittest.TestCase):
    def save(self, groups):
        output = MemoryFile()
        with patch.object(Path, 'open', return_value=output):
            digest = e._save_case(Path('synthetic.npz'), groups, lambda: None)
        raw = output.getvalue()
        self.assertEqual(digest, e.previous._sha(raw))
        return raw

    def test_nonempty_and_empty_case_round_trip(self):
        selected = dataset.rank_ordinals('A01', 'market', 0, e.STEPS[0], 5)
        group = SimpleNamespace(candidate_count=3, features=matrix(1), label=0, operation='WATER',
                                teacher_ordinal=selected[0], terminal_ordinal=selected[0])
        raw = self.save([(e.STEPS[0], group)])
        rows = [turn(step, hour, total=5, supported=1, retained=1) if i == 0 else turn(step, hour)
                for i, (step, hour) in enumerate(zip(e.STEPS, e.HOURS))]
        doc = case(raw, rows)
        validated = e._case_result(e.previous._encoded(doc), head='a'*40, role='A01', opponent='market', side=0)
        with patch.object(e.previous, '_read', return_value=raw):
            groups = e._load_case(Path('synthetic.npz'), validated, lambda: None)
        self.assertEqual(len(groups), 1)
        self.assertEqual((groups[0].teacher_ordinal, groups[0].terminal_ordinal, groups[0].target_index),
                         (selected[0], selected[0], 0))
        np.testing.assert_array_equal(groups[0].features, group.features)
        self.assertFalse(groups[0].features.flags.writeable)
        with self.assertRaises(ValueError): groups[0].features.setflags(write=True)

        empty = self.save([])
        empty_case = case(empty, [turn(step, hour) for step, hour in zip(e.STEPS, e.HOURS)])
        with patch.object(e.previous, '_read', return_value=empty):
            self.assertEqual(e._load_case(Path('synthetic.npz'), empty_case, lambda: None), [])

    def test_invalid_npy_headers_are_rejected_before_numpy_load(self):
        original = self.save([])
        with zipfile.ZipFile(io.BytesIO(original)) as archive:
            entries = {item.filename: archive.read(item) for item in archive.infolist()}
        for shape, dtype, fortran in (((10**12, 373), '<f4', False), ((0, 373), '<f8', False),
                                      ((0, 373), '<f4', True), ((1, 373), '<f4', False)):
            header = io.BytesIO()
            np.lib.format.write_array_header_1_0(header, dict(descr=dtype, fortran_order=fortran, shape=shape))
            output = io.BytesIO()
            with zipfile.ZipFile(output, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
                for name, value in entries.items():
                    archive.writestr(name, header.getvalue() if name == 'features.npy' else value)
            raw = output.getvalue()
            doc = case(raw, [turn(step, hour) for step, hour in zip(e.STEPS, e.HOURS)])
            with self.subTest(shape=shape, dtype=dtype, fortran=fortran), \
                    patch.object(e.previous, '_read', return_value=raw), \
                    patch.object(np, 'load', side_effect=AssertionError('allocation must not start')):
                with self.assertRaises(ValueError): e._load_case(Path('synthetic.npz'), doc, lambda: None)

    def test_invalid_final_split_header_is_rejected_before_numpy_load(self):
        output = io.BytesIO()
        np.savez_compressed(output, **e._split_arrays([], lambda: None))
        with zipfile.ZipFile(io.BytesIO(output.getvalue())) as archive:
            entries = {item.filename: archive.read(item) for item in archive.infolist()}
        header = io.BytesIO()
        np.lib.format.write_array_header_1_0(header, dict(descr='<f4', fortran_order=False,
                                                         shape=(10**12, 373)))
        altered = io.BytesIO()
        with zipfile.ZipFile(altered, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
            for name, value in entries.items():
                archive.writestr(name, header.getvalue() if name == 'features.npy' else value)
        raw = altered.getvalue()
        with patch.object(Path, 'open', return_value=io.BytesIO(raw)), \
                patch.object(np, 'load', side_effect=AssertionError('allocation must not start')):
            with self.assertRaises(ValueError):
                e._verify_split(Path('synthetic.npz'), [], e.previous._sha(raw), lambda: None)

    def test_selected_ordinal_and_hour_require_exact_integers(self):
        row = turn(e.STEPS[0], e.HOURS[0], total=1, supported=1, retained=1)
        e._validate_turn(row, role='A01', opponent='market', side=0, step=e.STEPS[0], hour=e.HOURS[0])
        for altered in (dict(row, step=float(e.STEPS[0])), dict(row, hour=False),
                        dict(row, selected_ordinals=[False])):
            with self.subTest(altered=altered), self.assertRaises(ValueError):
                e._validate_turn(altered, role='A01', opponent='market', side=0,
                                 step=e.STEPS[0], hour=e.HOURS[0])

    def test_model_schema_pins_operation_columns(self):
        _obs, _cfg, _records, schema = make_stream()
        self.assertEqual(e._model_schema(schema)['schema_sha256'], schema['schema_sha256'])
        altered = dict(schema, feature_names=list(reversed(schema['feature_names'])))
        with self.assertRaises(ValueError): e._model_schema(altered)

    def test_historical_registration_binds_private_source_identity(self):
        binding = e.previous._encoded({'synthetic': True})
        registration = e.previous._encoded({**e._registration(),
                                           'binding_commitment_sha256': e.previous._sha(binding)})
        def git(command, **kwargs):
            return registration if 'show' in command else (e.BASE_COMMIT + '\n').encode('ascii')
        with patch.object(e.subprocess, 'check_output', side_effect=git), \
                patch.object(e.previous, '_read', return_value=binding):
            self.assertEqual(e._registered_binding('a'*40), e.previous._sha(binding))
        with patch.object(e.subprocess, 'check_output', side_effect=git), \
                patch.object(e.previous, '_read', return_value=binding + b' '):
            with self.assertRaises(ValueError): e._registered_binding('a'*40)


if __name__ == '__main__':
    unittest.main()
