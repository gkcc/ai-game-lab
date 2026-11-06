"""Pure inventory/model-identity fixtures; no real artifact is opened."""
from copy import deepcopy
import unittest
from unittest.mock import patch

from learning.task_harvest_return_pairs_v1 import contract, sources


def witness():
    # This is a structural toy witness, not a cryptographic archive proof.
    # archive() separately checks the salted commitment and actual file bytes.
    entries = [dict(root=root, path=f'records/{index:03d}.bin', bytes=0,
                    sha256=f'{index + 1:064x}')
               for root, count in (('C01', 52), ('C02', 52), ('private', 41))
               for index in range(count)]
    amount, remainder = divmod(318604494, 145)
    for index, entry in enumerate(entries):
        entry['bytes'] = amount + (1 if index < remainder else 0)
    return dict(schema='task-harvest-model-dev-publication-witness-v1',
                registration_commit=contract.PRIOR_REG_COMMIT, domain='task-harvest-model-dev-v1',
                files=entries, bytes_total=318604494, commitment_sha256=contract.PRIOR_COMMITMENT)


class SourceTests(unittest.TestCase):
    def test_exact_inventory_has_145_files_and_318604494_bytes_without_io(self):
        value = witness()
        original = deepcopy(value)
        with patch('pathlib.Path.read_bytes', side_effect=AssertionError('Pure helper attempted a read')), \
             patch('pathlib.Path.open', side_effect=AssertionError('Pure helper attempted an open')):
            checked = sources.inventory(value)
        self.assertEqual(len(checked), 145)
        self.assertEqual(sum(entry['bytes'] for entry in checked.values()), 318604494)
        self.assertEqual(list(checked), sorted(checked))
        self.assertEqual({root: sum(key[0] == root for key in checked) for root in sources.ROOTS},
                         {'private': 41, 'C01': 52, 'C02': 52})
        self.assertEqual(value, original)

    def test_file_and_byte_counts_types_and_identity_are_exact(self):
        for defect in ('file_count', 'total', 'entry_total', 'float_total', 'bool_bytes',
                       'huge_entry', 'root_count', 'commitment', 'registration'):
            with self.subTest(defect=defect):
                value = witness()
                if defect == 'file_count':
                    value['files'].pop()
                elif defect == 'total':
                    value['bytes_total'] += 1
                elif defect == 'entry_total':
                    value['files'][0]['bytes'] += 1
                elif defect == 'float_total':
                    value['bytes_total'] = float(value['bytes_total'])
                elif defect == 'bool_bytes':
                    value['files'][0]['bytes'] = True
                elif defect == 'huge_entry':
                    value['files'][0]['bytes'] = 128 * 1024**2 + 1
                elif defect == 'root_count':
                    value['files'][0]['root'], value['files'][0]['path'] = 'private', 'records/extra.bin'
                    value['files'].sort(key=lambda row: (row['root'], row['path']))
                elif defect == 'commitment':
                    value['commitment_sha256'] = '0' * 64
                else:
                    value['registration_commit'] = '0' * 40
                with self.assertRaises(ValueError):
                    sources.inventory(value)

    def test_inventory_rejects_unsafe_paths_duplicate_keys_and_wrong_order(self):
        for path in ('../escape.bin', '/absolute.bin', 'C' + ':/outside.bin', 'records\\bad.bin',
                     'records//bad.bin', 'records/./bad.bin', 'records/../bad.bin', 'records/\nbad.bin'):
            with self.subTest(path=repr(path)):
                value = witness()
                value['files'][0]['path'] = path
                with self.assertRaises(ValueError):
                    sources.inventory(value)
        for defect in ('duplicate', 'order', 'root'):
            with self.subTest(defect=defect):
                value = witness()
                if defect == 'duplicate':
                    value['files'][1]['path'] = value['files'][0]['path']
                elif defect == 'order':
                    value['files'][0], value['files'][1] = value['files'][1], value['files'][0]
                else:
                    value['files'][0]['root'] = 'unregistered'
                with self.assertRaises(ValueError):
                    sources.inventory(value)

    def test_unique_model_uses_sha_not_model_filename_or_map_order(self):
        expected, other = 'a' * 64, 'b' * 64
        odd_name = 'artifacts/odd-public-fixture/not-an-epoch-or-checkpoint-label.npz'
        assets = {'artifacts/likely/model.npz': other, odd_name: expected,
                  'artifacts/also-matching/notes.txt': expected}
        with patch('pathlib.Path.read_bytes', side_effect=AssertionError('Pure helper read weights')):
            self.assertEqual(sources.unique_model(assets, expected), [odd_name, expected])
            self.assertEqual(sources.unique_model(dict(reversed(list(assets.items()))), expected), [odd_name, expected])

    def test_unique_model_rejects_absent_ambiguous_or_invalid_matching_binding(self):
        expected = 'a' * 64
        for assets in (
            {},
            {'artifacts/source.txt': expected},
            {'artifacts/first.npz': expected, 'artifacts/second.npz': expected},
            {'../escape.npz': expected},
            {'C' + ':/outside.npz': expected},
            {'/absolute.npz': expected},
        ):
            with self.subTest(paths=list(assets)):
                with self.assertRaises(ValueError):
                    sources.unique_model(assets, expected)


if __name__ == '__main__':
    unittest.main()
