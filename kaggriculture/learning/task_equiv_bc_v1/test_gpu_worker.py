"""In-memory archive tests; no real model, accelerator or filesystem writes."""
import io
import unittest
import zipfile

import numpy as np

from learning.task_equiv_bc_v1 import gpu_worker


def sample_archive():
    count, width = 48, 3
    x = np.zeros((count * width, 373), dtype=np.float32)
    offsets = np.arange(0, (count + 1) * width, width, dtype=np.int32)
    labels = np.zeros(count, dtype=np.int32)
    roles = np.repeat(np.array([0, 1], dtype=np.int32), 24)
    positive, verified = np.zeros(count * width, dtype=bool), np.zeros(count * width, dtype=bool)
    for i in range(count):
        positive[i * width] = verified[i * width] = True
        if i in (0, 1, 24, 25):
            positive[i * width + 1] = True
            verified[i * width + 1:i * width + 3] = True
    stream = io.BytesIO()
    np.savez_compressed(stream, features=x, offsets=offsets, teacher=labels,
                        positive=positive, verified=verified, roles=roles)
    return stream.getvalue()


def replace_member(raw, name, content):
    output = io.BytesIO()
    with zipfile.ZipFile(io.BytesIO(raw)) as source, zipfile.ZipFile(output, 'w', zipfile.ZIP_DEFLATED) as target:
        for item in source.infolist():
            target.writestr(item.filename, content if item.filename == name else source.read(item))
    return output.getvalue()


class ArchiveTests(unittest.TestCase):
    def test_device_identity_is_private_and_mandatory(self):
        contract = {'schema': 'task-equiv-bc-gpu-contract-v1', 'registration_commit': 'a' * 40,
                    'groups_sha256': 'b' * 64, 'checkpoint_sha256': 'c' * 64,
                    'schema_sha256': 'd' * 64, 'device_kind': 'synthetic-device'}
        gpu_worker.validate_contract(contract, 'a' * 40)
        for value in ('', ' space ', 'bad\nidentity', 123):
            with self.subTest(value=value), self.assertRaises(ValueError):
                gpu_worker.validate_contract(dict(contract, device_kind=value), 'a' * 40)

    def test_valid_private_batch_and_seed_coverage(self):
        groups, counts = gpu_worker.unpack_groups(sample_archive())
        self.assertEqual(len(groups), 48)
        self.assertEqual(counts, [2, 2])
        self.assertEqual(groups[0]['features'].shape, (3, 373))

    def test_oversized_npy_shape_rejected_before_loading(self):
        header = io.BytesIO()
        np.lib.format.write_array_header_1_0(header, {'descr': '<f4', 'fortran_order': False,
                                                     'shape': (48 * 256 + 1, 373)})
        with self.assertRaises(ValueError):
            gpu_worker.unpack_groups(replace_member(sample_archive(), 'features.npy', header.getvalue()))

    def test_oversized_decompressed_member_rejected(self):
        raw = replace_member(sample_archive(), 'features.npy', bytes(24 * 1024**2 + 1))
        with self.assertRaises(ValueError):
            gpu_worker.unpack_groups(raw)

    def test_unexpected_or_duplicate_members_rejected(self):
        raw, output = sample_archive(), io.BytesIO()
        with zipfile.ZipFile(io.BytesIO(raw)) as source, zipfile.ZipFile(output, 'w') as target:
            for item in source.infolist():
                target.writestr(item.filename, source.read(item))
            target.writestr('unknown.npy', b'x')
        with self.assertRaises(ValueError):
            gpu_worker.unpack_groups(output.getvalue())


if __name__ == '__main__':
    unittest.main()
