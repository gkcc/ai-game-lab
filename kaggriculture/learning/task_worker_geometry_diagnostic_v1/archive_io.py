"""Authenticate frozen training snapshots and decode only five allowed arrays."""
import hashlib
import io
import json
import math
from pathlib import Path
import struct
import zipfile

import numpy as np

REGISTRATION_SHA256 = '9f95b190268158cb99f3f1a09c0dd0c58a24be8b1bfac7ae30c9ebe3c3ab55a5'
HEADER_LIMIT = 4096
OPERATIONS = ('DIG', 'FERTILIZE', 'HARVEST', 'PLANT', 'WATER')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def snapshot(path, maximum):
    with Path(path).open('rb') as stream:
        raw = stream.read(maximum + 1)
    require(len(raw) <= maximum, 'Snapshot exceeds registered size')
    return raw


def registered_inputs(repo, root):
    registration = repo / 'review/evidence/task-teacher-merge-bc-v1-20261028/registration.json'
    raw = snapshot(registration, 4 * 1024**2)
    require(hashlib.sha256(raw).hexdigest() == REGISTRATION_SHA256, 'Training registration differs')
    cfg = json.loads(raw.decode('utf-8-sig'))
    artifacts = (root / 'artifacts').resolve(strict=True)
    schemas, paths = [], {}
    for name in ('old', 'new'):
        schema_path = (root / cfg['schema_path'][name]).resolve(strict=True)
        archive_path = (root / cfg['archive_path'][name]).resolve(strict=True)
        require(schema_path.is_relative_to(artifacts) and archive_path.is_relative_to(artifacts),
                'Registered source outside artifacts')
        schema = snapshot(schema_path, 4 * 1024**2)
        require(hashlib.sha256(schema).hexdigest() == cfg['schema_sha256'], 'Schema bytes differ')
        schemas.append(schema)
        paths[name] = archive_path
    require(schemas[0] == schemas[1], 'Source schemas differ')
    schema = json.loads(schemas[0].decode('utf-8-sig'))
    require(schema['ops'] == list(OPERATIONS) and len(schema['shared_feature_names']) == 275
            and len(schema['feature_names']) == 98, 'Schema dimensions differ')
    return cfg, schema, paths


def decode_arrays(raw, expected_sha, groups, candidates, max_uncompressed):
    require(hashlib.sha256(raw).hexdigest() == expected_sha, 'Training archive identity differs')
    expected = {'shared': ((groups, 275), np.dtype('float32')),
                'candidates': ((candidates, 98), np.dtype('float32')),
                'offsets': ((groups + 1,), np.dtype('int64')),
                'teacher_choice_label': ((groups,), np.dtype('int64')),
                'ops': ((candidates,), np.dtype('uint8'))}
    source = io.BytesIO(raw)
    with zipfile.ZipFile(source) as archive:
        members = archive.infolist()
        require(sum(item.file_size for item in members) <= max_uncompressed, 'Archive decoded budget exceeded')
        for name, (expected_shape, expected_dtype) in expected.items():
            matches = [item for item in members if item.filename == name + '.npy']
            require(len(matches) == 1, 'Missing or duplicate array member')
            item = matches[0]
            require(not item.flag_bits & 1 and item.compress_type == zipfile.ZIP_STORED,
                    'Unsupported array encoding')
            with archive.open(item) as member:
                version = np.lib.format.read_magic(member)
                require(version in ((1, 0), (2, 0)), 'Unsupported array header')
                prefix_size = 2 if version == (1, 0) else 4
                prefix = member.read(prefix_size)
                require(len(prefix) == prefix_size, 'Truncated array header')
                length = struct.unpack('<H' if prefix_size == 2 else '<I', prefix)[0]
                require(length <= HEADER_LIMIT, 'Array header exceeds limit')
                header = member.read(length)
                require(len(header) == length, 'Truncated array header')
                reader = (np.lib.format.read_array_header_1_0 if version == (1, 0)
                          else np.lib.format.read_array_header_2_0)
                shape, fortran, dtype = reader(io.BytesIO(prefix + header), max_header_size=HEADER_LIMIT)
                require(shape == expected_shape and dtype == expected_dtype and not fortran,
                        'Array header does not match registered shape')
                require(member.tell() <= HEADER_LIMIT and member.tell() + math.prod(shape) * dtype.itemsize
                        == item.file_size, 'Array member byte size differs')
    source.seek(0)
    with np.load(source, allow_pickle=False) as archive:
        # Do not access or decode the registered seed array.
        result = {name: archive[name].copy() for name in expected}
    return result


def read_archive(path, cfg, name):
    limits = cfg['data_limits']
    require(path.stat().st_size <= limits['max_archive_bytes'], 'Archive exceeds compressed budget')
    raw = snapshot(path, limits['max_archive_bytes'])
    return decode_arrays(raw, cfg['archive_sha256'][name], cfg['expected_groups'][name],
                         cfg['expected_candidates'][name], limits['max_uncompressed_bytes'])
