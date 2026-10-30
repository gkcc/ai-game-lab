"""Load preflight masks only after source provenance is independently verified.

The manifest digest is fixed locally before the future training attempt. Matching
mask lengths alone cannot bind masks to a particular source array: the caller
must first authenticate the data registration, full source manifest, arrays,
role assignments and old/new source order. No mask is a model feature.
"""
import hashlib
import io
import json
import re

import numpy as np


KINDS = ('seen_in_train', 'label_conflict', 'qualifying_plant')
ROLES = ('teacher_dev', 'conditional_dev')
SHA = re.compile(r'[0-9a-f]{64}\Z')
COMMIT = re.compile(r'[0-9a-f]{40}\Z')


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('Duplicate manifest field')
        result[key] = value
    return result


def load_masks(manifest_bytes, read_file, *, manifest_sha256, registration_commit, role_lengths):
    """Return private masks from bounded file bytes, or fail closed.

    ``read_file(name, maximum_bytes)`` must itself enforce a regular file under
    the registered local directory and limit its read to the supplied maximum.
    ``role_lengths`` comes from the separately authenticated, ordered views.
    """
    if (type(manifest_bytes) is not bytes or len(manifest_bytes) > 65536
            or not callable(read_file) or type(manifest_sha256) is not str
            or not SHA.fullmatch(manifest_sha256)
            or hashlib.sha256(manifest_bytes).hexdigest() != manifest_sha256
            or type(registration_commit) is not str or not COMMIT.fullmatch(registration_commit)):
        raise ValueError('Invalid local mask manifest identity')
    try:
        manifest = json.loads(manifest_bytes.decode('utf-8'), object_pairs_hook=_object)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError('Invalid local mask manifest') from exc
    if (type(manifest) is not dict or set(manifest) !=
            {'schema', 'registration_commit', 'conditional_source_order', 'masks'}
            or manifest['schema'] != 'task-pair-data-local-masks-v1'
            or manifest['registration_commit'] != registration_commit
            or manifest['conditional_source_order'] != ['old', 'new']
            or type(manifest['masks']) is not list):
        raise ValueError('Local mask manifest contract differs')
    if type(role_lengths) is not dict or set(role_lengths) != set(ROLES):
        raise ValueError('Invalid authenticated source lengths')
    expected = {}
    for role, count in (('teacher_dev', 1), ('conditional_dev', 2)):
        lengths = role_lengths[role]
        if (type(lengths) not in (tuple, list) or len(lengths) != count
                or any(type(n) is not int or not 1 <= n <= 1_000_000 for n in lengths)):
            raise ValueError('Invalid authenticated source lengths')
        for source, length in enumerate(lengths):
            for kind in KINDS:
                expected[f'{role}-{source}-{kind}.npy'] = (role, source, kind, length)
    entries = {}
    for item in manifest['masks']:
        if (type(item) is not dict or set(item) != {'name', 'sha256', 'length'}
                or type(item['name']) is not str or item['name'] not in expected
                or item['name'] in entries or type(item['length']) is not int
                or item['length'] != expected[item['name']][3]
                or type(item['sha256']) is not str or not SHA.fullmatch(item['sha256'])):
            raise ValueError('Invalid or duplicate local mask entry')
        entries[item['name']] = item
    if set(entries) != set(expected):
        raise ValueError('Incomplete local mask set')
    result = {role: [{} for _ in role_lengths[role]] for role in ROLES}
    for name, (role, source, kind, length) in expected.items():
        maximum = length + 65536
        raw = read_file(name, maximum)
        if (type(raw) is not bytes or len(raw) > maximum
                or hashlib.sha256(raw).hexdigest() != entries[name]['sha256']):
            raise ValueError('Local mask bytes differ')
        stream = io.BytesIO(raw)
        try:
            version = np.lib.format.read_magic(stream)
            if version == (1, 0):
                shape, _, dtype = np.lib.format.read_array_header_1_0(stream, max_header_size=4096)
            elif version == (2, 0):
                shape, _, dtype = np.lib.format.read_array_header_2_0(stream, max_header_size=4096)
            else:
                raise ValueError('Unsupported local mask array version')
        except (ValueError, OSError, EOFError, TypeError) as exc:
            raise ValueError('Invalid local mask array') from exc
        # Check the small header and exact payload size before allocating an
        # array; a short file can claim a huge shape in its NPY header.
        if shape != (length,) or dtype != np.dtype(bool) or len(raw) - stream.tell() != length:
            raise ValueError('Invalid local mask array')
        payload = np.frombuffer(raw, dtype=np.uint8, count=length, offset=stream.tell())
        if np.any(payload > 1):
            raise ValueError('Invalid local mask values')
        mask = payload.astype(bool, copy=True)
        result[role][source][kind] = mask
    for shards in result.values():
        for shard in shards:
            if np.any(shard['label_conflict'] & ~shard['seen_in_train']):
                raise ValueError('Conflict outside training overlap')
            if np.any(shard['label_conflict']):
                raise ValueError('Registered zero-conflict gate is not satisfied')
    return result
