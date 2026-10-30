"""Bounded in-memory task checkpoints; no filesystem, games or GPU.

The caller must authenticate the expected checkpoint SHA and the separately
registered deployment-visible schema SHA before calling ``load_checkpoint``.
Only immutable byte snapshots are accepted; no file is reopened after hashing.
The explicit checks below follow the existing TaskNetwork contract and reuse
its constants, schema hashing and binary-feature normalization mask. They do
not execute TaskNetwork's path-based constructor; real file-entry compatibility
must be verified later with the existing scorer on a separately written artifact.
"""

from dataclasses import dataclass
import hashlib
import io
import json
import math
import re
from types import MappingProxyType
import zipfile
import zlib

import numpy as np

from learning.task_bc_v1 import model


MAX_ARCHIVE_BYTES = 2 * 1024 * 1024
MAX_UNCOMPRESSED_BYTES = 1024 * 1024
MAX_HEADER_BYTES = 4096
MAX_METADATA_BYTES = 128 * 1024
HEX_SHA = re.compile(r'[0-9a-f]{64}\Z')
HEX_COMMIT = re.compile(r'[0-9a-f]{40}\Z')
NAMES = ('w0', 'b0', 'w1', 'b1', 'w2', 'b2', 'mean', 'scale', 'metadata')
SHAPES = {'w0': (model.DIMENSIONS[0], model.DIMENSIONS[1]), 'b0': (model.DIMENSIONS[1],),
          'w1': (model.DIMENSIONS[1], model.DIMENSIONS[2]), 'b1': (model.DIMENSIONS[2],),
          'w2': (model.DIMENSIONS[2], model.DIMENSIONS[3]), 'b2': (model.DIMENSIONS[3],),
          'mean': (model.DIMENSIONS[0],), 'scale': (model.DIMENSIONS[0],), 'metadata': ()}
PARAMS = ('w0', 'b0', 'w1', 'b1', 'w2', 'b2')
SCHEMA_KEYS = ('schema_version', 'schema_sha256', 'shared_feature_names',
               'feature_names', 'ops', 'dimensions')


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _unique_object(pairs):
    result = {}
    for name, value in pairs:
        _require(name not in result, 'Duplicate metadata field')
        result[name] = value
    return result


def _reject_constant(value):
    raise ValueError('Nonstandard metadata numeric constant')


def _parse_metadata(array):
    try:
        text = str(array)
        value = json.loads(text, object_pairs_hook=_unique_object, parse_constant=_reject_constant)
    except (ValueError, TypeError, UnicodeError, OverflowError, RecursionError):
        raise ValueError('Invalid checkpoint metadata') from None
    _require(type(value) is dict, 'Invalid checkpoint metadata')
    return value


def _inspect_header(archive, info, name):
    """Check one NPY header and its exact payload size without an array."""
    with archive.open(info, 'r') as stream:
        version = np.lib.format.read_magic(stream)
        if version == (1, 0):
            shape, fortran, dtype = np.lib.format.read_array_header_1_0(stream, max_header_size=MAX_HEADER_BYTES)
        elif version == (2, 0):
            shape, fortran, dtype = np.lib.format.read_array_header_2_0(stream, max_header_size=MAX_HEADER_BYTES)
        else:
            raise ValueError('Unsupported checkpoint NPY version')
        _require(shape == SHAPES[name] and type(fortran) is bool and not dtype.hasobject
                and dtype.fields is None and dtype.subdtype is None, 'Invalid checkpoint array header')
        if name in PARAMS:
            _require(dtype.kind == 'f' and dtype.itemsize == 4, 'Checkpoint weights must be float32')
        elif name in ('mean', 'scale'):
            _require(dtype.kind == 'f' and dtype.itemsize in (4, 8), 'Invalid normalization dtype')
        else:
            _require(dtype.kind == 'U' and 0 < dtype.itemsize <= MAX_METADATA_BYTES,
                     'Invalid metadata dtype')
        count = math.prod(shape) if shape else 1
        payload_size = count * dtype.itemsize
        _require(payload_size <= MAX_UNCOMPRESSED_BYTES
                and stream.tell() <= MAX_HEADER_BYTES + 16
                and info.file_size == stream.tell() + payload_size, 'Invalid checkpoint payload size')
        return shape, fortran, dtype, count, stream.tell(), payload_size


def _read_member(archive, info, header):
    """Decode only after the entire archive's headers have been checked."""
    shape, fortran, dtype, count, header_size, payload_size = header
    with archive.open(info, 'r') as stream:
        prefix = stream.read(header_size)
        _require(len(prefix) == header_size, 'Truncated checkpoint header')
        payload = stream.read(payload_size + 1)
        _require(len(payload) == payload_size, 'Truncated or oversized checkpoint payload')
    array = np.frombuffer(payload, dtype=dtype, count=count).reshape(shape, order='F' if fortran else 'C')
    return np.array(array, copy=True, order='K')


def _decode_npz(raw):
    """Decode a small, exact member set without calling ``np.load``."""
    try:
        with zipfile.ZipFile(io.BytesIO(raw), 'r') as archive:
            infos = archive.infolist()
            names = [info.filename for info in infos]
            _require(len(names) == len(NAMES) and len(set(names)) == len(names)
                    and set(names) == {name + '.npy' for name in NAMES},
                    'Unexpected or duplicate checkpoint members')
            _require(all(not (info.flag_bits & 1) and info.compress_type in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED)
                         and info.file_size >= 0 and info.compress_size >= 0 for info in infos),
                     'Unsupported checkpoint compression')
            _require(sum(info.file_size for info in infos) <= MAX_UNCOMPRESSED_BYTES
                    and sum(info.compress_size for info in infos) <= MAX_ARCHIVE_BYTES,
                    'Checkpoint contents exceed budget')
            # No member array is materialized until every member header is known
            # to have an allowed shape, dtype and exact bounded payload size.
            headers = {info.filename[:-4]: _inspect_header(archive, info, info.filename[:-4])
                       for info in infos}
            arrays = {}
            for info in infos:
                name = info.filename[:-4]
                arrays[name] = _read_member(archive, info, headers[name])
            return arrays
    except (zipfile.BadZipFile, EOFError, OSError, RuntimeError, TypeError, UnicodeError,
            OverflowError, zlib.error) as exc:
        raise ValueError('Invalid checkpoint archive') from None


def _validate_schema(metadata, expected_schema_sha):
    _require(type(metadata) is dict and all(key in metadata for key in SCHEMA_KEYS),
             'Missing task schema metadata')
    shared, features = metadata['shared_feature_names'], metadata['feature_names']
    _require(type(shared) is list and len(shared) == 275 and type(features) is list and len(features) == 98
            and all(type(name) is str and 0 < len(name) <= 512 and '\x00' not in name
                    for name in shared + features)
            and len(set(shared)) == len(shared) and len(set(features)) == len(features),
            'Invalid task feature names')
    _require(model.DIMENSIONS == (373, 128, 64, 1)
            and type(metadata['schema_version']) is str and metadata['schema_version'] == model.SCHEMA_VERSION
            and type(metadata['ops']) is list and all(type(op) is str for op in metadata['ops'])
            and metadata['ops'] == list(model.OPS)
            and type(metadata['dimensions']) is list
            and all(type(dimension) is int for dimension in metadata['dimensions'])
            and metadata['dimensions'] == list(model.DIMENSIONS)
            and type(metadata['schema_sha256']) is str and HEX_SHA.fullmatch(metadata['schema_sha256'])
            and metadata['schema_sha256'] == expected_schema_sha
            and metadata['schema_sha256'] == model.schema_hash(features, shared),
            'Task schema does not match the independently bound schema')
    return {key: metadata[key] for key in SCHEMA_KEYS}


def _validate_arrays(arrays, schema):
    mean, scale = arrays['mean'], arrays['scale']
    _require(isinstance(mean, np.ndarray) and isinstance(scale, np.ndarray)
            and mean.shape == (373,) and scale.shape == (373,)
            and mean.dtype.kind == 'f' and mean.dtype.itemsize in (4, 8)
            and scale.dtype.kind == 'f' and scale.dtype.itemsize in (4, 8)
            and bool(np.isfinite(mean).all()) and bool(np.isfinite(scale).all())
            and bool(np.all(scale >= 1)), 'Invalid task normalization')
    continuous = model.continuous_mask(schema['shared_feature_names'] + schema['feature_names'])
    _require(continuous.shape == (373,) and continuous.dtype == np.dtype(bool)
            and bool(np.all(mean[~continuous] == 0)) and bool(np.all(scale[~continuous] == 1)),
            'Non-continuous features must retain identity normalization')
    for name in PARAMS:
        value = arrays[name]
        _require(isinstance(value, np.ndarray) and value.shape == SHAPES[name]
                and value.dtype.kind == 'f' and value.dtype.itemsize == 4
                and bool(np.isfinite(value).all()), 'Invalid task weight')


@dataclass(frozen=True, repr=False)
class Checkpoint:
    """Private immutable-by-default arrays; values and metadata must not be logged."""
    params: object
    mean: np.ndarray
    scale: np.ndarray
    schema_json: str
    bound_schema_sha256: str

    def __repr__(self):
        return '<PrivateTaskCheckpoint>'


def load_checkpoint(raw, *, expected_sha256, expected_schema_sha256):
    """Validate one immutable NPZ snapshot against pre-bound identities."""
    _require(type(raw) is bytes and 0 < len(raw) <= MAX_ARCHIVE_BYTES
            and type(expected_sha256) is str and HEX_SHA.fullmatch(expected_sha256)
            and type(expected_schema_sha256) is str and HEX_SHA.fullmatch(expected_schema_sha256),
            'Invalid checkpoint snapshot identity')
    _require(hashlib.sha256(raw).hexdigest() == expected_sha256, 'Checkpoint SHA differs')
    arrays = _decode_npz(raw)
    metadata = _parse_metadata(arrays['metadata'])
    schema = _validate_schema(metadata, expected_schema_sha256)
    _validate_arrays(arrays, schema)
    params = {}
    for name in PARAMS:
        # Native float32 is the original training representation.
        params[name] = np.array(arrays[name], dtype=np.float32, copy=True)
        params[name].setflags(write=False)
    mean, scale = arrays['mean'].copy(), arrays['scale'].copy()
    mean.setflags(write=False)
    scale.setflags(write=False)
    schema_json = json.dumps(schema, sort_keys=True, ensure_ascii=True, separators=(',', ':'))
    return Checkpoint(MappingProxyType(params), mean, scale, schema_json, expected_schema_sha256)


def _validated_new_params(params):
    _require(type(params) is dict and set(params) == set(PARAMS), 'Invalid new parameter set')
    result = {}
    for name in PARAMS:
        value = params[name]
        _require(isinstance(value, np.ndarray) and value.shape == SHAPES[name]
                and value.dtype.kind == 'f' and value.dtype.itemsize == 4
                and bool(np.isfinite(value).all()), 'Invalid new parameter array')
        result[name] = np.array(value, dtype=np.float32, copy=True)
    return result


def _different(first, second):
    return any(not np.array_equal(first[name], second[name]) for name in PARAMS)


def _export_arm(snapshot, params, schema, provenance, arm):
    metadata = dict(schema)
    metadata['pair_training'] = {
        'schema': 'task-pair-training-provenance-v1',
        'arm': arm,
        'training_source': provenance['training_source'],
        'initialization': provenance['initialization'],
        'registration_commit': provenance['registration_commit'],
        'completed_updates': provenance['completed_updates'],
        'worker_coefficient': 0.0 if arm == 'control' else 0.25,
    }
    encoded = json.dumps(metadata, sort_keys=True, ensure_ascii=True, separators=(',', ':'))
    _require(len(encoded) * 4 <= MAX_METADATA_BYTES, 'Export metadata exceeds limit')
    stream = io.BytesIO()
    np.savez_compressed(stream, **params, mean=snapshot.mean, scale=snapshot.scale,
                        metadata=np.asarray(encoded))
    result = stream.getvalue()
    _require(len(result) <= MAX_ARCHIVE_BYTES, 'Export exceeds archive limit')
    return result


def export_pair(snapshot, control_params, auxiliary_params, *, provenance):
    """Return two private NPZ snapshots; caller must independently prove provenance.

    The narrow provenance enum prevents paths, original seeds and raw source
    identities from being embedded. Real use is accepted only for a completed
    64-update pair; synthetic-only exports are explicitly marked as such.
    The caller remains responsible for proving those claims and for exclusive
    file writes and actual TaskNetwork reload in the registered future run.
    """
    _require(isinstance(snapshot, Checkpoint) and type(provenance) is dict
            and set(provenance) == {'training_source', 'initialization', 'registration_commit', 'completed_updates'},
            'Invalid paired export provenance')
    source, initialization = provenance['training_source'], provenance['initialization']
    _require((source, initialization) in (('original_teacher_train', 'frozen_early_task_head'),
                                           ('synthetic_only', 'synthetic_fixture'))
            and type(provenance['registration_commit']) is str
            and HEX_COMMIT.fullmatch(provenance['registration_commit'])
            and type(provenance['completed_updates']) is int
            and 0 <= provenance['completed_updates'] <= 64
            and (source != 'original_teacher_train' or provenance['completed_updates'] == 64),
            'Invalid paired export provenance')
    _require(type(snapshot.schema_json) is str and len(snapshot.schema_json) * 4 <= MAX_METADATA_BYTES
            and type(snapshot.bound_schema_sha256) is str and HEX_SHA.fullmatch(snapshot.bound_schema_sha256)
            and hasattr(snapshot.params, 'keys') and set(snapshot.params.keys()) == set(PARAMS),
            'Invalid baseline snapshot')
    schema = _parse_metadata(np.asarray(snapshot.schema_json))
    schema = _validate_schema(schema, snapshot.bound_schema_sha256)
    baseline = {name: snapshot.params[name] for name in PARAMS}
    _validate_arrays(dict(baseline, mean=snapshot.mean, scale=snapshot.scale), schema)
    control = _validated_new_params(control_params)
    auxiliary = _validated_new_params(auxiliary_params)
    _require(_different(control, baseline) and _different(auxiliary, baseline)
            and _different(control, auxiliary), 'New paired weights must differ')
    return (_export_arm(snapshot, control, schema, provenance, 'control'),
            _export_arm(snapshot, auxiliary, schema, provenance, 'auxiliary'))
