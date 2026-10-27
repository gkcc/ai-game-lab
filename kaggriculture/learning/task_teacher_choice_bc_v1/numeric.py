"""Strict teacher-choice archive. Snapshot and decoded arrays require extra RAM.

Archive/array budgets bound inputs, not total process memory or peak RSS.
"""
import math
import io
import struct
import hashlib
from pathlib import Path
import zipfile
import numpy as np
from learning.task_choice_v1.features import OPS

DTYPES = dict(shared=np.dtype('float32'),candidates=np.dtype('float32'),offsets=np.dtype('int64'),
              teacher_choice_label=np.dtype('int64'),ops=np.dtype('uint8'),seed=np.dtype('uint32'))
KEYS = frozenset(DTYPES)
HEADER_LIMIT = 4096


def _limits(max_groups,max_candidates,max_archive_bytes,max_uncompressed_bytes):
    if any(type(x) is not int or x < 0 for x in (max_groups,max_candidates,max_archive_bytes,max_uncompressed_bytes)):
        raise ValueError('Invalid archive limits')


def validate(arrays,max_groups,max_candidates):
    if type(max_groups) is not int or type(max_candidates) is not int or min(max_groups,max_candidates) < 0:
        raise ValueError('Invalid array limits')
    if not isinstance(arrays,dict) or set(arrays) != KEYS: raise ValueError('Invalid array fields')
    if any(not isinstance(arrays[k],np.ndarray) or arrays[k].dtype != dtype for k,dtype in DTYPES.items()):
        raise ValueError('Invalid array dtype')
    s,c,o = arrays['shared'],arrays['candidates'],arrays['offsets']
    if s.ndim != 2 or s.shape[1] != 275 or not len(s) or len(s) > max_groups or c.ndim != 2 or c.shape[1] != 98 or len(c) > max_candidates:
        raise ValueError('Invalid dimensions or limits')
    n = len(s)
    if o.shape != (n+1,) or o[0] != 0 or o[-1] != len(c) or np.any(o[1:] <= o[:-1]):
        raise ValueError('Invalid offsets')
    if arrays['teacher_choice_label'].shape != (n,) or arrays['seed'].shape != (n,) or arrays['ops'].shape != (len(c),):
        raise ValueError('Invalid metadata dimensions')
    if (not np.isfinite(s).all() or not np.isfinite(c).all() or np.any(arrays['ops'] >= len(OPS))
        or np.any(arrays['teacher_choice_label'] < 0) or np.any(arrays['teacher_choice_label'] >= np.diff(o))):
        raise ValueError('Invalid numeric values')
    return arrays


def write_new(path,arrays,max_groups,max_candidates):
    validate(arrays,max_groups,max_candidates)
    with Path(path).open('xb') as stream: np.savez(stream,**arrays)


def write_stream(stream,arrays,max_groups,max_candidates):
    """For caller-owned empty streams, e.g. in-memory regression tests."""
    validate(arrays,max_groups,max_candidates)
    if stream.tell() != 0 or stream.read(1): raise ValueError('Stream must be empty')
    stream.seek(0)
    np.savez(stream,**arrays)


def read(path,max_groups,max_candidates,max_archive_bytes,max_uncompressed_bytes,expected_sha256):
    _limits(max_groups,max_candidates,max_archive_bytes,max_uncompressed_bytes)
    if not isinstance(expected_sha256,str) or len(expected_sha256) != 64 or any(c not in '0123456789abcdef' for c in expected_sha256):
        raise ValueError('Expected archive SHA required')
    if hasattr(path,'seek'):
        path.seek(0)
        snapshot = path.read(max_archive_bytes+1)
    else:
        with Path(path).open('rb') as source:
            snapshot = source.read(max_archive_bytes+1)
    if not isinstance(snapshot,bytes) or len(snapshot) > max_archive_bytes: raise ValueError('Archive exceeds compressed budget')
    if hashlib.sha256(snapshot).hexdigest() != expected_sha256: raise ValueError('Archive identity changed')
    stream = io.BytesIO(snapshot)
    with zipfile.ZipFile(stream) as archive:
        infos = archive.infolist()
        if len(infos) != len(KEYS) or {i.filename for i in infos} != {k+'.npy' for k in KEYS}:
            raise ValueError('Invalid archive fields')
        if any(i.flag_bits & 1 or i.compress_type != zipfile.ZIP_STORED for i in infos):
            raise ValueError('Invalid archive encoding')
        if sum(i.file_size for i in infos) > max_uncompressed_bytes:
            raise ValueError('Archive exceeds decoded budget')
        shapes = {}
        for info in infos:
            key = info.filename[:-4]
            with archive.open(info) as member:
                version = np.lib.format.read_magic(member)
                if version not in ((1,0),(2,0)): raise ValueError('Unsupported array header')
                length_size = 2 if version == (1,0) else 4
                prefix = member.read(length_size)
                if len(prefix) != length_size: raise ValueError('Truncated array header')
                length = struct.unpack('<H' if length_size == 2 else '<I',prefix)[0]
                if length > HEADER_LIMIT: raise ValueError('Array header exceeds limit')
                header = member.read(length)
                if len(header) != length: raise ValueError('Truncated array header')
                header_stream = io.BytesIO(prefix+header)
                reader = np.lib.format.read_array_header_1_0 if version == (1,0) else np.lib.format.read_array_header_2_0
                shape,fortran,dtype = reader(header_stream,max_header_size=HEADER_LIMIT)
                if dtype != DTYPES[key] or fortran or not isinstance(shape,tuple) or any(type(x) is not int or x < 0 for x in shape):
                    raise ValueError('Invalid array header')
                if member.tell() > HEADER_LIMIT or member.tell()+math.prod(shape)*dtype.itemsize != info.file_size:
                    raise ValueError('Array size inconsistent')
                shapes[key] = shape
        if (len(shapes['shared']) != 2 or shapes['shared'][1] != 275 or not shapes['shared'][0] or shapes['shared'][0] > max_groups
            or len(shapes['candidates']) != 2 or shapes['candidates'][1] != 98 or shapes['candidates'][0] > max_candidates):
            raise ValueError('Declared dimensions exceed limits')
        n,m = shapes['shared'][0],shapes['candidates'][0]
        if (shapes['offsets'] != (n+1,) or shapes['teacher_choice_label'] != (n,)
            or shapes['seed'] != (n,) or shapes['ops'] != (m,)):
            raise ValueError('Declared metadata dimensions inconsistent')
    stream.seek(0)
    with np.load(stream,allow_pickle=False) as loaded:
        arrays = {k:loaded[k] for k in loaded.files}
    return validate(arrays,max_groups,max_candidates)
