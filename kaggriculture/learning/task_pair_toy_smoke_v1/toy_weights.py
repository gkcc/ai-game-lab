"""Build fixed synthetic checkpoint bytes from independently supplied schemas.

This module performs no filesystem IO, training, inference or game execution.
It emits in-memory NPZ bytes for a future separately authorized toy smoke, not
learned policies. Schema hashes pin the existing public feature encoders; a
future intentional encoder change requires a separately reviewed revision.
"""
import hashlib
import io
import json

import numpy as np

from decision.api import PRODUCTS
from learning.bc_crop.agent import TEACHER_SHA
from learning.bc_crop.features import CROPS, SCHEMA_VERSION as CROP_VERSION
from learning.bc_crop.features import schema_hash as crop_schema_hash
from learning.task_bc_v1.model import DIMENSIONS
from learning.task_choice_v1.features import OPS, SCHEMA_VERSION as TASK_VERSION
from learning.task_choice_v1.features import schema_hash as task_schema_hash


# From the public pinned encoders on a synthetic controller context. These are
# schema hashes, not private checkpoints or identities of a trained model.
EXPECTED_CROP_SCHEMA = '49f545bb22616626c88ab113240b538f99e2cc54de87d82ac5bf286775377125'
EXPECTED_TASK_SCHEMA = '235eef551a799e7dc61e8b25448f9b7f6215bf6f97d044bf84b6edf9bd91ea56'
LOCAL_BASE = ('unit.index', 'unit.x', 'unit.y', 'target.x', 'target.y',
              *('carried_unit.' + item for item in PRODUCTS))


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _names(value, size):
    _require(type(value) is list and len(value) == size
             and all(type(name) is str and name and len(name) <= 256
                     and all(ord(char) >= 32 and ord(char) != 127 for char in name)
                     for name in value)
             and len(set(value)) == size, 'Invalid feature names')
    return list(value)


def _schemas(crop_schema, task_schema):
    _require(type(crop_schema) is dict and type(task_schema) is dict, 'Schema objects required')
    crop = _names(crop_schema.get('features'), 289)
    shared = _names(task_schema.get('shared_feature_names'), 275)
    local = _names(task_schema.get('feature_names'), 98)
    _require(crop_schema.get('schema_version') == CROP_VERSION
             and crop_schema.get('crops') == list(CROPS)
             and crop_schema.get('teacher_sha256') == TEACHER_SHA
             and crop_schema.get('schema_sha256') == EXPECTED_CROP_SCHEMA
             and crop_schema_hash(crop) == EXPECTED_CROP_SCHEMA,
             'Crop schema differs from pinned encoder')
    _require(task_schema.get('schema_version') == TASK_VERSION
             and task_schema.get('ops') == list(OPS)
             and task_schema.get('schema_sha256') == EXPECTED_TASK_SCHEMA
             and task_schema_hash(local, shared) == EXPECTED_TASK_SCHEMA,
             'Task schema differs from pinned encoder')
    _require(len(set(shared + local)) == 373 and local[:len(LOCAL_BASE)] == list(LOCAL_BASE)
             and all(name in crop for name in LOCAL_BASE)
             and [name for name in crop if name not in LOCAL_BASE] == shared,
             'Task and crop schema relationship differs')
    return crop, shared, local


def _archive(metadata, arrays):
    stream = io.BytesIO()
    np.savez_compressed(stream, metadata=np.asarray(json.dumps(metadata, sort_keys=True,
                                                             separators=(',', ':'))), **arrays)
    data = stream.getvalue()
    return dict(data=data, sha256=hashlib.sha256(data).hexdigest())


def _task(shared, local, output_sign, output_bias):
    metadata = dict(schema_version=TASK_VERSION, schema_sha256=EXPECTED_TASK_SCHEMA,
                    shared_feature_names=shared, feature_names=local, ops=list(OPS),
                    dimensions=list(DIMENSIONS), source='synthetic_toy', trained=False)
    arrays = dict(mean=np.zeros(373, dtype=np.float32), scale=np.ones(373, dtype=np.float32),
                  w0=np.zeros((373, 128), dtype=np.float32), b0=np.zeros(128, dtype=np.float32),
                  w1=np.zeros((128, 64), dtype=np.float32), b1=np.zeros(64, dtype=np.float32),
                  w2=np.zeros((64, 1), dtype=np.float32), b2=np.asarray([output_bias], dtype=np.float32))
    arrays['w0'][275 + local.index('target.x'), 0] = 1.
    arrays['w1'][0, 0] = 1.
    arrays['w2'][0, 0] = output_sign
    return _archive(metadata, arrays)


def _crop(names):
    metadata = dict(schema_version=CROP_VERSION, crops=list(CROPS), teacher_sha256=TEACHER_SHA,
                    features=names, schema_sha256=EXPECTED_CROP_SCHEMA,
                    source='synthetic_toy', trained=False)
    arrays = dict(mean=np.zeros(289, dtype=np.float32), scale=np.ones(289, dtype=np.float32),
                  w0=np.zeros((289, 128), dtype=np.float32), b0=np.zeros(128, dtype=np.float32),
                  w1=np.zeros((128, 64), dtype=np.float32), b1=np.zeros(64, dtype=np.float32),
                  w2=np.zeros((64, 5), dtype=np.float32), b2=np.asarray([0, 1, 0, 0, 0], dtype=np.float32))
    return _archive(metadata, arrays)


def build(*, crop_schema, task_schema):
    """Return in-memory untrained model archives for three fixed toy roles.

    ``crop_schema`` supplies crop ``features``, ``schema_version``, ``crops``,
    ``teacher_sha256`` and ``schema_sha256``; ``task_schema`` supplies task
    ``shared_feature_names``, ``feature_names``, ``schema_version``, ``ops``
    and ``schema_sha256``. The supplied hashes and names must match the pinned
    public encoders, including their order. Extra source fields are not copied.

    Two task heads score the nonnegative ``target.x`` feature with opposite
    signs; they can select different candidates when that coordinate differs,
    but no particular real-game choice or competitive value is guaranteed.
    The auxiliary head adds a constant bias and serves only as a distinct panel
    placeholder. Digests identify returned synthetic bytes, not source data.
    """
    crop, shared, local = _schemas(crop_schema, task_schema)
    result = dict(schema='task-pair-toy-weight-archives-v1', trained=False,
                  crop=_crop(crop), tasks={
                      'baseline': _task(shared, local, 1., 0.),
                      'control': _task(shared, local, -1., 0.),
                      'auxiliary': _task(shared, local, 1., .25),
                  })
    values = [result['crop']['sha256'], *(item['sha256'] for item in result['tasks'].values())]
    _require(len(set(values)) == 4, 'Synthetic archive identities collide')
    return result
