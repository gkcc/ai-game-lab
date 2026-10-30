"""Compose existing paired components in memory; caller authenticates sources.

There is no file I/O, device selection, source authentication or process
supervision here. Production must authenticate inputs, enforce its own budget
and device, exclusively write outputs and verify real TaskNetwork file reload.
"""

from dataclasses import dataclass
import hashlib
import json
from types import MappingProxyType

import numpy as np

from learning.task_bc_v1.model import TaskNetwork
from learning.task_pair_checkpoint_v1 import snapshot
from learning.task_pair_eval_adapter_v1 import adapter as evaluator
from learning.task_pair_host_export_v1.bridge import copy_pair_to_host
from learning.task_pair_numpy_scorer_v1 import scorer
from learning.task_pair_orchestration_v1.core import train_pair


_SOURCES = ('teacher_dev', 'conditional_old', 'conditional_new')
_MODELS = ('baseline', 'control', 'auxiliary')


@dataclass(frozen=True, repr=False)
class TrainingExports:
    """Private baseline snapshot and two in-memory checkpoint byte strings."""
    baseline: object
    control_bytes: bytes
    auxiliary_bytes: bytes
    completed_updates: int
    auxiliary_nonzero_updates: int

    def __repr__(self):
        return '<PrivateInMemoryTrainingExports>'


@dataclass(frozen=True, repr=False)
class DevSource:
    """Caller-authenticated view, masks and complete expected cluster set."""
    view: object
    masks: object
    expected_clusters: object

    def __repr__(self):
        return '<PrivateDevelopmentSource>'


@dataclass(frozen=True, repr=False)
class PrivateEvaluation:
    """Private model/source metrics; never a standalone game evaluation."""
    metrics: object

    def __repr__(self):
        return '<PrivateDevelopmentEvaluation>'


def _freeze_checkpoint(checkpoint, expected_schema_sha256):
    """Validate fixed-size checkpoint arrays, then copy before callbacks."""
    if (type(checkpoint) is not snapshot.Checkpoint
            or type(expected_schema_sha256) is not str
            or snapshot.HEX_SHA.fullmatch(expected_schema_sha256) is None
            or checkpoint.bound_schema_sha256 != expected_schema_sha256
            or type(checkpoint.schema_json) is not str
            or len(checkpoint.schema_json) * 4 > snapshot.MAX_METADATA_BYTES
            or type(checkpoint.params) is not MappingProxyType
            or len(checkpoint.params) != len(snapshot.PARAMS)
            or any(key not in checkpoint.params for key in snapshot.PARAMS)):
        raise ValueError('Baseline checkpoint contract differs')
    schema = snapshot._validate_schema(snapshot._parse_metadata(np.asarray(checkpoint.schema_json)),
                                       expected_schema_sha256)
    original = {name: checkpoint.params[name] for name in snapshot.PARAMS}
    original['mean'], original['scale'] = checkpoint.mean, checkpoint.scale
    snapshot._validate_arrays(original, schema)
    arrays = {name: np.array(value, copy=True, order='K') for name, value in original.items()}
    snapshot._validate_arrays(arrays, schema)
    for value in arrays.values():
        value.setflags(write=False)
    schema_json = json.dumps(schema, sort_keys=True, ensure_ascii=True, separators=(',', ':'))
    return snapshot.Checkpoint(MappingProxyType({key: arrays[key] for key in snapshot.PARAMS}),
                               arrays['mean'], arrays['scale'], schema_json,
                               expected_schema_sha256)


def train_and_export(data, checkpoint, class_weights, *, expected_schema_sha256, device,
                     rng_seed, updates, batch_size, training_source, registration_commit, check):
    """Run one paired budget and return two private checkpoint snapshots.

    The only synthetic budget is two 8-group CPU updates. The production-shaped
    contract is 64x256 on an explicitly supplied GPU; that shape is not proof
    of source, device model or a registered run. The caller must authenticate
    the checkpoint and training mapping before invocation. No files are opened.
    """
    if (not callable(check) or type(registration_commit) is not str
            or snapshot.HEX_COMMIT.fullmatch(registration_commit) is None
            or type(updates) is not int or type(batch_size) is not int
            or type(training_source) is not str):
        raise ValueError('Invalid in-memory training contract')
    if training_source == 'synthetic_only':
        if updates != 2 or batch_size != 8 or getattr(device, 'platform', None) != 'cpu':
            raise ValueError('Synthetic paired budget differs')
        initialization = 'synthetic_fixture'
    elif training_source == 'original_teacher_train':
        if updates != 64 or batch_size != 256 or getattr(device, 'platform', None) != 'gpu':
            raise ValueError('Production paired budget differs')
        initialization = 'frozen_early_task_head'
    else:
        raise ValueError('Invalid in-memory training source')
    baseline = _freeze_checkpoint(checkpoint, expected_schema_sha256)
    result = train_pair(data, baseline.params, baseline.mean, baseline.scale, class_weights,
                        device=device, rng_seed=rng_seed, updates=updates,
                        batch_size=batch_size, check=check)
    required_aux = 1 if training_source == 'synthetic_only' else 32
    if (type(result.completed_updates) is not int or result.completed_updates != updates
            or type(result.auxiliary_nonzero_updates) is not int
            or not required_aux <= result.auxiliary_nonzero_updates <= updates):
        raise ValueError('Paired update evidence is below the fixed gate')
    check()
    control, auxiliary = copy_pair_to_host(result, device=device)
    check()
    provenance = {'training_source': training_source, 'initialization': initialization,
                  'registration_commit': registration_commit, 'completed_updates': result.completed_updates}
    first, second = snapshot.export_pair(baseline, control, auxiliary, provenance=provenance)
    check()
    return TrainingExports(baseline, first, second, result.completed_updates,
                           result.auxiliary_nonzero_updates)


def _memory_network(checkpoint, expected_schema_sha256):
    """Build a scorer from validated memory; this is not TaskNetwork(path, sha)."""
    fixed = _freeze_checkpoint(checkpoint, expected_schema_sha256)
    model = TaskNetwork.__new__(TaskNetwork)
    model.metadata = json.loads(fixed.schema_json)
    model.mean = fixed.mean.astype(np.float64, copy=True)
    model.scale = fixed.scale.astype(np.float64, copy=True)
    model.params = {name: fixed.params[name].astype(np.float64, copy=True, order='K')
                    for name in snapshot.PARAMS}
    model.sha = None
    model.last = None
    for value in (model.mean, model.scale, *model.params.values()):
        value.setflags(write=False)
    return model


def models_from_memory(exports, *, expected_schema_sha256):
    """Create baseline/control/auxiliary scoring objects without file reload.

    SHA of each newly produced byte string is computed for its in-memory decode;
    that self-consistency check does not authenticate a model's provenance.
    """
    if (type(exports) is not TrainingExports
            or type(exports.control_bytes) is not bytes or type(exports.auxiliary_bytes) is not bytes
            or not 0 < len(exports.control_bytes) <= snapshot.MAX_ARCHIVE_BYTES
            or not 0 < len(exports.auxiliary_bytes) <= snapshot.MAX_ARCHIVE_BYTES
            or type(exports.completed_updates) is not int or type(exports.auxiliary_nonzero_updates) is not int
            or not 1 <= exports.auxiliary_nonzero_updates <= exports.completed_updates):
        raise ValueError('Invalid in-memory paired exports')
    models = {'baseline': _memory_network(exports.baseline, expected_schema_sha256)}
    for name, raw in (('control', exports.control_bytes), ('auxiliary', exports.auxiliary_bytes)):
        item = snapshot.load_checkpoint(raw, expected_sha256=hashlib.sha256(raw).hexdigest(),
                                        expected_schema_sha256=expected_schema_sha256)
        models[name] = _memory_network(item, expected_schema_sha256)
    return MappingProxyType(models)


def evaluate_models(models, sources, *, expected_schema_sha256, check):
    """Evaluate three caller-supplied TaskNetworks on three separate sources.

    The call snapshots each model and mask before any external callback. Views,
    model identity and source/mask roles must be independently authenticated and
    kept stable by the caller. All exact values are private and existing dev
    data, including unmatched subsets, are not independent game results.
    """
    if (type(models) not in (dict, MappingProxyType) or set(models) != set(_MODELS)
            or type(sources) not in (dict, MappingProxyType) or set(sources) != set(_SOURCES)
            or not callable(check)):
        raise ValueError('Invalid development evaluation inputs')
    frozen_sources = {}
    for name in _SOURCES:
        source = sources[name]
        if type(source) is not DevSource:
            raise ValueError('Invalid development source')
        count, _ = evaluator._view(source.view, expected_schema_sha256)
        clusters = evaluator._expected_clusters(source.expected_clusters)
        masks = evaluator._masks(source.masks, count)
        for mask in masks.values():
            mask.setflags(write=False)
        frozen_sources[name] = DevSource(source.view, masks, clusters)
    reference = frozen_sources['teacher_dev'].view
    frozen_models = {name: scorer._snapshot_model(reference, models[name], expected_schema_sha256)
                     for name in _MODELS}
    for model in frozen_models.values():
        for source in frozen_sources.values():
            scorer._verify_schema(source.view, model, expected_schema_sha256)
    metrics = {}
    for name in _MODELS:
        metrics[name] = {}
        for source_name in _SOURCES:
            source = frozen_sources[source_name]
            blocks = scorer.score_blocks(source.view, frozen_models[name],
                                         expected_schema_sha256=expected_schema_sha256, check=check)
            metrics[name][source_name] = evaluator.evaluate_source(
                source.view, blocks, source.masks, source=source_name,
                expected_clusters=source.expected_clusters, expected_schema_sha256=expected_schema_sha256)
    return PrivateEvaluation(metrics)
