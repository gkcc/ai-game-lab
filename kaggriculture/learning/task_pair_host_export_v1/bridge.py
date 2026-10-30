"""Copy paired device parameters into bounded host arrays for the exporter.

This helper authenticates neither a model, training result, device identity nor
provenance. The caller must separately establish each of those facts, enforce
the registered 64-update budget when applicable, and supervise the process.
Nothing is written, logged, or reloaded here.
"""

import math
from types import MappingProxyType

import numpy as np

from learning.task_pair_checkpoint_v1 import snapshot
from learning.task_pair_orchestration_v1.core import PairResult


_PARAMS = ('w0', 'b0', 'w1', 'b1', 'w2', 'b2')
_SHAPES = {'w0': (373, 128), 'b0': (128,), 'w1': (128, 64), 'b1': (64,),
           'w2': (64, 1), 'b2': (1,)}
MAX_HOST_BYTES = 512 * 1024


def _snapshot_inputs(result, device, jax):
    if (type(result) is not PairResult
            or type(result.completed_updates) is not int or not 1 <= result.completed_updates <= 1024
            or type(result.auxiliary_nonzero_updates) is not int
            or not 1 <= result.auxiliary_nonzero_updates <= result.completed_updates
            or result.both_arms_changed is not True or result.arms_differ is not True
            or type(result.control_params) is not MappingProxyType
            or type(result.experiment_params) is not MappingProxyType
            or not isinstance(device, jax.Device) or device.platform not in ('cpu', 'gpu')
            or device not in jax.devices(device.platform)
            or tuple(snapshot.PARAMS) != _PARAMS
            or snapshot.model.DIMENSIONS != (373, 128, 64, 1)
            or any(snapshot.SHAPES.get(name) != _SHAPES[name] for name in _PARAMS)):
        raise ValueError('Invalid paired result or selected device')
    mappings = (result.control_params, result.experiment_params)
    # Refuse an oversized mapping before iterating or copying it. With exactly
    # six keys, membership of every fixed name also rules out extra keys.
    for mapping in mappings:
        if len(mapping) != len(_PARAMS) or any(name not in mapping for name in _PARAMS):
            raise ValueError('Paired parameter keys differ')
    sources = tuple({name: mapping[name] for name in _PARAMS} for mapping in mappings)
    bytes_required = 0
    for arm in sources:
        for name in _PARAMS:
            leaf = arm[name]
            if (not isinstance(leaf, jax.Array) or leaf.shape != _SHAPES[name]
                    or leaf.dtype != np.dtype('float32') or leaf.devices() != {device}):
                raise ValueError('Paired parameter shape, dtype or device differs')
            bytes_required += math.prod(_SHAPES[name]) * 4
    if bytes_required > MAX_HOST_BYTES:
        raise ValueError('Paired host copy exceeds byte limit')
    return sources


def copy_pair_to_host(result, *, device):
    """Return two independent private float32 dictionaries, or fail closed.

    All keys, production shapes, float32 dtypes, device placement, evidence
    types and the combined copy budget are checked before any transfer. Every
    source leaf is synchronized and its device checked again; copied host values
    must be finite. At least one host parameter must differ between the arms.

    The caller must authenticate the PairResult, actual device and checkpoint,
    keep them stable during this call, and cross-check the update budget and
    provenance before passing the returned dicts to ``snapshot.export_pair``.
    This function does not prove 64 real updates, save files, reload a model,
    or establish official game performance.
    """
    import jax

    sources = _snapshot_inputs(result, device, jax)
    # Synchronize all leaves before copying any of them to the host.
    for arm in sources:
        for name in _PARAMS:
            leaf = arm[name]
            leaf.block_until_ready()
            if leaf.devices() != {device}:
                raise ValueError('Paired parameter moved off the selected device')
    copies = []
    for arm in sources:
        host = {}
        for name in _PARAMS:
            value = jax.device_get(arm[name])
            if (not isinstance(value, np.ndarray) or value.shape != _SHAPES[name]
                    or value.dtype != np.dtype('float32')):
                raise ValueError('Host parameter shape or dtype differs')
            owned = np.array(value, dtype=np.float32, copy=True, order='C')
            if not bool(np.isfinite(owned).all()):
                raise ValueError('Host parameter contains a nonfinite value')
            host[name] = owned
        copies.append(host)
    if not any(not np.array_equal(copies[0][name], copies[1][name]) for name in _PARAMS):
        raise ValueError('Paired host parameters are identical')
    return copies[0], copies[1]
