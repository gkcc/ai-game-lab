"""Feed only source feature rows to the existing NumPy task scorer.

The caller authenticates the view, model, source/feature semantics and schema;
this module does not load a model, authenticate weights or execute a game.
"""

import re

import numpy as np

from learning.task_bc_v1.model import DIMENSIONS, OPS, SCHEMA_VERSION, TaskNetwork, continuous_mask
from learning.task_pair_data_preflight_v1.core import RaggedView
from learning.task_pair_eval_adapter_v1.adapter import LogitBlock, feature_blocks


_SHA256 = re.compile(r'[0-9a-f]{64}\Z')
_PARAM_SHAPES = {'w0': (373, 128), 'b0': (128,), 'w1': (128, 64),
                 'b1': (64,), 'w2': (64, 1), 'b2': (1,)}
MAX_MODEL_SNAPSHOT_BYTES = 512 * 1024


def _verify_schema(view, model, expected):
    """Compare model and view schema to an independently fixed expected value."""
    if (type(view) is not RaggedView or type(model) is not TaskNetwork
            or type(expected) is not str or _SHA256.fullmatch(expected) is None
            or view.schema_digest != expected):
        raise ValueError('Task scorer view or schema differs')
    metadata = getattr(model, 'metadata', None)
    if (type(metadata) is not dict or type(metadata.get('schema_version')) is not type(SCHEMA_VERSION)
            or metadata['schema_version'] != SCHEMA_VERSION
            or metadata.get('schema_sha256') != expected
            or type(metadata.get('shared_feature_names')) is not list
            or type(metadata.get('feature_names')) is not list
            or tuple(metadata['shared_feature_names']) != view.shared_names
            or tuple(metadata['feature_names']) != view.feature_names
            or type(metadata.get('ops')) is not list or metadata['ops'] != list(OPS)
            or type(metadata.get('dimensions')) is not list
            or not all(type(value) is int for value in metadata['dimensions'])
            or metadata['dimensions'] != list(DIMENSIONS)):
        raise ValueError('Task scorer model schema differs')


def _snapshot_model(view, model, expected):
    """Validate bounded model metadata, then copy before any external callback."""
    _verify_schema(view, model, expected)
    params = getattr(model, 'params', None)
    if (type(params) is not dict or len(params) != len(_PARAM_SHAPES)
            or set(params) != set(_PARAM_SHAPES)):
        raise ValueError('Task scorer parameter keys differ')
    sources = {name: params[name] for name in _PARAM_SHAPES}
    sources['mean'] = getattr(model, 'mean', None)
    sources['scale'] = getattr(model, 'scale', None)
    shapes = dict(_PARAM_SHAPES, mean=(373,), scale=(373,))
    source_bytes = snapshot_bytes = 0
    for name, shape in shapes.items():
        value = sources[name]
        if (not isinstance(value, np.ndarray) or value.shape != shape
                or value.dtype.kind != 'f' or value.dtype.itemsize not in (4, 8)):
            raise ValueError('Task scorer array shape or dtype differs')
        source_bytes += int(value.nbytes)
        snapshot_bytes += int(value.size) * np.dtype('float64').itemsize
    if source_bytes > MAX_MODEL_SNAPSHOT_BYTES or snapshot_bytes > MAX_MODEL_SNAPSHOT_BYTES:
        raise ValueError('Task scorer snapshot exceeds byte limit')
    # TaskNetwork.score converts weights and features to float64. The private
    # snapshot uses that same representation, without changing the source.
    copies = {name: np.array(value, dtype=np.float64, copy=True, order='K')
              for name, value in sources.items()}
    if (any(not bool(np.isfinite(value).all()) for value in copies.values())
            or np.any(copies['scale'] < 1)):
        raise ValueError('Task scorer snapshot has invalid values')
    mask = continuous_mask(list(view.shared_names) + list(view.feature_names))
    if not np.all(copies['mean'][~mask] == 0) or not np.all(copies['scale'][~mask] == 1):
        raise ValueError('Task scorer binary normalization differs')
    for value in copies.values():
        value.setflags(write=False)
    frozen = TaskNetwork.__new__(TaskNetwork)
    frozen.metadata = {'schema_version': SCHEMA_VERSION, 'schema_sha256': expected,
                       'shared_feature_names': list(view.shared_names),
                       'feature_names': list(view.feature_names),
                       'ops': list(OPS), 'dimensions': list(DIMENSIONS)}
    frozen.mean, frozen.scale = copies['mean'], copies['scale']
    frozen.params = {name: copies[name] for name in _PARAM_SHAPES}
    return frozen


def score_blocks(view, model, *, expected_schema_sha256, check,
                 max_groups=256, max_candidates=65_536):
    """Yield original-order full scores and first-maximum local choices.

    The caller must supply a previously authenticated TaskNetwork and schema,
    and keep model and data unchanged. Only each group's 275 shared features
    and its 98-column candidate features are passed to ``TaskNetwork.score``.
    Group bounds determine slicing; labels, source clusters, overlap masks and
    group identities never become model inputs. ``check`` is a sampled hook, not
    a hard deadline. Returned scores and choices are private in-memory outputs.

    The existing block adapter enforces bounded whole-group blocks and validates
    the view's canonical schema. Its evaluator must consume the entire iterator
    before a metric is accepted, so an interrupted iterator is not a result.
    No independent source/model authentication or file reload occurs here.
    """
    if not callable(check):
        raise ValueError('Task scorer requires a check callback')
    # Generator setup occurs when iteration begins. Capture bounded weights and
    # normalization before any caller-provided check callback can run.
    frozen_model = _snapshot_model(view, model, expected_schema_sha256)
    for block in feature_blocks(view, expected_schema_sha256=expected_schema_sha256,
                                max_groups=max_groups, max_candidates=max_candidates):
        check()
        _verify_schema(view, model, expected_schema_sha256)
        scores = np.empty(len(block.candidates), dtype=np.float64)
        choices = np.empty(block.stop - block.start, dtype=np.int64)
        for group in range(block.stop - block.start):
            check()
            lo, hi = int(block.offsets[group]), int(block.offsets[group + 1])
            # Call the original scorer directly; neither IDs nor supervision are
            # supplied as an input or an extra model-side callback argument.
            result = TaskNetwork.score(frozen_model, block.shared[group], block.candidates[lo:hi])
            if (not isinstance(result, np.ndarray) or result.shape != (hi - lo,)
                    or result.dtype not in (np.dtype('float32'), np.dtype('float64'))
                    or not bool(np.isfinite(result).all())):
                raise ValueError('Task scorer returned invalid candidate scores')
            scores[lo:hi] = result
            choices[group] = int(np.argmax(result))
        check()
        _verify_schema(view, model, expected_schema_sha256)
        scores.setflags(write=False)
        choices.setflags(write=False)
        yield LogitBlock(block.start, block.stop, scores, choices)
